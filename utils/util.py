from types import SimpleNamespace
import os
import torch
import torch.distributed as dist
import time
import datetime
from collections import defaultdict, deque
from torch import sigmoid
from torchmetrics.classification import BinaryAUROC, AveragePrecision, BinaryCohenKappa
# from torcheval.metrics.functional import binary_auroc, binary_auprc


class RecursiveNamespace(SimpleNamespace):

    @staticmethod
    def map_entry(entry):
        if isinstance(entry, dict):
            return RecursiveNamespace(**entry)

        return entry

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, val in kwargs.items():
            if type(val) == dict:
                setattr(self, key, RecursiveNamespace(**val))
            elif type(val) == list:
                setattr(self, key, list(map(self.map_entry, val)))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.DISTRIBUTED.DIST_URL), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.DISTRIBUTED.DIST_URL,
                                         world_size=args.DISTRIBUTED.WORLD_SIZE, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

class NestedTensor(object):
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

def is_main_process():
    return get_rank() == 0

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)
class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            if k in ['loss', 'cls_loss', 'localization_loss', 'lr']:
                self.meters[k].update(v)
            else:
                self.meters[k] = v
    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            if isinstance(meter, SmoothedValue):
                meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

class PerformanceMetrics(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, device, bin_thresh=0.5):
        self.device = device
        self.preds = torch.empty((0, 1), device=device, dtype=torch.float32)
        self.targets = torch.empty((0, 1), device=device, dtype=torch.int)
        self.tp = torch.zeros(1, device=device, dtype=torch.int64)
        self.tn = torch.zeros(1, device=device, dtype=torch.int64)
        self.fp = torch.zeros(1, device=device, dtype=torch.int64)
        self.fn = torch.zeros(1, device=device, dtype=torch.int64)
        self.bin_thresh = bin_thresh

    def update(self, outputs, targets):
        self.preds = torch.cat((self.preds, outputs), 0)
        self.targets = torch.cat((self.targets, targets.to(int)), 0)
        if outputs.size(1) == 1:
            preds = (sigmoid(outputs) > self.bin_thresh) * 1
        else:
            preds = outputs.argmax(-1)
        targets_bools = targets > 0
        preds_bools = preds > 0
        self.tp += sum(targets_bools * preds_bools)
        self.tn += sum(~targets_bools * ~preds_bools)
        self.fp += sum(~targets_bools * preds_bools)
        self.fn += sum(targets_bools * ~preds_bools)
    @property
    def sensitivity(self):
        return (self.tp / (self.tp + self.fn)).item()

    @property
    def specificity(self):
        return (self.tn / (self.tn + self.fp)).item()

    @property
    def precision(self):
        return (self.tp / (self.tp + self.fp)).item()

    @property
    def f1(self):
        return (2 * self.tp / (2 * self.tp + self.fp + self.fn)).item()

    @property
    def accuracy(self):
        return ((self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)).item()
    @property
    def auroc(self):
        auroc = BinaryAUROC(thresholds=None).to(self.device)
        return auroc(self.preds, self.targets).item()
    @property
    def auprc(self):
        auprc = AveragePrecision(task="binary").to(self.device)
        return auprc(self.preds, self.targets).item()
    @property
    def cohen_kappa(self):
        cohen_kappa = BinaryCohenKappa().to(self.device)
        return cohen_kappa(self.preds, self.targets).item()

    # def __str__(self):
    #     return self.fmt.format(
    #         median=self.median,
    #         avg=self.avg,
    #         global_avg=self.global_avg,
    #         max=self.max,
    #         value=self.value)

def attention_softmax_2d(attn, apply_log=True):
    if apply_log:
        return torch.nn.functional.log_softmax((attn.view(*attn.size()[:2], -1)), dim=2).view_as(attn)
    else:
        return torch.nn.functional.softmax((attn.view(*attn.size()[:2], -1)), dim=2).view_as(attn)



