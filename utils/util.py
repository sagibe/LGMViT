from types import SimpleNamespace
import os
import numpy as np
import cv2
import torch
import torch.distributed as dist
import time
import datetime
from collections import defaultdict, deque
from torch import sigmoid
from torchmetrics.classification import BinaryAUROC, AveragePrecision, BinaryCohenKappa

class RecursiveNamespace(SimpleNamespace):
    """
    A subclass of SimpleNamespace that recursively converts dictionaries
    and lists of dictionaries into RecursiveNamespace instances, allowing
    nested attribute-style access.
    """
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
    """
    A class for logging and displaying metrics during training or evaluation loops, with support for
    smoothing, synchronization across distributed processes, and periodic printing.

    Attributes:
    - meters (defaultdict of SmoothedValue): Stores smoothed metric values with custom formatting.
    - delimiter (str): Delimiter for joining metric log strings.

    Methods:
    - update(**kwargs): Updates metric values with new data; adds or smooths values over iterations.
    - synchronize_between_processes(): Synchronizes all SmoothedValue meters across processes.
    - add_meter(name, meter): Adds a custom meter to the logger.
    - log_every(iterable, print_freq, header): Iterates over an iterable while logging metrics at
      a specified frequency. Displays estimated time remaining, current time, and memory usage.

    Special Methods:
    - __str__(): Returns a formatted string of all current meters.
    - __getattr__(attr): Provides attribute access to meters or raises an AttributeError if not found.
    """
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
    """
    A class for calculating and storing various performance metrics for binary classification.

    Attributes:
    - device (torch.device): The device (e.g., CPU or GPU) to store tensors and perform calculations.
    - bin_thresh (float): The threshold for converting prediction probabilities to binary class labels.
    - preds (torch.Tensor): Stores concatenated predictions over batches.
    - targets (torch.Tensor): Stores concatenated true labels over batches.
    - tp, tn, fp, fn (torch.Tensor): Track counts of true positives, true negatives, false positives,
      and false negatives, respectively, for binary classification.

    Methods:
    - update(outputs, targets): Updates the stored predictions, targets, and metric counters
      (tp, tn, fp, fn) with new batch data.
    - f1 (property): Computes the F1 score based on tp, fp, and fn.
    - accuracy (property): Computes the accuracy of predictions.
    - auroc (property): Computes the Area Under the Receiver Operating Characteristic Curve.
    - auprc (property): Computes the Area Under the Precision-Recall Curve.
    - cohen_kappa (property): Computes Cohen's Kappa, a measure of classification reliability.
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

def resize_scan(scan, size=256):
    """
    Resize each slice of a 3D medical scan to a specified size.

    This function takes a 3D numpy array representing a medical scan (e.g., CT or MRI)
    and resizes each 2D slice to the specified size using bicubic interpolation.

    Parameters:
    scan (numpy.ndarray): A 3D numpy array of shape (num_slices, height, width)
                          representing the input scan.
    size (int, optional): The desired width and height of each slice after resizing.
                          Default is 256.

    Returns:
    numpy.ndarray: A 3D numpy array of shape (num_slices, size, size) containing
                   the resized scan.
    """
    scan_rs = np.zeros((len(scan), size, size))
    for idx in range(len(scan)):
        cur_slice = scan[idx, :, :]
        cur_slice_rs = cv2.resize(cur_slice, (size, size), interpolation=cv2.INTER_CUBIC)
        scan_rs[idx, :, :] = cur_slice_rs
    return scan_rs

def min_max_norm_scan(scan):
    """
    Apply min-max normalization to a 3D medical scan.

    This function performs min-max normalization on the entire 3D scan, scaling
    the voxel values to a range between 0 and 1. The normalization is applied
    globally across all slices of the scan, preserving the relative intensity
    differences between different parts of the scan.

    Parameters:
    scan (numpy.ndarray): A 3D numpy array representing the input scan.
                          The array can be of any shape, typically
                          (num_slices, height, width) for medical imaging data.

    Returns:
    numpy.ndarray: A 3D numpy array of the same shape as the input, containing
                   the normalized scan with values scaled to the range [0, 1].

    """
    return (scan - scan.min()) / (scan.max() - scan.min())

def min_max_norm_slice(scan):
    """
    Apply min-max normalization to a 3D medical scan on a slice-by-slice basis.

    This function performs min-max normalization independently for each 2D slice
    of a 3D scan, scaling the pixel values of each slice to a range between 0 and 1.

    Parameters:
    scan (numpy.ndarray): A 3D numpy array representing the input scan.
                          Expected shape is (num_slices, height, width).

    Returns:
    numpy.ndarray: A 3D numpy array of the same shape as the input, containing
                   the normalized scan with values scaled to the range [0, 1]
                   for each slice independently.

    """
    scan_norm = np.zeros_like(scan)
    for idx in range(len(scan)):
        cur_slice = scan[idx, :, :]
        if cur_slice.max() > cur_slice.min():
            cur_slice_norm = (cur_slice - cur_slice.min()) / (cur_slice.max() - cur_slice.min())
            scan_norm[idx, :, :] = cur_slice_norm
    return scan_norm

def attention_softmax_2d(attn, apply_log=True):
    """
    Apply 2D softmax normalization to a batch of attention maps.

    This function takes a batch of attention maps and applies softmax (or log_softmax)
    normalization across the spatial dimensions.

    Parameters:
    attn (torch.Tensor): A 4D tensor of shape [scans (usually 1), batch_size (slices), height, width]
                         representing a batch of attention maps.
    apply_log (bool, optional): If True, applies log_softmax instead of regular softmax.
                                Default is True.

    Returns:
    torch.Tensor: A tensor of the same shape as the input, containing the normalized
                  attention maps.
    """
    if apply_log:
        return torch.nn.functional.log_softmax((attn.view(*attn.size()[:2], -1)), dim=2).view_as(attn)
    else:
        return torch.nn.functional.softmax((attn.view(*attn.size()[:2], -1)), dim=2).view_as(attn)

def min_max_normalize(batch_maps):
  """Applies min max normalization on a batch of 2D maps.

  Args:
    batch_maps: A 3D PyTorch tensor of shape (batch_size, height, width).

  Returns:
    A 3D PyTorch tensor of the same shape as `batch_maps` with the values
    min-max normalized.
  """

  # Get the min and max values of each map.
  min_values = batch_maps.amin(dim=(-2, -1), keepdim=True)
  max_values = batch_maps.amax(dim=(-2, -1), keepdim=True)

  # Normalize each map.
  normalized_maps = (batch_maps - min_values) / (max_values - min_values)

  return normalized_maps

