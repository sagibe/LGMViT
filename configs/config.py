from easydict import EasyDict as edict
import yaml

def get_default_config():
    """
    Add default config.
    """
    cfg = edict()

    # TRAINING
    cfg.TRAINING = edict()
    cfg.TRAINING.LR = 0.00001
    cfg.TRAINING.LR_DROP = 12
    cfg.TRAINING.BATCH_SIZE = 1
    cfg.TRAINING.WEIGHT_DECAY = 0.0001
    cfg.TRAINING.EPOCHS = 50
    cfg.TRAINING.CLIP_MAX_NORM = 0.1
    cfg.TRAINING.EVAL_INTERVAL = 1
    cfg.TRAINING.RESUME = ''
    cfg.TRAINING.START_EPOCH = 0
    cfg.TRAINING.EVAL = False
    cfg.TRAINING.NUM_WORKERS = 4
    cfg.TRAINING.CLS_THRESH = 0.5

    # MODEL
    cfg.MODEL = edict()
    cfg.MODEL.PRETRAINED_WEIGHTS = ''
    cfg.MODEL.NUM_CLASSES = 2
    # MODEL.POSITION_EMBEDDING
    cfg.MODEL.POSITION_EMBEDDING = edict()
    cfg.MODEL.POSITION_EMBEDDING.TYPE = 'sine'
    cfg.MODEL.POSITION_EMBEDDING.Z_SIZE = 40
    cfg.MODEL.POSITION_EMBEDDING.MODE = 'interpolate'
    # MODEL.BACKBONE
    cfg.MODEL.BACKBONE = edict()
    cfg.MODEL.BACKBONE.NAME = 'resnet101'
    cfg.MODEL.BACKBONE.RETURN_INTERM_LAYERS = False
    cfg.MODEL.BACKBONE.DILATION = False
    # MODEL.TRANSFORMER
    cfg.MODEL.TRANSFORMER = edict()
    cfg.MODEL.TRANSFORMER.NUM_LAYERS = 6
    cfg.MODEL.TRANSFORMER.FORWARD_EXPANSION_RATIO = 4
    cfg.MODEL.TRANSFORMER.EMBED_SIZE = 2048
    cfg.MODEL.TRANSFORMER.HEADS = 8
    cfg.MODEL.TRANSFORMER.DROP_PATH = 0.1
    cfg.MODEL.TRANSFORMER.FORWARD_DROP_P = 0.1

    # DATA
    cfg.DATA = edict()
    cfg.DATA.DATASET_DIR = '/mnt/DATA2/Sagi/Data/'
    cfg.DATA.DATASETS = 'PICAI/processed_data/processed_data_t2w_bias_corr_resgist_t2w_hist_stnd_normalized/'
    cfg.DATA.DATA_LIST = None
    cfg.DATA.DATA_FOLD = 0
    cfg.DATA.INPUT_SIZE = 128
    cfg.DATA.MODALITIES = 'all'  # 'all' for all available modalities. For specific modalities in a list of the desired ones (example = [])
    cfg.DATA.OUTPUT_DIR = '/mnt/DATA2/Sagi/Models/ProLesClassifier/'
    # DATA.PREPROCESS
    cfg.DATA.PREPROCESS = edict()
    cfg.DATA.PREPROCESS.RESIZE_MODE = 'interpolate'  # options: interpolate or padding
    cfg.DATA.PREPROCESS.MASK_PROSTATE = True # apply prostate mask on scan
    cfg.DATA.PREPROCESS.CROP_PROSTATE = True # crop scan according to prostate mask
    cfg.DATA.PREPROCESS.CROP_PADDING = 0 # padding around prostate crop (0 for minimal cropping)

    # TEST
    cfg.TEST = edict()
    cfg.TEST.DATASET_PATH = '/mnt/DATA2/Sagi/Data/PICAI/processed_data/processed_data_t2w_bias_corr_resgist_t2w_hist_stnd_normalized/fold_0/val/'
    cfg.TEST.BATCH_SIZE = 1
    cfg.TEST.CLIP_MAX_NORM = 0.1
    cfg.TEST.CHECKPOINT = 34
    cfg.TEST.NUM_WORKERS = 4
    cfg.TEST.CLS_THRESH = 0.5
    cfg.TEST.OUTPUT_DIR = '/mnt/DATA2/Sagi/Models/ProLesClassifier/'

    # DISTRIBUTED
    cfg.DISTRIBUTED = edict()
    cfg.DISTRIBUTED.WORLD_SIZE = 1
    cfg.DISTRIBUTED.DIST_URL = 'env://'

    return cfg

def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(filename, cfg, base_cfg=None):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        if base_cfg is not None:
            _update_config(base_cfg, exp_config)
        else:
            _update_config(cfg, exp_config)
