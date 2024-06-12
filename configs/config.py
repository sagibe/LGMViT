from easydict import EasyDict as edict
import yaml

def get_default_config():
    """
    Add default config.
    """
    cfg = edict()

    # TRAINING
    cfg.TRAINING = edict()
    cfg.TRAINING.INPUT_SIZE = 256
    cfg.TRAINING.LR = 0.00001
    cfg.TRAINING.LR_DROP = 12
    cfg.TRAINING.SCAN_SEG_SIZE = 32
    cfg.TRAINING.BATCH_SIZE = 32
    cfg.TRAINING.LAST_BATCH_MIN_RATIO = 0
    cfg.TRAINING.MAX_SCAN_SIZE = None
    cfg.TRAINING.WEIGHT_DECAY = 0.0001
    cfg.TRAINING.EPOCHS = 25
    cfg.TRAINING.CLIP_MAX_NORM = 0
    cfg.TRAINING.EVAL_INTERVAL = 1
    cfg.TRAINING.SAVE_CKPT_INTERVAL = 1
    cfg.TRAINING.SAVE_BEST_CKPT_CRITERION = 'f1'
    cfg.TRAINING.RESUME = ''
    cfg.TRAINING.START_EPOCH = 0
    cfg.TRAINING.EVAL = False
    cfg.TRAINING.NUM_WORKERS = 4
    cfg.TRAINING.CLS_THRESH = 0.5
    #cfg.TRAINING.LOSS
    cfg.TRAINING.LOSS = edict()
    cfg.TRAINING.LOSS.TYPE = 'bce'
    # cfg.TRAINING.LOSS.LOCALIZATION_LOSS
    cfg.TRAINING.LOSS.LOCALIZATION_LOSS = edict()
    cfg.TRAINING.LOSS.LOCALIZATION_LOSS.USE = False
    cfg.TRAINING.LOSS.LOCALIZATION_LOSS.TYPE = 'kl'
    cfg.TRAINING.LOSS.LOCALIZATION_LOSS.SPATIAL_FEAT_SRC = 'attn'
    cfg.TRAINING.LOSS.LOCALIZATION_LOSS.ATTENTION_METHOD = 'raw_attn'
    cfg.TRAINING.LOSS.LOCALIZATION_LOSS.FUSION_BETA = 0.5
    cfg.TRAINING.LOSS.LOCALIZATION_LOSS.SPATIAL_FEAT_INTERPOLATION = 'bilinear'
    cfg.TRAINING.LOSS.LOCALIZATION_LOSS.FEAT_CHANNEL_REDUCTION = 'squeeze_mean'
    cfg.TRAINING.LOSS.LOCALIZATION_LOSS.SPATIAL_MAP_NORM = 'softmax'
    cfg.TRAINING.LOSS.LOCALIZATION_LOSS.GT_SEG_PROCESS_METHOD = 'gauss'
    cfg.TRAINING.LOSS.LOCALIZATION_LOSS.GT_SEG_PROCESS_KERNEL_SIZE = 75
    cfg.TRAINING.LOSS.LOCALIZATION_LOSS.ALPHA = 10
    cfg.TRAINING.LOSS.LOCALIZATION_LOSS.PATIENT_LIST = None


    # MODEL
    cfg.MODEL = edict()
    cfg.MODEL.PRETRAINED_WEIGHTS = ''
    cfg.MODEL.NUM_CLASSES = 2
    cfg.MODEL.PATCH_SIZE = 16
    # MODEL.POSITION_EMBEDDING
    cfg.MODEL.POSITION_EMBEDDING = edict()
    cfg.MODEL.POSITION_EMBEDDING.TYPE = 'sine'
    cfg.MODEL.POSITION_EMBEDDING.Z_SIZE = 40
    cfg.MODEL.POSITION_EMBEDDING.FIT_MODE = 'interpolate'
    # MODEL.VIT_ENCODER
    cfg.MODEL.VIT_ENCODER = edict()
    cfg.MODEL.VIT_ENCODER.NUM_LAYERS = 12
    cfg.MODEL.VIT_ENCODER.FORWARD_EXPANSION_RATIO = 4
    cfg.MODEL.VIT_ENCODER.EMBED_SIZE = 768
    cfg.MODEL.VIT_ENCODER.HEADS = 12
    cfg.MODEL.VIT_ENCODER.DROP_PATH = 0.1
    cfg.MODEL.VIT_ENCODER.FORWARD_DROP_P = 0.1
    cfg.MODEL.VIT_ENCODER.USE_CLS_TOKEN = True
    cfg.MODEL.VIT_ENCODER.ATTENTION_3D = False

    # DATA
    cfg.DATA = edict()
    cfg.DATA.DATASET_DIR = '/mnt/DATA2/Sagi/Data/Prostate_MRI/'
    cfg.DATA.DATASETS = 'PICAI/processed_data/processed_data_t2w_bias_corr_resgist_t2w_hist_stnd_normalized/'
    cfg.DATA.ANNOT_TYPE = None
    cfg.DATA.DATA_SPLIT_FILE = '/mnt/DATA1/Sagi/Data/Prostate_MRI/processed_data/picai/train_val_splits.json'
    cfg.DATA.DATA_FOLD = 0
    cfg.DATA.OUTPUT_DIR = '/mnt/DATA2/Sagi/Models/LGMViT/'
    # DATA.PREPROCESS
    cfg.DATA.PREPROCESS = edict()
    cfg.DATA.PREPROCESS.RESIZE_MODE = 'interpolate'  # options: interpolate or padding
    cfg.DATA.PREPROCESS.GLAND_SEG_DIR = None
    cfg.DATA.PREPROCESS.MASK_ORGAN = True  # apply organ mask on scan
    cfg.DATA.PREPROCESS.CROP_ORGAN_SLICES = True  # crop scan according to prostate mask
    cfg.DATA.PREPROCESS.CROP_ORGAN_SPATIAL = True  # crop scan according to prostate mask
    cfg.DATA.PREPROCESS.SCAN_NORM_MODE = 'slice'
    cfg.DATA.PREPROCESS.CROP_PADDING = 0  # padding around prostate crop (0 for minimal cropping)

    # TEST
    cfg.TEST = edict()
    cfg.TEST.DATASET_PATH = ''
    cfg.TEST.BATCH_SIZE = 1
    cfg.TEST.CLIP_MAX_NORM = 0.1
    cfg.TEST.CHECKPOINT = 34
    cfg.TEST.NUM_WORKERS = 4
    cfg.TEST.CLS_THRESH = 0.5
    cfg.TEST.OUTPUT_DIR = '/mnt/DATA2/Sagi/Models/LGMViT/'

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
