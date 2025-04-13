from easydict import EasyDict as edict
import yaml

def get_default_config():
    """
    Initialize and return the default config
    """
    cfg = edict()

    # DATA
    cfg.DATA = edict()
    cfg.DATA.DATASET_DIR = ''   # Full path to dataset parent directory
    cfg.DATA.DATASETS = ''  # Name of dataset directory
    cfg.DATA.DATA_SPLIT_FILE = ''  # Path to dataset split file. Files located at: datasets/data_splits
    cfg.DATA.ANNOT_TYPE = None  # Annotation type to use if there is more than one (for LiTS17 options are: 'lesion', 'organ')
    # DATA.PREPROCESS
    cfg.DATA.PREPROCESS = edict()
    cfg.DATA.PREPROCESS.RESIZE_MODE = 'interpolate'  # Options: 'interpolate' or 'padding'. Resize method
    cfg.DATA.PREPROCESS.MASK_ORGAN = True  # Apply organ mask on scan
    cfg.DATA.PREPROCESS.CROP_ORGAN_SLICES = True    # Crop scan in the slices dimension according to organ mask
    cfg.DATA.PREPROCESS.CROP_ORGAN_SPATIAL = True   # Crop scan in the spatial dimensions according to organ mask
    cfg.DATA.PREPROCESS.CROP_PADDING = 0  # Padding around organ crop (0 for minimal cropping)
    cfg.DATA.PREPROCESS.SCAN_NORM_MODE = 'slice'    # Options: 'slice', 'scan', None. Minmax normalization option (for 'slice' normalization is applied separately for each slice, for 'scan' it is applied on the entire scan at once)

    # TRAINING
    cfg.TRAINING = edict()
    cfg.TRAINING.OUTPUT_DIR = '/mnt/DATA1/Sagi/Models/LGMViT/'  # Path to parent directory where the experiments (checkpoints) will be saved to
    cfg.TRAINING.INPUT_SIZE = 256   # Spatial input size (H and W)
    cfg.TRAINING.LR = 0.00001   # Initial learning rate
    cfg.TRAINING.SCAN_SEG_SIZE = 32 # Number of slices for each forward pass. This number must be a factor of the batch size. Reduce this number in cases where the desired batch size exceeds the GPU memory.
    cfg.TRAINING.BATCH_SIZE = 32 # Effective batch size
    cfg.TRAINING.LAST_BATCH_MIN_RATIO = 0 # Value between 0 and 1 (0 represents no size limit). The minimum size of the last batch (set by the ratio with the batch size) of a scan that is acceptable. Scans where the last batch is smaller than this number will be trimmed at the edges of the scan.
    cfg.TRAINING.MAX_SCAN_SIZE = None   # The maximum size (number of slices) of a scan that is acceptable. For scans with more slices than this number, a random continuous seg the size of this number will be selected (cropped). Set None for no size limit.
    cfg.TRAINING.WEIGHT_DECAY = 0.0001  # Weight decay parameter
    cfg.TRAINING.EPOCHS = 25    # Number of epochs to train
    cfg.TRAINING.CLIP_MAX_NORM = 0  # Clip gradients
    cfg.TRAINING.EVAL_INTERVAL = 1  # Interval length (in epochs) for evaluation on validation set
    cfg.TRAINING.SAVE_CKPT_INTERVAL = 1 # Interval length (in epochs) to save checkpoints
    cfg.TRAINING.SAVE_BEST_CKPT_CRITERION = 'f1'    # Options: 'f1', 'acc', 'auroc', 'auprc', 'cohen_kappa'. Criterion in which to save checkpoint for best epoch.
    cfg.TRAINING.RESUME = ''    # Resume from checkpoint. Options: 'latest' (resumes from the last saved epoch), insert full path to checkpoint, insert url to checkpoint
    cfg.TRAINING.START_EPOCH = 0 # Initialize the epoch count of training session
    cfg.TRAINING.EVAL = False
    cfg.TRAINING.NUM_WORKERS = 4
    cfg.TRAINING.CLS_THRESH = 0.5   # Binary classification threshold for evaluation of the model's prediction
    #cfg.TRAINING.LOSS
    cfg.TRAINING.LOSS = edict()
    cfg.TRAINING.LOSS.TYPE = 'bce'  # Classification loss type (currently only supports binary cross entropy)
    # cfg.TRAINING.LOSS.LOCALIZATION_LOSS
    cfg.TRAINING.LOSS.LOCALIZATION_LOSS = edict()
    cfg.TRAINING.LOSS.LOCALIZATION_LOSS.USE = False     # Apply localization loss
    cfg.TRAINING.LOSS.LOCALIZATION_LOSS.TYPE = 'kl'     # Localization loss type. Options: 'kl', 'mse', 'l1', 'mse_fgbg'(RobustViT), 'res'
    cfg.TRAINING.LOSS.LOCALIZATION_LOSS.SPATIAL_FEAT_SRC = 'fusion'     # Options: 'attn', 'bb_feat', 'fusion'. Source for attribution map for localization supervision
    cfg.TRAINING.LOSS.LOCALIZATION_LOSS.ATTENTION_METHOD = 'last_layer_attn'   # Options: 'last_layer_attn', 'rollout', 'relevance_map'(GAE)
    cfg.TRAINING.LOSS.LOCALIZATION_LOSS.FUSION_BETA = 0.5   # Value between 0 and 1. Weighting parameter of the EAFEM
    cfg.TRAINING.LOSS.LOCALIZATION_LOSS.SPATIAL_FEAT_INTERPOLATION = 'bilinear' # Options:  'bilinear', 'nearest'. Interpolation method to resize attribution maps
    cfg.TRAINING.LOSS.LOCALIZATION_LOSS.FEAT_CHANNEL_REDUCTION = 'squeeze_mean' # Optiions: 'squeeze_mean', 'select_max', 'squeeze_max'. Channel reduction methods in attribution map generation process (heads/embedding reduction)
    cfg.TRAINING.LOSS.LOCALIZATION_LOSS.SPATIAL_MAP_NORM = 'softmax'    # Options: 'softmax', 'minmax'. Attribution masp normalization methods in localization loss
    cfg.TRAINING.LOSS.LOCALIZATION_LOSS.GT_SEG_PROCESS_METHOD = 'gauss' # Options: 'None', 'gauss', 'learned_S1', 'learned_S2', 'learned_D1', 'learned_D2', 'learned_d1'. GT process method
    cfg.TRAINING.LOSS.LOCALIZATION_LOSS.GT_SEG_PROCESS_KERNEL_SIZE = 51 # Kernel size for gt process (for gauss)
    cfg.TRAINING.LOSS.LOCALIZATION_LOSS.ALPHA = 10  # Weighting parameter of the localization loss
    cfg.TRAINING.LOSS.LOCALIZATION_LOSS.PATIENT_LIST = None # Path to file of the list of cases to apply localization loss on (for ablation). Files located at: datasets/LGS_patient_lists

    # MODEL
    cfg.MODEL = edict()
    cfg.MODEL.PRETRAINED_WEIGHTS = ''   # Path to pretrained weights
    cfg.MODEL.NUM_CLASSES = 2   # Number of classes for classification (currently only supports 2 - binary classification)
    cfg.MODEL.PATCH_SIZE = 16   # Patch size for patch embedding module
    # MODEL.POSITION_EMBEDDING
    cfg.MODEL.POSITION_EMBEDDING = edict()
    cfg.MODEL.POSITION_EMBEDDING.TYPE = 'sine'  # Options: 'sine', 'learned', None. Type of positional encoding
    cfg.MODEL.POSITION_EMBEDDING.Z_SIZE = 40    # Max number of slices (for depth dimension of positional encoding)
    cfg.MODEL.POSITION_EMBEDDING.FIT_MODE = 'interpolate'   # Interpolation type to fit positional encoding if needed.
    # MODEL.VIT_ENCODER
    cfg.MODEL.VIT_ENCODER = edict()
    cfg.MODEL.VIT_ENCODER.NUM_LAYERS = 12   # Number of ViT encoder blocks
    cfg.MODEL.VIT_ENCODER.FORWARD_EXPANSION_RATIO = 4   # MLP expansion parameter in ViT block
    cfg.MODEL.VIT_ENCODER.EMBED_SIZE = 768  # Embedding size
    cfg.MODEL.VIT_ENCODER.HEADS = 12    # Number of heads
    cfg.MODEL.VIT_ENCODER.DROP_PATH = 0.1   # Drop path ratio
    cfg.MODEL.VIT_ENCODER.FORWARD_DROP_P = 0.1  # Drop path ratio
    cfg.MODEL.VIT_ENCODER.USE_CLS_TOKEN = True  # Use class token (currently mandatory)
    cfg.MODEL.VIT_ENCODER.ATTENTION_3D = False  # Apply 3D attention instead of 2D (Currently not supported, need to fix bug)

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
