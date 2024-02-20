import yaml
import os
from easydict import EasyDict as edict
from configs.config import get_default_config, update_config_from_file

SETTINGS = {
    'dataset_name': 'brats20_split3',
    'base_config': 'vit_B16_2D_cls_token_brats20_split3_input256_res_d2_a10',
    # 'grid_search_params': [0.95],
    # 'grid_search_params_names': ['0_95']
    # 'grid_search_params': [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99],
    # 'grid_search_params_names': ['0_01', '0_05', '0_1', '0_25', '0_5', '0_75', '0_9', '0_99'],
    'grid_search_params': [1, 5, 25, 50, 100, 250, 500, 1000, 2000, 5000, 10000],
    'grid_search_params_names': None,

}

def main(settings):
    config = get_default_config()
    if settings['base_config'] is not None:
        update_config_from_file(f"{settings['dataset_name']}/{settings['base_config']}.yaml", config)

    cur_config = config
    if settings['grid_search_params_names'] is None:
        settings['grid_search_params_names'] = settings['grid_search_params']

    if len(settings['grid_search_params_names']) != len(settings['grid_search_params']):
        raise ValueError(f"grid_search_params and grid_search_params_names must be the same len")
    for idx, param in enumerate(settings['grid_search_params']):
        #brats20
        # cur_yaml_name = f'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b{settings["grid_search_params_names"][idx]}_kl_a500_gtproc_gauss_51'
        # cur_yaml_name = f'vit_B16_2D_cls_token_brats20_split3_input256_robust_vit_a{settings["grid_search_params_names"][idx]}'
        cur_yaml_name = f'vit_B16_2D_cls_token_brats20_split3_input256_res_d2_a{settings["grid_search_params_names"][idx]}'

        # isles22
        # cur_yaml_name = f'vit_B16_2D_cls_token_isles22_input128dwi_lgm_fusion_b{settings["grid_search_params_names"][idx]}_kl_a1000_gtproc_gauss_51'
        # cur_yaml_name = f'vit_B16_2D_cls_token_isles22_input128dwi_robust_vit_a{settings["grid_search_params_names"][idx]}'
        # cur_yaml_name = f'vit_B16_2D_cls_token_isles22_input128dwi_res_d2_a{settings["grid_search_params_names"][idx]}'

        # config_name_template = f'vit_B16_2D_cls_token_brats20_split3_input256_lgm_fusion_b0_95_kl_a500_gtproc_gauss_51'
        # cur_config.TRAINING.LOSS.LOCALIZATION_LOSS.FUSION_BETA = param
        cur_config.TRAINING.LOSS.LOCALIZATION_LOSS.ALPHA = param

        save_path = f"{settings['dataset_name']}/{cur_yaml_name}.yaml"
        if os.path.exists(save_path):
            raise ValueError(f"path : {save_path} already exsits")
        save_dict = nested_easydict_to_dict(cur_config)
        with open(save_path, 'w') as outfile:
            yaml.dump(save_dict, outfile, default_flow_style=False, sort_keys=False)

    print('Done!')

def nested_easydict_to_dict(nested_odict):
   # Convert the nested ordered dictionary into a regular dictionary and store it in the variable "result".
   result = dict(nested_odict)

   # Iterate through each key-value pair in the dictionary.
   for key, value in result.items():

       # Check if the value is an instance of the OrderedDict class.
       if isinstance(value, dict):
           # If the value is an instance of the OrderedDict class, recursively call the function on that value and store the returned dictionary in the "result" dictionary.
           result[key] = nested_easydict_to_dict(value)
   return result
if __name__ == '__main__':
    settings = SETTINGS
    main(settings)