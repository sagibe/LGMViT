import wandb

def init_wandb(project_name, run_name, wandb_group, cfg={}):
    """
    Set up a new run with Weights & Biases (wandb).

    Args:
        project_name (str): Name of the project to associate the run with.
        config (dict, optional): Config template containing the configuration parameters. Defaults to None.

    Returns:
        wandb.Run: The newly created wandb.Run object.
    """

    def flatten_dict(nested_dict, parent_key='', sep='_'):
        """
        Flatten a nested dictionary to a single level.

        Args:
            nested_dict (dict): The nested dictionary to flatten.
            parent_key (str): The key of the parent dictionary (used for recursion).
            sep (str): The separator to use between keys.

        Returns:
            dict: The flattened dictionary.
        """
        flattened_dict = {}
        try:
            for key, value in nested_dict.items():
                new_key = f"{parent_key}{sep}{key}" if parent_key else key
                if isinstance(value, dict):
                    flattened_dict.update(flatten_dict(value, parent_key=new_key, sep=sep))
                else:
                    flattened_dict[new_key] = value
        except Exception as e:
            pass
        return flattened_dict

    config = {}

    for key, value in cfg.items():
        if key in ('TRAINING', 'MODEL', 'DATA'):
            if isinstance(value, dict):
                config[key] = {sub_key: sub_value for sub_key, sub_value in value.items()}
            else:
                config[key] = value
    wandb.init(
        project=project_name,
        name=run_name,
        config=flatten_dict(config),
        group=wandb_group,
    )
    run = wandb.run
    return run

def wandb_logger(train_stats, val_stats=None, epoch=None):
    if val_stats:
        wandb.log(
            {"Train/Loss": train_stats['loss'],
             "Train/Classification_Loss": train_stats['cls_loss'],
             "Train/Localization_Loss": train_stats['localization_loss'],
             "Train/Accuracy": train_stats['acc'],
             "Train/Sensitivity": train_stats['sensitivity'],
             "Train/Specificity": train_stats['specificity'],
             "Train/Precision": train_stats['precision'],
             "Train/F1": train_stats['f1'],
             "Train/AUROC": train_stats['auroc'],
             "Train/AUPRC": train_stats['auprc'],
             "Train/Cohens_Kappa": train_stats['cohen_kappa'],
             'Train/lr': train_stats['lr'],
             "Validation/Loss": val_stats['loss'],
             "Validation/Accuracy": val_stats['acc'],
             "Validation/Sensitivity": val_stats['sensitivity'],
             "Validation/Specificity": val_stats['specificity'],
             "Validation/Precision": val_stats['precision'],
             "Validation/F1": val_stats['f1'],
             "Validation/AUROC": val_stats['auroc'],
             "Validation/AUPRC": val_stats['auprc'],
             "Validation/Cohens_Kappa": val_stats['cohen_kappa'],
             "epoch": epoch})
    else:
        wandb.log(
            {"Train/Loss": train_stats['loss'],
             "Train/Classification_Loss": train_stats['cls_loss'],
             "Train/Localization_Loss": train_stats['localization_loss'],
             "Train/Accuracy": train_stats['acc'],
             "Train/Sensitivity": train_stats['sensitivity'],
             "Train/Specificity": train_stats['specificity'],
             "Train/Precision": train_stats['precision'],
             "Train/F1": train_stats['f1'],
             "Train/AUROC": train_stats['auroc'],
             "Train/AUPRC": train_stats['auprc'],
             "Train/Cohens_Kappa": train_stats['cohen_kappa'],
             'Train/lr': train_stats['lr'],
             "epoch": epoch})
