import argparse
import yaml
from util import printl, printl_file_content
from util import prepare_saving_dir
import torch
import numpy as np
from box import Box


def main(parse_args, configs, valid_fold_number, test_fold_number):
    if type(configs.fix_seed) == int:
        torch.manual_seed(configs.fix_seed)
        torch.random.manual_seed(configs.fix_seed)
        np.random.seed(configs.fix_seed)

    torch.cuda.empty_cache()
    curdir_path, result_path, checkpoint_path, log_path = prepare_saving_dir(parse_args)

    printl(f"{'=' * 128}")
    printl("               ______   ______   .__   __. .___________..______          ___   .___________.  ______ .______      ")
    printl("              /      | /  __  \  |  \ |  | |           ||   _  \        /   \  |           | /      ||   _  \     ")
    printl("             |  ,----'|  |  |  | |   \|  | `---|  |----`|  |_)  |      /  ^  \ `---|  |----`|  ,----'|  |_)  |    ")
    printl("             |  |     |  |  |  | |  . `  |     |  |     |      /      /  /_\  \    |  |     |  |     |      /     ")
    printl("             |  `----.|  `--'  | |  |\   |     |  |     |  |\  \----./  _____  \   |  |     |  `----.|  |\  \----.")
    printl("              \______| \______/  |__| \__|     |__|     | _| `._____/__/     \__\  |__|      \______|| _| `._____|")
    printl()
    printl(f"{'=' * 128}", log_path=log_path)
    printl(configs.description, log_path=log_path)
    printl(f"{'=' * 128}", log_path=log_path)
    printl(f"Result Directory: {result_path}", log_path=log_path)
    printl(f"Checkpoint Directory: {checkpoint_path}", log_path=log_path)
    printl(f"Log Directory: {log_path}", log_path=log_path)
    printl(f"Current Working Directory: {curdir_path}", log_path=log_path)
    printl(f"{'=' * 128}", log_path=log_path)
    printl_file_content(parse_args.config_path, log_path=log_path)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ContraTCR: A tool for training and predicting TCR binding using '
                                                 'contrastive learning.')
    parser.add_argument("--config_path", help="Path to the configuration file. Defaults to "
                                              "'./config/default/config.yaml'. This file contains all necessary "
                                              "parameters and settings for the operation.",
                        default='./config/default/config.yaml')
    parser.add_argument("--mode", help="Operation mode of the script. Use 'train' for training the model and "
                                       "'predict' for making predictions using an existing model. Default mode is "
                                       "'train'.", default='train')
    parser.add_argument("--result_path", default='./result/default/',
                        help="Path where the results will be stored. If not set, results are saved to "
                             "'./result/default/'. This can include prediction outputs or saved models.")
    parser.add_argument("--resume_path", default=None,
                        help="Path to a previously saved model checkpoint. If specified, training or prediction will "
                             "resume from this checkpoint. By default, this is None, meaning training starts from "
                             "scratch.")

    parse_args = parser.parse_args()

    config_path = parse_args.config_path
    with open(config_path) as file:
        config_dict = yaml.full_load(file)
        configs = Box(config_dict)

    for i in range(1):
        valid_fold_number = i
        if valid_fold_number == 4:
            test_fold_number = 0
        else:
            test_fold_number = valid_fold_number + 1
        main(parse_args, configs, valid_fold_number, test_fold_number)


