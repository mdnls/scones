import argparse
import traceback
import shutil
import logging
import yaml
import sys
import torch
import numpy as np
import copy
from scones.runners import *

import os
from scones.runners import GaussianRunner

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--config', type=str, required=True,  help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp', help='Path for saving running related data.')
    parser.add_argument('--doc', type=str, required=True, help='A string for documentation purpose. '
                                                               'Will be the name of the log folder.')
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--dry_run', action="store_true", help="If true, no code is excuted and the script is dry-run.")
    parser.add_argument('--overwrite', action="store_true", help="If true, automatically overwrite without asking.")
    parser.add_argument('-i', '--image_folder', type=str, default='images', help="The folder name of samples")
    parser.add_argument('--save_labels', action="store_true", help="If set to true then sampling will also save labels and bproj details of source.")
    args = parser.parse_args()
    args.log_path = os.path.join('scones', args.exp, 'logs', args.doc)

    # parse config file
    with open(os.path.join('scones/conf', args.config), 'r') as f:
        config = yaml.load(f)
    new_config = dict2namespace(config)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    os.makedirs(os.path.join(args.exp, 'image_samples'), exist_ok=True)
    args.image_folder = os.path.join(args.exp, 'image_samples', args.image_folder)
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)
        if(args.save_labels):
            os.makedirs(os.path.join(args.image_folder, "labels"))
    else:
        overwrite = args.overwrite
        if(not overwrite):
            response = input("Image folder already exists. Overwrite? (Y/N) ")
            if response.upper() == 'Y':
                overwrite = True

        if overwrite:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
            if (args.save_labels):
                os.makedirs(os.path.join(args.image_folder, "labels"))
        else:
            print("Output image folder exists. Program halted.")
            sys.exit(0)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device
    args.device = device

    # todo: rethink the config design to avoid translation
    new_config.ncsn.device = device
    new_config.compatibility.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    logging.info("Config =")
    print(">" * 80)
    config_dict = copy.copy(vars(config))
    print(yaml.dump(config_dict, default_flow_style=False))
    print("<" * 80)

    if(args.dry_run):
        print("Dry run successful!")
        print(f"GPU Availability: {torch.cuda.is_available()}")
    else:
        try:
            if(config.target.data.dataset.upper() in ["GAUSSIAN", "GAUSSIAN-HD"]):
                runner = GaussianRunner(args, config)
            else:
                runner = SCONESRunner(args, config)
            runner.sample()
        except:
            logging.error(traceback.format_exc())

    return 0

if __name__ == '__main__':
    sys.exit(main())
