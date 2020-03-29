import numpy as np
import os
import os.path as osp
import argparse

Config ={}
# machine specific
Config['aws'] = False
Config['root_path_aws'] = '/home/ec2-user/hw4/polyvore_outfits'
Config['root_path_pc'] = '/home/shashank/Desktop/Coursework/Sem2/DeepLearning/Week4/polyvore_outfits_hw/polyvore_outfits'
Config['meta_file'] = 'polyvore_item_metadata.json'
Config['compatible_train'] = 'pairwise_compatibility_train.txt'
Config['compatible_valid'] = 'pairwise_compatibility_valid.txt'
# Config['compatible_test'] = 'compatibility_train.txt'
Config['checkpoint_path'] = 'ckpts'
Config['use_cuda'] = True
Config['shutown'] = False

# Debug options
Config['debug'] = False
Config['debug_size'] = 10000

# Training Options
Config['Custom_model'] = True
Config['num_epochs'] = 10
Config['batch_size'] = 64
Config['learning_rate'] = 0.001