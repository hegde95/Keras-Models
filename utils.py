import numpy as np
import os
import os.path as osp
import argparse

Config ={}
# you should replace it with your own root_path
# Config['root_path'] = '/mnt/jiali/data/outfits/new_polyvore/test/polyvore_outfits'
Config['root_path'] = '/home/shashank/Desktop/Coursework/Sem2/DeepLearning/Week4/polyvore_outfits_hw/polyvore_outfits'
Config['meta_file'] = 'polyvore_item_metadata.json'
Config['checkpoint_path'] = 'ckpts'
Config['use_cuda'] = True
Config['debug'] = True
Config['num_epochs'] = 20
Config['batch_size'] = 64

Config['learning_rate'] = 0.001
Config['num_workers'] = 8