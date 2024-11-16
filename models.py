import os.path as osp
import argparse

import torch
from torch_geometric.datasets import MoleculeNet


parser = argparse.ArgumentParser()
# parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
parser.add_argument("--device_id", type=int, default=-1)

if __name__ == '__main__':
    args = parser.parse_args()

    if torch.cuda.is_available():
        if args.device_id >= 0:
            device = f'cuda:{args.device_id}'
        else:
            device = 'cuda'
    else:
        device = 'cpu'
    
    device = torch.device(device)

    path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'RLPD')
    dataset = MoleculeNet(path, "ESOL")
    data = dataset[0].to(device)





    print('all complete')