import os.path as osp
import argparse
from typing import Any, Dict, List
from tqdm import tqdm
import wandb

import torch
from torch_geometric.loader import DataLoader
from torch_geometric import seed_everything

from dataset import MoleculeDataset
from models import RGCNEncoder, GINE

parser = argparse.ArgumentParser()
# parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
parser.add_argument("--device_id", type=int, default=-1)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--name", default='MCDK_MDR1') # PPB, MCDK_MDR1
parser.add_argument("--task", default='regression')
parser.add_argument("--num_epochs", default=30,type=int)
parser.add_argument("--train_ratio", default=0.6)
parser.add_argument("--valid_ratio", default=0.2)
parser.add_argument("--num_layers", default=2)
parser.add_argument("--hidden_channels", default=32)
parser.add_argument("--num_blocks", default=4)
parser.add_argument("--lr", default=0.01)
parser.add_argument("--evaluate_on_test", action='store_true', default=False)
parser.add_argument("--use_wandb", action='store_true', default=False)
parser.add_argument("--wandb_entity", default="team_ryoji")
parser.add_argument("--wandb_project", default="insilico")
parser.add_argument("--wandb_name", default="test")


x_map: Dict[str, List[Any]] = {
    'atomic_num':
    list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'CHI_TETRAHEDRAL',
        'CHI_ALLENE',
        'CHI_SQUAREPLANAR',
        'CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL',
    ],
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-5, 7)),
    'num_hs':
    list(range(0, 9)),
    'num_radical_electrons':
    list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map: Dict[str, List[Any]] = {
    'bond_type': [
        'UNSPECIFIED',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'QUADRUPLE',
        'QUINTUPLE',
        'HEXTUPLE',
        'ONEANDAHALF',
        'TWOANDAHALF',
        'THREEANDAHALF',
        'FOURANDAHALF',
        'FIVEANDAHALF',
        'AROMATIC',
        'IONIC',
        'HYDROGEN',
        'THREECENTER',
        'DATIVEONE',
        'DATIVE',
        'DATIVEL',
        'DATIVER',
        'OTHER',
        'ZERO',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOANY',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
    ],
    'is_conjugated': [False, True],
}


def test(model, loader, device, task, args=None, run=None, epoch=None, evaluate_on_test=False, mean=None, std=None):
    model.eval()
    if task == 'regression':
        loss_fn = torch.nn.MSELoss()
    
    losses = []
    preds = []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        if pred.ndim > 1:
            pred = pred.squeeze()
        if pred.ndim == 0:
            pred = pred.unsqueeze(0)
        if evaluate_on_test:
            preds.append(pred.detach())
        else:
            if mean is not None and std is not None:
                y = (batch.y - mean)/std
            else:
                y = batch.y
            loss = torch.sqrt(loss_fn(pred, y))
            losses.append(loss.item())
    
    if evaluate_on_test:
        preds = torch.cat(preds)
        return preds

    loss_avg = sum(losses)/len(losses)
    stats = {'valid/RMSE_loss': loss_avg}
    if args is not None and args.use_wandb:
        run.log(stats, step=epoch, commit=True)

    return loss_avg

def train(model, optimizer, num_epochs, train_loader, valid_loader, device, task, checkpoint_dir, args, mean, std):
    model.train()
    optimizer.zero_grad()

    if args.use_wandb:
        run = wandb.init(entity=args.wandb_entity, project=args.wandb_project,
                            name=args.wandb_name)
        run.config.update(args)
    else:
        run = False

    if task == 'regression':
        loss_fn = torch.nn.MSELoss()
    losses = []
    best_result = float("-inf")
    best_epoch = -1
    for epoch in range(num_epochs):
        for batch in train_loader:
            assert len(batch) == batch.y.shape[0]
            batch = batch.to(device)
            pred = model(batch)
            if pred.ndim > 1:
                pred = pred.squeeze()
            if mean is not None and std is not None:
                y = (batch.y - mean)/std
            else:
                y = batch.y
            loss = torch.sqrt(loss_fn(pred, y))
            losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            optimizer.zero_grad()

        loss_avg = sum(losses)/len(losses)
        print(f'Epoch {epoch}: Train RMSE Loss {loss_avg}')

        stats = {'train/RMSE_loss': loss_avg}
        if args.use_wandb:
            run.log(stats, step=epoch, commit=False)

        result = test(model, valid_loader, device, task, args, run, epoch, mean=mean, std=std)
        print(f'Epoch {epoch}: Valid RMSE Loss {result}')

        if result > best_result:
            best_result = result
            best_epoch = epoch

            print("Save checkpoint to model_epoch_%d.pth" % epoch)
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(state, f"{checkpoint_dir}/model_epoch_{epoch}.pth")

    print(f'Loading best model from {best_epoch}')
    state = torch.load(f"{checkpoint_dir}/model_epoch_{best_epoch}.pth", map_location=device)
    model.load_state_dict(state["model"])
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(state, f"{checkpoint_dir}/best_model.pth")

    return model


if __name__ == '__main__':
    args = parser.parse_args()

    seed_everything(42)

    if torch.cuda.is_available():
        if args.device_id >= 0:
            device = f'cuda:{args.device_id}'
        else:
            device = 'cuda'
    else:
        device = 'cpu'
    
    device = torch.device(device)

    dataset = MoleculeDataset(root='data', name=args.name, task=args.task, evaluate_on_test=args.evaluate_on_test)


    pw = args.num_workers > 0
    if args.evaluate_on_test:
        test_loader = DataLoader(dataset, batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.num_workers,
                                    pin_memory=True, persistent_workers=pw)
    else:
        index = torch.randperm(len(dataset))

        num_trains = int(len(dataset)*args.train_ratio)
        num_valids = int(len(dataset)*args.valid_ratio)
        train_indices = index[:num_trains]
        valid_indices = index[num_trains:num_trains+num_valids]
        test_indices = index[num_trains+num_valids:]

        assert train_indices.size(0)+valid_indices.size(0)+test_indices.size(0) == len(dataset)

        pw = args.num_workers > 0
        train_loader = DataLoader(dataset[train_indices], batch_size=args.batch_size,
                                        shuffle=True, num_workers=args.num_workers,
                                        pin_memory=True, persistent_workers=pw)

        valid_loader = DataLoader(dataset[valid_indices], batch_size=args.batch_size,
                                        shuffle=False, num_workers=args.num_workers,
                                        pin_memory=True, persistent_workers=pw)
        
        test_loader = DataLoader(dataset[test_indices], batch_size=args.batch_size,
                                        shuffle=False, num_workers=args.num_workers,
                                        pin_memory=True, persistent_workers=pw)

    num_layers = args.num_layers
    hidden_channels = args.hidden_channels
    num_blocks = args.num_blocks
    assert hidden_channels % num_blocks == 0
    num_node_features = len(x_map)
    edge_attr_dim = len(e_map)
    edge_attr_num_features = []
    for attr in ['bond_type', 'stereo', 'is_conjugated']:
        edge_attr_num_features.append(len(e_map['bond_type']))

    # model = RGCNEncoder(num_layers, hidden_channels, num_blocks, 
    #                     num_node_features, edge_attr_dim, edge_attr_num_features)
    if args.name in ['PPB']:
        use_sigmoid_last = True
    else:
        use_sigmoid_last = False

    model = GINE(num_node_features, 1, edge_attr_dim, use_sigmoid_last=use_sigmoid_last)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.evaluate_on_test:
        state = torch.load(f"{dataset.root}/best_model.pth", map_location=device)
        model.load_state_dict(state["model"])
    else:
        model = train(model, optimizer, args.num_epochs, train_loader, valid_loader, device, args.task, dataset.root, args, dataset.mean, dataset.std)
    args.use_wandb = False # don't log the test result on wandb

    result = test(model, test_loader, device, args.task, evaluate_on_test=args.evaluate_on_test, mean=dataset.mean, std=dataset.std)

    if args.evaluate_on_test:
        torch.save(result, osp.join(dataset.root, 'test_result.pt'))
    else:
        print(f'Performance of Best Model on Test (RMSE) is {result}')

    
    print('All Complete!')