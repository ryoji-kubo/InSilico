import pandas as pd
import os.path as osp
from torch_geometric.utils import from_smiles
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
import torch

class MoleculeDataset(InMemoryDataset):
    '''Knowledge Graph Dataset. This can be used to fasten the precomputing step.
    This Dataset does Computation of Split-wise Data + Subgraph Computation
    Args:
        root: the directory that has the dataset stored.
        name: the name of the dataset ('Family')
        split: the name of the split to precompute train/valid/test
        partition_id: the id of the partition
        num_partitions: the total number of partitions
        transform
        pre_transform
    '''
    def __init__(self, 
                 root='dataset',
                 name='CYP2C9',
                 task='regression',
                 evaluate_on_test = False,
                 transform=None, 
                 pre_transform=None):
        self.task = task
        self.folder = osp.join(root, name)
        self.name = name
        self.evaluate_on_test = evaluate_on_test
        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        if self.evaluate_on_test == False and name != 'PPB':
            self.mean = torch.mean(self.y)
            self.std = torch.std(self.y)
            torch.save((self.mean, self.std), osp.join(self.processed_dir, 'mean_and_std.pt'))
        else:
            self.mean = None
            self.std = None
            

    @property
    def raw_file_names(self):
        if self.evaluate_on_test == False:
            return [f'/storage/ryoji/InSilico/Testsets/{self.name}_data.csv']
        else:
            return [f'/storage/ryoji/InSilico/Testsets/{self.name}_testset.csv']
    
    @property
    def processed_file_names(self): # return the full file name
        if self.evaluate_on_test == False:
            return ['data.pt']
        else:
            return ['test_data.pt']
    
    def download(self):
        pass

    def process(self):
        data_list = []
        for raw_file in self.raw_file_names:
            test_df = pd.read_csv(raw_file)
            for index, row in tqdm(test_df.iterrows(), desc='Converting Smile strings to Graphs', total=test_df.shape[0]):
                smile_col = [col for col in row.keys() if 'smile' in col.lower()]
                assert len(smile_col) == 1
                smile_col = smile_col[0]
                smile_str = row[smile_col]
                data = from_smiles(smile_str)

                if self.evaluate_on_test == False:
                    y_col = [col for col in row.keys() if 'value' in col.lower()]
                    assert len(y_col) == 1
                    y_col = y_col[0]

                if data.edge_index.shape[1] == 0:
                    continue
                
                if self.task == 'regression':
                    if self.evaluate_on_test == False:
                        if self.name == 'PPB':
                            data.y = row[y_col]/100 # the label is between 0 - 100, convert it to [0, 1]
                        else:
                            data.y = row[y_col]
                # data.y = torch.randint(1, (1, )) # for now, a random label
                data_list.append(data)

        torch.save(self.collate(data_list),  self.processed_paths[0])

if __name__ == '__main__':
    # Example
    dataset = MoleculeDataset(root='data', name='CYP2C9')

    print("Complete")