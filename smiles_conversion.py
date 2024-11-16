import pandas as pd
from torch_geometric.utils import from_smiles

if __name__ == '__main__':
    test_df = pd.read_csv('/storage/ryoji/InSilico/Testsets/CYP2C9_testset.csv')
    # Example
    smile_str = test_df.iloc[0]['SMILES']



    print("Complete")