import pandas as pd
import torch

dataset = 'TNF-A'
# MDCK_MDR1, CYP2C9, PPB, HepG2, TNF-A
#  CYP2C9, TNF-A

test_result = f'/storage/ryoji/InSilico/data/{dataset}/real_result.pt'
test_result = torch.load(test_result)
test_result = test_result.tolist()
df = pd.DataFrame({'result':test_result})
df.to_csv(f'/storage/ryoji/InSilico/data/{dataset}/{dataset}_predictions.csv')

print('Done')


