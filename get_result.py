import pandas as pd
import torch

dataset = 'PPB'
# MDCK_MDR1, CYP2C9, PPB, HepG2, TNF-A
#  CYP2C9, TNF-A

if dataset in ['HepG2']:
    unit = 'nM'

test_result = f'/storage/ryoji/InSilico/data/{dataset}/test_result.pt'
test_result = torch.load(test_result)

# mean_file = f'/storage/ryoji/InSilico/data/{dataset}/processed/mean_and_std.pt'
# mean, std = torch.load(mean_file)

# test_result = (test_result*std + mean) * 10**(-3)

test_result = test_result*100

real_result = f'/storage/ryoji/InSilico/data/{dataset}/real_result.pt'

torch.save(test_result, real_result)

print('Done')


