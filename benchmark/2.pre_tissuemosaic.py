import pandas as pd 
import scanpy as sc 
import sys, os 

dset = sys.argv[1]
d = sc.read_h5ad(f"_data/{dset}/cells_normalized_harm_typed.h5ad")
d.obs[['x', 'y']] = d.obsm['spatial']

d.obsm['celltype'] = pd.get_dummies(d.obs['celltype'])

os.makedirs(f"_data/{dset}/sample_ads", exist_ok=True)

for sid in d.obs.sid.unique(): 
    d_sid = d[d.obs.sid == sid]
    d_sid.write_h5ad(f"_data/{dset}/sample_ads/{sid}.h5ad") 
