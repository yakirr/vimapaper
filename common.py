import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import entropy
import os
import glob

def integration(d):
    A = d.obsp['connectivities']# (d.obsp['connectivities'] > 0).astype(np.float32)
    A /= A.sum(axis=1)
    S = pd.get_dummies(d.obs.sid).astype(np.float32)
    baseline = np.power(2, entropy(d.obs.sid.value_counts() / len(d), base=2))
    perplexities = np.power(2, entropy(np.array(A.dot(S)), axis=1, base=2)) / baseline
    return perplexities

def write_perplexities(dsetname, ds, outpath):
    # compute perplexities
    perplexities = {}
    perplexities['vima'] = np.concatenate([integration(d) for d in ds])
    for file in glob.glob(f'../benchmark/_results/{dsetname}_*_noharm.h5ad'):
        method = os.path.basename(file).split('_')[1]
        d = sc.read_h5ad(file)
        perplexities[method] = integration(d)

    # merge data
    data = []
    labels = []
    for k, v in perplexities.items():
        data.append(v)
        labels.extend([k] * len(v))
    perplexities = pd.DataFrame({'perplexity': np.concatenate(data), 'method': labels})
    perplexities['dset'] = dsetname
    perplexities.to_csv(outpath, index=False)
    print(perplexities.groupby('method').describe())

def collate_cc(dsetname, D, outpath):
    ps = []
    nposs = []
    nnegs = []
    ns = []
    methods = []
    for file in glob.glob(f'../benchmark/_results/{dsetname}_*_noharm.h5ad'):
        d = sc.read_h5ad(file)
        methods.append(os.path.basename(file).split('_')[1])
        ps.append(d.uns['clustercc_globalp'])
        nposs.append(d.uns['clustercc_npos'])
        nnegs.append(d.uns['clustercc_nneg'])
        ns.append(len(d))
    df = pd.DataFrame({'method':methods, 'p':ps, 'npos':nposs, 'nneg':nnegs, 'ntotal':ns}).set_index('method', drop=True)
    df.loc['vima'] = {'p':D.uns['vima_p'], 'npos':(D.obs.sig_mncoef > 0).sum(), 'nneg':(D.obs.sig_mncoef < 0).sum(), 'ntotal':len(D)}
    df['frac_pos'] = df.npos / df.ntotal
    df['frac_neg'] = df.nneg / df.ntotal
    df.sort_values('p', inplace=True)
    df.to_csv(outpath)
    print(df)