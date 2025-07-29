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

def collate_cc(dsetname, D=None, outpath=None):
    results = []
    for f in glob.glob(f'../benchmark/_results/{dsetname}*_noharm.h5ad'):
        method = os.path.basename(f).split('_')[1]
        d = sc.read_h5ad(f)
        results.append({'method': method,
                        'microniche':False,
                        'harmonized':False,
                        'bonferroni':False,
                        'p': d.uns['clustercc_globalp'],
                        'npos': d.uns['clustercc_npos'],
                        'nneg': d.uns['clustercc_nneg'],
                        'ntotal': len(d)
                        })
        results.append({'method': method,
                        'microniche':True,
                        'harmonized':False,
                        'bonferroni':False,
                        'p': d.uns['mncc_p'],
                        'npos': d.uns['mncc_npos'],
                        'nneg': d.uns['mncc_nneg'],
                        'ntotal': len(d)
                        })
        results.append({'method': method,
                        'microniche':False,
                        'harmonized':False,
                        'bonferroni':True,
                        'p': min(d.uns['clustercc_minp'], 1),
                        'npos': np.nan,
                        'nneg': np.nan,
                        'ntotal': len(d)
                        })
        
        for f in glob.glob(f'_results/{dsetname}*_harm.h5ad'):
            method = os.path.basename(f).split('_')[1]
            d = sc.read_h5ad(f)
            results.append({'method': method,
                            'microniche':False,
                            'harmonized':True,
                            'bonferroni':False,
                            'p': d.uns['clustercc_globalp'],
                            'npos': d.uns['clustercc_npos'],
                            'nneg': d.uns['clustercc_nneg'],
                            'ntotal': len(d)
                            })

    if D is not None:
        results.append({'method': 'vima',
                        'p': D.uns['vima_p'],
                        'npos': (D.obs.sig_mncoef > 0).sum(),
                        'nneg': (D.obs.sig_mncoef < 0).sum(),
                        'ntotal': len(D)
        })

    results = pd.DataFrame(results).set_index('method')
    results['frac_pos'] = results.npos / results.ntotal
    results['frac_neg'] = results.nneg / results.ntotal
    results.sort_values('p', inplace=True)
    mainresults = results[
        ~(results.microniche==True) & ~(results.harmonized==True) & ~(results.bonferroni==True)
        ][['p','npos','nneg','ntotal','frac_pos','frac_neg']]

    if outpath is not None:
        supp_outpath = outpath.rsplit('.', 1)
        supp_outpath = supp_outpath[0] + '_supp.' + supp_outpath[1]
        results.to_csv(supp_outpath)
        mainresults.to_csv(outpath)

    return results
    