import anndata as ad
import squidpy as sq 
import cellcharter as cc
import pandas as pd
import scanpy as sc
import numpy as np 
import scvi 
from lightning.pytorch import seed_everything 
import os, sys, gc 

seed_everything(12345)
scvi.settings.seed = 12345
it = 1

def run_cellcharter(dsetname):
    os.makedirs(f'_embeddings', exist_ok=True)
    d = sc.read_h5ad(f'_data/{dsetname}/cells_counts.h5ad')
    print(d.shape)

    sc.pp.filter_genes(d, min_counts=3)
    sc.pp.filter_cells(d, min_counts=3)

    print(type(d))
    if dsetname == 'ALZ': 
        d.layers['counts'] = d.X.copy()
        sc.pp.normalize_total(d, target_sum=np.median(d.X.sum(axis=1)))
        sc.pp.log1p(d)
        
        scvi.model.SCVI.setup_anndata(
            d, 
            layer="counts", 
            batch_key='sid',
        )

        model = scvi.model.SCVI(d)
        model.train(early_stopping=True, enable_progress_bar=True)
        rep = 'X_scVI'
        d.obsm[rep] = model.get_latent_representation(d).astype(np.float32)
    else:
        d.layers['raw'] = d.X.copy()
        for samp in d.obs['sid'].cat.categories: 
            d.X[d.obs.sid == samp] = sc.pp.scale(d[d.obs.sid == samp], copy=True).X 
        
        condition_key = 'sid'
        trvae_epochs = 10 
        early_stopping_kwargs = {
            "early_stopping_metric": "val_unweighted_loss",
            "threshold": 0,
            "patience": 2,
            "reduce_lr": True,
            "lr_patience": 13,
            "lr_factor": 0.1,
        }

        from scarches.dataset.trvae.data_handling import remove_sparsity
        d = remove_sparsity(d) # Necessary step in trVAE documentation 
        source_conditions = d.obs.sid.unique().tolist()

        d.X = d.X.astype(np.float32)
        trvae = cc.tl.TRVAE(
            adata=d, 
            condition_key=condition_key, 
            conditions=source_conditions, 
            hidden_layer_sizes=[128, 128], 
            recon_loss='mse'
        )

        trvae.train(
            n_epochs=trvae_epochs,
            alpha_epoch_anneal=trvae_epochs,
            early_stopping_kwargs=early_stopping_kwargs
        )

        rep = 'X_trVAE'
        d.obsm[rep] = trvae.get_latent(d.X, d.obs['sid'])

    sq.gr.spatial_neighbors(d, library_key='sid', coord_type='generic', delaunay=True)
    cc.gr.remove_long_links(d)
    cc.gr.aggregate_neighbors(d, n_layers=3, use_rep=rep, out_key='X_cellcharter', sample_key='sid')

    autok = cc.tl.ClusterAutoK(
        n_clusters=(2,10), 
        max_runs=10,
        convergence_tol=0.001
    )
    autok.fit(d, use_rep='X_cellcharter')
    d.obs['cluster_method'] = autok.predict(d, use_rep='X_cellcharter')

    sc.pp.neighbors(d, use_rep='X_cellcharter') 
    sc.tl.umap(d)
    sc.tl.leiden(d, resolution=1, key_added='leiden1') 
    d.write(f'_embeddings/{dsetname}_cellcharter_noharm.h5ad')

if __name__ == '__main__': 
    dset = sys.argv[1]
    run_cellcharter(dset) 
