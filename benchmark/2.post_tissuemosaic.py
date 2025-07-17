import scanpy as sc 
import anndata as ad 
import glob, sys

dset = sys.argv[1] 

def preprocess_featurized_h5ad(file):
    d = sc.read_h5ad(file)
    d = d[d.obs.dino_spot_features_valid == True]
    return(d)

d = ad.concat([preprocess_featurized_h5ad(file) for file in glob.glob(f'_data/{dset}/sample_ads_featurized/*')])
sc.pp.scale(d) 
sc.tl.pca(d, key_added='dino_PCA') 
sc.pp.neighbors(d, use_rep='dino_PCA', key_added='dino_neighbors')
sc.tl.umap(d, use_rep='dino_neighbors')
sc.tl.leiden(d, resolution=1, niterations=2, key_added='leiden1')
d.write_h5ad(f'_embeddings/{dset}_tm_final.h5ad')
