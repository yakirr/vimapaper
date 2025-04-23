Overall structure of benchmarking code:
1) `1.prepdata.ipynb` is reponsible for creating all versions of each dataset required by all methods. This is written to `_data/ALZ`, `_data/UC`, etc.
1) For each method, there is a notebook called `2.methodname.ipynb` that is responsible for using the method to create some sort of embedding of either patches, cells, or spots. This embedding needs to be an AnnData object whose `obs` field contains the following fields: `sid` specifying sample id, `donor` specifying donor id, `leiden` specifying leiden cluster, and optionally `method_cluster` if there is some additional method-specific cluster assignment. This embedding also needs to have a nearest-neighbor graph and UMAP representation computed. The result should be saved as `_embeddings/DATASETNAME_METHODNAME_noharm.h5ad` (with the "noharm" suffix indicating that harmony has not been run on this representation).
1) `3.harmonize.ipynb` is responsible for taking all embeddings and producing a harmonized version of each of them.
1) `4.assess.ipynb` is responsible for taking all embeddings, both harmonized and not, conducting association testing, and saving the results.

The following directories are untracked to do file size issues but are used by the code above:
1) `_data/DATASETNAME` stores, for each dataset, all of the different versions/formats of the data required by the different methods
2) `_embeddings` stores embeddings produced by all notebooks of the form `2.*.ipynb` and the subsequent harmonized embeddings produced by `3.harmonize.ipynb`.
3) `_results` stores the results produced by `4.assess.ipynb`.