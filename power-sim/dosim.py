import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import os, pickle, torch
import tqdm
import vima
import argparse

# basic setup
# torch.set_default_device('mps') # edit as appropriate to hardware
vima.set_seed(0) # for reproducibility
print("Basic setup complete (vima seed set).")
os.makedirs(f'./_results', exist_ok=True)
mnsize = 400
pixelsize = 10

#arguments
parser = argparse.ArgumentParser(description='Run dosim simulation')
parser.add_argument('--ncase', type=int, default=40, help='number of case microniches')
parser.add_argument('--ratio', type=int, default=2, help='ratio of #case/#control microniches')
parser.add_argument('--datamask', type=str, default='../../ALZ/alz-data/10u/pca_k=10_harmony/*.nc', help='glob path to data')
parser.add_argument('--samplemeta', type=str, default='../../ALZ/alz-data/sea-ad_cohort_donor_metadata_encoded_20240924.tsv', help='path to sample metadata TSV (Donor ID as index)')
parser.add_argument('--seed', type=int, default=0, help='random seed')
args = parser.parse_args()
ncase = args.ncase
ratio = args.ratio
seed = args.seed
outname = f'ratio{ratio:.2f}_ncase{ncase}_seed{seed}'

print("Parsed arguments.")
print(args)
print(outname)

# read in samples
samples = vima.read_samples(args.datamask, vima.default_parser)
print(f"Read {len(samples)} samples from datamask: {args.datamask}")

# read in sample metadata
samplemeta = pd.read_csv(args.samplemeta, sep='\t').set_index('Donor ID', drop=True)
samplemeta['case'] = samplemeta['Consensus Clinical Dx (choice=Control)'] != 'Checked'
print(f"Loaded sample metadata for {len(samplemeta)} donors from {args.samplemeta}")

# filter samples to those with enough patches
Pdense = vima.PatchCollection(samples, max_frac_empty=0.5, standardize=False)
sid_to_donor = pd.DataFrame(pd.Series({s.sid: s.donor for s in samples.values()}), columns=['donor'])
sid_to_donor['npatches'] = Pdense.meta.sid.value_counts()
sid_to_donor.npatches = sid_to_donor.npatches.fillna(0)
filtered_sids = pd.merge(samplemeta, sid_to_donor, left_index=True, right_on='donor', how='right'
                         ).sort_values('npatches', ascending=False
                        ).drop_duplicates(subset='donor', keep='first').index
filtered_sids = filtered_sids[sid_to_donor.loc[filtered_sids,'npatches'] >= 100]
filtered_samples = {s.sid: s for s in samples.values() if s.sid in filtered_sids}
print(f"Filtered samples to {len(filtered_samples)} donors with >=100 patches")

# choose patch to use as microniche signal
indexsample = list(filtered_samples.values())[0]
mn = indexsample.sel(x=slice(5000,5000+mnsize), y=slice(5000,5000+mnsize)).copy()
hpc2 = indexsample.sel(marker='hPC2').data.flatten()
magnitude_added_to_hpc2 = 2*np.std(hpc2[hpc2 != 0])
magnitude_noise = magnitude_added_to_hpc2/10
mn.loc[dict(marker='hPC2')] += magnitude_added_to_hpc2
print("Selected index sample and extracted microniche patch (mn) with boosted hPC2 signal")

# add in case/ctrl signal
def add_signal(sample, nmns):
    dense_mask = vima.d.union_patches_in_sample(Pdense.meta, sample).data.copy()
    dense_mask[-mn.shape[0]:,:] = False
    dense_mask[:,-mn.shape[1]:] = False
    mask_y, mask_x = np.where(dense_mask)
    groundtruth = np.zeros_like(dense_mask)

    for i in range(nmns):
        idx = np.random.randint(0, len(mask_x))
        y0, x0 = mask_y[idx], mask_x[idx]
        
        signal = mn.copy() + np.random.randn(*mn.shape) * magnitude_noise
        sample.data[y0:y0+mn.shape[0], x0:x0+mn.shape[1]] = signal #indexing is [y, x]
        groundtruth[y0:y0+mn.shape[0], x0:x0+mn.shape[1]] = 1 #indexing is [y, x]
    return groundtruth
print("Defined add_signal function for injecting microniche signals")

print(f"Adding in signal with random seed {seed}")
np.random.seed(seed)
groundtruths = {}
samplemeta = pd.DataFrame(columns=['sid','case'])
for s in tqdm.tqdm(filtered_samples.values()):
    case = np.random.choice([True, False])
    nmns = int(sid_to_donor.loc[s.sid].npatches) // 100
    groundtruths[s.sid] = add_signal(s, nmns if case else nmns // ratio)
    samplemeta = pd.concat([
        samplemeta,
        pd.DataFrame({'sid':[s.sid], 'case':[case]})
    ], ignore_index=True)
samplemeta = samplemeta.set_index('sid', drop=True)
print(f"Added signal for {len(groundtruths)} samples")

#### Run method ####
print("Starting analysis: constructing patch collections and training pipeline")
samples = filtered_samples

# choose which patches to train on
P = vima.PatchCollection(samples)
print(len(P), 'patches selected for training')

# choose which patches to do case/ctrl analysis on
Pdense = vima.PatchCollection(samples, max_frac_empty=0.5, sid_nums=P.sid_nums)
print(len(Pdense), 'dense patches selected for case/control analysis')

# add in ground truth info
Pdense.meta['case'] = 0.
for sid in Pdense.meta.sid.unique():
    gt = groundtruths[sid]
    mask = Pdense.meta.sid == sid
    for idx, row in Pdense.meta[mask].iterrows():
        x, y, ps = row['x'], row['y'], row['patchsize']
        Pdense.meta.loc[idx, 'case'] = gt[y:y+ps, x:x+ps].mean()
print("Annotated Pdense.meta with ground-truth case scores")

# train model
print("Training models: phase 1 (short) starting")
models = vima.models.cVAE(P.nmarkers, P.nsamples)
log = vima.train(models, P, n_epochs=10)
print("Phase 1 training complete. Starting phase 2 (fine-tune)")
log2 = vima.train(models, Pdense, n_epochs=20)
torch.save(models.state_dict(), f'_results/{outname}_model.pt')
print(f"Training complete. Model saved to _results/{outname}_model.pt")

# load model
models = vima.models.cVAE(P.nmarkers, P.nsamples)
models.load_state_dict(torch.load(f'_results/{outname}_model.pt'))
print(f"Loaded model from _results/{outname}_model.pt")

# apply models and build nearest-neighbor graphs
print("Computing latent representations and associations; saving fingerprints and graph")
ds = vima.latentreps(models, Pdense)
pickle.dump(ds, open(f'_results/{outname}_fingerprints.pkl', 'wb'))
p, D = vima.association(ds, samplemeta.case.astype(float), 'sid')
D.write(f'_results/{outname}_D.h5ad')

print('done')