import pandas as pd

# to embed text as editable characters rather than low-level vector shapes
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42   # TrueType, not Type 3
matplotlib.rcParams['ps.fonttype']  = 42   # in case you ever export EPS

fs_axislabel = 10
fs_figsubpanel = 14
fs_legend = 8

methodnames = {'vima':'VIMA',
                'patchcelltypeabundance': 'Cell type ab.',
                'patchavgmm': 'Avg. Expr.',
                'canvas':'CANVAS',
                'stagate':'STAGATE',
                'utag':'UTAG',
                'tissuemosaic':'TissueMosaic',
                'cellcharter':'CellCharter',
                'cluster':'2-layer ConvNet, Clustering',
                'vima-nocvae-noresnet':'2-layer ConvNet, Microniches',
                'vima-nosid':'ResNet AE, no cVAE',
                }

def write_sourcedata(sourcedata, filename):
    with pd.ExcelWriter(filename) as writer:
        for name in sorted(sourcedata.keys()):
            df = sourcedata[name]
            sheet_name = str(name)[:31].replace(':', '.').replace('/', '_').replace('\\', '_')
            df.to_excel(writer, sheet_name=sheet_name, index=False)