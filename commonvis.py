import pandas as pd

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