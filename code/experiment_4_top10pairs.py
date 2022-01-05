# run this file after experiment_4_hitmiss or experiment_4_facr.
#%%
from random import paretovariate
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
import pandas as pd
import numpy as np
import parameters
import os
from dataTransform import PreprocessBeforeModel
from coefficient_analysis import clean_en_text
import parameters
#%%
# comment one of two below.
cls = 'hitmiss'
# cls = 'facr'
if cls=='hitmiss':
    downRegulate_path = parameters.hitmiss_downPath
    upRegulate_path = parameters.hitmiss_upPath
    file_path = os.path.join(parameters.experiment4_hitmiss_csvout_dir, 'crop1_hitmiss_sample100_layer1_weightdeacy0.025_lr0.0001_drop0.2_std0.1.csv')
    save_path = os.path.join(parameters.experiment1_hitmiss_csvout_dir, 'top10_hitmiss_areaPairs.csv')
else:
    downRegulate_path = parameters.FACR_downPath
    upRegulate_path = parameters.FACR_upPath
    file_path = os.path.join(parameters.experiment4_facr_csvout_dir, 'crop1_facr_sample100_layer1_weightdeacy0.025_lr0.0001_drop0.2_std0.1.csv')    
    save_path = os.path.join(parameters.experiment4_facr_csvout_dir, 'top10_facr_areaPairs.csv')
#%%
preprocess = PreprocessBeforeModel()
preprocess.loadRelateAreaPair(downRegulate_path, upRegulate_path)
mapindex = preprocess.getcoefmatindx()
print(mapindex.shape)
#%%
# according the area index filtered above to get brain area name.
with open(os.path.join(parameters.rootpath, "samplearea.txt")) as f:
        arealist = f.readlines()
        samplearea = [clean_en_text(a) for a in arealist]
df = pd.read_csv(file_path, index_col=0)
data = df['meanInfluence'].values
index = np.argsort(data)[::-1]
top10_pairs = mapindex[index[:10], :]
area1 = np.array([samplearea[i] for i in top10_pairs[:,0]]).reshape(-1, 1)
area2 = np.array([samplearea[i] for i in top10_pairs[:,1]]).reshape(-1, 1)
columns = ['area1_index', 'area2_index', 'area1', 'area2']
result = pd.DataFrame(np.concatenate((top10_pairs, area1, area2), axis=1), columns=columns)
result.to_csv(save_path, index=None)
print(result)

# %%

