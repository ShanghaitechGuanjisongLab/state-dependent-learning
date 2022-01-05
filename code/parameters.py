from genericpath import exists
import os
import sys
import torch
# experiment 1 
# constructing a model use control group data, and test the model using control group data,
# in which training set and test set are seperated in ratio 8:2.

# experiment 2
# single mouse cross validation.
# select one mouse data as test set, others as training data.

# experiment 3
# test model which trained in Control Group data, using Experiment Group data.

# experiment 4
# to search permutation importance.
# first trained a model using control group data, and test the model in normal test_X, test_y to get a normal accuracy.
# then shuffle a feature, to boserve the feature importance to model.

rootpath = r"E:\fmri_new_version"
sys.path.append(os.path.join(rootpath, 'code'))
DEVICE = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
########################################################################## Control Group
# FA and CR model
CR_matfile = os.path.join(rootpath, r"dataoriconmat\con_CNOandPBS\con_CNOandPBS_CR_baselinedata.mat")
FA_matfile = os.path.join(rootpath, r"dataoriconmat\con_CNOandPBS\con_CNOandPBS_FA_baselinedata.mat")
# FACR_downPath = os.path.join(rootpath, r"data_allarea_cutoff140\FA_CRBothDown_joinedarea0.08.npy")
# FACR_upPath = os.path.join(rootpath, r"data_allarea_cutoff140\FA_CRBothUp_joinedarea0.08.npy")
FACR_downPath = os.path.join(rootpath, r"data_allarea_cutoff140\FA_CRBothDown_joinedarea0.05.npy")
FACR_upPath = os.path.join(rootpath, r"data_allarea_cutoff140\FA_CRBothUp_joinedarea0.05.npy")

# Hit and Miss model
hit_matfile = os.path.join(rootpath, r"dataoriconmat\con_CNOandPBS\con_CNOandPBS_Hit_baselinedata.mat")
miss_matfile = os.path.join(rootpath, r"dataoriconmat\con_CNOandPBS\con_CNOandPBS_Miss_baselinedata.mat")
# hitmiss_downPath = os.path.join(rootpath, r"data_allarea_cutoff140\HIT_MISSBothDown_joinedarea0.08.npy")
# hitmiss_upPath = os.path.join(rootpath, r"data_allarea_cutoff140\HIT_MISSBothUp_joinedarea0.08.npy")
hitmiss_downPath = os.path.join(rootpath, r"data_allarea_cutoff140\HIT_MISSBothDown_joinedarea0.03.npy")
hitmiss_upPath = os.path.join(rootpath, r"data_allarea_cutoff140\HIT_MISSBothUp_joinedarea0.03.npy")
##########################################################################Control Group

##########################################################################Experiment Group
exp_PBS_CR_matfile = os.path.join(rootpath, r"dataoriexpmat\exppbs\exp_PBS_CR_baselinedata.mat")
exp_PBS_FA_matfile = os.path.join(rootpath, r"dataoriexpmat\exppbs\exp_PBS_FA_baselinedata.mat")
exp_PBS_hit_matfile = os.path.join(rootpath, r"dataoriexpmat\exppbs\exp_PBS_Hit_baselinedata.mat")
exp_PBS_miss_matfile = os.path.join(rootpath, r"dataoriexpmat\exppbs\exp_PBS_Miss_baselinedata.mat")
exp_CNO_CR_matfile = os.path.join(rootpath, r"dataoriexpmat\expcno\exp_CNO_CR_baselinedata.mat")
exp_CNO_FA_matfile = os.path.join(rootpath, r"dataoriexpmat\expcno\exp_CNO_FA_baselinedata.mat")
exp_CNO_hit_matfile = os.path.join(rootpath, r"dataoriexpmat\expcno\exp_CNO_Hit_baselinedata.mat")
exp_CNO_miss_matfile = os.path.join(rootpath, r"dataoriexpmat\expcno\exp_CNO_Miss_baselinedata.mat")

##########################################################################Experiment Group

# output path
csv_outdir = r"csvOutPut"

experiment1_facr_csvout_dir = os.path.join(rootpath, csv_outdir, r"experiment1_FACR")
if not os.path.exists(experiment1_facr_csvout_dir):
    os.mkdir(experiment1_facr_csvout_dir)

experiment1_hitmiss_csvout_dir = os.path.join(rootpath, csv_outdir, r"experiment1_hitmiss")
if not os.path.exists(experiment1_hitmiss_csvout_dir):
    os.mkdir(experiment1_hitmiss_csvout_dir)

experiment2_hitmiss_csvout_dir = os.path.join(rootpath, csv_outdir, r"experiment2_hitmiss")
experiment2_facr_csvout_dir = os.path.join(rootpath, csv_outdir, r"experiment2_FACR")
if not os.path.exists(experiment2_hitmiss_csvout_dir):
    os.mkdir(experiment2_hitmiss_csvout_dir)
if not os.path.exists(experiment2_facr_csvout_dir):
    os.mkdir(experiment2_facr_csvout_dir)

experiment3_hitmiss_csvout_dir = os.path.join(rootpath, csv_outdir, r"experiment3_hitmiss")
experiment3_facr_csvout_dir = os.path.join(rootpath, csv_outdir, r"experiment3_FACR")
if not os.path.exists(experiment3_hitmiss_csvout_dir):
    os.mkdir(experiment3_hitmiss_csvout_dir)
if not os.path.exists(experiment3_facr_csvout_dir):
    os.mkdir(experiment3_facr_csvout_dir)

experiment4_hitmiss_csvout_dir = os.path.join(rootpath, csv_outdir, r"experiment4_hitmiss")
experiment4_facr_csvout_dir = os.path.join(rootpath, csv_outdir, r"experiment4_FACR")
if not os.path.exists(experiment4_hitmiss_csvout_dir):
    os.mkdir(experiment4_hitmiss_csvout_dir)
if not os.path.exists(experiment4_facr_csvout_dir):
    os.mkdir(experiment4_facr_csvout_dir)

# model path
model_outdir = r"modelOutPut"
experiment1_facr_modelOut_dir = os.path.join(rootpath, model_outdir, r"experiment1_FACR")
experiment1_hitmiss_modelOut_dir = os.path.join(rootpath, model_outdir, r"experiment1_hitmiss")

experiment2_hitmiss_modelOut_dir = os.path.join(rootpath, model_outdir, r"experiment2_hitmiss")
experiment2_facr_modelOut_dir = os.path.join(rootpath, model_outdir, r"experiment2_FACR")

experiment3_hitmiss_modelOut_dir = os.path.join(rootpath, model_outdir, r"experiment3_hitmiss")
experiment3_facr_modelOut_dir = os.path.join(rootpath, model_outdir, r"experiment3_FACR")

experiment4_hitmiss_modelOut_dir = os.path.join(rootpath, model_outdir, r"experiment4_hitmiss")
experiment4_facr_modelOut_dir = os.path.join(rootpath, model_outdir, r"experiment4_FACR")

for i in [experiment1_hitmiss_modelOut_dir, experiment1_facr_modelOut_dir,
          experiment2_hitmiss_modelOut_dir, experiment2_facr_modelOut_dir,
          experiment3_hitmiss_modelOut_dir, experiment3_facr_modelOut_dir,
          experiment4_hitmiss_modelOut_dir, experiment4_facr_modelOut_dir,
            ]:
    if not os.path.exists(i):
        os.mkdir(i)

