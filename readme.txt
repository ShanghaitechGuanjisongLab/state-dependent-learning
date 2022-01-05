folder description：
code folder: Contains code for all model data preprocessing and modeling. 
	the file experiment_* represent the main test for  this research, the function of every experiment_*.py file is writen in file head. if want to execute a experiment, just run the experiment_*.py file.
	the coefficient_analysis.py, to get the area-pairs whos perseaon's coefficient have same trend in control group.
	the dataTransform.py, preprocess data before modeling. The preprocess method is the same as introduced in aritice method.
	the parameters.py, contains many parameter that you would adjust.
	the showimg.ipynb contains the main result shows related to experiment.

csvOutput folder: Contains all csv output during training model. The subfolder in it contains different experiment ouput.

data_allarea_cutoff140 folder: the output folder of coefficient_analysis.py. 

dataoriconmat folder: control group data, including control group with CNO injection or PBS injection. The detail of experiment group is introduced in article.

dataoriexpmat folder: experiment group data, including experiment group with CNO injection or PBS injection. The detail of control group is introduced in article

samplearea.txt: including all recorded brain region, the sequence of brain region is important.
