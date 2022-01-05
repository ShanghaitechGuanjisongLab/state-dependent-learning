# to search permutation importance.
# first trained a model using control group data, and test the model in normal test_X, test_y to get a normal accuracy.
# then shuffle a feature, to boserve the feature importance to model.
# this file is for facr model.
#%%
from dataTransform import *
from model import *
from sklearn.model_selection import KFold
import parameters
import copy

positive_mat = parameters.CR_matfile
negative_mat = parameters.FA_matfile
downRegulation_path = parameters.FACR_downPath
upRegulation_path = parameters.FACR_upPath

model_save_path = parameters.experiment4_facr_modelOut_dir

positive_samples_train, negative_samples_train, positive_samples_test, negative_samples_test = dataTransformForSingleModel(positive_mat, negative_mat, downRegulation_path, upRegulation_path)

# generate test data set
new_positive_test, new_negative_test = expandSmallData(positive_samples_test, negative_samples_test)
test_X, test_y = generateraw(new_positive_test, new_negative_test)
# determine which samples should be kflod.
pos_sample_num, neg_sample_num = positive_samples_train.shape[0], negative_samples_train.shape[0]
kfold_num = round(pos_sample_num/neg_sample_num) if pos_sample_num>neg_sample_num else round(neg_sample_num/pos_sample_num)
kf = KFold(n_splits=kfold_num, shuffle=True, random_state=0)
if pos_sample_num > neg_sample_num:
    samples_retain, samples_toKfold = negative_samples_train, positive_samples_train
else:
    samples_retain, samples_toKfold = positive_samples_train, negative_samples_train
sample_num = 1
single_mouse_performance = []
for _, downsample_train_index in kf.split(samples_toKfold):
    new_class1_train = samples_toKfold[downsample_train_index,:]
    new_class2_train = samples_retain
    train_X, train_y = generateraw(new_class1_train, new_class2_train)
    # set parameter
    layers = 1
    num_input = train_X.shape[1]
    num_output = 2
    num_epochs = 500
    fold = 2
    dropout = 0.2
    std = 0.1
    lr = 0.0001
    weight_decay=0.025
    modelSave = os.path.join(model_save_path, r'crop1_facr_sample{}_layer{}_weightdeacy{}_lr{}_drop{}_std{}.pt'
                                            .format(sample_num, layers, weight_decay, lr, dropout, std))
    work = newWrok()
    net = work.createNet(layers, num_input, fold, num_output, dropout, std)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    df = work.train(net, train_X, train_y, test_X, test_y, loss, num_epochs, 60, params=None, lr=lr, optimizer=optimizer, schedule=None, saveModelPath=modelSave)
    single_mouse_performance.append(np.mean(df['test acc'][int(-min(num_epochs/4, 100)):]))
    sample_num +=1
#%%
best_per = 0
for cur in os.listdir(model_save_path):
    cur_model = work.createNet(layers, num_input, fold, num_output, dropout, std).to(DEVICE)
    cur_model.load_state_dict(torch.load(os.path.join(model_save_path, cur)))
    normal_performance = work.predict_prob(cur_model, test_X, test_y)
    if normal_performance>best_per:
        best_model = cur_model
        best_per = normal_performance
print("mean performance in best model: ", best_per)
#%%
# after train a model, now shuffle each feature, to get permutation importance of each feature
sample_times = 100
result_array = np.zeros((test_X.shape[1], sample_times)) # every column represent a sample
for s in range(sample_times):
    for f in range(test_X.shape[1]):
        new_test_X = copy.deepcopy(test_X)
        shuffle(new_test_X[:,f])
        acc = work.predict_prob(best_model, new_test_X, test_y)
        influence = best_per-acc
        result_array[f, s] = influence
        del new_test_X
    print("sample {} done".format(s))
mean_influence = np.mean(result_array, axis=1).reshape(-1, 1)
result = np.concatenate((mean_influence, result_array), axis=1)
columns = ['meanInfluence'] + ['sample{}'.format(i) for i in range(sample_times)]
#%%
df = pd.DataFrame(data=result, columns=columns)
df.to_csv(os.path.join(parameters.experiment4_facr_csvout_dir, r"crop1_facr_sample{}_layer{}_weightdeacy{}_lr{}_drop{}_std{}.csv"
                        .format(sample_times, layers, weight_decay, lr, dropout, std)))

# %%



