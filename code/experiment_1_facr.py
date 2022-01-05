# use all control mouse as data set, including pbs and cno injection during test.hitmiss model. FACR model.
#%%
import parameters
import dataTransform
from dataTransform import *
from model import *
from sklearn.model_selection import KFold
#%%
model_save_path = parameters.experiment1_facr_modelOut_dir
CR_samples_train, FA_samples_train, CR_samples_test, FA_samples_test = dataTransformForSingleModel(parameters.CR_matfile, parameters.FA_matfile, parameters.FACR_downPath, parameters.FACR_upPath)
#%%
new_CR_test, new_FA_test = expandSmallData(CR_samples_test, FA_samples_test)
test_X, test_y = generateraw(new_CR_test, new_FA_test)
#%%
kfold_num = 10
kf = KFold(n_splits=kfold_num, shuffle=True, random_state=0)
sample_num = 1
for _, downsample_train_index in kf.split(CR_samples_train):
    new_CR_train = CR_samples_train[downsample_train_index,:]
    new_FA_train = FA_samples_train
    # print(new_CR_train.shape, new_FA_train.shape)
    train_X, train_y = generateraw(new_CR_train, new_FA_train)
    num_input = train_X.shape[1]
    num_output = 2
    num_epochs = 500
    fold = 2
    max_test = 0
    layers = 1
    dropout = 0.2
    weight_decay = 0.025
    lr = 0.0001
    std = 0.1
    modelSave = os.path.join(model_save_path, r"crop1_FACR_sample{}_layer{}_weightdeacy{}_lr{}_drop{}_std{}.pt"
                                            .format(sample_num, layers, weight_decay, lr, dropout, std))
    # for layers in [1]:
    #     for dropout in [0.1, 0.2, 0.4]:
    #         for weight_decay in [0.05, 0.025]:
    #             for lr in [0.0001]:
    #                 for std in [0.01, 0.1]:
    work = newWrok()
    net = work.createNet(layers, num_input, fold, num_output, dropout, std)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    df = work.train(net, train_X, train_y, test_X, test_y, loss=loss, num_epochs=num_epochs,
                    batch_size=60, params=None, lr=lr, optimizer=optimizer, schedule=None, saveModelPath=modelSave)
    mean_test = np.mean(df['test acc'][int(-min(num_epochs/4, 100)):])
    if mean_test > max_test:
        max_test_df = df
        max_test = mean_test
        print('mean_test:', mean_test)
        final_layers, final_weight_decay, final_lr, final_dropout, final_std = layers, weight_decay, lr, dropout, std
        print("layer{}, weight_decay{}, lr{}, dropout{}, std{}".format(final_layers, final_weight_decay, final_lr, final_dropout, final_std))
    max_test_df.to_csv(os.path.join(parameters.experiment1_facr_csvout_dir, r'crop1_FACR_sample{}_layer{}_weightdeacy{}_lr{}_drop{}_std{}.csv'
                        .format(sample_num, final_layers, final_weight_decay, final_lr, final_dropout, final_std)))
    sample_num += 1
    print(sample_num)
    
# %%
