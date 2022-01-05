# single mouse cross validation.
# select one mouse data as test set, others as training data.
# this file is for hitmiss model.
import parameters
from model import *
from dataTransform import *
from sklearn.model_selection import KFold


positive_mat = parameters.hit_matfile
negative_mat = parameters.miss_matfile
downRegulation_path = parameters.hitmiss_downPath
upRegulation_path = parameters.hitmiss_upPath

model_save_path = parameters.experiment2_hitmiss_modelOut_dir

mouseid_list = PreprocessMouseCrossValidate.getAllMouseId(positive_mat)
for mouseid in mouseid_list:
    positive_samples_train, negative_samples_train, positive_samples_test, negative_samples_test = dataTransformForMouseCV(positive_mat, negative_mat, downRegulation_path, upRegulation_path, mouseid)
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
        modelSave = os.path.join(model_save_path, r'mouseid{}_crop1_hitmiss_sample{}_layer{}_weightdeacy{}_lr{}_drop{}_std{}.pt'
                                            .format(mouseid,sample_num, layers, weight_decay, lr, dropout, std))
        work = newWrok()
        net = work.createNet(layers, num_input, fold, num_output, dropout, std)
        loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        df = work.train(net, train_X, train_y, test_X, test_y, loss, num_epochs, 60, params=None, lr=lr, optimizer=optimizer, schedule=None, saveModelPath=modelSave)
        single_mouse_performance.append(np.mean(df['test acc'][int(-min(num_epochs/4, 100)):]))
        df.to_csv(os.path.join(parameters.experiment2_hitmiss_csvout_dir, r"mouseid{}_crop1_hitmiss_sample{}_layer{}_weightdeacy{}_lr{}_drop{}_std{}.csv"
                                .format(mouseid, sample_num, layers, weight_decay, lr, dropout, std)))
        sample_num +=1


