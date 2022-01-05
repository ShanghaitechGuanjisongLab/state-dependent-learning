#%%
from typing import final
from numpy.lib.function_base import copy
import torch
from torch import device, dropout, nn
from torch.nn import init
import pandas as pd
import time
import random
import numpy as np

DEVICE = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
#%%
def generateraw(positive_samples:np.array, negative_samples:np.array):
    """merge positive samples and negative samples, and transform positive_samples and negative_sample which come from dataTransform to torch tensor
    Args:
        positive_samples (np.array): positive sample, with feature and label, last column represent label, others columns are features
        negtive_samples (np.array): netative sample. positive_sample number should equal to negative_sample number.
    Returns:
        [type]: features and labels.
    """
    all_samples = np.concatenate((positive_samples, negative_samples), axis=0)
    index = list(range(all_samples.shape[0]))
    random.shuffle(index)
    all_samples = all_samples[index, :]
    labels = torch.LongTensor(all_samples[:,-1].squeeze())
    features = np.array(np.delete(all_samples, -1, 1), dtype=np.float32)
    features = torch.from_numpy(features)
    features2 = features.expand(1, features.shape[0], features.shape[1])
    features2 = features2.permute(1,2,0)
    return features2, labels
def data_iter(batch_size, features, labels):
    num_expmples = len(features)
    indices = list(range(num_expmples))
    random.shuffle(indices)  # 样本读取顺序是随机的
    for i in range(0, num_expmples, batch_size):
        j = torch.LongTensor(indices[i:min(i + batch_size, num_expmples)])
        batchfeature = features.index_select(0,j)
        batchfeature1 = batchfeature.expand(1, batchfeature.shape[0], batchfeature.shape[1], batchfeature.shape[2])
        batchfeature2 = batchfeature1.permute(1,0,2,3)
        yield batchfeature2, labels.index_select(0, j)

def expandSmallData(positive_samples:np.array, negative_samples:np.array):
    len_p = positive_samples.shape[0]
    len_n = negative_samples.shape[0]
    if len_p > len_n:
        copy_num = round(len_p/len_n)
        new_negative = None
        for _ in range(copy_num):
            copied = np.copy(negative_samples)
            if new_negative is None:
                new_negative = copied
            else:
                new_negative = np.concatenate((new_negative, copied), axis=0)
        return positive_samples, new_negative
    else:
        copy_num = round(len_n/len_p)
        new_positive = None
        for _ in range(copy_num):
            copied = np.copy(positive_samples)
            if new_positive is None:
                new_positive = copied
            else:
                new_positive = np.concatenate((new_positive, copied), axis=0)
        return new_positive, negative_samples


#%%
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)
class newWrok():
    def __init__(self) -> None:
        self.net = None
        self.max_test = 0.5

    def createNet(self, layers, num_inputs, fold, num_output, dropout, std):
        if layers <=1:
            self.net = nn.Sequential(FlattenLayer(),
                                nn.BatchNorm1d(num_inputs),
                                nn.Dropout(dropout),
                                nn.Linear(num_inputs, num_inputs//fold),
                                nn.ReLU(),
                                nn.BatchNorm1d(num_inputs//fold),
                                nn.Dropout(dropout),
                                nn.Linear(num_inputs//fold, num_output))     
        else:
            self.net = nn.Sequential(FlattenLayer(),
                            nn.BatchNorm1d(num_inputs),
                            )
            for layer in range(layers):
                num_hiddens = num_inputs//fold
                self.net.add_module('linear{}'.format(layer), nn.Linear(num_inputs, num_hiddens))
                self.net.add_module('func{}'.format(layer), nn.ReLU())
                self.net.add_module('BN{}'.format(layer), nn.BatchNorm1d(num_hiddens))
                self.net.add_module('DP{}'.format(layer), nn.Dropout(dropout))
                num_inputs = num_hiddens
            self.net.add_module('outlinear', nn.Linear(num_hiddens, num_output))
            for params in self.net.parameters():
                init.normal_(params, mean=0, std=std)
        return self.net

    def train(self, net, train_X, train_y, test_X, test_y, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None, schedule=None,saveModelPath=None):
        epochlist = []
        losslist = []
        trainacclist = []
        testacclist = []
        timelist = []
        best_test = 0
        net = net.to(device=DEVICE)
        for epoch in range(0, num_epochs):
            start = time.time()
            train_iter = data_iter(batch_size, train_X, train_y)
            test_iter = data_iter(batch_size, test_X, test_y)
            train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
            for X, y in train_iter:
                X = X.to(device=DEVICE)
                y = y.to(device=DEVICE)
                y_hat = net(X)
                l = loss(y_hat, y).sum()
                if optimizer is not None:
                    optimizer.zero_grad()
                elif params is not None and params[0].grad is not None:
                    for param in params:
                        param.grad.data.zero_()
                l.backward()
                if optimizer is None:
                    torch.optim.sgd(params, lr, batch_size)
                else:
                    optimizer.step()
                    if schedule:
                        schedule.step()
                train_l_sum += l.item()
                train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
                # print(y.shape)
                n += y.shape[0]
            test_acc = self.evaluate_accuracy(test_iter, net)
            if saveModelPath:
                if test_acc > best_test:
                    torch.save(net.state_dict(), saveModelPath)
                    best_test = test_acc
            interval = time.time() - start
            if epoch%50==0:
                print('epoch %d, loss%.4f, trian acc %.3f, test acc %.3f' %
                    (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
            epochlist.append(epoch+1)
            losslist.append(train_l_sum / n)
            trainacclist.append(train_acc_sum / n)
            testacclist.append(test_acc)
            timelist.append(interval)
        df = pd.DataFrame({'epoch': epochlist, 
                            'loss': losslist,
                            'train acc': trainacclist,
                            'test acc': testacclist,
                            'time': timelist}
                            )
        return df

    def evaluate_accuracy(self, data_iter, net, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        acc_sum, n = 0.0, 0
        # with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device=DEVICE)
            y = y.to(device=DEVICE)
            if isinstance(net, torch.nn.Module):
                net.eval() 
                acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
                net.train()
            else:
                if ('is_training' in net.__code__.co_varnames):
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            # print(y.shape)
            n += y.shape[0]
        return acc_sum / n

    def predict_prob(self,net,test_X, test_y):
        acc_sum, n = 0.0, 0
        # with torch.no_grad():
        test_iter = data_iter(60, test_X, test_y)
        for X, y in test_iter:
            X = X.to(device=DEVICE)
            y = y.to(device=DEVICE)
            if isinstance(net, torch.nn.Module):
                net.eval() 
                acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
        return acc_sum / n

#%%
if __name__ == "__main__":
    from dataTransform import *
    import parameters
    CR_samples_train, FA_samples_train, CR_samples_test, FA_samples_test = dataTransformForSingleModel(parameters.CR_matfile, parameters.FA_matfile, parameters.FACR_downPath, parameters.FACR_upPath)

    new_CR_train, new_FA_train = expandSmallData(CR_samples_train, FA_samples_train)
    new_CR_test, new_FA_test = expandSmallData(CR_samples_test, FA_samples_test)

    train_X, train_y = generateraw(new_CR_train, new_FA_train)
    test_X, test_y = generateraw(new_CR_test, new_FA_test)

    layers = 2
    num_input = train_X.shape[1]
    num_output = 2
    num_hiddens = 2*num_input
    dropout = 0.4
    lr = 0.01
    weight_decay = 0.025
    work = newWrok()
    net = work.createNet(layers, num_input,num_hiddens ,num_output, dropout)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    num_epochs = 20
    df = work.train(net, train_X, train_y, test_X, test_y, loss=loss, num_epochs=num_epochs,
                    batch_size=30, params=None, lr=lr, optimizer=optimizer, schedule=None)