#%%
import sys
import scipy.io as io
import numpy as np
import random
from random import shuffle
from sklearn import preprocessing
import os

from torch._C import dtype
import parameters

#%%
class PreprocessBeforeModel():
    def __init__(self) -> None:
        self.areapair1 = None
        self.areapair2 = None
        self.areaindex = None
        self.coefmatindex = None

    def loadRelateAreaPair(self, BothDown_np_path:str, BothUp_np_path:str):
        BothDown = np.load(BothDown_np_path)
        BothUp = np.load(BothUp_np_path)
        self.areapair1, self.areapair2 = BothDown, BothUp
        self.areaindex = list(set(BothDown.reshape((-1,))).union(set(BothUp.reshape((-1,)))))

    def extract_mat(mat_file:str):
        # mat_file, hit trial or FA trial or CR or miss trial
        # extract data from singel mat file, and seperant data into train set and test set in ratio 8:2
        pos_temp = io.loadmat(mat_file)
        all_data = pos_temp["All_Data"] # (channel, t_point, trial)
        indx = np.arange(all_data.shape[2]).reshape((-1,))
        random.seed(0)
        shuffle(indx)
        # seperate data to train set and test set
        train_data_indx = indx[0:int((0.8*all_data.shape[2])//1)]
        test_data_indx = indx[int((0.8*all_data.shape[2])//1):]
        # only brain region 0-140 were selected to 
        train_data = all_data[:141, :, train_data_indx]
        test_data = all_data[:141, :, test_data_indx]
        # 
        return train_data, test_data

    def getcoefmatindx(self):
        row = len(self.areaindex)  # row = col
        coefmatidex = []
        print('loading area-pairs screened befroe')
        for i in range(row): 
            for j in range(row):
                for n in range(self.areapair1.shape[0]):
                    if (np.array([self.areaindex[i], self.areaindex[j]])== self.areapair1[n,:]).all():
                        coefmatidex.append((i, j))
                        # print("area1:{} ,area2:{}".format(joinedareaunion[i],joinedareaunion[j]))
                for m in range(self.areapair2.shape[0]):
                    if (np.array([self.areaindex[i], self.areaindex[j]]) == self.areapair2[m,:]).all():
                        coefmatidex.append((i,j))
                        # print("area1:{} ,area2:{}".format(joinedareaunion[i],joinedareaunion[j]))
        print("single col={}".format(len(coefmatidex)))
        return np.array(coefmatidex)

    def extractupstri(self, coefmat, coefmatindx):
        # 提取矩阵中固定位置的值，不包含对角线
        # joinedarea 参加的脑区对，n行，2列
        upstri = []
        for xy in coefmatindx:
            i, j = xy
            upstri.append(coefmat[i, j])
            # print("area1:{} ,area2:{}".format(self.areaindex[i],self.areaindex[j]))
        return np.array(upstri)

    def coef(self, meantrialmat,windowsize, coefmapindx):
        """compute coef map of a single trial, the size shoule be (len(coefmapindx), 10-windowsize+1)
        Parameters
        ----------
        meantrialmat : [np.array] 
            sigle trial mat, size(channel, t_poiot)
        windowsize : [int]
            frame numbers that are used to compute coefficient
        joinedareaunion : [type]
            [description]
        areapair1 : [type]
            [description]
        areapair2 : [type]
            [description]
        coefmapindx : [type]
            [description]
        """
        _, t_point = meantrialmat.shape
        coefmatlist = []
        for st in range(t_point-windowsize+1):
            submat = preprocessing.scale(meantrialmat[:, st:st+windowsize])
            coefmat = np.corrcoef(submat, rowvar=True)
            upstricoef = self.extractupstri(coefmat, coefmapindx)
            coefmatlist.append(upstricoef)
        coefmatlist = np.array(coefmatlist)
        return coefmatlist.T

    def generateFeatureForSingleTrial(self, coefmap,positive_label:bool,crop=[-1])->np.array:
        """[according to a coefmap of a sigle trial to generate samples which input to network model directly]

        Parameters
        ----------
        coefmap : np.array
            coefficientmap, size should be (len(coefmapindex),(11-window)), in which each column represent a coefficent of two region coeff in a windowsize.
        positive_label : bool
            this label is positive label or negative label
        crop : list, optional
            used to expand data, usd the columns of data in coefmap two or more samples with same label, by default [-1]
        Returns
        -------
        np.array()
            samples of which the number is equal to length of crop.
        """
        features = coefmap[:,crop].reshape(len(crop), -1)
        if positive_label:
            labels = np.ones((len(crop),), dtype=int)
        else:
            labels = np.zeros((len(crop),), dtype=int)
        featureWithLabel = np.concatenate((features, labels.reshape(-1, 1)), axis=1)
        return featureWithLabel

    def generateFeature(self, data_set:np.array, positive_label:bool)->np.array:
        """generate final samples which could input to model according data_set which extract from mat_file.

        Parameters
        ----------
        data_set : np.array
            a data set that come from function extract_mat. Size should be (number of channels(brain regions), time_point, trials)
        positive_label : bool
            this data set is poistive dataset or negative dataset.
        Returns
        -------
        np.array
            the final samples that could input to netword model, the final column are labels, others columns are features.
        """
        trialnum = data_set.shape[2]
        if self.coefmatindex is None:
            self.coefmatindex = self.getcoefmatindx()
        samplesToModel = None
        for t in range(trialnum):
            sigle_mat = np.squeeze(data_set[:, :, t])
            coefmap = self.coef(sigle_mat, 5, self.coefmatindex)
            singleTrileToSamples = self.generateFeatureForSingleTrial(coefmap, positive_label, crop=[-1])
            samplesToModel = singleTrileToSamples if samplesToModel is None else np.concatenate((samplesToModel, singleTrileToSamples), axis=0)
        return samplesToModel

class PreprocessMouseCrossValidate():
    def __init__(self) -> None:
        pass

    def getAllMouseId(mat_file):
        pos_temp = io.loadmat(mat_file)
        re = pos_temp["re"].reshape(-1, )
        all_mouse_id = list(set(re))
        return all_mouse_id
    @staticmethod
    def extract_mat(mat_file:str, test_mouse_id):
        pos_temp = io.loadmat(mat_file)
        all_data = pos_temp["All_Data"]
        re = pos_temp["re"].reshape(-1, )
        index = np.arange(0, len(re),dtype=int)
        train_data = all_data[:141, :, index[test_mouse_id!=re]]
        test_data = all_data[:141, :, index[test_mouse_id==re]]
        return train_data, test_data

       
def dataTransformForMouseCV(positive_matfile:str, negtive_matfile:str, downRegulatePath:str, upRegulatePath:str, test_mouse_id):
    precessor = PreprocessBeforeModel()
    precessor.loadRelateAreaPair(downRegulatePath, upRegulatePath)
    # process positive label data
    positive_traindata, positive_testdata = PreprocessMouseCrossValidate.extract_mat(positive_matfile, test_mouse_id)
    positive_samples_train, positive_samples_test = precessor.generateFeature(positive_traindata, True), precessor.generateFeature(positive_testdata, True)
    # process negative label data
    negative_traindata, negative_testdata = PreprocessMouseCrossValidate.extract_mat(negtive_matfile, test_mouse_id)
    negative_samples_train, negative_samples_test = precessor.generateFeature(negative_traindata,False), precessor.generateFeature(negative_testdata, False)
    return positive_samples_train, negative_samples_train, positive_samples_test, negative_samples_test


def dataTransformForSingleModel(positive_matfile:str, negtive_matfile:str, downRegulatePath:str, upRegulatePath:str):
    precessor = PreprocessBeforeModel()
    precessor.loadRelateAreaPair(downRegulatePath, upRegulatePath)
    # process positive label data
    positive_traindata, positive_testdata = PreprocessBeforeModel.extract_mat(positive_matfile)
    positive_samples_train, positive_samples_test = precessor.generateFeature(positive_traindata, True), precessor.generateFeature(positive_testdata, True)
    # process negative label data
    negative_traindata, negative_testdata = PreprocessBeforeModel.extract_mat(negtive_matfile)  
    negative_samples_train, negative_samples_test = precessor.generateFeature(negative_traindata,False), precessor.generateFeature(negative_testdata, False)
    return positive_samples_train, negative_samples_train, positive_samples_test, negative_samples_test


if __name__ =="__main__":
    # CR_samples_train, FA_samples_train, CR_samples_test, FA_samples_test = dataTransformForSingleModel(parameters.CR_matfile, parameters.FA_matfile, parameters.FACR_downPath, parameters.FACR_upPath)
    # for i in [CR_samples_train, FA_samples_train, CR_samples_test, FA_samples_test]:
    #     print(i.shape)
    #     print(np.all(i[:,-1]==1))

    # hit_samples_train, miss_samples_train, hit_samples_test, miss_samples_test = dataTransformForSingleModel(parameters.hit_matfile, parameters.miss_matfile, parameters.hitmiss_downPath, parameters.hitmiss_upPath)
    mouseid_list = PreprocessMouseCrossValidate.getAllMouseId(parameters.hit_matfile)
    for mouseid in mouseid_list:
        hit_samples_train, miss_samples_train, hit_samples_test, miss_samples_test = dataTransformForMouseCV(parameters.hit_matfile, parameters.miss_matfile, parameters.hitmiss_downPath, parameters.hitmiss_upPath, mouseid)
        for i in [hit_samples_train, miss_samples_train, hit_samples_test, miss_samples_test]:
            print(i.shape)
            print(np.all(i[:,-1]==1))
        
