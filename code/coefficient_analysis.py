import os
import scipy.io as io
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import preprocessing
import pandas as pd
import parameters

def coef(meantrialmat, start,windowsize):
    submat = meantrialmat[:, start:start+windowsize]
    coefmat = np.corrcoef(submat)
    return coefmat
def extractupstri(coefmat):
    row, col = coefmat.shape
    upstri = []
    for i in range(0, row-1):
        for j in range(i+1, col):
            upstri.append(coefmat[i,j])

    return np.array(upstri)

# 每个trial做一次时间相关性图，然后再每个trial中取平均
def batch2(datapath, targetindex):
    filelist = os.listdir(datapath)
    allcoefbytime = []
    n = 1
    for file in filelist[0:8]:
        curfile = os.path.join(datapath, file)
        mat = io.loadmat(curfile)
        mat = mat["All_Data"]
        submat = mat[targetindex, :, :].squeeze()
        windowsize = 5
        channel, t_point, trial = submat.shape
        coefbytimepertrial = [] # 每个trial中的 时间相关二维图
        for tr_idx in range(trial):
            meantrial = preprocessing.scale(submat[:,:,tr_idx])  # 标准化
            coefbytime = []
            for st in range(t_point-windowsize+1):
                coefmat = coef(meantrial, st, 5)
                coefbytime.append(extractupstri(coefmat))
            coefbytime = np.array(coefbytime).T
            coefbytimepertrial.append(coefbytime)
        # print(np.array(coefbytimepertrial).shape)
        meancoef = np.mean(np.array(coefbytimepertrial), axis=0)
        # print(meancoef.shape)
        if n <= 1:
            allcoefbytime = np.copy(meancoef)
        else:
            allcoefbytime = np.concatenate((allcoefbytime, meancoef), axis=1)
        print(allcoefbytime.shape)
        n += 1
    return allcoefbytime

def batch3(datapath, targetindex):
    filelist = os.listdir(datapath)
    allcoefbytime = []
    n = 1
    for file in filelist[0:8]:
        curfile = os.path.join(datapath, file)
        mat = io.loadmat(curfile)
        mat = mat["All_Data"]
        submat = mat[targetindex, :, :].squeeze()
        windowsize = 5
        channel, t_point, trial = submat.shape
        coefbytimepertrial = [] # 每个trial中的 时间相关二维图
        for tr_idx in range(trial):
            meantrial = preprocessing.scale(submat[:,:,tr_idx])  # 标准化
            coefbytime = []
            for st in range(t_point-windowsize+1):
                coefmat = coef(meantrial, st, 5)
                coefbytime.append(extractupstri(coefmat))
            coefbytime = np.array(coefbytime).T
            coefbytimepertrial.append(coefbytime)
        # print(np.array(coefbytimepertrial).shape)
        meancoef = np.mean(np.array(coefbytimepertrial), axis=0)
        lasttwocol = np.mean(meancoef[:, -2:-1], axis=1).reshape((-1, 1))  # coef表中的最后两列取平均，然后hit和miss做差，cr和fa做差
        # print(meancoef.shape)
        if n <= 1:
            allcoefbytime = np.copy(lasttwocol)
        else:
            allcoefbytime = np.concatenate((allcoefbytime, lasttwocol), axis=1)
        print(allcoefbytime.shape)
        n += 1
    return allcoefbytime  # 返回在最后两列做平均后的表
    # plt.figure()
    # plt.imshow(allcoefbytime.repeat([100], axis=1), cmap=plt.cm.hot_r)
    # plt.colorbar()

def compareconsitency(arr1, arr2, cutoff):
    """
    比较两种实验条件下（control 和 CNO）的coef的变化是否具有一致性
    """
    bothup = []
    bothdown = []
    for i, ele in enumerate(arr1):
        if arr1[i] > cutoff and arr2[i] > cutoff:
            bothup.append(i)
        if arr1[i] < -cutoff and arr2[i] < -cutoff:
            bothdown.append(i)
    return bothup, bothdown

def indxtoarea(indxarr, areaidx, targetindex, samplearea):
    rowname = []
    colname = []
    rowidxlist = []
    colidxlist = []
    for i in indxarr:  # 第几行的脑区对
        rowidx, colidx = areaidx[i]  # targetarea中的索引
        rowidxlist.append(rowidx)
        colidxlist.append(colidx)
        # srowidx, scolidx = targetindex[rowidx], targetindex[colidx]
        rowname.append(samplearea[rowidx])
        colname.append(samplearea[colidx])
    df = pd.DataFrame({"area1_index":rowidxlist,
                        "area2_index":colidxlist,
                        "area1": rowname,
                        "area2": colname})
    return df

def getsortindx(mat, col, sorted_mat=False):
    """
    得到排序的索引, 是否需要排序后的mat,返回索引或排序后的矩阵
    :param mat: 原矩阵
    :param col: 按照哪一列进行排序
    """
    sortindx = np.mean(mat[:, col], axis=1).argsort()
    sortmat = mat[sortindx]
    if sorted_mat:
        return sortmat
    else:
        return sortindx

def gettargetindex(matpath=None, samplearea=None, targetarea=None): # 是文件
    if matpath:
        aadict = io.loadmat(matpath)
        arr = aadict["index"].reshape((-1,))
        return arr
    else:
        targetindex = []
        for i,sample in enumerate(samplearea):
            for target in targetarea:
                # if sample.find(target) != -1:
                #     targetindex.append(i)
                similarity = string_similar(sample, target)
                if similarity > 0.95:
                    targetindex.append(i)
        targetindex = np.array(list(set(targetindex))).reshape((-1, ))
        return targetindex


def getareaidx(indxmat):
    """
    产生一个索引字典
    """
    row, col = indxmat.shape
    areaidx = {}
    n = 0
    for i in range(0, row-1):
        for j in range(i+1, col):
            areaidx[n] = (i, j)
            n += 1
    return areaidx


import difflib
def string_similar(s1, s2):
    """
    比较两个字符串的相似性
    """
    return difflib.SequenceMatcher(None, s1, s2).quick_ratio()

import re
def clean_en_text(text):
    # keep English, digital and space
    comp = re.compile('[^A-Z^a-z^0-9^ ]')
    return comp.sub('', text)


if __name__ == "__main__":
    rootpath = parameters.rootpath
    with open(os.path.join(rootpath, "samplearea.txt")) as f:
        arealist = f.readlines()
        samplearea = [clean_en_text(a) for a in arealist]
    aa = io.loadmat(os.path.join(rootpath, "mainAreaIndex\map.mat"))["finalmap"].reshape((-1,))
    targetarea = [clean_en_text(i[0]) for i in aa]
    targetindexpath = os.path.join(rootpath, "mainAreaIndex\index.mat")
    con_path = os.path.join(rootpath, "data2")
    exp_path = os.path.join(rootpath, "data")
    # save_path = r"C:\Users\chenx\Documents\fMRI data analysis\data_allarea"
    save_path = os.path.join(rootpath, "data_allarea_cutoff140")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    targetindex = gettargetindex(samplearea=samplearea, targetarea=samplearea)
    targetindex = targetindex[:140] # 对于前140个脑区做相关性分析， 后面的脑区基本没有信号，因此删除掉。
    # targetindex = gettargetindex(matpath=targetindexpath)
    print(len(targetindex))
    indxmat = np.ones((len(targetindex), len(targetindex)))
    areaidx = getareaidx(indxmat)
    try:
        con_coef2 = np.load(os.path.join(save_path, "con_coef2.npy"))
        exp_coef2 = np.load(os.path.join(save_path, "exp_coef2.npy"))
        con_lasttwomean = np.load(os.path.join(save_path, "con_lasttwomean.npy"))
        exp_lasttwomean = np.load(os.path.join(save_path, "exp_lasttwomean.npy"))
        print("Have loaded coefmat")
    except:
        print("loading coefmat...")
    if not (("con_coef2" in vars()) or ("exp_coef2" in vars())):
        con_coef2 = batch2(con_path, targetindex)
        exp_coef2 = batch2(exp_path, targetindex)
        np.save(os.path.join(save_path, "con_coef2.npy"), con_coef2)
        np.save(os.path.join(save_path, "exp_coef2.npy"), exp_coef2)
        # control组和对照组中，最后两个时间窗口的矩阵图
        con_lasttwomean = batch3(con_path, targetindex)
        exp_lasttwomean = batch3(exp_path, targetindex)
        np.save(os.path.join(save_path, "con_lasttwomean.npy"), con_lasttwomean)
        np.save(os.path.join(save_path, "exp_lasttwomean.npy"), exp_lasttwomean)
    else:
        print("have loaded data")
    exp_con_coef2 = np.concatenate((exp_coef2, con_coef2), axis=1)
    # 计算ΔR
    mrow, mcol = con_lasttwomean.shape
    con_delta = []
    exp_dalta = []
    for c in np.arange(1, mcol, 2):
        con_delta.append(con_lasttwomean[:,c] - con_lasttwomean[:, c-1])
        exp_dalta.append(exp_lasttwomean[:,c] - exp_lasttwomean[:, c-1])
    con_delta = np.array(con_delta).T
    exp_delta = np.array(exp_dalta).T
    print(con_delta.shape)
    # control组中协同变化的脑区
    for cutoff in [0.03, 0.08]:  # idx 对应着第几行的脑区pair
        FA_CR_both_up_idx, FA_CR_both_down_idx = compareconsitency(con_delta[:,0], con_delta[:,2], cutoff)
        HIT_MISS_both_up_idx, HIT_MISS_both_down_idx = compareconsitency(con_delta[:,1], con_delta[:,3], cutoff)
        # 实验组中协同变化的脑区
        # exp_FA_CR_both_up_idx, exp_FA_CR_both_down_idx = compareconsitency(exp_delta[:,0], exp_delta[:,2], cutoff)
        # exp_HIT_MISS_both_up_idx, exp_HIT_MISS_both_down_idx = compareconsitency(exp_delta[:,1], exp_delta[:,3], cutoff)
        # 取交集（control组和实验组中变化趋势相同的脑区）
        # FA_CRBothUp = list(set(FA_CR_both_up_idx).intersection(set(exp_FA_CR_both_up_idx)))
        # FA_CRBothDown = list(set(FA_CR_both_down_idx).intersection(set(exp_FA_CR_both_down_idx)))
        # HIT_MISSBothUp = list(set(HIT_MISS_both_up_idx).intersection(set(exp_HIT_MISS_both_up_idx)))
        # HIT_MISSBothDown = list(set(HIT_MISS_both_down_idx).intersection(set(exp_HIT_MISS_both_down_idx)))
        foralt = [FA_CR_both_up_idx, FA_CR_both_down_idx, HIT_MISS_both_up_idx, HIT_MISS_both_down_idx]
        name = ["FA_CRBothUp", "FA_CRBothDown", "HIT_MISSBothUp", "HIT_MISSBothDown"]

        writer3 = pd.ExcelWriter(os.path.join(save_path, "pairareasort{}.xlsx".format(cutoff)))
        for na, ele in zip(name, foralt):
            ele = np.array(ele)
            ma = len(ele)
            row, col = exp_con_coef2.shape
            if ele.size != 0:
                sortidx = getsortindx(exp_con_coef2[ele], [-7, -8])
                pairarea = indxtoarea(ele[sortidx] ,areaidx, targetindex, samplearea)  # 实际上的第几个脑区对（与matlab上对应）
                pairarea.to_excel(writer3, sheet_name=na, index=True)
                np.save(os.path.join(save_path, "{}_joinedarea{}.npy".format(na, cutoff)), np.array(pairarea[["area1_index", "area2_index"]]))
                plt.figure()
                if ma > col:
                    copy = ma//col + 1
                    plt.imshow(exp_con_coef2[ele][sortidx].repeat([copy], axis=1), cmap=plt.cm.hot_r)
                    plt.vlines(np.arange(0,col,6)*copy-0.5, 0, ma, colors="c",linestyles="dashed",linewidth=0.8)
                else:
                    copy = 1
                    plt.imshow(exp_con_coef2[ele][sortidx].repeat([copy], axis=1), cmap=plt.cm.hot_r)
                    plt.vlines(np.arange(0,col,6)*copy-0.5, 0, ma, colors="c",linestyles="dashed",linewidth=0.8)
                plt.xticks([])
                plt.title(na)
                plt.colorbar()
                plt.savefig(os.path.join(save_path, "{}{}.png".format(na, cutoff)), dpi=600)
            print("ones list done")
        writer3.save()
    plt.show()