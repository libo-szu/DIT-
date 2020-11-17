import os
import numpy as np
import random
import  sklearn.svm as svm
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from scipy.io import loadmat
from sklearn.decomposition import PCA

def get_data(train_list):
    data = loadmat("MCAD_AFQ_competition.mat", mat_dtype=True)
    train_set = data["train_set"]
    train_diagnose = data["train_diagnose"]
    train_population = data["train_population"]
    train_sites = data['train_sites']
    train_old = train_population[:, 1]
    train_sex = train_population[:, 0]
    max_old=np.max(train_old)
    min_old=np.min(train_old)
    train_data = []
    train_label = []
    for file in train_list:
        index_file=int(file.replace(".npy",""))
        data = np.load("./new_data/" + file)
        label = np.load("./new_label/" + file).tolist()
        max_index = label.index(max(label))
        # if (max_index > 0):
        #     max_index = 1
        old_info=train_old[index_file]
        old_info=(old_info-min_old)/(max_old-min_old)
        sex_info=train_sex[index_file]
        # train_data_add=data.flatten()
        # pca = PCA(n_components=100)
        # # print(data.T.shape)
        # newX = pca.fit_transform(data)
        # invX = pca.inverse_transform(newX)
        # new_data=pca.transform(data)
        # print(new_data.shape)
        # print(pca.explained_variance_ratio_)
        train_data_add=data.flatten()
        train_data_add=np.append(train_data_add,sex_info)
        train_data_add=np.append(train_data_add,old_info)

        train_data.append(train_data_add)
        train_label.append(max_index)
    return train_data, train_label

def train():
    all_data_list = os.listdir("./new_data/")
    val1, val2, val3, val4, val5 = all_data_list[:int(700 * 0.2)],\
                                   all_data_list[ int(700 * 0.2):int(700 * 0.4)], \
                                   all_data_list[int(700 * 0.4):int(700 * 0.6)], \
                                   all_data_list[int(700 * 0.6):int(700 * 0.8)],\
                                   all_data_list[int(700 * 0.8):]

    list_vals = [val1, val2, val3, val4, val5]
    for item in list_vals:
        val_list = item
        train_list = [item for item in all_data_list if item not in val_list]
        train_data, train_label = get_data(train_list)
        val_data, val_label = get_data(val_list)
        clf = RandomForestClassifier()
        # print(len(train_data))
        train_data=np.array(train_data)
        train_label=np.array(train_label)
        # clf=AdaBoostClassifier()
        clf.fit(train_data, train_label)
        count = 0
        cc = 0
        for index, item in enumerate(val_data):
            cc += 1
            pred = clf.predict([item])
            # print(pred)
            label = val_label[index]
            if (pred[0] == label):
                count += 1
        print(count / cc)
if __name__=="__main__":
    train()
