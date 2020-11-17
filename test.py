from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from data_loader import transformer_loader
import math
import time
import copy
import random
import os
from scipy.io import loadmat



class Trans(nn.Module):
    def __init__(self):
        super(Trans, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=100, nhead=10,dropout=0.3,dim_feedforward=1024)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6,norm="T")
        self.linear = nn.Sequential(nn.Flatten(),
                                    nn.Linear(100*20, 512),
                                    nn.Dropout(0.1),
                                    # nn.Linear(1024, 512),
                                    # nn.Dropout(0.1),
                                    nn.Linear(512, 2),
                                    nn.Dropout(0.1),
                                    nn.Softmax())

    def forward(self,x):
        x=self.transformer_encoder(x)
        x=self.linear(x)
        return x

def get_data(path_mat_data):
    data = loadmat(path_mat_data, mat_dtype=True)
    train_set = data["test_set"]
    list_data=[]
    num_test_data=train_set[0, 0][0].shape[0]
    print(num_test_data)
    for i in range(20):
        di = train_set[i, 0][0]
        list_data.append(di)
    all_data_list=[]
    for k in range(num_test_data):
        new_data = np.zeros((20, 100))
        for j in range(20):
            data = list_data[j]
            one_ = data[k, :]

            if (math.isnan(np.sum(one_))):
                one_[one_ != 0] = 0
            new_data[j, :] = one_
        min_data=np.min(new_data)
        max_data=np.max(new_data)
        new_data=(new_data-min_data)/(max_data-min_data)
        # print(torch.from_numpy(new_data).size())
        all_data_list.append(torch.from_numpy(new_data))
    list_datas=[]
    for i,item in enumerate(all_data_list):
        input_data=torch.zeros(1,20,100)
        input_data[0,:,:]=item
        list_datas.append(input_data)
    return list_datas
def test(model_path):
    path="MCAD_AFQ_test.mat"

    data=get_data(path)
    model=Trans().cuda()
    ps=[]
    class_list=[]
    for i,input_data in enumerate(data):
        class1_p,class2_p=0,0
        nums_class1,nums_class2=0,0
        input_=Variable(input_data).float().cuda()
        for file in os.listdir(model_path):
            model.load_state_dict(torch.load(model_path+file))
            pred = model(input_)
            score_p, prediction = torch.max(pred, 1)
            class1_p+=score_p.cpu().detach().numpy()[0]
            prediction = prediction.cpu().detach().numpy()[0]

            score_p_min, _ = torch.min(pred, 1)
            class2_p += score_p_min.cpu().detach().numpy()[0]
            if(prediction==0):
                nums_class1+=1
            else:
                nums_class2+=1
        if(nums_class1>nums_class2):
            pred_class=1
        else:
            pred_class=3
        ps.append([class1_p/3,class2_p/3])
        class_list.append(pred_class)
    return class_list,ps
if __name__=="__main__":
    import pandas as pd
    pw=pd.ExcelWriter("2class.xlsx")
    model_path="./model_12/"
    class_list,ps=test(model_path)
    lists=[]
    for i in range(len(class_list)):
        if(class_list[i]==3):
            index=1
        else:
            index=0
        if(ps[i][index]<0.5):
            p=1-ps[i][index]
        else:
            p=ps[i][index]
        lists.append([class_list[i],p])
    df2=pd.DataFrame(lists,columns=["class","p"])
    df2.to_excel(pw,"sheet")
    pw.save()



