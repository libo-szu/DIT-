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
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=100, nhead=2,dropout=0.3,dim_feedforward=1024)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6,norm="T")
        self.linear = nn.Sequential(nn.Flatten(),
                                    nn.Linear(100*20, 512),
                                    nn.Dropout(0.1),
                                    # nn.Linear(1024, 512),
                                    # nn.Dropout(0.1),
                                    nn.Linear(512, 3),
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
        input_=Variable(input_data).float().cuda()
        p1,p2,p3=0,0,0

        model.load_state_dict(torch.load(model_path))
        pred = model(input_)
        score_p, prediction = torch.max(pred, 1)
        pred=pred.cpu().detach().numpy()[0]

        # print(pred.shape)
        class_list.append([prediction.cpu().detach().numpy()[0]+1,pred[0],pred[1],pred[2]])


    return class_list
if __name__=="__main__":
    result=test("./model123/1.pt")
    # print(result)

    import pandas as pd
    pw=pd.ExcelWriter("3class.xlsx")

    df2=pd.DataFrame(result,columns=["class","p","h","j"])
    df2.to_excel(pw,"sheet")
    pw.save()



