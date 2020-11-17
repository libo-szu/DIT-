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
        # print(x.size())
        x=self.transformer_encoder(x)
        # print(x.size())
        x=self.linear(x)
        return x

def binary_acc(preds, y):

    # preds = torch.round(preds)
    score_p, prediction = torch.max(preds, 1)
    score_t, target = torch.max(y, 1)
    # print("hhhhhhhhhhhhhhh")
    # print(prediction)
    # print(target)

    correct = torch.eq(prediction, target).float()
    acc = correct.sum() /( len(correct))
    return acc


#训练函数
def train(model, iterator, optimizer, criteon):

    avg_loss = []
    avg_acc = []
    model.train()        #表示进入训练模式
    # print("-------------------")
    # print(len(iterator))
    for i, (data,label,mask) in enumerate(iterator):
        # print("*****************")
        # print(data.size())
        # print(label.size())
        # print(mask.size())
        data=Variable(data).float().cuda()
        mask=Variable(mask).float().cuda()
        label=Variable(label).float().cuda()
        # print("*****************")
        # print(data.size())
        # print(label.size())
        # print(mask.size())
        # data = Variable(data).type(torch.LongTensor)
        pred = model(data)

        # print(pred.size())
        # print("hhhhhhhhhhhh")
        # similarity = torch.cosine_similarity(pred, label)
        #
        # loss = 1 - similarity
        loss = criteon(pred, label)
        acc = binary_acc(pred, label).item()   #计算每个batch的准确率
        avg_loss.append(loss.item())
        avg_acc.append(acc)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_acc = np.array(avg_acc).mean()
    avg_loss = np.array(avg_loss).mean()
    return avg_loss, avg_acc


#评估函数
def evaluate(model, iterator, criteon):

    avg_loss = []
    avg_acc = []
    model.eval()         #表示进入测试模式

    with torch.no_grad():
        for (data,label,mask) in iterator:

            data = Variable(data).float().cuda()
            mask = Variable(mask).float().cuda()
            label = Variable(label).float().cuda()
            pred = model(data)
            # score_p, prediction = torch.max(pred, 1)
            # score_t, target = torch.max(label, 1)
            # print("hhhhhhhhhhhhhhh")
            # print(prediction)
            # print(target)
            loss = criteon(pred,label)
            acc = binary_acc(pred, label).item()
            avg_loss.append(loss.item())
            avg_acc.append(acc)

    avg_loss = np.array(avg_loss).mean()
    avg_acc = np.array(avg_acc).mean()
    return avg_loss, avg_acc
if __name__=="__main__":
    all_data_list = os.listdir("./new_data/")
    # random.shuffle(all_data_list)
    val_list=random.sample(range(100),20)+random.sample(range(100,200),20)+random.sample(range(200,300),20)+random.sample(range(300,400),20)+random.sample(range(400,500),20)+random.sample(range(500,600),20)+random.sample(range(600,700),20)
    val_list=[str(item)+".npy" for  item in val_list]
    # val1,val2,val3,val4,val5=all_data_list[:int(700*0.2)],all_data_list[int(700*0.2):int(700*0.4)],all_data_list[int(700*0.4):int(700*0.6)],all_data_list[int(700*0.6):int(700*0.8)],all_data_list[int(700*0.8):]
    # list_vals=[val1,val2,val3,val4,val5]
    # for item in list_vals:
    #     val_list=item
    from scipy.io import loadmat
    data = loadmat("MCAD_AFQ_competition.mat", mat_dtype=True)
    print(data.keys())
    # print(data["train_set"])
    # print(data["train_set"]0][0][.shape)
    train_set = data["train_set"]
    train_diagnose = data["train_diagnose"]
    # all_data_list = []
    # for i, item in enumerate(train_diagnose):
    #     if (item[0] == 1 or item[0]==3):
    #         all_data_list.append(str(i) + ".npy")
    sorted(all_data_list)
    # val_list=random.sample(all_data_list,int(len(all_data_list)*0.2))
    print(all_data_list)
    print(val_list)
    print(len(val_list))
    # val_list = random.sample(all_data_list, int(700 * 0.2))
    train_list = [item for item in all_data_list if item not in val_list]
    # random.shuffle(train_list)
    train_dataset = transformer_loader(train_list,
                                               transform=transforms.Compose([
                                                   transforms.ToTensor(),

                                               ]))
    train_iterator = DataLoader(train_dataset, batch_size=32,
                                   shuffle=True, num_workers=1)
    print(len(train_list))
    print(len(train_iterator))
    dev_dataset = transformer_loader(val_list,
                                               transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                               ]))

    dev_iterator = DataLoader(dev_dataset, batch_size=32,
                                   shuffle=False, num_workers=1)

    # model=resnet18()
    # model =MyTransformerModel(p_drop=0.12).cuda()
    model=Trans().cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criteon = nn.BCELoss()
    accs=[]
    losses_list=[]

    best_valid_acc = float('-inf')
    for epoch in range(200):
        start_time = time.time()
        train_loss, train_acc = train(model, train_iterator, optimizer, criteon)
        dev_loss, dev_acc = evaluate(model, dev_iterator, criteon)
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        if dev_acc > best_valid_acc:          #只要模型效果变好，就保存
            best_valid_acc = dev_acc
            torch.save(model.state_dict(),'wordavg-model.pt')
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs:.2f}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {dev_loss:.3f} |  Val. Acc: {dev_acc*100:.2f}%')
        accs.append(dev_acc)
        losses_list.append(dev_loss)
    import matplotlib.pyplot as plt
    plt.plot(range(200),losses_list)
    plt.show()
    plt.plot(range(200),accs)
    plt.show()
