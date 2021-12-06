from torch.utils.data import DataLoader,Dataset
import pickle
from model import ViralGCN
import torch
from torch import nn



class Mydataset(Dataset):
    def __init__(self,path):
        with open(path+'growthset_3h_30_train.pickle','rb') as growthsize:
            self.y=pickle.load(growthsize)
            growthsize.close()
        with open(path+'graphset_3h_30_train.pickle','rb') as graphs:
            self.x=pickle.load(graphs)
            graphs.close()

    def __getitem__(self, index):
        data_x=self.x[index]
        data_y=self.y[index]

        return data_x,data_y
    def __len__(self):
        return len(self.y)

class MSE_n_loss(nn.Module):
    def __init__(self):
        super(MSE_n_loss,self).__init__()
    def forward(self,y_pre,n,y_t):
        # se = torch.square(y_pre-y_t)
        # mse = se.mean()
        le = torch.log(y_pre + 1) - torch.log(y_t + 1)
        sle = torch.pow(le, 2)
        sle = sle.mean()
        loss=sle+0.001*n
        return loss

class SLE_loss(nn.Module):
    def __init__(self):
        super(SLE_loss,self).__init__()
    def forward(self,y_pre,y_t):
        le = torch.log(y_pre + 1) - torch.log(y_t + 1)
        sle = torch.pow(le, 2)
        sle = sle.mean()
        return sle

dataset=Mydataset('weibo_train/')

train_size = int(len(dataset) * 0.8)
validate_size = int(len(dataset))-train_size

train_dataset, validate_dataset, = torch.utils.data.random_split(dataset, [train_size, validate_size])



train_dataloader=DataLoader(dataset=train_dataset,batch_size=None,shuffle=True)
validate_dataloader=DataLoader(dataset=validate_dataset,batch_size=None,shuffle=False)


num_layers=3
input_size=64
hidden_size=32
output_size=48
numinterval=18



loss_fun=MSE_n_loss()
evaluation=SLE_loss()
hyperparameter=list(range(10))
for n_s_n in hyperparameter:

    cas_gcn=ViralGCN(num_layers,input_size,hidden_size,output_size,numinterval,aggr_func='mean',num_sam_nodes=n_s_n,gru=False)
    optimizer=torch.optim.Adam(filter(lambda p: p.requires_grad, cas_gcn.parameters()),lr=0.005)


    for name, parameters in cas_gcn.named_parameters():
        print(name, ':', parameters.size(),':',parameters.requires_grad)
    train_loss=[]
    validation_loss=[]

    for epoch in range(4):
        print('training epoch=',epoch+1,'n:',100+100*n_s_n)
        cas_gcn.train()
        train_x = []
        train_y = []
        train_yl=[]
        drop=len(train_dataset)%20
        for i, data in enumerate(train_dataloader):
            x=data[0]
            y=data[1]
            if i<len(train_dataset)-drop:
                train_x.append(x)
                train_y.append(y)
                train_yl.append(y)
            if (i+1)%20==0:
                y_pre,n_sam=cas_gcn(train_x)
                y_pre=torch.squeeze(y_pre)
                y_t=torch.tensor(train_y)
                loss=loss_fun(y_pre,n_sam,y_t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if i == 19:
                    train_y_pre = y_pre
                else:
                    train_y_pre = torch.cat((train_y_pre, y_pre))
                train_x=[]
                train_y=[]
        train_yl=torch.tensor(train_yl)
        t_loss_n=loss_fun(train_y_pre,n_sam,train_yl)
        t_loss=evaluation(train_y_pre,train_yl)
        train_loss.append(t_loss)
        print('train_SLE:', t_loss,'train_MSE_n:',t_loss_n,'n_sam:',n_sam)
        print(train_loss)

        cas_gcn.eval()
        validate_x = []
        validate_y = []
        drop=len(validate_dataset)%50
        for i, data in enumerate(validate_dataloader):
            x = data[0]
            y = data[1]
            if i<len(validate_dataset)-drop:
                validate_x.append(x)
                validate_y.append(y)
            if (i + 1) % 50 == 0:
                with torch.no_grad():
                    y_pre,n_sam=cas_gcn(validate_x)
                    y_pre = torch.squeeze(y_pre)
                if i == 49:
                    validate_y_pre = y_pre
                else:
                    validate_y_pre = torch.cat((validate_y_pre, y_pre))
                validate_x = []
        validate_y_t = torch.tensor(validate_y)
        v_loss = evaluation(validate_y_pre, validate_y_t)
        v_loss_n=loss_fun(validate_y_pre,n_sam,validate_y_t)
        validation_loss.append(v_loss)
        print('validation_SLE:', v_loss,'validation_MSE_n:',v_loss_n,'n_sam:',n_sam)
        print(validation_loss)
        # if len(validation_loss) > 5:
        #     l1 = validation_loss[-2] - validation_loss[-1]
        #     l2 = validation_loss[-3] - validation_loss[-2]
        #     if l1 < 0.01 and l2 < 0.01:
        #         print('loss not decline for 2 epoches,break at epoch', epoch + 1)
        #         break
        # if (epoch+1)%5==0:
        #     query = input('save model or not:')
        #     if query == 'yes':
        #         with open('3h_30_validation_loss.pickle','wb') as f:
        #             pickle.dump(validation_loss,f)
        #             f.close()
        #
        #         torch.save(cas_gcn.state_dict(), 'viralgcn_3h_30(train_n).pt')




