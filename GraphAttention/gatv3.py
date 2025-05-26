import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from torch_geometric.data import Dataset
import os
import os.path as osp
from torch.nn import Linear, LeakyReLU, Softmax, BatchNorm1d, AvgPool2d, Sigmoid
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool, GATConv, GATv2Conv, TransformerConv, MFConv, SAGEConv, EdgeConv, EdgePooling
import torch_geometric.transforms as T
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import math
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("--drop", type=float, default=0)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--bs', type=int, default=50)
#parser.add_argument('--chn',type=int)
args = parser.parse_args()
drop=float(args.drop)
num_epochs=int(args.epochs)
lr=float(args.lr)
bs=int(args.bs)
#chn=int(args.chn)
act1=LeakyReLU(0.1)
act2=Sigmoid()
print('===== TESTING HYPERPARAMETERS =====')
print('LAYERS: 7')
print('DROPOUT: '+str(drop))
print('EPOCHS: '+str(num_epochs))
print('LEARNING RATE: '+str(lr))
print('BATCH SIZE: '+str(bs))

#print('CHANNELS: '+str(chn))
# os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
class RBDDataset_train(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
    
    @property
    def raw_file_names(self):
        return ['file.pt']

    @property
    def processed_file_names(self):
        return ['file.pt']
    
    def process(self):
        idx = 0
        #for raw_path in self.raw_paths:
        for files in os.listdir('train_r1_test_r2/train/raw/'):
            # Read data from `raw_path`.
            # if osp.isfile(osp.join(self.processed_dir, f'data_{idx}.pt'))==False:
            #     data = torch.load('train_r1_r2/train/raw/'+files)
            #     if self.pre_filter is not None and not self.pre_filter(data):
            #         continue
            #     if self.pre_transform is not None:
            #         data = self.pre_transform(data)
            #     torch.save([data,files], osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):
        ctr=0
        for files in os.listdir('train_r1_test_r2/train/raw/'):
            ctr+=1
        return ctr

    def get(self, idx):
        # for files in os.listdir('sample_graphs/processed/'):
        #   cur_ind = int(files.split('_')[-1].replace('.pt',''))
        #   if cur_ind == int(idx):
        #       data = torch.load(osp.join(self.processed_dir, files))
        #       break
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data
class RBDDataset_val(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
    
    @property
    def raw_file_names(self):
        return ['file.pt']

    @property
    def processed_file_names(self):
        return ['file.pt']
    
    def process(self):
        idx = 0
        #for raw_path in self.raw_paths:
        for files in os.listdir('train_r1_test_r2/val/raw/'):
            # Read data from `raw_path`.
            # if osp.isfile(osp.join(self.processed_dir, f'data_{idx}.pt'))==False:
            #     data = torch.load('train_r1_r2/val/raw/'+files)
            #     if self.pre_filter is not None and not self.pre_filter(data):
            #         continue
            #     if self.pre_transform is not None:
            #         data = self.pre_transform(data)
            #     torch.save([data,files], osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):
        ctr=0
        for files in os.listdir('train_r1_test_r2/val/raw/'):
            ctr+=1
        return ctr

    def get(self, idx):
        # for files in os.listdir('sample_graphs/processed/'):
        #   cur_ind = int(files.split('_')[-1].replace('.pt',''))
        #   if cur_ind == int(idx):
        #       data = torch.load(osp.join(self.processed_dir, files))
        #       break
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data
class RBDDataset_test(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
    
    @property
    def raw_file_names(self):
        return ['file.pt']

    @property
    def processed_file_names(self):
        return ['file.pt']
    
    def process(self):
        idx = 0
        #for raw_path in self.raw_paths:
        for files in os.listdir('train_r1_test_r2/test/raw/'):
            # Read data from `raw_path`.
            # if osp.isfile(osp.join(self.processed_dir, f'data_{idx}.pt'))==False:
            #     data = torch.load('train_r1_test_r2/test/raw/'+files)
            #     if self.pre_filter is not None and not self.pre_filter(data):
            #         continue
            #     if self.pre_transform is not None:
            #         data = self.pre_transform(data)
            #     torch.save([data,files], osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):
        ctr=0
        for files in os.listdir('train_r1_test_r2/test/raw/'):
            ctr+=1
        return ctr

    def get(self, idx):
        # for files in os.listdir('sample_graphs/processed/'):
        #   cur_ind = int(files.split('_')[-1].replace('.pt',''))
        #   if cur_ind == int(idx):
        #       data = torch.load(osp.join(self.processed_dir, files))
        #       break
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GATv2Conv(2, 20, edge_dim=2, heads=2)
        self.conv2 = GATv2Conv(40, 40, edge_dim=2, heads=2)
        self.conv3 = GATv2Conv(80, 80, edge_dim=2, heads=2)
        # self.conv4 = GATv2Conv(160, 160, edge_dim=3, heads=2)
        # self.conv5 = GATv2Conv(320, 320, edge_dim=3, heads=2)
        # self.conv6 = GATv2Conv(640, 640, edge_dim=3, heads=2)
        # self.conv7 = GATv2Conv(1280, 1280, edge_dim=3, heads=2)
        
        # self.conv31 = GCNConv(22, 40)
        # self.conv32 = GCNConv(40, 80)
        # self.conv33 = GCNConv(80, 160)
        # self.conv34 = GCNConv(160, 320)
        # self.conv35 = GCNConv(320, 640)
        # self.conv36 = GCNConv(640, 1280)
        # self.conv37 = GCNConv(1280, 2560)
        
        # self.conv41 = SAGEConv(22, 40, project = True)
        # self.conv42 = SAGEConv(40, 80, project = True)
        # self.conv43 = SAGEConv(80, 160, project = True)
        # self.conv44 = SAGEConv(160, 320, project = True)
        # self.conv45 = SAGEConv(320, 640, project = True)
        # self.conv46 = SAGEConv(640, 1280, project = True)
        # self.conv47 = SAGEConv(1280, 2560, project = True)
        
        self.bn1 = BatchNorm1d(40)
        self.bn2 = BatchNorm1d(80)
        self.bn3 = BatchNorm1d(160)
        # self.bn4 = BatchNorm1d(320)
        # self.bn5 = BatchNorm1d(640)
        # self.bn6 = BatchNorm1d(1280)
        # self.bn7 = BatchNorm1d(2560)

        # self.bnc1 = BatchNorm1d(1280)
        # self.bnc2 = BatchNorm1d(640)
        
        # self.lin1 = Linear(2560, 1280)
        # self.lin2 = Linear(1280, 640)
        # self.lin3 = Linear(640, 2)

        # self.bnc1 = BatchNorm1d(80)
        # self.bnc2 = BatchNorm1d(40)

        self.lin1 = Linear(160, 2)
        # self.lin2 = Linear(80, 40)
        # self.lin3 = Linear(40, 3)

    def forward(self, x, edge_index, edge_attr, batch):
        x11 = self.conv1(x, edge_index, edge_attr)
        # x12 = self.conv31(x, edge_index)
        # x13 = self.conv41(x, edge_index)
        # x1 = self.bn1(x11 + x12 + x13)
        x1 = self.bn1(x11)
        x1 = act1(x1)
        x1 = F.dropout(x1, p=drop, training=self.training)
        
        x21 = self.conv2(x1, edge_index, edge_attr)
        # x22 = self.conv32(x1, edge_index)
        # x23 = self.conv42(x1, edge_index)
        # x2 = self.bn2(x21 + x22 + x23)
        x2 = self.bn2(x21)
        x2 = act1(x2)
        x2 = F.dropout(x2, p=drop, training=self.training)

        x31 = self.conv3(x2, edge_index, edge_attr)
        # x32 = self.conv33(x2, edge_index)
        # x33 = self.conv43(x2, edge_index)
        # x3 = self.bn3(x31 + x32 + x33)
        x3 = self.bn3(x31)
        x3 = act1(x3)
        x3 = F.dropout(x3, p=drop, training=self.training)

        # x41 = self.conv4(x3, edge_index, edge_attr)
        # x42 = self.conv34(x3, edge_index)
        # x43 = self.conv44(x3, edge_index)
        # x4 = self.bn4(x41 + x42 + x43)
        # # x4 = self.bn4(x43)
        # x4 = act1(x4)
        # x4 = F.dropout(x4, p=drop, training=self.training)

        # x51 = self.conv5(x4, edge_index, edge_attr)
        # x52 = self.conv35(x4, edge_index)
        # x53 = self.conv45(x4, edge_index)
        # x5 = self.bn5(x51 + x52 + x53)
        # # x5 = self.bn5(x53)
        # x5 = act1(x5)
        # x5 = F.dropout(x5, p=drop, training=self.training)

        # x61 = self.conv6(x5, edge_index, edge_attr)
        # x62 = self.conv36(x5, edge_index)
        # x63 = self.conv46(x5, edge_index)
        # x6 = self.bn6(x61 + x62 + x63)
        # # x6 = self.bn6(x63)
        # x6 = act1(x6)
        # x6 = F.dropout(x6, p=drop, training=self.training)

        # x71 = self.conv7(x6, edge_index, edge_attr)
        # x72 = self.conv37(x6, edge_index)
        # x73 = self.conv47(x6, edge_index)
        # x7 = self.bn7(x71 + x72 + x73)
        # # x7 = self.bn7(x73)
        # x7 = act1(x7)
        # x7 = F.dropout(x7, p=drop, training=self.training)
        
        xfinal = global_mean_pool(x3, batch)
        #xfinal = act1(xfinal)
        #xfinal = F.dropout(xfinal, p=drop, training=self.training)
        xclass = self.lin1(xfinal)
        #xclass = self.bnc1(xclass)
        #xclass = act1(xclass)
        #xclass = F.dropout(xclass, p=drop, training=self.training)
        #xclass2 = self.lin2(xclass)
        #xclass2 = self.bnc2(xclass2)
        #xclass2 = act1(xclass2)
        #xclass2 = F.dropout(xclass2, p=0.5, training=self.training)
        #xclass3 = self.lin3(xclass2)
        
        return xclass

def train():
    # ctr=0
    running_loss=0.0
    for data in train_loader:  # Iterate in batches over the training dataset.
        g = data[0]
        g=g.to(device=device)
        out = model(g.x, g.edge_index, g.edge_attr, g.batch)  # Perform a single forward pass.
        loss = criterion(out, g.y)  # Compute the loss
        #print(tot)
        print(loss)
        running_loss+=loss.item() * len(data)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients
        # ctr+=1
        # print(ctr*bs)
    return running_loss

def test(loader):
    preds=[]
    true=[]
    names=[]
    for data in loader:  # Iterate in batches over the training/test dataset.
        g = data[0]
        name = data[1]
        g=g.to(device=device)
        out = model(g.x, g.edge_index, g.edge_attr, g.batch)  
        out = Softmax(dim=1)(out)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        preds = preds + list(pred.cpu())
        names = names + list(name)
        true = true + list(g.y.cpu())
    true_cpu=[]
    preds_cpu=[]
    for a, b in zip(true, preds):
        true_cpu.append(int(a))
        preds_cpu.append(int(b.cpu()))
    return balanced_accuracy_score(true, preds), accuracy_score(true, preds)

def split_dataset(dataset):
    indices = list(range(dataset.len()))
    classes = dataset.classes()
    print(classes)
    train_indices, val_indices = train_test_split(indices, test_size=0.1, stratify=classes)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    return train_dataset, val_dataset
#def main():
print(torch.cuda.is_available())
set_train=RBDDataset_train(root='train_r1_test_r2/train/')
#set_train.process()
set_val=RBDDataset_val(root='train_r1_test_r2/val/')
#set_val.process()
set_test=RBDDataset_test(root='train_r1_test_r2/test/')
#set_test.process()
# set_val=PDZDatasetVal(root='profaff_graphs_test/')
# set_test=PDZDatasetTest(root='../negin/graphs/')
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
train_loader=DataLoader(set_train, batch_size=bs, shuffle=True)
val_loader=DataLoader(set_val, batch_size=bs, shuffle=False)
test_loader=DataLoader(set_test, batch_size=bs, shuffle=False)

base_drop = 'model_'+str(drop)+'_'
base_lr = '_lr_'+str(lr)+'_bs_'+str(bs)
l = []
for files in os.listdir('binary/saved_models_split/'):
	if base_drop in files and base_lr in files:
		l.append([files, int(files.split('_')[5])])
print(l)
if len(l) > 0:
	l.sort(key=lambda x:x[1], reverse=True)
	recent_model = l[0][0]
	model = torch.load('binary/saved_models_split/'+recent_model)
	print('STARTING FROM CHECKPOINT '+recent_model)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	criterion = torch.nn.CrossEntropyLoss()
	print('Training Start')
	loss_values=[]
	model.train()
	for epoch in range(l[0][1], num_epochs):
		running_loss = train()
		print('Epoch '+str(epoch+1)+' Complete')
		loss_values.append(running_loss / len(train_loader))
		if (epoch+1) % 25 == 0:
			print('TESTING CHECKPOINT')
			model.eval()
			train_acc_balanced, train_acc = test(train_loader)
			val_acc_balanced, val_acc = test(val_loader)
			test_acc_balanced, test_acc = test(test_loader)
			model_name = 'model_'+str(drop)+'_layers_3_epochs_'+str(epoch+1)+'_lr_'+str(lr)+'_bs_'+str(bs)+'_acc_'+str(round(val_acc_balanced,3))+'_train_'+str(round(train_acc_balanced,3))
			print('TRAIN ACC: '+str(train_acc_balanced))
			print('VAL ACC: '+str(val_acc_balanced))
			print('TEST ACC: '+str(test_acc_balanced))
			with open('binary/model_out/'+model_name,'w') as f:
				f.write(str(train_acc_balanced)+','+str(val_acc_balanced)+','+str(test_acc_balanced))
	            #f.write(str(train_acc_balanced)+','+str(val_acc_balanced))
			torch.save(model, 'binary/saved_models_split/'+model_name+'.pth')
			model.train()
else:
	print('STARTING FROM SCRATCH')
	model = GCN(hidden_channels=1000).to(device=device)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	criterion = torch.nn.CrossEntropyLoss()
	print('Training Start')
	loss_values=[]
	model.train()
	for epoch in range(num_epochs):
	    running_loss = train()
	    print('Epoch '+str(epoch+1)+' Complete')
	    loss_values.append(running_loss / len(train_loader))
	    if (epoch+1) % 25 == 0:
	        print('TESTING CHECKPOINT')
	        model.eval()
	        train_acc_balanced, train_acc = test(train_loader)
	        val_acc_balanced, val_acc = test(val_loader)
	        test_acc_balanced, test_acc = test(test_loader)
	        model_name = 'model_'+str(drop)+'_layers_3_epochs_'+str(epoch+1)+'_lr_'+str(lr)+'_bs_'+str(bs)+'_acc_'+str(round(val_acc_balanced,3))+'_train_'+str(round(train_acc_balanced,3))
	        print('TRAIN ACC: '+str(train_acc_balanced))
	        print('VAL ACC: '+str(val_acc_balanced))
	        print('TEST ACC: '+str(test_acc_balanced))
	        with open('binary/model_out/'+model_name,'w') as f:
	            f.write(str(train_acc_balanced)+','+str(val_acc_balanced)+','+str(test_acc_balanced))
	            #f.write(str(train_acc_balanced)+','+str(val_acc_balanced))
	        torch.save(model, 'binary/saved_models_split/'+model_name+'.pth')
	        model.train()
ctrs = []
ctr=0
for a in loss_values:
    ctrs.append(ctr)
    ctr+=1
plt.plot(ctrs, loss_values)
plt.title(model_name)
plt.ylabel('loss values')
plt.savefig('binary/loss_plots/'+model_name+'.png')
# if __name__ == "__main__":
#     main()