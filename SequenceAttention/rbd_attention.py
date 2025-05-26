import torch
from torch.nn import Softmax
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import random
from torch.nn.utils.rnn import pad_sequence
#torch.set_printoptions(threshold=10_000)
import time
import argparse
from torch import linalg as LA
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
#parser.add_argument("--drop", type=float, default=0.2)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=1e-6)
parser.add_argument('--bs', type=int, default=50)
parser.add_argument('--att_drop', type=float, default=0.00)
args = parser.parse_args()
torch.autograd.set_detect_anomaly(True)
def load_data(data, round_, device):
	Y_temp = []
	X_temp = []
	for samples in data:
		if '[' in samples[0]:
			seq_data = torch.load('../embed/r2/'+samples[0]+'.pt')['mean_representations'][33]
		else:
			seq_data = torch.load('../embed/r1/'+samples[0]+'.pt')['mean_representations'][33]
		#seq_data = torch.load('embed/'+round_+'/'+samples[0]+'.pt')['mean_representations'][33]
		X_temp.append(seq_data)
		Y_temp.append(int(samples[1]))	
	Y = torch.tensor(Y_temp).to(device=device)
	X = torch.stack(X_temp).to(device=device)
	return X, Y

class Net(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.lnorm1=torch.nn.LayerNorm(2560)
		self.att1=torch.nn.MultiheadAttention(embed_dim=2560, num_heads=args.heads, add_bias_kv=True, dropout=args.att_drop)
		self.lnorm2=torch.nn.LayerNorm(2560)
		self.f1=torch.nn.Linear(2560, 640)
		self.f2=torch.nn.Linear(640, 160)
		self.f3=torch.nn.Linear(160, 2)
	def forward(self, x):
		x = self.lnorm1(x)
		x, _ = self.att1(x, x, x)
		x = self.lnorm2(x)
		x = self.f1(x)
		x = F.relu(x)
		x = self.f2(x)
		x = F.relu(x)
		x = self.f3(x)
		return x

def test(data, round_, model, device):
	bs=args.bs
	preds=[]
	true=[]
	for j in range(len(data)//bs):
		temp_names = list(data[j*bs:(j+1)*bs])
		cur_x, cur_y = load_data(temp_names, round_, device)
		out = model(cur_x)
		out = Softmax(dim=1)(out)
		pred = out.argmax(dim=1)
		preds = preds + list(pred.cpu())
		true = true + list(cur_y.cpu())
	temp_names = list(data[(j+1)*bs:])
	cur_x, cur_y = load_data(temp_names, round_, device)
	out = model(cur_x)
	out = Softmax(dim=1)(out)
	pred = out.argmax(dim=1)
	preds = preds + list(pred.cpu())
	true = true + list(cur_y.cpu())
	true_cpu=[]
	preds_cpu=[]
	for a, b in zip(true, preds):
		true_cpu.append(int(a))
		preds_cpu.append(int(b.cpu()))
	return balanced_accuracy_score(true, preds), accuracy_score(true, preds)

def main():
	# NEW PROTOCOL, NO CV, AVOID LEAKAGE
	# TRAIN
	# df_train = pd.read_csv('round_1_train_resample_binary.csv')
	# df_val = pd.read_csv('round_1_val_binary.csv')
	# df_test = pd.read_csv('round_2_test_binary.csv')
	df_train = pd.read_csv('round_1_2_train_binary.csv')
	df_val = pd.read_csv('round_1_2_val_binary.csv')
	device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = Net()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	criterion = torch.nn.CrossEntropyLoss()
	num_epochs=args.epochs
	bs=args.bs
	model.to(device=device)
	train_dataset = list(zip(df_train['name'], df_train['class']))
	model.train()
	loss_values=[]
	for i in range(num_epochs):
		running_loss=0
		random.shuffle(train_dataset)
		for j in range(len(train_dataset)//bs):
			temp_names = list(train_dataset[j*bs:(j+1)*bs])
			#print(((k+1)*bs)/len(train_dataset))
			cur_x, cur_y = load_data(temp_names, 'r1', device)
			y_pred = model(cur_x)
			loss = criterion(y_pred, cur_y)
			print(loss)
			print('$$$$$$$$$$')
			running_loss += loss.item() * len(temp_names)
			#if k % 10 == 0:
			
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
		# GET REMAINDER OF DATASET
		temp_names = list(train_dataset[(j+1)*bs:])
		#print(len(temp_names))
		#print(len(train_dataset))
		cur_x, cur_y = load_data(temp_names, 'r1', device)
		y_pred = model(cur_x)
		loss = criterion(y_pred, cur_y)
		running_loss += loss.item() * len(temp_names)
		print(loss)
		print('$$$$$$$$$$')
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()
		print('EPOCH '+str(i+1)+ ' COMPLETE')
		if (i + 1) % 25 == 0:
			#TEST
			print('TESTING CHECKPOINT')
			model.eval()
			model_name = 'model_'+str(i+1)+'_'+str(args.bs)+'_'+str(args.lr)+'_'+str(args.heads)+'_'+str(args.att_drop)
			#Training Set
			train_acc_bal, train_acc = test(train_dataset, 'r1', model, device)
			#Validation Set
			
			
			val_dataset = list(zip(df_val['name'], df_val['class']))
			val_acc_bal, val_acc = test(val_dataset, 'r1', model, device)
			#Test Set
						
			# test_dataset = list(zip(df_test['name'], df_test['class']))
			# test_acc_bal, test_acc = test(test_dataset, 'r2', model, device)

			with open('model_out/'+model_name+'.txt','w') as f:
				#f.write(str(train_acc_bal)+','+str(val_acc_bal)+','+str(test_acc_bal))
				f.write(str(train_acc_bal)+','+str(val_acc_bal))
			torch.save(model.state_dict(),'saved_models_combo/'+model_name+'.pt')
			model.train()
		loss_values.append(running_loss / len(train_dataset))
	ctrs = []
	ctr=0
	for a in loss_values:
		ctrs.append(ctr)
		ctr+=1
	plt.plot(ctrs, loss_values)
	plt.title(model_name)
	plt.ylabel('loss values')
	plt.savefig('loss_plots/'+model_name+'.png')

if __name__ == '__main__':
	main()