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
import os
# parser = argparse.ArgumentParser()
#parser.add_argument("--drop", type=float, default=0.2)
# parser.add_argument('--model_name', type=str, default='model_175_200_0.0001_8_0.0')
# args = parser.parse_args()
for files in os.listdir('best_model_combo/'):
	model_name = files
	heads = int(model_name.split('_')[4])
	drop = float(model_name.replace('.pt','').split('_')[5])

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
		self.att1=torch.nn.MultiheadAttention(embed_dim=2560, num_heads=heads, add_bias_kv=True, dropout=drop)
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
	names=[]
	predlist=[]
	true=[]
	probs=[]
	for j in range(len(data)//100):
		temp_names = list(data[j*100:(j+1)*100])
		cur_x, cur_y = load_data(temp_names, round_, device)
		out = model(cur_x)
		out = Softmax(dim=1)(out)
		pred = out.argmax(dim=1)
		predlist = predlist + [tensor.item() for tensor in pred]
		true = true + [tensor.item() for tensor in cur_y]
		names = names + [item[0] for item in temp_names]
		probs = probs + [tensor.item() for tensor in out[:,1]]
		print(len(names))
	temp_names = list(data[(j+1)*100:])
	cur_x, cur_y = load_data(temp_names, round_, device)
	out = model(cur_x)
	out = Softmax(dim=1)(out)
	pred = out.argmax(dim=1)
	predlist = predlist + [tensor.item() for tensor in pred]
	true = true + [tensor.item() for tensor in cur_y]
	names = names + [item[0] for item in temp_names]
	probs = probs + [tensor.item() for tensor in out[:,1]]
	print(len(names))
	# true_cpu=[]
	# preds_cpu=[]
	# for a, b in zip(true, preds):
	# 	true_cpu.append(int(a))
	# 	preds_cpu.append(int(b.cpu()))
	# return balanced_accuracy_score(true, preds), accuracy_score(true, preds), names, true_cpu, preds_cpu
	return names, true, predlist, probs

def main():
	# NEW PROTOCOL, NO CV, AVOID LEAKAGE
	# TRAIN
	device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = Net()
	#model.load_state_dict(torch.load('saved_models_combo/'+args.model_name+'.pt'))
	for files in os.listdir('best_model_combo/'):
		model_name = files
		heads = int(model_name.split('_')[4])
		drop = float(model_name.replace('.pt','').split('_')[5])

		model.load_state_dict(torch.load('best_model_combo/'+model_name))
		model.eval()
		model.to(device=device)

		print(' === Train === ')
		df_train = pd.read_csv('round_1_2_train_binary.csv')
		train_dataset = list(zip(df_train['name'], df_train['class']))
		train_names, train_true, train_pred, train_prob = test(train_dataset, 'r1', model, device)
		df_train = pd.DataFrame(zip(train_names, train_true, train_pred, train_prob), columns=['name','true','pred','prob'])
		df_train = df_train.sort_values(by=['prob'], ascending=False)
		df_train.to_csv('rbd_attention_combo_train_probs.csv',index=False)

		print(' === Val ===')
		df_val = pd.read_csv('round_1_2_val_binary.csv')
		val_dataset = list(zip(df_val['name'], df_val['class']))
		val_names, val_true, val_pred, val_prob = test(val_dataset, 'r1', model, device)
		df_val = pd.DataFrame(zip(val_names, val_true, val_pred, val_prob), columns=['name','true','pred','prob'])
		df_val = df_val.sort_values(by=['prob'], ascending=False)
		df_val.to_csv('rbd_attention_combo_val_probs.csv',index=False)

		# print(' === Test ===')
		# df_test = pd.read_csv('round_2_test_binary.csv')
		# test_dataset = list(zip(df_test['name'], df_test['class']))
		# test_names, test_true, test_pred, test_prob = test(test_dataset, 'r1', model, device)
		# df_test = pd.DataFrame(zip(test_names, test_true, test_pred, test_prob), columns=['name','true','pred','prob'])
		# df_test = df_test.sort_values(by=['prob'], ascending=False)
		# df_test.to_csv('rbd_attention_split_test_probs.csv',index=False)
	# print('50')
	# df_test = pd.read_csv('round_2_test.csv')
	# test_dataset = list(zip(df_test['name'], df_test['class']))
	# test_acc_bal, test_acc = test(test_dataset, 'r2', model, device)
	# print('100')
	# df_test_100 = pd.read_csv('round_2_test_100.csv')
	# test_dataset_100 = list(zip(df_test_100['name'], df_test_100['class']))
	# test_acc_bal_100, test_acc_100 = test(test_dataset_100, 'r2', model, device)
	# print('No Filter')
	# df_test_uf = pd.read_csv('round_2_test_uf.csv')
	# test_dataset_uf = list(zip(df_test_uf['name'], df_test_uf['class']))
	# test_acc_bal_uf, test_acc_uf = test(test_dataset_uf, 'r2', model, device)
	
	# with open('test_out/'+args.model_name+'.txt','w') as f:
	# 	f.write(str(test_acc_bal)+','+str(test_acc_bal_100)+','+str(test_acc_bal_uf))
	
	#df_test = pd.read_csv('round_2_test.csv')
	# print(' === Train === ')
	# df_train = pd.read_csv('round_1_2_train.csv')
	# train_dataset = list(zip(df_train['name'], df_train['class']))
	# train_names, train_true, train_pred, train_prob = test(train_dataset, 'r1', model, device)
	# df_train = pd.DataFrame(zip(train_names, train_true, train_pred, train_prob), columns=['name','true','pred','prob'])
	# df_train.to_csv('rbd_attention_combo_train_probs.csv',index=False)


	# print(' === Val ===')
	# df_val = pd.read_csv('round_1_2_val.csv')
	# val_dataset = list(zip(df_val['name'], df_val['class']))
	# val_names, val_true, val_pred, val_prob = test(val_dataset, 'r1', model, device)
	# df_val = pd.DataFrame(zip(val_names, val_true, val_pred, val_prob), columns=['name','true','pred','prob'])
	# df_val.to_csv('rbd_attention_combo_val_probs.csv',index=False)

	#test_acc_bal, test_acc, names, labels, predictions = test(test_dataset, 'r1', model, device)
	#df = pd.DataFrame(zip(names, labels, predictions), columns=['name','bind','pred'])
	# df_og = pd.read_csv('LY010LY011_Output_AnalysedDec1824_argmax.csv')
	# counts = []
	# agree = []
	# for a, b, c in zip(df['bind'], df['pred'], df_og['rij ref']):
	# 	counts.append(c)
	# 	agree.append(int(a==b))
	# df['agree'] = agree
	# df['counts'] = counts
	# df.to_csv('round_1_2_test_preds_with_counts.csv', index=False)
if __name__ == '__main__':
	main()