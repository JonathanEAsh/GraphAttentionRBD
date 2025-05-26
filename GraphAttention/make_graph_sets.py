import pandas as pd
import subprocess
import os
df = pd.read_csv('round_1_val.csv')
val_set = list(df['name'])
train_ctr = 0
val_ctr = 0
print('SPLIT PROTOCOL')
for files in os.listdir('raw_graphs_binary/r1/'):
	if files.replace('.pt','') in val_set:
		subprocess.call(['cp','raw_graphs_binary/r1/'+files, 'train_r1_test_r2/val/raw/'])
		val_ctr += 1
	else:
		subprocess.call(['cp','raw_graphs_binary/r1/'+files, 'train_r1_test_r2/train/raw/'])
		train_ctr += 1
	if (train_ctr + val_ctr) % 1000 == 0:
		print('TRAIN: '+str(train_ctr))
		print('VAL: '+str(val_ctr))
test_ctr=0
for files in os.listdir('raw_graphs_binary/r2/'):
	subprocess.call(['cp','raw_graphs_binary/r2/'+files, 'train_r1_test_r2/test/raw/'])
	test_ctr += 1
	if test_ctr % 1000 == 0:
		print('TRAIN: '+str(train_ctr))
		print('VAL: '+str(val_ctr))
		print('TEST: '+str(test_ctr))

# print('COMBO PROTOCOL')
# df = pd.read_csv('round_1_2_val.csv')
# val_set = list(df['name'])
# train_ctr = 0
# val_ctr = 0
# for files in os.listdir('raw_graphs_binary/r1/'):
# 	if files.replace('.pt','') in val_set:
# 		subprocess.call(['cp','raw_graphs_binary/r1/'+files, 'train_r1_r2/val/raw/'])
# 		val_ctr += 1
# 	else:
# 		subprocess.call(['cp','raw_graphs_binary/r1/'+files, 'train_r1_r2/train/raw/'])
# 		train_ctr += 1
# 	if (train_ctr + val_ctr) % 1000 == 0:
# 		print('TRAIN: '+str(train_ctr))
# 		print('VAL: '+str(val_ctr))
# for files in os.listdir('raw_graphs_binary/r2/'):
# 	if files.replace('.pt','') in val_set:
# 		subprocess.call(['cp','raw_graphs_binary/r2/'+files, 'train_r1_r2/val/raw/'])
# 		val_ctr += 1
# 	else:
# 		subprocess.call(['cp','raw_graphs_binary/r2/'+files, 'train_r1_r2/train/raw/'])
# 		train_ctr += 1
# 	if (train_ctr + val_ctr) % 1000 == 0:
# 		print('TRAIN: '+str(train_ctr))
# 		print('VAL: '+str(val_ctr))