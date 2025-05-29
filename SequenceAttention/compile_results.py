import os
import pandas as pd

names=[]
epochs=[]
bs=[]
lr=[]
heads=[]
attdrop=[]
test=[]
train=[]
val=[]

for files in os.listdir('model_out/'):
	#if 'knn' in files and '4mer' in files:
	print(files)
	with open('model_out/'+files) as f:
		lines=f.readlines()
		x = lines[0]
		train.append(float(x.split(',')[0]))
		val.append(float(x.split(',')[1]))
		#test.append(float(x.split(',')[2]))
		# y = lines[1]
		# best_train.append(float(y.split(',')[1]))
		# best_test.append(float(y.split(',')[3]))
	base=files.replace('.txt','')
	names.append(files.replace('.txt',''))
	epochs.append(int(base.split('_')[1]))
	bs.append(int(base.split('_')[2]))
	lr.append(float(base.split('_')[3]))
	heads.append(int(base.split('_')[4]))
	attdrop.append(float(base.split('_')[5]))
	# if 'model_out' in files:
	# 	base=files.replace('.txt','')
	# 	print(base)
	# 	if fold:
	# 		line_num=int(base.split('_')[7])*num_folds
	# 	else:
	# 		line_num=int(base.split('_')[7])
	# 	with open(files) as f:
	# 		lines=f.readlines()
	# 		if len(lines)>line_num+150+16:
	# 			if fold:
	# 				train.append(round(float(lines[-3].strip().split(' ')[-1]),3))
	# 				test.append(round(float(lines[-1].strip().split(' ')[-1]),3))
	# 			else:
	# 				train.append(round(float(lines[-2].strip().split(' ')[-1]),3))
	# 				test.append(round(float(lines[-1].strip().split(' ')[-1]),3))
	# 			drop.append(float(base.split('_')[3]))
	# 			layers.append(int(base.split('_')[5]))
	# 			epochs.append(int(base.split('_')[7]))
	# 			lr.append(float(base.split('_')[9]))
	# 			bs.append(int(base.split('_')[11]))
#old_df=pd.read_csv('quick_cv_4mer_three_layers_2head_bi.csv')
#df=pd.DataFrame(zip(names, epochs, bs, lr, heads, attdrop, train, val, test),columns=['names','epochs','bs','lr','heads','att_drop','train','val','test'])
df=pd.DataFrame(zip(names, epochs, bs, lr, heads, attdrop, train, val),columns=['names','epochs','bs','lr','heads','att_drop','train','val'])
#df=pd.concat([old_df, df])
df=df.sort_values(by='val',axis=0,ascending=False)
df.to_csv('rbd_self_attention_3fc_relu_attdrop_combo_binary.csv',index=False)
