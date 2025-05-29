import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score
import random
import pickle
from sklearn.model_selection import StratifiedKFold
import numpy as np
import statistics
# data = {'color': ['red', 'blue', 'green', 'red'], 
#         'size': ['small', 'medium', 'large', 'medium'],
#         'price': [10, 20, 30, 15]}
# df = pd.DataFrame(data)

# # One-hot encode categorical features
# encoder = OneHotEncoder(sparse_output=False)
# encoded_data = encoder.fit_transform(df[['color', 'size']])
# encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['color', 'size']))

# # Combine encoded features with the target variable
# X = encoded_df
# y = df['price']

# # Train a linear regression model
# model = LinearRegression()
# model.fit(X, y)

# # Get feature weights
# weights = pd.DataFrame({'feature': X.columns, 'weight': model.coef_})
# print(weights)
class_map = {0:1, 1:0, 2:0}
mut_list = []
df=pd.read_csv('round_1_results_combined.csv')
#df = pd.read_csv('sample.csv')
lib_505 = ['498','499','500','501','505']
lib_456 = ['456','473','475','476','485','486']
# tr_names=[]
# tr_seqs=[]
# tr_bind_1=[]
# tr_bind_2=[]
# val_names=[]
# val_seqs=[]
# val_bind_1=[]
# val_bind_2=[]
all_names=[]
all_seqs=[]
all_bind=[]
for a, z, x in zip(df['name'], df['seq'], df['class']):
	if '(' in a:
		check = 0
		muts = a[1:-1].split(',')
		for b in muts:
			if len(b)>0:
				ind = b[1:-1]
				if ind in lib_456:
					check = 1
					if b not in mut_list:
						mut_list.append(b)
		if check:
			check = 0
			all_names.append(a)
			all_seqs.append(z)
			all_bind.append(class_map[x])
			# rd = random.randint(1,10)
			# if rd != 1:
			# 	tr_names.append(a)
			# 	tr_seqs.append(z)
			# 	tr_bind_1.append(class_map_1[x])
			# 	tr_bind_2.append(class_map_2[x])
			# else:
			# 	val_names.append(a)
			# 	val_seqs.append(z)
			# 	val_bind_1.append(class_map_1[x])
			# 	val_bind_2.append(class_map_2[x])
df8 = pd.DataFrame(zip(all_names, all_bind),columns=['name','class'])
#df8.to_csv('505_var.csv',index=False)
print(mut_list)
md = {}
for a in lib_456:
	temp=[]
	for b in mut_list:
		if a in b:
			if len(temp) == 0:
				temp.append(b[0])
			temp.append(b[-1])	
	md[a]=temp
print(md)
# print(len(tr_names))
# print(len(val_names))
aa_list = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']

all_feats=[]
for a, b in zip(all_names, all_seqs):
	curr_onehot = []
	for indices in md.keys():
		cur_aa = b[int(indices)-333]
		for c in aa_list:
			if c == cur_aa:
				curr_onehot.append(1)
			else:
				curr_onehot.append(0)
	examined = []
	for ind1 in md.keys():
		aa1 = b[int(ind1)-333]
		for ind2 in md.keys():
			ex_str = ind1+'_'+ind2
			ex_str_1 = ind2+'_'+ind1
			if ind2 != ind1 and ex_str not in examined and ex_str_1 not in examined:
				examined.append(ex_str)
				examined.append(ex_str_1)
				aa2 = b[int(ind2)-333]
				for i in aa_list:
					for j in aa_list:
						if i == aa1 and j == aa2:
							curr_onehot.append(1)
						else:
							curr_onehot.append(0)
	examined = []
	for ind1 in md.keys():
		aa1 = b[int(ind1)-333]
		for ind2 in md.keys():
			aa2 = b[int(ind2)-333]
			for ind3 in md.keys():
				ex_str = ind1+'_'+ind2+'_'+ind3
				ex_str_1 = ind1+'_'+ind3+'_'+ind2
				ex_str_2 = ind2+'_'+ind1+'_'+ind3
				ex_str_3 = ind2+'_'+ind3+'_'+ind1
				ex_str_4 = ind3+'_'+ind1+'_'+ind2
				ex_str_5 = ind3+'_'+ind2+'_'+ind1
				if ind2 != ind1 and ind3 != ind2 and ind3 != ind1 and ex_str not in examined and ex_str_1 not in examined and ex_str_2 not in examined and ex_str_3 not in examined and ex_str_4 not in examined and ex_str_5 not in examined:
					examined.append(ex_str)
					examined.append(ex_str_1)
					examined.append(ex_str_2)
					examined.append(ex_str_3)
					examined.append(ex_str_4)
					examined.append(ex_str_5)
					aa3 = b[int(ind3)-333]
					for i in aa_list:
						for j in aa_list:
							for k in aa_list:
								if i == aa1 and j == aa2 and k == aa3:
									curr_onehot.append(1)
								else:
									curr_onehot.append(0)
	all_feats.append(curr_onehot)
	if len(all_feats) % 100 == 0:
		print(str(len(all_feats))+' OUT OF '+str(len(all_names)))
skf = StratifiedKFold(n_splits=10)
print('TRAINING START Ver 1')
best_score = 0
all_feats = np.array(all_feats)
all_bind = np.array(all_bind)
all_names = np.array(all_names)
perform = []
for i, (train_index, test_index) in enumerate(skf.split(all_feats, all_bind)):
	print('FOLD '+str(i+1))
	# TRAIN
	tr_feats = all_feats[train_index]
	tr_bind = all_bind[train_index]
	tr_names = all_names[train_index]
	cl = LogisticRegression()
	cl.fit(tr_feats, tr_bind)
	# VALIDATE
	val_feats = all_feats[test_index]
	val_bind = all_bind[test_index]
	val_names = all_names[test_index]
	pred = cl.predict(val_feats)
	score = balanced_accuracy_score(val_bind, pred)
	perform.append(score)
	if score > best_score:
		tr_pred = cl.predict(tr_feats)
		tr_probs = cl.predict_proba(tr_feats)
		tr_prob_b = [i[1] for i in tr_probs]
		tr_prob_nb = [i[0] for i in tr_probs]
		val_probs = cl.predict_proba(val_feats)
		val_prob_b = [i[1] for i in val_probs]
		val_prob_nb = [i[0] for i in val_probs]

		print(str(score)+' OVER '+str(best_score))
		best_score = score
		df_tr = pd.DataFrame(zip(tr_names, tr_bind, tr_pred, tr_prob_b, tr_prob_nb), columns=['name','bind','pred','prob_b','prob_nb'])
		df_tr.to_csv('456_tr_preds_uptosingle.csv', index=False)
		df_val = pd.DataFrame(zip(val_names, val_bind, pred, val_prob_b, val_prob_nb), columns=['name','bind','pred','prob_b','prob_nb'])
		df_val.to_csv('456_val_preds_uptosingle.csv', index=False)
		col = [str(i) for i in range(len(val_feats[0]))]
		coef = []
		for a in cl.coef_[0]:
			coef.append(a)
		weights = pd.DataFrame(zip(col, coef), columns=['feature','weight'])
		weights.to_csv('weights_505_uptosingle.csv',index=False)
		with open('model_505_uptosingle.pkl','wb') as f:
			pickle.dump(cl, f)
print('BA ACROSS TEN FOLDS')
print(str(round(max(perform)*100,3))+'% BEST, '+str(round(statistics.mean(perform)*100,3))+'% AVG, '+str(round(statistics.stdev(perform)*100,3))+'% STDEV')


# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# NOW DOING TRIPLES - NEW METHOD w/ Stratified CV
# 498 499 500 501 505
# Features 1-100: single body - 20 per residue
# Features 101-4100: double body - 400 per residue pair
# Features 4101-84100: triple body - 8000 per residue pair
# PERFORMANCE - Worse = NB
# SINGLE ONLY: 89.0% BEST, 78.5% AVG, 10.8% STDEV
# SINGLE + DOUBLE: 92.7% BEST, 79.9% AVG, 10.6% STDEV
# SINGLE + DOUBLE + TRIPLE: 93.9% BEST, 79.5% AVG, 10.3% STDEV

# 456 473 475 476 485 486
# Features 1-120: single body - 20 per residue
# Features 121-6120: double body - 400 per residue pair
# Features 6121-166120: triple body - 8000 per residue triplet
# PERFORMANCE - Worse = NB
# SINGLE ONLY: 68.4% BEST, 61.3% AVG, 5.6% STDEV
# SINGLE + DOUBLE: 78.7% BEST, 67.9% AVG, 6.9% STDEV
# SINGLE + DOUBLE + TRIPLE: 78.9% BEST, 68.3% AVG, 6.8% STDEV
