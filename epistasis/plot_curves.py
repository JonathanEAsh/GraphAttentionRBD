import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, PrecisionRecallDisplay, auc, roc_curve, RocCurveDisplay
import matplotlib.pyplot as plt
from scipy.special import softmax
import os 


feats = ['oh','emme','combo','esm']
md = {'log_val':[],'log_te':[]}
for a in md.keys():
	for b in feats:
		for files in os.listdir('results_cv/'):
			sp = files.split('_')
			if 'mpnn_tr_ultra_te_'+a.split('_')[1] in files and sp[-1].replace('.csv','')==a.split('_')[0] and sp[-2] == b:
				print(files)
				df = pd.read_csv('results_cv/'+files)
				true = list(df['bind'])
				pred = list(df['pred'])
				prob = list(df['prob_b'])
				# precision recall
				fig, ax = plt.subplots(layout='constrained')
				precision1, recall1, _ = precision_recall_curve(true, prob)
				av1 = average_precision_score(true, prob)
				disp1 = PrecisionRecallDisplay(precision=precision1, recall=recall1)
				disp1.plot(color='red', label='Average Precision = '+str(round(av1,3)), ax=ax)
				ax.axline((0,true.count(1) / (len(true)-true.count(1))), slope=0, color='black',linestyle='--')
				plt.xlim(0,1.2)
				plt.ylim(0,1.2)
				plt.xticks([0.5,1], size=20)
				plt.yticks([0.5,1], size=20)
				plt.xlabel('Recall', size=25)
				plt.ylabel('Precision', size=25)
				plt.title(a+'_'+b)
				plt.legend(loc='upper right', fontsize=15)
				#plt.savefig('curves/prec_rec_'+a+'_'+b+'.png')
				plt.close()

				# ROC
				fig, ax = plt.subplots(layout='constrained')
				fpr, tpr, thresholds = roc_curve(true, prob)
				roc_auc = auc(fpr, tpr)
				display1 = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
				display1.plot(color='red', label='Area Under Curve = '+str(round(roc_auc,3)), ax=ax)
				ax.axline((0,0), slope=1, color='black',linestyle='--')
				plt.xlim(-0.1,1.2)
				plt.ylim(0,1.2)
				plt.xticks([0.5,1], size=20)
				plt.yticks([0.5,1], size=20)
				plt.xlabel('False Positive Rate', size=25)
				plt.ylabel('True Positive Rate', size=25)
				plt.title(a+'_'+b)
				plt.legend(loc='upper right', fontsize=15)
				#plt.savefig('curves/roc_'+a+'_'+b+'.png')
				plt.close()

				print('AUC: '+str(round(roc_auc,2)))
				print('AUPRC: '+str(round(av1,2)))

# md = {'505':[],'456':[],'mpnn':[]}
# models = ['svc','rf','knn','nb','log']
# for a in models:
# 	print(a+' ====')
# 	for files in os.listdir('df/'):
# 		if 'val' in files and files.split('_')[-2].replace('.csv','') == a and '5fold' in files:
# 			lib = files.split('_')[0]
# 			print(files)
# 			df = pd.read_csv('df/'+files)
# 			true = list(df['bind'])
# 			pred = list(df['pred'])
# 			prob = list(df['prob_b'])
# 			# precision recall
# 			fig, ax = plt.subplots(layout='constrained')
# 			precision1, recall1, _ = precision_recall_curve(true, prob)
# 			av1 = average_precision_score(true, prob)
# 			disp1 = PrecisionRecallDisplay(precision=precision1, recall=recall1)
# 			disp1.plot(color='red', label='Average Precision = '+str(round(av1,3)), ax=ax)
# 			ax.axline((0,true.count(1) / (len(true)-true.count(1))), slope=0, color='black',linestyle='--')
# 			plt.xlim(0,1.2)
# 			plt.ylim(0,1.2)
# 			plt.xticks([0.5,1], size=20)
# 			plt.yticks([0.5,1], size=20)
# 			plt.xlabel('Recall', size=25)
# 			plt.ylabel('Precision', size=25)
# 			plt.title(a+'_'+lib)
# 			plt.legend(loc='upper right', fontsize=15)
# 			plt.savefig('curves/prec_rec_'+a+'_'+lib+'.png')
# 			plt.close()

# 			# ROC
# 			fig, ax = plt.subplots(layout='constrained')
# 			fpr, tpr, thresholds = roc_curve(true, prob)
# 			roc_auc = auc(fpr, tpr)
# 			display1 = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
# 			display1.plot(color='red', label='Area Under Curve = '+str(round(roc_auc,3)), ax=ax)
# 			ax.axline((0,0), slope=1, color='black',linestyle='--')
# 			plt.xlim(-0.1,1.2)
# 			plt.ylim(0,1.2)
# 			plt.xticks([0.5,1], size=20)
# 			plt.yticks([0.5,1], size=20)
# 			plt.xlabel('False Positive Rate', size=25)
# 			plt.ylabel('True Positive Rate', size=25)
# 			plt.title(a+'_'+lib)
# 			plt.legend(loc='upper right', fontsize=15)
# 			plt.savefig('curves/roc_'+a+'_'+lib+'.png')
# 			plt.close()




# lib_num = '456'
# file_list = [lib_num+'_val_preds.csv', lib_num+'_tr_preds.csv',lib_num+'_val_preds_uptodouble.csv', lib_num+'_tr_preds_uptodouble.csv',\
# 			lib_num+'_val_preds_uptosingle.csv', lib_num+'_tr_preds_uptosingle.csv']

# for a in file_list:
# 	df = pd.read_csv(a)
# 	# PRECISION RECALL
# 	fig, ax = plt.subplots()
# 	# For each class
# 	print(a)
# 	precision1, recall1, _ = precision_recall_curve(df['bind'], df['prob_b'])
# 	av1 = average_precision_score(df['bind'], df['prob_b'])
# 	disp1 = PrecisionRecallDisplay(precision=precision1, recall=recall1)
# 	disp1.plot(color='red', label='Worse=NB, AP='+str(round(av1,3)), ax=ax)

# 	if 'val' in a:
# 		prec_name = 'Library '+lib_num+' Validation Set Precision Recall Curves'
# 		filename = 'precision_recall_val'
# 	else:
# 		prec_name = 'Library '+lib_num+' Training Set Precision Recall Curves'
# 		filename = 'precision_recall_tr'
# 	if 'double' in a:
# 		prec_name = prec_name + '\nSingle and Double Body Terms'
# 		filename=filename+'_double.png'
# 	elif 'single' in a:
# 		prec_name = prec_name + '\nSingle Body Terms'
# 		filename=filename+'_single.png'
# 	else:
# 		prec_name = prec_name + '\nSingle, Double, and Triple Body Terms'
# 		filename=filename+'_triple.png'
# 	plt.title(prec_name)
# 	plt.savefig(lib_num+'_plots/'+filename)
# 	# plt.show()
# 	plt.close()

# 	# ROC
# 	fig, ax = plt.subplots()
# 	# For each class
# 	fpr, tpr, thresholds = roc_curve(df['bind'], df['prob_b'])
# 	roc_auc = auc(fpr, tpr)
# 	display1 = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
# 	display1.plot(color='red', label='Worse=NB, AUC='+str(round(roc_auc,3)), ax=ax)

# 	# plt.show()
# 	if 'val' in a:
# 		prec_name = 'Library '+lib_num+' Validation Set ROC Curves'
# 		filename = 'roc_val'
# 	else:
# 		prec_name = 'Library '+lib_num+' Training Set ROC Curves'
# 		filename = 'roc_tr'
# 	if 'double' in a:
# 		prec_name = prec_name + '\nSingle and Double Body Terms'
# 		filename=filename+'_double.png'
# 	elif 'single' in a:
# 		prec_name = prec_name + '\nSingle Body Terms'
# 		filename=filename+'_single.png'
# 	else:
# 		prec_name = prec_name + '\nSingle, Double, and Triple Body Terms'
# 		filename=filename+'_triple.png'
# 	plt.title(prec_name)
# 	plt.savefig(lib_num+'_plots/'+filename)
# 	# plt.show()
# 	plt.close()