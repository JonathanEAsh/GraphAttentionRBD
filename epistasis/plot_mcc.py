import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, mean_absolute_error, f1_score, accuracy_score, average_precision_score, auc, roc_curve
md = {'505':[],'456':[]}
models = ['single','double','triple']
files = ['505_val_preds_uptosingle.csv','456_val_preds_uptosingle.csv',\
'505_val_preds_uptodouble.csv','456_val_preds_uptodouble.csv',\
'505_val_preds_uptotriple.csv','456_val_preds_uptotriple.csv']
for a in models:
	print(a+' ====')
	for file in files:
		if a in file:
			lib = file.split('_')[0]
			print(files)
			if 'triple' in file:
				df=pd.read_csv(file.replace('_uptotriple',''))
			else:
				df = pd.read_csv(file)
			true = df['bind']
			pred = df['pred']
			prob = df['prob_b']
			av1 = average_precision_score(true, prob)
			fpr, tpr, thresholds = roc_curve(true, prob)
			roc_auc = auc(fpr, tpr)
			md[lib].append(round(av1,2))
			#md[lib].append(round(f1_score(true, pred),2))
			#md[lib].append(round(matthews_corrcoef(true, pred),2))
			#md[lib].append(round(100*balanced_accuracy_score(true, pred),1))
			#md[lib].append(round(100*accuracy_score(true, pred),1))

print(md)
libnames = ['LY005','LY006']



#colors = ['coral','firebrick','red']
#colors = ['lightsteelblue','cornflowerblue','royalblue']
colors = ['blue','orange','green']

f, ax = plt.subplots(layout='constrained')
xpos = [1,2]
big = []
for i in range(len(models)):
	temp = []
	for j in md.values():
		temp.append(j[i])
	big.append(temp)
print(big)

svc = ax.bar([i-0.2 for i in xpos], big[0], color=colors[0], width=0.2, label='Up To Single-Body', edgecolor='black')
#ax.bar_label(svc,size=14)
rf = ax.bar([i for i in xpos], big[1], color=colors[1], width=0.2, label='Up To Double-Body', edgecolor='black')
#ax.bar_label(rf,size=14)
knn = ax.bar([i+0.2 for i in xpos], big[2], color=colors[2], width=0.2, label='Up To Triple-Body', edgecolor='black')
#ax.bar_label(knn,size=14)
# nb = ax.bar([i+0.5 for i in xpos], big[3], color=colors[3], width=0.2, label='NB', edgecolor='black')
# ax.bar_label(nb)
# ax2.set_ylim([0,6])
ax.set_ylim([0, 160])
ax.set_yticks([50,90])
ax.set_yticklabels([50,90],size=20)
ax.set_xticks(xpos)
ax.set_xticklabels(libnames,size=25)
ax.legend(loc='upper left',fontsize=20)#, bbox_to_anchor=(-0.05,1.04))
#ax.set_title('Best Validation BA Achieved By\nDifferent Simple Models Separated By Library', size=18)
#ax.set_title('Best Logistic Regression Validation BA\nSeparated By Library And Feature Set', size=18)
ax.set_xlabel('Library Name',size=30)
ax.set_ylabel('Balanced Accuracy',size=30)
#ax.set_ylabel('Matthews Correlation Coefficient', size=15)
#plt.savefig('plots/ba_binary_color.png')
#plt.savefig('plots/mcc_binary.png')

plt.show()
