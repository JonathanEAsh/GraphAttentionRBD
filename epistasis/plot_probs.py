import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
from sklearn.metrics import r2_score
df = pd.read_csv('clonal_validation/probs/combo_binary_onlyultra.csv')
files = ['505_tr_preds.csv','505_val_preds.csv','456_tr_preds.csv','456_val_preds.csv']
for file in files:
	df2 = pd.read_csv(file)
	bi = []
	nb = []
	for a, b in zip(df2['bind'], df2['prob_b']):
		if a == 1:
			bi.append(b)
		else:
			nb.append(b)
	plot = plt.violinplot([bi, nb])
	plt.xticks([1,2], labels=['Nonbind','Bind'])
	if 'tr' in file:
		plt.title('True Label vs Predicted Binary Binding Probability\n'+file.split('_')[0]+' Training Set')
	else:
		plt.title('True Label vs Predicted Binary Binding Probability\n'+file.split('_')[0]+' Validation Set')
	plt.xlabel('Class')
	plt.ylabel('Predicted Binding Probability')
	plt.savefig(file.replace('.csv','_prob_violin.png'))
	plt.show()



	pred_prob = []
	true_prob = []
	for a, b, x in zip(df2['name'], df2['prob_b'], df2['bind']):
		for c, d, y in zip(df['name'], df['prob_b'], df['bind']):
			if c == a:
				if x == 1 and d >=0.5:
					pred_prob.append(b)
					true_prob.append(d)
				elif x == 0 and d <= 0.5:
					pred_prob.append(b)
					true_prob.append(d)
				break
	pred_prob = np.array(pred_prob)
	true_prob = np.array(true_prob)
	a, b = np.polyfit(true_prob, pred_prob, 1)
	r = r2_score(true_prob, pred_prob)
	plt.scatter(true_prob, pred_prob, s=5)
	plt.plot(true_prob, a*true_prob + b, color='red', label='R2 = '+str(round(r,3)))
	plt.xlabel('True Binding Probability')
	plt.ylabel('Predicted Binding Probability')
	if 'tr' in file:
		plt.title('True vs Predicted Binary Binding Probability\n'+file.split('_')[0]+' Training Set')
	else:
		plt.title('True vs Predicted Binary Binding Probability\n'+file.split('_')[0]+' Validation Set')
	plt.legend()
	plt.savefig(file.replace('.csv','_prob_scatter.png'))
	plt.show()