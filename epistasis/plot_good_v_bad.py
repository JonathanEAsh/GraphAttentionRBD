import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, mean_absolute_error
md = {'456':[5.419, 0.038],'505':[14.693, 0.706]}

print(md)
libnames = ['LY006','LY005']



colors = ['blue','red']
#colors = ['lightsteelblue','cornflowerblue','royalblue']

f, ax = plt.subplots(layout='constrained')
xpos = [1,2]
big = []
for i in range(len(libnames)):
	temp = []
	for j in md.values():
		temp.append(j[i])
	big.append(temp)
print(big)

svc = ax.bar([i-0.1 for i in xpos], big[0], color=colors[0], width=0.2, label='Positive Epistatic Samples', edgecolor='black')
ax.bar_label(svc)
rf = ax.bar([i+0.1 for i in xpos], big[1], color=colors[1], width=0.2, label='Negative Epistatic Samples', edgecolor='black')
ax.bar_label(rf)
# knn = ax.bar([i+0.2 for i in xpos], big[2], color=colors[2], width=0.2, label='Up To Single-Body', edgecolor='black')
# ax.bar_label(knn)
# nb = ax.bar([i+0.5 for i in xpos], big[3], color=colors[3], width=0.2, label='NB', edgecolor='black')
# ax.bar_label(nb)
# ax2.set_ylim([0,6])
#ax.set_ylim([0.5,1])
ax.set_xticks(xpos)
ax.set_xticklabels(libnames,size=13)
ax.legend(loc='upper left')#, bbox_to_anchor=(-0.05,1.04))
#ax.set_title('Best Validation BA Achieved By\nDifferent Simple Models Separated By Library', size=18)
ax.set_title('Percentage of Binders Among\nStrong Positive and Negative Beta-Containing\nSamples Separated By Library', size=18)
ax.set_xlabel('Library Name',size=15)
#ax.set_ylabel('Balanced Accuracy',size=15)
ax.set_ylabel('Percentage of Binders', size=15)
#plt.savefig('plots/ba_reddy.png')
plt.savefig('plots/good_v_bad.png')

plt.show()
