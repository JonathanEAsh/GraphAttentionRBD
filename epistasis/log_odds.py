import pandas as pd
files_456 = ['456_betas_worse_b_norm.csv','456_betas_worse_nb_norm.csv']
files_505 = ['505_betas_worse_b_norm.csv','505_betas_worse_nb_norm.csv']
#456
betas = []
difference = []
df1 = pd.read_csv(files_456[0])
df2 = pd.read_csv(files_456[1])
for a, b in zip(df1['betas'], df1['values']):
	if a.count('_') == 2:
		for c, d in zip(df2['betas'], df2['values']):
			if c == a:
				betas.append(a)
				difference.append(b - d)
				break
df3 = pd.DataFrame(zip(betas, difference), columns=['betas','difference_wb-wnb'])
df3.to_csv('456_log_odds.csv', index=False)

#505
betas = []
difference = []
df1 = pd.read_csv(files_505[0])
df2 = pd.read_csv(files_505[1])
for a, b in zip(df1['betas'], df1['values']):
	if a.count('_') == 2:
		for c, d in zip(df2['betas'], df2['values']):
			if c == a:
				betas.append(a)
				difference.append(b - d)
				break
df4 = pd.DataFrame(zip(betas, difference), columns=['betas','difference_wb-wnb'])
df4.to_csv('505_log_odds.csv', index=False)

import matplotlib.pyplot as plt
hist1 = df3['difference_wb-wnb'].hist(grid=False)
plt.title('Proportional Odds Assumption Library 456')
plt.xlabel('Single-Body Coefficient Differences')
plt.ylabel('Counts')
plt.savefig('456_odds.png')
plt.show()
hist2 = df4['difference_wb-wnb'].hist(grid=False)
plt.title('Proportional Odds Assumption Library 505')
plt.xlabel('Single-Body Coefficient Differences')
plt.ylabel('Counts')
plt.savefig('505_odds.png')
plt.show()