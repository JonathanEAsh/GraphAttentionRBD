import pandas as pd
import matplotlib.pyplot as plt
df_456 = pd.read_csv('456_var_gen_scores.csv')
b_scores = []
nb_scores = []
for a, b in zip(df_456['genetic_scores'], df_456['class']):
	if b == 1:
		b_scores.append(a)
	else:
		nb_scores.append(a)
fig, ax = plt.subplots(figsize=(5,8))
plot = plt.violinplot([nb_scores, b_scores],positions=[1,1.5])
plt.xticks([1,1.5], labels=['Nonbind','Bind'], size=25)
plt.title('LY006 Genetic Scores',size=30)
plt.xlabel('Class',size=30)
plt.ylabel('Genetic Scores', size=30)
plt.yticks([])
plt.savefig('456_scores.png')
plt.close()

df_505 = pd.read_csv('505_var_gen_scores.csv')
b_scores = []
nb_scores = []
for a, b in zip(df_505['genetic_scores'], df_505['class']):
	if b == 1:
		b_scores.append(a)
	else:
		nb_scores.append(a)
fig, ax = plt.subplots(figsize=(5,8))
plot = plt.violinplot([nb_scores, b_scores],positions=[1,1.5])
plt.xticks([1,1.5], labels=['Nonbind','Bind'], size=25)
plt.title('LY005 Genetic Scores', size=30)
plt.xlabel('Class',size=30)
plt.ylabel('Genetic Scores',size=30)
plt.yticks([])
plt.savefig('505_scores.png')
plt.close()