import pandas as pd
df = pd.read_csv('esm_stats_bind.csv')
names=[]
avgs=[]
bind=[]
ctr=0
for a, b, c in zip(df['name'], df['avgs'], df['bind']):
	if '[' in a:
		names.append(a)
		avgs.append(b)
		bind.append(c)
		if len(names) == 500:
			df2 = pd.DataFrame(zip(names, avgs, bind),columns=['name','avgs','bind'])
			df2.to_csv('r2_samples_split/r2_'+str(ctr)+'.csv', index=False)
			ctr+=1
			names = []
			avgs = []
			bind = []
df2 = pd.DataFrame(zip(names, avgs, bind),columns=['name','avgs','bind'])
df2.to_csv('r2_samples_split/r2_'+str(ctr)+'.csv', index=False)	
