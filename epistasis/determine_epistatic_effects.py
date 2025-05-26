import pandas as pd
files = ['456_betas_norm.csv','505_betas_norm.csv']
b0 = [-9.503733154976636, -8.7854326897221]
for file in files:
	df = pd.read_csv(file)
	names = []
	composite = []
	difference = []
	for a, b in zip(df['betas'], df['values']):
		# must be at least double
		if a.count('_')==4:
			ind1 = a.split('_')[1]+'_'+a.split('_')[2]
			ind2 = a.split('_')[3]+'_'+a.split('_')[4]
			# search singles
			cur_comp = 'b0,'
			scores = [b0[files.index(file)]]
			for c, d in zip(df['betas'], df['values']):
				if c.count('_')==2:
					if ind1 in c or ind2 in c:
						cur_comp+=c+','
						scores.append(d)
			names.append(a)
			composite.append(cur_comp[:-1])
			difference.append(b - sum(scores))
		elif a.count('_')==6:
			ind1 = a.split('_')[1]+'_'+a.split('_')[2]
			ind2 = a.split('_')[3]+'_'+a.split('_')[4]
			ind3 = a.split('_')[5]+'_'+a.split('_')[6]
			# search singles
			cur_comp = 'b0,'
			scores = [b0[files.index(file)]]
			for c, d in zip(df['betas'], df['values']):
				if c.count('_')==2:
					if ind1 in c or ind2 in c or ind3 in c:
						cur_comp+=c+','
						scores.append(d)
				elif c.count('_')==4:
					if (ind1 in c and ind2 in c) or (ind1 in c and ind3 in c) or (ind2 in c and ind3 in c):
						cur_comp+=c+','
						scores.append(d) 
			names.append(a)
			composite.append(cur_comp[:-1])
			difference.append(b - sum(scores))
	df2 = pd.DataFrame(zip(names,composite,difference),columns=['name','composite','difference'])
	df2 = df2.sort_values(by=['difference'], ascending=False)
	df2.to_csv(file.replace('.csv','_epistasis.csv'),index=False)
