import pandas as pd
import statistics
#files = ['505_betas_norm.csv']
files = ['456_betas_norm.csv','505_betas_norm.csv']
b0 = [-9.503733154976636, -8.7854326897221]
for file in files:
	cur_b0 = b0[files.index(file)]
	df = pd.read_csv(file)
	df_genetic = pd.read_csv(file.split('_')[0]+'_var_gen_scores.csv')
	names = []
	composite = []
	difference = []
	md = {}
	# STEP 1 - SINGLE-BODY TERMS
	ctr=0
	for a, b in zip(df['betas'], df['values']):
		# must be at most single
		if a.count('_')==2:
			temp = []
			for gen_scores, elements in zip(df_genetic['genetic_scores'], df_genetic['components']):
				cur_comp = elements.split(',')
				if a in cur_comp:
					temp.append(gen_scores)
			md[a]=[statistics.mean(temp)-cur_b0, 'b0'] # main effects
			ctr+=1
			if ctr%100 == 0:
 				print(ctr)
	# STEP 2 - DOUBLE-BODY TERMS
	for a, b in zip(df['betas'], df['values']):
		# must be double
		if a.count('_')==4:
			ind1 = a.split('_')[1]+'_'+a.split('_')[2]
			ind2 = a.split('_')[3]+'_'+a.split('_')[4]
			temp = []
			for gen_scores, elements in zip(df_genetic['genetic_scores'], df_genetic['components']):
				cur_comp = elements.split(',')
				if a in cur_comp:
					temp.append(gen_scores)
			cur_val = statistics.mean(temp) # mean genetic score of variants containing pair
			cur_comp_str = ''
			for c in df['betas']:
				if c.count('_')==2:
					if ind1 in c or ind2 in c:
						cur_comp_str+=c+',' # single body components
			comp_list = cur_comp_str[:-1].split(',')
			temp = []
			for components in comp_list:
				temp.append(md[components][0]) # single body effects
			md[a]=[cur_val - (sum(temp) + cur_b0), cur_comp_str+'b0']
			ctr+=1
			if ctr%100 == 0:
 				print(ctr)
	# STEP 3 - TRIPLE-BODY TERMS
	for a, b in zip(df['betas'], df['values']):
		# must be triple
		if a.count('_')==6:
			ind1 = a.split('_')[1]+'_'+a.split('_')[2]
			ind2 = a.split('_')[3]+'_'+a.split('_')[4]
			ind3 = a.split('_')[5]+'_'+a.split('_')[6]
			temp = []
			for gen_scores, elements in zip(df_genetic['genetic_scores'], df_genetic['components']):
				cur_comp = elements.split(',')
				if a in cur_comp:
					temp.append(gen_scores)
			cur_val = statistics.mean(temp) # mean genetic score of variants containing triplet
			cur_comp_str = ''
			for c in df['betas']:
				if c.count('_')==2:
					if ind1 in c or ind2 in c or ind3 in c:
						cur_comp_str+=c+',' # single body components
				elif c.count('_')==4:
					if (ind1 in c and ind2 in c) or (ind1 in c and ind3 in c) or (ind2 in c and ind3 in c):
						cur_comp_str+=c+',' # double body components
			comp_list = cur_comp_str[:-1].split(',')
			temp = []
			for components in comp_list:
				temp.append(md[components][0]) # single and double body effects
			md[a]=[cur_val - (sum(temp) + cur_b0), cur_comp_str+'b0']
			ctr+=1
			if ctr%100 == 0:
 				print(ctr)
	# STEP 4 - PUT IT ALL TOGETHER
	for keys, values in md.items():
		names.append(keys)
		composite.append(values[1])
		difference.append(values[0])
		df2 = pd.DataFrame(zip(names,composite,difference),columns=['name','composite','difference'])
		df2 = df2.sort_values(by=['difference'], ascending=False)
		df2.to_csv(file.replace('.csv','_epistasis.csv'),index=False)

# for file in files:
# 	df = pd.read_csv(file)
# 	df_genetic = pd.read_csv(file.split('_')[0]+'_var_gen_scores.csv')
# 	names = []
# 	composite = []
# 	difference = []
# 	ctr=0
# 	for a, b in zip(df['betas'], df['values']):
# 		# must be at least double
# 		if a.count('_')==4:
# 			ind1 = a.split('_')[1]+'_'+a.split('_')[2]
# 			ind2 = a.split('_')[3]+'_'+a.split('_')[4]
# 			# search singles
# 			cur_comp_str = ''
# 			scores = [b0[files.index(file)]]
# 			for c, d in zip(df['betas'], df['values']):
# 				if c.count('_')==2:
# 					if ind1 in c or ind2 in c:
# 						cur_comp_str+=c+','
# 			comp_list = cur_comp_str[:-1].split(',')
# 			for components in comp_list:
# 				temp = []
# 				for gen_scores, elements in zip(df_genetic['genetic_scores'], df_genetic['components']):
# 					cur_comp = elements.split(',')
# 					if components in cur_comp:
# 						temp.append(gen_scores)
# 				scores.append(statistics.mean(temp))
# 				# scores contains lower order values to be subtracted
# 			temp = []
# 			for gen_scores, elements in zip(df_genetic['genetic_scores'], df_genetic['components']):
# 				cur_comp = elements.split(',')
# 				if a in cur_comp: #highest order term
# 					temp.append(gen_scores)
# 			cur_val = statistics.mean(temp)
# 			names.append(a)
# 			composite.append(cur_comp_str +'b0')
# 			difference.append(cur_val - sum(scores))
# 		elif a.count('_')==6:
# 			ind1 = a.split('_')[1]+'_'+a.split('_')[2]
# 			ind2 = a.split('_')[3]+'_'+a.split('_')[4]
# 			ind3 = a.split('_')[5]+'_'+a.split('_')[6]
# 			# search singles
# 			cur_comp_str = ''
# 			scores = [b0[files.index(file)]]
# 			for c, d in zip(df['betas'], df['values']):
# 				if c.count('_')==2:
# 					if ind1 in c or ind2 in c or ind3 in c:
# 						cur_comp_str+=c+','
# 				elif c.count('_')==4:
# 					if (ind1 in c and ind2 in c) or (ind1 in c and ind3 in c) or (ind2 in c and ind3 in c):
# 						cur_comp_str+=c+','
# 			comp_list = cur_comp_str[:-1].split(',')
# 			for components in comp_list:
# 				temp = []
# 				for gen_scores, elements in zip(df_genetic['genetic_scores'], df_genetic['components']):
# 					cur_comp = elements.split(',')
# 					if components in cur_comp:
# 						temp.append(gen_scores)
# 				scores.append(statistics.mean(temp))
# 				# scores contains lower order values to be subtracted
# 			temp = []
# 			for gen_scores, elements in zip(df_genetic['genetic_scores'], df_genetic['components']):
# 				cur_comp = elements.split(',')
# 				if a in cur_comp: #highest order term
# 					temp.append(gen_scores)
# 			cur_val = statistics.mean(temp)
# 			names.append(a)
# 			composite.append(cur_comp_str+'b0')
# 			difference.append(cur_val - sum(scores))
# 		ctr+=1
# 		if ctr%100 == 0:
# 			print(ctr)
# 	df2 = pd.DataFrame(zip(names,composite,difference),columns=['name','composite','difference'])
# 	df2 = df2.sort_values(by=['difference'], ascending=False)
# 	df2.to_csv(file.replace('.csv','_epistasis.csv'),index=False)
