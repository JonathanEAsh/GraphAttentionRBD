import pandas as pd
l = ['456', '505']
for t in l:
	df_gen_scores = pd.read_csv(t+'_var_gen_scores.csv')
	df_epistatis = pd.read_csv(t+'_betas_norm_epistasis.csv')
	good = []
	bad = []
	for a, b in zip(df_epistatis['name'], df_epistatis['difference']):
		if b >= 1:
			good.append(a)
		elif b <= -1:
			bad.append(a)
	good_names = []
	good_b = []
	good_c = []
	bad_names = []
	bad_b = []
	bad_c = []
	for a, b, z in zip(df_gen_scores['class'], df_gen_scores['components'], df_gen_scores['name']):
		cur = b.split(',')
		for c in cur:
			if c in good:
				good_b.append(a)
				good_names.append(z)
				good_c.append(c)
			elif c in bad:
				bad_b.append(a)
				bad_names.append(z)
				bad_c.append(c)
	# df = pd.DataFrame(zip(good_names, good_b, good_c), columns=['name','class','good_component'])
	# df.to_csv('good_v_bad/'+t+'_good.csv', index=False)
	# df2 = pd.DataFrame(zip(bad_names, bad_b, bad_c), columns=['name','class','bad_component'])
	# df2.to_csv('good_v_bad/'+t+'_bad.csv', index=False)
	print('LIBRARY '+t)
	print('PERCENTAGE OF VARIANTS CONTAINING GOOD BETAS THAT ARE BINDERS')
	print(str(round(100*(sum(good_b)/len(good_b)),3))+'%')
	print('PERCENTAGE OF VARIANTS CONTAINING BAD BETAS THAT ARE BINDERS')
	print(str(round(100*(sum(bad_b)/len(bad_b)),3))+'%')
# LIBRARY 456
# PERCENTAGE OF VARIANTS CONTAINING GOOD BETAS THAT ARE BINDERS
# 5.419%
# PERCENTAGE OF VARIANTS CONTAINING BAD BETAS THAT ARE BINDERS
# 0.038%
# LIBRARY 505
# PERCENTAGE OF VARIANTS CONTAINING GOOD BETAS THAT ARE BINDERS
# 14.693%
# PERCENTAGE OF VARIANTS CONTAINING BAD BETAS THAT ARE BINDERS
# 0.706%
