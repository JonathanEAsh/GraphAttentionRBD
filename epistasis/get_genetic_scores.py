import pandas as pd
md_456 = {'456': ['F', 'S', 'I', 'T'], '473': ['Y', 'F', 'H', 'V', 'L', 'D'], '475': ['A', 'D', 'V', 'L', 'H', 'P'], '476': ['G', 'L', 'S', 'W', 'A', 'V'], '485': ['G', 'A', 'V', 'S', 'L', 'W'], '486': ['F', 'I', 'S', 'T']}
md_505 = {'498': ['R', 'Q', 'V', 'L', 'G', 'E'], '499': ['P', 'F', 'Y', 'S', 'L', 'H'], '500': ['T', 'W', 'S', 'R', 'M', 'L'], '501': ['Y', 'T', 'S', 'N', 'H', 'P'], '505': ['H', 'Y', 'F', 'L', 'D', 'V']}
lib1 = ['498', '499', '500', '501', '505']
lib1_double = ['498_499', '498_500', '498_501', '498_505', '499_500', '499_501', '499_505', '500_501', '500_505', '501_505']
lib1_triple = ['498_499_500', '498_499_501', '498_499_505', '498_500_501', '498_500_505', '498_501_505', '499_500_501', '499_500_505', '499_501_505', '500_501_505']
lib2 = ['456', '473', '475', '476', '485', '486']
lib2_double = ['456_473', '456_475', '456_476', '456_485', '456_486', '473_475', '473_476', '473_485', '473_486', '475_476', '475_485', '475_486', '476_485', '476_486', '485_486']
lib2_triple = ['456_473_475', '456_473_476', '456_473_485', '456_473_486', '456_475_476', '456_475_485', '456_475_486', '456_476_485', '456_476_486', '456_485_486', '473_475_476', '473_475_485', '473_475_486', '473_476_485', '473_476_486', '473_485_486', '475_476_485', '475_476_486', '475_485_486', '476_485_486']

# file_list_456 = ['456_betas_worse_b.csv','456_betas_worse_nb.csv']
# file_list_505 = ['505_betas_worse_b.csv','505_betas_worse_nb.csv']

df_b0 = pd.read_csv('intercept_norm.csv')
ctr = 0
df_var = pd.read_csv('456_var.csv')
scores = []
components = []
for a in df_var['name']:
	muts = a[1:-1].split(',')
	md_cur = {}
	for b in muts:
		if len(b) > 0:
			md_cur[b[1:-1]]=b[-1]
	for b in md_456.keys():
		if b not in md_cur.keys():
			md_cur[b]=md_456[b][0]
	# print(muts)
	# print(md_cur)
	beta_list = []
	for i in lib2:
		beta_list.append('beta_'+i+'_'+md_cur[i])
	for i in lib2_double:
		ind1 = i.split('_')[0]
		ind2 = i.split('_')[1]
		beta_list.append('beta_'+ind1+'_'+md_cur[ind1]+'_'+ind2+'_'+md_cur[ind2])
	for i in lib2_triple:
		ind1 = i.split('_')[0]
		ind2 = i.split('_')[1]
		ind3 = i.split('_')[2]
		beta_list.append('beta_'+ind1+'_'+md_cur[ind1]+'_'+ind2+'_'+md_cur[ind2]+'_'+ind3+'_'+md_cur[ind3])
	beta_str='b0,'
	for i in beta_list:
		beta_str+=i+','
	# print(beta_list)
	beta_vals = []
	df_wb = pd.read_csv('456_betas_norm.csv')
	for c, d in zip(df_wb['betas'], df_wb['values']):
		if c in beta_list:
			beta_vals.append(d)
	for c, d in zip(df_b0['cutoff'], df_b0['b0_norm']):
		if '456' in c:
			beta_vals.append(d)
	scores.append(sum(beta_vals))
	components.append(beta_str[:-1])
	ctr+=1
	if ctr%100 == 0:
		print(ctr)
df_var['components'] = components
df_var['genetic_scores'] = scores

df_var.to_csv('456_var_gen_scores.csv', index=False)

df_var = pd.read_csv('505_var.csv')
scores = []
components = []
ctr=0
for a in df_var['name']:
	muts = a[1:-1].split(',')
	md_cur = {}
	for b in muts:
		if len(b) > 0:
			md_cur[b[1:-1]]=b[-1]
	for b in md_505.keys():
		if b not in md_cur.keys():
			md_cur[b]=md_505[b][0]
	# print(muts)
	# print(md_cur)
	beta_list = []
	for i in lib1:
		beta_list.append('beta_'+i+'_'+md_cur[i])
	for i in lib1_double:
		ind1 = i.split('_')[0]
		ind2 = i.split('_')[1]
		beta_list.append('beta_'+ind1+'_'+md_cur[ind1]+'_'+ind2+'_'+md_cur[ind2])
	for i in lib1_triple:
		ind1 = i.split('_')[0]
		ind2 = i.split('_')[1]
		ind3 = i.split('_')[2]
		beta_list.append('beta_'+ind1+'_'+md_cur[ind1]+'_'+ind2+'_'+md_cur[ind2]+'_'+ind3+'_'+md_cur[ind3])
	# print(beta_list)
	beta_str='b0,'
	for i in beta_list:
		beta_str+=i+','
	beta_vals = []
	df_wb = pd.read_csv('505_betas_norm.csv')
	for c, d in zip(df_wb['betas'], df_wb['values']):
		if c in beta_list:
			beta_vals.append(d)
	for c, d in zip(df_b0['cutoff'], df_b0['b0_norm']):
		if '505' in c:
			beta_vals.append(d)
	scores.append(sum(beta_vals))
	components.append(beta_str[:-1])
	ctr+=1
	if ctr%100 == 0:
		print(ctr)
df_var['components'] = components
df_var['genetic_scores'] = scores
df_var.to_csv('505_var_gen_scores.csv', index=False)


