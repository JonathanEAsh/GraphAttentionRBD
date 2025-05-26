import pandas as pd
import statistics
df = pd.read_csv('intercepts.csv')
lib1 = ['498', '499', '500', '501', '505']
lib1_double = ['498_499', '498_500', '498_501', '498_505', '499_500', '499_501', '499_505', '500_501', '500_505', '501_505']
lib1_triple = ['498_499_500', '498_499_501', '498_499_505', '498_500_501', '498_500_505', '498_501_505', '499_500_501', '499_500_505', '499_501_505', '500_501_505']
lib2 = ['456', '473', '475', '476', '485', '486']
lib2_double = ['456_473', '456_475', '456_476', '456_485', '456_486', '473_475', '473_476', '473_485', '473_486', '475_476', '475_485', '475_486', '476_485', '476_486', '485_486']
lib2_triple = ['456_473_475', '456_473_476', '456_473_485', '456_473_486', '456_475_476', '456_475_485', '456_475_486', '456_476_485', '456_476_486', '456_485_486', '473_475_476', '473_475_485', '473_475_486', '473_476_485', '473_476_486', '473_485_486', '475_476_485', '475_476_486', '475_485_486', '476_485_486']
norm_intercept = []
for a, b in zip(df['file'], df['value']):
	name = a.split('_')[1].replace('.pkl','')+'_betas.csv'

	df_betas = pd.read_csv(name)
	if '456' in a:
		means = []
		# single
		for x in lib2:
			temp = []
			for c, d in zip(df_betas['betas'], df_betas['values']):
				if c.count('_') == 2 and x in c:
					temp.append(d)
			means.append(statistics.mean(temp))
		# double
		for y in lib2_double:
			ind1 = y.split('_')[0]
			ind2 = y.split('_')[1]
			temp = []
			for c, d in zip(df_betas['betas'], df_betas['values']):
				if c.count('_') == 4 and ind1 in c and ind2 in c:
					temp.append(d)
			means.append(statistics.mean(temp))
		# triple
		for y in lib2_triple:
			ind1 = y.split('_')[0]
			ind2 = y.split('_')[1]
			ind3 = y.split('_')[2]
			temp = []
			for c, d in zip(df_betas['betas'], df_betas['values']):
				if c.count('_') == 6 and ind1 in c and ind2 in c and ind3 in c:
					temp.append(d)
			means.append(statistics.mean(temp))
		norm_intercept.append(b + sum(means))
	else:
		means = []
		# single
		for x in lib1:
			temp = []
			for c, d in zip(df_betas['betas'], df_betas['values']):
				if c.count('_') == 2 and x in c:
					temp.append(d)
			means.append(statistics.mean(temp))
		# double
		for y in lib1_double:
			ind1 = y.split('_')[0]
			ind2 = y.split('_')[1]
			temp = []
			for c, d in zip(df_betas['betas'], df_betas['values']):
				if c.count('_') == 4 and ind1 in c and ind2 in c:
					temp.append(d)
			means.append(statistics.mean(temp))
		# triple
		for y in lib1_triple:
			ind1 = y.split('_')[0]
			ind2 = y.split('_')[1]
			ind3 = y.split('_')[2]
			temp = []
			for c, d in zip(df_betas['betas'], df_betas['values']):
				if c.count('_') == 6 and ind1 in c and ind2 in c and ind3 in c:
					temp.append(d)
			means.append(statistics.mean(temp))
		norm_intercept.append(b + sum(means))
df['b0_norm']=norm_intercept
df.to_csv('intercept_norm.csv', index=False)