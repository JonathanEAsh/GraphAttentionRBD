import pandas as pd
nat_505 = ['beta_498_R', 'beta_498_R_499_P', 'beta_498_R_500_T', 'beta_498_R_501_Y', 'beta_498_R_505_H', 'beta_498_R_499_P_500_T', 'beta_498_R_499_P_501_Y', 'beta_498_R_499_P_505_H', \
			'beta_498_R_500_T_501_Y', 'beta_498_R_500_T_505_H', 'beta_498_R_501_Y_505_H', 'beta_499_P', 'beta_499_P_500_T', 'beta_499_P_501_Y', 'beta_499_P_505_H', 'beta_499_P_500_T_501_Y', \
			'beta_499_P_500_T_505_H', 'beta_499_P_501_Y_505_H', 'beta_500_T', 'beta_500_T_501_Y', 'beta_500_T_505_H', 'beta_500_T_501_Y_505_H', 'beta_501_Y', 'beta_501_Y_505_H', 'beta_505_H']
nat_456 = ['beta_456_F', 'beta_456_F_473_Y', 'beta_456_F_475_A', 'beta_456_F_476_G', 'beta_456_F_485_G', 'beta_456_F_486_F', 'beta_456_F_473_Y_475_A', 'beta_456_F_473_Y_476_G', 'beta_456_F_473_Y_485_G', \
			'beta_456_F_473_Y_486_F', 'beta_456_F_475_A_476_G', 'beta_456_F_475_A_485_G', 'beta_456_F_475_A_486_F', 'beta_456_F_476_G_485_G', 'beta_456_F_476_G_486_F', 'beta_456_F_485_G_486_F', 'beta_473_Y', \
			'beta_473_Y_475_A', 'beta_473_Y_476_G', 'beta_473_Y_485_G', 'beta_473_Y_486_F', 'beta_473_Y_475_A_476_G', 'beta_473_Y_475_A_485_G', 'beta_473_Y_475_A_486_F', 'beta_473_Y_476_G_485_G', 'beta_473_Y_476_G_486_F', \
			'beta_473_Y_485_G_486_F', 'beta_475_A', 'beta_475_A_476_G', 'beta_475_A_485_G', 'beta_475_A_486_F', 'beta_475_A_476_G_485_G', 'beta_475_A_476_G_486_F', 'beta_475_A_485_G_486_F', 'beta_476_G', 'beta_476_G_485_G', \
			'beta_476_G_486_F', 'beta_476_G_485_G_486_F', 'beta_485_G', 'beta_485_G_486_F', 'beta_486_F']
print('================== 456 ==================')
df = pd.read_csv('456_betas_norm_epistasis.csv')
nat_456_name = []
nat_456_con = []
nat_456_score = []
mut_456_name = []
mut_456_con = []
mut_456_score = []

for a, b, c in zip(df['name'], df['difference'], df['composite']):
	if a.count('_') == 4:
		sp = c.split(',')[:-1]
		if (sp[0] in nat_456 and sp[1] not in nat_456) or (sp[1] in nat_456 and sp[0] not in nat_456):
			nat_456_name.append(a)
			nat_456_con.append(c)
			nat_456_score.append(b)
		elif sp[0] not in nat_456 and sp[1] not in nat_456:
			mut_456_name.append(a)
			mut_456_con.append(c)
			mut_456_score.append(b)
df2 = pd.DataFrame(zip(nat_456_name, nat_456_con, nat_456_score), columns=['name','composite','difference'])
df2.to_csv('clonal_validation/456_nat_double.csv', index=False)
df3 = pd.DataFrame(zip(mut_456_name, mut_456_con, mut_456_score), columns=['name','composite','difference'])
df3.to_csv('clonal_validation/456_mut_double.csv', index=False)

print('================== 505 ==================')
df = pd.read_csv('505_betas_norm_epistasis.csv')
nat_505_name = []
nat_505_con = []
nat_505_score = []
mut_505_name = []
mut_505_con = []
mut_505_score = []

for a, b, c in zip(df['name'], df['difference'], df['composite']):
	if a.count('_') == 4:
		sp = c.split(',')[:-1]
		if (sp[0] in nat_505 and sp[1] not in nat_505) or (sp[1] in nat_505 and sp[0] not in nat_505):
			nat_505_name.append(a)
			nat_505_con.append(c)
			nat_505_score.append(b)
		elif sp[0] not in nat_505 and sp[1] not in nat_505:
			mut_505_name.append(a)
			mut_505_con.append(c)
			mut_505_score.append(b)
df2 = pd.DataFrame(zip(nat_505_name, nat_505_con, nat_505_score), columns=['name','composite','difference'])
df2.to_csv('clonal_validation/505_nat_double.csv', index=False)
df3 = pd.DataFrame(zip(mut_505_name, mut_505_con, mut_505_score), columns=['name','composite','difference'])
df3.to_csv('clonal_validation/505_mut_double.csv', index=False)

