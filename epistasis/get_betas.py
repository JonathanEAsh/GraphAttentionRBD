import pandas as pd
import math
cutoff_1_456 = 119
cutoff_2_456 = 6119
cutoff_1_505 = 99
cutoff_2_505 = 4099
aa_list = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
lib1 = ['498','499','500','501','505']
lib1_double = []
examined = []
for ind1 in lib1:
	for ind2 in lib1:
		ex_str = ind1+'_'+ind2
		ex_str_1 = ind2+'_'+ind1
		if ind2 != ind1 and ex_str not in examined and ex_str_1 not in examined:
			examined.append(ex_str)
			examined.append(ex_str_1)
			lib1_double.append(ex_str)
lib1_triple = []
examined = []
for ind1 in lib1:
	for ind2 in lib1:
		for ind3 in lib1:
			ex_str = ind1+'_'+ind2+'_'+ind3
			ex_str_1 = ind1+'_'+ind3+'_'+ind2
			ex_str_2 = ind2+'_'+ind1+'_'+ind3
			ex_str_3 = ind2+'_'+ind3+'_'+ind1
			ex_str_4 = ind3+'_'+ind1+'_'+ind2
			ex_str_5 = ind3+'_'+ind2+'_'+ind1
			if ind2 != ind1 and ind3 != ind2 and ind3 != ind1 and ex_str not in examined and ex_str_1 not in examined and ex_str_2 not in examined and ex_str_3 not in examined and ex_str_4 not in examined and ex_str_5 not in examined:
				examined.append(ex_str)
				examined.append(ex_str_1)
				examined.append(ex_str_2)
				examined.append(ex_str_3)
				examined.append(ex_str_4)
				examined.append(ex_str_5)
				lib1_triple.append(ex_str)
lib2 = ['456','473','475','476','485','486']
lib2_double = []
examined = []
for ind1 in lib2:
	for ind2 in lib2:
		ex_str = ind1+'_'+ind2
		ex_str_1 = ind2+'_'+ind1
		if ind2 != ind1 and ex_str not in examined and ex_str_1 not in examined:
			examined.append(ex_str)
			examined.append(ex_str_1)
			lib2_double.append(ex_str)
lib2_triple = []
examined = []
for ind1 in lib2:
	for ind2 in lib2:
		for ind3 in lib2:
			ex_str = ind1+'_'+ind2+'_'+ind3
			ex_str_1 = ind1+'_'+ind3+'_'+ind2
			ex_str_2 = ind2+'_'+ind1+'_'+ind3
			ex_str_3 = ind2+'_'+ind3+'_'+ind1
			ex_str_4 = ind3+'_'+ind1+'_'+ind2
			ex_str_5 = ind3+'_'+ind2+'_'+ind1
			if ind2 != ind1 and ind3 != ind2 and ind3 != ind1 and ex_str not in examined and ex_str_1 not in examined and ex_str_2 not in examined and ex_str_3 not in examined and ex_str_4 not in examined and ex_str_5 not in examined:
				examined.append(ex_str)
				examined.append(ex_str_1)
				examined.append(ex_str_2)
				examined.append(ex_str_3)
				examined.append(ex_str_4)
				examined.append(ex_str_5)
				lib2_triple.append(ex_str)

print(lib1)
print(lib1_double)
print(lib1_triple)
print(lib2)
print(lib2_double)
print(lib2_triple)

md_456 = {'456': ['F', 'S', 'I', 'T'], '473': ['Y', 'F', 'H', 'V', 'L', 'D'], '475': ['A', 'D', 'V', 'L', 'H', 'P'], '476': ['G', 'L', 'S', 'W', 'A', 'V'], '485': ['G', 'A', 'V', 'S', 'L', 'W'], '486': ['F', 'I', 'S', 'T']}
md_505 = {'498': ['R', 'Q', 'V', 'L', 'G', 'E'], '499': ['P', 'F', 'Y', 'S', 'L', 'H'], '500': ['T', 'W', 'S', 'R', 'M', 'L'], '501': ['Y', 'T', 'S', 'N', 'H', 'P'], '505': ['H', 'Y', 'F', 'L', 'D', 'V']}
## 456 ternary
df = pd.read_csv('weights_456.csv')
coef = []
val = []
for a, b in zip(df['feature'], df['weight']):
	if a < cutoff_1_456:
		# single body
		if b != 0:
			cur_ind = lib2[math.floor(a / 20)]
			cur_aa = aa_list[a%20]
			coef.append('beta_'+cur_ind+'_'+cur_aa)
			val.append(b)
	elif a < cutoff_2_456:
		# double body
		if (a-cutoff_1_456) % 400 == 0:
			cur_base = a
		if (a-cutoff_1_456) % 20 == 0:
			cur_base_2 = a
		if b != 0:
			ind_entry = lib2_double[int((cur_base-cutoff_1_456)/400)]
			cur_ind_1 = ind_entry.split('_')[0]
			cur_ind_2 = ind_entry.split('_')[1]
			cur_aa_1 = aa_list[math.floor((a-cur_base-1)/20)]
			cur_aa_2 = aa_list[a-cur_base_2-1]
			if cur_aa_1 not in md_456[cur_ind_1]:
				print('error')
				print(cur_aa_1 +' NOT IN '+cur_ind_1)
				print(a)
			if cur_aa_2 not in md_456[cur_ind_2]:
				print('error')
				print(cur_aa_2 +' NOT IN '+cur_ind_2)
				print(a)
			coef.append('beta_'+cur_ind_1+'_'+cur_aa_1+'_'+cur_ind_2+'_'+cur_aa_2)
			val.append(b)
	else:
		# triple body
		if (a-cutoff_2_456) % 8000 == 0:
			cur_base = a
		if (a-cutoff_2_456) % 400 == 0:
			cur_base_2 = a
		if (a-cutoff_2_456) % 20 == 0:
			cur_base_3 = a
		if b != 0:
			ind_entry = lib2_triple[int((cur_base-cutoff_2_456)/8000)]
			cur_ind_1 = ind_entry.split('_')[0]
			cur_ind_2 = ind_entry.split('_')[1]
			cur_ind_3 = ind_entry.split('_')[2]
			cur_aa_1 = aa_list[math.floor((a-cur_base-1)/400)]
			cur_aa_2 = aa_list[math.floor((a-cur_base_2-1)/20)]
			cur_aa_3 = aa_list[a-cur_base_3-1]
			if cur_aa_1 not in md_456[cur_ind_1]:
				print('error')
				print(cur_aa_1 +' NOT IN '+cur_ind_1)
				print(a)
			if cur_aa_2 not in md_456[cur_ind_2]:
				print('error')
				print(cur_aa_2 +' NOT IN '+cur_ind_2)
				print(a)
			if cur_aa_3 not in md_456[cur_ind_3]:
				print('error')
				print(cur_aa_3 +' NOT IN '+cur_ind_3)
				print(a)
			coef.append('beta_'+cur_ind_1+'_'+cur_aa_1+'_'+cur_ind_2+'_'+cur_aa_2+'_'+cur_ind_3+'_'+cur_aa_3)
			val.append(b)
df_final = pd.DataFrame(zip(coef, val), columns=['betas','values'])
df_final.to_csv('456_betas.csv',index=False)

## 505 ternary
df = pd.read_csv('weights_505.csv')
coef = []
val = []
for a, b in zip(df['feature'], df['weight']):
	if a < cutoff_1_505:
		# single body
		if b != 0:
			cur_ind = lib1[math.floor(a / 20)]
			cur_aa = aa_list[a%20]
			coef.append('beta_'+cur_ind+'_'+cur_aa)
			val.append(b)
	elif a < cutoff_2_505:
		# double body
		if (a-cutoff_1_505) % 400 == 0:
			cur_base = a
		if (a-cutoff_1_505) % 20 == 0:
			cur_base_2 = a
		if b != 0:
			ind_entry = lib1_double[int((cur_base-cutoff_1_505)/400)]
			cur_ind_1 = ind_entry.split('_')[0]
			cur_ind_2 = ind_entry.split('_')[1]
			cur_aa_1 = aa_list[math.floor((a-cur_base-1)/20)]
			cur_aa_2 = aa_list[a-cur_base_2-1]
			cur_name = 'beta_'+cur_ind_1+'_'+cur_aa_1+'_'+cur_ind_2+'_'+cur_aa_2
			if cur_aa_1 not in md_505[cur_ind_1]:
				print('error')
				print(cur_aa_1 +' NOT IN '+cur_ind_1)
				print(a)
				print(cur_name)
			if cur_aa_2 not in md_505[cur_ind_2]:
				print('error')
				print(cur_aa_2 +' NOT IN '+cur_ind_2)
				print(a)
				print(cur_name)
			coef.append(cur_name)
			val.append(b)
	else:
		# triple body
		if (a-cutoff_2_505) % 8000 == 0:
			cur_base = a
		if (a-cutoff_2_505) % 400 == 0:
			cur_base_2 = a
		if (a-cutoff_2_505) % 20 == 0:
			cur_base_3 = a
		if b != 0:
			ind_entry = lib1_triple[int((cur_base-cutoff_2_505)/8000)]
			cur_ind_1 = ind_entry.split('_')[0]
			cur_ind_2 = ind_entry.split('_')[1]
			cur_ind_3 = ind_entry.split('_')[2]
			cur_aa_1 = aa_list[math.floor((a-cur_base-1)/400)]
			cur_aa_2 = aa_list[math.floor((a-cur_base_2-1)/20)]
			cur_aa_3 = aa_list[a-cur_base_3-1]
			cur_name = 'beta_'+cur_ind_1+'_'+cur_aa_1+'_'+cur_ind_2+'_'+cur_aa_2+'_'+cur_ind_3+'_'+cur_aa_3
			if cur_aa_1 not in md_505[cur_ind_1]:
				print('error')
				print(cur_aa_1 +' NOT IN '+cur_ind_1)
				print(a)
				print(cur_name)
			if cur_aa_2 not in md_505[cur_ind_2]:
				print('error')
				print(cur_aa_2 +' NOT IN '+cur_ind_2)
				print(a)
				print(cur_name)
			if cur_aa_3 not in md_505[cur_ind_3]:
				print('error')
				print(cur_aa_3 +' NOT IN '+cur_ind_3)
				print(a)
				print(cur_name)
			coef.append(cur_name)
			val.append(b)
df_final = pd.DataFrame(zip(coef, val), columns=['betas','values'])
df_final.to_csv('505_betas.csv',index=False)
