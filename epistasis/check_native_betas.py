import pandas as pd
import matplotlib.pyplot as plt
print('================= 456 =================')

md_456 = {'456': 'F', '473': 'Y', '475': 'A', '476': 'G', '485': 'G', '486': 'F'}
nat_456 = []
ex1 = []
ex2 = []
for keys, values in md_456.items():
	nat_456.append('beta_'+keys+'_'+values)
	
	for keys2, values2 in md_456.items():
		if keys2 != keys:
			ind1 = keys2+'_'+values2+'_'+keys+'_'+values
			ind2 = keys+'_'+values+'_'+keys2+'_'+values2
			# print(ind1)
			# print(ind2)
			if ind1 not in ex1 and ind2 not in ex1:
				ex1.append(ind1)
				ex1.append(ind2)
				nat_456.append('beta_'+ind2)
				
	for keys2, values2 in md_456.items():
		for keys3, values3 in md_456.items():
			if keys3 != keys2 and keys3 != keys and keys2 != keys:
				ind1 = keys+'_'+values+'_'+keys2+'_'+values2+'_'+keys3+'_'+values3
				ind2 = keys+'_'+values+'_'+keys3+'_'+values3+'_'+keys2+'_'+values2
				ind3 = keys2+'_'+values2+'_'+keys+'_'+values+'_'+keys3+'_'+values3
				ind4 = keys2+'_'+values2+'_'+keys3+'_'+values3+'_'+keys+'_'+values
				ind5 = keys3+'_'+values3+'_'+keys2+'_'+values2+'_'+keys+'_'+values
				ind6 = keys3+'_'+values3+'_'+keys+'_'+values+'_'+keys2+'_'+values2
				if ind1 not in ex2 and ind2 not in ex2 and ind3 not in ex2 and ind4 not in ex2 and ind5 not in ex2 and ind6 not in ex2:
					ex2.append(ind1)
					ex2.append(ind2)
					ex2.append(ind3)
					ex2.append(ind4)
					ex2.append(ind5)
					ex2.append(ind6)
					nat_456.append('beta_'+ind1)
print(nat_456)
nr = []
for a in nat_456:
	if a not in nr:
		nr.append(a)
print(len(nat_456))
print(len(nr))

df = pd.read_csv('456_betas_norm_epistasis.csv')
d = list(df['name'])
for a in nat_456:
	if a not in d:
		print(a)
vals_nat = []
vals_mut = []
for a, b in zip(df['name'], df['difference']):
	if a in nat_456:
		vals_nat.append(b)
	else:
		vals_mut.append(b)
plt.hist(vals_nat, bins=100, label='native', color='blue', alpha=0.5)
plt.hist(vals_mut, bins=100, label='mutant', color='red', alpha=0.5)
plt.title('Library 456 Native vs Mutant Epistasis Effects')
plt.xlabel('Epistasis Effect')
plt.ylabel('Beta Counts')
plt.legend()
plt.savefig('456_epistasis_by_native.png')
plt.show()
print('================= 505 =================')
md_505 = {'498': 'R', '499': 'P', '500': 'T', '501': 'Y', '505': 'H'}
nat_505 = []
ex1 = []
ex2 = []
for keys, values in md_505.items():
	nat_505.append('beta_'+keys+'_'+values)
	
	for keys2, values2 in md_505.items():
		if keys2 != keys:
			ind1 = keys2+'_'+values2+'_'+keys+'_'+values
			ind2 = keys+'_'+values+'_'+keys2+'_'+values2
			# print(ind1)
			# print(ind2)
			if ind1 not in ex1 and ind2 not in ex1:
				ex1.append(ind1)
				ex1.append(ind2)
				nat_505.append('beta_'+ind2)
				
	for keys2, values2 in md_505.items():
		for keys3, values3 in md_505.items():
			if keys3 != keys2 and keys3 != keys and keys2 != keys:
				ind1 = keys+'_'+values+'_'+keys2+'_'+values2+'_'+keys3+'_'+values3
				ind2 = keys+'_'+values+'_'+keys3+'_'+values3+'_'+keys2+'_'+values2
				ind3 = keys2+'_'+values2+'_'+keys+'_'+values+'_'+keys3+'_'+values3
				ind4 = keys2+'_'+values2+'_'+keys3+'_'+values3+'_'+keys+'_'+values
				ind5 = keys3+'_'+values3+'_'+keys2+'_'+values2+'_'+keys+'_'+values
				ind6 = keys3+'_'+values3+'_'+keys+'_'+values+'_'+keys2+'_'+values2
				if ind1 not in ex2 and ind2 not in ex2 and ind3 not in ex2 and ind4 not in ex2 and ind5 not in ex2 and ind6 not in ex2:
					ex2.append(ind1)
					ex2.append(ind2)
					ex2.append(ind3)
					ex2.append(ind4)
					ex2.append(ind5)
					ex2.append(ind6)
					nat_505.append('beta_'+ind1)
print(nat_505)
nr = []
for a in nat_505:
	if a not in nr:
		nr.append(a)
print(len(nat_505))
print(len(nr))

df = pd.read_csv('505_betas_norm_epistasis.csv')
d = list(df['name'])
for a in nat_505:
	if a not in d:
		print(a)
vals_nat = []
vals_mut = []
for a, b in zip(df['name'], df['difference']):
	if a in nat_505:
		vals_nat.append(b)
	else:
		vals_mut.append(b)
plt.hist(vals_nat, bins=100, label='native', color='blue', alpha=0.5)
plt.hist(vals_mut, bins=100, label='mutant', color='red', alpha=0.5)
plt.title('Library 505 Native vs Mutant Epistasis Effects')
plt.xlabel('Epistasis Effect')
plt.ylabel('Beta Counts')
plt.legend()
plt.savefig('505_epistasis_by_native.png')
plt.show()