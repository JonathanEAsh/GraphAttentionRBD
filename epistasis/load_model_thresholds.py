import pickle
import pandas as pd
import statistics
files = ['model_456.pkl','model_505.pkl']
en = ['beta0_456','beta0_505']
val = []
for a in files:
	with open(a,'rb') as f:
		model = pickle.load(f)
		l = model.intercept_
		#val.append(statistics.mean(l))
		val.append(l[0])
		# val_worse.append(l[1])
		# val_like.append(l[2])
df = pd.DataFrame(zip(files,en,val), columns=['file','cutoff','value'])
df.to_csv('intercepts.csv',index=False)