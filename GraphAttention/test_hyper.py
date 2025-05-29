import subprocess
import time
import os.path
import pandas as pd
drop_list=[0.0, 0.1, 0.2]
layer_list=[6,7,8,9]
lr_list=[0.01, 0.001, 0.0001, 0.00001]
bs_list=[50,100,200,300]
epoch_list=[200]
channel_list=[80,160,320]
print('Total Combinations: '+str(len(drop_list)*len(lr_list)*len(bs_list)*len(epoch_list)))
# drop_list=[0.1]
# layer_list=[2]
# lr_list=[0.005]
# bs_list=[100]
# epoch_list=[5]
com_ctr=0
incom_ctr=0
tot=0
ccb=0
sdk=0
# df=pd.read_csv('quick_cv_4mer_three_layers_2head_bi.csv')
# complete_list=list(df['names'])
for i in drop_list:
	#for j in layer_list:
	for k in lr_list:
		for l in bs_list:
			for m in epoch_list:
				#for n in channel_list:
				# incomplete=1
				file_name='model_'+str(i)+'_layers_3_epochs_25_lr_'+str(k)+'_bs_'+str(l)
				check = 0
				for files in os.listdir('binary/model_out/'):
					if file_name in files:
						check = 1

				if not check:
					print('NOT FOUND: '+file_name)
				#if file_name not in complete_list:
					incom_ctr+=1
					process1=subprocess.Popen(['squeue','-u','ja961'], stdout=subprocess.PIPE)
					process2=subprocess.Popen(['grep','gpu'], stdin=process1.stdout, stdout=subprocess.PIPE)
					process3=subprocess.check_output(['wc','-l'], stdin=process2.stdout)
					#print(int(process3))
					while int(process3)>=199:
						print('SLEEP')
						time.sleep(300)
						process1=subprocess.Popen(['squeue','-u','ja961'], stdout=subprocess.PIPE)
						process2=subprocess.Popen(['grep','gpu'], stdin=process1.stdout, stdout=subprocess.PIPE)
						process3=subprocess.check_output(['wc','-l'], stdin=process2.stdout)
					if incom_ctr % 2 == 0:
						time.sleep(15)
					# if incom_ctr%2==0:
					# 	subprocess.call(['sbatch','--time=1:00:00','--mem=32G','--partition=p_sdk94_1','--gres=gpu:1','--requeue','./run_model.sh',str(i),str(j),str(m),str(k),str(l)])
					# else:
					# 	subprocess.call(['sbatch','--time=1:00:00','--mem=32G','--partition=p_ccb_1','--gres=gpu:1','--requeue','./run_model.sh',str(i),str(j),str(m),str(k),str(l)])
					if sdk < 2:
						subprocess.call(['sbatch','--time=72:00:00','--mem=32G','--partition=p_sdk94_1','--gres=gpu:1','--requeue','./run_model.sh',str(i),str(m),str(k),str(l)])
						sdk+=1
					elif ccb < 2:
						subprocess.call(['sbatch','--time=72:00:00','--mem=32G','--partition=p_ccb_1','--gres=gpu:1','--requeue','./run_model.sh',str(i),str(m),str(k),str(l)])
						ccb+=1
					else:
						subprocess.call(['sbatch','--time=72:00:00','--mem=32G','--partition=gpu','--gres=gpu:1','--requeue','./run_model.sh',str(i),str(m),str(k),str(l)])

				else:
					print('FOUND: '+file_name)
					com_ctr+=1
					# with open(file_name) as f:
					# 	x=f.readlines()
					# 	if len(x)>m*5:
					# 		incomplete=0
					# 		com_ctr+=1
				# if incomplete:
					
				tot+=1
print(com_ctr)
print(incom_ctr)
print(tot)
