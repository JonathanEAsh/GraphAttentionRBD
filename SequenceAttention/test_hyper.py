import subprocess
import os
import time
lr_list=[0.0001, 0.00001, 0.000001]
bs_list=[50, 100, 200]
epoch_list=[200]
head_list=[1, 2, 4, 8, 16]
att_drop_list=[0.1, 0.0]
ccb_ctr=2
sdk_ctr=2
for a in bs_list:
	for b in lr_list:
		for c in epoch_list:
			for d in head_list:
				for e in att_drop_list:
					model_name = 'model_'+str(c)+'_'+str(a)+'_'+str(b)+'_'+str(d)+'_'+str(e)+'.txt'
					if not os.path.isfile('quick_model_out/'+model_name):
						print('NOT COMPLETE: '+model_name)
						process1=subprocess.Popen(['squeue','-u','ja961'], stdout=subprocess.PIPE)
						process2=subprocess.check_output(['wc','-l'], stdin=process1.stdout)
						while int(process2)>=200:
							print('SLEEP')
							time.sleep(300)
							process1=subprocess.Popen(['squeue','-u','ja961'], stdout=subprocess.PIPE)
							process2=subprocess.check_output(['wc','-l'], stdin=process1.stdout)
						if ccb_ctr < 2:
							subprocess.call(['sbatch','--time=3-00:00:00','--mem=64G','--partition=p_ccb_1','--gres=gpu:1','--requeue','./submit_new_model.sh',str(a),str(b),str(c),str(d),str(e)])
							ccb_ctr+=1
						elif sdk_ctr < 2:
							subprocess.call(['sbatch','--time=3-00:00:00','--mem=64G','--partition=p_sdk94_1','--gres=gpu:1','--requeue','./submit_new_model.sh',str(a),str(b),str(c),str(d),str(e)])
							ccb_ctr+=1
						else:
							subprocess.call(['sbatch','--requeue','--time=3-00:00:00','--mem=64G','--partition=gpu','--gres=gpu:1','./submit_new_model.sh',str(a),str(b),str(c),str(d),str(e)])
					else:
						print('COMPLETED: '+model_name)