import subprocess
import os
for files in os.listdir('r1_samples_split/'):
	subprocess.call(['sbatch','--time=3:00:00','--mem=64G','./submit_graph.sh','r1_samples_split/'+files])
for files in os.listdir('r2_samples_split/'):
	subprocess.call(['sbatch','--time=3:00:00','--mem=64G','./submit_graph.sh','r2_samples_split/'+files])
