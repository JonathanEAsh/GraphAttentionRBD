from pyrosetta import *
init()
from joey_utils import find_atom_coords, get_distance, interaction_energy, interaction_energy_split, index_selector, set_sf_weight, sasa_metric, total_energy
DSSP = pyrosetta.rosetta.protocols.moves.DsspMover()

import pandas as pd
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-pdb", "--input_path", help="Input path", type=str)
args = parser.parse_args()
input_path=str(args.input_path)
sfxn=get_fa_scorefxn()
def get_distance_matrix(pose):
	matrix=[]
	for i in range(len(pose.chain_sequence(1))):
		temp=[]
		ref_coord=find_atom_coords(pose, i+1)
		for j in range(len(pose.chain_sequence(2))):
			target_coord=find_atom_coords(pose, j+1+len(pose.chain_sequence(1)))
			temp.append(round(get_distance(ref_coord, target_coord),3))
		matrix.append(temp)
	return matrix
def get_energy_matrix(pose):
	matrix=[]
	sfxn(pose)
	for i in range(len(pose.chain_sequence(1))):
		temp=[]
		ref_sel=index_selector(i+1)
		for j in range(len(pose.chain_sequence(2))):
			target_sel=index_selector(j+1+len(pose.chain_sequence(1)))
			temp.append(round(interaction_energy(pose, sfxn, ref_sel, target_sel),3))
		matrix.append(temp)
	return matrix
def get_interface(pose):
	matrix=[]
	for i in range(len(pose.chain_sequence(1))):
		temp=[]
		ref_coord=find_atom_coords(pose, i+1)
		for j in range(len(pose.chain_sequence(2))):
			target_coord=find_atom_coords(pose, j+1+len(pose.chain_sequence(1)))
			temp.append(round(get_distance(ref_coord, target_coord),3))
		matrix.append([i,min(temp)])
	matrix.sort(key=lambda x:x[1])
	interface_str=''
	for i in matrix[:48]:
		interface_str+=str(i[0])+','
	return interface_str[:-1]
def get_interface_and_pep(pose):
	matrix=[]
	for i in range(len(pose.chain_sequence(1))):
		temp=[]
		ref_coord=find_atom_coords(pose, i+1)
		for j in range(4):
			target_coord=find_atom_coords(pose, len(pose.sequence())-j)
			temp.append(round(get_distance(ref_coord, target_coord),3))
		matrix.append([i,min(temp)])
	matrix.sort(key=lambda x:x[1])
	interface_str=''
	for i in matrix[:30]:
		interface_str+=str(i[0])+','
	for i in range(4):
		interface_str+=str(len(pose.sequence())-i-1)+','
	return interface_str[:-1]
def get_knn(pose, interface):
	knn={}
	sfxn(pose)
	for a in interface.split(','):
		temp=[]
		cur_ind=int(a)+1
		ref_coord=find_atom_coords(pose, cur_ind)
		ref_sel=index_selector(cur_ind)
		for b in interface.split(','):
			if b != a:
				target_ind=int(b)+1
				target_coord=find_atom_coords(pose, target_ind)
				target_sel=index_selector(target_ind)
				cur_dist=round(get_distance(ref_coord, target_coord),3)
				cur_e=round(interaction_energy(pose, sfxn, ref_sel, target_sel),3)
				temp.append([b,cur_dist,cur_e])
		temp.sort(key=lambda x:x[1])
		new_temp=[]
		#for i in range(16): #use for knn
		for i in range(len(temp)): #use for complete
			new_temp.append(temp[i])
		knn[a]=new_temp
	return knn
def get_bi(pose, interface):
	knn={}
	metric_list=['fa_atr', 'fa_rep', 'fa_sol', 'fa_elec', 'lk_ball_wtd', 'hbond_sr_bb', 'hbond_lr_bb', 'hbond_bb_sc', 'hbond_sc']
	#partial_metric_list=['lk_ball_wtd','fa_atr','fa_elec']
	sfxn=get_fa_scorefxn()
	sfxn(pose)
	all_indices=[int(x)+1 for x in interface.split(',')]
	pep_indices=[]
	for i in range(4):
		pep_indices.append(max(all_indices)-i)
	for a in interface.split(','):
		temp=[]
		cur_ind=int(a)+1
		ref_coord=find_atom_coords(pose, cur_ind)
		ref_sel=index_selector(cur_ind)
		if cur_ind in pep_indices:
			for b in interface.split(','):
				if b != a:
					target_ind=int(b)+1
					target_coord=find_atom_coords(pose, target_ind)
					target_sel=index_selector(target_ind)
					cur_dist=round(get_distance(ref_coord, target_coord),3)
					cur_e=round(interaction_energy(pose, sfxn, ref_sel, target_sel),3)
					# cur_e=''
					# for i in metric_list:
					# 	#interact_metric = InteractionEnergyMetric()
					# 	for j in metric_list:
					# 		if j == i:
					# 			sfxn=set_sf_weight(sfxn, j, 1)
					# 		else:
					# 			sfxn=set_sf_weight(sfxn, j, 0)
					# 	sfxn(pose)
					# 	cur_e_split=round(interaction_energy(pose, sfxn, ref_sel, target_sel),3)
					# 	cur_e+=str(cur_e_split)+':'
					# temp.append([b,cur_dist,cur_e[:-1]])
					temp.append([b,cur_dist,cur_e])
		else:
			for b in interface.split(','):
				if b != a:
					target_ind=int(b)+1
					if target_ind in pep_indices:
						target_coord=find_atom_coords(pose, target_ind)
						target_sel=index_selector(target_ind)
						cur_dist=round(get_distance(ref_coord, target_coord),3)
						# cur_e=''
						# for i in metric_list:
						# 	#interact_metric = InteractionEnergyMetric()
						# 	for j in metric_list:
						# 		if j == i:
						# 			sfxn=set_sf_weight(sfxn, j, 1)
						# 		else:
						# 			sfxn=set_sf_weight(sfxn, j, 0)
						# 	sfxn(pose)
						# 	cur_e_split=round(interaction_energy(pose, sfxn, ref_sel, target_sel),3)
						# 	cur_e+=str(cur_e_split)+':'
						cur_e=round(interaction_energy(pose, sfxn, ref_sel, target_sel),3)
						# temp.append([b,cur_dist,cur_e[:-1]])
						temp.append([b,cur_dist,cur_e])
		temp.sort(key=lambda x:x[1])
		new_temp=[]
		#for i in range(16): #use for knn
		for i in range(len(temp)): #use for complete
			new_temp.append(temp[i])
		knn[a]=new_temp
	return knn
def calc_totale_sasa_ss(pose, interface):
	sfxn(pose)
	score_str=''
	sasa_str=''
	for i in interface.split(','):
		cur_ind=index_selector(int(i)+1)
		score_str+=str(round(total_energy(pose, sfxn, cur_ind),3))+','
		sasa_str+=str(round(sasa_metric(pose, cur_ind),3))+','
	DSSP.apply(pose)
	sec=pose.secstruct()
	return score_str[:-1], sasa_str[:-1], sec
	#return sasa_str[:-1], sec
df2=pd.read_csv('msa/conserved_indices_full.csv')
ctr=0
names=[]
dist=[]
ie=[]
interface=[]
connect=[]
seqs=[]
tote=[]
sasa=[]
ss=[]
#pers=[]
for files in os.listdir('../GOLD_decoys_split/GOLD_'+input_path):
	pose=pose_from_pdb('../GOLD_decoys_split/GOLD_'+input_path+'/'+files)
	# dist_mat=get_distance_matrix(pose)
	# inte_mat=get_energy_matrix(pose)
	#if_str=get_interface(pose)
	full_if_str=get_interface_and_pep(pose)
	knn_dict=get_bi(pose, full_if_str)
	node_str=''
	connect_str=''
	dist_str=''
	e_str=''
	for i, j in knn_dict.items():
		node_str+=i+','
		for a in j:
			connect_str+=a[0]+','
			dist_str+=str(a[1])+','
			e_str+=str(a[2])+','
		connect_str=connect_str[:-1]+';'
		dist_str=dist_str[:-1]+';'
		e_str=e_str[:-1]+';'
	interface.append(node_str[:-1])
	dist.append(dist_str[:-1])
	connect.append(connect_str[:-1])
	ie.append(e_str[:-1])
	
	tote_str, sa_str, ss_str = calc_totale_sasa_ss(pose, full_if_str)
	#sa_str, ss_str = calc_totale_sasa_ss(pose, full_if_str)
	tote.append(tote_str)
	sasa.append(sa_str)
	ss.append(ss_str)
	# print(len(pose.chain_sequence(1)))
	# print('$$$$$')
	# print(len(dist_mat))
	# print('$$$$$')
	# print(len(inte_mat))
	# print('$$$$$')
	# dist_str=''
	# for i in dist_mat:
	# 	sub_str=''
	# 	for j in i:
	# 		sub_str+=str(j)+','
	# 	sub_str=sub_str[:-1]+';'
	# 	dist_str+=sub_str
	# dist_str=dist_str[:-1]
	# inte_str=''
	# for i in inte_mat:
	# 	sub_str=''
	# 	for j in i:
	# 		sub_str+=str(j)+','
	# 	sub_str=sub_str[:-1]+';'
	# 	inte_str+=sub_str
	# inte_str=inte_str[:-1]
	names.append(files.split('.')[0])
	# dist.append(dist_str)
	# ie.append(inte_str)
	# interface.append(if_str)
	seqs.append(pose.sequence())
	# print(if_str)
	# print(if_str.count(',')+1)
	# indices_to_check=[int(x) for x in if_str.split(',')]
	# per_str=''
	# base=files.split('_')[0]+'_'+files.split('_')[1]
	# for a, b in zip(df2['name'],df2['percents']):
	# 	if a == base:
	# 		per_list=b.split(',')
	# 		print(len(per_list))
	# 		for c in indices_to_check:
	# 			per_str+=per_list[c]+','
	# 		break
	# pers.append(per_str[:-1])
	# print(per_str[:-1].count(',')+1)
	ctr+=1
	print('$$$$$$')
	print(ctr)
	print('$$$$$$')
	# if ctr == 5:
	#	break
print(len(names))
print(len(interface))
print(len(dist))
print(len(connect))
print(len(ie))
print(len(seqs))
df=pd.DataFrame(zip(names, interface, connect, dist, ie, seqs, tote, sasa, ss), columns=['name','interface','connect','dist','ie','seq','total_e','sasa','ss'])
df.to_csv('matrices_bi_4mer_30_tote_sasa_ss/dist_ie_matrix_'+input_path+'.csv', index=False)