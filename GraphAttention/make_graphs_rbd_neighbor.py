import os
import os.path as osp
import torch
from torch_geometric.data import Data, Dataset
import torch_geometric.transforms as T
import pandas as pd
import esm
from torch_geometric.utils import to_networkx, to_undirected, degree
import networkx as nx
import matplotlib.pyplot as plt
import math
from pyrosetta import *
init()
from joey_utils import chain_selector, neighbor_selector, selector_union, interaction_energy, index_selector, get_distance, find_atom_coords, index_selector
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default='esm_stats_bind.csv')
args = parser.parse_args()
def visualize_graph(G, color):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=True, node_color=color, cmap="Set1")
    #nx.draw(G)
    plt.show()

class RBDDataset(Dataset):
	def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
		super().__init__(root, transform, pre_transform, pre_filter)
	
	@property
	def raw_file_names(self):
		return ['file.pt']

	@property
	def processed_file_names(self):
		return ['file.pt']
    
	def process(self):
		idx = 0
		#for raw_path in self.raw_paths:
		for files in os.listdir('sample_graphs/raw/'):
			# Read data from `raw_path`.
			if osp.isfile(osp.join(self.processed_dir, f'data_{idx}.pt'))==False:
				data = torch.load('sample_graphs/raw/'+files)
				if self.pre_filter is not None and not self.pre_filter(data):
					continue
				if self.pre_transform is not None:
					data = self.pre_transform(data)
				torch.save([data,files], osp.join(self.processed_dir, f'data_{idx}.pt'))
			idx += 1

	def len(self):
		ctr=0
		for files in os.listdir('sample_graphs/raw/'):
			ctr+=1
		return ctr

	def get(self, idx):
		# for files in os.listdir('sample_graphs/processed/'):
		# 	cur_ind = int(files.split('_')[-1].replace('.pt',''))
		# 	if cur_ind == int(idx):
		# 		data = torch.load(osp.join(self.processed_dir, files))
		# 		break
		data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
		return data

def make_graph(name, esm, bind, x_ind, edge_index): #need esm embeddings, y (bind), x index, edge index - fill in node features and edge features based off of indices
	if '[' in name:
		cur_pose = pose_from_pdb('../decoys_r2/'+name+'.pdb')
	else:
		cur_pose = pose_from_pdb('../decoys_r1/'+name+'.pdb')
	cur_esm = [float(i) for i in esm.split(',')]
	edge_attr = []
	sfxn(cur_pose)
	for a in edge_index:
		cur_edge_pos_ind = [md[a[0]], md[a[1]]]
		coord1 = find_atom_coords(cur_pose, cur_edge_pos_ind[0])
		coord2 = find_atom_coords(cur_pose, cur_edge_pos_ind[1])
		dist = get_distance(coord1, coord2)
		seq_dist = abs(cur_edge_pos_ind[0] - cur_edge_pos_ind[1])
		# sel1 = index_selector(cur_edge_pos_ind[0])
		# sel2 = index_selector(cur_edge_pos_ind[1])
		# inte = interaction_energy(cur_pose, sfxn, sel1, sel2)	
		#edge_attr.append([seq_dist, inte, dist])
		edge_attr.append([seq_dist, dist])
	design_seq = cur_pose.sequence()
	x = []
	for a in x_ind:
		cur_pos_ind = md[a]

		temp = []
		if cur_pos_ind < len(ACE2_seq):
			temp.append(0)
		else:
			temp.append(1)
		# cur_aa = design_seq[cur_pos_ind - 1] #0-index
		# for b in aa_list:
		# 	if b == cur_aa:
		# 		temp.append(1)
		# 	else:
		# 		temp.append(0)
		temp.append(cur_esm[cur_pos_ind - 1])
		x.append(temp)
	x=torch.tensor(x, dtype=torch.float)
	edge_index=torch.tensor(edge_index, dtype=torch.long)
	edge_attr=torch.tensor(edge_attr, dtype=torch.float)
	print(x.shape)
	print(edge_index.shape)
	print(edge_attr.shape)
	data=Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr, y=torch.tensor([int(bind)]))
	return data
ACE2_seq="STIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFLKEQSTLAQMYPLQEIQNLTVKLQLQALQQNGSSVLSEDKSKRLNTILNTMSTIYSTGKVCNP"\
"DNPQECLLLEPGLNEIMANSLDYNERLWAWESWRSEVGKQLRPLYEEYVVLKNEMARANHYEDYGDYWRGDYEVNGVDGYDYSRGQLIEDVEHTFEEIKPLYEHLHAYVRAKLMNAYPSYISPIG"\
"CLPAHLLGDMWGRFWTNLYSLTVPFGQKPNIDVTDAMVDQAWDAQRIFKEAEKFFVSVGLPNMTQGFWENSMLTDPGNVQKAVCHPTAWDLGKGDFRILMCTKVTMDDFLTAHHEMGHIQYDMAY"\
"AAQPFLLRNGANEGFHEAVGEIMSLSAATPKHLKSIGLLSPDFQEDNETEINFLLKQALTIVGTLPFTYMLEKWRWMVFKGEIPKDQWMKKWWEMKREIVGVVEPVPHDETYCDPASLFHVSNDYSF"\
"IRYYTRTLYQFQFQEALCQAAKHEGPLHKCDISNSTEAGQKLFNMLRLGKSEPWTLALENVVGAKNMNVRPLLNYFEPLFTWLKDQNKNSFVGWSTDWSPYAD"
aa_list = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
pose = pose_from_pdb('../../struct_relaxed/wuhan.pdb')
mut_indices = [682, 720, 721, 742, 749, 763, 766, 770, 702, 703, 704, 705, 707, 709, 710, \
714, 715, 717, 718, 722, 723, 724, 725, 727, 733, 729, 736, 731, 706, 713, 730, 740, 741, \
750, 765, 764, 751, 738, 711, 743, 758, 761] # Starts from 1 (pose indexing)
edge_index_ogind = [] # pose index
x_ogind = [] # pose index
for i, a in enumerate(mut_indices):
	indsel = index_selector(a)
	neighbors = neighbor_selector(indsel)
	neighbors_bool = list(neighbors.apply(pose))
	cur_neighborhood = [i + 1 for i, b in enumerate(neighbors_bool) if b == 1]
	print(a)
	print(cur_neighborhood)
	for c in cur_neighborhood:
		edge_index_ogind.append([a,c])
		if [c,a] not in edge_index_ogind:
			edge_index_ogind.append([c,a])
		if c not in x_ogind:
			x_ogind.append(c)
	if a not in x_ogind:
		x_ogind.append(a)
print(x_ogind)
print(edge_index_ogind)
md={} # map graph index to pose index
md_i = {} # map pose index to graph index
x_ind = [] # graph index
edge_index = [] # graph index
x_ogind.sort()
print('$$$$$$$$$$$$$$$$$$$$$$$$$')
print(x_ogind)
print(edge_index_ogind)
for i, a in enumerate(x_ogind):
	md[i]=a
	md_i[a]=i
	x_ind.append(i)
for a in edge_index_ogind:
	temp = [md_i[a[0]], md_i[a[1]]]
	edge_index.append(temp)
print(x_ind)
print(edge_index)
print('$$$$$$$$$$$$$$$$$$$$$$$$$')
for a in x_ind:
	found = 0
	for b in edge_index:
		if a in b:
			found = 1
			break
	if found == 0:
		print(a)

print(len(x_ogind))
print(len(x_ind))
print(len(md.keys()))
print(len(md_i.keys()))
# uhoh=[]
# for a in edge_index:
# 	reverse = [a[1], a[0]]
# 	if reverse not in edge_index:
# 		print('not undirected')
# 		uhoh.append(a)




# print(int_ind)
# print(cha_int)
# print(chb_int)
# sa = ''
# for a in cha_int:
# 	sa += str(a)+'+'
# print(sa[:-1])
# sb = ''
# for a in chb_int:
# 	sb += str(a)+'+'
# print(sb[:-1])
# sfxn = get_fa_scorefxn()
# print(len(all_int))
# print(len(cha_int))
# print(len(chb_int))
# print(len(chb_int)*len(cha_int))
# # print((4*33)+(30*4))
class_map = {0:1, 1:0, 2:0}
sfxn = get_fa_scorefxn()
df = pd.read_csv(args.input_path)
print(df.shape)
for a, b, c in zip(df['name'],df['avgs'],df['bind']):
	cur_graph = make_graph(a, b, class_map[c], x_ind, edge_index)
	if '[' in a:
		torch.save(cur_graph, 'raw_graphs_binary/r2/'+a+'.pt')
	else:
		torch.save(cur_graph, 'raw_graphs_binary/r1/'+a+'.pt')
	# G = to_networkx(cur_graph)
	# print(list(G.nodes))
	# print(list(G.edges))
	#visualize_graph(G, color=cur_graph.x)
	# set_sample=RBDDataset(root='sample_graphs/')
	# set_sample.process()
	# print(set_sample.len())
	# data=set_sample.get(0)[0]
	# print(cur_graph.validate(raise_on_error=True))
	# print(cur_graph.num_nodes)
	# print(cur_graph.num_edges)
	# print(cur_graph.num_node_features)
	# print(cur_graph.has_isolated_nodes())
	# print(cur_graph.has_self_loops())
	# print(cur_graph.is_directed())
	# print(cur_graph['x'])
	# print(cur_graph['y'])
	# print(cur_graph['edge_index'])
	# print(cur_graph)
	# deg=degree(cur_graph.edge_index[0],cur_graph.num_nodes)
	# print(deg)
	# print(len(deg))
	# print(torch.sum(deg))
	#print(uhoh)
	# break