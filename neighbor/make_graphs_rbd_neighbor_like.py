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
from matplotlib.patches import Circle
import math
import statistics
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--input_path", type=str, default='esm_stats_bind.csv')
# args = parser.parse_args()
def visualize_graph(G, color):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, node_color=color, with_labels=False)
    #nx.draw(G)
    plt.savefig('graph.png')
    plt.show()

def calc_hamming(seq1, seq2):
	mutctr = 0
	for i, j in zip(seq1[(436-333):(509-333)], seq2[(436-333):(509-333)]):
		if i != j:
			mutctr+=1
	return mutctr

def count_neighbors(target, seqs):
	neighbors = []
	for a in seqs:
		cur_seq = a[1]
		dist = calc_hamming(target, cur_seq)
		if dist == 1:
			neighbors.append(a)
	return neighbors, len(neighbors)


print('here')
wt = 'TNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGNIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGNTPCNGVAGFNCYFPLQSYGFRPTYGVGHQPYRVVVLSFELLHAPATVCG'
lib_505 = ['498','499','500','501','505']
lib_456 = ['456','473','475','476','485','486']
seqs_505 = [['wt',wt,0]]
seqs_456 = [['wt',wt,0]]
aa_list = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
df = pd.read_csv('round_1_results_combined.csv')
for a, b, c in zip(df['name'], df['seq'], df['class']):
	if '(' in a:
		check_456 = 0
		check_505 = 0
		muts = a[1:-1].split(',')
		for z in muts:
			if len(z)>0:
				ind = z[1:-1]
				if ind in lib_505:
					check_505 = 1
				elif ind in lib_456:
					check_456 = 1
		if check_505:
			seqs_505.append([a,b,c])
		elif check_456:
			seqs_456.append([a,b,c])
	# if len(seqs_456) == 1000:
	# 	break
# 505 Graph
print(len(seqs_505))
print(len(seqs_456))
x = []
y = []
edge_index = []
for index, a in enumerate(seqs_505):
	x.append(index)
	y.append(a[2])
#for index, a in enumerate(seqs_505):
	cur_seq = a[1]
	if index % 100 == 0:
		print('====')
		print(index)
		print('====')
	for index_2, b in enumerate(seqs_505):
		target_seq = b[1]
		if cur_seq != target_seq:
			cur_dist = calc_hamming(cur_seq, target_seq)
			if cur_dist == 1:
				edge_index.append([index, index_2])

cur_graph = Data(x=torch.tensor(x), edge_index=torch.tensor(edge_index).t().contiguous(), y=torch.tensor(y))
# transform = T.RemoveIsolatedNodes()
# cur_graph = transform(cur_graph)
color_list = []
for a in cur_graph.y:
	if a == 0:
		color_list.append('green')
	elif a == 1:
		color_list.append('grey')
	else:
		color_list.append('red')

previous = []
big = []
iter_ = 0
temp = []
alt_edge_index = []
alt_edge_index_bad = []
while len(temp) != 0 or iter_ == 0:
	if iter_ > 0:
		temp = []
	if iter_ == 0:
		for a in edge_index:
			if a[0] == 0 and (seqs_505[a[1]][2]==0):
				temp.append(a[1])
				big.append(a[1])
				alt_edge_index.append(a)
				#print(seqs_505[a[1]][0])
	else:
		for a in edge_index:
			# check = 1
			# for b in range(len(previous)):
			# 	if a[1] in previous[b]:
			# 		check = 0
			# 		break
			if (a[0] in previous[-1]) and (a[1] != 0) and (a[1] not in big) and (seqs_505[a[1]][2]==0):
				big.append(a[1])
				temp.append(a[1])
				alt_edge_index.append(a)
			elif (a[1] != 0) and (a[1] not in big) and (seqs_505[a[1]][2]==0) and (seqs_505[a[0]][2]==0):
				alt_edge_index_bad.append(a)
				#print(a)
	previous.append(temp)
	print('ITER '+str(iter_))
	print(len(previous[-1]))
	print(previous[-1])
	iter_ +=1
tot = 0
for i, a in enumerate(previous):
	# print('LEVEL '+str(i+1))
	# print(a)
	tot += len(a)
	# for b in a:
	# 	print(str(b)+': '+str(seqs_505[b][0]))
print(tot)
print(tot / len(x))
print('Worse: '+str(y.count(1)))
print('Like: '+str(y.count(0)))
print('Active: '+str(y.count(0) + y.count(1)))

big_nr = []
for a in big:
	if a not in big_nr:
		big_nr.append(a)
print(len(big))
print(len(big_nr))

edgenames = []
seq_edges = []
for a in alt_edge_index:
	if seqs_505[a[0]][2] != 0:
		print('NONBINDER IN GOOD')
		print(seqs_505[a[0]][0])
	if seqs_505[a[1]][2] != 0:
		print('NONBINDER IN GOOD')
		print(seqs_505[a[1]][0])
	edgenames.append([seqs_505[a[0]][0], seqs_505[a[1]][0]])
	seq_edges.append([seqs_505[a[0]][1], seqs_505[a[1]][1]])
edgenames_bad = []
seq_edges_bad = []
for a in alt_edge_index_bad:
	if seqs_505[a[0]][2] != 0:
		print('NONBINDER IN BAD')
		print(seqs_505[a[0]][0])
	if seqs_505[a[1]][2] != 0:
		print('NONBINDER IN BAD')
		print(seqs_505[a[1]][0])
	edgenames_bad.append([seqs_505[a[0]][0], seqs_505[a[1]][0]])
	seq_edges_bad.append([seqs_505[a[0]][1], seqs_505[a[1]][1]])
print(edgenames[0])
print(edgenames_bad[0])

nodes_good = []
for a in edgenames:
	if a[0] not in nodes_good:
		nodes_good.append(a[0])
	if a[1] not in nodes_good:
		nodes_good.append(a[1])
nodes_bad = []
for a in edgenames_bad:
	if a[0] not in nodes_bad:
		nodes_bad.append(a[0])
	if a[1] not in nodes_bad:
		nodes_bad.append(a[1])
#print(nodes_bad)

seqs_good = []
for a in seq_edges:
	if a[0] not in seqs_good:
		seqs_good.append(a[0])
	if a[1] not in seqs_good:
		seqs_good.append(a[1])
seqs_bad = []
for a in seq_edges_bad:
	if a[0] not in seqs_bad:
		seqs_bad.append(a[0])
	if a[1] not in seqs_bad:
		seqs_bad.append(a[1])



counts_bad = []
for a, b in zip(seqs_bad, nodes_bad):
	#print(b)
	neighbor_list, neighbor_count = count_neighbors(a, seqs_505)
	#print(neighbor_list)
	counts_bad.append(neighbor_count)
counts_good = []
for a, b in zip(seqs_good, nodes_good):
	#print(b)
	neighbor_list, neighbor_count = count_neighbors(a, seqs_505)
	#print(neighbor_list)
	counts_good.append(neighbor_count)
print('GOOD: AVG: '+str(round(statistics.mean(counts_good),3))+' STDEV: '+str(round(statistics.stdev(counts_good),3)))
print('BAD: AVG: '+str(round(statistics.mean(counts_bad),3))+' STDEV: '+str(round(statistics.stdev(counts_bad),3)))


#print(edgenames)
#df3 = pd.read_csv('tsne_505.csv')
df3 = pd.read_csv('505_circle_like.csv')
x_good = []
y_good = []

x_bad = []
y_bad = []

for a, b, c in zip(df3['name'], df3['x'], df3['y']):
	if a in nodes_good:
		x_good.append(b)
		y_good.append(c)
	else:
		x_bad.append(b)
		y_bad.append(c)
fig, ax = plt.subplots(figsize=(10,10))

plt.scatter(x_bad, y_bad, s=40, zorder=1, color='red', label='Disconnected')
plt.scatter(x_good, y_good, s=20, zorder=1, color='green', label='Connected')


all_names = list(df3['name'])
all_x = list(df3['x'])
all_y = list(df3['y'])
ctr=0
for a in edgenames:
	#print(a)
	name1 = a[0]
	x1 = all_x[all_names.index(name1)]
	y1 = all_y[all_names.index(name1)]

	name2 = a[1]
	x2 = all_x[all_names.index(name2)]
	y2 = all_y[all_names.index(name2)]

	plt.plot([x1,x2],[y1,y2],color='black',linewidth=0.5,zorder=0)
	ctr+=1
	# if ctr == 50:
	# 	break
plt.legend()

for i in range(5):
	circle = Circle((0,0), i+1, edgecolor='black', linestyle=':',facecolor='none',zorder=0)
	ax.add_patch(circle)

#plt.savefig('505_circle_like_only_thick_connected.png')
plt.show()
print(len(nodes_good))
print(len(nodes_bad))
print(len(nodes_good)/(len(nodes_good)+len(nodes_bad)))
# LIB 505
# LIKE + WORSE
# PERCENTAGE OF EDGES - 31.44%
# 2441 / 2447 ACTIVE VARIANTS INCLUDED
# LIKE
# PERCENTAGE OF EDGES - 5.28%
# 410 / 411 ACTIVE VARIANTS INCLUDED

# LIB 456
# LIKE + WORSE
# PERCENTAGE OF EDGES - 58.41%
# 11418 / 11419 ACTIVE VARIANTS INCLUDED
# LIKE
# PERCENTAGE OF EDGES - 0.87%
# 170 / 193 ACTIVE VARIANTS INCLUDED
# NUM NEIGHBORS
# GOOD: AVG: 25.865 STDEV: 0.433
# BAD: AVG: 25.852 STDEV: 0.445


# level1 = []
# level2 = []
# level3 = []
# level4 = []
# level5 = []
# level6 = []
# for a in edge_index:
# 	if a[0] == 0:
# 		level1.append(a[1])
# for a in edge_index:
# 	if a[0] in level1 and a[1] not in level1 and a[1] != 0:
# 		level2.append(a[1])
# for a in edge_index:
# 	if a[0] in level2 and a[1] not in level2 and a[1] not in level1:
# 		level3.append(a[1])
# for a in edge_index:
# 	if a[0] in level3 and a[1] not in level3 and a[1] not in level2 and a[1] not in level1:
# 		level4.append(a[1])
# for a in edge_index:
# 	if a[0] in level4 and a[1] not in level4 and a[1] not in level3 and a[1] not in level2 and a[1] not in level1:
# 		level5.append(a[1])
# for a in edge_index:
# 	if a[0] in level5 and a[1] not in level5 and a[1] not in level4 and a[1] not in level3 and a[1] not in level2 and a[1] not in level1:
# 		level6.append(a[1])
# print('LEVEL 1')
# for a in level1:
# 	print(seqs_505[a][0])
# print('LEVEL 2')
# for a in level2:
# 	print(seqs_505[a][0])
# print('LEVEL 3')
# for a in level3:
# 	print(seqs_505[a][0])
# print('LEVEL 4')
# for a in level4:
# 	print(seqs_505[a][0])
# print('LEVEL 5')
# for a in level5:
# 	print(seqs_505[a][0])
# print('LEVEL 6')
# for a in level6:
# 	print(seqs_505[a][0])

# G = to_networkx(cur_graph)
# print(list(G.nodes))
# print(list(G.edges))
# visualize_graph(G, color=color_list)
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
