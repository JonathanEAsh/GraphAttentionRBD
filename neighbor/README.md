NEIGHBORHOOD SCRIPTS
========================
make_graphs_rbd_neighbor_like.py - Creates 1-hamming-distance graph for specified library. Plots the graph using concentric circles to place samples based on mutational distance from WT. 

456/505_neighbor_like.csv - Contains coordinates for concentric circle plot.

round_1_results_combined.csv - Contains the names, sequences, and binding classes of all samples from round 1 (LY005+6+8). For class, 0 = Like WT, 1 = Worse than WT, 2 = NB. NB and Worse are typically lumped together for a binary class representation downstream (0 = NB, 1 = Bind). 
