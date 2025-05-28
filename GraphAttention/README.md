GRAPH ATTENTION SCRIPTS
========================
gatv3.py - Trains GAT model with split dataset task (train 90% LY005+6+8, val 10% LY005+6+8,test LY010)
gatv3_combo.py - Trains GAT model with combined dataset task (train 90% LY005+6+8+10, val 10% LY005+6+8+10)
- Both model scripts will output performance metrics into binary/model_out, checkpoints into binary/saved_models, and loss plots into binary/loss_plots. Checkpoints are performed every 25 epochs. Loss plots are saved after the full 200 epoch training run.

get_dist_and_e.py - Pre-compute CA distances and interaction energies for faster graph creation
make_graph_sets.py - copies graphs from raw_graphs for graph dataset creation
make_graphs_rbd_neighbor.py - create RBD graphs based on Rosetta-generated decoys of samples
monitor_graphs.py - parallelize graph creation process into batches of 1000
process_graphs.py - perform graph processing to finalize torch_geometric dataset creation
round_1_2_val.csv - validation samples for the combo task
round_1_val.csv - validation samples for the split task
split_samples.py - perform data split into batches of 1000. Creates files in r1_samples_split and r2_samples_split
test_hyper.py - perform hyperparameter optimization for graph model training. Submits separate training jobs for each hyperparameter combination

binary/compile_results + compile_results_combo.py - compile performance metrics from model_out for both split and combo protocols, respectively.
binary/test_gat_split + combo.py - perform model inference on relevant dataset, recording prediction confidence for each individual sample.

To replicate the paper's GAT model split training regiment for the full feature set:
1 - "mkdir -p raw_graphs_binary/r1 raw_graphs_binary/r2"
2 - Ensure Rosetta decoys are downloaded. Default paths are ../decoys_r1/ and ../decoys_r2/. Native structure Wuhan.pdb also required. Default path ../../struct_relaxed/wuhan.pdb
3 - "python monitor_graphs.py"
4 - "mkdir -p train_r1_test_r2/train/raw train_r1_test_r2/train/processed train_r1_test_r2/val/raw train_r1_test_r2/val/processed train_r1_test_r2/test/raw train_r1_test_r2/test/processed"
5 - "python make_graph_sets.py"
6 - "python process_graphs.py"
7 - "mkdir -p binary/model_out/ binary/saved_models/ binary/loss_plots/
8 - "python test_hyper.py"

Necessary files not included in this repo: esm_stats_bind.csv, variant decoys, wuhan.pdb
========================
When training is complete:
9 - "cd binary"
10 - "python compile_results.py"
11 - select best model, redirect to it with the relevant testing script "python test_gat_split.py" for individual predictions and probabilities
========================
