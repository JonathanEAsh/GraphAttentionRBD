SEQUENCE ATTENTION SCRIPTS
========================
compile_results.py - compile performance metrics from model_out

rbd_attention.py - Train sequence attention model using ESM embeddings

round_1_2_train_binary.csv - combined training set (90% LY005+6+8+10)

round_1_2_val_binary.csv - combined validation set (10% LY005+6+8+10)

round_1_train_binary.csv - split training set (90% LY005+6+8)

round_1_val_binary.csv - split validation set (10% LY005+6+8)

round_2_test_binary.csv - split test set (LY010)

test_hyper.py - perform sequence attention hyperparameter optimization

test_rbd_attention_combo / split.py - perform model inference with trained seq attention model for combo and split protocols, respectively. Outputs individual sample labels and confidence.

===========

To replicate the paper's Sequence Attention model split training regiment:

1 - "cd ../"

2 - "mkdir -p embed/r1 embed/r2"

3 - "python extract_esm_embeddings.py" - precomputes ESM embeddings for model training

4 - "cd SequenceAttention"

5 - "python test_hyper.py"


========================

When training is complete:

6 - "python compile_results.py"

7 - select best model, redirect to it with the relevant testing script "python test_rbd_attention_split.py" for individual predictions and probabilities


