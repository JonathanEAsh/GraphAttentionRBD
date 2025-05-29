EPISTASIS SCRIPTS
========================
456_var.csv - LY006 samples

505_var.csv - LY005 samples

check_native_betas.py - examine the values of betas corresponding to the native state or pairs/triplets of states

determine_epistatic_effects_by_genscore.py - calculate main and epistatic effects using variant genetic scores

encode_muts.py - train logistic regression model and save the coefficients and prediction probabilities for downstream analysis

find_best_and_worst.py - find the epistatic terms with the largest positive and negative impact (>1 and <-1, respectively). Retrieve samples containing said betas and calculate the percentage of binders in the set.

find_variants_for_validation.py - separate samples based on their incorporation of states corresponding to the wt sequence.

get_betas.py - Name model coefficients based on corresponding state

get_genetic_scores.py - compute variant genetic scores based on zero-sum betas

load_model_thresholds.py - extract saved model intercept for genetic score calculation

normalize_betas / intercept.py - enforce zero sum constraint on model coefficients

plot_curves.py - plot roc and prc curves

plot_epistasis.py - plot epistatic effect histogram

plot_genetic_scores.py - make violin plots of genetic scores vs activity

plot_good_v_bad.py - make bar plot of percentage of binders in extreme beta containing samples

plot_mcc.py - make bar plot of performance metrics according to degree of features used by model

round_1_results_combined.csv required from neighborhood subdir

==========

To replicate the paper's epistatic analysis for the full feature set:

1 - "python encode_muts.py"

2 - "python get_betas.py"

3 - "python normalize_betas.py" - repeat for both libraries

4 - "python load_model_thresholds.py" 

5 - "python normalize_intercept.py"

6 - "python get_genetic_scores.py"

7 - "python determine_epistatic_effects_by_genscore.py"

