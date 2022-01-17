# Recommendation System Simulations
![plot](./figures/Framework.pdf?raw=true)
## Required libraries:
- pandas
- numpy
- tensorflow
- scipy

## To run
- Modify required parameters in AlgorithmicBias_Github.py
- Typical simulation can be run with:
$ python AlgorithmicBias_Github.py 

## What this code does
Output are all parameters for model simulations and time-varying features of model, including item popularity, which can be used to calculate Gini coefficient, cumulative item popularity, etc.

## Output
File name lists basic parameters for code, including whether uniform_beta = True (feature of teacher model)
Pickle file with the following keys:
- 'P': Teacher model P
- 'Q': Teacher model Q
- 'n': Number of agents (default = 4000)
- 'm': Number of items (default = 200)
- 'k': Latent features in teacher model (default = 4)
- 'beta': Beta scalar parameter for teacher model. If uniform_beta = True, this is always 0.0. 
- 'r': Number of items recommended before model is retrained
- 'fract_available': Initial data student model is trained on (default = 0.1%)
- 'epsilon': Parameter for epsilon-greedy strategy (epsilon = 0.0 for greedy strategy, 0.1 for epsilon greedy strategy, and 1.0 for random strategy)
- 'embeddings': Latent features in student model ("k'" in paper; default = 5)
- 'realization': Always 1. Modifications to code can allow multiple realizations to be saved in 1 .pkl file
- 'sim_data': data recorded at each timestep
- 'final_R_views': Final user-item matrix (R^{data} in the paper)
- 'final_U': Final P matrix for student model
- 'final_V': Final Q^T matrix for student model
- 'gt_U': Teacher model P (legacy)
- 'gt_V': Teacher model Q^T (legacy)

### sim_data 
This key contains several features for each timestep (listed in order):

- simulation timestep (values from 1 to m)
- min_val: minimum valence error (Brier score)
- Number of user-item pairs not recommended
- Popularity of each item (how many times they were chosen). Used to find mean item popularity, Gini coefficient **(statistics for Figs. 2, 4, 5b-c)**
- 2 statistics:
    - **(Legacy):** MSE error between student and teacher model probabilities
    - Brier score between trained student model and all collected user-item pairs **(statistic for Fig. 3, 5a)**
- **(Legacy):** [the item the student model predicts each user is most likely to choose, the rank the student model gives to the ground truth most preferred item for each agent]
- **(Legacy):** Correlations between student model-predicted popular items or users who pick many items and the teacher model ground truth

## Citation
Anonymized for review purposes
