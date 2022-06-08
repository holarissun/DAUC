# DAUC: a Density-based Approach for Uncertainty Categorization
Code for paper Latent Density Models for Uncertainty Categorization.

Requirements:

- ddu_dirty_mnist
- PyTorch v1.8.1
- UQ360
- alibi_detect

## To Reproduce Our Results:
- Prepare Dataset with

python3 prep_data_dmnist.py

- Uncertainty Explanation:

python3 train_dmnist.py --latent_unit 40 --seed_k 0 --epoch 30 --bandwidth 1.0

- Inverse Direction:

python3 train_tabular.py --dataset Digits

