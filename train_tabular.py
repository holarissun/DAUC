import argparse
import torch

parser = argparse.ArgumentParser()

parser.add_argument("--seed_k",default=0, type=int)
parser.add_argument("--gpu_idx",default=None, type=int)
parser.add_argument("--batch_size",default=16, type=int)
parser.add_argument("--repeat",default=0, type=int)
parser.add_argument("--epochs",default=300, type=int)
parser.add_argument("--kernel",default='gaussian', type=str)
parser.add_argument("--bandwidth",default=0.01, type=float)
parser.add_argument("--matrix_mode",default='valid',type=str)
parser.add_argument("--validation_split",default=0.85,type=float)
parser.add_argument("--latent_unit",default=10,type=int)
parser.add_argument("--drop_testing",default=0.0,type=float)
parser.add_argument("--IDM_thres",default=0.8,type=float)
parser.add_argument("--args.boundary_thres",default=0.8,type=float)
parser.add_argument("--dataset",default=None,type=str) # Covtype, SubSpam, Digits


args = parser.parse_args()
args.data_k = args.seed_k
if args.gpu_idx is not None:
    torch.cuda.set_device(args.gpu_idx)
else:
    torch.cuda.set_device(int(args.repeat%8)) # auto-alloc gpus
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import ddu_dirty_mnist
import torchvision
from torch.autograd import Variable
import torch.utils.data.dataloader as Data
import numpy as np
import os
import matplotlib.pyplot as plt
from IPython import embed

'''Covtype'''
tot_res_jan = []
for REPEAT_I in range(10):
    



    if args.dataset == 'SubSpam':
        import pandas as pd
        import random
        from sklearn.ensemble import RandomForestClassifier
        X = pd.read_csv('../IRL_imputation/dataset/spambase.data',',',header=None).values[:,:-1]
        y = pd.read_csv('../IRL_imputation/dataset/spambase.data',',',header=None).values[:,-1].astype(int)
        subset_idx = random.sample(range(X.shape[0]), 4601) # subset, size = 4601
        forest = RandomForestClassifier(random_state=0)
        forest.fit(X,y)
        importances = np.abs(forest.feature_importances_)
        costs_ind = (importances/importances.max())>0.1
        X = X[subset_idx]
        y = y[subset_idx]
        X = X[:,costs_ind]
    elif args.dataset == 'Digits':
        from sklearn.datasets import load_digits
        dataset = load_digits()
        X, y = dataset['data'], dataset['target']
    elif args.dataset == 'Covtype':
        from sklearn.datasets import fetch_covtype
        import random
        dataset = fetch_covtype()
        X, y = dataset['data'], dataset['target']-1
        subset_idx = random.sample(range(X.shape[0]), 30000)
        print('subset', len(subset_idx))
        X = X[subset_idx]
        y = y[subset_idx]


    def split_data(data_x, data_y):
        total_num = len(data_x)
        train_prop, valid_prop, test_prop = 0.4, 0.4, 0.2
        n_train = int(total_num * train_prop)
        n_valid = int(total_num * valid_prop)
        n_test = total_num - n_train - n_valid

        full_idx = np.random.choice(total_num,total_num,False)  

        train_arr = data_x[full_idx[:n_train]]
        train_label = data_y[full_idx[:n_train]].astype(int)
        valid_arr = data_x[full_idx[n_train:n_train + n_valid]]
        valid_label = data_y[full_idx[n_train:n_train + n_valid]].astype(int)
        test_arr = data_x[full_idx[-n_test:]]
        test_label = data_y[full_idx[-n_test:]].astype(int)
        print(train_arr.shape, valid_arr.shape, test_arr.shape)
        return train_arr, train_label, valid_arr, valid_label, test_arr, test_label

    train_arr, train_label, valid_arr, valid_label, test_arr, test_label = split_data(X, y)
    args.latent_unit = train_arr.shape[1]
    args.n_class = len(set(train_label))


    '''train.py'''


    np.set_printoptions(suppress=True)
    os.makedirs('saved_models/unified/', exist_ok = True)
    os.makedirs('saved_results/Tabular_results/',exist_ok = True)
    os.makedirs('saved_figs/Tabular_results/',exist_ok = True)


    alias = f"LOG_BS_{args.batch_size}_DS{args.dataset}_ep_{args.epochs}_latent{args.latent_unit}_bw{args.bandwidth}_bd{args.args.boundary_thres}_repeat{args.repeat}"

    batch_size = args.batch_size





    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import RidgeClassifier
    clf = RidgeClassifier().fit(train_arr, train_label)
    '''evaluation.py, preprocessing'''

    total_x = np.vstack((train_arr, valid_arr, test_arr))
    total_x_features = np.zeros((total_x.shape[0], args.latent_unit))
    total_x_preds = np.zeros((total_x.shape[0],))
    for i in range(np.ceil((train_label.shape[0]+valid_label.shape[0]+test_label.shape[0])/100).astype(int)):
        batch_x = total_x[i*100: (i+1)*100]
        total_x_features[i*100: (i+1)*100, :] = torch.as_tensor(batch_x)
        total_x_preds[i*100: (i+1)*100] = clf.predict(torch.as_tensor(batch_x))

    train_pred = total_x_preds[:train_arr.shape[0]].astype(int)
    valid_pred = total_x_preds[train_arr.shape[0]:-test_arr.shape[0]].astype(int)
    test_pred = total_x_preds[-test_arr.shape[0]:].astype(int)
    assert test_pred.shape[0] + train_pred.shape[0] + valid_pred.shape[0] == total_x_preds.shape[0]

    train_features = total_x_features[:train_arr.shape[0]]
    valid_features = total_x_features[train_arr.shape[0]:-test_arr.shape[0]]
    test_features = total_x_features[-test_arr.shape[0]:]

    train_corr_pred = (train_label == train_pred)

    normalized_train_features = (train_features - train_features.mean()) / (train_features.std() + 1e-4)
    normalized_valid_features = (valid_features - valid_features.mean()) / (valid_features.std() + 1e-4)
    normalized_test_features = (test_features - train_features.mean()) / (train_features.std() + 1e-4)


    if args.matrix_mode == 'valid':
        pred_label_train_data_dict = {}
        for i in range(len(valid_pred)):
            try:
                pred_label_train_data_dict[f'{valid_label[i]}_{valid_pred[i]}']
            except:
                pred_label_train_data_dict[f'{valid_label[i]}_{valid_pred[i]}'] = []
            pred_label_train_data_dict[f'{valid_label[i]}_{valid_pred[i]}'].append(normalized_valid_features[i])



    valid_acc = (valid_pred == valid_label).sum() / len(valid_pred)





    from sklearn.neighbors import KernelDensity

    '''build corpus density estimators'''
    corpus_density_matrix = {}
    for label_set_i in list(set(train_label)):
        for pred_set_j in list(set(train_label)):
            try:
                data_ij = pred_label_train_data_dict[f'{label_set_i}_{pred_set_j}']
                corpus_density_matrix[f'{label_set_i}_{pred_set_j}'] = KernelDensity(kernel=args.kernel, bandwidth=args.bandwidth).fit(data_ij)
            except:
                corpus_density_matrix[f'{label_set_i}_{pred_set_j}'] = -1

    max_diag_list = []
    max_diag_idx_list = []

    corpus_density_mat = np.zeros((len(test_pred), len(set(test_label)), len(set(test_label)))) - 99999999.



    if len(test_arr) < 5000:
        training_diag = {}
        training_all = []
        for label_set_i in list(set(train_label)):
            for pred_set_j in list(set(train_label)):
                kde_model = corpus_density_matrix[f'{label_set_i}_{pred_set_j}']
                if kde_model != -1:
                    corpus_density_mat[:, label_set_i, pred_set_j] = kde_model.score_samples(normalized_test_features)
                    training_all.append(kde_model.score_samples(normalized_train_features))
            # training_diag is for thresholding outliers. 
    else:
        import multiprocessing
        from multiprocessing import Process
        from multiprocessing import Pool
        import time
        N_SAMPLE = normalized_test_features.shape[0]
        N_EACH_TACKLE = 1000
        PARALLEL_SOLUTION = np.zeros((N_SAMPLE,))
        N_THRED = int(np.ceil(N_SAMPLE / N_EACH_TACKLE))

        os.makedirs('tempbu',exist_ok=True)
        def func_fill_tensor(kde, ftr, index_to_fill, label_set_i, pred_set_j):
            temp_var = kde.score_samples(ftr[index_to_fill * N_EACH_TACKLE : (index_to_fill + 1)*N_EACH_TACKLE])
            np.save(f'tempbu/{alias}_{index_to_fill}_{label_set_i}_{pred_set_j}', temp_var)

        max_diag_list = []
        max_diag_idx_list = []
        pred_diag_list = []
        pred_max_list = []
        corpus_density_mat = np.zeros((len(test_pred), len(set(test_label)), len(set(test_label)))) - 99999999.
        training_diag = {}
        start_time = time.time()
        temp_start_time = time.time()
        for label_set_i in list(set(train_label)):
            for pred_set_j in list(set(train_label)):
                kde_model = corpus_density_matrix[f'{label_set_i}_{pred_set_j}']
                if kde_model != -1:
                    process_list = []
                    for thred_i in range(N_THRED):
                        p = Process(target=func_fill_tensor, args=(kde_model, 
                                                                   normalized_test_features,
                                                                   thred_i,
                                                                   label_set_i,
                                                                   pred_set_j
                                                                  ))

                        process_list.append(p)
                        p.start()
                    for p in process_list:
                        p.join()
                    for thred_i in range(N_THRED):
                        temp_var = np.load(f"tempbu/{alias}_{thred_i}_{label_set_i}_{pred_set_j}.npy")
                        corpus_density_mat[thred_i * N_EACH_TACKLE : (thred_i + 1)*N_EACH_TACKLE, label_set_i, pred_set_j] = temp_var
                    temp_start_time = time.time()




    import os
    import multiprocessing
    from multiprocessing import Process
    from multiprocessing import Pool
    import time
    N_SAMPLE = normalized_train_features.shape[0]
    N_EACH_TACKLE = 1000
    PARALLEL_SOLUTION = np.zeros((N_SAMPLE,))
    N_THRED = int(np.ceil(N_SAMPLE / N_EACH_TACKLE))

    def func_fill_tensor(kde, ftr, index_to_fill, label_set_i, pred_set_j):
        temp_var = kde.score_samples(ftr[index_to_fill * N_EACH_TACKLE : (index_to_fill + 1)*N_EACH_TACKLE])
        np.save(f'tempbu/2{alias}_{index_to_fill}_{label_set_i}_{pred_set_j}', temp_var)



    start_time = time.time()
    train_density_mat = np.zeros((len(train_pred), len(set(test_label)), len(set(test_label)))) - 99999999.
    for label_set_i in list(set(train_label)):
        for pred_set_j in list(set(train_label)):
            kde_model = corpus_density_matrix[f'{label_set_i}_{pred_set_j}']
            if kde_model != -1:
                process_list = []
                for thred_i in range(N_THRED):
                    p = Process(target=func_fill_tensor, args=(kde_model, 
                                                               normalized_train_features,
                                                               thred_i,
                                                               label_set_i,
                                                               pred_set_j
                                                              )) 
                    p.start()
                    process_list.append(p)
                for p in process_list:
                    p.join()
                for thred_i in range(N_THRED):
                    temp_var = np.load(f"tempbu/2{alias}_{thred_i}_{label_set_i}_{pred_set_j}.npy")
                    train_density_mat[thred_i * N_EACH_TACKLE : (thred_i + 1)*N_EACH_TACKLE, label_set_i, pred_set_j] = temp_var

                assert train_density_mat.std() != 0



    train_pred_diag_list = []

    for train_idx_i in range(len(train_pred)):
        max_diag_value =  -99999999.0
        arg_max_idx = -1
        for i in range(len(set(train_label))):
            if i == args.data_k:
                continue
            if max_diag_value < train_density_mat[train_idx_i, i, i]:
                arg_max_idx = i
                max_diag_value = train_density_mat[train_idx_i, i, i]
        train_pred_diag_list.append(train_density_mat[train_idx_i, train_pred[train_idx_i], train_pred[train_idx_i]])

    pred_max_list = []    
    pred_diag_list = []    
    for test_idx_i in range(len(test_pred)):
        max_diag_value =  -99999999.0
        arg_max_idx = -1
        for i in range(len(set(train_label))):
            if i == args.data_k:
                continue
            if max_diag_value < corpus_density_mat[test_idx_i, i, i]:
                arg_max_idx = i
                max_diag_value = corpus_density_mat[test_idx_i, i, i]
        max_diag_idx_list.append(arg_max_idx)        
        max_diag_list.append(max_diag_value)
        pred_diag_list.append(corpus_density_mat[test_idx_i, test_pred[test_idx_i], test_pred[test_idx_i]])
        pred_max_list.append(corpus_density_mat[test_idx_i].max())



    for threshold_for_boundary in [args.boundary_thres]:
        underfit_score = []
        boundary_score = []
        trusted_list = []
        mask_trusted = np.zeros((len(test_arr)))
        mask_underfit = np.zeros((len(test_arr)))
        mask_trusted_score = np.zeros((len(test_arr)))
        mask_underfit_score = np.zeros((len(test_arr)))

        valid_uf_score_list = []
        for i in range(len(test_arr)):
            underfit_score.append(np.clip(np.sum(np.exp( (corpus_density_mat[i])[:,test_pred[i]] - pred_diag_list[i])), -1, 100))
            boundary_score.append(np.clip(np.sum(np.exp(np.diagonal(corpus_density_mat[i]) - pred_diag_list[i])  ), -1, 100) )
        for i in range(len(test_arr)):
            acc_thres_underfit = np.quantile(underfit_score, valid_acc)
            picked_thres_boundary = np.quantile(boundary_score, args.args.boundary_thres)

            if boundary_score[i] <= picked_thres_boundary:
                mask_trusted_score[i] = 1
            else:
                mask_trusted_score[i] = 0

            if underfit_score[i] >= acc_thres_underfit:
                mask_underfit_score[i] = -1

            if corpus_density_mat[i].argmax()//10 != corpus_density_mat[i].argmax()%10 :
                mask_underfit[i] = -1
            elif (( np.diagonal(corpus_density_mat[i]) * threshold_for_boundary - corpus_density_mat[i][test_pred[i], test_pred[i]]) > 0).sum() <= 1:
                trusted_list.append(i)
                mask_trusted[i] = 1
            else:
                continue # here we can track the suspicious misclassifications


    tot_out_list = []
    pie_chart_classes = [0,0,0,0,0,0]

    log_output = []
    class_labels = np.zeros((len(mask_underfit_score)))
    class_acc_list = []
    for i in range(len(mask_trusted_score)):
            if mask_trusted_score[i] == 0 and mask_underfit_score[i] != -1:
                log_output.append('boundary')
                pie_chart_classes[1] +=1
                class_labels[i] = 2 # boundary
            elif mask_underfit_score[i] == -1 and mask_trusted_score[i] != 0:
                log_output.append('underfit')
                pie_chart_classes[2] +=1
                class_labels[i] = 3 # underfit
            elif mask_trusted_score[i] == 0 and  mask_underfit_score[i] == -1:
                log_output.append('B & U')
                pie_chart_classes[3] +=1
                class_labels[i] = 4 # B & U
            else:
                log_output.append('trusted')
                pie_chart_classes[4] +=1
                class_labels[i] = 5
    for i in range(1,6):
        mask = class_labels == i
        if mask.sum() > 0:
            class_acc_list.append(100*(test_pred[mask] == test_label[mask]).sum() / mask.sum())
        else:
            class_acc_list.append(0.0)
    mask = np.ones((len(test_label),)).astype(bool)
    class_acc_list.append(100*(test_pred[mask] == test_label[mask]).sum() / mask.sum())
    labels=['OOD','Boundary','Underfit','B&U','trusted','Overall']
    pie_chart_classes = np.asarray(pie_chart_classes)
    porcent = 100.*pie_chart_classes/pie_chart_classes.sum()
    labels = [f'{i} - acc {np.round(j,2)} % - {np.round(k,2)} %' for i,j,k in zip(labels, class_acc_list, porcent)]

    print('baseline:', class_acc_list[3])

    '''use scores'''

    minor_idx = ((mask_underfit_score == -1 )*( mask_trusted_score == 0)) # B&I Class

    print('number of minority examples', minor_idx.sum())
    if minor_idx.sum()<=1:
        continue
    minor_test_arr = test_arr[minor_idx]
    minor_test_label = test_label[minor_idx]
    minor_test_features = test_features[minor_idx]
    normalized_minor_test_features = normalized_test_features[minor_idx]


    from sklearn.neighbors import KernelDensity
    test_kde_model = KernelDensity(kernel=args.kernel, bandwidth=args.bandwidth).fit(normalized_minor_test_features)

    train_density_score = np.zeros((len(train_arr)))

    import multiprocessing
    from multiprocessing import Process
    from multiprocessing import Pool
    import time
    start_time = time.time()
    N_SAMPLE = normalized_train_features.shape[0]
    N_EACH_TACKLE = 1000
    PARALLEL_SOLUTION = np.zeros((N_SAMPLE,))
    N_THRED = int(np.ceil(N_SAMPLE / N_EACH_TACKLE))

    os.makedirs('tempbu',exist_ok=True)
    def func_fill_tensor(kde, ftr, index_to_fill):
        temp_var = kde.score_samples(ftr[index_to_fill * N_EACH_TACKLE : (index_to_fill + 1)*N_EACH_TACKLE])
        np.save(f'tempbu/{alias}_{index_to_fill}', temp_var)
    process_list = []
    for thred_i in range(N_THRED):
        p = Process(target=func_fill_tensor, args=(test_kde_model, 
                                                   normalized_train_features,
                                                   thred_i
                                                  ))
        process_list.append(p)
        p.start()
    for p in process_list:
        p.join()
    for thred_i in range(N_THRED):
        temp_var = np.load(f"tempbu/{alias}_{thred_i}.npy")
        train_density_score[thred_i * N_EACH_TACKLE : (thred_i + 1)*N_EACH_TACKLE] = temp_var


    results_to_log_score = []
    results_to_log_overall = []
    for quantile_q in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        selected_train_idx = train_density_score >= np.quantile(train_density_score, quantile_q)
        selected_train_arr = train_arr[selected_train_idx]
        selected_train_features = train_features[selected_train_idx]
        selected_train_label = train_label[selected_train_idx]


        clf = RidgeClassifier().fit(selected_train_features, selected_train_label)
        results_to_log_score.append(clf.score(minor_test_features, minor_test_label))
    tot_res_jan.append(results_to_log_score)
tot_res_jan = np.asarray(tot_res_jan)
np.save(f'{args.dataset}_BnI_results.npy', tot_res_jan)
plt.plot([0.1*i for i in range(10)],np.mean(tot_res_jan, 0 ), label = 'w/ Filtered Training Data')
plt.plot([0.1*i for i in range(10)],[np.mean(tot_res_jan[:100],0)[0]]*10, label = 'w/o Filtered Training Data')
plt.fill_between([0.1*i for i in range(10)], np.mean(tot_res_jan[:],0)[:10] + np.std(tot_res_jan[:],axis=0)[:10], np.mean(tot_res_jan[:],0)[:10] - np.std(tot_res_jan[:], axis=0)[:10], alpha=0.25)
plt.legend(loc='upper left')
plt.title(f'{args.dataset}')
plt.xlabel('Match Quantile')
plt.ylabel('Prediction Acc.')

plt.show()

