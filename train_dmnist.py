import ddu_dirty_mnist
import argparse
import torch
import torchvision
from torch.autograd import Variable
import torch.utils.data.dataloader as Data
import numpy as np
import os
import matplotlib.pyplot as plt
from IPython import embed
parser = argparse.ArgumentParser()

parser.add_argument("--seed_k",default=0, type=int)
parser.add_argument("--gpu_idx",default=None, type=int)
parser.add_argument("--batch_size",default=64, type=int)
parser.add_argument("--repeat",default=0, type=int)
parser.add_argument("--epochs",default=15, type=int)
parser.add_argument("--kernel",default='gaussian', type=str)
parser.add_argument("--bandwidth",default=1.0, type=float)
parser.add_argument("--matrix_mode",default='valid',type=str)
parser.add_argument("--validation_split",default=0.85,type=float)
parser.add_argument("--balance_test_set",action = 'store_true')
parser.add_argument("--latent_unit",default=10,type=int)
parser.add_argument("--drop_testing",default=0.1,type=float)
parser.add_argument("--boundary_thres",default=0.8,type=float)
parser.add_argument("--activate_oocd",action = 'store_true')
parser.add_argument("--weighted",action = 'store_true')
args = parser.parse_args()

args.data_k = args.seed_k


'''load data and train model'''


test_arr = np.load(f'data/DMNIST/{args.data_k}_{args.validation_split}_test_arr.npy')
test_label = np.load(f'data/DMNIST/{args.data_k}_{args.validation_split}_test_label.npy')
valid_arr = np.load(f'data/DMNIST/{args.data_k}_{args.validation_split}_valid_arr.npy')
valid_label = np.load(f'data/DMNIST/{args.data_k}_{args.validation_split}_valid_label.npy')
train_arr = np.load(f'data/DMNIST/{args.data_k}_{args.validation_split}_train_arr.npy')
train_label = np.load(f'data/DMNIST/{args.data_k}_{args.validation_split}_train_label.npy')
test_dirty_boundary = np.load(f'data/DMNIST/{args.data_k}_{args.validation_split}_test_dirty_boundary.npy')
valid_dirty_boundary = np.load(f'data/DMNIST/{args.data_k}_{args.validation_split}_valid_dirty_boundary.npy')

'''train.py'''


np.set_printoptions(suppress=True)
os.makedirs('saved_models/unified/', exist_ok = True)
os.makedirs('saved_results/Density_approach/',exist_ok = True)
os.makedirs('saved_figs/Density_approach/',exist_ok = True)


alias = f"Dirty_OOCD_Weighted_{args.weighted}_{args.activate_oocd}_mis_{args.data_k}_ep_{args.epochs}_latent{args.latent_unit}_bw{args.bandwidth}_val{args.validation_split}_test{args.drop_testing}_bd{args.boundary_thres}_repeat{args.repeat}"


if args.gpu_idx is not None:
    torch.cuda.set_device(args.gpu_idx)
else:
    torch.cuda.set_device(int(args.repeat%8)) # auto-alloc gpus
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kk = args.data_k
batch_size = args.batch_size

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_normal_(m.weight, gain=1)
        #torch.nn.init.xavier_normal_(tensor, gain=1)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.xavier_normal_(m.weight, gain=1)
        torch.nn.init.zeros_(m.bias)

        
        
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(64 * 3 * 3, args.latent_unit),
            torch.nn.ReLU()
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(args.latent_unit, 10)
        )
        
        #torch.nn.init.xavier_uniform_(tensor, gain=1.0)
        
        
    def forward_feature(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.view(conv3_out.size(0), -1)
        out = self.dense(res)
        return out
    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.view(conv3_out.size(0), -1)
        out = self.dense(res)
        out = self.out(out)
        return out


model = Net().cuda()
model.apply(weights_init)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
loss_func = torch.nn.CrossEntropyLoss()


train_model_flag = True
print('training models...')
for epoch in range(args.epochs):
    print('epoch {}'.format(epoch + 1))
    # training-----------------------------
    train_loss = 0.
    train_acc = 0.
    for iter_i in range(int(train_label.shape[0]/ batch_size)):
        idx = np.random.randint(0, train_label.shape[0], batch_size)
        batch_x = torch.as_tensor(train_arr[idx]).float().cuda().unsqueeze(1)
        batch_y = torch.as_tensor(train_label[idx]).cuda()

        #batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        train_loss += loss.cpu().detach().item()
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.cpu().detach().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (train_label.shape[0]), \
                                                   train_acc / (train_label.shape[0])))

    # evaluation--------------------------------
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for batch_i in range(20): # split validation set 
        batch_x = torch.as_tensor(valid_arr[batch_i* int(valid_arr.shape[0]/20) : (batch_i + 1)* int(valid_arr.shape[0]/20)]).float()
        batch_y = torch.as_tensor(valid_label[batch_i* int(valid_arr.shape[0]/20): (batch_i + 1)* int(valid_arr.shape[0]/20)])
        out = model(batch_x.cuda().unsqueeze(1))
        loss = loss_func(out, batch_y.cuda()).cpu().detach().item()
        eval_loss += loss
        pred = torch.max(out, 1)[1].cpu().detach()
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.cpu().detach().item()

    print('Validation Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (valid_label.shape[0]), eval_acc / (valid_label.shape[0])))

model.cuda()



'''evaluation.py, preprocessing'''

total_x = np.vstack((train_arr, valid_arr, test_arr))
total_x_features = np.zeros((total_x.shape[0], args.latent_unit))
total_x_preds = np.zeros((total_x.shape[0],))
for i in range(np.ceil((train_label.shape[0]+valid_label.shape[0]+test_label.shape[0])/100).astype(int)):
    batch_x = total_x[i*100: (i+1)*100]
    total_x_features[i*100: (i+1)*100, :] = model.forward_feature(torch.as_tensor(batch_x).float().unsqueeze(1).cuda()).cpu().detach().numpy()
    total_x_preds[i*100: (i+1)*100] = torch.max(model.forward(torch.as_tensor(batch_x).float().unsqueeze(1).cuda()), 1)[1].cpu().detach()

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


# added new feature: drop 0.1 error examples for testing. args.drop_testing = 0.1
collected_error_idx = []
collected_error_features = []
collected_corr_idx = []
collected_corr_features = []


collected_error_pred = []
collected_error_label = []
collected_corr_pred = []
collected_corr_label = []

if args.matrix_mode == 'valid':
    pred_label_train_data_dict = {}
    for i in range(len(valid_pred)):
        try:
            pred_label_train_data_dict[f'{valid_label[i]}_{valid_pred[i]}']
        except:
            pred_label_train_data_dict[f'{valid_label[i]}_{valid_pred[i]}'] = []
            
        if valid_label[i] != valid_pred[i]:
            if np.random.random() > args.drop_testing:
                pred_label_train_data_dict[f'{valid_label[i]}_{valid_pred[i]}'].append(normalized_valid_features[i])
            else:
                collected_error_features.append(normalized_valid_features[i])
                collected_error_idx.append(i)
                collected_error_pred.append(valid_pred[i])
                collected_error_label.append(valid_label[i])
        else:
            if np.random.random() < args.drop_testing and valid_dirty_boundary[i] == False:
                collected_corr_features.append(normalized_valid_features[i])
                collected_corr_idx.append(i)
                collected_corr_pred.append(valid_pred[i])
                collected_corr_label.append(valid_label[i])
            else:
                pred_label_train_data_dict[f'{valid_label[i]}_{valid_pred[i]}'].append(normalized_valid_features[i])
elif args.matrix_mode == 'train':
    pred_label_train_data_dict = {}
    for i in range(len(train_pred)):
        try:
            pred_label_train_data_dict[f'{train_label[i]}_{train_pred[i]}']
        except:
            pred_label_train_data_dict[f'{train_label[i]}_{train_pred[i]}'] = []
        pred_label_train_data_dict[f'{train_label[i]}_{train_pred[i]}'].append(normalized_train_features[i])
        
print('train_label is',list(set(train_label)))


valid_acc = (valid_pred == valid_label).sum() / len(valid_pred)
amb_acc = (valid_pred[valid_dirty_boundary] == valid_label[valid_dirty_boundary]).sum() / np.sum(valid_dirty_boundary)
mnist_acc = (valid_pred[~valid_dirty_boundary] == valid_label[~valid_dirty_boundary]).sum() / np.sum(~valid_dirty_boundary)

print(f'validation performance: {valid_acc.round(4)} , On Ambiguous-MNIST: {amb_acc.round(4)} ; On MNIST: {mnist_acc.round(4)}', )
np.save(f'saved_results/Density_approach/validation_stats_{alias}', (valid_pred, valid_label, valid_dirty_boundary))





from sklearn.neighbors import KernelDensity

'''build corpus density estimators'''
corpus_density_matrix = {}
for label_set_i in list(set(train_label)):
    print('label_set_i',label_set_i)
    for pred_set_j in list(set(train_label)):
        try:
            data_ij = pred_label_train_data_dict[f'{label_set_i}_{pred_set_j}']
            corpus_density_matrix[f'{label_set_i}_{pred_set_j}'] = KernelDensity(kernel=args.kernel, bandwidth=args.bandwidth).fit(data_ij)
        except:
            corpus_density_matrix[f'{label_set_i}_{pred_set_j}'] = -1

max_diag_list = []
max_diag_idx_list = []
pred_diag_list = []
pred_max_list = []
corpus_density_mat = np.zeros((len(test_pred), len(set(test_label)), len(set(test_label)))) - 99999999.
collected_error_cdm = np.zeros((len(collected_error_idx), len(set(test_label)), len(set(test_label)))) - 99999999.
collected_corr_cdm = np.zeros((len(collected_corr_idx), len(set(test_label)), len(set(test_label)))) - 99999999.

corpus_density_weight = np.zeros((len(set(test_label)), len(set(test_label))))
collected_error_weight = np.zeros((len(set(test_label)), len(set(test_label))))
collected_corr_weight = np.zeros((len(set(test_label)), len(set(test_label))))
for i in range(len(valid_pred)):
    corpus_density_weight[valid_label[i], valid_pred[i]] += 1.0/len(valid_pred)
for i in range(len(collected_error_idx)):
    collected_error_weight[collected_error_label[i], collected_error_pred[i]] += 1.0/len(collected_error_idx)
for i in range(len(collected_corr_idx)):
    collected_corr_weight[collected_corr_label[i], collected_corr_pred[i]] += 1.0/len(collected_corr_idx)



training_diag = {}
training_all = []
for label_set_i in list(set(train_label)):
    print(label_set_i)
    for pred_set_j in list(set(train_label)):
        kde_model = corpus_density_matrix[f'{label_set_i}_{pred_set_j}']
        if kde_model != -1:
            corpus_density_mat[:, label_set_i, pred_set_j] = kde_model.score_samples(normalized_test_features)
            training_all.append(kde_model.score_samples(normalized_train_features))
            if args.drop_testing >0:
                collected_error_cdm[:, label_set_i, pred_set_j] = kde_model.score_samples(collected_error_features)
                collected_corr_cdm[:, label_set_i, pred_set_j] = kde_model.score_samples(collected_corr_features)
    training_diag[f'{label_set_i}'] = corpus_density_matrix[f'{label_set_i}_{label_set_i}'].score_samples(pred_label_train_data_dict[f'{label_set_i}_{label_set_i}'])
    # training_diag is for thresholding outliers. 
    
train_density_mat = np.zeros((len(train_pred), len(set(test_label)), len(set(test_label)))) - 99999999.
for label_set_i in list(set(train_label)):
    print('training_density',label_set_i)
    for pred_set_j in list(set(train_label)):
        kde_model = corpus_density_matrix[f'{label_set_i}_{pred_set_j}']
        if kde_model != -1:
            train_density_mat[:, label_set_i, pred_set_j] = kde_model.score_samples(normalized_train_features)

train_pred_diag_list = []
    
for train_idx_i in range(len(train_pred)):
    max_diag_value =  -99999999.0
    arg_max_idx = -1
    for i in range(10):
        if i == args.data_k:
            continue
        if max_diag_value < train_density_mat[train_idx_i, i, i]:
            arg_max_idx = i
            max_diag_value = train_density_mat[train_idx_i, i, i]
    train_pred_diag_list.append(train_density_mat[train_idx_i, train_pred[train_idx_i], train_pred[train_idx_i]])
    
    
    
for test_idx_i in range(len(test_pred)):
    max_diag_value =  -99999999.0
    arg_max_idx = -1
    for i in range(10):
        if i == args.data_k:
            continue
        if max_diag_value < corpus_density_mat[test_idx_i, i, i]:
            arg_max_idx = i
            max_diag_value = corpus_density_mat[test_idx_i, i, i]
    max_diag_idx_list.append(arg_max_idx)        
    max_diag_list.append(max_diag_value)
    pred_diag_list.append(corpus_density_mat[test_idx_i, test_pred[test_idx_i], test_pred[test_idx_i]])
    pred_max_list.append(corpus_density_mat[test_idx_i].max())
    

'''evaluation.py ood'''

prec_recall = []
#prec_recall_ood = []
#for threshold_i in [0. + 0.001*i for i in range(50)]:
for threshold_i in [0.]:
    quantile = []
    for label_set_i in list(set(test_label)):
        if label_set_i == args.data_k:
            quantile.append(-1)
        else:
            quantile.append(np.quantile(training_diag[f'{label_set_i}'], threshold_i))
    quantile = np.asarray(quantile)
    #ood_list = np.logical_or(max_diag_list < quantile[test_pred], max_diag_idx_list != test_pred)
    ood_list = (pred_diag_list < quantile[test_pred])
    ood_score1 = pred_diag_list / quantile[test_pred]
    # calculate precision, recall
    precision = (test_label[ood_list] == args.data_k).sum() / len(test_label[ood_list])
    recall = (test_label[ood_list] == args.data_k).sum() / (test_label == args.data_k).sum()
    prec_recall.append((precision, recall, ood_score1))

np.save(f'saved_results/Density_approach/oocd_prec_rec_score_{alias}', prec_recall)

prec_recall = []
#prec_recall_ood = []
#for threshold_i in [0. + 0.001*i for i in range(50)]:
for threshold_i in [0.0]:
    quantile = np.quantile(np.asarray(training_all).max(0), threshold_i)
    #ood_list = np.logical_or(max_diag_list < quantile[test_pred], max_diag_idx_list != test_pred)
    ood_list = pred_max_list < quantile
    ood_score2 = pred_max_list / quantile
    # calculate precision, recall
    precision = (test_label[ood_list] == args.data_k).sum() / len(test_label[ood_list])
    recall = (test_label[ood_list] == args.data_k).sum() / (test_label == args.data_k).sum()
    prec_recall.append((precision, recall, ood_score2))
    print('ood',prec_recall)
np.save(f'saved_results/Density_approach/ood_prec_rec_score_{alias}', prec_recall)


'''evaluation.py, oo(correct)d'''

if args.activate_oocd:
    prec_recall = []
    #prec_recall_ood = []
    #for threshold_i in [0. + 0.001*i for i in range(50)]:
    for threshold_i in [0.]:
        quantile = []
        for label_set_i in list(set(test_label)):
            if label_set_i == args.data_k:
                quantile.append(-1)
            else:
                quantile.append(np.quantile(training_diag[f'{label_set_i}'], threshold_i))
        quantile = np.asarray(quantile)
        #ood_list = np.logical_or(max_diag_list < quantile[test_pred], max_diag_idx_list != test_pred)
        ood_list = (pred_diag_list < quantile[test_pred])
        ood_score1 = pred_diag_list / quantile[test_pred]
        # calculate precision, recall
        precision = (test_label[ood_list] == args.data_k).sum() / len(test_label[ood_list])
        recall = (test_label[ood_list] == args.data_k).sum() / (test_label == args.data_k).sum()
        prec_recall.append((precision, recall, ood_score1))

    np.save(f'saved_results/Density_approach/oocd_prec_rec_score_{alias}', prec_recall)

    
    
'''evaluation.py, in-distribution'''

iD_num = (test_dirty_boundary>=0).sum() # number of indistribution samples



IDM_score = []
boundary_score = []
trusted_list = []
mask_trusted = np.zeros((len(test_arr)))
mask_IDM = np.zeros((len(test_arr)))
mask_trusted_score = np.zeros((len(test_arr)))
mask_IDM_score = np.zeros((len(test_arr)))

valid_uf_score_list = []



for i in range(len(test_arr)):
    if ood_list[i]:
        IDM_score.append(-1)
        boundary_score.append(-1)
    else:
        if args.weighted:
            IDM_score.append(np.clip(np.sum(
                np.exp( 
                corpus_density_weight[:, test_pred[i]] * corpus_density_mat[i, :, test_pred[i]]  - 
                corpus_density_weight[test_pred[i], test_pred[i]] * pred_diag_list[i]
                      )
                ), -1, 100))
            boundary_score.append(np.clip(np.sum(
                np.exp(
                    np.diagonal(corpus_density_weight * corpus_density_mat[i]) - 
                    corpus_density_weight[test_pred[i], test_pred[i]] * pred_diag_list[i]
                      )  
                ), -1, 100) )
        else:
            IDM_score.append(np.clip(np.sum(np.exp( (corpus_density_mat[i])[:,test_pred[i]] - pred_diag_list[i])), -1, 100))
            boundary_score.append(np.clip(np.sum(np.exp(np.diagonal(corpus_density_mat[i]) - pred_diag_list[i])  ), -1, 100) )
for i in range(len(test_arr)):
    acc_thres_IDM = np.quantile(IDM_score, valid_acc * 0.5)
    picked_thres_boundary = np.quantile(boundary_score, args.boundary_thres)
    
    if boundary_score[i] <= picked_thres_boundary:
        mask_trusted_score[i] = 1
    else:
        mask_trusted_score[i] = 0

    if IDM_score[i] > acc_thres_IDM:
        mask_IDM_score[i] = -1

    if corpus_density_mat[i].argmax()//10 != corpus_density_mat[i].argmax()%10 :
        mask_IDM[i] = -1
    elif (( np.diagonal(corpus_density_mat[i]) * 0.8 - corpus_density_mat[i][test_pred[i], test_pred[i]]) > 0).sum() <= 1:
        trusted_list.append(i)
        mask_trusted[i] = 1
    else:
        continue # here we can track the suspicious misclassifications
prec_boundary = (((1-mask_trusted)[:iD_num] == 1) * (test_dirty_boundary[:iD_num] == 1)).sum() / ((1-mask_trusted)[:iD_num] == 1).sum()
rec_boundary = (((1-mask_trusted)[:iD_num] == 1) * (test_dirty_boundary[:iD_num] == 1)).sum() / (test_dirty_boundary[:iD_num] == 1).sum()

prec_boundary_score = (((1-mask_trusted_score)[:iD_num] == 1) * (test_dirty_boundary[:iD_num] == 1)).sum() / ((1-mask_trusted_score)[:iD_num] == 1).sum()
rec_boundary_score = (((1-mask_trusted_score)[:iD_num] == 1) * (test_dirty_boundary[:iD_num] == 1)).sum() / (test_dirty_boundary[:iD_num] == 1).sum()



F1 = 2/(1/prec_boundary + 1/rec_boundary)
F1_score = 2/(1/prec_boundary_score + 1/rec_boundary_score)
print('new precision ', prec_boundary, prec_boundary_score )
print('new recall ', rec_boundary , rec_boundary_score)
print('F1', F1, F1_score)




'''for in-distribution IDM samples (validation of method only)'''
if args.drop_testing > 0.0: # only activated in demonstrative example.
    mask_IDM_valid = np.zeros((len(collected_error_idx) + len(collected_corr_idx)))
    mask_IDM_valid_score = np.zeros((len(collected_error_idx) + len(collected_corr_idx)))
    
    #prop = len(collected_corr_idx) / (len(collected_error_idx)+len(collected_corr_idx))
    prop = 0.5
    
    for i in range(len(collected_error_idx)):
        if args.weighted:
            vp_i = valid_pred[collected_error_idx[i]]
            valid_uf_score = np.clip(np.sum(
                np.exp(
                    collected_error_weight[:, vp_i] * collected_error_cdm[i, :, vp_i] 
                    - collected_error_weight[vp_i, vp_i] * collected_error_cdm[i, vp_i, vp_i] 
                      )
                ), -1, 100)
        else:
            valid_uf_score = np.clip(np.sum(np.exp(collected_error_cdm[i][:,valid_pred[collected_error_idx[i]]] - collected_error_cdm[i][valid_pred[collected_error_idx[i]],valid_pred[collected_error_idx[i]]] )), -1, 100)
        valid_uf_score_list.append(valid_uf_score)
        
        if collected_error_cdm[i].argmax()//10 != collected_error_cdm[i].argmax()%10 :
            mask_IDM_valid[i] = -1

    for i in range(len(collected_corr_idx)):
        if args.weighted:
            vp_i = valid_pred[collected_corr_idx[i]]
            valid_uf_score = np.clip(np.sum(
                np.exp(
                    collected_corr_weight[:, vp_i] * collected_corr_cdm[i, :, vp_i] 
                    - collected_corr_weight[vp_i, vp_i] * collected_corr_cdm[i, vp_i, vp_i] 
                      )
                ), -1, 100)
        else:
            valid_uf_score = np.clip(np.sum(np.exp(collected_corr_cdm[i][:,valid_pred[collected_corr_idx[i]]] - collected_corr_cdm[i][valid_pred[collected_corr_idx[i]],valid_pred[collected_corr_idx[i]]] )), -1, 100)
        valid_uf_score_list.append(valid_uf_score)
        if collected_corr_cdm[i].argmax()//10 != collected_corr_cdm[i].argmax()%10 :
            mask_IDM_valid[i + len(collected_error_idx)] = -1
    valid_thres_IDM = np.quantile(valid_uf_score_list, prop)
    for i in range(len(collected_error_idx)):
        if valid_uf_score_list[i] > valid_thres_IDM:
            mask_IDM_valid_score[i] = -1
    for i in range(len(collected_corr_idx)):        
        if valid_uf_score_list[i + len(collected_error_idx)] > valid_thres_IDM:
            mask_IDM_valid_score[i + len(collected_error_idx)] = -1
            
    prec_uf = (mask_IDM_valid[:len(collected_error_idx)] == -1).sum() / (mask_IDM_valid == -1).sum()
    prec_uf_score = (mask_IDM_valid_score[:len(collected_error_idx)] == -1).sum() / (mask_IDM_valid_score == -1).sum()
    rec_uf = (mask_IDM_valid[:len(collected_error_idx)] == -1).sum() / (len(collected_error_idx) ) 
    rec_uf_score = (mask_IDM_valid_score[:len(collected_error_idx)] == -1).sum() / (len(collected_error_idx) ) 

    F1_uf =  2/(1/prec_uf + 1/rec_uf)
    F1_uf_score = 2/(1/prec_uf_score + 1/rec_uf_score)
    print('prec of validation (score) ', prec_uf_score )
    print('prec of validation ',  prec_uf)
    print('recall of validation (score) ',  rec_uf_score)
    print('recall of validation ', rec_uf )
    print('F1-IDM', F1_uf, F1_uf_score)
    import pandas as pd
    box_df_valid = pd.DataFrame()
    oracle_gt = []
    for i,score in enumerate(mask_IDM_valid_score):
        if i <= len(collected_error_idx):
            oracle_gt.append('IDM')
        else:
            oracle_gt.append('Correct')
    box_df_valid['Ocacle'] = oracle_gt
    box_df_valid['IDM_score'] = valid_uf_score_list

    box_df_valid.to_csv(f"saved_results/Density_approach/validufboxcsv_{alias}.csv", index = False)

prec_recall_boundary = (prec_boundary,rec_boundary,F1)
prec_recall_boundary_score = (prec_boundary_score,rec_boundary_score,F1_score)

prec_recall_IDM = (prec_uf,rec_uf,F1_uf)
prec_recall_IDM_score = (prec_uf_score,rec_uf_score,F1_uf_score)


np.save(f'saved_results/Density_approach/boundary_{alias}', prec_recall_boundary)
np.save(f'saved_results/Density_approach/boundary_score_{alias}', prec_recall_boundary_score)
np.save(f'saved_results/Density_approach/IDM_{alias}', prec_recall_IDM)
np.save(f'saved_results/Density_approach/IDM_score_{alias}', prec_recall_IDM_score)


'''for box-plots'''

import pandas as pd
box_df = pd.DataFrame()
oracle_gt = []
for i in test_dirty_boundary:
    if i ==0:
        oracle_gt.append('MNIST')
    elif i == 1:
        oracle_gt.append('Ambiguous')
    else:
        oracle_gt.append('OOD')
box_df['Ocacle'] = oracle_gt
box_df['ood_score1'] = ood_score1
box_df['ood_score2'] = ood_score2
box_df['IDM_score'] = IDM_score
box_df['boundary_score'] = boundary_score

box_df.to_csv(f"saved_results/Density_approach/boxcsv_{alias}.csv", index = False)