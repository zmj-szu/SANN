import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold,StratifiedKFold
from lifelines.utils import concordance_index
import tensorflow as tf
import torch
import torch.nn.functional as F
import math, copy
import matplotlib.pyplot as plt
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pytorchtools import EarlyStopping
from imblearn.over_sampling import SMOTE  
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_curve,auc
from scipy import interp
from sklearn.calibration import calibration_curve
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows


seed_value =528
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
tf.keras.utils.set_random_seed(seed_value)
tf.config.experimental.enable_op_determinism()
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled =True
# Sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
# Somer's D
def somers_d(y_true, y_pred):
    n = len(y_true)
    rankings = rankdata(y_pred, method='average')
    c = np.sum(y_true)
    d = np.sum(rankings[y_true == 1]) - c * (c + 1) / 2
    return d / (c * (n - c))
# Harrell's C
def harrell_c(y_true, y_pred):
    return 2 * roc_auc_score(y_true, y_pred) - 1
# G Index
def g_index(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return 2 * (sensitivity * specificity) - 1
def calculate_sensitivity_and_specificity(y_true,y_pred):
 
    TN, FP, FN, TP = confusion_matrix(y_true,y_pred).ravel()

    specificity = TN / (TN + FP)

    negative_predictive_value = TN / (TN + FN)

    if np.isnan(negative_predictive_value):
        negative_predictive_value = 0

    return  specificity,negative_predictive_value
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadedAttention(nn.Module):
    """
         :param h
         :param d_model
         :param query: Q
         :param key: K
         :param value: V
         :param dropout
         :param mask:
         :return:
         """
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)


        # 3) "Concat" using x view and apply x final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class selfattetionann(nn.Module):
    def __init__(self,h,d_model,dropout_multihead,dropout,num_te):
        super().__init__()
        self.input_to_hidden_layer_1 = nn.Linear(num_te,10)
        self.hidden_layer_activation_1 = nn.ReLU()
        self.input_to_hidden_layer_2 = nn.Linear(10,10)
        self.hidden_layer_activation_2 = nn.ReLU()
        self.input_to_hidden_layer_3 = nn.Linear(10,10)
        self.hidden_layer_activation_3 = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)  
        self.hidden_to_output_layer = nn.Linear(10,1)
        self.hidden_to_output_activation = nn.Sigmoid()
        self.selfattention = MultiHeadedAttention(h=h, d_model=d_model, dropout=dropout_multihead)
    def forward(self, x):
        x = self.selfattention(x, x, x, mask=None)
        x = x.squeeze(-1)
        x = self.input_to_hidden_layer_1(x)
        x = self.hidden_layer_activation_1(x)
        x = self.input_to_hidden_layer_2(x)
        x = self.hidden_layer_activation_2(x)
        x = self.input_to_hidden_layer_3(x)
        x = self.hidden_layer_activation_3(x)
        x = self.dropout_2(x)  
        x = self.hidden_to_output_layer(x)
        x = self.hidden_to_output_activation(x)
        return x

class MyDataset(Dataset):
    def __init__(self,x,y):
        self.x = x.clone().detach() # torch.tensor(x).float()
        self.y = y.clone().detach() # torch.tensor(y).float()
    def __len__(self):
        return len(self.x)
    def __getitem__(self, ix):
        return self.x[ix], self.y[ix]



data_df = pd.read_excel('/data_5.0/data_process5.0.xlsx')
data = data_df.drop('MT', axis=1)
# year = data['FollowYear'] #smote
# val_data = pd.read_excel('/data_7.0/CU_ExV_updating.xlsx')

##
batch_size = 40
epoch = 200
h= 1
num_te = 31
d_model = 1
n_splits = 10
dropout_multihead = 0.01
initial_lr = 0.01
dropout = 0.0005
weight_decay = 0.0001
validation_split=0.05
patience =30
##
##
X = data
Y = data_df['MT']
##smote
smote = SMOTE()  
X, Y = smote.fit_resample(X, Y)  
year = X['Follow-up']
##
X = X.drop('Follow-up', axis=1)
X_display = X.columns.tolist()

standard_s1 = MinMaxScaler()  
X = standard_s1.fit_transform(X)  
num_columns = X.shape[0]
data_tensor = torch.tensor(X)
X = data_tensor.reshape(num_columns, num_te, d_model)

Y = torch.tensor(Y).float()
Y = Y.unsqueeze(1)

# val_x = val_data.drop('Follow-up', axis=1)
# val_x = val_x.drop('MT', axis=1)
# standard_s3 = MinMaxScaler() 
# val_x = standard_s3.fit_transform(val_x)  
# num_columns = val_x.shape[0]
# data_tensor = torch.tensor(val_x)
# val_x = data_tensor.reshape(num_columns, num_te, d_model)
#
#
# val_y = val_data['MT']
# val_y = torch.tensor(val_y).float()
# val_y = val_y.unsqueeze(1)
#
# val_year = val_data['Follow-up']
# workbook = Workbook()
# workbook.remove(workbook.active)  # Remove the default sheet created by openpyxl



skfolds = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed_value)

accuracy_score_list, c_index_list, f1_score_list = [], [], []
auc_score_list, brier_score_list, sensitivity_list = [], [], []
specificity_list, precision_list, npv_list = [], [], []
f1_score_list, youden_j_list = [], []
threshold_list = []


for train_index, test_index in skfolds.split(X, Y):
    model = selfattetionann(h, d_model, dropout_multihead, dropout, num_te)
    opt = optim.Adam(model.parameters(), lr=initial_lr,weight_decay=weight_decay)
    # lossfuc = nn.BCELoss()
    lossfuc = nn.MSELoss()
    loss_history_train = []
    loss_history_val = []

    # val_size = int(validation_split * len(train_index))
    # val_index = train_index[:val_size]
    # train_index = train_index[val_size:]
    X_train_fold, y_train_fold = X[train_index], Y[train_index]

    # X_val_fold, y_val_fold = X[val_index], Y[val_index]
    X_test_fold, y_test_fold = X[test_index], Y[test_index]

    # X_test_fold = val_x
    # y_test_fold = val_y

    y_year = year.iloc[test_index]
    # y_year = val_year

    # early_stopping = EarlyStopping(patience=patience, verbose=True)
    ds = MyDataset(X_train_fold, y_train_fold)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    for _ in range(epoch):
        for data in dl:
         x, y = data
         opt.zero_grad()
         loss_value = lossfuc(model(x),y)
         loss_value.backward()
         opt.step()
        loss_value_train = lossfuc(model(X_train_fold), y_train_fold)
        loss_history_train.append(loss_value_train)
        # loss_value_val = lossfuc(model(X_val_fold), y_val_fold)
        # loss_history_val.append(loss_value_val)
        # early_stopping(loss_value_train, model)
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break
    loss_history_np_train = [tensor.detach().numpy() for tensor in loss_history_train]
    # loss_history_np_val = [tensor.detach().numpy() for tensor in loss_history_val]
    # plt.plot(loss_history_np_train, 'b', label='Train loss', lw=2, alpha=.8)
    # # # plt.plot(loss_history_np_val, 'r', label='Valid loss')
    # # plt.title('Train and Valid Loss')
    # plt.legend()
    # plt.xlabel('epochs')
    # plt.ylabel('loss value')
    # plt.show()
    
    y_pred_prob = model(X_test_fold)
    y_pred_prob = y_pred_prob.detach().numpy()
    y_pred_prob = pd.DataFrame(y_pred_prob)

    y_test_fold = y_test_fold .detach().numpy()
    y_test_fold = pd.DataFrame(y_test_fold)


    thresholds = np.linspace(0, 1, 1000)

    best_threshold = None
    max_diff = -1

    for threshold in thresholds:
 
        y_pred = (y_pred_prob >= threshold).astype(int)
        
        sensitivity = recall_score(y_test_fold, y_pred)
        specificity, negative_predictive_value = calculate_sensitivity_and_specificity(y_test_fold, y_pred)  #

        
        diff = abs(sensitivity - (1 - specificity))
        if diff > max_diff:
            max_diff = diff
            best_threshold = threshold
           

 
    y_pred = (y_pred_prob >= best_threshold).astype(int)
    
    c_index = concordance_index(y_year, 1 - y_pred_prob, y_test_fold)
    Auc = roc_auc_score(y_test_fold, y_pred_prob)  #
    brier_score = brier_score_loss(y_test_fold, y_pred_prob, pos_label=1)  #
    sensitivity = recall_score(y_test_fold, y_pred)
    specificity, negative_predictive_value = calculate_sensitivity_and_specificity(y_test_fold, y_pred)  #
    accuracy = accuracy_score(y_test_fold, y_pred)  #
    precision = precision_score(y_test_fold, y_pred, zero_division=0)  #
    f1 = f1_score(y_test_fold, y_pred)  #
    youden_j = sensitivity + specificity - 1  #

    c_index_list.append(c_index)
    auc_score_list.append(Auc)
    brier_score_list.append(brier_score)
    accuracy_score_list.append(accuracy)
    precision_list.append(precision)
    sensitivity_list.append(sensitivity)
    specificity_list.append(specificity)
    npv_list.append(negative_predictive_value)
    f1_score_list.append(f1)
    youden_j_list.append(youden_j)
    threshold_list.append(best_threshold)

    print(
            f'Fold: {len(c_index_list)}, C-index: {c_index:.4f}, AUC: {Auc:.4f}, Brier Score: {brier_score:.4f}, '
            f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Sensitivity: {sensitivity:.4f}, '
            f'Specificity: {specificity:.4f}, NPV: {negative_predictive_value:.4f}, F1: {f1:.4f}, '
            f'Youden\'s J: {youden_j:.4f}',)
  
    print(f'Threshold probability: {best_threshold:.4f}')


print(f'c_index: {np.mean(c_index_list):.2f} (+/- {np.std(c_index_list) * 2:.2f})')
print(f'auc: {np.mean(auc_score_list):.2f} (+/- {np.std(auc_score_list) * 2:.2f})')
print(f'brier: {np.mean(brier_score_list):.2f} (+/- {np.std(brier_score_list) * 2:.2f})')
print(f'Threshold probability: {np.mean(threshold_list):.2f} (+/-{np.std(threshold_list) * 2:.2f})')
print(f'Sensitivity: {np.mean(sensitivity_list):.2f} (+/- {np.std(sensitivity_list) * 2:.2f})')
print(f'Specificity: {np.mean(specificity_list):.2f} (+/- {np.std(specificity_list) * 2:.2f})')
print(f'accuracy: {np.mean(accuracy_score_list):.2f} (+/- {np.std(accuracy_score_list) * 2:.2f})')
print(f'precision: {np.mean(precision_list):.2f} (+/- {np.std(precision_list) * 2:.2f})')
print(f'NPV: {np.mean(npv_list):.2f} (+/-{np.std(npv_list) * 2:.2f})')
print(f'f1_score: {np.mean(f1_score_list):.2f} (+/- {np.std(f1_score_list) * 2:.2f})')
print(f'youden_j: {np.mean(youden_j_list):.2f} (+/-{np.std(youden_j_list) * 2:.2f})')

metrics_df = pd.DataFrame({
    'C-index': [np.mean(c_index_list)],
    'AUC': [np.mean(auc_score_list)],
    'Brier Score': [np.mean(brier_score_list)],
    'Threshold probability':[np.mean(threshold_list)],
    'Sensitivity': [np.mean(sensitivity_list)],
    'Specificity': [np.mean(specificity_list)],
    'Accuracy': [np.mean(accuracy_score_list)],
    'Precision': [np.mean(precision_list)],
    'Negative Predictive Value': [np.mean(npv_list)],
    'F1-score': [np.mean(f1_score_list)],
    "Youden's J statistic": [np.mean(youden_j_list)]
})

# save
metrics_df.to_excel('./smote_selfAttention_ann.xlsx', index=False)


