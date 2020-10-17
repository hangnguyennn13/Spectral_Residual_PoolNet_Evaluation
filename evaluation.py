import os
import torch
import numpy as np
from PIL import Image
import sys 

# comment out the one that you dont need to test
# evaluate poolnet
ground_truth_path = "./data/DUTS-TE/Ground_Truth"
predicted_img_path = "./data/poolnet_result"

# evaluate Spectral residual 
ground_truth_path = "./data/DUTS-TE/Ground_Truth"
predicted_img_path = "./data/my_model_result"

# load ground truth images

gt_list = []
gt_paths = []
path = ground_truth_path  # ground image path directory
test = os.listdir(path)
for i in test: 
    if(i[-3:] == 'png'):
        gt_list.append(i)
gt_list = sorted(gt_list)
for i in range(len(gt_list)):
    gt_paths.append(os.path.join(path, gt_list[i]))

gt_paths = sorted(gt_paths)

# load predicted images 

pred_paths = []
path = predicted_img_path # predicted image directory
test_pred = sorted(os.listdir(path))
for i in range(len(test_pred)):
    pred_paths.append(os.path.join(path, test_pred[i]))

pred_paths = sorted(pred_paths)

def prec_rec(y_true, y_pred, beta2):
    
    eps = sys.float_info.epsilon
    tp = torch.sum(y_true * y_pred)
    all_p_pred = torch.sum(y_pred)
    all_p_true = torch.sum(y_true)
    
    prec = (tp + eps) / (all_p_pred + eps)
    rec = (tp + eps) / (all_p_true + eps)
    # print(prec)
    # print(rec)
    
    return prec, rec


overall_mae = 0
total_prec = 0
total_rec = 0 
for j in range(len(gt_paths)):
        try:
            gt = np.array(Image.open(gt_paths[j]).convert('LA')) / 255
            pred = np.array(Image.open(pred_paths[j]).convert('LA')) / 255 
            mae = np.sum(np.abs(pred - gt)) / (pred.shape[:2][0] * pred.shape[:2][1])

            gt_arr = torch.from_numpy(np.array(gt)).float()
            pred_arr = torch.from_numpy(np.array(pred)).float()

            y_true1 = torch.reshape(gt_arr, (1,-1))
            y_pred1 = torch.reshape(pred_arr, (1,-1))

            prec, rec = prec_rec(y_true1, y_pred1,0.3)
            print(prec, rec)
            total_prec = total_prec + prec
            total_rec = total_rec + rec

            overall_mae = overall_mae + mae
        except:
            print('error')

beta2 = 0.3 
overall_fb = (1+beta2) * (total_prec * total_rec) / ((beta2 * total_prec + total_rec) * len(gt_paths))
print('MAE', overall_mae/len(gt_paths) )
print('Precision', total_prec/len(gt_paths) )
print('Recall', total_rec/len(gt_paths) )