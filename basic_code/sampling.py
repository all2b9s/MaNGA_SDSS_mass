from .CNN_result import CNN_result
from basic_code.CNN_result import M_L
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

def sampling(model, Data, 
             if_scramble = False, # Do random scramble or not
             if_Gaussian = False, # Do Gaussian smoothing or not
             para = [], # parameters for Gaussian kernel [sigma, kernel_size]
             N_FIGS = 9):
    mode = 'test'
    test_set = Data[int(0.8*len(Data)):,:,:,:]
    where_ctr = test_set[:,(N_FIGS-3):(N_FIGS-2),:,:].bool().reshape([len(test_set),10000]).to('cpu')
    gmass = []
    bin_num = []
    VT_diff = []
    # int(len(test_set))
    for i in range(0, int(len(test_set))):
        result = CNN_result(model,i,Data)
        if if_Gaussian:
            result.Gaussian(k_size=para[1], sigma=para[0])
            
        if if_scramble:
            result.VT_lize(if_scramble = True)
        else:
            result.VT_lize()

        r_img = result.x[0,2,:,:].to('cpu').detach().numpy()
        M_L_r = M_L(result.x[0,(1,2),:,:].to('cpu').detach().numpy(), -0.306, 1.097)
        bell_mass = result.VT_bined(M_L_r+r_img+9)

        gmass.append([result.total_mass()[0],result.total_mass()[1],(10**bell_mass).sum()*0.16]) # pred, true, bell
        VT_ctr = test_set[i,6,:,:].to('cpu').detach().numpy()
        bin_num.append(np.sum(VT_ctr))

        pred_1d = torch.tensor(result.mass_pred_VT.ravel())
        truth_1d = torch.tensor(result.mass_tru.ravel())
        bell_1d  = torch.tensor(bell_mass.ravel())
        if not i:
            all_pred =  pred_1d[where_ctr[i]]
            all_truth = truth_1d[where_ctr[i]]
            all_bell = bell_1d[where_ctr[i]]
        else:
            all_pred = torch.cat([all_pred,pred_1d[where_ctr[i]]])
            all_truth = torch.cat([all_truth,truth_1d[where_ctr[i]]])
            all_bell = torch.cat([all_bell,bell_1d[where_ctr[i]]])

        VT_tru = truth_1d[where_ctr[i]]
        VT_pred = pred_1d[where_ctr[i]]
        VT_diff_temp = pred_1d[where_ctr[i]]-truth_1d[where_ctr[i]]
        VT_diff_std ,VT_diff_mean = torch.std_mean(VT_diff_temp)
        VT_diff.append([(10**VT_tru).mean(), (10**VT_pred).mean(),VT_diff_mean.numpy(), VT_diff_std.numpy()])
        #if i%200 ==0:
            #print(i)

    #gmass[gmass==0]=1

    gmass = np.array(gmass)
    gmass = gmass.reshape([len(gmass),3])
    bad = np.argwhere(gmass[:,0]==0).ravel()
    bin_num = np.array(bin_num)
    bin_num = bin_num.ravel()
    VT_diff = np.array(VT_diff)
    VT_diff[bad] = 0
    print(i)
    print(torch.std_mean(all_pred-all_truth))
    print()
    return all_truth, all_pred, all_bell, gmass, VT_diff