import numpy as np
import torch 
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import glob
import os
import urllib.request
import optuna
from optuna.trial import TrialState
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
# nohup python -u MaNGA_VT_train.py 3 >> ./logs/mastar_i_only.out 2>&1 &

#device=torch.device(1 if torch.cuda.is_available() else 'cpu')
#device_valid = torch.device('cpu')
device = torch.device(int(sys.argv[1]) if torch.cuda.is_available() else 'cpu')
print(device)
N_FIGS = 9 

Data = torch.load('./catalog/dr17_mastar.pt',map_location=device)
#Data[:,8,:,:] = Data[:,8,:,:]+np.log10(1/0.25)
#Data = Data[:,[1,3,4,5],:,:]
#Data = torch.tensor(Data)
normed_Data = torch.clone(Data)

train_set = Data[:int(0.7*len(Data)),:,:,:].to(device)
valid_set = Data[int(0.7*len(Data)):int(0.8*len(Data)),:,:,:].to(device)
test_set = Data[int(0.8*len(Data)):,:,:,:].to(device)


study_name = 'Mastar_i_only'
storage = 'sqlite:///./models/'+study_name+'.db'
bands = ['i']

def Conv2D(in_channels,out_channels,kernel_size,padding):
    conv = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride=1, padding = padding),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2)
        )
    return conv

class CNN_noVT(nn.Module):
    band_to_index = {'u': 0, 'g': 1, 'r': 2, 'i': 3, 'z': 4}
    
    def __init__(self, bands = ['u', 'g', 'r', 'i', 'z'], mid=64 , out = 1,
                kernel_size = 5,
                n_layers  =3):
        padding = int(kernel_size/2)
        super(CNN_noVT, self).__init__()
        self.mask = list(map(lambda band: self.band_to_index[band], bands))
        self.redshift = Conv2D(6,6,1,0)
        self.conv_1 =Conv2D(len(bands)+2, mid,kernel_size, padding = padding)
        conv = []
        #self.norm = nn.BatchNorm2d(3)
        for i in range(0,n_layers-1):
            conv.append(
                nn.Conv2d(in_channels = mid, out_channels = mid, kernel_size = kernel_size, stride=1, padding = padding),
            )
            conv.append(nn.ReLU())
            #conv.append(nn.BatchNorm2d(mid))
        self.conv = nn.Sequential(*conv)
        self.conv_f1 =Conv2D(mid,int(mid/2),kernel_size,padding = padding)
        self.conv_f2 =Conv2D(int(mid/2),out,kernel_size,padding = padding)
        #self.conv_f3 =Conv2D(out,out,1,padding = 0)
    def forward(self, x):
        #img = torch.clone(x[:,:6,:,:])
        #img[:,5,:,:]*=100
        img = torch.cat([x[:,self.mask,:,:],x[:,5:6,:,:]],dim = 1)
        img[:,-1,:,:]*=100
        VT = torch.clone(x[:,7:,:,:])
        #img = self.redshift(img)#*0.1+img[:,:5,:,:]
        
        x_ = torch.cat([img,VT],dim=1).clone()
        x_ = self.conv_1(x_)
        x_ = self.conv(x_)
        x_ = self.conv_f1(x_)
        x_ = self.conv_f2(x_)#-np.log10(0.25)
        #x_ = 0.1*self.conv_f3(x_)+x_
        #x_ = x_*filter
        return x_, torch.log10(torch.sum(10**x_))  
    
def objective(trial):
    ############################################################################
    num_layers = trial.suggest_int("num_layers", 3, 5)
    kernel_size = 2*trial.suggest_int("kernels_size/2", 0, 5)+1
    mid = trial.suggest_int("num_filters", 2**5, 2**7,log=True) 
    lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    #batch_size =  trial.suggest_int("batch_size", 1, 128,log=True)
    ############################################################################
    
    model = CNN_noVT(bands = bands, mid = mid,kernel_size=kernel_size,n_layers=num_layers).to(device)
    criterion                   = nn.MSELoss()
    optimizer                   = torch.optim.Adam(model.parameters(),lr=lr,weight_decay = weight_decay)
    num_epochs = 40
    batch_size =32
    
    dataset = torch.utils.data.TensorDataset(train_set[:,:(N_FIGS-1),:,:], train_set[:,(N_FIGS-1):,:,:])
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    valid_dataset = torch.utils.data.TensorDataset(valid_set[:,:(N_FIGS-1),:,:], valid_set[:,(N_FIGS-1):,:,:])
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=256, shuffle=True)
    
    min_valid = 1e40
    fout = './models/'+study_name+'_1.out'   
    fmodel = './models/'+study_name+'/'+study_name+'_%d.pt'%(trial.number)
    path = './models/'+study_name+'/'
    folder = os.path.exists(path)
    if not folder:                  
        os.makedirs(path)
    f = open(fout, 'a')
    f.write('Trial %d starts! \n' % (trial.number))
    n_count = 0
    for epoch in range(num_epochs):
        loss_mean = 0
        if n_count>=5:
            return min_valid
        for i, (x_batch, y_batch) in enumerate(train_loader):
            model = model.to(device)
            optimizer.zero_grad()
            z, mass_pred = model(x_batch)
            mass_tru = torch.log10(torch.sum(10**y_batch))
            loss = criterion(z,y_batch)+criterion(mass_pred,mass_tru)
            loss.backward()
            optimizer.step()
        #if epoch%10 ==0:
        loss_mean = loss_mean/2993
        with torch.no_grad():
            valid_loss = 0
            for i, (x_batch,y_batch) in enumerate(valid_loader):
                z, mass_pred = model(x_batch)
                mass_tru = torch.log10(torch.sum(10**y_batch))
                loss = criterion(z,y_batch)+criterion(mass_pred,mass_tru)
                valid_loss += loss
            valid_loss/=(i+1)
            #print(valid_loss)
            #test_ = model(test_set[:,:5,:,:])
            #test_loss = criterion(test_ , test_set[:,5:,:,:])
        
        if valid_loss<min_valid:
            n_count = 0  
            min_valid = valid_loss
            torch.save(model, fmodel)
        else:
            n_count+=1
        
        trial.report(min_valid,epoch)
        
        #print('%d %.5e %.5e '% (epoch,loss,valid_loss))
        f.write('%d %.5e %.5e \n' % (epoch,loss,valid_loss))
    f.close()
    torch.cuda.empty_cache()
    return min_valid

def main():
########################################################################################################
    n_trials       = 100  # set to None for infinite
    startup_trials = 40
########################################################################################################
    try:
        optuna.delete_study(study_name=study_name,storage= storage)
    except:
        pass
    sampler = optuna.samplers.TPESampler(n_startup_trials=startup_trials)
    study = optuna.create_study(direction="minimize",
                                study_name=study_name,
                                sampler=sampler,
                                storage=storage,
                                load_if_exists=True)
    #study.enqueue_trial({"num_filters": 256,"lr":5e-5,"batch_size":1})
    study.optimize(objective, n_trials=n_trials,timeout = 3600*48)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    print(df)
    
if __name__ == "__main__":
    main()