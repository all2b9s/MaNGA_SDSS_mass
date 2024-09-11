import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm as tqdm
import cv2
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import torch 
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

from astropy.io import fits
from matplotlib import pyplot as plt

import seaborn as sns

device=torch.device(0 if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
print(device)


id_list = np.loadtxt('./catalog/dr17_id.txt',dtype=str)
test_id = id_list[int(0.8*len(id_list)):]
NSAid_list = np.loadtxt('./NSAid_list.txt').astype(int)

class CNN_result():
    N_FIGS = 9
    def __init__(self,model, index, data ,mode='test'):
        self.model = model
        self.index = index
        self.mode = mode
        if mode == 'all':
            self.manga_id = id_list[index]
            self.x = data[index:(index+1),:,:,:]

        if mode == 'test':
            self.test_set = data[int(0.8*len(data)):,:,:,:].to(device)
            self.manga_id = test_id[index]
            self.x= (self.test_set[index:(index+1),:,:,:].clone())
        
    def get_img(self, mode):
        temp = self.x[:,:(self.N_FIGS-1),:,:]
        if mode == 'pred':
            img = (self.model(temp)[0]*self.x[:,(self.N_FIGS-2):(self.N_FIGS-1),:,:]).to('cpu').detach().numpy()
        if mode == 'truth':
            img = self.x[:,(self.N_FIGS-1):,:,:].to('cpu').detach().numpy()

        return img[0][0]
    
    def VT_bined(self, _image):
        #hdu_bin = fits.open('./MaNGA_train/bin_id/'+str(self.manga_id)+'_bin_id.fits',dtype=np.float32)
        #bin_id = torch.tensor(hdu_bin[0].data.astype(np.float32)).reshape([1,1,100,100])
        bin_id = self.bin_id.ravel()
        #bin_id = bin_id[scramble_idx]
        bin_id = bin_id.numpy()
        bin_count = np.unique(bin_id, return_counts = True)[1]
        bin_inverse = np.unique(bin_id, return_inverse = True)[1]
        image_1d = _image.ravel()
        image_VT = np.zeros([len(bin_id)])
        for i in range(len(bin_count)):
            _sum = np.sum(10**image_1d[bin_inverse==i])
            image_VT[bin_inverse==i]=np.log10(_sum/bin_count[i])
        image_VT = image_VT.reshape(_image.shape)
        return image_VT
    
    def VT_lize(self, do_scramble = False):
        temp = self.x
        mask = test_set[self.index:(self.index+1),(N_FIGS-2),:,:]
        #temp[:,:3,:,:] = (temp[:,:3,:,:])*test_set[index:(index+1),4:5,:,:] #overall 
        #temp[:,0,:,:] = (temp[:,0,:,:]+0)*test_set[index:(index+1),(N_FIGS-2),:,:] #g band
        #temp[:,1,:,:] = (temp[:,1,:,:]+0)*test_set[index:(index+1),(N_FIGS-2),:,:] #r band
        #temp[:,2,:,:] = (temp[:,2,:,:]+0.5)*test_set[index:(index+1),(N_FIGS-2),:,:] #i band
        temp[:,5,:,:] = (temp[:,5,:,:])
        temp[:,:5,:,:] = temp[:,:5,:,:]*mask-10*(1-mask)
        
        hdu_bin = fits.open('./MaNGA_train/bin_id/'+str(self.manga_id)+'_bin_id.fits',dtype=np.float32)
        self.bin_id = torch.tensor(hdu_bin[0].data.astype(np.float32)).reshape([1,1,100,100])
        if do_scramble:
            self.bin_id = self.scramble(self.bin_id) #scramble the VT_bin
            temp = self.scramble(temp, keep=1)  #scramble the data
        mass_tru = self.get_img('truth')
        self.mass_pred = self.get_img('pred')
        
        mass_pred_VT = self.VT_bined(self.mass_pred)
        mass_tru[mass_tru<-10]=0
        self.mass_tru = self.VT_bined(mass_tru)
        mass_pred_VT = mass_pred_VT.reshape(self.mass_pred.shape)
        #mass_pred_VT = mass_pred_VT-(-0.1*mass_pred_VT+0.82)
        self.mass_pred_VT = mass_pred_VT*temp[0,(self.N_FIGS-2),:,:].to('cpu').detach().numpy()
        return self.mass_pred_VT, self.mass_tru, self.manga_id

    def total_mass(self):
        mass_pred_VT = torch.tensor(10**self.mass_pred_VT)
        mass_pred_VT[mass_pred_VT==1] = 0
        gmass_pred = torch.sum(0.16*mass_pred_VT)

        mass_tru = torch.tensor(10**self.mass_tru)
        mass_tru[mass_tru==1] = 0
        gmass_tru = torch.sum(0.16*mass_tru)

        return gmass_pred, gmass_tru

    def scramble(self,temp,keep = 0):
        mask = self.x[:,(self.N_FIGS-2),:,:].to('cpu').bool().numpy().ravel()
        scramble = torch.ones(temp.shape)
        for idx in range(len(temp[0])):
            map = temp[:,idx,:,:].ravel()
            if not keep:
                self.scramble_idx = torch.randperm(len(map[mask]))
            map[mask] = map[mask][self.scramble_idx]
            scramble[:,idx,:,:] = map.reshape(temp.shape)
        return scramble
    
    def Gaussian(self, k_size=5, sigma = 2.5):
        for i in range(0,5):
            self.x[0,i:(i+1),:,:] = torch.tensor(self._gauss(self.x[0,i,:,:].to('cpu').numpy(), k_size, sigma))
            
    
    def _gauss(self, temp, kernel_size = 5, sigma=2.5):
        def gaussian_kernel(size, sigma):
            x, y = np.meshgrid(np.arange(-size // 2 + 1, size // 2 + 1), np.arange(-size // 2 + 1, size // 2 + 1))
            kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            return kernel / np.sum(kernel)
        gaussian_filter = gaussian_kernel(kernel_size, sigma)
        result = convolve(10**temp, gaussian_filter,mode='nearest')
        return np.log10(result)

def M_L(bands, a, b):
    #print(bands[0]-bands[1])
    return a+b*(bands[0]-bands[1])

def bands(index, band, Data):
    bands =  Data[int(len(Data)*0.8)+i,(band[0],band[1]),:,:].to('cpu').detach().numpy()
    bands = (10**bands).sum(axis=(1,2))
    bands = np.log10(bands)
    return bands

def p90_pred(index, model, test_set):
    manga_id = test_id[index]
    f_name = './Data/p90_image/'+manga_id+'.fits'
    hdulp90 = fits.open(f_name)
    image = hdulp90[0].data
    image[:,:,:,:5][image[:,:,:,:5]<=0]=10**-10
    image[:,:,:,:5] = np.log10(image[:,:,:,:5])
    redshift = test_set[index,5,:,:].max()

    p90_sam = np.zeros([1,8,len(image[0]),len(image[0])])
    for i in range(0,5):
        p90_sam[0,i,:,:] = image[0,:,:,i]*image[0,:,:,5]
    p90_sam[0,5,:,:] = np.ones([len(image[0]),len(image[0])])*redshift.to('cpu').numpy()
    p90_sam[0,7,:,:] = image[0,:,:,5]

    p90_sam = p90_sam.astype(np.float32)
    p90_tsam = torch.tensor(p90_sam).to(device)

    p90_pixel, p90_total = model(p90_tsam)
    p90_pixel = p90_pixel.to('cpu').detach().numpy()
    #plt.imshow(p90_pixel[0,0])
    return p90_pixel, p90_total.to('cpu').detach().numpy()+np.log10(0.16), image

def nsa_read(index, item = 'ELPETRO_FLUX'):
    manga_id = test_id[index]
    NSA_id = NSAid_list[index]
    mag = NSA_data[item][NSA_id]
    #print(mag)
    return mag