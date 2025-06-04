import numpy as np
import pandas as pd
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




id_list = np.loadtxt('./catalog/dr17_id.txt',dtype=str)
test_id = id_list[int(0.8*len(id_list)):]
NSAid_list = np.loadtxt('./catalog/NSAid_list.txt').astype(int)

class CNN_result():
    N_FIGS = 9
    def __init__(self,model, index, data ,mode='test', do_scramble = False):
        self.model = model
        self.index = index
        self.mode = mode
        self.device = data.device
        if mode == 'all':
            self.manga_id = id_list[index]
            self.x = data[index:(index+1),:,:,:]

        if mode == 'test':
            self.test_set = data[int(0.8*len(data)):,:,:,:].to(self.device)
            self.manga_id = test_id[index]
            self.x= (self.test_set[index:(index+1),:,:,:].clone()).to(self.device)
            hdu_bin = fits.open('./MaNGA_train/bin_id/'+str(self.manga_id)+'_bin_id.fits',dtype=np.float32)
            self.bin_id = torch.tensor(hdu_bin[0].data.astype(np.float32)).reshape([1,1,100,100])
        
        if do_scramble:
            mask = torch.clone(self.x[0,(self.N_FIGS-2),:,:]).to(torch.bool).to(self.x.device)
            self.scramble_idx = torch.randperm(len(self.x[0,0][mask]))
            self.x = self.scramble(self.x)  #scramble the data
            try:
                self.bin_id = self.scramble(self.bin_id) #scramble the VT_bin
            except:
                pass
        
    def get_img(self, mode):
        temp = torch.clone(self.x[:,:(self.N_FIGS-1),:,:]).to(self.device)
        if mode == 'pred':
            img = (self.model(temp)[0]*temp[:,(self.N_FIGS-2):(self.N_FIGS-1),:,:]).to('cpu').detach().numpy()
        if mode == 'truth':
            img = self.x[:,(self.N_FIGS-1):,:,:].to('cpu').detach().numpy()

        return img[0][0]
    
    def VT_bined(self, _image, mask = None):
        bin_id = torch.clone(self.bin_id)
        #bin_id = bin_id[scramble_idx]
        bin_id = bin_id.ravel().numpy()
        bin_inverse = np.unique(bin_id, return_inverse = True)[1]
        bin_count = np.unique(bin_id, return_counts = True)[1]
        image_1d = _image.ravel()
        image_VT = np.zeros([len(bin_id)])-1
        
        for i in range(1,len(bin_count)):
            if mask is not None:
                mask_1d = mask.ravel()
                _sum = np.sum(10**(image_1d*mask_1d)[bin_inverse==i])
                bin_count[i]-=(~mask_1d[bin_inverse == i]).sum()
            else:
                _sum = np.sum(10**image_1d[bin_inverse==i])
            
            try:
                image_VT[bin_inverse==i]=np.log10(_sum/bin_count[i])
            except:
                image_VT = -1
        image_VT = image_VT.reshape(_image.shape)
        #image_VT*=self.x[0,(self.N_FIGS-2),:,:].to('cpu').numpy()
        return image_VT
    
    def VT_lize(self):
        temp = torch.clone(self.x)
        mask = temp[0,(self.N_FIGS-2),:,:]
        #temp[:,:3,:,:] = (temp[:,:3,:,:])*test_set[index:(index+1),4:5,:,:] #overall 
        #temp[:,0,:,:] = (temp[:,0,:,:]+0)*test_set[index:(index+1),(N_FIGS-2),:,:] #g band
        #temp[:,1,:,:] = (temp[:,1,:,:]+0)*test_set[index:(index+1),(N_FIGS-2),:,:] #r band
        #temp[:,2,:,:] = (temp[:,2,:,:]+0.5)*test_set[index:(index+1),(N_FIGS-2),:,:] #i band
        temp[:,5,:,:] = (temp[:,5,:,:])
        temp[:,:5,:,:] = temp[:,:5,:,:]*mask-10*(1-mask)
        mass_tru = self.get_img('truth')
        self.mass_pred = self.get_img('pred')
        
        mass_pred_VT = self.VT_bined(self.mass_pred)
        mass_tru[mass_tru<-10]=0
        self.mass_tru = self.VT_bined(mass_tru)
        mass_pred_VT = mass_pred_VT.reshape(self.mass_pred.shape)
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
    
    def get_bands_mask(self, band_indexs):
        mask = torch.ones([1,1,100,100]).to(self.x.device)
        for index in band_indexs:
            mask*=(self.x[0:1,index:(index+1)]>-9)
        return (mask).bool().to('cpu').numpy()

    def scramble(self,temp):
        mask = torch.clone(self.x[:,(self.N_FIGS-2),:,:]).to('cpu').bool().numpy().ravel()
        scramble = torch.ones(temp.shape)
        for idx in range(temp.shape[1]):
            map = temp[:,idx,:,:].ravel()
            map[mask] = map[mask][self.scramble_idx]
            scramble[:,idx,:,:] = map.reshape(temp[:,idx,:,:].shape)
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
    p90_tsam = torch.tensor(p90_sam).to(test_set.device)

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
