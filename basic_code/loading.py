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



def loading(mass_m = 'Mastar', #The model for the stellar mass
            N_FIGS = 9,  # The number of layers in our dataset. Basically no need to change that.
            device = torch.device(0 if torch.cuda.is_available() else 'cpu'), # Device 
            no_z = False):
    if mass_m == 'Mastar':
        Data = torch.load('./catalog/dr17_mastar.pt',map_location=device)
        if no_z:
            from basic_code.CNN_no_z import CNN_noVT
        else:
            from basic_code.CNN_noVT_Mastar import CNN_noVT
        model = CNN_noVT(in_chan=8 , mid=99 , out = 1,
                kernel_size = 3,
                n_layers  =4).to(device)
        model.load_state_dict(torch.load('./models/Mastar_noVT.pt',map_location=device))

    if mass_m == 'MILES':
        Data = torch.load('./catalog/dr17_firefly_.pt',map_location=device)
        Data[:,8,:,:] = Data[:,8,:,:]+np.log10(1/0.25)
        if no_z:
            from basic_code.CNN_no_z import CNN_noVT
        else:
            from basic_code.CNN_noVT_MILES import CNN_noVT
        model = CNN_noVT(in_chan=8 , mid=35 , out = 1,
                kernel_size = 11,
                n_layers  =3).to(device)
        model.load_state_dict(torch.load('./models/MILES_noVT.pt',map_location=device))
        
    if mass_m == 'PCA':
        Data = torch.load('./catalog/dr17_PCA.pt',map_location=device)
        if no_z:
            from basic_code.CNN_no_z import CNN_noVT
        else:
            from basic_code.CNN_noVT_Mastar import CNN_noVT
        model = CNN_noVT(in_chan=8 , mid=52 , out = 1,
                kernel_size = 5,
                n_layers  =3).to(device)
        model.load_state_dict(torch.load('./models/PCA_noVT.pt',map_location=device))
    
    return model, Data