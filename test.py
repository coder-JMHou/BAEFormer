import h5py
import torch
import sys
from model import Net
import numpy as np
sys.path.append(r"C:\Users\89709\Desktop\pansharpening\zeroshot\experiment\Toolbox")
import scipy.io as sio

satellite = 'wv3'
device=torch.device('cuda:1')
ckpt = 'weights/'
model = Net(lms_features=8).to(device)
weight = torch.load(ckpt)
model.load_state_dict(weight)

file_path = 'E:/testing_wv3/test_wv3_multiExm1.h5'
dataset = h5py.File(file_path, 'r')
ms = np.array(dataset['ms'], dtype=np.float32) / 2047.0
lms = np.array(dataset['lms'], dtype=np.float32) / 2047.0
pan = np.array(dataset['pan'], dtype=np.float32) / 2047.0

ms = torch.from_numpy(ms).float().to(device)
lms = torch.from_numpy(lms).float().to(device)
pan = torch.from_numpy(pan).float().to(device)
for i in range(len(dataset['ms'])):
    model.eval()
    with torch.no_grad():
        out = model(pan[i:i+1], lms[i:i+1])
        I_SR = torch.squeeze(out * 2047).cpu().detach().numpy()  # HxWxC
        I_SR = I_SR.transpose(1, 2, 0)
        sio.savemat('./result/' + satellite + '_reduced_result_'+str(i) + '.mat',
                    {'I_SR': I_SR})