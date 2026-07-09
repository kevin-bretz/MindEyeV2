#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time 
from subprocess import call
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import copy

from tqdm import tqdm
from PIL import Image
from sklearn.preprocessing import StandardScaler

import webdataset as wds
import sys

# nsd_access is from this repo: https://github.com/tknapen/nsd_access
# also see https://cvnlab.slite.page/p/dC~rBTjqjb/How-to-get-the-data for how to download the NSD data!
from nsd_access import NSDAccess
nsd_path = '/scratch/gpfs/KNORMAN/natural-scenes-dataset'
nsda = NSDAccess(nsd_path)

import nibabel as nib

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:",device)


# In[ ]:


tmp = '/scratch/gpfs/KNORMAN'
shared1000 = np.load("shared1000.npy") # download from https://huggingface.co/datasets/pscotti/mindeyev2/tree/main


# In[1]:


for sub in [0]: #,1,2,3,4,5,6,7]:
    subject=f'subj0{sub+1}'
    subj=subject
    print(subject)

    abs_cnt = -1
    abs_notshared1000_cnt = -1
    abs_shared1000_cnt = -1

    # load coco 73k indices
    indices_path = "COCO_73k_subj_indices.hdf5"
    hdf5_file = h5py.File(indices_path, "r")
    indices = hdf5_file[f"{subj}"][:]

    nsessions_allsubj=np.array([40, 40, 32, 30, 40, 32, 40, 30]) 
    nsessions=nsessions_allsubj[sub];
    ntrials = 750*nsessions
    print(nsessions,ntrials)

    print(time.strftime("\nCurrent time: %H:%M:%S", time.localtime())) 

    file = f"/scratch/gpfs/KNORMAN/natural-scenes-dataset/nsddata/ppdata/{subject}/func1pt8mm/roi/nsdgeneral.nii.gz"
    nifti = nib.load(file) 
    mask = nifti.get_data()
    mask[mask<1] = 0 
    nsdgeneral_mask = mask

    for tar in tqdm(range(nsessions)):
        sess=tar+1

        behav = nsda.read_behavior(subject=subject, 
                    session_index=sess, 
                    trial_index=[]) 

        # pull single-trial betas and mask them
        betas = nsda.read_betas(subject=subject, 
                            session_index=sess, 
                            trial_index=[], # empty list as index means get all for this session
                            data_type='betas_fithrf', # GLMSingle beta2
                            data_format='func1pt8mm') 

        # betas = betas[mask]
        betas = np.moveaxis(betas,-1,0)

        vox_include = copy.deepcopy(nsdgeneral_mask)
        ncsnr = nib.load(f"{subject}_ncsnr.nii.gz").get_fdata()
        ncsnr[ncsnr<.15] = np.nan 
        if tar==0: print("voxels left:", len(vox_include[vox_include>0]))
        vox_include[np.isnan(ncsnr)] -= 1 # keep all nsdgeneral voxels even if they are below the threshold
        vox_include[vox_include<0] = 0
        if tar==0: print("voxels left after ncsnr thresholding:", len(vox_include[vox_include>0])) # subj01 = 49329

        betas = betas.reshape(len(betas),-1)
        betas = betas[:,vox_include.flatten().astype(bool)]
        shape = betas.shape
        scalar = StandardScaler(with_mean=True, with_std=True).fit(betas) # YOU SHOULD EXCLUDE SHARED1000 FROM THIS (NOT DONE HERE BUT DONE IN ACTUAL MINDEYE2 PAPER)
        betas_mean = scalar.mean_
        betas_std = scalar.scale_
        betas = (betas - betas_mean) / betas_std
        betas = betas.reshape(shape).astype('float16') # (1, 15724)    

        globals()[f'betas_ses{sess}'] = betas  
        globals()[f'behav_ses{sess}'] = behav   
        print(betas.shape)

    for tar in range(nsessions):
        sess=tar+1

        if sess==1:
            betas_all = globals()[f'betas_ses{sess}']
        else:
            betas_all = np.vstack((betas_all,globals()[f'betas_ses{sess}']))
        print(betas_all.shape)

    with h5py.File(f'betas_{subject}.hdf5', 'w') as f:
        f.create_dataset('betas', data=betas_all)
    print(f"saved betas_{subject}.hdf5")

    os.makedirs(f"{tmp}/mindeyev2_wds/{subj}",exist_ok=True)
    os.makedirs(f"{tmp}/mindeyev2_wds/{subj}/train",exist_ok=True)
    os.makedirs(f"{tmp}/mindeyev2_wds/{subj}/test",exist_ok=True)
    sink1 = wds.TarWriter(f"{tmp}/mindeyev2_wds/{subj}/test/0.tar")
    for tar in tqdm(range(nsessions)):
        behav = globals()[f'behav_ses{tar+1}']

        sink2 = wds.TarWriter(f"{tmp}/mindeyev2_wds/{subj}/train/{tar}.tar")
        for i in range(len(behav)):
            abs_cnt += 1                

            trial_numbers = np.where(indices==indices[abs_cnt])[0]
            assert np.isin(abs_cnt,trial_numbers)
            trial_numbers[trial_numbers == abs_cnt] = -1 # current trial becomes negative 1
            if len(trial_numbers) == 1:
                trial_numbers = np.append(trial_numbers, -1)
                trial_numbers = np.append(trial_numbers, -1)
            if len(trial_numbers) == 2:
                trial_numbers = np.append(trial_numbers, -1)
            assert len(trial_numbers) == 3

            sess=tar+1
            behav = globals()[f'behav_ses{sess}']
            behav_matrix = np.ones((1, 17))*-1
            jjj=-1
            for j in range(1):
                jj = i-j
                jjj += 1

                if jj >= 0:
                    # change NaNs to negative-one integers
                    iscorrect = behav.iloc[jj]['ISCORRECT']
                    if np.isnan(iscorrect): iscorrect = -1

                    isoldcurrent = behav.iloc[jj]['ISOLDCURRENT']
                    if np.isnan(isoldcurrent): isoldcurrent = -1

                    iscorrectcurrent = behav.iloc[jj]['ISCORRECTCURRENT']
                    if np.isnan(iscorrectcurrent): iscorrectcurrent = -1

                    rt = behav.iloc[jj]['RT']
                    if np.isnan(rt): rt = -1

                    changemind = behav.iloc[jj]['CHANGEMIND']
                    if np.isnan(changemind): changemind = -1

                    button = behav.iloc[jj]['BUTTON']
                    if np.isnan(button): button = -1

                    total1 = behav.iloc[jj]['TOTAL1']
                    if np.isnan(total1): total1 = -1

                    total2 = behav.iloc[jj]['TOTAL2']
                    if np.isnan(total2): total2 = -1

                    coco73 = int(behav.iloc[jj]['73KID'])-1
                    assert coco73 >= 0 and coco73 < 730000

                    behavior = {
                        "cocoidx": coco73, #0
                        "subject": sub+1,                          #1
                        "session": int(behav.iloc[jj]['SESSION']), #2
                        "run": int(behav.iloc[jj]['RUN']),         #3
                        "trial": int(behav.iloc[jj]['TRIAL']),     #4
                        "global_trial": (int(behav.iloc[jj]['SESSION'])-1)*750 + jj,        #5
                        "time": int(behav.iloc[jj]['TIME']),       #6
                        "isold": int(behav.iloc[jj]['ISOLD']),     #7
                        "iscorrect": iscorrect,                    #8
                        "rt": rt, # 0 = no RT                      #9
                        "changemind": changemind,                  #10
                        "isoldcurrent": isoldcurrent,              #11
                        "iscorrectcurrent": iscorrectcurrent,      #12
                        "total1": total1,   #13
                        "total2": total2,   #14
                        "button": button,                          #15
                        "shared1000": shared1000[int(behav.iloc[jj]['73KID'])-1], #16
                    }

                    assert (int(behav.iloc[jj]['SESSION'])-1)*750 + jj >= 0
                    assert (int(behav.iloc[jj]['SESSION'])-1)*750 + jj < 27750

                    behav_matrix[jjj] = np.array(list(behavior.values()))

            past_behav_matrix = np.ones((15, 17))*-1
            jjj=-1
            for j in range(1,16):
                jj = i-j
                jjj += 1

                if jj >= 0:
                    # change NaNs to negative-one integers
                    iscorrect = behav.iloc[jj]['ISCORRECT']
                    if np.isnan(iscorrect): iscorrect = -1

                    isoldcurrent = behav.iloc[jj]['ISOLDCURRENT']
                    if np.isnan(isoldcurrent): isoldcurrent = -1

                    iscorrectcurrent = behav.iloc[jj]['ISCORRECTCURRENT']
                    if np.isnan(iscorrectcurrent): iscorrectcurrent = -1

                    rt = behav.iloc[jj]['RT']
                    if np.isnan(rt): rt = -1

                    changemind = behav.iloc[jj]['CHANGEMIND']
                    if np.isnan(changemind): changemind = -1

                    button = behav.iloc[jj]['BUTTON']
                    if np.isnan(button): button = -1

                    total1 = behav.iloc[jj]['TOTAL1']
                    if np.isnan(total1): total1 = -1

                    total2 = behav.iloc[jj]['TOTAL2']
                    if np.isnan(total2): total2 = -1

                    coco73 = int(behav.iloc[jj]['73KID'])-1
                    assert coco73 >= 0 and coco73 < 730000

                    behavior = {
                        "cocoidx": coco73, #0
                        "subject": sub+1,                          #1
                        "session": int(behav.iloc[jj]['SESSION']), #2
                        "run": int(behav.iloc[jj]['RUN']),         #3
                        "trial": int(behav.iloc[jj]['TRIAL']),     #4
                        "global_trial": (int(behav.iloc[jj]['SESSION'])-1)*750 + jj,        #5
                        "time": int(behav.iloc[jj]['TIME']),       #6
                        "isold": int(behav.iloc[jj]['ISOLD']),     #7
                        "iscorrect": iscorrect,                    #8
                        "rt": rt, # 0 = no RT                      #9
                        "changemind": changemind,                  #10
                        "isoldcurrent": isoldcurrent,              #11
                        "iscorrectcurrent": iscorrectcurrent,      #12
                        "total1": total1,   #13
                        "total2": total2,   #14
                        "button": button,                          #15
                        "shared1000": shared1000[int(behav.iloc[jj]['73KID'])-1], #16
                    }

                    assert (int(behav.iloc[jj]['SESSION'])-1)*750 + jj >= 0
                    assert (int(behav.iloc[jj]['SESSION'])-1)*750 + jj < 27750

                    past_behav_matrix[jjj] = np.array(list(behavior.values()))

            future_behav_matrix = np.ones((15, 17))*-1
            jjj=-1
            for j in range(1,16):
                jj = i+j
                jjj += 1

                if jj >= 0 and jj<750:
                    # change NaNs to negative-one integers
                    iscorrect = behav.iloc[jj]['ISCORRECT']
                    if np.isnan(iscorrect): iscorrect = -1

                    isoldcurrent = behav.iloc[jj]['ISOLDCURRENT']
                    if np.isnan(isoldcurrent): isoldcurrent = -1

                    iscorrectcurrent = behav.iloc[jj]['ISCORRECTCURRENT']
                    if np.isnan(iscorrectcurrent): iscorrectcurrent = -1

                    rt = behav.iloc[jj]['RT']
                    if np.isnan(rt): rt = -1

                    changemind = behav.iloc[jj]['CHANGEMIND']
                    if np.isnan(changemind): changemind = -1

                    button = behav.iloc[jj]['BUTTON']
                    if np.isnan(button): button = -1

                    total1 = behav.iloc[jj]['TOTAL1']
                    if np.isnan(total1): total1 = -1

                    total2 = behav.iloc[jj]['TOTAL2']
                    if np.isnan(total2): total2 = -1

                    coco73 = int(behav.iloc[jj]['73KID'])-1
                    assert coco73 >= 0 and coco73 < 730000

                    behavior = {
                        "cocoidx": coco73, #0
                        "subject": sub+1,                          #1
                        "session": int(behav.iloc[jj]['SESSION']), #2
                        "run": int(behav.iloc[jj]['RUN']),         #3
                        "trial": int(behav.iloc[jj]['TRIAL']),     #4
                        "global_trial": (int(behav.iloc[jj]['SESSION'])-1)*750 + jj,        #5
                        "time": int(behav.iloc[jj]['TIME']),       #6
                        "isold": int(behav.iloc[jj]['ISOLD']),     #7
                        "iscorrect": iscorrect,                    #8
                        "rt": rt, # 0 = no RT                      #9
                        "changemind": changemind,                  #10
                        "isoldcurrent": isoldcurrent,              #11
                        "iscorrectcurrent": iscorrectcurrent,      #12
                        "total1": total1,   #13
                        "total2": total2,   #14
                        "button": button,                          #15
                        "shared1000": shared1000[int(behav.iloc[jj]['73KID'])-1], #16
                    }

                    assert (int(behav.iloc[jj]['SESSION'])-1)*750 + jj >= 0
                    assert (int(behav.iloc[jj]['SESSION'])-1)*750 + jj < 27750

                    future_behav_matrix[jjj] = np.array(list(behavior.values()))

            olds_behav_matrix = np.ones((3, 17))*-1
            jjj=-1
            for j in range(3):
                jj = trial_numbers[j]

                if jj>=0:
                    jjj += 1
                    old_session = int(np.floor(jj / 750)) + 1
                    old_trial = jj % 750
                    behav = globals()[f'behav_ses{old_session}']
                    jj = old_trial

                    # change NaNs to negative-one integers
                    iscorrect = behav.iloc[jj]['ISCORRECT']
                    if np.isnan(iscorrect): iscorrect = -1

                    isoldcurrent = behav.iloc[jj]['ISOLDCURRENT']
                    if np.isnan(isoldcurrent): isoldcurrent = -1

                    iscorrectcurrent = behav.iloc[jj]['ISCORRECTCURRENT']
                    if np.isnan(iscorrectcurrent): iscorrectcurrent = -1

                    rt = behav.iloc[jj]['RT']
                    if np.isnan(rt): rt = -1

                    changemind = behav.iloc[jj]['CHANGEMIND']
                    if np.isnan(changemind): changemind = -1

                    button = behav.iloc[jj]['BUTTON']
                    if np.isnan(button): button = -1

                    total1 = behav.iloc[jj]['TOTAL1']
                    if np.isnan(total1): total1 = -1

                    total2 = behav.iloc[jj]['TOTAL2']
                    if np.isnan(total2): total2 = -1

                    coco73 = int(behav.iloc[jj]['73KID'])-1
                    assert coco73 >= 0 and coco73 < 730000

                    behavior = {
                        "cocoidx": coco73, #0
                        "subject": sub+1,                          #1
                        "session": int(behav.iloc[jj]['SESSION']), #2
                        "run": int(behav.iloc[jj]['RUN']),         #3
                        "trial": int(behav.iloc[jj]['TRIAL']),     #4
                        "global_trial": (int(behav.iloc[jj]['SESSION'])-1)*750 + jj,        #5
                        "time": int(behav.iloc[jj]['TIME']),       #6
                        "isold": int(behav.iloc[jj]['ISOLD']),     #7
                        "iscorrect": iscorrect,                    #8
                        "rt": rt, # 0 = no RT                      #9
                        "changemind": changemind,                  #10
                        "isoldcurrent": isoldcurrent,              #11
                        "iscorrectcurrent": iscorrectcurrent,      #12
                        "total1": total1,   #13
                        "total2": total2,   #14
                        "button": button,                          #15
                        "shared1000": shared1000[int(behav.iloc[jj]['73KID'])-1], #16
                    }

                    assert (int(behav.iloc[jj]['SESSION'])-1)*750 + jj >= 0
                    assert (int(behav.iloc[jj]['SESSION'])-1)*750 + jj < 27750

                    olds_behav_matrix[jjj] = np.array(list(behavior.values()))

            behav = globals()[f'behav_ses{sess}']
            # Check if this is a shared1000 trial
            if shared1000[int(behav.iloc[i]['73KID'])-1]:
                abs_shared1000_cnt += 1
            else:
                abs_notshared1000_cnt += 1

            with torch.no_grad(): #https://cvnlab.slite.page/p/fRv4lz5V2F/Untitled
                if shared1000[int(behav.iloc[i]['73KID'])-1]:
                    sink1.write({
                        "__key__": "sample%09d" % abs_shared1000_cnt,
                        "behav.npy": behav_matrix,
                        "past_behav.npy": past_behav_matrix,
                        "future_behav.npy": future_behav_matrix,
                        "olds_behav.npy": olds_behav_matrix,
                    })
                    assert behav_matrix[-1,0] < 73000
                else:
                    sink2.write({
                        "__key__": "sample%09d" % abs_notshared1000_cnt,
                        "behav.npy": behav_matrix,
                        "past_behav.npy": past_behav_matrix,
                        "future_behav.npy": future_behav_matrix,
                        "olds_behav.npy": olds_behav_matrix,
                    })
                    assert behav_matrix[-1,0] < 73000
        sink2.close()
    sink1.close()

    print(time.strftime("\nCurrent time: %H:%M:%S", time.localtime())) 

