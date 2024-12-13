# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 14:47:52 2023

@author: user
"""
import numpy as np
import matplotlib.pyplot as plt
import collections
from keras.models import Sequential
from keras.layers import TimeDistributed, Dense, Input, Flatten
from keras.layers import ConvLSTM2D, BatchNormalization, MaxPooling3D
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from random import shuffle
import tensorflow as tf
import os
import sys

output_cwt_dir_PatchClampGouwens = '../output/PatchClampGouwens/CWT/TwoSweeps'
output_cwt_dir_PatchSeqGouwens = '../output/PatchSeqGouwens/CWT/TwoSweeps'
output_dir = '../source/classification'
input_dir = './classification'



    
def main():
    #print(tf.config.list_physical_devices('GPU'))    
    with open(output_dir+"/Processed_cell_labels.txt", "w") as file:
        pass #create empty txt file each time the code runs
    categories = ['Pvalb', 'Excitatory', 'Vip/Lamp5', 'Sst']
    data_dict = {}   
    with open(os.path.join(input_dir,'cell_types_GouwensAll.txt'), 'r') as file:
        for line in file:
            # Split the line at the ':' character to separate key and value
            value, key = line.strip().split(' ')
            # Add the key-value pair to the dictionary
            data_dict[int(key)] = value
    cells_list = list(data_dict.items())
    available_ids_dict = collections.defaultdict(list)
    available_ids_list = []
    with open(output_dir +"/Processed_cell_labels.txt", "a") as file: # "a" stand for append mode
        for i in range(len(cells_list)):
            single_cell = cells_list[i]
            cell_specimen_id = int(single_cell[0])
            cell_category = single_cell[1]
            if cell_category != 'Unsure':
                try:
                    cell_cwt = np.load(os.path.join(output_cwt_dir_PatchClampGouwens,"cell_%d.npy" %cell_specimen_id))
                    available_ids_dict['%d' %cell_specimen_id].append(cell_category)
                    available_ids_list.append(cell_specimen_id)
                    file.write("%s %d \n" %(cell_category, cell_specimen_id))
                except:
                    print('The cell %d is not in Patch Clamp, cells in Patch Seq will be looked' % cell_specimen_id)
                    try:
                         cell_cwt = np.load(os.path.join(output_cwt_dir_PatchSeqGouwens,"cell_%d.npy" %cell_specimen_id))
                         available_ids_dict['%d' %cell_specimen_id].append(cell_category)
                         available_ids_list.append(cell_specimen_id)
                         file.write("%s %d \n" %(cell_category, cell_specimen_id))
                    except:
                         print("Cell %d does not exist in either Patch Clamp and Patch Seq" %cell_specimen_id)
                         continue
                    
                    
                    #print("Cell %d does not exist" %cell_specimen_id)
                    continue
            else:
                continue
    
    
if __name__ == '__main__':
    print('Launching main')
    main()
