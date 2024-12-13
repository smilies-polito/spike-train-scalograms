import collections
import os
import numpy as np
from random import shuffle

from sklearn.model_selection import train_test_split

output_cwt_dir_PatchClampGouwens = '../../output/PatchClampGouwens/CWT/CWT_TwoSweeps_New_w_40and50'
output_cwt_dir_PatchSeqGouwens = '../../output/PatchSeqGouwens/CWT/CWT_TwoSweeps_New_w_PS_SupraThresh_40and50'
input_dir="."
output_models = '../output/Models'

categories = ['Pvalb', 'Excitatory', 'Vip/Lamp5', 'Sst']
data_dict = {}
print(os.getcwd())
print(os.path.isdir('../../output'))
with open(os.path.join(input_dir, 'cell_types_GouwensAll_new.txt'), 'r') as file:
    for line in file:
        # Split the line at the ':' character to separate key and value
        value, key = line.strip().split(' ')
        # Add the key-value pair to the dictionary
        data_dict[int(key)] = value
cells_list = list(data_dict.items())

available_ids_dict = collections.defaultdict(list)
available_ids_list = []
for i in range(len(cells_list)):
    single_cell = cells_list[i]
    cell_specimen_id = int(single_cell[0])
    cell_category = single_cell[1]
    if cell_category != 'Unsure':
        if os.path.isfile(os.path.join(output_cwt_dir_PatchClampGouwens, "cell_%d.npy" % cell_specimen_id)):
            available_ids_dict['%d, Clamp' % cell_specimen_id] = cell_category
            available_ids_list.append(cell_specimen_id)
        else:
            print('The cell %d is not in Patch Clamp, cells in Patch Seq will be looked' % cell_specimen_id)
            if os.path.isfile(os.path.join(output_cwt_dir_PatchSeqGouwens, "cell_%d.npy" % cell_specimen_id)):
                available_ids_dict['%d, Seq' % cell_specimen_id] = cell_category
                available_ids_list.append(cell_specimen_id)
            else:
                print("Cell %d does not exist in either Patch Clamp and Patch Seq" % cell_specimen_id)
                continue

            # print("Cell %d does not exist" %cell_specimen_id)
            continue
    else:
        continue
print('Formation of list is done, size: %d' % len(available_ids_list))


items = list(available_ids_dict.items())
shuffle(items)
available_ids_dict = dict(items)

all_classes = list(available_ids_dict.values())
all_ids = list(available_ids_dict.keys())

X_train, X_test, y_train_class_names, y_test_class_names = train_test_split(all_ids, all_classes, test_size=0.2,
                                                                            random_state=42, stratify=all_classes)
print(len(X_test))
# File path
file_path = "../../output/Test_split.txt"

# Save list to file
with open(file_path, "w") as file:
    for item in X_test:
        file.write(f"{item}\n")

print(f"Test List saved to {file_path}")

# File path
file_path = "../../output/Train_split.txt"

# Save list to file
with open(file_path, "w") as file:
    for item in X_train:
        file.write(f"{item}\n")

print(f"List saved to {file_path}")
