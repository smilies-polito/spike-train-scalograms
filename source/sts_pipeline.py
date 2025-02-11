import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import collections
from random import shuffle
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from skopt import BayesSearchCV
from skopt.space import Categorical, Real, Integer
from sklearn.base import BaseEstimator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score
import pickle
import wandb

#setting seeds for reproducibility
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
np.random.seed(42)
import random
random.seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = False


output_cwt_dir_PatchClampGouwens = os.path.join("..","data", "PatchClampGouwens_CWT")
output_cwt_dir_PatchSeqGouwens = os.path.join("..","data","PatchSeqGouwens_CWT")
input_dir = os.path.join('..', 'data')
input_dir_split=os.path.join('..', "data","Split")
output_models = os.path.join('..',"output", "Models")
best_models_path=os.path.join('..', 'models')
epochs = 1000
#epochs = 1

"""
param_space = {
        'batch_size': Categorical([64, 128]),
        #'learning_rate':{
        #   "values": [0.0001, 0.001]}, # Values within the specified range
        'learning_rate': Real(0.0001, 0.001, prior='log-uniform'),
        'lr_decay': Categorical([0, 0.01]), #lr_decay= 0 means no decay
        'lr_decay_period': Categorical([10]),
        'early_stopping': Categorical([10]),
        'dropout_rate': Categorical([0, 0.3, 0.5, 0.7])
        }


#Just to check whether the code properly runs w/o encountering errors
param_space = {
        'batch_size': Categorical([64, 128]),
        #'learning_rate': Real(0.0001, 0.001, prior='log-uniform'), 
        'learning_rate': Categorical([0.001]),
         'lr_decay': Categorical([0.01]), #lr_decay= 0 means no decay
        'lr_decay_period': Categorical([10]),
        'early_stopping': Categorical([10]),
        'dropout_rate': Categorical([0, 0.5])
        #'dropout_rate': Real(0.0, 1.0)  # Continuous range between 0 and 1
        #'dropout_rate': Categorical([0, 0.3, 0.5, 0.7])
        }
"""

def generate_arrays_for_prediction(available_ids_list, batch_size):
    num_classes = 4
    batch_data = []
    idx = 1
    for id in available_ids_list:
        cell_dir_PatchSeqGouwens = os.path.join(output_cwt_dir_PatchSeqGouwens,"cell_%d.npy" %id)
        cell_dir_PatchClampGouwens = os.path.join(output_cwt_dir_PatchClampGouwens,"cell_%d.npy" %id)
        if os.path.exists(cell_dir_PatchSeqGouwens):
            cell_cwt = np.load(os.path.join(cell_dir_PatchSeqGouwens))
        elif os.path.exists(cell_dir_PatchClampGouwens):
            cell_cwt = np.load(os.path.join(cell_dir_PatchClampGouwens))

        col_size = 4800 
        

        if cell_cwt.shape == (180,col_size, 2):
            # convert it to tensorflow
            #tf_cell_cwt = tf.convert_to_tensor(cell_cwt)
            if idx < batch_size:
                batch_data.append(cell_cwt)
                idx+=1
            else:
                batch_data.append(cell_cwt)
                yield (torch.Tensor(np.array(batch_data)))

                batch_data = []
                idx=1


        else:
            print('Cell %d has size mismatch size: %s' %(id, str(cell_cwt.shape)))
            continue

        if id == available_ids_list[-1] and len(batch_data) != 0:
            yield (torch.Tensor(np.array(batch_data)))

def load_cell_for_interpretability(cell_id, device, size):
    batch_data=[]
    cell_dir_PatchSeqGouwens = os.path.join(output_cwt_dir_PatchSeqGouwens, "cell_%d.npy" % cell_id)
    cell_dir_PatchClampGouwens = os.path.join(output_cwt_dir_PatchClampGouwens, "cell_%d.npy" % cell_id)
    if os.path.exists(cell_dir_PatchSeqGouwens):
        cell_cwt = np.load(os.path.join(cell_dir_PatchSeqGouwens))
    elif os.path.exists(cell_dir_PatchClampGouwens):
        cell_cwt = np.load(os.path.join(cell_dir_PatchClampGouwens))

    col_size = 4800

    if cell_cwt.shape == (180, col_size, 2):
        batch_data.append(cell_cwt)
        cwt_batch=(torch.Tensor(np.array(batch_data)))
        # # Move data to the same device
        cwt_batch = cwt_batch.to(device)
        # The model expects input tensors with the shape
        # [batch_size, channels, pixels_x, pixels_y]
        cwt_batch  = cwt_batch.permute(0, 3, 1, 2)

        resized_batch_data = F.interpolate(cwt_batch, size=size, mode='bilinear', align_corners=False)
        # Add a third channel by replicating the first channel
        added_channel = resized_batch_data[:, :1, :, :]  # Taking the first channel
        resized_batch_data = torch.cat((resized_batch_data, added_channel), dim=1)  # Now (1, 3, 224, 224)
        return resized_batch_data
    else:
        print('Cell %d has size mismatch size: %s' % (id, str(cell_cwt.shape)))
        return



#This function generates list of y_test with the consideration of Cell size mismatch
def generate_list_for_y_test(cell_ids, cell_labels):
    y_test_class_names = []
    for id, category in zip(cell_ids, cell_labels):
        cell_dir_PatchSeqGouwens = os.path.join(output_cwt_dir_PatchSeqGouwens,"cell_%d.npy" %id)
        cell_dir_PatchClampGouwens = os.path.join(output_cwt_dir_PatchClampGouwens,"cell_%d.npy" %id)
        if os.path.exists(cell_dir_PatchSeqGouwens):
            cell_cwt = np.load(os.path.join(cell_dir_PatchSeqGouwens))
        elif os.path.exists(cell_dir_PatchClampGouwens):
            cell_cwt = np.load(os.path.join(cell_dir_PatchClampGouwens))
        col_size = 4800
        if cell_cwt.shape == (180,col_size, 2):
            y_test_class_names.append(category)
        else:
            print('Cell %d has size mismatch size: %s' %(id, str(cell_cwt.shape)))
            continue
    return y_test_class_names

def generate_arrays(cell_ids, cell_labels, batch_size):
    num_classes = 4     


    #maintaining ids and labels consistent, I shuffle cells to avoid having batches with always the same sequence of data inside
    ids_and_labels = list(zip(cell_ids, cell_labels))
    shuffle(ids_and_labels)
    cell_ids, cell_labels = zip(*ids_and_labels)

    #FOR DEBUG ONLY: CHECK DATA CONSISTENCY AFTER SHUFFLING
    print(cell_ids[:10], cell_labels[:10])
    #######END DEBUG

    batch_data = []
    batch_labels = []
    idx = 1
    for id, category in zip(cell_ids, cell_labels):
        cell_dir_PatchSeqGouwens = os.path.join(output_cwt_dir_PatchSeqGouwens,"cell_%d.npy" %id)
        cell_dir_PatchClampGouwens = os.path.join(output_cwt_dir_PatchClampGouwens,"cell_%d.npy" %id)
        if os.path.exists(cell_dir_PatchSeqGouwens):
            cell_cwt = np.load(os.path.join(cell_dir_PatchSeqGouwens))
        elif os.path.exists(cell_dir_PatchClampGouwens):
            cell_cwt = np.load(os.path.join(cell_dir_PatchClampGouwens))
        
        category_arr = np.ones(1)
        
        if category == 'Pvalb':
            category_arr *= 0
        elif category == 'Excitatory':
            category_arr *= 1
        elif category == 'Vip/Lamp5':
            category_arr *= 2
        elif category == 'Sst':
            category_arr *= 3
        #The nn.CrossEntropyLoss() expects class indices as target labels, not one-hot encoded vectors
        #category_arr =  to_categorical(category_arr, num_classes, dtype=torch.uint8)   

        col_size = 4800
        

        if cell_cwt.shape == (180,col_size, 2):          
            # convert it to tensorflow
            #tf_cell_cwt = tf.convert_to_tensor(cell_cwt)
            if idx < batch_size:
                batch_data.append(cell_cwt)
                batch_labels.append(category_arr)
                idx+=1
            else:
                batch_data.append(cell_cwt)
                batch_labels.append(category_arr)
                batch_labels_arr = np.concatenate([arr for arr in batch_labels], axis=0)
                yield (torch.Tensor(np.array(batch_data)), torch.LongTensor(batch_labels_arr))
                
                batch_data = []
                batch_labels = []
                idx=1

        else:
            print('Cell %d has size mismatch size: %s' %(id, str(cell_cwt.shape)))
            continue
    
        
        if id == cell_ids[-1] and len(batch_data) != 0:
            batch_labels_arr = np.concatenate([arr for arr in batch_labels], axis=0)
            yield (torch.Tensor(np.array(batch_data)), torch.LongTensor(batch_labels_arr))

def remove_sizeMismatched_cells(available_ids_list, available_ids_dict):
    print(f"Before removing size mismatched cells: list {len(available_ids_list)} and dict {len(available_ids_dict)}")
    
    num_classes = 4
    idx = 1
    ids_to_remove = []  # To store IDs that need to be removed from both list and dict

    for ids in available_ids_list:
        cell_specimen_id = int(ids)
        
        # Assume these are the paths based on your existing code
        cell_dir_PatchSeqGouwens = os.path.join(output_cwt_dir_PatchSeqGouwens, f"cell_{cell_specimen_id}.npy")
        cell_dir_PatchClampGouwens = os.path.join(output_cwt_dir_PatchClampGouwens, f"cell_{cell_specimen_id}.npy")
        
        if os.path.exists(cell_dir_PatchSeqGouwens):
            cell_cwt = np.load(cell_dir_PatchSeqGouwens)
        elif os.path.exists(cell_dir_PatchClampGouwens):
            cell_cwt = np.load(cell_dir_PatchClampGouwens)
        else:
            print(f"Cell {cell_specimen_id} data not found")
            continue

        col_size = 4800
        
        if cell_cwt.shape == (180, col_size, 2):
            # If the size matches, do nothing and continue to the next cell
            continue
        else:
            print(f"Cell {cell_specimen_id} has size mismatch size: {cell_cwt.shape}")
            ids_to_remove.append(ids)

    # Remove IDs with size mismatch from both list and dict
    for id_to_remove in ids_to_remove:
        available_ids_list.remove(id_to_remove)
        available_ids_dict.pop(str(id_to_remove))
    print(f"The new number of cells in the list {len(available_ids_list)} and dict {len(available_ids_dict)}")

def remove_sizeMismatched_cells(available_ids_list, available_ids_dict):
    print(f"Before removing size mismatched cells: list {len(available_ids_list)} and dict {len(available_ids_dict)}")
    
    num_classes = 4
    idx = 1
    ids_to_remove = []  # To store IDs that need to be removed from both list and dict

    for ids in available_ids_list:
        cell_specimen_id = int(ids)
        
        # Assume these are the paths based on your existing code
        cell_dir_PatchSeqGouwens = os.path.join(output_cwt_dir_PatchSeqGouwens, f"cell_{cell_specimen_id}.npy")
        cell_dir_PatchClampGouwens = os.path.join(output_cwt_dir_PatchClampGouwens, f"cell_{cell_specimen_id}.npy")
        
        if os.path.exists(cell_dir_PatchSeqGouwens):
            cell_cwt = np.load(cell_dir_PatchSeqGouwens)
        elif os.path.exists(cell_dir_PatchClampGouwens):
            cell_cwt = np.load(cell_dir_PatchClampGouwens)
        else:
            print(f"Cell {cell_specimen_id} data not found")
            continue

        col_size = 4800
        
        if cell_cwt.shape == (180, col_size, 2):
            # If the size matches, do nothing and continue to the next cell
            continue
        else:
            print(f"Cell {cell_specimen_id} has size mismatch size: {cell_cwt.shape}")
            ids_to_remove.append(ids)

    # Remove IDs with size mismatch from both list and dict
    for id_to_remove in ids_to_remove:
        available_ids_list.remove(id_to_remove)
        available_ids_dict.pop(str(id_to_remove))
    print(f"The new number of cells in the list {len(available_ids_list)} and dict {len(available_ids_dict)}")



def BayesSearch(train_data_dict, test_data_dict, model_no, freeze_level):
    
    global model_name    
    #Check which pre-trained model is wanted to be used
    if model_no == 1:
        model_name = 'ResNet18'
    elif model_no == 2:
        model_name = 'Incep_v3'
    elif model_no == 3:
        model_name = 'DenseNet121'
    elif model_no == 4:
        model_name = 'MobileNet_v2'

    model_folder = os.path.join(output_models, model_name +"_best_models")
    if not os.path.isdir(model_folder):
         os.mkdir(model_folder)

    global num_CV_inner
    num_CV_inner = 3
    
    global global_mean_epoch_counts
    global_mean_epoch_counts = {}
    #X_train, X_test, y_train_class_names, y_test_class_names = train_test_split(all_ids, all_classes, test_size=0.2, random_state=42, stratify=all_classes)
    X_train = [cell for cell in train_data_dict.keys()]
    y_train_class_names = [l for l in train_data_dict.values()]

    X_test = [cell for cell in test_data_dict.keys()]
    y_test_class_names = [l for l in test_data_dict.values()]

    cv_inner = StratifiedKFold(n_splits=num_CV_inner, shuffle=True, random_state=42)

    estimator = MyEstimator(model_no, freeze_level)
    # Initialize Bayesian optimization for hyperparameter search
    opt = BayesSearchCV(
        estimator=estimator,
        search_spaces=param_space,
        n_iter=50,  # Number of iterations for optimization
        cv = cv_inner,
        random_state=42
        )
    
    opt.fit(X_train, y_train_class_names)
        
    best_estimator = opt.best_estimator_
    best_params = opt.best_params_
    best_model = best_estimator.model
    batch_size_best = best_params['batch_size']
    size_model_best = best_estimator.size_model

    # Save the best model as pickle
    with open(os.path.join(model_folder, 'best_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
    # Save the best estimator as pickle
    with open(os.path.join(model_folder, 'best_estimator.pkl'), 'wb') as f:
        pickle.dump(best_estimator, f)
    
    test_whole_generator = generate_arrays_for_prediction(X_test, batch_size_best) 
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to the chosen device
    best_model = best_model.to(device)


    best_model.eval()  # Set the model to evaluation mode    
    y_pred_list = []  # To store the predictions

    with torch.no_grad():
        for batch_data in test_whole_generator:  # No need to unpack, as it only yields batch_data
            # # Move data to the same device
            batch_data = batch_data.to(device)
            #The model expects input tensors with the shape 
            #[batch_size, channels, pixels_x, pixels_y]
            batch_data = batch_data.permute(0, 3, 1, 2)
                
            resized_batch_data = F.interpolate(batch_data, size=size_model_best, mode='bilinear', align_corners=False)
            # Add a third channel by replicating the first channel
            added_channel = resized_batch_data[:, :1, :, :]  # Taking the first channel
            resized_batch_data = torch.cat((resized_batch_data, added_channel), dim=1)  # Now (1, 3, 224, 224)
                
            outputs = best_model(resized_batch_data)  # Forward pass
            _, predicted = torch.max(outputs, 1)  # Get the predicted class indices
            y_pred_list.append(predicted.cpu().numpy())  # NumPy does not support GPU tensors directly, so first move to CPU

    # Concatenate the predictions into a single numpy array
    y_pred = np.concatenate(y_pred_list, axis=0)
    # Convert integer labels in y_test to class names
    y_pred_class_names = [label_to_class[label] for label in y_pred]
    y_test_list = [label_to_number[label] for label in y_test_class_names]
    y_test = np.array(y_test_list)

    wandb.init(
        project="sts-source_project",
        group= freeze_level+"_"+model_name + "_hyperparam_search",
        name="best_model",
        config=best_params  # Logging the params directly as config
        )
    
    balanced_test_accuracy = balanced_accuracy_score(y_test, y_pred)*100
    wandb.log({"Balanced test accuracy": balanced_test_accuracy})

    cr = classification_report(y_test_class_names, y_pred_class_names, target_names = categories, output_dict=True)
    cr_df = pd.DataFrame(cr).transpose()
    cr_df.reset_index(inplace=True)

    # Rename the index column to 'class'
    cr_df.rename(columns={'index': 'class'}, inplace=True)
    wandb.log({"classification_report": wandb.Table(dataframe=cr_df)})


    conf_matrix = confusion_matrix(y_test, y_pred, labels=range(len(categories)))
    # Plot heatmap
    plot_confusion_matrix(conf_matrix)
    # Finish W&B run
    wandb.finish()


    # Get the cross-validation results
    cv_results = opt.cv_results_

    # Extract the mean test scores and their corresponding parameter configurations
    mean_test_scores = cv_results['mean_test_score']
    params_list = cv_results['params']
        
    for score, params in zip(mean_test_scores, params_list):
        batch_size = params['batch_size']
        learning_rate = params['learning_rate']
        lr_decay = params['lr_decay']
        lr_decay_period = params['lr_decay_period']
        early_stopping = params['early_stopping']
        dropout_rate = params['dropout_rate']
        # Generate the run name based on the parameters
        run_name = f"BS_{batch_size}_LR_{str(learning_rate).replace('.', '_')}_LR_decay_{str(lr_decay).replace('.', '_')}_LR_decay_period_{lr_decay_period}_ES_{early_stopping}_DR_{str(dropout_rate).replace('.', '_')}"
            
        print(f"Logging configuration: {run_name}")

        wandb.init(
        project="sts-source_project",
        group= freeze_level+"_"+model_name + "_hyperparam_search",
        name=run_name,
        config=params  # Logging the params directly as config
        )
            
        params_key = tuple(sorted(params.items()))
        epochs_inner_cv = global_mean_epoch_counts[params_key]
        mean_epochs = np.mean(epochs_inner_cv)
        rounded_mean_epochs = round(mean_epochs)  # Round the mean epochs to closest integer

        wandb.log({'Mean accuracy': score, 'Mean epochs': rounded_mean_epochs})
        wandb.finish()

#class MyEstimator(BaseEstimator):
class MyEstimator:
    def __init__(self, model_no, freeze_level, **params):
        self.model_no = model_no
        self.params = params
        self.freeze_level=freeze_level
        self.model = None     # To store the trained model
        self.size_model = None  # To store the size of the trained model
        self.epoch_counts = []

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {'model_no': self.model_no, 'freeze_level': self.freeze_level, **self.params}
    def set_params(self, **params):
        if 'model_no' in params:
            self.model_no = params.pop('model_no')
        if 'freeze_level' in params:
            self.freeze_level=params.pop('freeze_level')
        self.params.update(params)
        return self

    
    def fit(self, X_train_inner, y_train_inner, **params):

        batch_size = self.params['batch_size']
        learning_rate = self.params['learning_rate']
        lr_decay = self.params['lr_decay']
        lr_decay_period = self.params['lr_decay_period']
        early_stopping = self.params['early_stopping']
        dropout_rate = self.params['dropout_rate']
        
        params_key = tuple(sorted(self.params.items()))
        
        if params_key in global_mean_epoch_counts:
            if len(global_mean_epoch_counts[params_key]) < num_CV_inner:
                param_config = {
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'lr_decay': lr_decay,
                    'lr_decay_period': lr_decay_period,
                    'early_stopping': early_stopping,
                    'dropout_rate': dropout_rate
                }

                sweep_config = {
                    'method': 'bayes',
                    'name': 'sts-source_project',
                    'metric': {
                        'goal': 'maximize',
                        'name': 'balanced_test_accuracy'
                    },
                    'dataset': "PatchClamp&PatchSeq",
                    'architecture': model_name,
                    'parameters': param_config
                }

                wandb.init(
                        project = "sts-source_project",
                        group = self.freeze_level+"_"+ model_name +"_hyperparam_search" + "_innerCVs",
                        name = f"BS_{batch_size}_LR_{str(learning_rate).replace('.', '_')}_LR_decay_{str(lr_decay).replace('.', '_')}_LR_decay_period_{lr_decay_period}_ES_{early_stopping}_DR_{str(dropout_rate).replace('.', '_')}",
                        config = sweep_config)

                X_train_inner, X_val_inner, y_train_inner, y_val_inner = train_test_split(X_train_inner, y_train_inner, test_size=0.2,
                                                                                          random_state=42,
                                                                                          stratify=y_train_inner)
                #this model will not be used
                _, self.size_model = self._load_model()
                # Train the model using the current split
                self.model = self._train_model(X_train_inner, y_train_inner, X_val_inner, y_val_inner, **params)
        
            else:
                # Get the best epoch
                epochs_inner_cv = global_mean_epoch_counts[params_key]
                mean_epochs = np.mean(epochs_inner_cv)
                rounded_mean_epochs = round(mean_epochs)
            
                param_config = {
                    'epoch': rounded_mean_epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'lr_decay': lr_decay,
                    'lr_decay_period': lr_decay_period,
                    'early_stopping': early_stopping,
                    'dropout_rate': dropout_rate
                }

                sweep_config = {
                    'method': 'bayes',
                    'name': 'sts-source_project',
                    'metric': {
                        'goal': 'maximize',
                        'name': 'balanced_test_accuracy'
                    },
                    'dataset': "PatchClamp&PatchSeq",
                    'architecture': model_name,
                    'parameters': param_config
                }

                wandb.init(
                    project = "sts-source_project",
                    group = self.freeze_level+"_"+model_name +"_final_run_with_best_params",
                    name = f"Epocs_{rounded_mean_epochs}_BS_{batch_size}_LR_{str(learning_rate).replace('.', '_')}_LR_decay_{str(lr_decay).replace('.', '_')}_LR_decay_period_{lr_decay_period}_ES_{early_stopping}_DR_{str(dropout_rate).replace('.', '_')}",
                    config = sweep_config)
                #this model will not be used
                _, self.size_model = self._load_model()
                self.model = self._final_train_model(X_train_inner, y_train_inner, rounded_mean_epochs, **params)
        
        else:
        
            param_config = {
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'lr_decay': lr_decay,
                'lr_decay_period': lr_decay_period,
                'early_stopping': early_stopping,
                'dropout_rate': dropout_rate
            }

            sweep_config = {
                'method': 'bayes',
                'name': 'sts-source_project',
                'metric': {
                    'goal': 'maximize',
                    'name': 'test_accuracy'
                },
                'dataset': "PatchClamp&PatchSeq",
                'architecture': model_name,
                'parameters': param_config
            }

            wandb.init(
                    project = "sts-source_project",
                    group = self.freeze_level+"_"+model_name +"_hyperparam_search" + "_innerCVs",
                    name = f"BS_{batch_size}_LR_{str(learning_rate).replace('.', '_')}_LR_decay_{str(lr_decay).replace('.', '_')}_LR_decay_period_{lr_decay_period}_ES_{early_stopping}_DR_{str(dropout_rate).replace('.', '_')}",
                    config = sweep_config)

                            
            X_train_inner, X_val_inner, y_train_inner, y_val_inner = train_test_split(X_train_inner, y_train_inner, test_size=0.2, random_state=42, stratify= y_train_inner)
            #this model will not be used
            _, self.size_model = self._load_model()
            # Train the model using the current split
            self.model = self._train_model(X_train_inner, y_train_inner, X_val_inner, y_val_inner, **params)
        
        return self.model   
    
    #def score(self, X_test_inner, model, size_model, **params):
    def score(self, X_test_inner, y_train_inner, **params):
        # Check if the model has been trained
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")
        """
        batch_size = self.params['batch_size']
        learning_rate = self.params['learning_rate']
        lr_decay = self.params['lr_decay']
        lr_decay_period = self.params['lr_decay_period']
        early_stopping = self.params['early_stopping']
        dropout_rate = self.params['dropout_rate']        
        """        
        
        y_test_inner_class_names = generate_list_for_y_test(X_test_inner, y_train_inner)
        
        balanced_test_accuracy = self._evaluate_model(self.model, self.size_model, X_test_inner, y_test_inner_class_names, **params)
        wandb.log({"Balanced test accuracy": balanced_test_accuracy})
        wandb.finish()
        
        # Return the test accuracy as the score for Bayesian optimization
        return balanced_test_accuracy 

    
    def _train_model(self, X_train_inner, y_train_inner, X_val_inner, y_val_inner,  **params):

        batch_size = self.params['batch_size']
        learning_rate = self.params['learning_rate']
        lr_decay = self.params['lr_decay']
        lr_decay_period = self.params['lr_decay_period']
        early_stopping = self.params['early_stopping']
        dropout_rate = self.params['dropout_rate']
        
        params_key = tuple(sorted(self.params.items()))
        if params_key not in global_mean_epoch_counts:
            global_mean_epoch_counts[params_key] = []
            
        # Load the specified model
        model, size_model = self._load_model()
              
        # Check for GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to the chosen device
        model = model.to(device)
        print(next(model.parameters()).device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
        
        
        #5 cv-> 0.8 train, 3 cv->2/3 train
        #len_train_loader = int((len(available_ids_list)*0.8*(2/3)*0.8)//batch_size)
        #len_val_loader = int((len(available_ids_list)*0.8*(2/3)*0.2)//batch_size)

        #3 cv-> 2/3 train, 2 cv->1/2 train
        #len_train_loader = int((len(available_ids_list)*(2/3)*(1/2)*0.8)//batch_size)
        #len_val_loader = int((len(available_ids_list)*(2/3)*(1/2)*0.2)//batch_size)
        
        best_val_loss = np.inf
        epochs_no_improve = 0
        
        # Train your model using X_train_inner and validate using X_inner_val
        # Training loop
        for epoch in range(epochs):
            model.train()
            total_train_loss = 0.0
            correct_train_predictions = 0
            total_train_samples = 0
                                
            # Create DataLoader for training and validation
            train_loader = generate_arrays(X_train_inner, y_train_inner, batch_size)
            val_loader = generate_arrays(X_val_inner, y_val_inner, batch_size)
            
            #Even though it shuffles in generate_arrays(), this part is only for attain the number of batches
            #it is indifferent of the ordering ---> this block is just for counting how many batches one have. 
            train_loader_temp = generate_arrays(X_train_inner, y_train_inner, batch_size)
            val_loader_temp = generate_arrays(X_val_inner, y_val_inner, batch_size)
            len_train_loader = count_batches(train_loader_temp)
            len_val_loader = count_batches(val_loader_temp)
            print(f"Number of train batches: {len_train_loader}")
            print(f"Number of val batches: {len_val_loader}")
            
            #lr_decay_user = 0 means no decay applied
            # Check if it's time to add learning rate decay
            if (epoch + 1) % lr_decay_period == 0 and lr_decay != 0:
                # Apply learning rate decay by updating each parameter group
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= lr_decay
                                
            # Training
            for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):
                # Move data to the same device
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                
                optimizer.zero_grad()  # Zero the gradients
                
                #The model expects input tensors with the shape
                #[batch_size, channels, pixels_x, pixels_y]
                batch_data = batch_data.permute(0, 3, 1, 2)
                
                resized_batch_data = F.interpolate(batch_data, size=size_model, mode='bilinear', align_corners=False)
                # Add a third channel by replicating the first channel
                added_channel = resized_batch_data[:, :1, :, :]  # Taking the first channel
                resized_batch_data = torch.cat((resized_batch_data, added_channel), dim=1)  # Now (1, 3, 224, 224)
                
                outputs = model(resized_batch_data)  # Forward pass
                if self.model_no == 2: #If the model is Inception v3
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    loss = criterion(logits, batch_labels.squeeze())  # Calculate loss
                    _, predicted = torch.max(logits, 1)
                else:
                    loss = criterion(outputs, batch_labels.squeeze())  # Calculate loss
                    _, predicted = torch.max(outputs, 1)
                
                loss.backward()  # Backward pass
                optimizer.step()  # Update weights
                total_train_loss += loss.item()
                
                correct_train_predictions += (predicted == batch_labels.squeeze()).sum().item()
                total_train_samples += batch_labels.size(0)
                print("The number of total train samples:", total_train_samples)
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len_train_loader}], Loss: {loss.item():.4f}")
            
            train_accuracy = correct_train_predictions / total_train_samples
            print("The train accuracy:", train_accuracy)
            average_train_loss = total_train_loss / len_train_loader #average training loss per batch
            
            # Log training metrics
            wandb.log({"inner_train_loss": average_train_loss, "inner_train_accuracy": train_accuracy}, step=epoch)

            print("Now, the trained parameters will be tested on validation set")
            # Validation
            model.eval()
            total_val_loss = 0.0
            correct_val_predictions = 0
            total_val_samples = 0
        
            with torch.no_grad():
                for batch_idx, (val_data, val_labels) in enumerate(val_loader):
                    # Move data to the same device
                    val_data, val_labels = val_data.to(device), val_labels.to(device)
                    
                    #The model expects tensors with the shape
                    #[batch_size, channels, pixels_x, pixels_y]
                    val_data = val_data.permute(0, 3, 1, 2)
                    
                    resized_val_data = F.interpolate(val_data, size=size_model, mode='bilinear', align_corners=False)
                    # Add a third channel by replicating the first channel
                    added_channel = resized_val_data[:, :1, :, :]  # Taking the first channel
                    resized_val_data = torch.cat((resized_val_data, added_channel), dim=1)  # Now (1, 3, 224, 224)
                    
                    
                    val_outputs = model(resized_val_data)
                    val_loss = criterion(val_outputs, val_labels.squeeze())
        
                    total_val_loss += val_loss.item()
                    _, val_predicted = torch.max(val_outputs, 1)
                    correct_val_predictions += (val_predicted == val_labels.squeeze()).sum().item()
                    total_val_samples += val_labels.size(0)
                    print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len_val_loader}], Loss: {val_loss.item():.4f}")
            
            val_accuracy = correct_val_predictions / total_val_samples
            average_val_loss = total_val_loss / len_val_loader
            
            # Log validation metrics
            wandb.log({"inner_val_loss": average_val_loss, "inner_val_accuracy": val_accuracy}, step=epoch)

            # Print and save metrics
            print(f"\nEpoch: {epoch + 1} - Train Loss: {average_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f} | Val Loss: {average_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}\n")

            
            factor =0.95 #i want at least a 5% improvemment to avoid early stopping 
            if average_val_loss < factor*best_val_loss: #in original begum code, she was looking at val_loss (validation loss on the last batch), I corrected the code
                best_val_loss = average_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve == early_stopping:
                print(f'Early stopping after {early_stopping} epochs of no improvement.')
                wandb.log({"Epoch": epoch})
                break

        global_mean_epoch_counts[params_key].append(epoch + 1)
        
        # No longer this is needed because _final_train_model deals with the final run w/ best params
        #if len(global_mean_epoch_counts[params_key]) > num_CV_inner:
         #   wandb.finish()

        return model
    
    def _final_train_model(self, X_train_inner, y_train_inner, rounded_mean_epochs, **params):
        batch_size = self.params['batch_size']
        learning_rate = self.params['learning_rate']
        lr_decay = self.params['lr_decay']
        lr_decay_period = self.params['lr_decay_period']
        early_stopping = self.params['early_stopping']
        dropout_rate = self.params['dropout_rate']


        # Load the specified model
        model, size_model = self._load_model()

        # Check for GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to the chosen device
        model = model.to(device)
        print(next(model.parameters()).device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
        
        best_val_loss = np.inf
        epochs_no_improve = 0
        
        # Training loop
        for epoch in range(rounded_mean_epochs):
            model.train()
            total_train_loss = 0.0
            correct_train_predictions = 0
            total_train_samples = 0

            # Create DataLoader for training and validation
            train_loader = generate_arrays(X_train_inner, y_train_inner, batch_size)

            #Even though it shuffles in generate_arrays(), this part is only for attain the number of batches
            #it is indifferent of the ordering
            train_loader_temp = generate_arrays(X_train_inner, y_train_inner, batch_size)
            len_train_loader = count_batches(train_loader_temp)
            print(f"Number of train batches: {len_train_loader}")

            #lr_decay_user = 0 means no decay applied
            # Check if it's time to add learning rate decay
            if (epoch + 1) % lr_decay_period == 0 and lr_decay != 0:
                # Apply learning rate decay by updating each parameter group
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= lr_decay

            # Training
            for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):
                # Move data to the same device
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

                optimizer.zero_grad()  # Zero the gradients

                #The model expects input tensors with the shape
                #[batch_size, channels, pixels_x, pixels_y]
                batch_data = batch_data.permute(0, 3, 1, 2)

                resized_batch_data = F.interpolate(batch_data, size=size_model, mode='bilinear', align_corners=False)
                # Add a third channel by replicating the first channel
                added_channel = resized_batch_data[:, :1, :, :]  # Taking the first channel
                resized_batch_data = torch.cat((resized_batch_data, added_channel), dim=1)  # Now (1, 3, 224, 224)

                outputs = model(resized_batch_data)  # Forward pass
                if self.model_no == 2: #If the model is Inception v3
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    loss = criterion(logits, batch_labels.squeeze())  # Calculate loss
                    _, predicted = torch.max(logits, 1)
                else:
                    loss = criterion(outputs, batch_labels.squeeze())  # Calculate loss
                    _, predicted = torch.max(outputs, 1)

                loss.backward()  # Backward pass
                optimizer.step()  # Update weights
                total_train_loss += loss.item()

                correct_train_predictions += (predicted == batch_labels.squeeze()).sum().item()
                total_train_samples += batch_labels.size(0)
                print("The number of total train samples:", total_train_samples)
                print(f"Epoch [{epoch+1}/{rounded_mean_epochs}], Batch [{batch_idx}/{len_train_loader}], Loss: {loss.item():.4f}")

            train_accuracy = correct_train_predictions / total_train_samples
            print("The train accuracy:", train_accuracy)
            average_train_loss = total_train_loss / len_train_loader #average training loss per batch

            # Log training metrics
            wandb.log({"final_train_loss": average_train_loss, "final_train_accuracy": train_accuracy}, step=epoch)
        
        wandb.finish()
        return model


    def _evaluate_model(self, model, size_model, X_test_inner, y_test_inner_class_names, **params):
        batch_size = self.params['batch_size']
        #Evaluate the trained model on test set
        print("Generation of X_test has started")
        test_whole_generator = generate_arrays_for_prediction(X_test_inner, batch_size) 
        print("size of X_test is ", len(X_test_inner))
        
        # Move model to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        print("Based on X_test, the prediction will be made")
        model.eval()  # Set the model to evaluation mode
        
        y_pred_inner_list = []  # To store the predictions

        with torch.no_grad():
            for batch_data in test_whole_generator:  # No need to unpack, as it only yields batch_data
                # # Move data to the same device
                batch_data = batch_data.to(device)
                #The model expects input tensors with the shape 
                #[batch_size, channels, pixels_x, pixels_y]
                batch_data = batch_data.permute(0, 3, 1, 2)
                
                resized_batch_data = F.interpolate(batch_data, size=size_model, mode='bilinear', align_corners=False)
                # Add a third channel by replicating the first channel
                added_channel = resized_batch_data[:, :1, :, :]  # Taking the first channel
                resized_batch_data = torch.cat((resized_batch_data, added_channel), dim=1)  # Now (1, 3, 224, 224)
                
                outputs = model(resized_batch_data)  # Forward pass
                _, predicted = torch.max(outputs, 1)  # Get the predicted class indices
                y_pred_inner_list.append(predicted.cpu().numpy())  # NumPy does not support GPU tensors directly, so first move to CPU

        # Concatenate the predictions into a single numpy array
        y_pred_inner = np.concatenate(y_pred_inner_list, axis=0)

        print("size of y_pred_inner: %d" %(len(y_pred_inner)))
        #print("size of inner_test_idx ", len(inner_test_idx))
        print("size of y_test_inner_class_names ", len(y_test_inner_class_names))

        #Find the predicted class indices
        print("y_pred_inner (first 10 values after np.argmax)", y_pred_inner[0:10])

        y_test_inner_list = [label_to_number[label] for label in y_test_inner_class_names]
        y_test_inner = np.array(y_test_inner_list)
        print("size of y_test_inner: %d" %(len(y_test_inner)))
        
        # Convert integer labels in y_test to class names
        y_pred_inner_class_names = [label_to_class[label] for label in y_pred_inner]

        if len(np.unique(y_pred_inner_class_names)) != 4:
            print("y_pred_inner does not contain 4 classes but", np.unique(y_pred_inner_class_names))
            print("y_pred_inner ( first 10 class values)", y_pred_inner_class_names[0:10])
            print("y_test_inner contains classes of", np.unique(y_test_inner_class_names))
        else:
            print("y_pred_inner exactly contains 4 classes", np.unique(y_pred_inner_class_names))
        
        
        #accuracy = accuracy_score(y_test_inner, y_pred_inner)
        #return accuracy
        
        balanced_accuracy = balanced_accuracy_score(y_test_inner, y_pred_inner)
        return balanced_accuracy * 100
        
    
    def _load_model(self):
        dropout_rate = self.params['dropout_rate']
        #Check which pre-trained model is wanted to be used
        if self.model_no == 1: 
            #Load ResNet18 model
            model = models.resnet18(pretrained=True)
            
            if self.freeze_level=='FULL':
                 # Freeze all parameters in the feature extractor model
                for param in model.parameters():
                    param.requires_grad = False
            
            num_ftrs = model.fc.in_features
            if dropout_rate == 0:
                model.fc = nn.Linear(num_ftrs, 4)
            else:
                model.fc = nn.Sequential(
                    nn.Dropout(p=dropout_rate),  # Dropout layer with 50% probability
                    nn.Linear(num_ftrs, 4)  # Fully connected layer with 4 output features
                )
            size_model = (224, 244)
        elif self.model_no == 2:
            #Load Inception-V3 model
            model = models.inception_v3(pretrained=True)

            if self.freeze_level=='FULL':
                 # Freeze all parameters in the feature extractor model
                for param in model.parameters():
                    param.requires_grad = False

            num_ftrs = model.fc.in_features
            if dropout_rate == 0:
                model.fc = nn.Linear(num_ftrs, 4)
            else:
                model.fc = nn.Sequential(
                    nn.Dropout(p=dropout_rate),  # Dropout layer with 50% probability
                    nn.Linear(num_ftrs, 4)  # Fully connected layer with 4 output features
                )
            size_model = (299, 299)
        elif self.model_no == 3:
            #Load DenseNet-121 model
            model = models.densenet121(pretrained=True)

            if self.freeze_level=='FULL':
                 # Freeze all parameters in the feature extractor model
                for param in model.parameters():
                    param.requires_grad = False

            num_ftrs = model.classifier.in_features
            if dropout_rate == 0:
                model.classifier = nn.Linear(num_ftrs, 4)
            else:
                model.classifier = nn.Sequential(
                    nn.Dropout(p=dropout_rate),  # Dropout layer with 50% probability
                    nn.Linear(num_ftrs, 4)  # Fully connected layer with 4 output features
                )
            size_model = (224, 244)
        elif self.model_no == 4:
            #Load MobileNet_V2 model
            model = models.mobilenet_v2(pretrained=True)

            if self.freeze_level=='FULL':
                 # Freeze all parameters in the feature extractor model
                for param in model.parameters():
                    param.requires_grad = False

            num_ftrs = model.classifier[-1].in_features
            if dropout_rate == 0:
                model.classifier[-1] = nn.Linear(num_ftrs, 4)
            else:
                model.classifier[-1] = nn.Sequential(
                    nn.Dropout(p=dropout_rate),  # Dropout layer with 50% probability
                    nn.Linear(num_ftrs, 4)  # Fully connected layer with 4 output features
                )
            size_model = (224, 244)

        # Check which parameters are unfrozen and count them
        trainable_params = 0
        print("Trainable Parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"  {name} - {param.numel()} parameters")
                trainable_params += param.numel()

        print(f"Total trainable parameters: {trainable_params}")


        return model, size_model



def count_batches(generator_func):
    count = 0
    for _ in generator_func:
        count += 1
    return count


def main(model_no, mode, freeze_level):
    wandb.login()
    
    if not os.path.isdir(output_models):
         os.mkdir(output_models)

    global categories
    #load cell IDs and their Labels
    categories = ['Pvalb', 'Excitatory', 'Vip/Lamp5', 'Sst']
    data_dict = {}
    with open(os.path.join(input_dir,'cell_types_GouwensAll_new.txt'), 'r') as file:
        for line in file:
            # Split the line at the ':' character to separate key and value
            value, key = line.strip().split(' ')
            # Add the key-value pair to the dictionary
            data_dict[int(key)] = value
    cells_list = list(data_dict.items())


    #Create Train and Test dictionaries with cell IDs and their dataset origin and label
    train_data_dict = {}
    with open(os.path.join(input_dir_split,'Train_split.csv'), 'r') as file:
        for line in file:
            # Split the line at the ':' character to separate key and value
            key, value = line.strip().split(', ')
            # Add the key-value pair to the dictionary
            train_data_dict[int(key)] = data_dict[int(key)]

    test_data_dict = {}
    with open(os.path.join(input_dir_split,'Test_split.csv'), 'r') as file:
        for line in file:
            # Split the line at the ':' character to separate key and value
            key, value = line.strip().split(', ')
            # Add the key-value pair to the dictionary
            test_data_dict[int(key)] = data_dict[int(key)]

    global label_to_class
    global label_to_number
    label_to_class = dict(zip(range(len(categories)), categories))
    label_to_number = {label: i for i, label in enumerate(categories)}
    
    global param_space
    param_space = {
        'batch_size': Categorical([32, 64]),
        'learning_rate': Real(0.001, 0.1*0.8, prior='log-uniform')
        'lr_decay': Categorical([0, 0.01, 0.1]), #lr_decay= 0 means no decay
        'lr_decay_period': Integer(5, 20),
        'early_stopping': Integer(10, 20),
        'dropout_rate': Categorical([0, 0.3, 0.5, 0.7])
        }
    
    # If test is TRUE it runs only the testing on best model otherwise it does everything
    if mode == "TEST":
        Testing_best_model(test_data_dict, model_no)
    elif mode == "EXPLAIN":
        Testing_best_model_with_explainability(test_data_dict, model_no)
    else:
        BayesSearch(train_data_dict, test_data_dict, model_no, freeze_level)

    


def Testing_best_model(test_data_dict, model_no):
     

    global model_name    
    #Check which pre-trained model is wanted to be used
    if model_no == 1:
        model_name = 'ResNet18'
    elif model_no == 2:
        model_name = 'Incep_v3'
    elif model_no == 3:
        model_name = 'DenseNet121'
    elif model_no == 4:
        model_name = 'MobileNet_v2'
    model_folder = os.path.join(best_models_path, model_name +"_best_models")
    if not os.path.isdir(model_folder):
         os.mkdir(model_folder)
    
    
    model_path = os.path.join(model_folder, 'best_model.pkl')
    estimator_path = os.path.join(model_folder, 'best_estimator.pkl')
    

    # Caricamento del modello
    with open(model_path, 'rb') as f:
       best_model  = pickle.load(f)

    with open(estimator_path, 'rb') as f:
       best_estimator  = pickle.load(f)

    batch_size_best = best_estimator.params['batch_size']
    best_params = best_estimator.params
    
    X_test = [cell for cell in test_data_dict.keys()]
    y_test_class_names = [l for l in test_data_dict.values()]

    test_whole_generator = generate_arrays_for_prediction(X_test,  batch_size_best) 
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to the chosen device
    best_model = best_model.to(device)
    size_model_best = best_estimator.size_model

    best_model.eval()  # Set the model to evaluation mode    
    y_pred_list = []  # To store the predictions

    with torch.no_grad():
        for batch_data in test_whole_generator:  # No need to unpack, as it only yields batch_data
            # # Move data to the same device
            batch_data = batch_data.to(device)
            #The model expects input tensors with the shape 
            #[batch_size, channels, pixels_x, pixels_y]
            batch_data = batch_data.permute(0, 3, 1, 2)
                
            resized_batch_data = F.interpolate(batch_data, size=size_model_best, mode='bilinear', align_corners=False)
            # Add a third channel by replicating the first channel
            added_channel = resized_batch_data[:, :1, :, :]  # Taking the first channel
            resized_batch_data = torch.cat((resized_batch_data, added_channel), dim=1)  # Now (1, 3, 224, 224)
                
            outputs = best_model(resized_batch_data)  # Forward pass
            _, predicted = torch.max(outputs, 1)  # Get the predicted class indices
            y_pred_list.append(predicted.cpu().numpy())  # NumPy does not support GPU tensors directly, so first move to CPU

    # Concatenate the predictions into a single numpy array
    y_pred = np.concatenate(y_pred_list, axis=0)
    # Convert integer labels in y_test to class names
    y_pred_class_names = [label_to_class[label] for label in y_pred]
    y_test_list = [label_to_number[label] for label in y_test_class_names]
    y_test = np.array(y_test_list)

    wandb.init(
        project="sts-source_project",
        group= freeze_level+"_"+model_name + "_hyperparam_search",
        name="best_model",
        config=best_params  # Logging the params directly as config
        )
    
    balanced_test_accuracy = balanced_accuracy_score(y_test, y_pred)*100
    wandb.log({"Balanced test accuracy": balanced_test_accuracy})

    cr = classification_report(y_test_class_names, y_pred_class_names, target_names = categories, output_dict=True)
    cr_df = pd.DataFrame(cr).transpose()
    cr_df.reset_index(inplace=True)

    # Rename the index column to 'class'
    cr_df.rename(columns={'index': 'class'}, inplace=True)
    wandb.log({"classification_report": wandb.Table(dataframe=cr_df)})


    conf_matrix = confusion_matrix(y_test, y_pred, labels=range(len(categories)))
    # Plot heatmap
    plot_confusion_matrix(conf_matrix)
    # Finish W&B run
    wandb.finish()


def Testing_best_model_with_explainability(test_data_dict, model_no):
    global model_name
    # Check which pre-trained model is wanted to be used
    if model_no == 1:
        model_name = 'ResNet18'
    elif model_no == 2:
        model_name = 'Incep_v3'
    elif model_no == 3:
        model_name = 'DenseNet121'
    elif model_no == 4:
        model_name = 'MobileNet_v2'
    model_folder = os.path.join(best_models_path, model_name + "_best_models")
    if not os.path.isdir(model_folder):
        os.mkdir(model_folder)

    model_path = os.path.join(model_folder, 'best_model.pkl')
    estimator_path = os.path.join(model_folder, 'best_estimator.pkl')

    # Caricamento del modello
    with open(model_path, 'rb') as f:
        best_model = pickle.load(f)

    with open(estimator_path, 'rb') as f:
        best_estimator = pickle.load(f)

    batch_size_best = best_estimator.params['batch_size']
    best_params = best_estimator.params

    X_test = [cell for cell in test_data_dict.keys()]
    y_test_class_names = [l for l in test_data_dict.values()]

    test_whole_generator = generate_arrays_for_prediction(X_test, batch_size_best)

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to the chosen device
    best_model = best_model.to(device)
    size_model_best = best_estimator.size_model

    best_model.eval()  # Set the model to evaluation mode
    y_pred_list = []  # To store the predictions

    with torch.no_grad():
        for batch_data in test_whole_generator:  # No need to unpack, as it only yields batch_data
            # # Move data to the same device
            batch_data = batch_data.to(device)
            # The model expects input tensors with the shape
            # [batch_size, channels, pixels_x, pixels_y]
            batch_data = batch_data.permute(0, 3, 1, 2)

            resized_batch_data = F.interpolate(batch_data, size=size_model_best, mode='bilinear', align_corners=False)
            # Add a third channel by replicating the first channel
            added_channel = resized_batch_data[:, :1, :, :]  # Taking the first channel
            resized_batch_data = torch.cat((resized_batch_data, added_channel), dim=1)  # Now (1, 3, 224, 224)

            outputs = best_model(resized_batch_data)  # Forward pass
            _, predicted = torch.max(outputs, 1)  # Get the predicted class indices
            y_pred_list.append(
                predicted.cpu().numpy())  # NumPy does not support GPU tensors directly, so first move to CPU

    # Concatenate the predictions into a single numpy array
    y_pred = np.concatenate(y_pred_list, axis=0)
    # Convert integer labels in y_test to class names
    y_pred_class_names = [label_to_class[label] for label in y_pred]
    y_test_list = [label_to_number[label] for label in y_test_class_names]
    y_test = np.array(y_test_list)

    wandb.init(
        project="sts-source_project",
        group=freeze_level + "_" + model_name + "_explain",
        name="best_model_interpretation",
        config=best_params  # Logging the params directly as config
    )

    print("Questi sono i primi 5 di test")
    print(y_test[:5])
    print(y_test_class_names[:5])
    print(len([p for p in y_pred_class_names if p=='Pvalb']))

    print("Questi sono i primi 5 predetti")
    print(y_pred[:5])
    print(y_pred_class_names[:5])

    wandb.log({"Predictions": wandb.Table(dataframe=pd.DataFrame({'Predictions':y_pred, 'True_labels': y_test}))})
    #wandb.log({"Test_set": wandb.Table(dataframe=pd.DataFrame(test_data_dict, index=range(len(test_data_dict))))})

    #for explainability, take one cell for each type
    torch.use_deterministic_algorithms(False)
    print(label_to_number)
    print(label_to_class)

    cell_pvalb= 765881829
    cell_sst= 805642715
    cell_sst2= 759959254
    cell_vip= 681631284
    cell_exct= 599334696

    #load the cells
    cwt_pvalb=load_cell_for_interpretability(cell_pvalb,device, size_model_best)
    cwt_vip = load_cell_for_interpretability(cell_vip, device, size_model_best)
    cwt_sst = load_cell_for_interpretability(cell_sst, device, size_model_best)
    cwt_sst2 = load_cell_for_interpretability(cell_sst2, device, size_model_best)
    cwt_exct =load_cell_for_interpretability(cell_exct, device, size_model_best)

    interpret(cwt_pvalb, 'Pvalb', best_model)
    interpret(cwt_sst, 'Sst', best_model)
    interpret(cwt_sst2, 'Sst2', best_model)
    interpret(cwt_vip, 'Vip', best_model)
    interpret(cwt_exct, 'Exct', best_model)

    #this part is equivalent to normally testing the whole model
    balanced_test_accuracy = balanced_accuracy_score(y_test, y_pred) * 100
    wandb.log({"Balanced test accuracy": balanced_test_accuracy})

    cr = classification_report(y_test_class_names, y_pred_class_names, output_dict=True)
    print(categories)
    print(cr)
    cr_df = pd.DataFrame(cr).transpose()
    print(cr_df)
    cr_df.reset_index(inplace=True)
    print(cr_df)

    # Rename the index column to 'class'
    cr_df.rename(columns={'index': 'class'}, inplace=True)
    print(cr_df)
    wandb.log({"classification_report": wandb.Table(dataframe=cr_df)})

    conf_matrix = confusion_matrix(y_test, y_pred, labels=range(len(categories)))
    # Plot heatmap
    plot_confusion_matrix(conf_matrix)
    # Finish W&B run
    wandb.finish()

def interpret(cell_cwt, label,  model):
    cell_cwt.requires_grad_()
    scores = model(cell_cwt)

    # Get the index corresponding to the maximum score and the maximum score itself.
    score_max_index = scores.argmax()
    score_max = scores[0, score_max_index]
    '''
    backward function on score_max performs the backward pass in the computation graph and calculates the gradient of 
    score_max with respect to nodes in the computation graph
    '''
    score_max.backward()
    '''
    Saliency would be the gradient with respect to the input image now. But note that the input image has 3 channels. 
    To derive a single class saliency value for each pixel (i, j),  we take the maximum magnitude
    across all colour channels.
    '''
    saliency, _ = torch.max(cell_cwt.grad.data.abs(), dim=1)
    cwt_image = cell_cwt[0, 0, :, :].detach().cpu().numpy()

    ##code to plot by sweep, considering only rheobase+40
    # Extract gradients for the first two channels
    cwt_channel_2 = cell_cwt[0, 1, :, :].detach().cpu().numpy()  # Channel 2 CWT image

    grad_channel_2 = cell_cwt.grad[:, 1, :, :].data.abs()  # Saliency for channel 2

    saliency_channel_2 = grad_channel_2[0].cpu().numpy()

    # Optional: Normalize the saliency map for better visualization
    saliency_channel_2 /= saliency_channel_2.max()

    # Flip both scalogram and saliency vertically
    channel_2_cwt = np.flipud(cwt_channel_2)
    channel_2_saliency = np.flipud(saliency_channel_2)


    # Create the figure
    print("Visualizing saliency")
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot scalogram with high-contrast colormap
    img1 = ax.imshow(channel_2_cwt, cmap="gray_r", interpolation="nearest")

    # Overlay saliency map with existing colormap
    img2 = ax.imshow(channel_2_saliency, cmap="jet", alpha=0.5, interpolation="nearest")

    title= 'Excitatory' if label=='Exct' else label

    # Labels and title
    ax.set_xlabel("Time", fontsize=16, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=16, fontweight="bold")
    ax.set_title(title, fontsize=18, fontweight="bold")

    # Move color bar below the figure and adjust its length
    cbar = plt.colorbar(img2, orientation="vertical", ax=ax, pad=0.05, shrink=0.8)
    cbar.set_label("Intensity", fontsize=14, fontweight="bold")
    cbar.ax.tick_params(labelsize=12)

    # Hide axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    # Show the plot
    plt.show()
    # Log to wandb
    wandb.log({f"saliency_cwt_sweeps_{label}": wandb.Image(plt)})
    plt.close()


def plot_confusion_matrix(conf_matrix): 
    plt.figure(figsize=(8, 8))  # Maintain square format 
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                cbar=False, annot_kws={"size": 24},  # Bigger font size for numbers 
                xticklabels=['Pvalb', 'Excitatory', 'Vip/Lamp5', 'Sst'],
                yticklabels=['Pvalb', 'Excitatory', 'Vip/Lamp5', 'Sst'])
 
    # Customize axis labels and ticks 
    plt.xlabel("Predicted Label", fontsize=21, fontweight='bold')  # Increase by 1 point 
    plt.ylabel("True Label", fontsize=21, fontweight='bold')      # Increase by 1 point 
    plt.xticks(fontsize=19)  # Increase tick size by 1 point 
    plt.yticks(fontsize=19)  # Increase tick size by 1 point 
    plt.gca().tick_params(axis='y', pad=10)  # Add spacing for y-ticks 

    plt.tight_layout() 
    plt.show()
    wandb.log({"confusion_matrix_heatmap": wandb.Image(plt)})




if __name__ == "__main__":
    print('Launching main')

    #global model_no

    if len(sys.argv) > 3:
        model_no = int(sys.argv[1])
        if model_no<1 or model_no>4:
            print("Wrong model number! Choose 1: ResNet18, 2:InceptionV3, 3:DenseNet121, 4: MobileNetV2")
        mode = sys.argv[2].upper().strip()
        print(mode)
        freeze_level = sys.argv[3].upper().strip() if len(sys.argv)==4 else "NONE" #By default no freezing if not specified
        if not ((mode=="TRAIN") or (mode=="TEST") or (mode=='EXPLAIN')):
            print("Select either TRAIN or TEST OR EXPLAIN")
            exit()
        if not((freeze_level=='FULL') or (freeze_level=='NONE')):
            print("Invalid option for level of freezing of deep feature extractor")
            exit()
    else:
        print("USAGE: sts_pipeline.py model_number mode [freeze_level] ")
        exit()
    main(model_no, mode, freeze_level)

