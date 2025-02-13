# spike-trains-scalograms
This repository contains the code of the Spike Train Scalograms (STS) pipeline for neuronal cell types classification. The STS pipeline combines fine-tuned Convolutional Neural Networks (CNN) with scalograms of spike trains obtained through Continuous Wavelet Transform (CWT) to classify neurons from raw Electrophysiological (EP) recordings rather than human-knowledge-based features.

## Release notes

STS v1.0: 

* First release of STS.

## How to cite

### STS Primary publication

* Amprimo, G., Martini, L., Bilir, B., Bardini, R., Savino, A., Olmo, G., Di Carlo, S., Spike Train Scalograms (STS): a Deep Learning Classification Pipeline
for Neuronal Cell Types, 2025 (submitted to IEEE EMBC 2025).

STS relies on two datasets of murine cortical interneurons.

### PatchSeqDataset
It leverages the patch-seq technique, that combines patch-clamp with single-cell RNA-Seq (scRNA-Seq) to collect EP and transcriptomic profiles of 4,200 mouse visual cortical GABAergic interneurons, reconstructing the morphological conformation of 517 of them. Primary publication:

* Gouwens, N. W., Sorensen, S. A., Baftizadeh, F., Budzillo, A., Lee, B. R., Jarsky, T., ... & Zeng, H. (2020). Integrated morphoelectric and transcriptomic classification of cortical GABAergic cells. Cell, 183(4), 935-953, https://doi.org/10.1016/j.cell.2020.09.057.


### PatchClampDataset
It provides EP data from 1,938 neurons of adult mouse visual cortical neurons and morphological reconstructions for 461 of them. Primary publication:

* Gouwens, N.W., Sorensen, S.A., Berg, J. et al. Classification of electrophysiological and morphological neuron types in the mouse visual cortex. Nat Neurosci 22, 1182–1195 (2019), https://doi.org/10.1038/s41593-019-0417-0.



## Experimental setup

Follow these steps to setup for reproducing the experiments provided in _Amprimo et al., 2025_.
1) Install `Singularity` from https://docs.sylabs.io/guides/3.0/user-guide/installation.html:
	* Install `Singularity` release 3.10.2, with `Go` version 1.18.4
	* Suggestion: follow instructions provided in _Download and install singularity from a release_ section after installing `Go`
	* Install dependencies from: https://docs.sylabs.io/guides/main/admin-guide/installation.html
2) Clone the STS repository in your home folder
```
git clone https://github.com/smilies-polito/spike-trains-scalograms.git
```

3) Move to the spike-train-scalograms source subfolder, and build the singularity container with 
```
cd spike-trains-scalograms/source
sudo singularity build STS.sif STS.def
```
or
```
cd spike-trains-scalograms/source
singularity build --fakeroot STS.sif STS.def
```

5) The STS pipeline exploits the Weight&Biases tool for logging train and test runs of the STS pipeline. Create an account at https://wandb.ai/site/ and configure credential during the first code run. 

# Reproducing STS pipeline

The STS pipeline described in In _Amprimo et al., 2025_, is implemented in source/sts_pipeline.py. The script can be run in three mode:
* **TRAIN**: This mode allows to retrain a certain configuration of the STS pipeline, by specifying which model pre-trained on ImageNet to leverage as deep feature extractor. 
* **TEST**: This mode reproduces the classification performance results of the optimal configurations of the STS pipeline reported in the manuscript. 
* **EXPLAIN**: This mode reproduces the explainability analysis of the optimal configurations of the STS pipeline using saliency maps.

## Data required

In order to reproduce the analysis, it is necessary to gather the scalograms generated from the _PatchSeqDataset_ and the _PatchClampDataset_ considered in this study. The scalograms are available as a Kaggle dataset at 
https://www.kaggle.com/datasets/smiliesatpolito/STS-data. The dataset contains also the optimal STS configurations trained during the experiment. These steps must be observed:

1) Unzip the best_model folder inside spike-trains-scalograms/models.

2) Unzip the two folders with the scalograms of the two datasets in spike-trains-scalograms/data


## Reproducing the analysis running the STS Singularity container


### Running the  STS pipeline

First activate a singularity shell using the constructed container.

```
singularity shell --nv ../STS_pytorch.sif
```
For all modes, the deep feature extractor model must be specified using a _model-no_ parameter that can assume values between **1** and **4**, as follow:

1 - ResNet18

2 -  InceptionV3

3 - DenseNet121

4 - MobileNetV2

Moreover, for the **TRAIN** mode, whether to freeze the parameters of the deep feature extractor or let them retrain completely can be selected specifying either the _freeze-level_ parameter with value **FULL** or **NONE**. In _Amprimo et al 2025_ the **NONE** parameter was employed, so the whole deep feature extractor was fine-tuned for the classification task.

The generic command to run the STS pipeline on the singularity shell is:

```
python sts_pipeline.py model-no mode [freeze-level] 
```

For instance, to run in **TRAIN** mode the STS pipeline with complete fine-tuning of DenseNet121, run:

```
python sts_pipeline.py 3 TRAIN NONE 
```

For simply reproducing the analysis of _Amprimo et al 2025_ for a certain optimal STS pipeline configuration, e.g., the one with InceptionV3, run in **TEST** mode using:

```
python sts_pipeline.py 2 TEST NONE 
```

To reproduce the explainability analysis, run in **EXPLAIN** mode using:

```
python sts_pipeline.py 2 EXPLAIN NONE 
```

## Running the baseline LGBM model

To reproduce the baseline shallow model trained on high-level EP features of spike trains, run the baseline_pipeline.py script.


```
python baseline_pipeline.py 
```

The EP feature for the neurons in the two datasets are obtained from the two csv files in data/Split:

1. PatchClamp_EP_features.csv
2. PatchSeq_EP_features.csv

PatchClamp_EP_features.csv was obtained with a custom script calling the DANDI (https://dandiarchive.org) APIs to access the DANDISET 000020 (https://doi.org/10.48324/dandi.000020/0.210913.1639). This script leverages the `dandi-cli` tool (10.5281/zenodo.3692138) to access raw EP data for each cell and compute EP features.

	Credits:  Allen Institute for Brain Science (2020). Patch-seq recordings from mouse visual cortex. Available from https://dandiarchive.org/dandiset/000020/ and https://github.com/dandisets/000020.

	Primary publication: Gouwens, N. W., Sorensen, S. A., Baftizadeh, F., Budzillo, A., Lee, B. R., Jarsky, T., ... & Zeng, H. (2020). Integrated morphoelectric and transcriptomic classification of cortical GABAergic cells. Cell, 183(4), 935-953, https://doi.org/10.1016/j.cell.2020.09.057.

PatchSeq_EP_features.csv can be downloaded at http://celltypes.brain-map.org/cell_types_specimen_details.csv. 

	Credits:  Allen Institute for Brain Science (2023). Cell Types dataset. Available from http://celltypes.brain-map.org/data (DOWNLOAD CELL FEATURE DATA button).

	Primary publication: Gouwens, N.W., Sorensen, S.A., Berg, J. et al. Classification of electrophysiological and morphological neuron types in the mouse visual cortex. Nat Neurosci 22, 1182–1195 (2019), https://doi.org/10.1038/s41593-019-0417-0.


## Repository structure

After retrieving and unzipping the kaggle datasets as described above, the folder and file structure should be as follows
```
|
├── data                                       // Data files
│    ├── Split                                 // Folder with train-test data splits and EP features for the baseline model
│    │    ├── PatchSeq_EP_features.csv         // EP features file for EP analysis of the PatchSeqDataset
│    │    │                                    // (Credits: Allen Institute for Brain Science (2020). 
│    │    │                                    // Patch-seq recordings from mouse visual cortex.
│    │    │                                    // Available from: 
│    │    │                                    // https://dandiarchive.org/dandiset/000020/
│    │    │                                    // https://github.com/dandisets/000020.)
│    │    │
│    │    ├── PatchClamp_EP_features.csv       // EP features file for the PatchClampDataset
│    │    │                                    // (Credits: Allen Institute for Brain Science (2023). 
│    │    │                                    // Allen SDK: https://github.com/alleninstitute/allensdk
│    │    │                                    // IPFX: https://github.com/alleninstitute/ipfx.)
│    │    │
│    │    ├── Train_split.csv                  // Data to consider as training samples
│    │    └── Test_split.csv                   // Data to consider as test samples   
│    │
│    ├── PatchClampGouwensCWT                  // Scalograms for PatchClampDataset  
│    │    └── ...
│    │
│    └── PatchSeqGouwensCWT                    // Scalograms for PatchSeqDataset  
│         └── ...   
│
├── source                                     // Scripts for STS pipeline and baseline LGBM model
│    ├── sts_pipeline.py                       // Python script for running the baseline pipeline
│    └── baseline_pipeline.py                  // Python script for cell line-based cell type label analysis
│
├── models                                     // Optimal deep feature extractor models trained in Amprimo et al. 2025
│    ├── DenseNet121_best_models				
│    │    └── ...                      
│    ├── InceptionV3_best_models         			
│    │    └── ...
│    └── ...
│
├── output                                     // Local output of the STS analysis (e.g., retrained deep feature extractor model)
│    └── ...                                  
│
└── README.md                                  // This README file          

```

