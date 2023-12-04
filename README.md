# DistPepFold

<a href="https://github.com/marktext/marktext/releases/latest">
   <img src="https://img.shields.io/badge/DistPepFold-v1.0.0-green">
   <img src="https://img.shields.io/badge/platform-Linux-green">
   <img src="https://img.shields.io/badge/Language-python3-green">
   <img src="https://img.shields.io/badge/dependencies-tested-green">
   <img src="https://img.shields.io/badge/licence-GNU-green">
</a>  

DistPepFold is a computational tool using deep learning for peptide docking.

Copyright (C) 2023 Zicong Zhang, Jacob Verburgt, Yuki Kagaya, Charles Christoffer, Daisuke Kihara, and Purdue University. 

License: GPL v3. (If you are interested in a different license, for example, for commercial use, please contact us.) 

Contact: Daisuke Kihara (dkihara@purdue.edu)

For technical problems or questions, please reach to Zicong Zhang (zhan1797@purdue.edu).

## Installation
<details>

### System Requirements
GPU: any GPU supports CUDA with at least 12GB memory. <br>
GPU is required for DistPepFold and no CPU version is available.

## Pre-required software
### Required 
Python 3 : https://www.python.org/downloads/     
### Optional for protein structure visualization
Pymol (for map visualization): https://pymol.org/2/    


## Environment set up  
### 1. [`Install git`](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) 
### 2. Clone the repository in your computer 
```
git clone  https://github.com/kiharalab/DistPepFold.git && cd DistPepFold
```
### 3. Build dependencies.   
#### 3.1 Install with anaconda (Recommended)
##### 3.1.1 [`install anaconda`](https://www.anaconda.com/download). 
##### 3.1.2 Install dependency in command line
Make sure you are in the DistPepFold directory and then run 
```
conda env create -f environment.yml
```
Each time when you want to run this software, simply activate the environment by
```
conda activate distpepfold
conda deactivate(If you want to exit) 
```




## Usage
   
```
bash pred.sh
```