# Predict Readmission of HFpEF patients using LV strains from cine CMR
**Author: Zhennong Chen, PhD**<br />

This is the GitHub repo based on an paper under rewivew: <br />
*Predicting Two-year Readmission of Heart Failure with Preserved Ejection Fraction using Comprehensive Left-ventricular Strain Analysis from Cardiac Magnetic Resonance and Deep Learning*<br />
Authors: Zhennong Chen, Hui Ren, Chagin Choi, Hyun Jin Cho, Sun-hyun Kim, Siyeop Yoon, Xiang Li, Quanzheng Li<br />

**Citation**: TBD

## Description
We have proposed to use ***LV strain maps*** derived from cine CMR with a ***graph convolutional network (GCN)*** to predict heart failure (HF)-related readmission in patients with ***heart failure with preserved ejection fraction (HFpEF)***.<br />
The main contributions  are as follows:<br />
(1) LV strain map is a [N,T] matrix where N is the number of AHA segments (regional) and T is the number of time frames (temporal). It is a comprehensive representation of LV function.<br />
(2) A Chebyshev GCN is used to encode the LV strain map. The edge matrix is defined according to the anatomical adjacency of AHA segments.<br />
(3) We also offer the option to add clinical data from electronic health record (EHR) as additional input.<br />


## User Guideline
### Environment Setup
The entire code is [containerized](https://www.docker.com/resources/what-container). This makes setting up environment swift and easy. Make sure you have nvidia-docker and Docker CE [installed](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) on your machine before going further. <br />
- You can build your own docker from the folder ```docker```. The docker image can be built by ```./docker_build.sh```, after that the docker container can be built by ```./docker_run.sh```. The installed packages can be referred to ```dockerfile``` and ```requirements.txt``` <br />
- You'll need  ```docker/docker_tensorflow``` for step 1 and ```docker/docker_torch``` for the step 2<br />

### Data Preparation (we have examples available)
You should prepare two things before running this step. Please refer to the `example_data` folder for guidance:

- **NIfTI images** of cine CMR data and **corresponding segmentation** (myocardium + LV bloodpool).
   - Please re-organize your data so that each time frame (3D volume) is a separate nii file. 
   - The segmentations were done by our [cineCMR segmentation foundation model](https://github.com/zhennongchen/cineCMR_SAM), or you can prepare by your own.
   - These are saved in ```example_data/raw_nii_images```

- **A patient list** that enumerates all your cases.  
   - To understand the expected format, please refer to the file:  
     `example_data/Patient_lists/patient_list.xlsx`.
   - The ground truth label of re-admission is also provided in this patient list


### Experiments
we have design our study into 2 steps, with each step having its own jupyter notebook.<br /> 
**step1: get LV strain map**: ```step1_calculate_LV_strain.ipynb```
1. We use published method [DeepStrain](https://github.com/moralesq/DeepStrain) to generate regional LV strain from cine CMR short-axis (SAX) views. we use the first time frame (end-diastole) as template. <br /> 
2. The generated strains are saved in ```example_data/results```

**step2: model***: use ```step2_model.ipynb```
1. It interpolates the number of time frames into 25 as default <br /> 
2. Training and prediction included  <br /> 

### Additional guidelines 
Please contact chenzhennong@gmail.com for any further questions.



