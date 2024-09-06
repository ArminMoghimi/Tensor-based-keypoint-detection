# Tensor-based Relative Radiometric Normalization (RRN) and its optimization using TRR and GA (codes and dataset)

This repository contains MATLAB codes and datasets utilized for relative radiometric normalization (RRN) of bi-temporal multi-spectral images in the following papers:
- Armin Moghimi, Turgay Celik & Ali Mohammadzadeh (2022) Tensor-based keypoint detection and switching regression model for relative radiometric normalization of bitemporal multispectral images, International Journal of Remote Sensing, 43:11, 3927-3956, DOI: 10.1080/01431161.2022.2102951) 
- Armin Moghimi, Turgay Celik, Ali Mohammadzadeh, Saied Pirasteh, and Jonathan Li (accepted in IGARS IEEE 2024 but not presented), Optimizing Relative Radiometric Modeling: Fine-tuning strategies using Trust-Region Reflective and Genetic Algorithms for Residual Error Minimization.

- Armin Moghimi, Turgay Celik, Ali Mohammadzadeh, Saied Pirasteh, and Jonathan Li (under review in IEEE GRSL 2024), Optimizing Relative Radiometric Modeling: Fine-Tuning Strategies with Trust-Region Reflective and Genetic Algorithms for Minimizing Residual Errors in Bitemporal Multispectral Imagery. 

## ISPRS ICWG III/IVa "Disaster Management
This dataset is also available in the "Datasets & Opensources" section of # ISPRS ICWG III/IVa "Disaster Management" of ISPRS: 

https://www2.isprs.org/commissions/comm3/icwg-3-4a/datasets/

## Overview
As presented in our papers, the MATLAB code implements relative radiometric normalization methods for unregistered satellite image pairs based on the WSST-SURF detector descriptors. The WSST-SURF has been developed on the great function OPENSURF,  an implementation of SURF (Speeded Up Robust Features)SURF by Dr. Dirk-Jan Kroon (Dirk-Jan Kroon (2024). OpenSURF (including Image Warp) (https://www.mathworks.com/matlabcentral/fileexchange/28300-opensurf-including-image-warp), MATLAB Central File Exchange. 

![Test Image 1](https://github.com/ArminMoghimi/Tensor-based-keypoint-detection/blob/main/Workflow11.jpg)

For code and datasets, please take a look at the supplementary material.

## Dependencies and Environment
The codes are developed and tested in MATLAB R2020a, with both OpenCV.3.4.1 and the VLFeat open-source libraries on a desktop computer with Intel(R) Core (TM) i7 CPU @ 2.40 GHz, 16.00GB RAM, running the Windows 10. To use the codes, you need some prerequisites as follows: 
- 	MATLAB 2018b or upper
- 	OpenCV (3.4.1) jsut for cv.affinetransformation
- 	VLFeat 0.9.21  (https://www.vlfeat.org/)

Having the required build tools for Windows and Visual Studio would be best. Please see https://github.com/kyamagu/mexopencv for instructions on downloading and installing OpenCV on your MATLAB software. Also, please see the https://www.vlfeat.org/ for downloading and installing VLFeat 0.9.21.

## Getting Started
After installing OpenCV 3.4.1 and VLFeat 0.9.21, it is enough to use only main.m for a quick start.
## Acknowledgment

I sincerely thank Dr. Dirk-Jan Kroon and Professor Vedaldi for their codes at every research project stage. 
