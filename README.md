# RadioShift: A Framework Measuring the Robustness of Lightweight Neural Networks over RF Signals

We build an end-to-end framework, named RadioShift, which can (a) automatically generate simulated I/Q frames with various domain shifts, (b) capture real-world I/Q frames, (c) train lightweight neural networks, and (c) test them on FPGAs given the RF signals we collect. The source code of our framework are divided into three separate repositories (as each of them can be used independently).

https://github.com/UCdasec/RadioShift  (Simulated RF Signal Generation)

https://github.com/UCdasec/RadioShift-SDR (Real-world RF Signal Generation)

https://github.com/UCdasec/Radio-ML  (Train Neural Networks on GPUs and Test Them on FPGAs)

## Overview:
This repository contains code and documents for generating simulated RF signals with different channel fading,  hardware imperfections, SNRs, etc. by using Matlab. We use it to generate simulated RF signals from different modulations (e.g., FM, BPSK, QPSK, etc.) 

## Reference
We use the source code and documents of this repository to collect simulated RF signals, which are used in the following paper.  

Anagh Mishra, Phu Le, Ryan Evans, Nirnimesh Ghose, Boyang Wang, "RadioShift: A Framework Measuing the Robustness of Lightweight Neural Networks over RF Signals," the IEEE National Aerospace and Electronics Conference (IEEE NAECON 2026), Cincinnati, OH, August 9-12, 2026, USA.

To completely reproduce research results in the above paper, one will need to use all the three repos mentioned above. The **RadioShift Dataset** used in our paper can be found below (last modified: July 2026). The simulated RF signals were generated using the code from this repo. The real-world RF signals were generated using the code from our RadioShift-SDR repo.  

https://mailuc-my.sharepoint.com/:f:/g/personal/wang2ba_ucmail_uc_edu/IgBnktfy-5hVT5pE_rpI9Y16AYZdUtU4ldAIK7P2yyZSo9w?e=fqZN27 

Note: the above link need to be updated every 6 months due to certain settings of OneDrive. If you find the links are expired and you cannot access the data, please feel free to email us (Dr. Boyang Wang, boyang.wang@uc.edu). We will be update the links as soon as we can (typically in 1~2 days). Thanks!

## Background

The Matlab code provided in this repositoy is closely derived from the Modulation Classification by Using FPGA

https://www.mathworks.com/help/releases/R2022b/deep-learning-hdl/ug/modulation-classification-by-using-FPGA.html

The folder consists of a Matlab Live file, ModClassification_ShiftSignalGeneration.mlx, along with helper files used throughout the program. The first half of this Matlab Live serves as a step by step demonstration on how new modulation schemes were introduced and provides visualizations in the form of IQ Constellation and Time Series Plots. The second half of this file provides the steps for generating signal traces with specified parameters for all 18 modulation schemes. Not only can users genereate seven new digital modulation schemes, but sampling error offsets can be carefully manipulated to simulated real-world wireless communication impairements.

## Tutorial Video and Document

We create a tutorial video regarding how to use our code to generate simulated RF data. The link is below  

https://ceas.mediaspace.kaltura.com/media/Tutorial_RF_Data_Generation/1_yl3mvlne 

https://ceas.mediaspace.kaltura.com/media/Tutorial_RF_Simulated_Data_Generation_Deep_Dive_Part+1/1_54lu3c06 
https://ceas.mediaspace.kaltura.com/media/Tutorial_RF_Simulated_Data_Generation_Deep_Dive_Part+2/1_a415g8sr
https://ceas.mediaspace.kaltura.com/media/Tutorial_RF_Simulated_Data_Generation_Deep_Dive_Part+3/1_nqntu9pe

We also provide a tutorial document (RF_Data_Generation_Matlab.pdf) as well. 

## Example Dataset 

The dataset below is an example of simulated RF dataset that we generated using this tutorial (15 modulations, SNR 0dB to 30dB, 1024 samples per I/Q frames, 0 ppm).  

https://mailuc-my.sharepoint.com/:f:/g/personal/wang2ba_ucmail_uc_edu/EvKqiU3it-dCpseB6lj337kBe3d7YQBRPhXNVgiJkwiDQw?e=zOqzue
