# RadioShift
The Matlab code provided in this repositoy is closely derived from the Modulation Classification by Using FPGA

https://www.mathworks.com/help/releases/R2022b/deep-learning-hdl/ug/modulation-classification-by-using-FPGA.html

The folder consists of a Matlab Live file, ModClassification_ShiftSignalGeneration.mlx, along with helper files used throughout the program. The first half of this Matlab Live serves as a step by step demonstration on how new modulation schemes were introduced and provides visualizations in the form of IQ Constellation and Time Series Plots. The second half of this file provides the steps for generating signal traces with specified parameters for all 18 modulation schemes. Not only can users genereate seven new digital modulation schemes, but sampling error offsets can be carefully manipulated to simulated real-world wireless communication impairements.

## Tutorial Video and Document

We create a tutorial video regarding how to use our code to generate simulated RF data. The link is below  

https://ceas.mediaspace.kaltura.com/media/Tutorial_RF_Data_Generation/1_yl3mvlne 

https://ceas.mediaspace.kaltura.com/media/Tutorial_RF_Simulated_Data_Generation_Deep_Dive_Part+1/1_54lu3c06 
https://ceas.mediaspace.kaltura.com/media/Tutorial_RF_Simulated_Data_Generation_Deep_Dive_Part+2/1_a415g8sr
https://ceas.mediaspace.kaltura.com/media/Tutorial_RF_Simulated_Data_Generation_Deep_Dive_Part+3/1_nqntu9pe

We also provide a tutorial document (ready only) as well. The link is below  

https://www.overleaf.com/read/qgkzqgmshhbc#56783d 

## Example Dataset 

The dataset below is an example of simulated RF dataset that we generated using this tutorial (15 modulations, SNR 0dB to 30dB, 1024 samples per I/Q frames, 0 ppm).  

https://mailuc-my.sharepoint.com/:f:/g/personal/wang2ba_ucmail_uc_edu/EvKqiU3it-dCpseB6lj337kBe3d7YQBRPhXNVgiJkwiDQw?e=zOqzue
