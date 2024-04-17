# XAI_FDA
Interpretable operator learning and posthoc/by-design XAI

Codes and data used in the examples presented in the paper:\
Interpreting and generalizing deep learning in physics-based problems with functional linear models \ 
https://arxiv.org/abs/2307.04569  


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \
Matlab codes: Matlab implementation uploaded. \
Python codes: To be added in the future. 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \
Instructions:\
Run the main.m file. Flag_nn_approx should be set to 1 for NN-driven results and 0 for data-driven results. \
Test case 1: Flag_method = 3 for EMNIST results and Flag_method = 1 for MNIST results. 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \
Data:\
The input training and test data (based on the data as well as the probed neural network) are available here: \
https://drive.google.com/drive/folders/1lUkeI_QE9GZRx5APfZ_mMla04qr4xdg-?usp=drive_link

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \
Using the codes on your own data:\
A sample Pytorch code is placed under pt2mat folder where you can see how  the *.pt files from Pytorch can be loaded and probed for generating the NN-driven data. In this code, you can also see how in this case vtu files (simulation results from FEniCS) are loaded to create the input/output training data.  
