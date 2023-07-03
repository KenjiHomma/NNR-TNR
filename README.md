# NNR-TNR

This is a sample python code of nuclear norm regularized (NNR) loop optimization for tensor network renormalization (TNR) (arXiv:2306.17479). 

# Requirements
- Python 3
- numpy
- scipy
- matplotlib
- ncon
  
# How to run
After installing the packages, you can run this code as, 
 ```
python3 NNR-TNR.py 
 ```
By default, The main file "NNR-TNR.py" produces the relative error of free energy density, conformal data and singular value spectrums of the crtical 2D ising model with Bond dimension $\chi =8$. The computation might take some times depending on the environment. (It ends in less than a minute for MacBook Air M1, 2020.) Once computation is done, RG step dependences of the conformal data and singualr value spectrums will be plotted and saved in /CFTdata001 and /spectrum001, respectively.

To change some of parameters, such as bond dimensions, number of RG steps, temperature, etc ..., one can add these parameters at the command line. For example, to increase the bond dimension to $\chi=16$
 ```
python3 NNR-TNR.py 16
 ```
For the detail of NNR loop optimizattion, please read our paper in arXiv:2306.17479.
