## Kurdyukov Alexander (BS21-AI-01, a.kurdyukov@innopolis.university)
### This repo is an unofficial fork of the [TorchGlocalK](https://github.com/fleanend/TorchGlocalK) repository
#### All main references can be found in `references/` directory
 
### To use this repo (in the way its creator was) it is needed to have an ability to run jupyter notebooks. This can be achieved in several ways:
1) Use [Google Colab](https://colab.research.google.com) and manualy config the kernel 
2) Create [Python virtual environment](https://docs.python.org/3/tutorial/venv.html) and [run local jupyter server](https://www.codecademy.com/article/how-to-use-jupyter-notebooks) with [confugured kernel](https://ipython.readthedocs.io/en/stable/install/kernel_install.html) (Recommended)

### If the user will choose 2nd option:
1) Navigate the folder with cloned repo in CMD or PowerShell 
2) Crete Python virtual environment and activate it `(On this step it is considered that the user has Python and Pip installed on his machine)`
3) The creator of this repo was using Python of version 3.9
4) Install all required packages that are listed [here](https://github.com/fleanend/TorchGlocalK/blob/main/README.md#3-requirements)
5) **Please note, that it is needed to install [PyTorch](https://pytorch.org/get-started/locally/) in appropriate for the user configuration manually!**
6) In activated virtual environment execute **```ipython kernel install --name "NAME_OF_THE_KERNEL" --user```** to add virtual environment as Python Kernel and to further use it to run jupyter notebooks
7) Execute **```jupyter notebook```** and wait untill the browser window with navigated repo will be opened
8) Enter user password (if needed) and navigate in the opened window **`/notebooks`** subdirectory and open any wanted notebook
9) In **`Kernels`** tab choose kernel from step **`4)`** 

### **`At this moment user should have an ability to run jupyter notebook's code blocks in any manner`**
#### Now if the user wants to reproduce the results or just test the solution then he/she can just run all the code cells in the `1.0-Glocal_K.ipynb` notebook. 

### Visualization of results of trainig:
#### If it is the case that the user would like to visualize results of fine-tuning phase then in the same-named code cell there declared and commented 4 lists for RMSEs and NDCGs. After runnning this cell you can add an extra cell right after it and use any plotting package([Matplotlib](https://matplotlib.org/stable/users/getting_started/) for example) to build graphs. 
#### Original visualizatons can be found in `reports/` directory.

### Getting recommendations:
#### At the end of the `1.0-Glocal_K.ipynb` notebook there are 3 code cells that can be run by the user (in the same way as all previous were `=)`). The user can follow the instructions to get top K movie IDs as recommendations for the specified user from the dataset.