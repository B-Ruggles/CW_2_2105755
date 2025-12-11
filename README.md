# Galaxy Classification, Comparing Traditional and Neural Network Approach.



## Overview
Using the Galaxy10 decals data-set, this project aims to test and compare a traditional random forest classifier and a Convolutional Neural Network (CNN) at the task of classifying galaxy morphology.  The project is structured in the following way. 

* Q1 - Traditional ML method (Random Forest Classifier) 
* Q2 - Basic Convolutional Neural Network 
* Q3 - Improving on Convolutional Neural Network using Data Augmentation.

The first two questions focus on the specifics of each method, question 3 focuses solely on how manipulating the images can improve the network. 



 

## The Data-set

The Galaxy10 DECALS data-set is a set of ~17,000 full colour images taken by the DESI Legacy survey. Images of galaxies have have been placed in to 10 classes (a list of 10 classes can be found in q1_main.py). Each image is made up of 256x256 pixels. 

Full details of the data-set can be found here: https://astronn.readthedocs.io/en/latest/galaxy10.html 

## Requirements 

``` 
numpy
matplotlib
pandas
scikit-learn
torch
torchvision
tqdm
astroNN
ipykernel 
```

## Usage 

### Folder Structure 

```
py/
├── functions.py
├── get_images.py
├── __pycache__
│   └── functions.cpython-312.pyc
├── Q1
│   └── main_q1.ipynb
├── Q2
│   ├── best_model.pt
│   └── main_q2.ipynb
└── Q3
    ├── best_model_q3.pt
    └── main_q3.ipynb
```

* Each part of the project is in notebook format. The notebooks are designed to be self explanatory, you can simply click **Run All** or you can step through each cell for a full tutorial.  
* Questions 2 and 3 focus on training CNNs, depending your hardware and choice of hyperparameters, training can take 2-4 hours. 
* It is not recommended to run the CNN models without the use of CUDA. 
* If you do not want to spend the time training the network yourself, you can use the pretrained models found in the Q2 and Q3 folders. 

### How to run 

```
#set up virtual environment 
Python3 -m venv venv
# activate virtual environment 
source venv/bin/activate
#open notebook
jupyter lab
```









