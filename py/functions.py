from astroNN.datasets import galaxy10
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score 
import torch.optim as optim 
import torch

# Function to get pandas dataframe of gaia star data
def get_data():
    images, labels = galaxy10.load_data()
    return images,labels 

#Function to split test and training data
def split_data_rf(X1,X2,Y,test_size):
    return train_test_split(X1,X2,Y,test_size=test_size,random_state=11)

#categorise galaxies 
# put each galaxy into a catagory based on the above classifaction 
def galaxy_type(g):
    sm = g['t01_smooth_or_features_a01_smooth_debiased']
    fe = g['t01_smooth_or_features_a02_features_or_disk_debiased']
    ed = g['t02_edgeon_a04_yes_debiased']
    sp = g['t04_spiral_a08_spiral_debiased']
    bar = g['t03_bar_a06_bar_debiased']

    if sm >= 0.8:
        return 'E'

    if sp >= 0.7 and fe >= 0.5:
        return 'Spiral'

    # Any reasonably clear disk (including edge-on) → Disk
    if fe >= 0.7 or ed >= 0.6:
        return 'Disk'

    return None

def split_data_torch(X,Y,I ):
  trn_idx, tst_idx = train_test_split(I,test_size=0.1,
                                      stratify= Y[I],random_state=11)
  xtrn = X[trn_idx]  
  xtst = X[tst_idx]
  ytrn = Y[trn_idx]
  ytst = Y[tst_idx]
  return xtrn,xtst,ytrn,ytst

def train(model,trainLoader,testLoader,criterion,optimizer,device,n_epoch):
  for e in range(n_epoch):
    #use model in training mode 
    model.train()
    #increment loss for each image 
    running_loss = 0.0
    # number of correct labels
    correct = 0
    # total number of samples 
    total = 0
    for imgs,lbl in trainLoader:
      # move image and label to GPU 
      imgs = imgs.to(device)
      labels = lbl.to(device)
      #clear old gradients 
      optimizer.zero_grad()
      # predict labels 
      outputs = model(imgs)
      # workout the loss for the prediction 
      loss = criterion(outputs,labels)
      # use backpropagation to calculate new weights 
      loss.backward()
      # update weights 
      optimizer.step()
      # get batch loss 
      running_loss += loss.item() * imgs.size(0)
      # return position of most likely catagory for each image in batch 
      _,preds = torch.max(outputs,1)
      # count the correctly predicted samples 
      correct += (preds == labels).sum().item()
      # increae the total samples 
      total += labels.size(0)
    # calculate train_loss for epoch 
    trn_loss = running_loss / total
    # calculate train accuracy 
    trn_acc = correct / total 
    # test trained model 
    test_acc = evaluate(model, testLoader, device)

    print(f"Epoch {e+1}/{n_epoch} "
        f"Train loss: {trn_loss:.4f}  "
        f"Train acc: {trn_acc:.3f}  "
        f"Test acc: {test_acc:.3f}")

def evaluate(model,testLoader,device):
  model.eval()
  correct = 0
  total = 0 
  # stops gradient calculations for testing 
  with torch.no_grad():
    for imgs,lbl in testLoader:
      imgs = imgs.to(device)
      labels = lbl.to(device)
      outputs = model(imgs)
      _,preds = torch.max(outputs,1)
      # count the correctly predicted samples 
      correct += (preds == labels).sum().item()
      # increae the total samples 
      total += labels.size(0)
    return correct / total if total > 0 else 0.0
  
def new_size(H,k,p,s):
  cnn_size = ((H+(2*p) - k) / s) + 1
  pool_size = ((H-K)/s) + 1
  return (cnn_size, pool_size)