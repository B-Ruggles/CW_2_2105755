import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score 
# Function to get pandas dataframe of gaia star data
def get_data():
    data_path = "/home/benr/ACT/CW2/gz2_hart16.csv.gz"
    df = pd.read_csv(data_path)
    return df 

#Function to split test and training data
def split_data_rf(X1,X2,Y,test_size):
    return train_test_split(X1,X2,Y,test_size=test_size,random_state=11)

#categorise galaxies 
# put each galaxy into a catagory based on the above classifaction 
def galaxy_type(galaxy):
 # distinct galaxy features 
 smooth = 't01_smooth_or_features_a01_smooth_debiased'
 disk   = 't01_smooth_or_features_a02_features_or_disk_debiased'
 edge_on = 't02_edgeon_a04_yes_debiased'
 spiral = 't04_spiral_a08_spiral_debiased'
 bar = 't03_bar_a06_bar_debiased'
 # categorise based on debiased weights 
 if galaxy[smooth] > 0.7:
    return 'E'
 if galaxy[edge_on] > 0.6:
   return 'S(edgeon)'
 if galaxy[disk] > 0.6 and galaxy[spiral] < 0.4:
   return 'S0'
 if galaxy[spiral] > 0.4:
   if galaxy[bar] > 0.5:
     return 'SB'
   else:
     return 'S (a b c)'
 return None 