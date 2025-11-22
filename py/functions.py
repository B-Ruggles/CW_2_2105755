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

