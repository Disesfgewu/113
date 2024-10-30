from datetime import datetime
import time

from loading_data import *
from normalize import *
from LSTM_model import *
# from regression import *
from VotingRegressor import *
# from ExtraTreesRegressor import *
from forcast import *

def main():
    start_time = time.time()
    #adding_title( "./data/ExampleTrainData(AVG)" )
    #adding_title( "./data/ExampleTrainData(IncompleteAVG)" )
    SourceData = loading_data( "./data/ExampleTrainData(AVG)" , True)
    AllOutPut = LSTM_data( SourceData ) 
    Regression_X_train , Regression_y_train = regression_data( SourceData )
    x_train , y_train = normal( AllOutPut , 12 )
    X_train = reshape( x_train )
    
    NowDateTime = datetime.now().strftime("%Y-%m")
    print( X_train )
    print( y_train )
    train( X_train , y_train , NowDateTime , batch_size = 128 , epochs = 100 )
    
    regression_modal( NowDateTime , AllOutPut , Regression_X_train , Regression_y_train )
    
    forcast( AllOutPut = AllOutPut , lstm = './model/WheatherLSTM_' + NowDateTime + '.h5' , regression_model = './model/WheatherRegression_' + NowDateTime )
    
    end_time = time.time() 
    execution_time = end_time - start_time
    
    print( f"exex time : {execution_time}") 
main()
