import numpy as np
import pandas as pd

def lagged_dataset(path,target_name,predictors_name,first_year,last_year,time_lag,sequence_length,file_name):
    target = pd.read_csv(path + target_name,index_col=0)
    target.index = pd.to_datetime(target.index)
    predictors = pd.read_csv(path + predictors_name,index_col=0)
    predictors.index = pd.to_datetime(predictors.index)


    time_lags = np.repeat(time_lag,predictors.shape[1])           # Set the first time lag considered for each predictor respect to the target
    sequence_lengths = np.repeat(sequence_length,predictors.shape[1])       # Set the sequence length for each predictor  

    # Filter the target to the years of interest
    target = target[(target.index.year>=first_year) & (target.index.year<=last_year)]

    # Check that first year of predictors is on less than the first year of target (necessary to compute higher lags than 180)

    if predictors.index[0].year >= target.index[0].year:
        raise ValueError('The first year of the predictors should be at least one less than the first year of the target to consider higher lags than 180.')


    # Create an empty dataset with the target dates
    lagged_dataset = target.copy()

    # Add a column for each time lag of each predictor
    for i,col in enumerate(predictors.columns):
        for j in range(sequence_lengths[i]):
            lagged_dataset[str(col)+'_lag'+str(time_lags[i]+j)] = predictors[col].shift(time_lags[i]+j)

    #Save the dataset

    lagged_dataset.to_csv(path+file_name+'.csv')