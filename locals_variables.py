import numpy as np
import pandas as pd
import xarray as xr

def main(first_year_train, last_year_test, local_latitude, local_longitude, first_clima, last_clima, path_predictors, path_output):

    # Make a datarange from first to last year and create a dataframe with that index
    date_range = pd.date_range(str(first_year_train), str(last_year_test+1), freq='D', closed='left')


    # Create a dataframe to store the data
    locals = pd.DataFrame(index=date_range)
    for var in ['msl','olr','sm1','t2m','tp','z500']:
        if var == 'sm1':    
            variable = 'swvl1'
        elif var == 'sic':
            variable = 'siconc'
        elif var == 'olr':
            variable = 'mtnlwrf'
        elif var == 'z500':
            variable = 'z'
        else:
            variable = var

        # Load the data and select local node
        daily_data_train = xr.open_dataset(path_predictors+'data_daily_'+var+'_1950_2010.nc')
        daily_data_test = xr.open_dataset(path_predictors+'data_daily_'+var+'_2011_2022.nc')

        data_train_como = daily_data_train.sel(latitude=local_latitude, longitude=local_longitude, method='nearest')
        data_test_como = daily_data_test.sel(latitude=local_latitude, longitude=local_longitude, method='nearest')

        # Remove the seasonal average
        daily_data_clima = data_train_como.sel(time=slice(str(first_clima), str(last_clima)))

        year_average = daily_data_clima.groupby('time.dayofyear').mean('time')
        year_average2 = np.append(np.append(year_average[variable].values, year_average[variable].values,axis=0), year_average[variable].values,axis=0)
        year_average_xarray = xr.DataArray(data=year_average2,dims=["dayofyear"],)
        year_average_smooth = year_average.rolling(dayofyear=30,min_periods=1, center=True).mean('time')
        year_average_smooth[variable] = year_average_xarray.rolling(dayofyear=30,min_periods=1, center=True).mean('time')[366:732]
        year_average_smooth_nonleap = year_average_smooth.sel(dayofyear=year_average_smooth['dayofyear']!=60)

        years = data_train_como.groupby('time.year').mean().year.values

        import calendar

        for year in years:
            is_leap_year = calendar.isleap(year)
            year_data = data_train_como.sel(time=data_train_como['time.year'] == year)

            if is_leap_year:
                diff = year_data[variable].values - year_average_smooth[variable].values
            else:
                diff = year_data[variable].values - year_average_smooth_nonleap[variable].values
            year_data[variable] = (('time'), diff)  
            data_train_como[variable].loc[dict(time=data_train_como['time.year'] == year)] = year_data[variable].values
        
        years = data_test_como.groupby('time.year').mean().year.values

        import calendar

        for year in years:
            is_leap_year = calendar.isleap(year)
            year_data = data_test_como.sel(time=data_test_como['time.year'] == year)

            if is_leap_year:
                diff = year_data[variable].values - year_average_smooth[variable].values
            else:
                diff = year_data[variable].values - year_average_smooth_nonleap[variable].values
            year_data[variable] = (('time'), diff)  
            data_test_como[variable].loc[dict(time=data_test_como['time.year'] == year)] = year_data[variable].values

        locals[var] = np.concatenate([data_train_como[variable].values, data_test_como[variable].values])

    locals.to_csv(path_output+'locals_variables.csv')