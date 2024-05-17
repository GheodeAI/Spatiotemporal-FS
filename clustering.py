from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.basemap import Basemap
import itertools
import xarray as xr
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import datetime
from matplotlib import pyplot as plt



def filter_xarray(data, min_lat=-90, max_lat=90, min_lon=-180, max_lon=180, months=[1,2,3,4,5,6,7,8,9,10,11,12],resolution=2):
    
    ''' This function filters the data by latitude and longitude'''
    
    #Filtering the data
    filtered_dataset = data.sel(latitude=np.arange(max_lat, min_lat-resolution,-resolution), longitude=np.arange(min_lon, max_lon+resolution,resolution))

    filtered_dataset = filtered_dataset.sel(time=filtered_dataset['time.month'].isin(months))
    
    return filtered_dataset


def perform_clustering(var, months, coord, numbe_of_clusters, norm, seasonal_soothing, first_year,last_year, first_clima,last_clima,resolution,path_predictors,path_output):

    # Define geographical coordinates

    if coord == 'World' or coord == 'all_but_atlantic':
        min_lat=-90
        max_lat=90
        min_lon=-180
        max_lon=180-resolution
    elif coord == 'Europe':
        min_lat=30
        max_lat=70
        min_lon=-16
        max_lon=44
    elif coord == 'North_Atlantic':
        min_lat=0
        max_lat=66
        min_lon=-90
        max_lon=40
    elif coord == 'Artic':
        min_lat=48
        max_lat=90
        min_lon=-180
        max_lon=180-resolution

    if coord == 'all_but_atlantic':
        all_but_atlantic = True
    else:
        all_but_atlantic = False

    # Data extraction from .nc files
    daily_data_train = xr.open_dataset(path_predictors+'data_daily_'+var+'_1950_2010.nc')
    
    daily_data_test = xr.open_dataset(path_predictors+'data_daily_'+var+'_2011_2022.nc')
    
    # Define the variable to perform the clustering
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

    # Perform the cluster only on the train years
    daily_data_train = daily_data_train.sel(time=slice(str(first_year)+'-01-01', str(int(last_year))+'-12-31'))
    data_clima_time = daily_data_train.sel(time=slice(str(first_clima)+'-01-01', str(int(last_clima))+'-12-31'))

    # Data preprocessing
    from clustering import filter_xarray
    # Data is filtered based on the geographical limits, months, resolution and years
    data_filtered = filter_xarray(daily_data_train, min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon, months=months,resolution=resolution)
    data_filtered_clima = filter_xarray(data_clima_time, min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon, months=months,resolution=resolution)

    # Perform the seasonal soothing
    if seasonal_soothing == True:

        year_average = data_filtered_clima.groupby('time.dayofyear').mean('time')
        year_average2 = np.append(np.append(year_average[variable].values, year_average[variable].values,axis=0), year_average[variable].values,axis=0)
        year_average_xarray = xr.DataArray(data=year_average2,dims=["dayofyear", "latitude", "longitude"],)
        year_average_smooth = year_average.rolling(dayofyear=30,min_periods=1, center=True).mean('time')
        year_average_smooth[variable] = year_average_xarray.rolling(dayofyear=30,min_periods=1, center=True).mean('time')[366:732,:,:]
        year_average_smooth_nonleap = year_average_smooth.sel(dayofyear=year_average_smooth['dayofyear']!=60)

        years = data_filtered.groupby('time.year').mean().year.values

        import calendar

        for year in years:
            is_leap_year = calendar.isleap(year)
            year_data = data_filtered.sel(time=data_filtered['time.year'] == year)

            if is_leap_year:
                diff = year_data[variable].values - year_average_smooth[variable].values
            else:
                diff = year_data[variable].values - year_average_smooth_nonleap[variable].values
            year_data[variable] = (('time', 'latitude', 'longitude'), diff)  
            data_filtered[variable].loc[dict(time=data_filtered['time.year'] == year)] = year_data[variable].values

            
    data_filtered_clima.close()

    # Reshape the data 
    data = data_filtered[variable].values
    
    data_res = data.reshape(data.shape[0], data.shape[1]*data.shape[2]).T

    # Apply a mask to the data to remove the ocean points in case of t2m or sm, and remove nan values
    if var == 't2m' or var == 'sm1':
        mask = np.load(path_output+'mask_ocean_europe.npy')
    else:
        mask = ~np.any(np.isnan(data_res), axis=1)

    # Apply a mask to the data to remove the points outside the Atlantic ocean if the coord is all_but_atlantic
    if all_but_atlantic==True:
        mask_2 = xr.where((data_filtered.longitude < -80) | (data_filtered.longitude > 20), True, False)
        mask_2 = mask_2.broadcast_like(data_filtered)

        mask_2 = np.moveaxis(mask_2.values, -1, 0)

        mask_2 = mask_2.reshape(data.shape[0], data.shape[1]*data.shape[2]).T

        mask_2 = mask_2[:,0]

        mask = mask & mask_2

    # Apply a mask to the data to remove the points with no sea ice for sic variable
    if var=='sic':
        
        data_sic = filter_xarray(daily_data_train, min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon, months=months,resolution=resolution)[variable].values

        data_res_sic = data_sic.reshape(data_sic.shape[0], data_sic.shape[1]*data_sic.shape[2]).T

        mask_sic = data_res_sic.sum(axis=1)!=0

        mask = mask & mask_sic

    # Mask the final data
    data_res_masked = data_res[mask]
 
    # Normalize each time series
    if norm==True:
        from sklearn.preprocessing import normalize
        data_res_masked = normalize(data_res_masked, axis=1, copy=True, return_norm=False)

    # Perform the clustering
    from clustering import cluster_model
    cluster = cluster_model(data_res_masked, numbe_of_clusters, var)
    cluster.check_data()
    cluster.kmeans()
    # cluster.agclustering()


    # Get the closest node to the cluster center
    centroids = cluster_model.get_closest2center2(cluster, data_res_masked)

    # Plot the clusters

    north=data_filtered.indexes['latitude'][0] 
    south=data_filtered.indexes['latitude'][-1]
    west=data_filtered.indexes['longitude'][0]
    east=data_filtered.indexes['longitude'][-1]

    cluster_model.plot_clusters(cluster,data_res_masked,mask,north,south,west,east,coord,resolution)

    # Get the data for the centroids 
    
    lat = np.arange( north, south-resolution,-resolution)
    lon = np.arange(west, east+resolution, resolution)

    import itertools
    iter = itertools.product(lat, lon)
    nodes_list=list(iter)
    nodes_list = np.array(nodes_list)[mask]

    lons_c = [np.array(nodes_list)[centroids][i][1] for i in range(len(np.array(nodes_list)[centroids]))]
    lats_c = [np.array(nodes_list)[centroids][i][0] for i in range(len(np.array(nodes_list)[centroids]))]

    # Once the cluster are created, read and process the test data

    data_filtered_test = filter_xarray(daily_data_test, min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon, months=months,resolution=resolution)

    # Apply seasonal forecasting

    if seasonal_soothing == True:

        years = data_filtered_test.groupby('time.year').mean().year.values

        import calendar

        for year in years:
            is_leap_year = calendar.isleap(year)
            year_data = data_filtered_test.sel(time=data_filtered_test['time.year'] == year)

            if is_leap_year:
                diff = year_data[variable].values - year_average_smooth[variable].values
            else:
                diff = year_data[variable].values - year_average_smooth_nonleap[variable].values

            year_data[variable] = (('time', 'latitude', 'longitude'), diff)  
            data_filtered_test[variable].loc[dict(time=data_filtered_test['time.year'] == year)] = year_data[variable].values

    # Merge the train and test data

    data_filtered_total = xr.concat([data_filtered, data_filtered_test], dim='time')
    data_filtered.close()
    data_filtered_test.close()

    # Create a dataframe with the centroids timeseries

    centroids_data = []
    for i in range(len(centroids)):
        centroid_data = data_filtered_total.sel(latitude=lats_c[i], longitude=lons_c[i])[variable].values
        centroids_data.append(centroid_data)

    centroids_dataframe = pd.DataFrame(centroids_data).T
    centroids_dataframe.index = data_filtered_total.time.values
    centroids_dataframe.columns = [var+coord+'_cluster'+str(i) for i in range(1, numbe_of_clusters+1)]


    # Get average data for each cluster, weighted averages are calculated. Batch size is adjusted to avoid memory errors

    data_cl_av = data_filtered_total[variable].values

    clusters_av_dataframe = pd.DataFrame(columns=[var+coord+'_cluster'+str(i) for i in range(1, numbe_of_clusters+1)])

    weights = np.cos(np.deg2rad(nodes_list[:,0]))

    # data_filtered_total.close()
    def weighted_average(data, weights):
        weighted_sum = np.dot(weights, data)
        total_weight = np.sum(weights)
        return weighted_sum / total_weight

    def calculate_weighted_average(data, weights, batch_size):
        num_rows = data.shape[0]
        result = np.zeros((data.shape[1],))
        
        for i in range(0, num_rows, batch_size):
            if i==190000:
                batch_size=878
            batch_data = data[i:i+batch_size]
            batch_weights = weights[i:i+batch_size]
            result += weighted_average(batch_data, batch_weights)
        
        return result / (num_rows / batch_size)

    for i in range(len(centroids)):
        # cluster_data = data_cl_av_masked[cluster.labels==i]
        cluster_mask = cluster.labels==i
        if var == 'sst' or var=='sic' or var=='sm1' or var=='t2m':
            data_cl_av_masked = data_cl_av.reshape(data_cl_av.shape[0], data_cl_av.shape[1]*data_cl_av.shape[2]).T[mask][cluster_mask]
            batch_size = 1000  
            cluster_average = calculate_weighted_average(data_cl_av_masked, weights[cluster_mask], batch_size)
        else:
            mask_cl = mask & cluster_mask
            data_cl_av_masked = data_cl_av.reshape(data_cl_av.shape[0], data_cl_av.shape[1]*data_cl_av.shape[2]).T[mask_cl]
            batch_size = 1000  
            cluster_average = calculate_weighted_average(data_cl_av_masked, weights[cluster_mask], batch_size)    
        # cluster_average = np.average(data_cl_av_masked, weights=weights[cluster_mask],axis=0)
        clusters_av_dataframe[var+coord+'_cluster'+str(i+1)] = cluster_average

    clusters_av_dataframe.index = data_filtered_total.time.values

    # Create a dataframe with the cluster labels

    labels_dataframe = pd.DataFrame(cluster.labels, columns=['cluster'])
    labels_dataframe['nodes_lat'] = np.array(nodes_list)[:,0]
    labels_dataframe['nodes_lon'] = np.array(nodes_list)[:,1]
    labels_dataframe['cluster'] = labels_dataframe['cluster']+1

    # Save the data

    centroids_dataframe.to_csv( path_output+ 'centroids'+var+coord+str(numbe_of_clusters)+'.csv')
    clusters_av_dataframe.to_csv( path_output+ 'averages'+var+coord+str(numbe_of_clusters)+'.csv')
    labels_dataframe.to_csv( path_output+ 'labels'+var+coord+str(numbe_of_clusters)+'.csv')

    return centroids,centroids_dataframe,clusters_av_dataframe,labels_dataframe


class cluster_model:

    def __init__(self, data, n_clusters, name):
        self.data = data
        self.n_clusters = n_clusters
        self.name = name
        self.correct_data_shape = True
        self.labels = None
        self.cluster_centers = None
        self.silhouette_score = None

    def check_data(self):
        if isinstance(self.data, np.ndarray) and len(self.data.shape) == 2:
            print('Data is a 2D numpy array')
            print('Please, be sure the data is in the correct format: (n_samples (nodes), n_features (variables (time))')
            self.correct_data_shape = True
        else:
            print('Data is not a 2D numpy array')
            self.correct_data_shape = False
        
    def kmeans(self):
        self.cluster = KMeans(n_clusters=self.n_clusters, random_state=0, init='k-means++',n_init=10, tol=0.0001)
        self.cluster.fit(self.data)
        self.labels = self.cluster.labels_
        # self.silhouette_score = silhouette_score(data, self.labels)
        self.cluster_centers = self.cluster.cluster_centers_
        self.silhouette_score = silhouette_score(self.data, self.labels)
        # self.cluster_centers = self.cluster.cluster_centers_
        # self.predictions = self.cluster.predict(data)

    def agclustering(self):
        self.cluster = AgglomerativeClustering(n_clusters=self.n_clusters)
        self.cluster.fit(self.data)
        self.labels = self.cluster.labels_
        

    def dendogram(self, method='single', metric='euclidean' ):
        self.linkage_matrix = linkage(self.data, method=method, metric=metric)
        plt.figure(figsize=(20, 15))
        plt.title("Dendrograms")
        dendrogram(self.linkage_matrix)
        plt.show()

    def get_closest2center2(self, data):
        index = [np.argmin(np.linalg.norm(data[self.labels==i] - self.cluster_centers[i], ord=2, axis=1)) for i in range(self.cluster_centers.shape[0])]
        absolute_indexes = [np.where(self.labels == i)[0][closest_index] for i, closest_index in enumerate(index)]
        print('Index of the closest cluster center for each sample', absolute_indexes)
        return absolute_indexes
    
    def get_closest2center(data, cluster_centers, labels):
        index = [np.argmin(np.linalg.norm(data[labels == i] - cluster_centers[i], ord=2, axis=1)) for i in range(cluster_centers.shape[0])]
        index_data = [np.argmin(np.linalg.norm(data[index[i]] - data, ord=2, axis=1)) for i in range(len(index))]
        print(index_data)
        return index_data

    def get_mean_clusters(self):
        mean_clusters = np.zeros((self.n_clusters))
        for i in range(self.n_clusters):
            mean_clusters[i] = np.mean(self.data[self.labels==i], axis=0)
        return mean_clusters

    def plot_clusters(cluster,data,mask,north,south,west,east,coord,resolution):
        if north == 90 and south == -90:
            s = 20
        else:
            s = 50
        plt.figure(figsize=(20, 10))
        if coord == 'Artic':
            map = Basemap(projection='npstere',boundinglat=48,lon_0=270,resolution='l')
        else:
            map = Basemap(projection='cyl',llcrnrlat=south,urcrnrlat=north, llcrnrlon=west,urcrnrlon=east,resolution='c')
        # Draw coastlines and fill continents
        map.drawcoastlines(linewidth=2)
        # map.fillcontinents(color='white', lake_color='white')
        lat = np.arange( north, south-resolution,-resolution)
        lon = np.arange(west, east+resolution, resolution)
        
        iter = itertools.product(lat, lon)
        nodes_list=list(iter)
        nodes_list = np.array(nodes_list)[mask]
        lons = [nodes_list[i][1] for i in range(len(nodes_list))]
        lats = [nodes_list[i][0] for i in range(len(nodes_list))]
        x, y = map(lons, lats)
        num_clusters = len(np.unique(cluster.labels))
        if num_clusters>10:
            cmap = plt.cm.get_cmap('tab20', num_clusters)
        else:
            cmap = plt.cm.get_cmap('tab10', num_clusters)
        map.scatter(x, y,c=cluster.labels, cmap=cmap, s=s, )
        # Add colorbar
        bounds = np.arange(num_clusters + 1) - 0.5
        cbar = plt.colorbar(ticks=np.arange(num_clusters), boundaries=bounds)
        cbar.set_ticklabels(np.arange(num_clusters)+1)
        cbar.ax.tick_params(labelsize=15)
        cbar.set_label('Cluster',fontsize=20)
        centroids = cluster_model.get_closest2center2(cluster, data)
        
        lons_c = [np.array(nodes_list)[centroids][i][1] for i in range(len(np.array(nodes_list)[centroids]))]
        lats_c = [np.array(nodes_list)[centroids][i][0] for i in range(len(np.array(nodes_list)[centroids]))]
        x_c, y_c = map(lons_c, lats_c)
        map.scatter(x_c, y_c,marker='x',linewidth=4, cmap=cmap, s=300,color='black') 
        # plt.show()
        # Set plot title
        plt.title(cluster.name,fontsize=25)
        plt.xlabel('Longitude',fontsize=20)
        plt.ylabel('Latitude',fontsize=20)
        plt.xticks(np.arange(west, east+2, 10),fontsize=15)
        plt.yticks(np.arange( north, south-2,-10),fontsize=15)
        # Show the plot
        plt.show()


        # if north == 90 and south == -90:
        #     s = 20
        # else:
        #     s = 100

        # plt.figure(figsize=(20, 10))
        # if coord == 'Artic':
        #     map = Basemap(projection='npstere',boundinglat=48,lon_0=270,resolution='l')
        # else:
        #     map = Basemap(projection='cyl',llcrnrlat=south,urcrnrlat=north, llcrnrlon=west,urcrnrlon=east,resolution='c')

        # # Draw coastlines and fill continents
        # map.drawcoastlines(linewidth=0.5)
        # map.fillcontinents(color='white', lake_color='white')

        # lat = np.arange( north, south-2,-2)
        # lon = np.arange(west, east+2, 2)
        
        # iter = itertools.product(lat, lon)
        # nodes_list=list(iter)

        # nodes_list = np.array(nodes_list)[mask]

        # lons = [nodes_list[i][1] for i in range(len(nodes_list))]
        # lats = [nodes_list[i][0] for i in range(len(nodes_list))]

        # x, y = map(lons, lats)
        # num_clusters = len(np.unique(cluster.labels))
        # if num_clusters>10:
        #     cmap = plt.cm.get_cmap('tab20', num_clusters)
        # else:
        #     cmap = plt.cm.get_cmap('tab10', num_clusters)
        # map.scatter(x, y,c=cluster.labels, cmap=cmap, s=s, )

        # # Add colorbar
        # bounds = np.arange(num_clusters + 1) - 0.5
        # cbar = plt.colorbar(ticks=np.arange(num_clusters), boundaries=bounds)
        # cbar.set_ticklabels(np.arange(num_clusters)+1)
        # cbar.set_label('Cluster')


        # centroids = cluster_model.get_closest2center(cluster, data)
        
        # lons_c = [np.array(nodes_list)[centroids][i][1] for i in range(len(np.array(nodes_list)[centroids]))]
        # lats_c = [np.array(nodes_list)[centroids][i][0] for i in range(len(np.array(nodes_list)[centroids]))]
        # x_c, y_c = map(lons_c, lats_c)
        # map.scatter(x_c, y_c,marker='x',linewidth=4, cmap=cmap, s=300,color='black') 
        # # plt.show()



        # # Set plot title
        # plt.title(cluster.name,fontsize=20)
        # plt.xlabel('Longitude')
        # plt.ylabel('Latitude')
        # plt.xticks(np.arange(west, east+2, 10))
        # plt.yticks(np.arange( north, south-2,-10))

        # # Show the plot
        # plt.show()
