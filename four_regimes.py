import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import ipdb
import cartopy.crs as ccrs
import matplotlib
from somoclu import Somoclu
from importlib import reload

import sys; sys.path.insert(0, '/home/pyfsiew/codes/')
import tools
import four_regimes as fr

if __name__ == "__main__":

    ### Read the data
    var = 'H850'
    var = 'H500_regrid_2.5x2.5'
    var = 'H500_regrid_1x1'
    # Read data and remove the seasonal cycle 
    var3d = tools.read_data(var, slicing=False, months='all', limit_year=False, compute=True)
    #var3d_anom = tools.remove_seasonal_cycle_and_detrend(var3d, detrend=True)
    var3d_anom = tools.remove_seasonal_cycle_and_detrend(var3d, detrend=False)
    if False: # apply running mean to the data
        var3d_anom = tools.filter_data(var3d_anom, filter_type='low')
    if False: # Take January as the background state (Falkena et al. 2019)
        mon_mask = var3d.time.dt.month.isin([1])
        climatology = var3d.sel(time=mon_mask).mean(dim='time')
        var3d_anom = var3d - climatology

    ### Select the period
    st_date='1980-06-01';  end_date='2021-08-31'; months = [6,7,8] #JJA
    st_date='1980-12-01';  end_date='2021-02-28';months = [12,1,2]
    data = var3d_anom.sel(time=slice(st_date, end_date))
    mon_mask = data.time.dt.month.isin(months)
    data = data.sel(time=mon_mask)
    data_raw = data.copy() # data will be modified soon. Here we keep the raw data which is not modified by remoiving winter mean or standardization

    if False: # Remove the winter-mean (is this step really necessary?)
        data  = tools.remove_winter_mean(data, months='NDJFM')
    if False: # Standardize grid by grid (to only obtain the spatial pattern rather than the magnitude) 
        data = (data - data.mean(dim='time')) / data.std(dim='time')
    if False: # Pentad-mean the data
        mean_day = 5
        data__rw_pentad = [data.sel(time=slice('%s-12-01'%yr, '%s-02-28'%(yr+1))).coarsen(time=mean_day, boundary='trim').mean() for yr in years]
        data_rw_pentad = xr.concat(data_rw_pentad, dim='time')
        data = data_pentad

    ### select the region
    lon1, lon2, lat1, lat2 = -90, 30, 20, 80 # Atlantic regime - Swinda 2019 RMES (6 clusters in k-mean clustering. Set radius0 as 1)
    lon1, lon2, lat1, lat2 = -60, 90, 60, 90 # Euro-Atlantic Arctic
    lon1, lon2, lat1, lat2 = -180, 180, 60, 90 
    data_xr = data.sel(longitude=slice(lon1,lon2)).sel(latitude=slice(lat1,lat2))
    data_xr_mean = data_xr.mean(dim='time')
    lat_weight = np.sqrt(np.cos(np.radians(data_xr.latitude))) # Do the square cos(lat) weight to each grid
    data_xr_weight = data_xr * lat_weight
    nt,ny,nx = data_xr_weight.shape
    data_reshape = np.reshape(data_xr_weight.values, [nt, ny*nx], order='F')

    ############# Start the run ###############################
    reload(fr)
    seeds = range(0,20)
    ms_ns = [(2,1), (3,1), (4,1), (2,2), (5,1), (6,1), (3,2), (7,1), (8,1), (4,2), (9,1), (3,3), (10,1), (5,2), 
            (11,1), (12,1), (6,2), (4,3), (13,1), (14,1), (7,2), (15,1), (5,3), (16,1), (8,2), (4,4), (17,1), (18,1), (6,3), (9,2), (19,1), (20,1), (5,4), (10,2)]
    ms_ns = [(5,5)]
    ms_ns = [(4,4)]
    seeds = [0]
    for m, n in ms_ns:
        for seed in seeds:
            print(m,n)
            centroid_maps, euclid_dist, node_date, freq = fr.som_fitting(data_reshape, data_xr_weight, m, n, seed=seed)
            nodes = list(node_date.keys())
            centroid_maps_raw = [data_raw.sel(time=node_date[node]).mean(dim='time') for node in nodes]

            if True: # Save the regime in files
                folder = '/home/pyfsiew/codes/som/node_dates/'
                #np.save(folder + 'node_dates_%sx%s_%s_r3_bubble.npy'%(m,n,seed), node_date) # For radius0=4 or 1
                #np.save(folder + 'node_distance_%sx%s_%s_r3_bubble.npy'%(m,n,seed), euclid_dist) # For radius0=4
                #np.save(folder + 'node_dates_%sx%s_%s.npy'%(m,n,seed), node_date)  # The most original one
                np.save(folder + 'node_dates_keeptrend_%sx%s_%s.npy'%(m,n,seed), node_date)  # The most original one

    if False:
        fr.plot_regime_composite(centroid_maps_raw, 3, 3, freq)
        fr.plot_regime_composite(centroid_maps, 3, 3, freq) # Plot the regime (SOM region)
        node_regression, node_correlation = fr.mangitude_regime(node_date, data_raw.sel(longitude=slice(-60,90)).sel(latitude=slice(60,90)), centroid_maps)
        #np.save(folder + 'regime_regressions.npy', node_regression) 
        fr.correlation_regimes_projection_ice(node_date, node_regression)

    if False: # Plot the distribution of minimum distance (This tells us whether a global minimum is reached)
        plt.close()
        fig, ax1 = plt.subplots(1,1,figsize=(3,3))
        ax1.hist(euclid_dist.min(axis=1), bins=20)
        ax1.axvline(np.median(distance.min(axis=1)), color='k')
        ax1.set_xlim(0,30000)
        plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(dt.date.today(), 'SOM_BMUdistance'), bbox_inches='tight', dpi=300)

    #$run -i regime_yearly_frequency(node_timeseries, data_xr.time)
    #%run -i plot_ice_distributions_in_regimes(node_date)
    #%run -i ice_change_in_persist_days
    #%run -i ice_distribution_in_regimes
    #$run -i plot_ice_anomaly
    #%run -i regimes_pdf_in_ice_persistence
    #%run -i regimes_in_ice_terciles
    #%run -i cyclone_frequency_regimes.py
    #%run -i blocking_frequency_regimes
    #$run -i determine_how_many_regimes

def som_fitting(data, data_xr_3d, m, n, seed=None, epochs=50):

    # m in row, n is column
    # seed=None means random initial guess

    time_shape = data.shape[0]
    data_tmean = data_xr_3d.mean(dim='time')
    epochs = 50
    topo = "hexagonal"
    topo = 'rectangular'
    maptype = 'toroid'
    maptype = 'planar'
    
    ### Create some random initial guess for SOM
    rng = np.random.default_rng(seed)
    random_node = np.float32(rng.normal(size=(m*n, data.shape[1])))
    ### Train the SOM map
    somobj = Somoclu(n, m, maptype=maptype, gridtype=topo, initialcodebook=random_node, neighborhood='gaussian')
    somobj.train(data=np.float32(data), epochs=epochs, radius0=3, radiusN=1, radiuscooling='linear', scale0=0.1, scaleN=0.01, scalecooling='linear')
    #somobj = Somoclu(m, n, maptype=maptype, gridtype=topo, initialization='pca', neighborhood='gaussian')
    #somobj.train(data=data, epochs=epochs, radius0=0, radiusN=1, radiuscooling='linear', scale0=0.1, scaleN=0.01, scalecooling='linear') # all-default

    ### Get the clusters map. # The reshape is only correct when it comes with longitude first and then latitude
    centroids = np.reshape(somobj.codebook,(n*m, data_tmean.shape[1], data_tmean.shape[0]))
    centroid_maps = []
    for i in range(n*m):
        centroid = centroids[i].transpose()
        centroid = xr.DataArray(centroid, dims=data_tmean.dims, coords=data_tmean.coords)
        centroid_maps.append(centroid)

    ### Get the Frquency, distance and the best matching unit
    distance = somobj.get_surface_state()
    # The second element is the row; first element shows the column of the n x m matrix
    bmus = somobj.get_bmus(distance)
    nodesum = np.zeros((m,n))
    for t in range(time_shape):
        nodesum[bmus[t,1],bmus[t,0]] += 1
    freq = nodesum/time_shape * 100

    ### Get the node of each day (timeseries)
    nodes = np.arange(1, m*n+1)
    nodes_matrix = nodes.reshape(m, n)
    node_date = {n:[] for n in nodes}
    node_timeseries = []
    for i, day in enumerate(data_xr_3d.time.to_index()):
        row_idx=bmus[i][1]; col_idx=bmus[i][0]
        node_no = nodes_matrix[row_idx, col_idx]
        node_date[node_no].append(day)
        node_timeseries.append(node_no)

    return centroid_maps, distance, node_date, freq


def plot_regime_composite(map_grid, row, col, freq):
    import pandas as pd
    reload(tools)

    grid = row*col
    cmap= 'coolwarm'
    mapcolors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#FFFFFF', '#FFFFFF', '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
    cmap= matplotlib.colors.ListedColormap(mapcolors)
    mapcolor_grid  = [cmap] * grid
    shading_level = np.linspace(-2,2,21)
    shading_level = np.linspace(-100,100,9)
    shading_level = np.linspace(-200,200,9)
    shading_level = np.linspace(-150,150,9)
    shading_level = np.linspace(-200,200,11)
    shading_level_grid = [shading_level] * grid
    clabels_row = [''] * grid
    top_title = [''] * col
    left_title = [''] * row
    ind_titles = None
    leftcorner_text= ['C' + str(i+1) + ':' + str(round(f,1)) +'%' for i, f in enumerate(freq.flatten())]
    projection=ccrs.PlateCarree(central_longitude=160); xsize=3; ysize=2
    projection=ccrs.NorthPolarStereo(); xsize=2; ysize=1.5
    projections=ccrs.Orthographic(central_longitude=0, central_latitude=90); xsize=1.5; ysize=1.5
    xlims = [-180,180]
    ylims = (65,90)
    region_boxes = None
    contour_map_grids = map_grid
    map_grid = [None] * len(map_grid)

    xylims = [((-3500000,4000000), (-4500000,900000))] * grid  # For regime (BKS)
    xylims = None
    tools.map_grid_plotting(map_grid, row, col, mapcolor_grid, shading_level_grid, clabels_row, top_titles=top_title, left_titles=left_title, 
                                ind_titles=ind_titles, projection=projection, xsize=xsize, ysize=ysize, gridline=False, region_boxes=region_boxes,
                                        ylim=ylims, xlim=xlims, set_xylim=xylims,
                                   fill_continent=True, colorbar=False, contour_map_grids=contour_map_grids, contour_clevels=shading_level_grid, leftcorner_text=leftcorner_text,
                                   box_outline_gray=True)

    fig_name = 'SOM_clusters_composite'
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0) # hspace is the vertical
    #plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=-0.53) # hspace is the vertical
    plt.savefig('/home/pyfsiew/graphs/%s_%s.png'%(dt.date.today(), fig_name), bbox_inches='tight', dpi=300, pad_inches=0.01)


def mangitude_regime(node_date, data_xr, node_map):


    nodes = list(node_date.keys())
    node_regression = {n:[] for n in nodes}
    node_correlation = {m:[] for m in nodes}

    for m in nodes:
        for day in node_date[m]:
            # Ymap is the weighted map. So it fits with the regime projection
            Ymap= data_xr.sel(time=day)
            Xmap = node_map[m-1] # index 0 is node 1 is node_map
            regression, correlation = tools.pattern_correlation(Ymap, Xmap)
            node_regression[m].append(regression)
            node_correlation[m].append(correlation)

    return node_regression, node_correlation


