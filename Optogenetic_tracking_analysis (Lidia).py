# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 18:50:18 2019
Code to analyse optogenetic experiments data from ontogenetic tracking videos with observed reversal differences:
    -Put all the FeatureN.HDF5 files for the same strain in one folder 
    -Change the path in datafolder for the path to your own folder, always with forward slashes, even if it's Microsoft computer'
    -Change in the section name the strains the names of nStrain1, 2 and nControl1, 2 to your strain names 
    
@author: Lidia
"""
import numpy as np
import pandas as pd
import statsmodels.stats.multitest as smm
from scipy import stats as sts
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import seaborn as sns
import os
from pathlib import Path
from numpy.linalg import norm
from scipy.spatial import distance
import numpy.ma as ma

#Import all HDF5 files from one strain (folder) into one variable and extract tables from them one by one. Be careful to be in the file path, add r'C: to file paths if run in Windows
datafolder = Path('/Users/lidia/Desktop/Optogenetics_Iris/') # this structure allows you to not have to change the rest of the paths in the code, it adapts them to Windows and Mac
Strain1 = sorted(list(datafolder.glob('Strain1(4707)/*_featuresN.hdf5')))# import all HDF5 FeaturesN files for interesting strain sorted in alphabetical order
Control1 = sorted(list(datafolder.glob('Control1/*_featuresN.hdf5')))# import all HDF5 FeaturesN files for control
Strain2 = sorted(list(datafolder.glob('Strain2(4838)/*_featuresN.hdf5')))# import all HDF5 FeaturesN files for interesting strain
Control2 = sorted(list(datafolder.glob('Control2/*_featuresN.hdf5')))# import all HDF5 FeaturesN files for control
Stimuli = pd.read_excel(datafolder / 'Stimulus_time.xlsx', sheet_name = 'All2', header = 1) #import the excel file with the mannually gathered stimuli points

#Creates a folder to store the results from the analysis 
newfolder = datafolder / 'Results'
if not os.path.exists(newfolder): #this prevents creating the folder if it already exists
    os.makedirs(newfolder)
    
#name the strains, make sure to leave a space before the name 
nStrain1 = ' AQ4707'
nControl1 = ' AQ4707 no ATR'
nStrain2 = ' AQ4838'
nControl2 = ' AQ4838 no ATR'
    
#Select timeseries data from files, adding all files from different videos for the same strain onto the same dataframe. To speed this part and avoid having to repeat the code I could create a def function and use map to apply it over the two Strain folders
Strain1_feat = pd.DataFrame() #dataframe to store information 
temporary1 = []                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
for iteration, i in enumerate(Strain1): 
    data = pd.read_hdf(i, 'timeseries_data') #select all timeseries_data tables from the hdf5 documents in the folder
    data = data.assign(Video = iteration)
    temporary1.append(data) #append all data
Strain1_feat = pd.concat(temporary1, ignore_index = True) #concatenate next to each other all the tables for all the files in the folder
Control1_feat = pd.DataFrame()
temporary2 = []                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
for iteration, i in enumerate(Control1): 
    data = pd.read_hdf(i, 'timeseries_data')
    data = data.assign(Video = iteration)
    temporary2.append(data)
Control1_feat = pd.concat(temporary2, ignore_index = True)
Strain2_feat = pd.DataFrame()
temporary3 = []                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
for iteration, i in enumerate(Strain2): 
    data = pd.read_hdf(i, 'timeseries_data')
    data = data.assign(Video = iteration)
    temporary3.append(data)
Strain2_feat = pd.concat(temporary3, ignore_index = True)
Control2_feat = pd.DataFrame()
temporary4 = []                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
for iteration, i in enumerate(Control2): 
    data = pd.read_hdf(i, 'timeseries_data')
    data = data.assign(Video = iteration)
    temporary4.append(data)
Control2_feat = pd.concat(temporary4, ignore_index = True)
#%% FIRST PART OF THE CODE : Calculate jensen shannon distance to discover the most significantly different features

Str1 = Strain1_feat[Strain1_feat.columns.difference(['worm_index'])].groupby ('timestamp').mean() #calculate the mean for each timestamp, for each feature from the different worms    
Cont1 = Control1_feat[Control1_feat.columns.difference(['worm_index'])].groupby ('timestamp').mean()#calculate the mean for each timestamp, for each feature from the different worms  
Str1_Cont1 = pd.merge(Str1, Cont1, on = 'timestamp')
Str2 = Strain2_feat[Strain2_feat.columns.difference(['worm_index'])].groupby('timestamp').mean()#calculate the mean for each timestamp, for each feature from the different worms  
Cont2 = Control2_feat[Control2_feat.columns.difference(['worm_index'])].groupby ('timestamp').mean()#calculate the mean for each timestamp, for each feature from the different worms  
Str2_Cont2 = pd.merge(Str2, Cont2, on = 'timestamp' )
Cont1_Cont2 = pd.merge(Cont1, Cont2, on = 'timestamp' )
Cont2_Cont1 = pd.merge(Cont2, Cont1, on = 'timestamp' )

jsd_result1 = []
Str1_Cont1 = Str1_Cont1.drop(index = 0).reset_index(drop = True) #most values at t=0 are NaN because in most videos the worm is not properly analysed. In order to omit this line we remove it 
for column in Str1_Cont1.columns[0:150]: #as there are 150 columns of Str1D and 150 of Cont1D, this way we compare each column of Str1D to one of Cont1D in order 
    x = np.absolute(np.array(((Str1_Cont1[column]).interpolate(method ='linear')).fillna(0))) #becomes p, the Str1D part of the dataframe. Using dropna() we omit the NaN values for calculating the magnitude and distance of the vector 
    n = Str1_Cont1.columns.get_loc(column)
    y = np.absolute(np.array((Str1_Cont1.iloc[:, n+149]).interpolate (method ='linear').fillna(0))) #becomes q, the Cont1D part of the dataframe, and it is the column x +149 because the two dataframes of 150 are concatenated 
    jsd1 = distance.jensenshannon(x, y)
    info = [column] 
    jsd_result1.append (info + [jsd1])
jsstrain1 = pd.DataFrame(data = jsd_result1)

jsd_result2 = []
Str2_Cont2 = Str2_Cont2.drop(index = 0).reset_index(drop = True) #most values at t=0 are NaN because in most videos the worm is not properly analysed. In order to omit this line we remove it 
for column in Str2_Cont2.columns[0:150]: #as there are 150 columns of Str1D and 150 of Cont1D, this way we compare each column of Str1D to one of Cont1D in order 
    x = np.absolute(np.array((Str2_Cont2[column]).interpolate(method ='linear'))) #becomes p, the Str1D part of the dataframe. Using dropna() we omit the NaN values for calculating the magnitude and distance of the vector 
    n = Str2_Cont2.columns.get_loc(column)
    y = np.absolute(np.array((Str2_Cont2.iloc[:, n+149]).interpolate (method ='linear'))) #becomes q, the Cont1D part of the dataframe, and it is the column x +149 because the two dataframes of 150 are concatenated 
    jsd2 = distance.jensenshannon(x, y)
    info = [column] 
    jsd_result2.append (info + [jsd2])
jsstrain2 = pd.DataFrame(data = jsd_result2)

jsd_result3 = []
Cont1_Cont2 = Cont1_Cont2.drop(index = 0).reset_index(drop = True) #most values at t=0 are NaN because in most videos the worm is not properly analysed. In order to omit this line we remove it 
for column in Cont1_Cont2.columns[0:150]: #as there are 150 columns of Str1D and 150 of Cont1D, this way we compare each column of Str1D to one of Cont1D in order 
    x = np.absolute(np.array((Cont1_Cont2[column]).interpolate(method ='linear'))) #becomes p, the Str1D part of the dataframe. Using dropna() we omit the NaN values for calculating the magnitude and distance of the vector 
    n = Cont1_Cont2.columns.get_loc(column)
    y = np.absolute(np.array((Cont1_Cont2.iloc[:, n+149]).interpolate (method ='linear'))) #becomes q, the Cont1D part of the dataframe, and it is the column x +149 because the two dataframes of 150 are concatenated 
    jsd3 = distance.jensenshannon(x, y)
    info = [column] 
    jsd_result3.append (info + [jsd3])
jsstrain3 = pd.DataFrame(data = jsd_result3)

jsd_result4 = []
Cont2_Cont1 = Cont2_Cont1.drop(index = 0).reset_index(drop = True) #most values at t=0 are NaN because in most videos the worm is not properly analysed. In order to omit this line we remove it 
for column in Cont2_Cont1.columns[0:150]: #as there are 150 columns of Str1D and 150 of Cont1D, this way we compare each column of Str1D to one of Cont1D in order 
    x = np.absolute(np.array((Cont2_Cont1[column]).interpolate(method ='linear'))) #becomes p, the Str1D part of the dataframe. Using dropna() we omit the NaN values for calculating the magnitude and distance of the vector 
    n = Cont2_Cont1.columns.get_loc(column)
    y = np.absolute(np.array((Cont2_Cont1.iloc[:, n+149]).interpolate (method ='linear'))) #becomes q, the Cont1D part of the dataframe, and it is the column x +149 because the two dataframes of 150 are concatenated 
    jsd4 = distance.jensenshannon(x, y)
    info = [column] 
    jsd_result4.append (info + [jsd4])
jsstrain4 = pd.DataFrame(data = jsd_result4)

#Plot 
jsstrain1[0] = [s.rstrip("_x") for s in jsstrain1[0]] #remove the x result from concatenating 2 dataframes in feature names
ax = jsstrain1.dropna().plot.bar (x = 0, y =1, figsize = (30,20), legend = False)
ax.set_xlabel('Features')
ax.set_ylabel('JSD')
ax.set_title('JSD values per feature between ATR and control in' + nStrain1)
plt.savefig(os.path.join(newfolder, ('JSD values per feature between ATR and control in' + nStrain1))) #saves all the plots as png in the selected folder
plt.close()

jsstrain2[0] = [s.rstrip("_x") for s in jsstrain2[0]] #remove the x result from concatenating 2 dataframes in feature names
ax = jsstrain2.dropna().plot.bar (x = 0, y =1, figsize = (30,20), legend = False)
ax.set_xlabel('Features')
ax.set_ylabel('JSD')
ax.set_title('JSD values per feature between ATR and control in' + nStrain2)
plt.savefig(os.path.join(newfolder, ('JSD values per feature between ATR and control in' + nStrain2))) #saves all the plots as png in the selected folder
plt.close()

jsstrain3[0] = [s.rstrip("_x") for s in jsstrain3[0]] #remove the x result from concatenating 2 dataframes in feature names
ax = jsstrain3.dropna().plot.bar (x = 0, y =1, figsize = (30,20), legend = False)
ax.set_xlabel('Features')
ax.set_ylabel('JSD')
ax.set_title('JSD values per feature between controls')
plt.savefig(os.path.join(newfolder, ('JSD values per feature between controls'))) #saves all the plots as png in the selected folder
plt.close()

jsstrain4[0] = [s.rstrip("_x") for s in jsstrain4[0]] #remove the x result from concatenating 2 dataframes in feature names
ax = jsstrain4.dropna().plot.bar (x = 0, y =1, figsize = (30,20), legend = False)
ax.set_xlabel('Features')
ax.set_ylabel('JSD')
ax.set_title('JSD values per feature between controls2')
plt.savefig(os.path.join(newfolder, ('JSD values per feature between controls2'))) #saves all the plots as png in the selected folder
plt.close()

#Calculate a meaningful threshold for the JSD data, in order to do this we take 9000 values(approx. the size
#of the Str1_Cont1 dataframe) from each column and compare them to 9000 values that are 9000 below them in the 
#dataframe, this is not random, but still implies that we are comparing without taking into account the timestamp
#values for the same feature from the same strain that should be similar, not taking into account stimuli.

jsd_result = []
for n in range (9000, 90000, 9000): # the stimulus is given at 3000s and each video is 9000s
   for column in Strain1_feat.columns[2:150]: #as there are 150 columns of Str1D and 150 of Cont1D, this way we compare each column of Str1D to one of Cont1D in order 
    x = np.absolute(np.array(((Strain1_feat.loc[n-9000:n, column]).interpolate(method ='linear')).fillna(0))) #becomes p, the Str1D part of the dataframe. Using dropna() we omit the NaN values for calculating the magnitude and distance of the vector 
    y = np.absolute(np.array((Strain1_feat.loc[n:n+9000, column]).interpolate (method ='linear').fillna(0))) #becomes q, the Cont1D part of the dataframe, and it is the column x +149 because the two dataframes of 150 are concatenated 
    jsd = distance.jensenshannon(x, y)
    info = [column] 
    jsd_result.append (info + [jsd])
jsstrain = pd.DataFrame(data = jsd_result)
jsstrainm1 = jsstrain.groupby(0).mean().reset_index()

jsd_result = []
for n in range (9000, 90000, 9000): # the stimulus is given at 3000s and each video is 9000s
   for column in Strain2_feat.columns[2:150]: #as there are 150 columns of Str1D and 150 of Cont1D, this way we compare each column of Str1D to one of Cont1D in order 
    x = np.absolute(np.array(((Strain2_feat.loc[n-9000:n, column]).interpolate(method ='linear')).fillna(0))) #becomes p, the Str1D part of the dataframe. Using dropna() we omit the NaN values for calculating the magnitude and distance of the vector 
    y = np.absolute(np.array((Strain2_feat.loc[n:n+9000, column]).interpolate (method ='linear').fillna(0))) #becomes q, the Cont1D part of the dataframe, and it is the column x +149 because the two dataframes of 150 are concatenated 
    jsd = distance.jensenshannon(x, y)
    info = [column] 
    jsd_result.append (info + [jsd])
jsstrain = pd.DataFrame(data = jsd_result)
jsstrainm2 = jsstrain.groupby(0).mean().reset_index()
    
differences1 = jsstrain1[0].loc[(jsstrain1[1].sub(jsstrainm1[1])) >0] #find out the features that are more distant between experiment and control than inside experiment
differences2 = jsstrain2[0].loc[(jsstrain2[1].sub(jsstrainm2[1])) >0] #find out the features that are more distant between experiment and control than inside experiment

ax = jsstrainm1.dropna().plot.bar(x = 0, y =1, figsize = (50,20), legend = False)
ax.set_xlabel('Features')
ax.set_ylabel('JSD')
ax.set_title('JSD values per feature in same strain to set threshold in' + nStrain1)
plt.savefig(os.path.join(newfolder, ('JSD values per feature in same strain to set threshold in' + nStrain1))) #saves all the plots as png in the selected folder
plt.close()

ax = jsstrainm2.dropna().plot.bar(x = 0, y =1, figsize = (50,20), legend = False)
ax.set_xlabel('Features')
ax.set_ylabel('JSD')
ax.set_title('JSD values per feature in same strain to set threshold in' + nStrain2)
plt.savefig(os.path.join(newfolder, ('JSD values per feature in same strain to set threshold in' + nStrain1))) #saves all the plots as png in the selected folder
plt.close()

#%% SECOND PART OF THE CODE: visualise variables

#Create list of features you are interested in (here I added all the ones related with reversal and speed + the ones the JSD found to be divergent)
ColInt = ['speed', 'relative_to_body_speed_midbody', 'speed_neck', 'speed_head_base',
       'speed_hips', 'speed_tail_base', 'speed_midbody', 'speed_head_tip',
       'speed_tail_tip', 'd_relative_to_body_speed_midbody', 'd_speed_neck',
       'd_speed_head_base', 'd_speed_hips', 'd_speed_tail_base',
       'd_speed_midbody', 'd_speed_head_tip', 'd_speed_tail_tip',
       'curvature_head', 'curvature_hips', 'curvature_midbody',
       'curvature_neck', 'curvature_tail', 'curvature_mean_head',
       'curvature_mean_neck', 'curvature_mean_midbody', 'curvature_mean_hips',
       'curvature_mean_tail', 'curvature_std_head', 'curvature_std_neck',
       'curvature_std_midbody', 'curvature_std_hips', 'curvature_std_tail',
       'path_curvature_body', 'path_curvature_tail', 'path_curvature_midbody',
       'path_curvature_head', 'd_curvature_head', 'd_curvature_hips',
       'd_curvature_midbody', 'd_curvature_neck', 'd_curvature_tail',
       'd_curvature_mean_head', 'd_curvature_mean_neck',
       'd_curvature_mean_midbody', 'd_curvature_mean_hips',
       'd_curvature_mean_tail', 'd_curvature_std_head', 'd_curvature_std_neck',
       'd_curvature_std_midbody', 'd_curvature_std_hips',
       'd_curvature_std_tail', 'd_path_curvature_body',
       'd_path_curvature_tail', 'd_path_curvature_midbody',
       'd_path_curvature_head', 'motion_mode'] + list(set((differences1.values).tolist() + (differences2.values).tolist()))

columns = Strain1_feat.columns[2:] #List of all but the first two columns, that are timestamp and worm index, not important when looping over features
Datlist = {nStrain1:Strain1_feat, nControl1:Control1_feat, nStrain2:Strain2_feat, nControl2:Control2_feat} #creates a directory of the dataframes in order to be able to loop over them 

#Align videos based on stimuli point and take 5 min before and 10 min after stimuli
Datlist_aligned = {} #creates dictionary to store results
Datlist_speed = {}
for key, dataframe in Datlist.items(): 
 dataframe = dataframe.groupby('Video') #groups data in dataframes in original dictionary by video
 temp = [] #list to store results
 speed = []
 for video, group in dataframe: #loops over the groups created in the previous step, so over every video for a strain 
    for index, row in Stimuli.iterrows(): #loops over the rows of the stimuli list
      if row['Video'] == video:  #controls that the stimuli taken is the one that corresponds to the video group that is being looped
          subset = group.loc[(group['timestamp']>= row[key]-300) & (group['timestamp'] <= row[key]+600)] #selects the data in the dataframe 5min above the stimuli and 6.3 min after stimuli
          subset = subset.reset_index (drop = True) # reset index of subset dataframe, so that it goes 0 to 7000 
          subset = subset.loc[~subset.index.duplicated(keep='first')]
          subset['Homogeneous_time'] = subset.index #save new index as new column to use it afterwards to group all the videos by timestamp
          spd = subset['speed_head_tip'].rolling(window=10).mean()
          temp.append(subset) #stores all subsets
          speed.append(spd)

 Strain_aligned = pd.concat(temp, ignore_index = True) #concats all subsets stored into 1 dataframe
 Speed_aligned = pd.DataFrame(data = speed).T
 Speed_aligned['Mean'] = Speed_aligned.mean(axis = 1)
 Datlist_aligned [key] = Strain_aligned #stores the dataframes for each strain onto a dictionary 
 Datlist_speed [key] = Speed_aligned
 
# #Plot per strain the distribution in time of the core features for this analysis from all the videos for the strainn
rawfolder = datafolder / 'Results/Raw_data/'
if not os.path.exists(rawfolder): #this prevents creating the folder if it already exists
    os.makedirs(rawfolder) #creates folder to store the figures from the following loop
# for key, dataframe in Datlist_aligned.items(): 
#     dataframe['timestamp_min'] = dataframe['Homogeneous_time'].apply(lambda x: x/600) #this step transforms timestamp from s to min taking into account 10frames/s framerate
#     for column in ColInt:
#         feat_time = pd.DataFrame(data = (dataframe['timestamp_min'], dataframe[column])).T #creates a dataframe with the timestamp in min and the column to be plotted
#         plt.figure()
#         plt.axvline (x=5, linestyle = 'dashed', color ='grey', zorder = 2) #plots a line at 5min, when the pulse takes place
#         hmap1 = plt.hist2d(x = feat_time['timestamp_min'], y = feat_time[column], bins=100, range = [(0, 15), (-40, 40)], cmap = 'plasma') #creates a histogram of feat_time with 100 bins of resolution and dimensions 0-15 in x axes -40 to 40 in y axes. 
#         plt.xlabel('Time (s)', size = 14)
#         plt.ylabel(column, size = 14)
#         plt.title(column + ' ' + 'per time in' + ' ' + key , size = 14)
#         plt.savefig(os.path.join(rawfolder, ( column + ' ' + 'per time in' + key + '.png'))) #saves all the plots as png in the selected folder
#         plt.close()

#Plot speed per timestamp for each video
Spd_mean = []
for key, dataframe in Datlist_speed.items(): 
  plt.style.use('seaborn-white')
  fig1 = dataframe.plot (color = 'grey' , linewidth = 0.5, figsize = (20,10), legend=False, zorder = 5)
  fig1.axvline (x=300, linestyle = 'dashed', color ='black', zorder = 0) #plots the vertical line at time 0 indicating when the egg laying event takes place, zorder indicates that it is plotted at the back of the plot
  dataframe ['Mean'].plot(color = 'black', linewidth = 2, zorder = 10)
  fig1.set_title('Speed head tip per time' + key)
  fig1.set_ylabel('Speed head tip')
  fig1.set_xlabel('Timestamp (s/10')
  plt.savefig(os.path.join(rawfolder, ( 'Speed head tip per time in' + key + '.png'))) #saves all the plots as png in the selected folder
  plt.close()
  dataframe['Mean' + key] = dataframe['Mean']
  Spd_mean.append(dataframe['Mean' + key]) #create a list of all the mean plots for the 4 strains to plot 
#Plot mean speed per timestamp for the 4 strains together 
Speed_mean = pd.DataFrame(data = Spd_mean). T
fig2 = Speed_mean.plot(linewidth = 0.5, figsize = (20,10), zorder = 5)
fig2.axvline (x=300, linestyle = 'dashed', color ='black', zorder = 0)
fig2.set_title('Mean speed head tip per time')
fig2.set_ylabel('Speed head tip μm/sec')
fig2.set_xlabel('Timestamp s/10')
plt.savefig(os.path.join(rawfolder, ( 'Mean speed head tip per time in all strains.png'))) #saves all the plots as png in the selected folder
plt.close() 
#%% THIRD PART OF THE CODE : Evaluate how different are the selected features per timestamp on the target strain vs. the control
ructureNewInt = 'relative_to_body_speed_midbody'
#Evaluate significant list of features for each time point between target strain and control (t-test for each time point, for each feature)
Strain1_group = Datlist_aligned [nStrain1].groupby('Homogeneous_time')
Control1_group = Datlist_aligned[nControl1].groupby('Homogeneous_time')
t_test_results1 = []
for time_point1, group1 in Strain1_group:
    for time_point2, group2 in Control1_group:
        for column in ColInt:
                if time_point2 == time_point1: #only calculates t-test between points at the same timestamp for the same feature between control and target strain
                    control1 = group1[column]
                    strain1 = group2[column]
                    _,p = sts.ttest_ind(control1, strain1, nan_policy = 'omit') 
                    info1 = [time_point1, column] #keeps the timestamp = time_point1 and the feature = column information for the later result table
                    if p == 'NaN' : #the p.value is 'NaN' because both dataframes don't have the same length, some videos were longer than others and thus there was more data
                        break       #to make the loop stop when p-value is 'NaN' and start from the top
                    t_test_results1.append(info1 + [p]) #stores the p-values with the information about the timestamp and feature they correspond to 
Strain1_results = pd.DataFrame(data = t_test_results1)
Strain1_results = Strain1_results.rename(columns = {0:'timestamp', 1:'feature', 2:'p-value'}) #creates new dataframe as a list of the p-value per feature grouping the features by timestamp (the t-test averages videos for same strain, so 1 value per feature per timestamp)

Strain2_group = Datlist_aligned [nStrain2].groupby('Homogeneous_time')
Control2_group = Datlist_aligned[nControl2].groupby('Homogeneous_time')
t_test_results2 = []
for time_point1, group1 in Strain2_group:
    for time_point2, group2 in Control2_group:
        for column in NewInt:
                if time_point2 == time_point1:
                    control2 = group1[column]
                    strain2 = group2[column]
                    _,p = sts.ttest_ind(control2, strain2, nan_policy = 'omit') 
                    info2 = [time_point1, column]
                    if p == 'NaN' :
                        break
                    t_test_results2.append(info2 + [p])
Strain2_results = pd.DataFrame(data = t_test_results2)
Strain2_results = Strain2_results.rename(columns = {0:'timestamp', 1:'feature', 2:'p-value'})

#get most significant features performing a two stage FDR correction (I use the number of ttest performed, so the number of features compared. You could also use the number of plates analysed (as features are extracted from averages of worms tracks per plate), or per worm (not recommend because at some points worm tracks are lost))
Strain1_results_corrected = smm.multipletests(Strain1_results['p-value'], alpha=0.05, method='fdr_tsbky', is_sorted = False, returnsorted = False) 
Strain1_results.insert(3, 'p-value-corrected', Strain1_results_corrected[1]) #adds the corrected p-values as a new column to the results dataframe
Strain2_results_corrected = smm.multipletests(Strain2_results['p-value'], alpha=0.05, method='fdr_tsbky', is_sorted = False, returnsorted = False) 
Strain2_results.insert(3, 'p-value-corrected', Strain2_results_corrected[1]) #adds the corrected p-values as a new column to the results dataframe

#Pivot results dataframes to be able to plot them 
Strain1_results["p-value-corrected"] = np.ma.filled(Strain1_results["p-value-corrected"].astype(float), np.nan) #from the t-test all p-values that were NaN were transformed to masked arrays to omit them, this transforms them back to NaN for the next step
Strain1_results_plot = pd.pivot_table(Strain1_results, index= 'timestamp', columns = 'feature', values = 'p-value-corrected') #pivots the results dataframe, to create a dataframe in which every row is a timestamp, each column is a different feature and the values are the corrected p-values
Strain2_results["p-value-corrected"] = np.ma.filled(Strain2_results["p-value-corrected"].astype(float), np.nan) #from the t-test all p-values that were NaN were transformed to masked arrays to omit them, this transforms them back to NaN for the next step
Strain2_results_plot = pd.pivot_table(Strain2_results, index= 'timestamp', columns = 'feature', values = 'p-value-corrected') #pivots the results dataframe, to create a dataframe in which every row is a timestamp, each column is a different feature and the values are the corrected p-values

#Plots results dataframes in a clustermap to see the difference significance of the features at different time points
sns.set()
cmap1 = sns.clustermap(Strain1_results_plot, vmin = 0, vmax = 0.05, figsize = (50, 20), row_cluster = False, col_cluster = False, cbar_pos=(0.1, .2, .03, .4))
cmap1.fig.suptitle(x=0.5, y=0.9, t = ('T-test significant features per time in'+ nStrain1), fontsize = 30 ) #adds title to figure
# cmap1.ax_heatmap.yaxis.set_major_formatter(tick.FuncFormatter(lambda x, pos: '{:,.2f}'.format(x/600))) #this line reduces the number of decimals from the ticks to 2 and formats the ticks label so that they represent minutes and not seconds*frame rate
cmap1.ax_heatmap.axhline (y= 3000, linestyle = 'dashed', color ='grey', zorder = 2) #plots a line at 5min, when the pulse takes place
plt.setp(cmap1.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.setp(cmap1.ax_heatmap.xaxis.get_majorticklabels(), rotation=90, fontsize = 15)
sns.set_style('ticks')
plt.savefig(os.path.join(newfolder, ('New T-test significant features per time in' + nStrain1 + ' with and without ATR.png'))) #saves all the plots as png in the selected folder

cmap2 = sns.clustermap(Strain2_results_plot, vmin = 0, vmax = 0.05, figsize = (50, 20), row_cluster = False, col_cluster = False, cbar_pos=(0.1, .2, .03, .4))
cmap2.fig.suptitle(x=0.5, y=0.9, t = ('T-test significant features per time in'+ nStrain2), fontsize = 30 ) #adds title to figure
cmap2.ax_heatmap.yaxis.set_major_formatter(tick.FuncFormatter(lambda x, pos: '{:,.2f}'.format(x/600))) #this line reduces the number of decimals from the ticks to 2 and formats the ticks label so that they represent minutes and not seconds*frame rate
cmap2.ax_heatmap.axhline (y= 3000, linestyle = 'dashed', color ='grey', zorder = 2) #plots a line at 5min, when the pulse takes place
plt.setp(cmap2.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.setp(cmap2.ax_heatmap.xaxis.get_majorticklabels(), rotation=90, fontsize = 15)
sns.set_style('ticks')
plt.savefig(os.path.join(newfolder, ('New T-test significant features per time in' + nStrain2 + ' with and without ATR.png'))) #saves all the plots as png in the selected folder

#%% FOURTH PART OF THE CODE: Calculate Jensen Shannon per timestamp to be able to compare between strains with several stimuli 
#Not by timestamp, JSD divergence and permutation test over it.
 
#I could use interpolate, to figure out the missing NaN values instead of dropna. However with interpolate there will still be loads of 'NaN' at the beginning and end of the videos. 
Strain1_group = Strain1_feat.groupby('timestamp')
Control1_group = Control1_feat.groupby('timestamp')
t_test_jsd_results1 = []
for time_point1, group1 in Strain1_group:
    for time_point2, group2 in Control1_group:
        for column in columns: 
            if time_point2 == time_point1: #only calculates t-test between points at the same timestamp for the same feature between control and target strain
                    x = np.array((group1[column]).dropna())
                    y = np.array((group2[column]).dropna())
                    if x.size != y.size: 
                        n = x.size - y.size
                        if n>0 : mx = ma.masked_array(x, mask = [x - x[0 : 0+n]])
                        if n<0 : my = ma.masked_array(y, mask = y[0 : 0-n])
                        jsd = distance.jensenshannon(mx, my) 
                    jsd1 = distance.jensenshannon(x, y) 
                    info1 = [time_point1, column] #keeps the timestamp = time_point1 and the feature = column information for the later result table
                    t_test_jsd_results1.append (info + [jsd1]) #stores the distance values with the information about the timestamp and feature they correspond to 
Strain1_results_jsd = pd.DataFrame(data = t_test_jsd_results1)
Strain1_results_jsd = Strain1_results_jsd.rename(columns = {0:'timestamp', 1:'feature', 2:'JSD_distance'}) #creates new dataframe as a list of the p-value per feature grouping the features by timestamp (the t-test averages videos for same strain, so 1 value per feature per timestamp)

Strain2_group = Strain2_feat.groupby('timestamp')
Control2_group = Control2_feat.groupby('timestamp')
t_test_jsd_results2 = []
for time_point2, group2 in Strain2_group:
    for time_point2, group2 in Control2_group:
        for column in columns: 
            if time_point2 == time_point1: #only calculates t-test between points at the same timestamp for the same feature between control and target strain
                    x = np.array((group1[column]).dropna())
                    y = np.array((group2[column]).dropna())
                    if x.size != y.size: 
                        n = x.size - y.size
                        if n>0 : mx = ma.masked_array(x, mask = [x - x[0 : 0+n]])
                        if n<0 : my = ma.masked_array(y, mask = y[0 : 0-n])
                        jsd = distance.jensenshannon(mx, my) 
                    jsd2 = distance.jensenshannon(x, y) 
                    info2 = [time_point2, column] #keeps the timestamp = time_point1 and the feature = column information for the later result table
                    t_test_jsd_results1.append (info + [jsd2]) #stores the distance values with the information about the timestamp and feature they correspond to 
Strain2_results_jsd = pd.DataFrame(data = t_test_jsd_results2)
Strain2_results_jsd = Strain2_results_jsd.rename(columns = {0:'timestamp', 1:'feature', 2:'JSD_distance'}) #creates new dataframe as a list of the p-value per feature grouping the features by timestamp (the t-test averages videos for same strain, so 1 value per feature per timestamp)


Strain1_results_jsd_plot = pd.pivot_table(Strain1_results_jsd, index= 'timestamp', columns = 'feature', values = 'JSD_distance') #pivots the results dataframe, to create a dataframe in which every row is a timestamp, each column is a different feature and the values are the corrected p-values
Strain2_results_jsd_plot = pd.pivot_table(Strain2_results_jsd, index= 'timestamp', columns = 'feature', values = 'JSD_distance') #pivots the results dataframe, to create a dataframe in which every row is a timestamp, each column is a different feature and the values are the corrected p-values

sns.set()
cmap1 = sns.clustermap(Strain1_results_jsd_plot, vmin = 0, vmax = 0.05, figsize = (50, 20), row_cluster = False, col_cluster = False, cbar_pos=(0.1, .2, .03, .4))
cmap1.fig.suptitle(x=0.5, y=0.9, t = ('T-test significant features per time in'+ nStrain1), fontsize = 30 ) #adds title to figure
cmap1.ax_heatmap.yaxis.set_major_formatter(tick.FuncFormatter(lambda x, pos: '{:,.2f}'.format(x/600))) #this line reduces the number of decimals from the ticks to 2 and formats the ticks label so that they represent minutes and not seconds*frame rate
cmap1.ax_heatmap.axhline (y= 3000, linestyle = 'dashed', color ='grey', zorder = 2) #plots a line at 5min, when the pulse takes place
plt.setp(cmap1.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.setp(cmap1.ax_heatmap.xaxis.get_majorticklabels(), rotation=90, fontsize = 15)
sns.set_style('ticks')
plt.savefig(os.path.join(newfolder, ('T-test significant features per time in' + nStrain1 + ' with and without ATR.png'))) #saves all the plots as png in the selected folder

cmap2 = sns.clustermap(Strain2_results_jsd_plot, vmin = 0, vmax = 0.05, figsize = (50, 20), row_cluster = False, col_cluster = False, cbar_pos=(0.1, .2, .03, .4))
cmap2.fig.suptitle(x=0.5, y=0.9, t = ('T-test significant features per time in'+ nStrain2), fontsize = 30 ) #adds title to figure
cmap2.ax_heatmap.yaxis.set_major_formatter(tick.FuncFormatter(lambda x, pos: '{:,.2f}'.format(x/600))) #this line reduces the number of decimals from the ticks to 2 and formats the ticks label so that they represent minutes and not seconds*frame rate
cmap2.ax_heatmap.axhline (y= 3000, linestyle = 'dashed', color ='grey', zorder = 2) #plots a line at 5min, when the pulse takes place
plt.setp(cmap2.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.setp(cmap2.ax_heatmap.xaxis.get_majorticklabels(), rotation=90, fontsize = 15)
sns.set_style('ticks')
plt.savefig(os.path.join(newfolder, ('T-test significant features per time in' + nStrain2 + ' with and without ATR.png'))) #saves all the plots as png in the selected folder


Strain1_group = Strain1_feat.groupby('timestamp')
Control1_group = Control1_feat.groupby('timestamp')
t_test_results1 = []
for time_point1, group1 in Strain1_group:
    for time_point2, group2 in Control1_group:
        for column in NewInt:
                if time_point2 == time_point1: #only calculates t-test between points at the same timestamp for the same feature between control and target strain
                    control1 = group1[column]
                    strain1 = group2[column]
                    _,p = sts.ttest_ind(control1, strain1, nan_policy = 'omit') 
                    info1 = [time_point1, column] #keeps the timestamp = time_point1 and the feature = column information for the later result table
                    if p == 'NaN' : #the p.value is 'NaN' because both dataframes don't have the same length, some videos were longer than others and thus there was more data
                        break       #to make the loop stop when p-value is 'NaN' and start from the top
                    t_test_results1.append(info1 + [p]) #stores the p-values with the information about the timestamp and feature they correspond to 
Strain1_results = pd.DataFrame(data = t_test_results1)
Strain1_results = Strain1_results.rename(columns = {0:'timestamp', 1:'feature', 2:'p-value'}) #creates new dataframe as a list of the p-value per feature grouping the features by timestamp (the t-test averages videos for same strain, so 1 value per feature per timestamp)

Strain2_group = Strain2_feat.groupby('timestamp')
Control2_group = Control2_feat.groupby('timestamp')
t_test_results2 = []
for time_point1, group1 in Strain2_group:
    for time_point2, group2 in Control2_group:
        for column in NewInt:
                if time_point2 == time_point1:
                    control2 = group1[column]
                    strain2 = group2[column]
                    _,p = sts.ttest_ind(control2, strain2, nan_policy = 'omit') 
                    info2 = [time_point1, column]
                    if p == 'NaN' :
                        break
                    t_test_results2.append(info2 + [p])
Strain2_results = pd.DataFrame(data = t_test_results2)
Strain2_results = Strain2_results.rename(columns = {0:'timestamp', 1:'feature', 2:'p-value'})
