# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 18:33:15 2019

@author: Lidia Ripoll Sanchez

Code to analyse speed egg laying correlation in videos of egg laying events.
The code was made following the analysis explained in Hardaker et al. 2001
This codegenerates plots for the mean speed and all individual event speeds of
the strain vs the 2min before and after the event, plot for the comparison 
between strains (normalized and not), and plots for the mean speed of the strain 
and a standard speed control, and plots for the mean speed of the strain for all
events, the mean for first events in cluster, intermediate events in cluster and 
last events in cluster. 
The code was created to analyse three different strains FX5190(FX), N2 and NG8048(NG). 
In order to run the code: 
    -change file paths to the paths of your files, 
    -adapt exp_speed for each strain and exp_events dataframes to the structure of your file, 
    -change the dimensions of all lists (num, num2, l1,l2,l3,m1,m2,m3) depending on 
    the number of columns (num, num2) or events (l1,l2,l3,m1,m2,m3) of your file 
    -when plotting change the name of each strain for the corresponding name of your 
    strain (Ei. if you loaded your file for strain AQ2000 in file_speedFX change FX5190
    in the plots for AQ2000)

"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import pandas as pd
import numpy as np
from scipy import stats

#Import files with experiment data from path location
file_speedFX = r'C:\Users\Lidia\Desktop\Collab_Denise\Final_speeds_all_FX.csv' # path to file + file name
file_speedN2 = r'C:\Users\Lidia\Desktop\Collab_Denise\Final_speeds_all_N2.csv' # path to file + file name
file_speedNG = r'C:\Users\Lidia\Desktop\Collab_Denise\Final_speeds_all_NG.csv' # path to file + file name
file_events = r'C:\Users\Lidia\Desktop\Collab_Denise\Final_events_useful.xlsx' # path to file + file name

#Create Dataframe objects for each file in the base of (read_excel or read_csv): exp_df_strain = pd.read_excel(io=file_name, header=number) 
exp_speed_FX = pd.read_csv(file_speedFX, header=1)
exp_speed_N2 = pd.read_csv(file_speedN2, header=1)
exp_speed_NG = pd.read_csv(file_speedNG, header=1)
exp_events = pd.read_excel(file_events, header=1, usecols= "A:G")

#%%Calculate absolute values for each strain speed values dataframe 
exp_speed_FX = exp_speed_FX.abs()
exp_speed_N2 = exp_speed_N2.abs()
exp_speed_NG = exp_speed_NG.abs()
#%%Calculate average speed value for a 10s window at 1s intervals using a rolling window
Speed10FX = exp_speed_FX.rolling(window=10).mean()
Speed10N2 = exp_speed_N2.rolling(window=10).mean()
Speed10NG = exp_speed_NG.rolling(window=10).mean()
#Return Frame columns to their initial values before rolling average 
num = ['1','2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
for i in num:
    n = i
    Speed10FX['Frame_FX_' + n ] = exp_speed_FX['Frame_FX_'+ n ]
    Speed10NG['Frame_NG_' + n ] = exp_speed_NG['Frame_NG_' + n]
    if (i == 1 or i == 5 or i == 15): #exp_speed_N2 has less columns so we need to use this to prevent running the loop over missing columns
        pass
        Speed10N2['Frame_N2_' + n ] = exp_speed_N2['Frame_N2_' + n]
#%%Create column control with 1200 Speed values from random points 
#create the empty variables to store loop values
contFX = []
contN2 = []
contNG = []
numN2 = ['2', '3', '4', '6', '7', '8', '9', '10', '11', '12', '13', '14', '16']
#for loop to pick 1200 random values from every speed column in dataframe
for i in num: 
    n = i
    contFX.append(Speed10FX['Speed_FX_' + n].sample(n=2400).values)
    contNG.append(Speed10NG['Speed_NG_' + n].sample(n=2400).values)
#exp_speed_N2 has less columns so we need to create a new loop to prevent running the loop over missing columns
for i in numN2:
    n = i
    contN2.append(Speed10N2['Speed_N2_' + n].sample(n=2400).values) 
#create control dataframe calculating the mean of all values taken getting 1200 at the end 
controlFX = pd.DataFrame(contFX).mean().dropna() #Use dropna to remove NaN null values that the loop could have randomly taken
controlN2 = pd.DataFrame(contN2).mean().dropna()
controlNG = pd.DataFrame(contNG).mean().dropna()

#%% Create dataframe with speed per egg laying event, 2min before and after event
#Select events for every strain 
FX = exp_events.loc[exp_events['Strain'] == 'FX']
N2 = exp_events.loc[exp_events['Strain'] == 'N2']
NG = exp_events.loc[exp_events['Strain'] == 'NG']
#create the empty variables to store loop values
tempFX = []
tempN2 = []
tempNG = []
#for loop over events dataframe to obtain the start and end frame for each video
for index, row in FX.iterrows():
    videoname = str (row['Video'])
    begin = row ['Start_frame']
    end = row ['End_frame']
    tempFX.append(Speed10FX[Speed10FX['Frame_' + videoname].between(begin, end)]['Speed_'+ videoname].values)
for index, row in N2.iterrows():
    videoname = str (row['Video'])
    begin = row ['Start_frame']
    end = row ['End_frame']
    tempN2.append(Speed10N2[Speed10N2['Frame_' + videoname].between(begin, end)]['Speed_'+ videoname].values)
for index, row in NG.iterrows():
    videoname = str (row['Video'])
    begin = row ['Start_frame']
    end = row ['End_frame']
    tempNG.append(Speed10NG[Speed10NG['Frame_' + videoname].between(begin, end)]['Speed_'+ videoname].values)

#store speed information in dataframe add control and mean(not count Control when calculating it) columns 
speedFX = pd.DataFrame(data = tempFX).transpose()
speednaFX = speedFX.fillna(method = 'bfill', axis = 1) #fill all nan values with the one in the same row and following column to prevent diluting the mean
speednaFX['Control'] = controlFX
speednaFX['Mean'] = speednaFX.loc[:,speednaFX.columns != 'Control'].mean(axis=1)

speedN2 = pd.DataFrame(data = tempN2).transpose()
speedN2.columns = speedN2.columns.astype(str)
speedN2 = speedN2.drop(columns= ['93', '94', '95', '96', '97']) #remove all nan value columns from dataset, it goes from 128 to 123
speednaN2 = speedN2.fillna(method = 'bfill', axis = 1) #fill all nan values with the one in the same position and following row to prevent diluting the mean
speednaN2['Control'] = controlN2
speednaN2['Mean'] = speednaN2.loc[:,speednaN2.columns != 'Control'].mean(axis=1)

speedNG = pd.DataFrame(data = tempNG).transpose()
speednaNG = speedNG.fillna(method = 'bfill', axis = 1) #fill all nan values with the one in the same position and following row to prevent diluting the mean
speednaNG['Control'] = controlNG
speednaNG['Mean'] = speednaNG.loc[:,speednaNG.columns != 'Control'].mean(axis=1)

#%% Create index time for plotting and average speed per 10s
#Add time column indicating the time in 0.1s
d = np.arange (-120, 120.1, dtype=int)
data = np.repeat (d, 10)
time = pd.Series (data) 
speednaFX.insert(0, "Time", time, True)
speednaN2.insert(0, "Time", time, True)
speednaNG.insert(0, "Time", time, True)
# (NOT NECESSARY) Calculate average speed value for every second (fps =10)
speedsFX = speednaFX.groupby('Time').mean()
speedsN2 = speednaN2.groupby('Time').mean()
speedsNG = speednaNG.groupby('Time').mean()
#%%Calculate normalised speed (dividing speed per second, by average speed in last 30s of 4min around event)
#create lists for the number of columns in each dataframe
l1 = list(range(0, 131))
l2 = list(range(0, 126))
l3 = list(range(0, 149))
#create empty variables to store values
nomFX = []
nomN2 = []
nomNG = []
#loop around the lists calculating the mean for the last 30 rows of each column and diviing each value of the column by it
for i in l1:
    n = i 
    m = str (n)
    meanFX = speedsFX.iloc[210:241, n].mean()
    if (np.isnan(meanFX) == True):
        meanFX = 0
    else:
        meanFX = meanFX
    nomFX.append(speedsFX.iloc[:, n].sub(meanFX))         
for i in l2:
    n= i
    m = str(n)
    meanN2 = speedsN2.iloc[210:241, n].mean()
    if (np.isnan(meanN2) == True):
        meanN2 = 0
    else:
        meanN2 = meanN2
    nomN2.append(speedsN2.iloc[:, n].sub(meanN2))
for i in l3:
    n= i
    m = str(n)
    meanNG = speedsNG.iloc[210:241, n].mean()
    if (np.isnan(meanNG) == True):
        meanNG = 0
    else:
        meanNG = meanNG
    nomNG.append(speedsNG.iloc[:, n].sub(meanNG))

#Create dataframe for normalized speed for each strain 
speednFX = pd.DataFrame(data = nomFX).transpose()
speednN2 = pd.DataFrame(data = nomN2).transpose()
speednNG = pd.DataFrame(data = nomNG).transpose()

#Calculate normalized mean speed for each strain averaging normalized speed for all columns but the control
speednFX['Mean_normalized'] = speednFX.loc[:,speednFX.columns != 'Control'].mean(axis=1)
speednNG['Mean_normalized'] = speednNG.loc[:,speednNG.columns != 'Control'].mean(axis=1)
speednN2['Mean_normalized'] = speednN2.loc[:,speednN2.columns != 'Control'].mean(axis=1)

#%%Identify clusters and create dataframe with mean for first, last, middle events and all events
#create dataframes average for each second but with all nan values (if there is any difference between event orders replacing nan values by the following value will remove differences,keeping nan will dilute average of events)
speedFX.insert(0, "Time", time, True)
speedN2.insert(0, "Time", time, True)
speedNG.insert(0, "Time", time, True)
# (NOT ESSENTIAL) Calculate average speed value for every second (fps =10)
speedcluFX = speedFX.groupby('Time').mean()
speedcluN2 = speedN2.groupby('Time').mean()
speedcluNG = speedNG.groupby('Time').mean()
#Create empty variables for first, middle and last events in cluster per strain
clu1FX = []
clumFX = []
clulFX = []
clu1N2 = []
clumN2 = []
clulN2 = []
clu1NG = []
clumNG = []
clulNG = []
#Adapt lists created before for the number of events on every strain, removing the ones we deleted from N2
m1 = list(range(0, 129))
m2 = list(range(0, 125))
m3 = list(range(0, 147))
N2.drop([223, 224, 225, 226, 227])
#For loops to create lists for every strain with the first, last and middle events of every cluster
for i in m1:
    n = i
    bf = FX.iloc[n-1, 3]
    af = FX.iloc[n+1, 3]
    frame = FX.iloc[n, 3]
    if (abs(frame-bf) > 1200) and (abs(frame-af) <= 1200):
        clu1FX.append(speedcluFX.iloc[:,n])
    elif (abs(frame-bf) <= 1200) and (abs(frame-af) <= 1200): 
        clumFX.append(speedcluFX.iloc[:,n])
    elif (abs(frame-bf) <= 1200) and (abs(frame-af) > 1200):
        clulFX.append(speedcluFX.iloc[:,n])
for i in m2:
    n = i
    bf = N2.iloc[n-1, 3]
    af = N2.iloc[n+1, 3]
    frame = N2.iloc[n, 3]
    if (abs(frame-bf) > 1200) and (abs(frame-af) <= 1200):
        clu1N2.append(speedcluN2.iloc[:,n])
    elif (abs(frame-bf) <= 1200) and (abs(frame-af) <= 1200): 
        clumN2.append(speedcluN2.iloc[:,n])
    elif (abs(frame-bf) <= 1200) and (abs(frame-af) > 1200):
        clulN2.append(speedcluN2.iloc[:,n])
for i in m3:
    n = i
    bf = NG.iloc[n-1, 3]
    af = NG.iloc[n+1, 3]
    frame = NG.iloc[n, 3]
    if (abs(frame-bf) > 1200) and (abs(frame-af) <= 1200):
        clu1NG.append(speedcluNG.iloc[:,n])
    elif (abs(frame-bf) <= 1200) and (abs(frame-af) <= 1200): 
        clumNG.append(speedcluNG.iloc[:,n])
    elif (abs(frame-bf) <= 1200) and (abs(frame-af) > 1200):
        clulNG.append(speedcluNG.iloc[:,n])

#Create dataframes with list get mean speed for first, last and middle events
cluster1FX = pd.DataFrame(data = clu1FX).transpose().mean(axis=1)
clustermFX = pd.DataFrame(data = clumFX).transpose().mean(axis=1)
clusterlFX = pd.DataFrame(data = clulFX).transpose().mean(axis=1) 
clusterFX = pd.concat([cluster1FX, clustermFX, clusterlFX, speedsFX['Mean']], axis=1)
clusterFX.columns =['First_event', 'Intermediate_events', 'Last_event', 'All_events']

cluster1N2 = pd.DataFrame(data = clu1N2).transpose().mean(axis=1)
clustermN2 = pd.DataFrame(data = clumN2).transpose().mean(axis=1)
clusterlN2 = pd.DataFrame(data = clulN2).transpose().mean(axis=1) 
clusterN2 = pd.concat([cluster1N2, clustermN2, clusterlN2, speedsN2['Mean']], axis=1)
clusterN2.columns =['First_event', 'Intermediate_events', 'Last_event', 'All_events']

cluster1NG = pd.DataFrame(data = clu1NG).transpose().mean(axis=1)
clustermNG = pd.DataFrame(data = clumNG).transpose().mean(axis=1)
clusterlNG = pd.DataFrame(data = clulNG).transpose().mean(axis=1) 
clusterNG = pd.concat([cluster1NG, clustermNG, clusterlNG, speedsNG['Mean']], axis=1)
clusterNG.columns =['First_event', 'Intermediate_events', 'Last_event', 'All_events']

#%%Plots with all events for every strain
#Plot all events per strain and mean speed per strain
plt.style.use('seaborn-white')
fig1 = speedsFX.plot (color = 'grey' , linewidth = 0.5, figsize = (20,10), legend=False, zorder = 5)
fig1.axvline (x=0, linestyle = 'dashed', color ='black', zorder = 0) #plots the vertical line at time 0 indicating when the egg laying event takes place, zorder indicates that it is plotted at the back of the plot
speedsFX ['Mean'].plot(color = 'black', linewidth = 2, zorder = 10)
fig1.set_title('Velocity pattern around invidiaul egg laying events in FX5190')
fig1.set_ylabel('Velocity μm/sec')
fig1.set_xlabel('Time sec')
plt.show()

fig2 = speedsN2.plot (color = 'grey' , linewidth = 0.5, figsize = (20,10), legend=False, zorder = 5)
fig2.axvline (x=0, linestyle = 'dashed', color ='black', zorder = 0)
speedsN2 ['Mean'].plot(color = 'black', linewidth = 2, zorder = 10)
fig2.set_title('Velocity pattern around invidiaul egg laying events in N2')
fig2.set_ylabel('Velocity μm/sec')
fig2.set_xlabel('Time sec')
plt.show()

fig3 = speedsNG.plot (color = 'grey' , linewidth = 0.5, figsize = (20,10), legend=False, zorder = 5)
fig3.axvline (x=0, linestyle = 'dashed', color ='black', zorder = 0)
speedsNG ['Mean'].plot(color = 'black', linewidth = 2, zorder = 10)
fig3.set_title('Velocity pattern around individual egg laying events in NG8048')
fig3.set_ylabel('Velocity μm/sec')
fig3.set_xlabel('Time sec')
plt.show()

#Plot all events per strain and mean speed per strain but with normalized speeds
fig4 = speednFX.plot (color = 'grey' , linewidth = 0.5, figsize = (20,10), legend=False, zorder = 5)
fig4.axvline (x=0, linestyle = 'dashed', color ='black', zorder = 0)
speednFX ['Mean_normalized'].plot(color = 'black', linewidth = 2, zorder = 10)
fig4.set_title('Normalized velocity pattern around individual egg laying events in FX5190')
fig4.set_ylabel('Velocity μm/sec')
fig4.set_xlabel('Time sec')
plt.show()

fig5 = speednN2.plot (color = 'grey' , linewidth = 0.5, figsize = (20,10), legend=False, zorder = 5)
fig5.axvline (x=0, linestyle = 'dashed', color ='black', zorder = 0)
speednN2 ['Mean_normalized'].plot(color = 'black', linewidth = 2, zorder = 10)
fig5.set_title('Normalized velocity pattern around individual egg laying events in N2')
fig5.set_ylabel('Velocity μm/sec')
fig5.set_xlabel('Time sec')
plt.show()

fig6 = speednNG.plot (color = 'grey' , linewidth = 0.5, figsize = (20,10), legend=False, zorder = 5)
fig6.axvline (x=0, linestyle = 'dashed', color ='black', zorder = 0)
speednNG ['Mean_normalized'].plot(color = 'black', linewidth = 2, zorder = 10)
fig6.set_title('Normalized velocity pattern around individual egg laying events in NG8048')
fig6.set_ylabel('Velocity μm/sec')
fig6.set_xlabel('Time sec')
plt.show()
#%%Plot for different strains 
#Calculation of standard deviation
errorFX = speedsFX.iloc[:,0:129].std(axis =1)
errorN2 = speedsN2.iloc[:,0:123].std(axis =1)
errorNG = speedsNG.iloc[:,0:147].std(axis =1)

#Plot mean speed with standard deviation, control  for the three strains
plt.style.use('seaborn-white')

fig7 = speedsFX['Mean'].plot (yerr=errorFX, figsize = (15,6))
fig7.axvline (x=0, linestyle = 'dashed', color ='black', zorder = 0)
speedsFX['Control'].plot (color='black', linestyle ='dashed')
fig7.set_title('Velocity pattern around egg laying events in FX5190')
fig7.set_ylabel('Velocity μm/sec')
fig7.set_xlabel('Time sec')
plt.show()

fig8 = speedsN2['Mean'].plot (yerr=errorN2, color='green', figsize = (15,6))
fig8.axvline (x=0, linestyle = 'dashed', color ='black', zorder = 0)
speedsN2['Control'].plot (color='black', linestyle ='dashed')
fig8.set_title('Velocity pattern around egg laying events in N2')
fig8.set_ylabel('Velocity μm/sec')
fig8.set_xlabel('Time sec')
plt.show()

fig9 = speedsNG['Mean'].plot (yerr=errorNG, color='pink', figsize = (15,6))
fig9.axvline (x=0, linestyle = 'dashed', color ='black', zorder = 0)
speedsNG['Control'].plot (color='black', linestyle ='dashed')
fig9.set_title('Velocity pattern around egg laying events in NG8048')
fig9.set_ylabel('Velocity μm/sec')
fig9.set_xlabel('Time sec')
plt.show()
#%%Plot all strains together
plt.style.use('seaborn-white')
#not normalized speed
fig8 = plt.figure(constrained_layout =True)
gsp = gs.GridSpec(2,1, figure=fig8)
ax = fig8.add_subplot(gsp[0,:])
for frame in[speedsFX, speedsN2, speedsNG]: ax.plot(frame['Mean'])
plt.axvline (x=0, linestyle = 'dashed', color ='black', zorder = 0)
plt.title('Mean velocity for N2 , FX5190 and NG8048 around egg laying event')
ax.set_ylabel('Velocity μm/sec')
ax.set_xlabel('Time sec')

#normalized speed
ax2 = fig8.add_subplot(gsp[1,:])
for frame in[speednFX, speednN2, speednNG]: ax2.plot(frame['Mean_normalized'])
plt.axvline (x=0, linestyle = 'dashed', color ='black', zorder = 0)
plt.title('Normalized mean velocity for N2 , FX5190 and NG8048 around egg laying event')
ax2.set_ylabel('Velocity μm/sec')
ax2.set_xlabel('Time sec')
fig8.legend(labels = ('FX5190', 'N2', 'NG8048'), bbox_to_anchor = (1.1,0.6), loc = 'center right')
plt.show()

#%%Plot clusters
fig9 = clusterFX.plot (figsize = (15,6))
fig9.axvline (x=0, linestyle = 'dashed', color ='black', zorder = 0)
fig9.set_title('Velocity pattern around egg laying events in FX5190')
fig9.set_ylabel('Velocity μm/sec')
fig9.set_xlabel('Time sec')
plt.show()

fig10 = clusterN2.plot (figsize = (15,6))
fig10.axvline (x=0, linestyle = 'dashed', color ='black', zorder = 0)
fig10.set_title('Velocity pattern around egg laying events in N2')
fig10.set_ylabel('Velocity μm/sec')
fig10.set_xlabel('Time sec')
plt.show()

fig11 = clusterNG.plot (figsize = (15,6))
fig11.axvline (x=0, linestyle = 'dashed', color ='black', zorder = 0)
fig11.set_title('Velocity pattern around egg laying events in NG8048')
fig11.set_ylabel('Velocity μm/sec')
fig11.set_xlabel('Time sec')
plt.show()

#Interesting data
#According to this plot intermediate events have opposite speeds at the same time than first and last events in FX5190
fig12 = clusterFX['All_events'].plot (figsize = (15,6), legend = True)
clusterFX['Intermediate_events'].plot (color = 'orange')
plt.show()
#We calculate a t test to see if this difference is significant (as egg laying events in clusters are layed at the same interval we consider that the variance is the same one for both populations and use ttest)
pvalueFX = stats.ttest_ind (clusterFX['First_event'], clusterFX['Intermediate_events'])
pvalueNG = stats.ttest_ind (clusterNG['First_event'], clusterNG['Intermediate_events'])
