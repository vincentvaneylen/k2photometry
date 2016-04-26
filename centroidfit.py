'''
% Routines to go from MAST pixel files to light curves. Use run_pipeline.run() for regular use of this, or run gotoflux()
% Author Vincent Van Eylen
% Contact vincent@phys.au.dk
% See Van Eylen et al. 2015 (ApJ) for details. Please reference this work if you found this code helpful!
'''

# general python files
import os
import matplotlib.pyplot as pl
import numpy as np
from lmfit import minimize, Parameters

# pipeline files
from auxiliaries import *


def sliceIterator(lst, sliceLen):
    for i in range(len(lst) - sliceLen + 1):
        yield lst[i:i + sliceLen]

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def median_filter(time,data,binsize=100):
  # do a running median filter dividing all data points by the median of their immediate surroundings
  i = 0
  data_filtered = []
  while i < len(time):
	  bin_begin = max(0, (i - binsize/2))
	  bin_end = min(len(time),(i+binsize/2))
	  the_bin = data[bin_begin:bin_end]
	  the_bin = sorted(the_bin)
	  median = np.median(the_bin) #[len(the_bin)/2]
	  data_filtered.append(data[i]/median)
	  i = i + 1
  return data_filtered

def spitzer_residual(params,time,data,Xc,Yc,robust=True):
  #
  # residual function used for calculating a fit to centroid (and time), borrowed from reducing data for Spitzer
  #

  # unpack all parameters (note: some may be fixed rather than variable)
  X1 = params['X1'].value
  X2 = params['X2'].value
  X3 = params['X3'].value
  Y1 = params['Y1'].value
  Y2 = params['Y2'].value
  Y3 = params['Y3'].value
  XY1 = params['XY1'].value
  XY2 = params['XY2'].value
  T0 = params['T0'].value
  T1 = params['T1'].value
  T2 = params['T2'].value
  T3 = params['T3'].value
  T4 = params['T4'].value
  TsinAmp = params['TsinAmp'].value
  TsinOff = params['TsinOff'].value

  mean_Xc = np.array(0.) #np.mean(Xc)
  mean_Yc = np.array(0.) #np.mean(Yc)
  time0 = time[0] - 1.#np.array(1994.0) #time[0]-1.

  model = (T0 + TsinAmp*np.sin((time-time0)+TsinOff) + T1*(time-time0) + T2*((time-time0)**2.) + T3*((time-time0)**3.) + T4*((time-time0)**4.) + X1*(Xc-mean_Xc) + X2*((Xc-mean_Xc)**2) + X3*((Xc-mean_Xc)**3) + Y1*(Yc-mean_Yc) + Y2*((Yc-mean_Yc)**2) + Y3*((Yc-mean_Yc)**3) + XY1*(Xc-mean_Xc)*(Yc-mean_Yc) + XY2*((Xc-mean_Xc)**2)*((Yc-mean_Yc)**2))

  residual = np.array(data-model)

  if robust:
    # calculate residual in a robust way
    residual = residual[np.abs(residual) < np.mean(residual) + 3.*np.std(residual)]

  return residual


def find_thruster_events(time,data,Xc,Yc,outputpath='',starname=''):
  #
  # Find events when the spacecruft thruster are fired. Usually no useful data points are gathered when this happens
  #

  diff_centroid = np.diff(Xc)**2 + np.diff(Yc)**2

  thruster_mask = diff_centroid < (1.5*np.mean(diff_centroid) + 0.*np.std(diff_centroid))
  thruster_mask1 = np.insert(thruster_mask,0, False) # this little trick helps us remove 2 data points each time instead of just 1
  thruster_mask2 = np.append(thruster_mask,False)
  thruster_mask = thruster_mask1*thruster_mask2

  time_thruster = time[ thruster_mask ]
  diff_centroid_thruster = diff_centroid[ thruster_mask[1:] ]

  Xc_clipped = Xc[:][thruster_mask]
  Yc_clipped = Yc[:][thruster_mask]
  time_clipped = time[:][thruster_mask]
  data_clipped = data[:][thruster_mask]


  #pl.figure('Data with / without thruster events')
  #pl.plot(time,data)
  #pl.plot(time_clipped,data_clipped)

  #pl.figure('Differential of centroid movement')
  #pl.plot(time[1:],diff_centroid)
  #pl.plot(time_thruster,diff_centroid_thruster,'*')

  pl.figure()
  pl.plot(time_clipped,data_clipped)
  pl.savefig(os.path.join(outputpath,'raw_nothrusters_' + str(starname) + '.png'))
  np.savetxt(os.path.join(outputpath, 'lightcurve_raw_nothrusters_' + str(starname) + '.txt'),np.transpose([time_clipped,np.array(data_clipped)/np.mean(data_clipped)]),header='Time, Flux')

  return [time_clipped,data_clipped,Xc_clipped,Yc_clipped]


def clean_data(time,data):
  #
  # Module for basic data cleaning up
  #

  time = time[0:]
  data = data[0:]

  pl.figure('Cleaning up')
  #pl.plot(time,data,'.')
  [data,time] = sigma_clip(data,3,dependent_var=time,top_only=True) # do sigma-clipping (but only at the top of light curve, in bottom outliers may be transit events
  [data,time] = sigma_clip(data,3,dependent_var=time,top_only=True)

  [data,time,lowerbound,upperbound] = running_sigma_clip(data,8,binsize=10,dependent_var=time)
  pl.plot(time,data,'.',color='grey')
  pl.xlabel('Time [d]')
  pl.ylabel('Relative flux')
  return time,data



def spitzer_fit(time,data,Xc,Yc,starname='',outputpath='',chunksize=300):
  #
  # Fit a polynomial to the data and return corrected data
  #

  outputfolder = os.path.join(outputpath,str(starname))

  # Remove NaN etc.
  time = np.array(time)[np.array(time) > 0.]
  data = np.array(data)[np.array(time) > 0.]
  Xc = np.array(Xc)[np.array(time) > 0.]
  Yc = np.array(Yc)[np.array(time) > 0.]

  data = np.array(data) / np.mean(data)

  params = Parameters() # fitting parameters, set to vary=false to fix
  params.add('X1', value = 0.,vary=True)
  params.add('X2', value = 0.,vary=True)
  params.add('X3', value = 0.,vary=True)
  params.add('Y1', value = 0.,vary=True)
  params.add('Y2', value = 0.,vary=True)
  params.add('Y3', value = 0.,vary=True)
  params.add('XY1', value = 0.,vary=True)
  params.add('XY2', value = 0.,vary=False)
  params.add('T0', value = 0.,vary=True)
  params.add('T1', value = 0.,vary=True)
  params.add('T2', value = 0.,vary=True) #
  params.add('T3', value = 0.,vary=True) #
  params.add('T4', value = 0.,vary=False)
  params.add('TsinAmp', value = 0.,vary=False)
  params.add('TsinOff', value = 0.,vary=False)

  # first divide data in different chunks
  chunksize=chunksize
  time_chunks = list(chunks(time,chunksize))
  data_chunks = list(chunks(data,chunksize))
  Xc_chunks = list(chunks(Xc,chunksize))
  Yc_chunks = list(chunks(Yc,chunksize))

  i = 0
  corrected_data = []
  pl.figure('Data correction Spitzer ' + str(starname))
  while i < len(time_chunks):
    fit = minimize(spitzer_residual, params, args=(time_chunks[i],data_chunks[i],Xc_chunks[i],Yc_chunks[i],False))#,method='leastsq') # first fit is not robust, to get a good first estimate
    fit = minimize(spitzer_residual, fit.params, args=(time_chunks[i],data_chunks[i],Xc_chunks[i],Yc_chunks[i],True))

    final_model = data_chunks[i] - spitzer_residual(fit.params,time_chunks[i],data_chunks[i],Xc_chunks[i],Yc_chunks[i],robust=False)
    corrected_data.append(data_chunks[i] - final_model) # + np.mean(data_chunks[i])

    pl.figure('Data correction Spitzer ' + str(starname))
    pl.plot(time_chunks[i],data_chunks[i],'*',label='Raw data')
    pl.plot(time_chunks[i],final_model,'*',label='Modeled data')

    pl.figure('Corrected data Spitzer ' + str(starname))
    pl.plot(time_chunks[i],corrected_data[i],'*')#,label='Corrected data')

    i = i + 1
  pl.legend()
  pl.savefig(os.path.join(outputfolder, 'centroiddetrended_lightcurve_' + str(starname) + '.png'))


  import itertools # to go from list of lists to one list again
  corrected_time = list(itertools.chain(*time_chunks))
  corrected_data = list(itertools.chain(*corrected_data))

  # finally do a broad running median filtering to remove remaining trends. can be turned off if one wants to keep long term trends
  corrected_data = np.array(median_filter(corrected_time,np.array(corrected_data)+1.,49))-1. #
  corrected_data = np.array(median_filter(corrected_time,np.array(corrected_data)+1.,49))-1. #

  return [corrected_time,corrected_data]

