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
import pyfits
import matplotlib as mpl
from matplotlib.colors import LogNorm
from scipy.ndimage import measurements

# pipeline files
from auxiliaries import *



def get_pixelfluxes(filename,inputfolder='',outputfolder='',starname=''):
  # Read a pixel file, and return the flux per pixel over time

  pixfile = pyfits.open(inputfolder+filename,memmap=True)  # open a FITS file
  #print f.info()
  L=len(pixfile[1].data) # number of images

  Xabs = pixfile[2].header['CRVAL2P'] # X position of pixel on kepler spacecraft
  Yabs = pixfile[2].header['CRVAL1P'] # Y position

  keplerid = pixfile[0].header['KEPLERID']
  kepmag = pixfile[0].header['Kepmag']
  obsmode = pixfile[0].header['OBSMODE']
  channel = pixfile[0].header['CHANNEL']
  module = pixfile[0].header['MODULE']
  RA = pixfile[0].header['RA_OBJ']
  DEC = pixfile[0].header['DEC_OBJ']

  obj_basics = [keplerid,kepmag,obsmode,channel,module,RA,DEC]
  obj_basics_head = '# 0 kepid, 1 kepmag, 2 obsmode, 3 channel, 4 module, 5 RA, 6 DEC'
  np.savetxt(os.path.join(outputfolder,'obj_basics_' + str(starname) + '.txt'),obj_basics,header=obj_basics_head,fmt='%s')


  # read dates and fluxes
  dates= []
  fluxes = []
  i = 0
  while i < L:
    # get dates and flux from file
    dates.append(np.array(pixfile[1].data[i][0]))
    flux = np.array(pixfile[1].data[i]['FLUX'], dtype='float64')

    # rework flux into 2D array
    NY=pixfile[2].header['NAXIS1']
    NX=pixfile[2].header['NAXIS2']
    flux = flux.reshape(NX, NY)

    fluxes.append(flux) # array over time of 2D array of flux-per-pixel

    i = i +1
  #print dates,fluxes
  return dates,fluxes,kepmag,Xabs,Yabs


def plot_pixelimages(dates,fluxes,outputfolder=''):
  # make plot of individual pixel images: CAREFUL this can take up a lot of space! This can be remorked to make nice movies of pixel files over time.
  i = 0
  folder = os.path.join(outputfolder,'pixelimages/')
  if not os.path.exists(folder):
    os.makedirs(folder)
  while i < len(dates):
    pl.figure(str(dates[i]))
    pl.imshow(fluxes[i],norm=LogNorm(),interpolation="none")
    pl.colorbar(orientation='vertical')
    pl.xlabel('X')
    pl.ylabel('Y')
    #pl.show()
    pl.savefig(os.path.join(folder,str(dates[i])+'.png'))
    pl.close()
    i = i + 1


def find_aperture(dates,fluxes,plot=True,starname='',outputfolder='',kepmag='na',cutoff_limit=2.):
  #
  # This definition reads a 2D array of fluxes (over time) and creates an aperture mask which can later be used to select those pixels for inclusion in light curve
  #

  # first sum all the flux over the different times, this assumes limited movement throughout the time series
  flux = np.nansum(fluxes,axis=0)

  # define which cutoff flux to use for including pixel in mask
  cutoff = cutoff_limit*np.median(flux) # perhaps a more elaborate way to define this could be found in the future but this seems to work pretty well.

  # define the aperture based on cutoff and make it into array of 1 and 0
  aperture =  np.array([flux > cutoff]) #scipy.zeros((np.shape(flux)[0],np.shape(flux)[1]), int)
  aperture = np.array(1*aperture)
  #print aperture
  outline_all = make_aperture_outline(aperture[0]) # an outline (ONLY for figure) of what we are including if we would make no breakups

  # this cool little trick allows us to measure distinct blocks of apertures, and only select the biggest one
  lw, num = measurements.label(aperture) # this numbers the different apertures distinctly
  area = measurements.sum(aperture, lw, index=np.arange(lw.max() + 1)) # this measures the size of the apertures
  aperture = area[lw].astype(int) # this replaces the 1s by the size of the aperture
  aperture = (aperture >= np.max(aperture))*1 # remake into 0s and 1s but only keep the largest aperture

  outline = make_aperture_outline(aperture[0]) # a new outline (ONLY for figure)

  if plot: # make aperture figure
    if not os.path.exists(outputfolder):
      os.makedirs(outputfolder)
    cmap = mpl.cm.get_cmap('Greys', 20)
    pl.figure('Aperture_' + str(starname))
    pl.imshow(flux,norm=LogNorm(),interpolation="none")#,cmap=cmap)
    pl.plot(outline_all[:, 0], outline_all[:, 1],color='green', zorder=10, lw=2.5)
    pl.plot(outline[:, 0], outline[:, 1],color='red', zorder=10, lw=2.5)#,label=str(kepmag))

    #pl.colorbar(orientation='vertical')
    pl.xlabel('X',fontsize=15)
    pl.ylabel('Y',fontsize=15)
    pl.legend()
    #pl.xlim([-1,18])
    #pl.ylim([-1,16])
    #pl.xticks([0,5,10,15])
    #pl.yticks([0,5,10,15])
    pl.tight_layout()
    pl.savefig(os.path.join(outputfolder,'aperture_' + str(starname)+'.pdf'))
    #pl.close()
    #pl.show()
  return aperture

def estimate_background(dates,fluxes,aperture,outputfolder='',starname=''):
  #
  # estimate the background flux level. this is just the average level of flux on the pixels after correcting for outliers (times the amount of pixels includeed)
  # (note that this may no longer be needed in later versions of K2 data release
  #

  number_of_pixels = np.sum(aperture)

  # estimate background by looking at median, after repeated clipping of what are stars.
  i = 0
  flux_bg_list = []
  while i < len(fluxes):
    flux = np.array(fluxes[i]).ravel()
    flux_clipped = sigma_clip(flux,3,iterative=False)
    flux_bg_list.append(np.median(flux_clipped))
    i = i + 1

  pl.figure()
  pl.plot(dates,flux_bg_list)#,label='')
  pl.xlabel('Time [d]')
  pl.ylabel('Background flux / pixel')
  pl.legend()
  pl.savefig(os.path.join(outputfolder,'background_' + str(starname) + '.png'))
  #pl.show()
  background = number_of_pixels * np.array(flux_bg_list)
  return background



def make_aperture_outline(frame, no_combined_images=1, threshold=0.5):
  ## this is a little module that defines so called outlines to be used for plotting apertures

  thres_val = no_combined_images * threshold
  mapimg = (frame > thres_val)
  ver_seg = np.where(mapimg[:,1:] != mapimg[:,:-1])
  hor_seg = np.where(mapimg[1:,:] != mapimg[:-1,:])

  l = []
  for p in zip(*hor_seg):
      l.append((p[1], p[0]+1))
      l.append((p[1]+1, p[0]+1))
      l.append((np.nan,np.nan))

  # and the same for vertical segments
  for p in zip(*ver_seg):
      l.append((p[1]+1, p[0]))
      l.append((p[1]+1, p[0]+1))
      l.append((np.nan, np.nan))


  segments = np.array(l)

  x0 = -0.5
  x1 = frame.shape[1]+x0
  y0 = -0.5
  y1 = frame.shape[0]+y0

  #   now we need to know something about the image which is shown
  #   at this point let's assume it has extents (x0, y0)..(x1,y1) on the axis
  #   drawn with origin='lower'
  # with this information we can rescale our points
  segments[:,0] = x0 + (x1-x0) * segments[:,0] / mapimg.shape[1]
  segments[:,1] = y0 + (y1-y0) * segments[:,1] / mapimg.shape[0]

  return segments



def get_lightcurve(dates,fluxes,aperture,starname='',plot=True,outputfolder='',Xabs=0,Yabs=0):
  #
  # Go from an array of fluxes per pixel (= X,Y,time) to a light curve, by summing the flux in the pixels that are included in the aperture
  #

  # get only the fluxes that full in the aperture. since aperture is defined as 0 everywhere outside we can just multiply
  aperture_fluxes = fluxes*aperture
  background_fluxes = estimate_background(dates,fluxes,aperture,outputfolder=outputfolder,starname=starname)

  # sum over axis 2 and 1 (the X and Y positions), (axis 0 is the time)
  f_t = np.nansum(np.nansum(aperture_fluxes,axis=2), axis=1) - background_fluxes

  # first make a matrix that contains the x and y positions
  x_pixels = [range(0,np.shape(aperture_fluxes)[2])] * np.shape(aperture_fluxes)[1]
  y_pixels = np.transpose([range(0,np.shape(aperture_fluxes)[1])] * np.shape(aperture_fluxes)[2])

  # multiply the position matrix with the aperture fluxes to obtain x_i*f_i and y_i*f_i
  xpos_times_flux = np.nansum( np.nansum( x_pixels*aperture_fluxes, axis=2), axis=1)
  ypos_times_flux = np.nansum( np.nansum( y_pixels*aperture_fluxes, axis=2), axis=1)

  # calculate centroids
  Xc = xpos_times_flux / f_t
  Yc = ypos_times_flux / f_t

  # remove the empty frames (why do they happen?), they have f_t zero
  Xc = np.array(Xc)[f_t > 0]
  Yc = np.array(Yc)[f_t > 0]
  dates = np.array(dates)[f_t > 0]
  f_t = np.array(f_t)[f_t > 0]

  Xc = np.array(Xc) + Xabs
  Yc = np.array(Yc) + Yabs

  if plot:
    # plot flux and centroid curve
    g, (ax1, ax2, ax3) = pl.subplots(3, sharex='col', num='timeseries_'+str(starname))
    ax1.plot(np.array(dates)[f_t>0],f_t[f_t>0],'*') # little hack to avoid values 0 that will skew the plot, but they are not (yet) removed from the LC
    ax1.set_ylabel('Flux [count]')
    ax2.plot(dates,Xc,'*')
    ax2.set_ylabel('Xc')
    ax3.plot(dates,Yc,'*')
    ax3.set_ylabel('Yc')
    ax3.set_xlabel('Time [d]')
    g.savefig(os.path.join(outputfolder,'flux_centroids_' + str(starname) + '.png'))
    #pl.show()

    pl.figure('Raw data')
    pl.plot(np.array(dates)[f_t>0],f_t[f_t>0],'.',color='grey')
    pl.xlabel('Time [d]')
    pl.ylabel('Flux [count]')
    pl.savefig(os.path.join(outputfolder,'raw_data_' + str(starname) + '.png'))
  return [dates,f_t,Xc,Yc]


def overwrite_centroids(centroidstar='',outputpath=''):
  # little module to overwrite centroids, can be used to take the centroid position of another star or average of stars (not used currently)

  folder = os.path.join(outputpath,centroidstar)
  t,Xc,Yc = np.loadtxt(os.path.join(folder,'centroids_' + str(centroidstar) + '.txt'),unpack=True)

  return Xc,Yc



def remove_known_outliers(t,f_t,Xc,Yc):
  # remove predefined known outliers

  t = np.array(t)
  f_t = np.array(f_t)
  Xc = np.array(Xc)
  Yc = np.array(Yc)
  pl.figure()
  pl.plot(t,f_t,'r.')
  outlier1 = [2064.11,2064.58] # some very large outliers
  outlier2 = [2060.0,2061.4] # very first day is not great
  outliers = [outlier1,outlier2]
  i = 0
  while i < len(outliers):
    mask = ~( (t > outliers[i][0])*(t < outliers[i][1]))
    t = t[mask]
    f_t = f_t[mask]
    Xc = Xc[mask]
    Yc = Yc[mask]
    i = i + 1

  pl.plot(t,f_t,'b.')

  return t,f_t,Xc,Yc



def gotoflux(starname,outputpath='',inputpath='',campaign=2,cutoff_limit=2.):
  '''
  Read a specific pixel file and extract a light curve from it
  '''


  filename = 'ktwo' + starname + '-c0' + str(campaign) + '_lpd-targ.fits' # filename as downloaded from MAST
  outputfolder = os.path.join(outputpath,str(starname))
  if not os.path.exists(outputfolder):
    os.makedirs(outputfolder)

  # Read fluxes per (X,Y) over time
  dates,fluxes,kepmag,Xabs,Yabs = get_pixelfluxes(filename,inputfolder=inputpath,outputfolder=outputfolder,starname=starname)

  # Define aperture
  aperture = find_aperture(dates,fluxes,plot=True,starname=starname,outputfolder=outputfolder,kepmag=kepmag,cutoff_limit=cutoff_limit)

  # Create light curve out of fluxes and a fixel aperture
  t,f_t,Xc,Yc = get_lightcurve(dates,fluxes,aperture,starname=starname,outputfolder=outputfolder,Xabs=Xabs,Yabs=Yabs)

  # Remove simple known outliers
  t,f_t,Xc,Yc = remove_known_outliers(t,f_t,Xc,Yc)

  np.savetxt(os.path.join(outputfolder,'lightcurve_raw_' + str(starname) + '.txt'),np.transpose([t,f_t,Xc,Yc]),header='Time, flux, Xc, Yc')
  np.savetxt(os.path.join(outputfolder,'centroids_' + str(starname) + '.txt'),np.transpose([t,Xc,Yc]),header='Time, Xc, Yc')


  return [t,f_t,Xc,Yc]