# general python files
import sys
import os
import pylab as pl
import numpy as np
import pyfits
import scipy
from os import listdir
from os.path import isfile, join
import sys

from run_pipeline import run


outputpath = 'example_reduced/'
inputpath = 'example_input/'

starnames = [ f[4:13] for f in listdir(inputpath) if isfile(join(inputpath,f)) ] # get all files in the folder, just grab the EPIC number of the filename

i = 0

exc_list = []
while i < len(starnames):
  print 'Now running stars, number '
  print str(i)
  run(starnames[i],outputpath=outputpath,inputpath=inputpath,makelightcurve=True,campaign=1)

  try:
    run(starnames[i],outputpath=outputpath,inputpath=inputpath,makelightcurve=True,campaign=1)
  
  except Exception as inst:
    print inst
    exc_list.append(inst)
  i = i + 1

print 'Module failed for some stars:'
print exc_list

print 'Done...'