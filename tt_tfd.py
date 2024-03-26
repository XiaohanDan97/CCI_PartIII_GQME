#package : numpy, ttpy

import numpy as np
from params import *

import math, tt
import tt.ksl
import time
import initial as ini

def tt_tfd(initial_state):

  #tt_heavu,tt_heavd: Heaviside functions, to calculate populations 
  #tt_ul,tt_ur: arrays to cut the wavepacket, to calculate coherences
  tt_heav_donor,tt_heav_accepter,tt_ul,tt_ur,y0 = ini.initial(initial_state)
  A = ini.construct_Hamil()

  RDO_arr = np.zeros((TIME_STEPS, DOF_E_SQ), dtype=np.complex_)
  t = np.arange(0, TIME_STEPS * DT, DT)

  # Propagation loop
  START_TIME = time.time()
  for ii in range(TIME_STEPS):

    y0 = tt.ksl.ksl(A, y0, DT)
    print(ii,t[ii])

    #this equal to the trace of nuclear DOFs, and then select electronic state
    RDO_arr[ii][0] = np.abs(tt.dot(tt_heav_donor * y0, tt_heav_donor * y0))
    RDO_arr[ii][3] = np.abs(tt.dot(tt_heav_accepter * y0, tt_heav_accepter * y0))
    #the coherence 
    RDO_arr[ii][2] = tt.dot(tt.matvec(tt_ul,y0), tt.matvec(tt_ur, y0))
    RDO_arr[ii][1] = tt.dot(tt.matvec(tt_ur,y0), tt.matvec(tt_ul, y0))

  print("\tPropagation time:", time.time() - START_TIME)

  return t,RDO_arr 
