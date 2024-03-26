#package : numpy, ttpy

import numpy as np
from params import *

import math, tt
import tt.ksl

def initial(istate):

  #initial state 
  #Build initial ground state at spin-Up state
  su = np.array([1,0])
  sd = np.array([0,1])
  tt_su = tt.tensor(su)
  tt_sd = tt.tensor(sd)
  e1 = np.sqrt(0.5) * (su + sd)
  e2 = np.sqrt(0.5) * (su + 1j * sd)
  tt_e1 = tt.tensor(e1)
  tt_e2 = tt.tensor(e2)  

  gs = np.zeros((occ))
  gs[0] = 1.
  tt_gs = tt.tensor(gs)
  if(istate==0):
    tt_psi0 = tt_su
  elif(istate==1):
    tt_psi0 = tt_e1
  elif(istate==2):
    tt_psi0 = tt_e2
  elif(istate==3):
    tt_psi0 = tt_sd
    
  for k in range(2 * DOF_N): # double space formation
      tt_psi0 = tt.kron(tt_psi0, tt_gs)
  
  y0 = tt_psi0 # initial wavepacket
  print(y0)

  # Extent y0 to higher rank: Add noise, for higher rank KSL propagation
  radd = np.array([1,9]) # the rank of the first bath mode core is capped by occ
  radd = np.append(radd, np.repeat(9, DOF_N * 2-3))
  radd = np.append(radd, np.array([9,1])) # and the last core
  tt_rand = tt.rand(occ, DOF_N * 2, radd) # random TT array with desired size and rank
  tt_rand = tt_rand * tt_rand.norm()**(-1) # Renormalize noise
  tt_rand = tt.kron(tt.ones(2,1), tt_rand)# Append electronic site
  y0 = y0 + tt_rand * 1e-10 # Ensure noise is small
  print('extent rank',y0)

  # Heaviside functions, for selecting electronic states from overall wavefunction
  tt_heavu = tt.kron(tt_su,tt.ones(occ, dim*2))
  tt_heavd = tt.kron(tt_sd,tt.ones(occ, dim*2))
  
  # arrays to cut the wavepacket, to calculate coherences
  ul = np.array([[1,0],[0,0]])
  ur = np.array([[0,1],[0,0]])
  tt_ul = tt.matrix(ul)
  tt_ur = tt.matrix(ur)
  for i in range(DOF_N * 2):
      tt_ul = tt.kron(tt_ul, tt.eye(occ,1))
      tt_ur = tt.kron(tt_ur, tt.eye(occ,1))

  return tt_heavu,tt_heavd,tt_ul,tt_ur,y0

def construct_Hamil():

  om = OMEGA_C / DOF_N * (1 - np.exp(-OMEGA_MAX/OMEGA_C))

  # initialize arrays for parameters
  freq = np.zeros((DOF_N)) # frequency
  ck = np.zeros((DOF_N))   # linear electron-phonon coupling constant
  gk = np.zeros((DOF_N))   # ck in occupation number representation
  thetak = np.zeros((DOF_N)) # temperature-dependent mixing parameter in TFD
  sinhthetak = np.zeros((DOF_N)) # sinh(theta)
  coshthetak = np.zeros((DOF_N)) # cosh(theta)
  for i in range(DOF_N):
      freq[i] = -OMEGA_C * np.log(1-(i+1) * om/(OMEGA_C)) # Ohmic frequency
      ck[i] = np.sqrt(XI * om) * freq[i] #Ohmic coupling constant
      gk[i] = -ck[i] / np.sqrt(2 * freq[i]) #Transfer ck to occ. num. representation
  
      thetak[i] = np.arctanh(np.exp(-BETA * freq[i]/2)) #theta, defined for harmonic models
      sinhthetak[i] = np.sinh(thetak[i]) #sinh(theta)
      coshthetak[i] = np.cosh(thetak[i]) #cosh(theta)
  
  
  # constructing Pauli operators
  px = np.array([[0,1],[1,0]])
  pz = np.array([[1,0],[0,-1]])
  # Build electronic site energy matrix
  He = EPSILON  * pz + GAMMA_DA  * px
  # TT-ize that energy matrix
  tt_He = tt.matrix(He)
  tt_He = tt.kron(tt_He, tt.eye(occ, DOF_N * 2))
  
  
  # Build number operator, corresponds to harmonic oscillator Hamiltonian
  numoc = np.diag(np.arange(0, occ, 1))
  # Initiate the TT-ized number operator as a zero TT array with shape of occ^N
  tt_numoc = tt.eye(occ, DOF_N)*0.
  # Construct number operator as TT
  for k in range(DOF_N):
      if k == 0:
          tmp = tt.kron(tt.matrix(numoc) * freq[k], tt.eye(occ, DOF_N - 1))
      elif 0 < k < DOF_N-1:
          tmp = tt.kron(tt.eye(occ,k-1), tt.matrix(numoc) * freq[k])
          tmp = tt.kron(tmp,tt.eye(occ, DOF_N - k))
      else:
          tmp = tt.kron(tt.eye(occ,k), tt.matrix(numoc) * freq[k])
      tt_numoc = tt_numoc + tmp
      tt_numoc = tt_numoc.round(eps)

  # Ensure correct dimensionality
  tt_Ie = tt.eye(2,1)
  tt_systemnumoc = tt.kron(tt_Ie, tt_numoc)
  tt_systemnumoc = tt.kron(tt_systemnumoc, tt.eye(occ, DOF_N))
  
  # create a duplicate of number operator for the ficticious system
  tt_tildenumoc = tt.kron(tt_Ie, tt.eye(occ, DOF_N))
  tt_tildenumoc = tt.kron(tt_tildenumoc, tt_numoc)
  
  thetak = np.zeros((DOF_N)) #temperature-dependent mixing parameter in TFD
  sinhthetak = np.zeros((DOF_N)) #sinh(theta)
  coshthetak = np.zeros((DOF_N)) #cosh(theta)
  for i in range(DOF_N):
      thetak[i] = np.arctanh(np.exp(-BETA * freq[i]/2)) #theta, defined for harmonic models
      sinhthetak[i] = np.sinh(thetak[i]) #sinh(theta)
      coshthetak[i] = np.cosh(thetak[i]) #cosh(theta)
  
  #Build displacement operator, corresponds to x operator in real space
  eneroc = np.zeros((occ, occ))
  for i in range(occ - 1):
      eneroc[i,i+1] = np.sqrt(i+1)
      eneroc[i+1,i] = eneroc[i,i+1]
  
  # initialize displacement operator
  tt_energy = tt.eye(occ, DOF_N)*0.
  for k in range(DOF_N):
      if k == 0:
          # coshtheta takes account for energy flow from real to ficticious system
          # thus takes account for temperature effect
          tmp = tt.kron(tt.matrix(eneroc) * gk[k] * coshthetak[k], tt.eye(occ, DOF_N - 1))
      elif 0 < k < DOF_N - 1:
          tmp = tt.kron(tt.eye(occ,k-1), tt.matrix(eneroc) * gk[k] * coshthetak[k])
          tmp = tt.kron(tmp,tt.eye(occ, DOF_N - k))
      else:
          tmp = tt.kron(tt.eye(occ,k), tt.matrix(eneroc) * gk[k] * coshthetak[k])
      tt_energy = tt_energy + tmp
      tt_energy = tt_energy.round(eps)
  tt_systemenergy = tt.kron(tt.matrix(pz), tt_energy)
  tt_systemenergy = tt.kron(tt_systemenergy, tt.eye(occ, DOF_N))
  
  
  # initialize displacement operator
  tt_tilenergy = tt.eye(occ, DOF_N)*0.
  for k in range(DOF_N):
      if k == 0:
          tmp = tt.kron(tt.matrix(eneroc) * gk[k] * sinhthetak[k], tt.eye(occ, DOF_N - 1))
      elif 0 < k < DOF_N - 1:
          tmp = tt.kron(tt.eye(occ,k-1), tt.matrix(eneroc) * gk[k] * sinhthetak[k])
          tmp = tt.kron(tmp, tt.eye(occ, DOF_N - k))
      else:
          tmp = tt.kron(tt.eye(occ,k), tt.matrix(eneroc) * gk[k] * sinhthetak[k])
      tt_tilenergy = tt_tilenergy + tmp
      tt_tilenergy = tt_tilenergy.round(eps)
  tt_tildeenergy = tt.kron(tt.matrix(pz), tt.eye(occ, DOF_N))
  tt_tildeenergy = tt.kron(tt_tildeenergy, tt_tilenergy)
  
  #The total propogation Hamiltonian
  # Note that ficticious Harmonic oscillators carry negative sign
  H = tt_He + tt_systemnumoc - tt_tildenumoc + tt_systemenergy + tt_tildeenergy
  H = H.round(eps)
  # Construct propagation operator, d/dt psi(t0)=A psi(t0)
  A = -1j * H

  return A

