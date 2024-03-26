import numpy as np
from params import *

def output_superoper_array(t,s_array,pre_name):

  for j in range(DOF_E_SQ):
    a = str(int(j/DOF_E))
    b = str(int(j%DOF_E))
    for k in range(DOF_E_SQ):
      c = str(int(k/DOF_E))
      d = str(int(k%DOF_E))

      # outfileStr for super-operator
      outfileStr = pre_name + a + b + c + d + PARAM_STR + ".txt"

      f = open(outfileStr, "w")

      for i in range(TIME_STEPS):
          f.write("%s\t%s\t%s\n"%(float(t[i]), s_array[i][j][k].real, s_array[i][j][k].imag))
      f.close()
  return 0

def read_superoper_array(pre_name):

    S_array_read = np.zeros((TIME_STEPS, DOF_E_SQ, DOF_E_SQ), dtype=np.complex_)
    time = np.zeros((TIME_STEPS), dtype=np.float_)

    for j in range(DOF_E_SQ):
        a = str(int(j/DOF_E))
        b = str(int(j%DOF_E))
        for k in range(DOF_E_SQ):
            c = str(int(k/DOF_E))
            d = str(int(k%DOF_E))

            # opens and reads the super-operator file
            time_read, Sreal, Simag = np.hsplit(
                np.loadtxt(pre_name+"%s%s%s%s"%(a,b,c,d) + PARAM_STR
                               + ".txt"), 3)

            # stores values in variables and combines real and imag parts of super-operator
            for i in range(TIME_STEPS):
                time[i] = time_read[i]
                S_array_read[i][j][k] = Sreal[i] + 1.j * Simag[i]

    return time,S_array_read

def output_operator_array(timeVec, sigma, pre_name):

    for j in range(DOF_E_SQ):
        a = str(int(j/DOF_E))
        b = str(int(j%DOF_E))

        # outfileStr for vectorize form of operator
        outfileGQMEStr = pre_name + a + b 
        outfileGQMEStr += PARAM_STR + ".txt"

        f = open(outfileGQMEStr, "w")

        for i in range(len(timeVec)):
            f.write("%s\t%s\t%s\n"%(timeVec[i], sigma[i][j].real, sigma[i][j].imag))
        f.close()

def read_operator_array(pre_name):

    O_array_read = np.zeros((TIME_STEPS, DOF_E_SQ), dtype=np.complex_)
    time = np.zeros((TIME_STEPS), dtype=np.float_)

    for j in range(DOF_E_SQ):
        a = str(int(j/DOF_E))
        b = str(int(j%DOF_E))

        # opens and reads the operator file
        time_read, Oreal, Oimag = np.hsplit(
            np.loadtxt(pre_name+"%s%s"%(a,b) + PARAM_STR
                           + ".txt"), 3)

        # stores values in variables and combines real and imag parts of U
        for i in range(TIME_STEPS):
            time[i] = time_read[i]
            O_array_read[i][j] = Oreal[i] + 1.j * Oimag[i]

    return time,O_array_read
