import numpy as np
import numpy.ma as MA
import matplotlib.pyplot as plt
import sys
from scipy.interpolate import interp1d
import time
import warnings
warnings.filterwarnings("ignore")
from functions_shear import *

var_list = sys.argv[1:]
for i in var_list:
    exec(i)

if dset == 4: #all = WITH DES WL
    label = 'L4'
    print('weak lensing likelihood included')

elif dset == 0: #reduced = WITHOUT DES WL
    label = 'L0'

#############################################################
#SHEAR POWER SPECTRUM
#############################################################

def isfloat(value):
    if value[-1] == ',':
        value = value[:-1]
    try:
        float(value)
        value = float(value)
        return value
    except ValueError:
        return value

"""if dset == 4:

    print('calculating shear power spectrum')
    print 'n(z) bin %s' % (which_bin)
    
    shearps_dir = '/n/dvorkin_lab/anadr/Neutrinos_DDE/DES_WPC135/shear_PS/nuLCDM/'
    
    ls = np.linspace(np.log10(2),np.log10(2500),50)
    ls = 10**ls

    shear_tot = []

    for num_file in range(ini,fin):

        f = open(shearps_dir + 'shearPS%s_n%s_%s.txt' % (dset,which_bin,num_file),'r')

        data_block = f.readlines()
        f.close()

        data = np.zeros((len(data_block),len(ls)))

        for (line_count,line) in enumerate(data_block):
            items = line.split()

            for count2,i in enumerate(items):
	        data[line_count][count2] = isfloat(i)

        for i in data:
	    shear_tot.append(i)

    arrs = [shear_tot]
    labs = ['shear']

    for i,j in zip(arrs,labs):

        for perc in [2.5,16,50,84,97.5]:
	
	    a = np.percentile(i,perc,axis=0)

            file1 = open('/n/home04/adiazrivero/Neutrinos_DDE/Results/CLs/DES_WPC135/Arrays_135/nuLCDM/%s%s_n%s_%s.txt' % (j,label,which_bin,perc), 'w')
            for item in a:
                file1.write('%s\n' % item)
            file1.close()"""

#########################################################

#Vivian grid
logkmin = -4.0
logkmax = 1.0
dlogk = 0.05
nk = (logkmax-logkmin)/dlogk

ks = 10**np.array([logkmin + dlogk*i for i in np.arange(1,nk+1,1)])

#########################################################

zstar = 1100
c = 2.99792458 * 1e5 #km/s

#########################################################

wanted_params = ['H1*','H100*','DISTL1*','DISTL100*','Pk1z0*','Pk100z5*']

chain_dir = '/n/dvorkin_lab/anadr/Neutrinos_DDE/DES_WPC135/chains/w_dark_energy_chain2/LCDM/post/'
fileloader = LoadFiles(chain_dir,'post_test%s' % label)
param_ind = fileloader.get_parameter_indices(wanted_params)

pk_params = []
hz_params = []
dl_params = []

for i in range(1,101):
    hz_params.append('H%s' % i)

for i in range(1,101):
    pk_params.append('P%sz0' % i)

for i in range(1,101):
    dl_params.append('DL%s' % i)

full_params = hz_params + dl_params + pk_params

#########################################################

zini = -0.05
dz = 0.04
nz = 100

z_Hz = [zini + dz*i for i in np.arange(1,nz+1,1)]

#########################################################

Pk0_tot = []
hz_tot = []
dl_tot = []

for num_file in np.arange(ini,fin,1):

    data,chain_len = fileloader.lcdm_read_hdp(num_file,full_params,param_ind)
    print('File number: %s --> %s chain samples' % (num_file,chain_len))

    for num_sample in range(chain_len):

        #start_time = time.time()
        Hz = np.zeros(len(z_Hz))

        for countz,i in enumerate(range(1,101)):
            Hz[countz] = c * data['H%s' % i][num_sample]

        hz_tot.append(Hz)
        
	#########################################################

        Pkz = np.zeros(len(ks))

        for countk,i in enumerate(range(1,101)):
            Pkz[countk] = data['P%sz0' % (i)][num_sample]

        Pk0_tot.append(Pkz)
        
	#########################################################

        DL = np.zeros(len(z_Hz))

        for countz,i in enumerate(range(1,101)):
            DL[countz] = data['DL%s' % i][num_sample]

        dl_tot.append(DL)

	print DL

        sys.exit()
        
	#########################################################

sys.exit()
 
#print time.time() - start_time

arrs = [hz_tot,Pk0_tot,dl_tot]
labs = ['hz','pk','dl']

for arr,lab in zip(arrs,labs):

    for perc in [2.5,16,50,84,97.5]:

	a = np.percentile(arr,perc,axis=0)

        file1 = open('/n/home04/adiazrivero/Neutrinos_DDE/Results/CLs/DES_WPC135/Arrays_135/nuLCDM/%s_%s_%s.txt' % (lab,perc,label), 'w')
        
	for item in a:
            file1.write('%s\n' % item)
        file1.close()

