import numpy as np
import numpy.ma as MA
import matplotlib.pyplot as plt
import sys
from scipy.interpolate import interp1d
import time
from functions_shear import *
import warnings
warnings.filterwarnings("ignore")

def isfloat(value):
    if value[-1] == ',':
        value = value[:-1]
    try:
        float(value)
        value = float(value)
        return value
    except ValueError:
        return value

var_list = sys.argv[1:]
for i in var_list:
    exec(i)

if num == 1:
    num = ''
    num_pca = 1

elif num == 4:
    num_pca = 1

elif num == 2 or num ==5:
    num_pca = 3

elif num == 3 or num == 6:
    num_pca = 5


#############################################################
#SHEAR POWER SPECTRUM
#############################################################

"""if num == 4 or num == 5 or num == 6:

    print 'n(z) bin %s' % (which_bin)

    ls = np.linspace(np.log10(2),np.log10(2500),50)
    ls = 10**ls

    shear_tot = []

    for num_file in range(ini,fin):
        
	#if num == 4 and which_bin == 1:      
	    #f = open('/n/dvorkin_lab/anadr/Neutrinos_DDE/DES_WPC135/shear_PS/nuwCDM/viv_kgrid/shearPS%s_n%s_%s_test.txt' % (num,which_bin,num_file),'r')

	dire = '/n/dvorkin_lab/anadr/Neutrinos_DDE/DES_WPC135/shear_PS/nuwCDM/viv_kgrid/Jan25_bugfix/'
	print(dire)
	#f = open('/n/dvorkin_lab/anadr/Neutrinos_DDE/DES_WPC135/shear_PS/nuwCDM/viv_kgrid/shearPS%s_n%s_%s.txt' % (num,which_bin,num_file),'r')
	f = open(dire + 'shearPS%s_n%s_%s.txt' % (num,which_bin,num_file),'r')

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

            #file1 = open('/n/home04/adiazrivero/Neutrinos_DDE/Results/CLs/DES_WPC135/Arrays_135/nuwCDM/%s%s_n%s_%s.txt' % (j,num,which_bin,perc), 'w')
            file1 = open('/n/home04/adiazrivero/Neutrinos_DDE/Results/CLs/DES_WPC135/Arrays_135/nuwCDM/Jan25_bugfix/%s%s_n%s_%s.txt' % (j,num,which_bin,perc), 'w')
            for item in a:
                file1.write('%s\n' % item)
            file1.close()
"""
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

wanted_params = ['H1*','H100*','WPC1','WPC%s' % num_pca,'DISTL1*','DISTL100*','Pk1z0*','Pk100z5*']

chain_dir = '/n/dvorkin_lab/anadr/Neutrinos_DDE/DES_WPC135/chains/w_dark_energy_chain2/post_pk/'
fileloader = LoadFiles(chain_dir,'post_test%s' % num)
param_ind = fileloader.get_parameter_indices(wanted_params)

wpc_params = []
pk_params = []
hz_params = []
dl_params = []

for i in range(1,101):
    hz_params.append('H%s' % i)

for i in range(1,num_pca+1):
    wpc_params.append('WPC%s' % i)

for j in np.arange(0,5.1,0.1):
    for i in range(1,101):
        if j == int(j):
            pk_params.append('P%sz%s' % (i,int(j)))
        else:
            pk_params.append('P%sz%s' % (i,j))

for i in range(1,101):
    dl_params.append('DL%s' % i)

full_params = hz_params + wpc_params + dl_params + pk_params

#########################################################

zini = -0.05
dz = 0.04
nz = 100

zarr = [zini + dz*i for i in np.arange(1,nz+1,1)]

#########################################################

pcs = np.loadtxt('WFIRST_outPCA_CosmoMC.dat', skiprows=1)
pc_z = pcs[:,0]
pc_vecs = pcs[:,1:num_pca+1]

print 'PCs loaded'

#########################################################

wz_tot = []
Pk0_tot = []
hz_tot = []
dl_tot = []

for num_file in np.arange(ini,fin,1):

    data,chain_len = fileloader.nuwcdm_read_hdp(num_file,full_params,param_ind)
    print('File number: %s --> %s chain samples' % (num_file,chain_len))

    for num_sample in range(chain_len):

        Hz = np.zeros(len(zarr))

        for countz,i in enumerate(range(1,101)):
            Hz[countz] = c*data['H%s' % i][num_sample]

        hz_tot.append(Hz)
	
        #########################################################

        Pkz = np.zeros(len(ks))

        for countk,i in enumerate(range(1,101)):
            Pkz[countk] = data['P%sz0' % (i)][num_sample]

        Pk0_tot.append(Pkz)

        #########################################################

        wz_alphas = np.zeros(num_pca)
        for count,i in enumerate(range(1,num_pca+1)):
            wz_alphas[count] = data['WPC%s' % i][num_sample]
        w_vec = wz_alphas * pc_vecs
        w_vec = np.sum(w_vec,axis=1) -1
        #w_spline = interp1d(pc_z, w_vec, kind='cubic')
        
        wz_tot.append(w_vec)

        #########################################################

        DL = np.zeros(len(zarr))

        for countz,i in enumerate(range(1,101)):
            DL[countz] = data['DL%s' % i][num_sample]

        dl_tot.append(DL)
        
	#########################################################

arrs = [wz_tot,hz_tot,Pk0_tot,dl_tot]
labs = ['wz','hz','pk','dl']

for arr,lab in zip(arrs,labs):

    for perc in [2.5,16,50,84,97.5]:

	a = np.percentile(arr,perc,axis=0)

        file1 = open('/n/home04/adiazrivero/Neutrinos_DDE/Results/CLs/DES_WPC135/Arrays_135/nuwCDM/%s_%s_%s.txt' % (lab,perc,num), 'w')
        
	for item in a:
            file1.write('%s\n' % item)
        file1.close()

