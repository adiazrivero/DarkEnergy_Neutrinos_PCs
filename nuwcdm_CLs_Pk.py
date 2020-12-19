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

#########################################################

#Vivian grid
logkmin = -4.0
logkmax = 1.0
dlogk = 0.05
nk = (logkmax-logkmin)/dlogk

ks = 10**np.array([logkmin + dlogk*i for i in np.arange(1,nk+1,1)])

#ks = open('ks_10.txt','r')
#ks = [float(i.strip()) for i in ks]

#########################################################

zstar = 1100
c = 2.99792458 * 1e5 #km/s

#########################################################

wanted_params = ['H1*','H100*','WPC1','WPC%s' % num_pca,'DISTL1*','DISTL100*','Pk1z%s*' % redshift,'Pk100z%s*' % redshift]


chain_dir = '/n/dvorkin_lab/anadr/Neutrinos_DDE/DES_WPC135/chains/w_dark_energy_chain2/post_pk/'
fileloader = LoadFiles(chain_dir,'post_test%s' % num)
param_ind = fileloader.get_parameter_indices(wanted_params)

print(param_ind)

wpc_params = []
pk_params = []
hz_params = []
dl_params = []

for i in range(1,101):
    hz_params.append('H%s' % i)

for i in range(1,num_pca+1):
    wpc_params.append('WPC%s' % i)

for i in range(1,101):
    pk_params.append('P%sz%s' % (i,redshift))

for i in range(1,101):
    dl_params.append('DL%s' % i)

full_params = hz_params + wpc_params + dl_params + pk_params

#########################################################

#zs = np.arange(0,5.1,0.1)

zini = -0.05
dz = 0.04
nz = 100

zarr = [zini + dz*i for i in np.arange(1,nz+1,1)]

#########################################################

Pk_tot = []

for num_file in np.arange(ini,fin,1):

    data,chain_len = fileloader.nuwcdm_read_hdp(num_file,full_params,param_ind)
    print('File number: %s --> %s chain samples' % (num_file,chain_len))

    for num_sample in range(chain_len):
        
	"""Pkz = np.zeros((len(zs),len(ks))

        for countz,j in enumerate(zs):
            for countk,i in enumerate(range(1,101)):
                if j == int(j):
                    Pkz[countz][countk] = data['P%sz%s' % (i,int(j))][num_sample]
                else:
                    Pkz[countz][countk] = data['P%sz%s' % (i,j)][num_sample]"""


        #Pkz = np.zeros((len(zs),100))
        Pkz = np.zeros(len(ks))

        for countk,i in enumerate(range(1,101)):
            Pkz[countk] = data['P%sz%s' % (i,redshift)][num_sample]

	print(Pkz)

	sys.exit()

        Pk_tot.append(Pkz)
	plt.loglog(ks,Pkz)
	plt.show()
	sys.exit()
        #########################################################

arrs = [Pk_tot]
labs = ['pk_z=%s' % redshift]

for arr,lab in zip(arrs,labs):

    for perc in [2.5,16,50,84,97.5]:

	a = np.percentile(arr,perc,axis=0)

        file1 = open('/n/home04/adiazrivero/Neutrinos_DDE/Results/CLs/DES_WPC135/Arrays_135/nuwCDM/bin%s/%s_%s_%s.txt' % (which_bin,lab,perc,num), 'w')
        
	for item in a:
            file1.write('%s\n' % item)
        file1.close()

