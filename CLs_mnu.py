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

wanted_params = ['mnu','S8*']
#wanted_params = ['H1*','H100*','WPC1','WPC%s' % num_pca,'DISTL1*','DISTL100*','Pk1z0*','Pk100z5*']

chain_dir = '/n/dvorkin_lab/anadr/Neutrinos_DDE/DES_WPC135/chains/w_dark_energy_chain2/post_pk/'
fileloader = LoadFiles(chain_dir,'post_test%s' % num)
param_ind = fileloader.get_parameter_indices(wanted_params)

print(param_ind)

full_params = ['mnu','S8*']

mnu_tot = []
s8_tot = []

for num_file in np.arange(ini,fin,1):

    #data,chain_len = fileloader.read_mnu(num_file,full_params,param_ind)
    data,chain_len = fileloader.read_mnu_s8(num_file,full_params,param_ind)
    print('File number: %s --> %s chain samples' % (num_file,chain_len))

    #print(min(data['mnu']),max(data['mnu']))

    mnu_tot.append(data['mnu'])
    s8_tot.append(data['S8*'])

plt.scatter(mnu_tot,s8_tot,s=0.01)
plt.xlim(0.1,0.9)
plt.ylim(0.7,0.9)
plt.show()

sys.exit()

p68 = np.percentile(np.array(mnu_tot),68)
p95 = np.percentile(np.array(mnu_tot),95)
p99 = np.percentile(np.array(mnu_tot),99)
print('68th percentile: %.3f' % p68) 
print('95th percentile: %.3f' % p95) 
print('99th percentile: %.3f' % p99) 


sys.exit()

arrs = [wz_tot,hz_tot,Pk0_tot,dl_tot]
labs = ['wz','hz','pk','dl']

for arr,lab in zip(arrs,labs):

    for perc in [2.5,16,50,84,97.5]:

	a = np.percentile(arr,perc,axis=0)

        file1 = open('/n/home04/adiazrivero/Neutrinos_DDE/Results/CLs/DES_WPC135/Arrays_135/nuwCDM/%s_%s_%s.txt' % (lab,perc,num), 'w')
        
	for item in a:
            file1.write('%s\n' % item)
        file1.close()

