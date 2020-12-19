import numpy as np
import numpy.ma as MA
import matplotlib.pyplot as plt
import sys
from scipy.interpolate import interp1d
import time
import warnings

warnings.filterwarnings("ignore")

c = 2.99792458 * 1e5 #km/s

def isfloat(value):

    try:
        float(value)
        value = float(value)
        return value

    except ValueError:
        return value

class LoadFiles:

    def __init__(self,directory,fileroot): #add types (Python 3+)
        self.directory = directory
        self.fileroot = fileroot

    def read_file(self,filename,column_names):

        f = open(filename)
	col_names = column_names
        data_block = f.readlines()
        f.close()

        data = {}

        for col_name in col_names:
            #data[col_name] = np.zeros(len(data_block))
            data[col_name] = [0] * len(data_block)

        for (line_count,line) in enumerate(data_block):

            #items = np.array(line.split())
            items = line.split()

            for (col_count,col_name) in enumerate(col_names):
                value = items[col_count]
                data[col_name][line_count] = isfloat(value)

        return data

    def get_parameter_indices(self,wanted_params):

        col_names = ['parameter','latex']
        all_params = self.read_file(self.directory + self.fileroot + '.paramnames',col_names)
	all_params = all_params['parameter']
        param_ind = []

        for name in wanted_params:
            ind = [i for i,e in enumerate(all_params) if e == name]
            param_ind.append(ind)

        param_ind = [item + 2 for sublist in param_ind for item in sublist]

        return param_ind

    def load_nz(self,nz_file,which_bin):

        cols = ['z','1','2','3','4']
        data = self.read_file(nz_file,cols)

        z_nz = np.array(data['z'])
        nz = np.array(data['%s' % which_bin])

        #have to normalize n(z)
        dz = z_nz[1]-z_nz[0]
        norm = 1/np.sum(dz*nz)
        nz_tot = norm*nz

        #print('normalization = %s' % norm)

        return z_nz,nz_tot

    def lcdm_read_hdp(self,filenumber,column_names,indlist):

        filename = self.directory + self.fileroot + '_%s.txt' % filenumber
        print(filename)

        f = open(filename)
        col_names = column_names

        data_block = f.readlines()
        f.close()

        data = {}

        for col_name in col_names:
            data[col_name] = np.zeros(len(data_block))

        for (line_count,line) in enumerate(data_block):

            #ind1 = [0]+[1]+[i for i in indlist[:3]] #weight,likelihood // om,ol,h0
            ind1 = [i for i in indlist[:2]] #H(z)
            ind2 = [i for i in indlist[2:4]] #D_L(z)
            ind3 = [i for i in indlist[4:]] #P(k,z)

            #items1 = np.array(line.split())[ind1]
            items1 = np.array(line.split())[ind1[0]:ind1[1]+1]
            items2 = np.array(line.split())[ind2[0]:ind2[1]+1]
            items3 = np.array(line.split())[ind3[0]:ind3[1]+1]

            items = np.concatenate([items1,items2,items3])

            for (col_count,col_name) in enumerate(col_names):
                value = items[col_count]
                data[col_name][line_count] = isfloat(value)

        return data,len(data_block)

    def nuwcdm_read_hdp(self,filenumber,column_names,indlist):

        filename = self.directory + self.fileroot + '_%s.txt' % filenumber
        print(filename)
        
	f = open(filename)
        col_names = column_names

        data_block = f.readlines()
        f.close()

        data = {}

        for col_name in col_names:
            data[col_name] = np.zeros(len(data_block))

        for (line_count,line) in enumerate(data_block):

            ind1 = [i for i in indlist[:2]] #H(z)
            ind2 = [i for i in indlist[2:4]] #w(z)
            ind3 = [i for i in indlist[4:6]] #D_L(z)
            ind4 = [i for i in indlist[6:]] #P(k,z)

            items1 = np.array(line.split())[ind1[0]:ind1[1]+1]
            items2 = np.array(line.split())[ind2[0]:ind2[1]+1]
            items3 = np.array(line.split())[ind3[0]:ind3[1]+1]
            items4 = np.array(line.split())[ind4[0]:ind4[1]+1]

            items = np.concatenate([items1,items2,items3,items4])

            for (col_count,col_name) in enumerate(col_names):

                value = items[col_count]
                data[col_name][line_count] = isfloat(value)
	
	return data,len(data_block)

    def lcdm_readData(self,filenumber,column_names,indlist):
        
	filename = self.directory + self.fileroot + '_%s.txt' % filenumber
        print(filename)

	f = open(filename)
        col_names = column_names

        data_block = f.readlines()
        f.close()

        data = {}

        for col_name in col_names:
            data[col_name] = np.zeros(len(data_block))

        for (line_count,line) in enumerate(data_block):

            ind1 = [0]+[1]+[i for i in indlist[:3]] #weight,likelihood // om,ol,h0
            ind2 = [i for i in indlist[3:5]] #H(z)
            ind3 = [i for i in indlist[5:]] #P(k,z)

            items1 = np.array(line.split())[ind1]
            items2 = np.array(line.split())[ind2[0]:ind2[1]+1]
            items3 = np.array(line.split())[ind3[0]:ind3[1]+1]

            items = np.concatenate([items1,items2,items3])

            for (col_count,col_name) in enumerate(col_names):
                value = items[col_count]
                data[col_name][line_count] = isfloat(value)

        return data,len(data_block)

    def nulcdm_readData(self,filenumber,column_names,indlist):

	filename = self.directory + self.fileroot + '_%s.txt' % filenumber
	print filename
	f = open(filename)
        col_names = column_names

        data_block = f.readlines()
        f.close()

        data = {}

        for col_name in col_names:
            data[col_name] = np.zeros(len(data_block))

        for (line_count,line) in enumerate(data_block):

            ind1 = [0]+[1]+[i for i in indlist[:4]] #weight,likelihood // om,ol,on,h0
            ind2 = [i for i in indlist[4:6]] #H(z)
            ind3 = [i for i in indlist[6:]] #P(k,z)

            items1 = np.array(line.split())[ind1]
            items2 = np.array(line.split())[ind2[0]:ind2[1]+1]
            items3 = np.array(line.split())[ind3[0]:ind3[1]+1]

            items = np.concatenate([items1,items2,items3])

            for (col_count,col_name) in enumerate(col_names):
                value = items[col_count]
                data[col_name][line_count] = isfloat(value)

        return data,len(data_block)

    def phant_readData(self,filenumber,column_names,indlist):

	filename = self.directory + self.fileroot + '_%s.txt' % filenumber
	print filename
	f = open(filename)
        col_names = column_names

        data_block = f.readlines()
        f.close()

        data = {}

        for col_name in col_names:
            data[col_name] = np.zeros(len(data_block))

        for (line_count,line) in enumerate(data_block):

            ind1 = [0]+[1]+[i for i in indlist[:4]] #weight,likelihood // om,ol,on,h0
            ind2 = [i for i in indlist[4:6]] #H(z)
            ind3 = [i for i in indlist[6:8]] #w(z)
            ind4 = [i for i in indlist[8:]] #P(k,z)

            items1 = np.array(line.split())[ind1]
            items2 = np.array(line.split())[ind2[0]:ind2[1]+1]
            items3 = np.array(line.split())[ind3[0]:ind3[1]+1]
            items4 = np.array(line.split())[ind4[0]:ind4[1]+1]

            items = np.concatenate([items1,items2,items3,items4])

            for (col_count,col_name) in enumerate(col_names):
                value = items[col_count]
                data[col_name][line_count] = isfloat(value)

        return data,len(data_block)

    def read_mnu(self,filenumber,column_names,indlist):

        filename = self.directory + self.fileroot + '_%s.txt' % filenumber
        print(filename)

        f = open(filename)
        col_names = column_names
     
        data_block = f.readlines()
        f.close()

        data = {}

        for col_name in col_names:
            data[col_name] = np.zeros(len(data_block))

        for (line_count,line) in enumerate(data_block):

            ind1 = [i for i in indlist[:4]] #weight,likelihood // om,ol,on,h0
            items = np.array(line.split())[ind1]

            for (col_count,col_name) in enumerate(col_names):

                value = items[col_count]
                data[col_name][line_count] = isfloat(value)

        return data,len(data_block)
    
    def read_mnu_s8(self,filenumber,column_names,indlist):

        filename = self.directory + self.fileroot + '_%s.txt' % filenumber
        print(filename)

        f = open(filename)
        col_names = column_names
     
        data_block = f.readlines()
        f.close()

        data = {}

        for col_name in col_names:
            data[col_name] = np.zeros(len(data_block))

        for (line_count,line) in enumerate(data_block):

            ind1 = [indlist[0]] #weight,likelihood // om,ol,on,h0
            ind2 = [indlist[1]] #weight,likelihood // om,ol,on,h0
            items1 = np.array(line.split())[ind1]
            items2 = np.array(line.split())[ind2]
            items = np.concatenate([items1,items2])

            for (col_count,col_name) in enumerate(col_names):

                value = items[col_count]
                data[col_name][line_count] = isfloat(value)

        return data,len(data_block)

def w(zs,w_spline): #dark energy equation of state
    w = np.zeros(zs.shape)
    large = zs > 3
    small = zs <= 3
    wsmall = -1 + w_spline(zs[small])
    w[large] = -1
    w[small] = wsmall
    return w

def n(zs,z_fornz,nz_tot,theory=False,bins='individual'):

    if theory == False:
        
	if bins == 'all_bins':
	    
	    if len(nz_tot) == 1:
		print 'wrong n_z!'
		sys.exit()

	    if zs > min(z_fornz) and zs < max(z_fornz):
	        n_arr = np.zeros(len(range(6)))
                for count,i in enumerate(range(6)):
    	            nz = nz_tot[i]
    	            nz = interp1d(z_fornz,nz)
    	            n_arr[count] = nz(zs)
                val = max(n_arr)
        
            else:
                val = np.zeros(zs.shape)
            
	    return val
	
	if bins == 'individual':
	    
	    """if len(nz_tot) != 1:
		print 'wrong n_z!'
		sys.exit()"""

	    if zs > min(z_fornz) and zs < max(z_fornz):
	        #nz_arr = nz_tot[0]
	        nz = interp1d(z_fornz,nz_tot)
	        val = nz(zs)
            
	    else:
                val = np.zeros(zs.shape)
	    
	    return val

    elif theory == True:
        z0 = 0.555
        alpha = 1.197
        beta = 1.193
        return (zs/z0)**alpha * np.exp(-(zs/z0)**beta)

def H(z,omegam,omeganu,omegal,H0): #hubble rate
    return H0 * np.sqrt((omegam+omeganu) * (1+z)**3 + omegal)

def chi(z,H_spline): #comoving distance
    
    zarr = np.linspace(0,z,300)
    dz = zarr[1]-zarr[0]
    
    return c * np.sum(dz * 1/H_spline(zarr))

def g(z,z_fornz,nz_tot,H_spline,theory,bins): #geometric lensing efficiency factor
    
    z2 = np.linspace(z,5.1,51)
    dz = z2[1]-z2[0]
    integrand = np.zeros(z2.shape)
    
    for count,i in enumerate(z2):
        integrand[count] = dz * n(i,z_fornz,nz_tot,theory,bins=bins) * (chi(i,H_spline) - chi(z,H_spline)) / chi(i,H_spline)
    
    return chi(z,H_spline) * np.sum(integrand)

def wind(z,z_fornz,nz_tot,H_spline,omegam,theory,bins): #window function
    return 1.5 * omegam * g(z,z_fornz,nz_tot,H_spline,theory,bins) * (1+z) * H_spline(0)**2

def weight(zs,z_fornz,nz_tot,H_spline,omegam,theory,bins): #lensing weight function
    
    func = np.zeros(zs.shape)
    
    for count,k in enumerate(zs):
	
	func[count] = wind(k,z_fornz,nz_tot,H_spline,omegam,theory,bins)**2 * chi(k,H_spline) / H_spline(k) * (1/c**3)
    
    return func

