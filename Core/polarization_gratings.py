# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 10:44:03 2020

@author: Noah A. Rubin
"""

'''   IMPORTING PACKAGES   '''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import minimize
from scipy.io import loadmat
import gdspy

def load_simulation_library(file_name):
    
    """
    This method is used to load simulation data to be used in generating CAD.
    
    Parameters:
        file_name (string): pointer to file with properly formatted .mat file containing library
    
    Returns:
        data (dict): dictionary containing arrays of every element in the library.
            keys: dxs (x-dimensions), dys (y-dimensions), phiXs (x-phase shifts), phiYs (y-phase shifts), txs (x-polarized transmission amplitudes), tys (y-polarized transmission amplitudes) 
    
    """
    
    data = loadmat(file_name)
    data['phiXs'] = data['phiXs'].squeeze()
    data['phiYs'] = data['phiYs'].squeeze()
    data['txs'] = data['txs'].squeeze()
    data['tys'] = data['tys'].squeeze()
    data['dxs'] = data['dxs'].squeeze()
    data['dys'] = data['dys'].squeeze()
    
    
    return data

class GratingUnitCell(object):
    
    """
    This is a generic class for gratings in which a Jones matrix is defined at each position, stored in a 'jcube'.
    This class implements methods that are relevant to any Jones matrix grating, such as the Fourier transform.
    
    This class and its methods assume that the grating, regardless of its unit cell geometry, is sampled on a square lattice whose sampling period is identical in the x and y directions (d_a = d_b).
    
    Attributes:
        Nx (int): Number of sampling locations along the x direction.
        Ny (int): Number of sampling locations along the y direction.
        angle (float): Angle between two grating vectors.
        jcube (Ny x Nx x 4 array of floats): Stores the Jones matrix at each position.
        XX, YY: meshgrids of lattice coordinates on the grid
        
    """
    
    def __init__(self,Nx,Ny):
        
        """
        The constructor for the GratingUnitCell class.
        
        Parameters:
            Nx (int): Number of sampling locations along the x direction.
            Ny (int): Number of sampling locations along the y direction.
            angle (float): Angle between two grating vectors. Default value is pi/2.
        
        """
        
        self.Nx=Nx
        self.Ny=Ny
        self.jcube=np.zeros([Ny,Nx,2,2])

        # create coordinate grid for general use
        self.XX, self.YY = np.meshgrid(np.arange(Nx), np.arange(Ny))
        #self.XX+=1 # FT computed based on indexing starting at 1
        #self.YY+=1
           
    def calculate_Jmn (self,m,n):
        
        """   
        Finds the Jones matrix associated with order (m,n) by Fourier transform.
        
        This method is best used for a one-time calculation of the weight of a particular order.
        
        Parameters:
            m (int): First index of diffraction order.
            n (int): Second index of diffraction order.
            
        Returns:
            output_matrix (2x2 float array): (m,n)th Jones matrix coefficient of the matrix Fourier series
        
        """
        
        output_matrix = np.zeros((2,2), dtype=np.complex)
        
        # calculate weights from analytical Fourier transform
        weights = self.calculate_weight_matrix(m, n)
        # make it the right shape
        weights4d = np.zeros([self.Ny, self.Nx, 2, 2], dtype=np.complex)
        weights4d[:, :, 0, 0] = weights
        weights4d[:, :, 1, 0] = weights
        weights4d[:, :, 0, 1] = weights
        weights4d[:, :, 1, 1] = weights
        
        # add the individual contribution of the jcube at each site
        output_matrix = np.sum(weights4d*self.jcube, axis=(0,1))
        
        return output_matrix
    
    
    def fourier_transform(self, a, b, m, n):
        
        """
        Finds the Fourier coefficient to multiply each Jones matrix in the lattice, later summed to find the Jones matrix Fourier coefficient.
        
        Compatible with vectorized input and output.
        
        Expression is given by J_{mn} = sum_{a, b} 1/(AB) * sinc(pi*m/A) * sinc(pi*n/B) * exp(-i*pi*(2*a*m/A + 2*b*n/B))
        
        Parameters:
            a (int): x (column) coordinate of lattice point under consideration
            b (int): y (row) coordinate of lattice point under consideration
            m (int): first index of diffraction order of interest
            n (int): second index of diffraction order of interest
            
        Returns:
            fourier (float): multiplier in computation of Fourier series
            
        """
        
        A = self.Nx
        B = self.Ny
        
        # We assume here that d_a = d_b, i.e., the lattice spacing between pillars is the same in x and y directions
        exp_term = np.exp(-1j*np.pi*((2*a)*m/A + (2*b)*n/B))
        first_sinc = np.sinc(np.pi*m/A * 1/np.pi) # need to divide by pi for np's sinc definition
        second_sinc = np.sinc(np.pi*n/B * 1/np.pi) # need to divide by pi for np's sinc definition
        fourier = (1/(A*B) * exp_term * first_sinc * second_sinc)
        
        return fourier
    
    def calculate_weight_matrix (self,m,n):
        
        """   
        Finds the weight matrix associated with order (m, n) computing the aggregated Fourier coefficient.
        
        This method is best used for a one-time calculation of the weight of a particular order whose Fourier coefficient will need to be found repeatedly.
        
        Parameters:
            m (int): First index of diffraction order.
            n (int): Second index of diffraction order.
            
        Returns:
            weight_matrix (Ny x Nx float array): matrix of weights from a vectorized computation of the analytical Fourier expression
        
        """
        
        # vectorize the computation of the weights
        weight_matrix = self.fourier_transform(self.XX, self.YY, m, n)
        
        return weight_matrix

class MetasurfaceGratingUnitCell(GratingUnitCell):
    
    """
    This class extends GratingUnitCell with methods and assumptions that apply to the design of one-layer metasurface polarization gratings.
    
    Attributes:
        Nx (int): Number of sampling locations along the x direction.
        Ny (int): Number of sampling locations along the y direction.
        angle (float): Angle between two grating vectors.
        phi_x (Ny x Nx float array): Array of phi_x phase angles.
        phi_y (Ny x Nx float array): Array of phi_y phase angles.
        theta (Ny x Nx float array): Array of theta phase angles.
        history (dict): History of grating parameters so that optimization progress can be recorded and analyzed later
    
    """
    
    def __init__(self, Nx, Ny, angle=np.pi/2, phi_x=None, phi_y=None, theta=None):
        
        """   
        The constructor for the MetasurfaceGratingUnitCell class.
        
        Parameters:
            Nx (int): Number of sampling locations along the x direction.
            Ny (int): Number of sampling locations along the y direction.
            angle (float): Angle between two grating vectors.
            phi_x ()
            
        Returns:
            weight_matrix (Ny x Nx float array): matrix of weights from a vectorized computation of the analytical Fourier expression
        
        """
        
        
        super(MetasurfaceGratingUnitCell, self).__init__(Nx,Ny)
        
        # assign default values to the angle arrays
        if phi_x is None:
            self.phi_x = 2*np.pi*np.random.rand(Ny, Nx)
        if phi_y is None:
            self.phi_y = 2*np.pi*np.random.rand(Ny, Nx)
        if theta is None:
            self.theta = 2*np.pi*np.random.rand(Ny, Nx)
        
        # populate the j_cube based on design of metasurface
        self.jcube = self.find_metasurface_jcube(self.phi_x, self.phi_y, self.theta)
        
        # create a container for optimization history
        self.history = {0: self.pack(self.phi_x, self.phi_y, self.theta)}
    
    def rotation_matrix(self, rot_angle):
        
        """
        Convenience function to generate a 2x2 rotation matrix.

        Parameters
        ----------
        rot_angle : float
            Rotation angle in [radians].

        Returns
        -------
        mat : ndarray
            2x2 Jones rotation matrix.

        """
        
        mat = np.array([[np.cos(rot_angle), -np.sin(rot_angle)], [np.sin(rot_angle), np.cos(rot_angle)]])
        return mat
    
    def make_matrix_array(self, arr):
        
        """   
        Function to turn (2, 2, Ny, Nx) array into (Ny, Nx, 2, 2) array for use with matrix multiplication. This is the preferred format for matrix multiplication. 
        
        Parameters:
            Nx (int): Number of sampling locations along the x direction.
            Ny (int): Number of sampling locations along the y direction.
            angle (float): Angle between two grating vectors.
            phi_x ()
            
        Returns:
            weight_matrix (Ny x Nx float array): matrix of weights from a vectorized computation of the analytical Fourier expression
        
        """
        
        return np.moveaxis(arr, [-2, -1], [0, 1])
        
    def find_metasurface_jcube(self, phi_x, phi_y, theta):
        
        """   
        Compute the metasurface's (Ny, Nx, 2, 2) jcube (array of Jones matrices) from its angular parameters
        
        Parameters:
            phi_x (Ny x Nx): array of phi_x angles
            phi_y (Ny x Nx): array of phi_y angles
            theta (Ny x Nx): array of physical rotation angles
            
        Returns:
            out (Ny, Nx, 2, 2): jcube from angular parameters
        
        """
        
        rot1 = self.make_matrix_array(self.rotation_matrix(-theta))
        rot2 = self.make_matrix_array(self.rotation_matrix(theta))
        diag = self.make_matrix_array(np.array([[np.exp(1j*phi_x), np.zeros([self.Ny, self.Nx])], [np.zeros([self.Ny, self.Nx]), np.exp(1j*phi_y)]]))
        
        out = rot1@diag@rot2
        
        return out
    
    def update_metasurface_parameters(self, phi_x, phi_y, theta):
        
        """   
        Update instance variables of the metasurface and recalculate the jcube (array of Jones matrices)
        
        Parameters:
            phi_x (Ny x Nx): array of phi_x angles
            phi_y (Ny x Nx): array of phi_y angles
            theta (Ny x Nx): array of physical rotation angles
            
        Returns:
        
        """
        
        self.phi_x = phi_x
        self.phi_y = phi_y
        self.theta = theta
        self.jcube = self.find_metasurface_jcube(phi_x, phi_y, theta)
        
    
    def pack(self, phi_x, phi_y, theta):
        
        """
        Take the 3 angle matrices and pack into a 1D array of length Ny*Nx*3 for optimization.
        
        Parameters:
            
        Returns:
            params1d (Ny*Nx*3, 1): 1D list of the angular parameters
        
        """
        
        dim = self.Ny*self.Nx
        phi_x1d = np.reshape(phi_x, dim)
        phi_y1d = np.reshape(phi_y, dim)
        theta1d = np.reshape(theta, dim)
        
        params1d = np.concatenate((phi_x1d, phi_y1d, theta1d))
        
        return params1d        
    
    
   
    def unpack (self, params1d): 
        
        """
        Take a packed version of the 3 angle matrices and unpack them for further use.
        
        Parameters:
            params1d (Ny*Nx*3, 1): 1D list of the angular parameters
            
        Returns:
            out_dict: dictionary with each arrays as a key
        
        """
        
        dim = self.Nx*self.Ny
        phi_x = np.reshape(params1d[:dim], [self.Ny, self.Nx])
        phi_y = np.reshape(params1d[dim:-dim], [self.Ny, self.Nx])
        theta = np.reshape(params1d[-dim:], [self.Ny, self.Nx])
        
        out_dict = {"theta": theta, "phi_x": phi_x, "phi_y": phi_y}
        
        return out_dict
    
    def optimize(self, merit_func, cons, cons_type, tolerance=0.005, negate=True):

        """
        Optimization function with custom defined merit functions and constraints
        
        Parameters:
            merit_func (lambda): merit function that accepts 1D list of parameters
            cons (list of lambdas): list of constraint functions
            cons_type (list of strings): what type of constraint each constraint function represents (e.g., 'eq' for equality)
            tolerance (float): tolerance for convergence of the residual of the optimiztaion algorithm
            negate (bool): if True, the displayed value of the merit function is multiplied by -1, as would be necessary for trying to maximize that merit function by minimization
            
        Returns:
            final_val (float): final value of the merit function
        
        """
        
        # iteration counter for optimization
        iter_number = 1
        
        init = self.pack(self.phi_x, self.phi_y, self.theta)
        if (not self.history):
            # evaluates to true if the dict is empty, i.e., we haven't optimized before
            self.history[0] = init
            iter_offset = 0
        else:
            iter_offset = max(list(self.history.keys())) # get maximum past iteration and append to the end of that
            
        # define the callback function to be called during optimization
        def status_func(state_vec):
            nonlocal iter_number
            # display status to console
            if negate:
                curr = -merit_func(state_vec)
            else:
                curr = merit_func(state_vec)
            print("Iteration: " , iter_number, " - Merit =", curr)
            # append to optimization history of the object
            self.history[iter_number+iter_offset] = state_vec
            iter_number += 1 # increment the iteration counter
        
        constraint_list = self.convert_constraints(cons, cons_type)
        
        opt_param = minimize(merit_func, init, constraints=constraint_list, tol=tolerance, options ={'disp': True,'maxiter': 10000}, callback = status_func)        
        opt_dictionary = self.unpack(opt_param.x)
        self.update_metasurface_parameters(opt_dictionary['phi_x'], opt_dictionary['phi_y'],opt_dictionary['theta'])
        
        final_val = opt_param.fun
        return final_val
    
    def convert_constraints(self, cons, cons_type):
        
        """
        Convert constraints into form mandated by minimize function.
        
        Parameters:
            cons (lambda list): merit function that accepts 1D list of parameters
            cons_type (list of strings): type of each constraint, 'eq' for equality and 'ineq' for inequality, of the same length as cons
            
        Returns:
            formatted_constraints (array of dicts): constraints formatted for minimize
        
        """
        # check if there are as many constraints as constraint types
        if len(cons)!=len(cons_type):
            raise ValueError('Constraint list and constraint type list are of unequal length.')
        
        # parse through the constraints and assemble the dicts mandated by minimize function
        formatted_constraints = []
        for constraint, constraint_type, i in zip(cons, cons_type, np.arange(len(cons))):
            if constraint_type=='eq':
                formatted_constraints.append({"type": "eq", "fun": constraint})
            elif constraint_type=='ineq':
                formatted_constraints.append({"type": "ineq", "fun": constraint})
            else:
                raise ValueError('Invalid constraint type.')
                
        return formatted_constraints
    
    def compute_weight_matrices(self, orders):
        
        """
        Compute spatially-varying Fourier weights for a specific set of orders, so that this only has to be done once.
        
        Parameters:
            orders (list of tuples): orders of interest
            
        Returns:
            weight_matrices (list of Nx x Ny arrays): matrices with spatially-varying Fourier weights
        
        """
        
        weight_matrices = np.empty([self.Ny, self.Nx, len(orders)], dtype = np.complex)
        
        for order, i in zip(orders, np.arange(len(orders))):
            m, n = order[0], order[1]
            weight_matrix = self.calculate_weight_matrix(m, n)
            weight_matrices[:, :, i] = weight_matrix
            
        return weight_matrices
    
    def calculate_Jmn_fast(self,m,n,weights,x):
        
        """   
        Finds the Jones matrix associated with order (m,n) by Fourier transform.
        
        This method is faster than calculate_Jmn because the weight matrix is handed to the function rather than calculated in-line.
        
        This is useful when the coefficient of a particular order will be calculated many times as the jcube changes.
        
        Parameters:
            m (int): First index of diffraction order.
            n (int): Second index of diffraction order.
            weights (Ny x Nx): matrix of weights on J_(a,b) at each lattice position. Calculated by, e.g., calculate_weight_matrix
            x (Ny x Ny x 3 list of floats): parameters of current grating, to be unpacked
            
        Returns:
            output_matrix (2x2 float array): (m,n)th Jones matrix coefficient of the matrix Fourier series
        
        """
        
        out_dict = self.unpack(x)
        phi_x, phi_y, theta = out_dict['phi_x'], out_dict['phi_y'], out_dict['theta']
        
        jcube = self.find_metasurface_jcube(phi_x, phi_y, theta)
        
        # make it the right shape, duplicate matrix suitably for multiplication
        weights4d = np.zeros([self.Ny, self.Nx, 2, 2], dtype = np.complex)
        weights4d[:, :, 0, 0] = weights
        weights4d[:, :, 1, 0] = weights
        weights4d[:, :, 0, 1] = weights
        weights4d[:, :, 1, 1] = weights
        
        # add the individual contribution of the jcube at each site
        output_matrix = np.sum(weights4d*jcube, axis=(0,1))
        
        return output_matrix
        
        
    def power_in_orders_fast(self, orders, weight_matrices, x):
        
        """
        Computation of output power in a set of orders. This can be useful for defining an optimization merit function.
        
        This function is made fast by using weight_matrices that have been previously calculated, saving a step of calculation.
        
        Parameters:
            orders (list of tuples): orders of interest
            weight_matrices (list of Nx x Ny arrays): matrices with spatially-varying Fourier weights, one for each order of interest, previously calculated by computer_weight_matrices
            x (Ny x Ny x 3 list of floats): parameters of current grating, to be unpacked
            
        Returns:
            power (list of floats): amount of power in each order
        
        """
        
        power = np.zeros(len(orders))
        
        for order, i in zip(orders, np.arange(len(orders))):
            m, n = order[0], order[1]
            weight_matrix = weight_matrices[:, :, i]
            order_matrix = self.calculate_Jmn_fast(m, n, weight_matrix, x)
            power[i] = self.matrix_power(order_matrix)
            
        return power
    
    def order_contrasts_fast(self, orders, polarizations, weight_matrices, x):
        
        """
        Computation of diattentuation of a given set of orders. This can be useful for defining optimization constraints.
        
        This function is made fast by using weight_matrices that have been previously calculated, saving a step of calculation.
        
        Parameters:
            orders (list of tuples): orders of interest
            polarizations (list of Jones vectors): list of Jones vectors to analyze for , the preferred polarizations.
            weight_matrices (list of Nx x Ny arrays): matrices with spatially-varying Fourier weights, one for each order of interest, previously calculated by computer_weight_matrices
            x (Ny x Ny x 3 list of floats): parameters of current grating, to be unpacked
            
        Returns:
            constrast (list of floats): list of each order's contrast with respect to the desired polarization state
        
        """
        
        contrast = np.zeros(len(orders))
        
        for order, polarization, i in zip(orders, polarizations, np.arange(len(orders))):
            m, n = order[0], order[1]
            weight_matrix = weight_matrices[:, :, i]
            order_matrix = self.calculate_Jmn_fast(m, n, weight_matrix, x)
            I = np.linalg.norm(order_matrix@polarization)
            perp = np.array([-polarization[1]/np.exp(1j*np.angle(polarization[1])), polarization[0]*np.exp(1j*np.angle(polarization[1]))])
            Iperp = np.linalg.norm(order_matrix@perp)
            contrast[i] = np.real((I-Iperp)/(I+Iperp))
            
        return contrast
    
    
    def matrix_power(self, mat):
        
        """
        Convert power expectation value of a Jones matrix.
        
        Parameters:
            mat (2 x 2 array): Jones matrix
            
        Returns:
            power (float): power expectation value for all possible input polarization states
        
        """
        
        power = 0.5 * np.trace(np.conj(mat.T)@mat)
        return np.abs(power)
    
    def total_matrix(self, orders):
        
        """
        Calculate total Jones matrix in a set of diffraction orders.
        
        Parameters:
            orders (list of tuples): list of diffraction orders
            
        Returns:
            mat (2 x 2 array): output Jones matrix
        
        """
        
        mat = np.zeros([2, 2], dtype = np.complex)
        for order in orders:
            jones = self.calculate_Jmn(order[0], order[1])
            mat+=np.conj(jones.T)@jones
        
        return mat
    
    def get_unitcell_design(self, library_dict, min_size=None):
        
        """
        Function to parse a library and find best-fit grating unitcell design.

        Parameters
        ----------
        library_dict : dictionary
            A library loaded from a .mat data file. The library contains arrays of different pillar widths (dxs), x phases (phiXs), heights (dys), y phases (phiYs), and transmission along x and y (txs and tys) at a given wavelength.
        min_size : float, optional
            If specified, the minimum size feature in [nm] that should be used in the design. The default is None.

        Returns
        -------
        design : dictionary
            The computed best-fit design as a dictionary with fields x_diams, y_diams, and thetas.

        """
        
        design = {"theta":self.theta, "x_diams": np.zeros([self.Ny, self.Nx]), "y_diams": np.zeros([self.Ny, self.Ny])}
        if not min_size == None:
            min_size = min_size*1e-9
        
        phiX, phiY, tx, ty = library_dict["phiXs"], library_dict["phiYs"], library_dict["txs"], library_dict["tys"]
        
        # expand the dimensions of phase arrays to permit array broadcasting
        phiXd, phiYd = np.expand_dims(self.phi_x, axis=2), np.expand_dims(self.phi_y, axis=2)
        
        # 3d array with merit function computed at each lattice position, (Ny, Nx, len(phiX))
        merit_array = np.abs(np.exp(1j*phiXd)-np.ones([self.Ny, self.Nx, 1])*tx*np.exp(1j*phiX))**2 + np.abs(np.exp(1j*phiYd)-np.ones([self.Ny, self.Nx, 1])*ty*np.exp(1j*phiY))**2
        chosen_index = np.argmin(merit_array, axis=2)
        
        design["x_diams"] = library_dict["dxs"][chosen_index]
        design["y_diams"] = library_dict["dys"][chosen_index]
        
        if min_size == None:
            return design
        else:
            truth_array = np.logical_and(design["x_diams"]>min_size, design["y_diams"]>min_size)
            
            for index, val in np.ndenumerate(truth_array):
                if val:
                    pass # size is above the minimum size here
                else:
                    # find the indices sorted suitability
                    listing = np.argsort(merit_array[index])
                    # find the dimensions sorted by suitability
                    x_dimensions = library_dict["dxs"][listing]
                    y_dimensions = library_dict["dys"][listing]
                    # find the first place where the dimensions meet the minimum size
                    loc = np.where(np.logical_and(x_dimensions>min_size, y_dimensions>min_size))[0][0]
                    # choose those dimensions for the design
                    design["x_diams"][index] = x_dimensions[loc]
                    design["y_diams"][index] = y_dimensions[loc]

            return design
    
    def get_unitcell_design_offset(self, library_dict, min_size=None, max_size=None, N_offset=1000):
        
        """
        Function to compute best-fit unitcell design while allowing overall phase to vary to produce best-fit.

        Parameters
        library_dict : dictionary
            A library loaded from a .mat data file. The library contains arrays of different pillar widths (dxs), x phases (phiXs), heights (dys), y phases (phiYs), and transmission along x and y (txs and tys) at a given wavelength.
        min_size : float, optional
            If specified, the minimum size feature in [nm] that should be used in the design. The default is None.
        max_size : float, optional
            If specified, the maximum size feature in [nm] that should be used in the design. The default is None.
        N_offset : int, optional
            Number of global phase offsets oh phiX and phiY in the design. The best design is chosen from all these offsets. The default is 1000.

        Returns
        -------
        design : dictionary
            The computed best-fit design as a dictionary with fields x_diams, y_diams, and thetas.

        """
        
        offsets=np.linspace(0,2*np.pi,N_offset)
        designs=dict()
        merits=np.zeros((N_offset,1))
        
        for ii in range(0,N_offset):
            
            design = {"theta": self.theta, "x_diams": np.zeros([self.Ny, self.Nx]), "y_diams": np.zeros([self.Ny, self.Ny])}
            
            phiX, phiY, tx, ty = library_dict["phiXs"], library_dict["phiYs"], library_dict["txs"], library_dict["tys"]
            
            dxs,dys=library_dict["dxs"],library_dict["dys"];
            
            # remove library elements that do not fulfill the minsize or max size condition
            
            # if there is a min_size
            if min_size is not None:
                # ...and also a max_size
                if max_size is not None:
                    condition = np.logical_and(np.logical_and(dxs>min_size*1e-9, dys>min_size*1e-9), np.logical_and(dxs<max_size*1e-9, dys<max_size*1e-9))
                    
                # just a min_size
                else:
                    condition = np.logical_and(dxs>min_size*1e-9, dys>min_size*1e-9)
            # if there is no min_size
            else:
                # ...but there is a max_size
                if max_size is not None:
                    condition = np.logical_and(dxs<max_size*1e-9, dys<max_size*1e-9)
                    
                # if there is neither a minimum or maximum specified
                else:
                    condition = np.logical_and(np.logical_and(dxs>0, dys>0), np.logical_and(dxs<np.inf, dys<np.inf))
            
            # find locations where condition holds true and filter library                   
            locator = np.where(condition)[0]
            phiX = phiX[locator]
            phiY = phiY[locator]
            tx = tx[locator]
            ty = ty[locator]
            dxs = dxs[locator]
            dys = dys[locator]
            
            # expand the dimensions of phase arrays to permit array broadcasting
            phiXd, phiYd = np.expand_dims(self.phi_x+offsets[ii], axis=2), np.expand_dims(self.phi_y+offsets[ii], axis=2)
            
            # 3d array with merit function computed at each lattice position, (Ny, Nx, len(phiX))
            merit_array = np.abs(np.exp(1j*phiXd)-np.ones([self.Ny, self.Nx, 1])*tx*np.exp(1j*phiX))**2 + np.abs(np.exp(1j*phiYd)-np.ones([self.Ny, self.Nx, 1])*ty*np.exp(1j*phiY))**2
            chosen_index = np.argmin(merit_array, axis=2)
            merit = np.sum(np.amin(merit_array, axis=2))
        
            design["x_diams"] = dxs[chosen_index]
            design["y_diams"] = dys[chosen_index]
            
            designs[ii]=design
            merits[ii]=merit
            
        meritpos=np.argmin(merits)
        return designs[meritpos]
    
    
    def get_cad (self, lx, ly, file_name, unit_cell_design, sep, size_offset=0):
        
        """
        Generate the CAD as a .gds file from a computed design.

        Parameters
        ----------
        lx : float
            Width of the final grating in [microns].
        ly : float
            Height of the final grating in [microns].
        file_name : str
            Output file name. Should end in '.gds' extension.
        unit_cell_design : dictionary
            A unit cell design generated by get_unitcell_design or get_unitcell_design_offset.
        sep : float
            Inter-element separation between structures in [microns].
        size_offset : float, optional
            If specified, a global size offset in [nm] subtracted from all feature sizes. Can be positive (features smaller) or negative (features larger). The default is 0.

        Returns
        -------
        None.

        """
        
        nx = np.int(  lx // (self.Nx*sep)  )
        ny = np.int(  ly // (self.Ny*sep)   )
        dimensions = unit_cell_design
        lx_mod = nx*self.Nx*sep
        ly_mod = ny*self.Ny*sep
        unitcell_name = 'single_unit_cell'
        unitcell = gdspy.Cell(unitcell_name)        
        
        # create unit cell
        for x in range(self.Nx):
            for y in range(self.Ny):
                dx = 1e6*dimensions["x_diams"][y,x] - 0.001*size_offset
                dy = 1e6*dimensions["y_diams"][y,x] - 0.001*size_offset
                theta = dimensions["theta"][y,x]
                rect = gdspy.Rectangle((0, 0), (dx , dy))
                rect.rotate( theta , center = (dx/2 , dy/2 ))
                rect.translate( (x+0.5)*sep - dx/2 , (y+0.5)*sep - dy/2 )
                unitcell.add(rect)
        
        # create metasurface        
        ref_cell = gdspy.Cell('whole_metasurface')
        cell_array = gdspy.CellArray(unitcell, nx, ny, (sep*self.Nx, sep*self.Ny), (-lx_mod/2, -ly_mod/2), magnification=1)
        ref_cell.add(cell_array)

        try:
            gdspy.write_gds(file_name, unit=1.0e-6, precision=1.0e-9)
            gdspy.Cell.cell_dict = {}

        except Exception as e:

            gdspy.Cell.cell_dict = {}
            print(e)
        gdspy.current_library = gdspy.GdsLibrary()
            
    def plot_design(self, design, sep, size_offset=0):
        
        """
        Convenience method to plot grating unit cell design without needing to generate or open a .gds file.

        Parameters
        ----------
        design : dictionary
            A design generated by the get_unitcell_design method.
        sep : float
            Inter-pillar separation, in [microns].
        size_offset : float, optional
            Size offset of all features in [nm]. Positive values mean smaller features. The default is 0.

        Returns
        -------
        fig : figure
            Output figure reference.
        ax : axis
            Output axis reference..

        """
        
        fig, ax = plt.subplots(1,1)
        for x in range(self.Nx):
            for y in range(self.Ny):
                dx = 1e6*design["x_diams"][y,x] - 0.001*size_offset
                dy = 1e6*design["y_diams"][y,x] - 0.001*size_offset
                theta = design["theta"][y,x]
                xy = ((x+0.5)*sep - dx/2 , (y+0.5)*sep - dy/2)
                rect = Rectangle(xy, dx , dy, angle=np.degrees(theta), rotation_point='center')
                ax.add_patch(rect)
                
        ax.set_xlim(0, self.Nx*sep)
        ax.set_ylim(0, self.Ny*sep)
        ax.set_aspect('equal')
        ax.set_xlabel('x (μm)')
        ax.set_ylabel('y (μm)')
        ax.set_title('Grating design')
        ax.grid(True)
        ax.set_axisbelow(True) # put gridlines behind rectangles
                
        return fig, ax
    
    def save(self, name):
        
        """
        
        Save the current grating design and its optimization history.        

        Parameters
        ----------
        name : str
            Filename for the .npz file to-be-saved. If name does not end in .npz, .npz will be appended.

        Returns
        -------
        None.

        """
        
        # check if .npz extension was specified; add it if not
        if name[-4:] != '.npz':
            name = name + '.npz'
        
        # save one dictionary for the state variables
        np.savez(name, phi_x=self.phi_x, phi_y=self.phi_y, theta=self.theta)
        # save a second dictionary for the optimization history, if any
        # recall that in savez all keys must be strings, so we must convert the history dictionary first
        mod_history = {}
        for key, val in self.history.items():
            mod_history[str(key)] = val
        np.savez(name[:-4]+'_history.npz', **mod_history)
        
    def load(self, name):
        
        """
        Load a previous grating design and its history.
        
        History is reinstated so that optimization can continue.

        Parameters
        ----------
        name : str
            Filename of .npz file. If name does not end in .npz, .npz will be appended.

        Returns
        -------
        None.

        """
        
        # check if .npz extension was specified; add it if not
        if name[-4:] == '.npz':
            file = np.load(name)
            loaded_history = np.load(name[:-4]+'_history.npz')
        else:
            file = np.load(name+'.npz')
            loaded_history = np.load(name+'_history.npz')
        self.phi_x, self.phi_y, self.theta = file['phi_x'], file['phi_y'], file['theta']
        self.jcube = self.find_metasurface_jcube(self.phi_x, self.phi_y, self.theta)
        
        # need to convert back to integer keys
        converted = {}
        for key, val in loaded_history.items():
            converted[int(key)] = val 
        self.history = converted
