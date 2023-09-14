# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 11:41:59 2021

@author: Noah Rubin
"""
#%% Package imports

import numpy as np
import matplotlib.pyplot as plt

#%% Definition of the Pauli matrices (with polarization convention) as a dictionary

pauli = {0: np.eye(2), 1: np.array([[1, 0], [0, -1]]), 2: np.array([[0, 1], [1, 0]]), 3: np.array([[0, -1j], [1j, 0]])}

#%% Function definitions

# turn a position on the Poincare sphere into a Stokes vector
def stokes_to_jones(stokes):
    """
    Parameters
    ----------
    stokes : ndarray
        Three-element Stokes state-of-polarization (SOP).

    Returns
    -------
    ndarray
        The corresponding Jones vector (2x1)

    """
    
    stokes = stokes/np.linalg.norm(stokes) # normalize
    mat  = stokes[0]*pauli[1] + stokes[1]*pauli[2] + stokes[2]*pauli[3] # compute matrix
    w, v = np.linalg.eig(mat) # compute its eigenbasis
    
    # find +1 eigenvector
    if w[0] > w[1]:
        return v[:, 0]
    else:
        return v[:, 1]
    
# turn a Jones vector into a position on the Poincare sphere
def jones_to_stokes(jones):
    """

    Parameters
    ----------
    jones : ndarray
        A 2x1 Jones vector to-be-converted.

    Returns
    -------
    ndarray
        The corresponding Stokes 3-vector, its position on the Poincare sphere

    """
    
    jones = jones/np.linalg.norm(jones) # normalize
    
    # Each Stokes component is a Pauli matrices sandwiched between the Jones vector <j|sigma|j>
    stokes1 = np.dot(np.conj(jones), jones@pauli[1])
    stokes2 = np.dot(np.conj(jones), jones@pauli[2])
    stokes3 = np.dot(np.conj(jones), jones@pauli[3])
    return np.real(np.array([stokes1, stokes2, stokes3]))

# turn a Jones matrix transformation into a Mueller matrix
def jones_mueller(J):
    """

    Parameters
    ----------
    J : ndarray
        2x2 Jones matrix to be converted.

    Returns
    -------
    mueller : ndarray
        4x4 (non-depolarizing) Mueller matrix corresponding to J.

    """
    mueller = np.zeros((4,4))
    
    for i in range(4):
        for j in range(4):
            mueller[i,j] = np.real(0.5*np.trace( np.dot(J, np.dot( pauli[j] , np.dot(np.transpose(np.conjugate(J)),pauli[i])   ))))
    
    return mueller

# get useful information out of an order's Jones matrix transformation
# such as the Jones matrix of a diffraction order that you simulate
def analyze_Jones_matrix(J):
    """
    Parameters
    ----------
    J : ndarray
        2x2 Jones matrix transformation.

    Returns
    -------
    efficiency : float
        Average intensity output averaged over all possible input polarization states, i.e., for unpolarized light.
    diattenuation : float
        Contrast in output intensity for max vs min, bounded in [0,1].
    analyzer_state : ndarray
        Position on Poincare sphere of polarization state being analyzed.

    """
    # get the Mueller matrix
    mueller = jones_mueller(J)
    # pull out the first row which dictates diattenuating behavior
    first_row = mueller[0, :]
    diattenuation = np.sqrt(np.sum(first_row[1:]**2)) / first_row[0]
    efficiency = first_row[0]
    analyzer_state = first_row[1:]/np.linalg.norm(first_row[1:])
    
    return efficiency, diattenuation, analyzer_state

# generate the Jones matrix of an analyzer with any average efficiency, diattenuation, and analyzer state
# mostly for testing purposes
def generate_analyzer_Jones_matrix(efficiency, diattenuation, stokes_state):
    """
    Parameters
    ----------
    efficiency : float
        Average efficiency of transformation for all polarizations, i.e., what would be observed for unpolarized light.
    diattenuation : float
        Normalized difference between max and min possible output states.
    stokes_state : ndarray
        Stokes three-vector (position on Poicnare sphere) of analyzed polarization state.

    Returns
    -------
    analyzer_Jones : ndarray
        Jones vector of analyzer with given parameters.

    """
    
    # generate the orthogonal pair of Jones vectors corresponding to the Stokes vector of the analyzer
    jones_vector = stokes_to_jones(stokes_state)
    jones_vector_perp = stokes_to_jones(-stokes_state)
    
    # convert the efficiency and diattenuation to common and differential loss using arctanh
    # trouble emerges here if you put in exactly 1.0 for diattenuation
    diff_loss = np.arctanh(diattenuation)
    
    # compute analyzer Jones matrix
    analyzer_Jones = (np.exp(diff_loss/2) * np.outer(jones_vector, np.conj(jones_vector)) + np.exp(-diff_loss/2) * np.outer(jones_vector_perp, np.conj(jones_vector_perp)))
    
    # compute efficiency of current matrix
    eff = 0.5 * np.trace(analyzer_Jones@analyzer_Jones)
    # then normalize matrix by desired efficiency to get correct matrix
    # kind of a hack, not the fully correct way
    # prevents us from dealing with full complexity of this problem while getting right answer
    # better way to do this would be to write a function that makes Mueller matrix of diattenuator
    # then convert it to Jones. But this is a little messy. Can do if needed.
    analyzer_Jones = np.sqrt(efficiency/eff) * analyzer_Jones

    return analyzer_Jones
    
# plot a polariation ellipse
def plot_polarization_ellipse(ax, jones_vector, sampling_points=100, rad=1):
    """
    Parameters
    ----------
    ax: axis to plot on
    jones_vector : ndarray
        2x1 Jones vector of the state to plot.
    sampling_points : int, optional
        Number of points on the ellipse to plot. The default is 100.
    rad : float, optional
        Scaling factor of ellipse size. The default is 1.

    Returns
    -------
    None.

    """
    # operating in Stokes formalism helps remove some discontinuities of trig functions
    stokes_state = jones_to_stokes(jones_vector)
    # extract parameters of ellipse, from definition of Jones vector: [cos(chi); sin(chi) * e^{i * phi}]
    chi, phi = 1/2 * np.arccos(stokes_state[0]), np.arctan2(stokes_state[2], stokes_state[1])
    
    # prepare parametric function of polarization ellipse
    t = np.linspace(0, 2*np.pi, sampling_points)
    x = rad * np.cos(chi) * np.cos(t)
    y = rad * np.sin(chi) * np.cos(t + phi)
    
    # plot the ellipse - red if it is left-handed, blue if it is right-handed
    if stokes_state[2]<0:
        ax.plot(x, y, 'r')
    else:
        ax.plot(x, y, 'b')
        
    # make the plot more aesthetic
    ax.axis('off')
    ax.set_aspect('equal', 'box')
