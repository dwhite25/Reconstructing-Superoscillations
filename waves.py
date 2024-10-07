import os
import csv
import math
import scipy
import joblib
import numpy as np
import sympy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal as ss
from numpy.fft import fft, ifft, fftfreq, fftshift, ifftshift

# classic sinc function
def sinc(x, k, **kwargs):
    r = x - k
    omega = 2*np.pi
    if 'omega' in kwargs:
        omega = kwargs['omega']
    power = 1
    if 'power' in kwargs:
        power = kwargs['power']
    y = 1
    if r != 0:
        y = (sp.sin(omega*r)/(omega*r))**power
    return y

# cosine function
def cosine(x, k, **kwargs):
    r = x - k
    y = sp.cos(sp.pi*r)
    return y

# unnormalized gaussian
def gaussian(x, k, **kwargs):
    omega=1/np.pi
    if 'omega' in kwargs:
        omega = kwargs['omega']
    r = x - k
    y = sp.exp(-(omega*r)**2)
    return y

# chirp wave
def chirp(x, k, **kwargs):
    r = x - k
    o = 10 - 0.5*r
    y = sp.pi*sp.sin(o*r)
    return y

# polynomial odd function
def poly(x, k, **kwargs):
    r = x - k
    y = 8*r - 14.3984*(r**3) + 4.77612*(r**5) - 0.82315*(r**7)
    return y

# sinc^10 function
def canvas(x, k, **kwargs):
    omega=2*np.pi
    m=10.
    if 'omega' in kwargs:
        omega = kwargs['omega']
    if 'm' in kwargs:
        m = kwargs['m']
    r = x - k
    z = omega*r/m
    y = (sp.sin(z)/(z+1e-10))**m
    return y

# canvased poly function
def canvased_poly(x, k, **kwargs):
    r = x - k
    c = canvas(x, k, **kwargs)
    p = poly(x, k, **kwargs)
    return c * p

# canvased chirp wave
def canvased_chirp(x, k, **kwargs):
    ca = canvas(x, k, **kwargs)
    ch = chirp(x, k, **kwargs)
    return ca*ch

# comb wave
def comb(x, k, **kwargs):
    r = x - k
    for i in range(10):
        if i == 0:
            f = sp.sin((i+1)*0.6*x)
        else:
            f += sp.sin((i+1)*0.6*x)
    return f

# canvased comb
def canvased_comb(x, k, **kwargs):
    ca = canvas(x, k, **kwargs)
    co = comb(x, k, **kwargs)
    return 0.45 * ca * co

# canvased linear climb
def canvased_linear(x, k, **kwargs):
    c = canvas(x, k, **kwargs)
    l = x - k
    return 6 * l * c

# Dr. Howell's most recent custom wave
def flatline(x, k, **kwargs):
    r = x - k
    sol = [ 0.65050157,
            0.48107457,
           -0.49920901,
           -0.10730917,
           -0.29207321,
            0.15058321,
            0.07248865]
    F_SO = 0
    for i in range(len(sol)):
        F_SO += (sol[i] * sp.sin(2*np.pi*(1+i/(len(sol)-1))*r))
    s = .05
    F_SO *= (sp.sin(s*r)/(s*(r+1e-8)))**8
    return F_SO

# superoscillation function with seven frequencies. total wave's bandwidth is 1/4 
# of the bandwidth in the superregion. more information in the metadata file in 
# '../MLE/ghost_waves/7peaks/' folder. 
def SevenPeaks(x, k, **kwargs):
    complx = False
    if 'complx' in kwargs:
        complx = kwargs['complx']
    r = x - k
    amps = [  121.58453965, 
             -612.02092549, 
             1361.03496265, 
            -1704.84150255, 
             1268.89246146, 
             -533.27342635, 
               99.59269209]
    freqs = np.array([10, 11, 12, 13, 14, 15, 16])/24
    f = 0
    for i in range(len(amps)):
        f += amps[i] * (sp.cos(2*np.pi*r*freqs[i]) + 1j*sp.sin(2*np.pi*r*freqs[i])*int(complx))
    return f

# superoscillation function with six frequencies. total wave's bandwidth is 1/4 
# of the bandwidth in the superregion. more information in the metadata file in 
# '../MLE/ghost_waves/6peaks/' folder. 
def SixPeaks(x, k, **kwargs):
    _2D = False
    if '_2D' in kwargs:
        _2D = kwargs['_2D']
    r = x - k
    amps = [ 17.52955301,
            -49.82368358,
             48.68712141,
             -6.97596888,
            -17.36600119,
              8.90783108]
    freqs = np.array([0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
    f = 0
    for i in range(len(amps)):
        f += amps[i] * (sp.cos(2*np.pi*r*freqs[i]) + 1j*sp.sin(2*np.pi*r*freqs[i])*int(_2D))
    return f

# ----------------------------------------------------------------------------------------------------------------
# creates a scattered wave based on a sympy function and lists of seps and amps
def create_wave(times, wf, seps, amps, y_off=0, **kwargs):
    _2D = False
    if '_2D' in kwargs:
        _2D = kwargs['_2D']
    base_y = 0
    complx = False
    if 'base_y' in kwargs:
        base_y = kwargs['base_y']
    if 'complx' in kwargs:
        complx = kwargs['complx']
    if complx == True:
        seps, amps = complexify(seps, amps)
    sim = np.zeros(len(times))
    if _2D == True:
        sim = np.zeros(len(times), dtype=np.complex128)
    for i in range(len(seps)):
        sim += (wf(times, seps[i], amps[i]) + base_y)
    sim += y_off
    return sim

# ----------------------------------------------------------------------------------------------------------------
# uses the above function to create a new wave for singular use
def create_new_wave(times, Func, seps, amps, y=0, **kwargs):
    x, k, A  = sp.symbols('x k A')
    wavefunc = A*Func(x, k, **kwargs)
    wf       = sp.lambdify((x, k, A), wavefunc)
    wave     = create_wave(times, wf, seps, amps, y_off=y, **kwargs)
    return wave

# creates complex scatter distribution of echoes, given some initial scatterers
def complexify(seps, amps, num_echoes=3):
    seps_copy = seps.copy()
    amps_copy = amps.copy()
    for j in range(1, min(len(amps), 3)):
        sepdist = seps[j] - seps[0]
        for i in range(num_echoes):
            seps_copy.append(seps_copy[j] + (i+1)*sepdist)
            amps_copy.append(amps[j] / ((-2.5)**(i+1)))
    df = pd.DataFrame({'seps': seps_copy, 'amps': amps_copy})
    result = df.groupby('seps', as_index=False)['amps'].sum()
    return result['seps'].to_numpy(), result['amps'].to_numpy()

# ----------------------------------------------------------------------------------------------------------------
# adds noise to a time series at a given signal-to-noise ratio.
# otherwise, optionally adds a provided noise time series to the wave.
def add_noise(wave, snr=100, noise=None):
    if noise is None:
        amp = np.sqrt(np.mean(wave**2))
        noise = np.random.normal(0, amp/snr, len(wave))
    return wave + noise

# ----------------------------------------------------------------------------------------------------------------
# get error between data and sim
def get_error(data, sim):
    error = data - sim
    return error

# ----------------------------------------------------------------------------------------------------------------
# returns the MSE between data and sim
def get_loss(data, sim):
    return np.mean((data - sim)**2)

# ----------------------------------------------------------------------------------------------------------------
# creates a scattered wave and returns the wave and its partials
def get_gradients(times, err, pk, pA, seps, amps, y_off=0, use_A=True):
    # storage spots for final results
    dAs = [[0. for i in range(len(times))] for j in range(len(seps))]
    dks = [[0. for i in range(len(times))] for j in range(len(amps))]
    for i in range(len(seps)):
        dks[i] = pk(times, seps[i], amps[i])
        if use_A == True:
            dAs[i] = pA(times, seps[i], amps[i])
    dy = 1
    dh_dAs = [np.mean(err*i) for i in dAs]
    dh_dks = [np.mean(err*i) for i in dks]
    dh_dy  = np.mean(err*dy)
    return dh_dks, dh_dAs, dh_dy

# ----------------------------------------------------------------------------------------------------------------
# updates values based on gradients
def descend(seps, amps, y, sgrads, agrads, ygrad, lr, b=0., start=0, use_y=True, use_A=True, **kwargs):
    seps[start:] += ((lr * np.array(sgrads[start:])) + b)
    if use_A == True:
        amps[start:] += ((lr * np.array(agrads[start:])) + b)
    if use_y == True:
        y += ((lr * ygrad) + b)
    return seps, amps, y

# ----------------------------------------------------------------------------------------------------------------
# main function. finds minimum MSE by way of gradient descent.
def find_min(times, data, Func, seps1, amps1, y=0, lr=0.001, b=0,
             threshold=1e-6, start=0, print_status=False, use_y=True, use_A=True, **kwargs):
    seps = seps1.copy()
    amps = amps1.copy()
    x, k, A  = sp.symbols('x k A')
    loss     = 10000
    min_loss = 10000
    old_loss = 1
    i        = 0
    j        = 0

    # create sympy function to be used for building sim wave
    wavefunc = A*Func(x, k, **kwargs)

    # make function and partials all numpy functions for faster computing
    wf = sp.lambdify((x, k, A), wavefunc)
    pA = sp.lambdify((x, k, A), sp.diff(wavefunc, A))
    pk = sp.lambdify((x, k, A), sp.diff(wavefunc, k))

    # main loop
    while j < 10:
        sim = create_wave(times, wf, seps, amps, y_off=y, **kwargs)
        err = get_error(data, sim)
        dks, dAs, dy = get_gradients(times, err, pk, pA, seps, amps, y_off=y, use_A=use_A)
        seps, amps, y = descend(seps, amps, y, dks, dAs, dy, lr, b=b, start=start, use_y=use_y, use_A=use_A)
        old_loss = loss
        loss = get_loss(data, sim)
        if print_status == True:
            if i < 25:
                print(i, '---', loss)
            elif i < 1000 and i % 100 == 0:
                print(i, '---', loss)
            elif i % 1000 == 0:
                print(i, '---', loss)

        i += 1
        if loss < min_loss:
            min_loss = loss
            stuff = [i, seps, amps, y, loss]
        if abs(loss-old_loss)/loss < threshold:
            j += 1
        else:
            j = 0

    return stuff

# ----------------------------------------------------------------------------------------------------------------
# write-to-file function
def write_to_file(filepath, stuff, start_sep1, seps, amps, offset, snr, length, thr, lr):
    if not os.path.exists(filepath):
        sim_sepnames = ['sim_sep'+str(i+1) for i in range(len(stuff[1]))]
        sim_ampnames = ['sim_amp'+str(i+1) for i in range(len(stuff[2]))]
        data_sepnames = ['data_sep'+str(i+1) for i in range(len(stuff[1]))]
        data_ampnames = ['data_amp'+str(i+1) for i in range(len(stuff[2]))]
        names = [*data_sepnames, *data_ampnames, 'data DC offset', 'SNR', '# time steps',
                'threshold', 'learning rate', 'sim starting separation', 'iterations',
                *sim_sepnames, *sim_ampnames, 'sim DC offset', 'MSE']
        # Open the CSV file in append mode and write the header
        with open(filepath, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(names)
    # Open the CSV file in append mode and write the data
    with open(filepath, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([*seps, *amps, offset, snr, length, thr, lr, start_sep1,
                         stuff[0], *stuff[1], *stuff[2], stuff[-2], stuff[-1]])