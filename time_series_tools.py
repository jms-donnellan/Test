# Test

import numpy as np
from astroML.utils import check_random_state

from astroML.time_series import  lomb_scargle
    
def LS_bootstrap_err_est(t, y, dy, omega,
                           generalized=True, subtract_mean=True,
                           N_bootstraps=100, random_state=None,
                           hist=False, plot_hist=True,Nbins=200):
    """Use a bootstrap analysis to compute Lomb-Scargle error estimation
    Parameters
    ----------
    The first set of parameters are passed to the lomb_scargle algorithm
    t : array_like
        sequence of times
    y : array_like
        sequence of observations
    dy : array_like
        sequence of observational errors
    omega : array_like
        frequencies at which to evaluate p(omega)
    generalized : bool
        if True (default) use generalized lomb-scargle method
        otherwise, use classic lomb-scargle.
    subtract_mean : bool
        if True (default) subtract the sample mean from the data before
        computing the periodogram.  Only referenced if generalized is False
    Remaining parameters control the bootstrap
    N_bootstraps : int
        number of bootstraps
    random_state : None, int, or RandomState object
        random seed, or random number generator
    Returns
    -------
    D : ndarray
        distribution of the height of the highest peak
    """
    random_state = check_random_state(random_state)
    t = np.asarray(t)
    y = np.asarray(y)
    dy = np.asarray(dy) + np.zeros_like(y)

    D = np.zeros(N_bootstraps)
    omegaD= np.zeros(N_bootstraps)
    
    for i in range(N_bootstraps):
        ind = random_state.randint(0, len(y), len(y))
        #print(ind)
        #print(t[ind], y[ind], dy[ind])
        #print(subtract_mean)
        ###el vector de tiempo es patológico, dado que hay una observación aislada en dos días distintos, aunque no sea correcto del todo vamos a conservar los dos puntos en esas noches en concreto para evitar que el mook periodogram de resultados raros (i.e. periodo=0) 
        ##ind[-2]=11
        ##ind[-1]=-1
        p = lomb_scargle(t[ind], y[ind], dy[ind], omega,
                         generalized=generalized, subtract_mean=subtract_mean)
        D[i] = p.max()
        omegaD[i]=omega[p.argmax()]
        
        #if omegaD[i]==min(omega):
        #    from matplotlib import pyplot as plt
        #    plt.plot(omega,p)
        #    plt.figure()
        #    print(t[ind],y[ind],dy[ind])
            
            
    if hist:
        
        if plot_hist:
            from matplotlib import pyplot as plt
            frecD=omegaD.copy()/(2*np.pi)
            
            plt.figure('bootstrap hist')
            plt.hist(frecD,normed=True, bins=Nbins)
            plt.hist(frecD,normed=True,histtype='step')

            plt.figure('bootstrap cumhist')
            Xcum=np.sort(D)
            Ycum=np.array(range(N_bootstraps))/float(N_bootstraps)
            plt.plot(Xcum,Ycum)
            #plt.xlim(Xcum,Xcum)
            plt.grid()
            plt.minorticks_on()
            plt.xlabel('')

        return D,omegaD
    else:
        return D

def LS_chi2(t, dy, y, omega,
            phase,A,B=0,
            plot=True):
    """Compute the Chi2 value and the reducted (Chi2_nu) for the given frecuendies.
        
    ----------
    The first set of parameters are passed to the lomb_scargle algorithm
    t : array_like
        sequence of times
    dy : array_like
        sequence of observational errors
        omega : array_like
        frequencies at which to evaluate p(omega)
    generalized : bool
        if True (default) use generalized lomb-scargle method
        otherwise, use classic lomb-scargle.
    subtract_mean : bool
        if True (default) subtract the sample mean from the data before
        computing the periodogram.  Only referenced if generalized is False
    frec : best estimated frecuency 
        frequency at which to evaluate p(omega)
    A : float
        Best fitted sine amplitude
    B : float or none
        If generalized, best constant shift of the sinusoidal.
    phase : float
        phase shift of the best fitted sine wave.
    Remaining parameters control the bootstrap
    N_bootstraps : int
        number of bootstraps
    random_state : None, int, or RandomState object
        random seed, or random number generator
    Returns
    -------
    D : ndarray
        distribution of the height of the highest peak
    """
    y = np.asarray(y)
    t = np.asarray(t)
    dy = np.asarray(dy) + np.zeros_like(y)
    
    
    
    Chi2=[]
    Chi2_nu=[]
    
    def bst_sin(frec,a,b,B,t): #we define the best sinusoidal model 
        w=2.*np.pi*frec
        best_sin=a*np.sin(w*t)+b*np.cos(w*t)+B#A*np.sin(2.*np.pi*frec* (t - phase))+B
        return(best_sin)
    
    from lmfit import Parameters, minimize
    p=Parameters()
    #p.add('phase',0., max=np.pi,min=-np.pi) 
    p.add('a',0.)   
    p.add('b',0.)
    p.add('B',0.)   
    p.add('frec',0.,vary=False)
    
    def residual(p):
        pd=p.valuesdict()
        return bst_sin(pd['frec'],pd['a'],pd['b'],pd['B'],np.array(t))-y
    

    for i in range(len(omega)):
        p['frec'].value=omega[i]/(2*np.pi)
        ph_shift = minimize(residual,p,method='Nelder')
        #phase=ph_shift.params['phase']
        #A=ph_shift.params['A']
        A=np.sqrt(ph_shift.params['a']**2+ph_shift.params['b']**2)
        phase=np.arctan2(ph_shift.params['b'],ph_shift.params['a'])
        B=ph_shift.params['B'].value

        
        y_mod = A*np.sin((omega[i]* t) + phase) + B
        
        Xi = np.sum(((y-y_mod)/dy)**2)
        
        
        Chi2.append(Xi)
        Chi2_nu.append(Xi/(len(y)-1))
        
        #if i==300:
        #    from matplotlib import pyplot as plt
        #    plt.figure()
        #    plt.plot(t,y)
        #    plt.plot(t,y_mod)
        #D[i] = p.max()
        #omegaD[i]=omega[p.argmax()]
        
        #if omegaD[i]==min(omega):
        #    from matplotlib import pyplot as plt
        #    plt.plot(omega,p)
        #    plt.figure()
        #    print(t[ind],y[ind],dy[ind])
        
        
    Chi2=np.array(Chi2)
    Chi2_nu=np.array(Chi2_nu)        
    if plot:
            from matplotlib import pyplot as plt
            frec=omega.copy()/(2*np.pi)
            
            f = plt.figure('Chi2')
            ax1 = f.add_subplot(111)
            ax1.plot(frec,Chi2)
            ax1.set_yscale('log')
            
            ax2 = ax1.twinx()
            ax2.plot(frec,Chi2_nu)
            #ax2.set_ylim(tuple([Chi2_nu.min(),Chi2_nu.max()]))
            ax2.set_ylabel(r'Chi^2_nu')
            ax2.set_yscale('log')
            
            
            plt.grid()
            plt.minorticks_on()
            plt.xlabel('frec [$d^{-1}$]')

            return Chi2,Chi2_nu
    
    return Chi2,Chi2_nu

def LS__semi_param_bootstrap_err_est(t, dy, frec,omega,
                                     residuals,phase,A,B=0,
                                     generalized=True, subtract_mean=True,
                                     N_bootstraps=100, random_state=None,
                                     hist=False, plot_hist=True,Nbins=200):
    """Use a semi-parametric-bootstrap analysis to compute Lomb-Scargle error estimation
        https://stats.stackexchange.com/questions/67519/bootstrapping-residuals-am-i-doing-it-right
    Parameters
        
    ----------
    The first set of parameters are passed to the lomb_scargle algorithm
    t : array_like
        sequence of times
    dy : array_like
        sequence of observational errors
        omega : array_like
        frequencies at which to evaluate p(omega)
    generalized : bool
        if True (default) use generalized lomb-scargle method
        otherwise, use classic lomb-scargle.
    subtract_mean : bool
        if True (default) subtract the sample mean from the data before
        computing the periodogram.  Only referenced if generalized is False
    frec : best estimated frecuency 
        frequency at which to evaluate p(omega)
    A : float
        Best fitted sine amplitude
    B : float or none
        If generalized, best constant shift of the sinusoidal.
    phase : float
        phase shift of the best fitted sine wave.
    Remaining parameters control the bootstrap
    N_bootstraps : int
        number of bootstraps
    random_state : None, int, or RandomState object
        random seed, or random number generator
    Returns
    -------
    D : ndarray
        distribution of the height of the highest peak
    """
    random_state = check_random_state(random_state)
    y = A*np.sin((2.*np.pi*frec* t) + phase)
    y = np.asarray(y)
    t = np.asarray(t)
    dy = np.asarray(dy) + np.zeros_like(y)
    
    
    D = np.zeros(N_bootstraps)
    omegaD= np.zeros(N_bootstraps)
    
    for i in range(N_bootstraps):
        ind = random_state.randint(0, len(t), len(t))
        #print(ind)
        #print(t[ind], y[ind], dy[ind])
        #print(subtract_mean)
        ###el vector de tiempo es patológico, dado que hay una observación aislada en dos días distintos, aunque no sea correcto del todo vamos a conservar los dos puntos en esas noches en concreto para evitar que el mook periodogram de resultados raros (i.e. periodo=0) 
        ##ind[-2]=11
        ##ind[-1]=-1
        y_mock = y + residuals[ind]
        p = lomb_scargle(t, y_mock, dy, omega,
                         generalized=generalized, subtract_mean=subtract_mean)
        D[i] = p.max()
        omegaD[i]=omega[p.argmax()]
        
        #if omegaD[i]==min(omega):
        #    from matplotlib import pyplot as plt
        #    plt.plot(omega,p)
        #    plt.figure()
        #    print(t[ind],y[ind],dy[ind])
            
            
    if hist:
        
        if plot_hist:
            from matplotlib import pyplot as plt
            frecD=omegaD.copy()/(2*np.pi)
            
            plt.figure('bootstrap hist')
            plt.hist(frecD,normed=True, bins=Nbins)
            plt.hist(frecD,normed=True,histtype='step')

            plt.figure('bootstrap cumhist')
            Xcum=np.sort(D)
            Ycum=np.array(range(N_bootstraps))/float(N_bootstraps)
            plt.plot(Xcum,Ycum)
            #plt.xlim(Xcum,Xcum)
            plt.grid()
            plt.minorticks_on()
            plt.xlabel('')

        return D,omegaD
    else:
        return D


def LS_bootstrap_sig(t, y, dy, omega,
                           generalized=True, subtract_mean=True,
                           N_bootstraps=100, random_state=None,
                           hist=False, plot_hist=True,Nbins=200):
    """Use a bootstrap analysis to compute Lomb-Scargle significance
    Parameters
    ----------
    The first set of parameters are passed to the lomb_scargle algorithm
    t : array_like
        sequence of times
    y : array_like
        sequence of observations
    dy : array_like
        sequence of observational errors
    omega : array_like
        frequencies at which to evaluate p(omega)
    generalized : bool
        if True (default) use generalized lomb-scargle method
        otherwise, use classic lomb-scargle.
    subtract_mean : bool
        if True (default) subtract the sample mean from the data before
        computing the periodogram.  Only referenced if generalized is False
    Remaining parameters control the bootstrap
    N_bootstraps : int
        number of bootstraps
    random_state : None, int, or RandomState object
        random seed, or random number generator
    Returns
    -------
    D : ndarray
        distribution of the height of the highest peak
    """
    random_state = check_random_state(random_state)
    t = np.asarray(t)
    y = np.asarray(y)
    dy = np.asarray(dy) + np.zeros_like(y)

    D = np.zeros(N_bootstraps)
    omegaD= np.zeros(N_bootstraps)
    
    for i in range(N_bootstraps):
        ind = random_state.randint(0, len(y), len(y))
        #print[ind]
        p = lomb_scargle(t, y[ind], dy[ind], omega,
                         generalized=generalized, subtract_mean=subtract_mean)
        D[i] = p.max()
        omegaD[i]=omega[p.argmax()]
    if hist:
        
        if plot_hist:
            from matplotlib import pyplot as plt
            frecD=omegaD.copy()/(2*np.pi)
            
            plt.figure('bootstrap hist')
            plt.hist(frecD,normed=True, bins=Nbins)
            plt.hist(frecD,normed=True,histtype='step')

            plt.figure('bootstrap cumhist')
            Xcum=np.sort(D)
            Ycum=np.array(range(N_bootstraps))/float(N_bootstraps)
            plt.plot(Xcum,Ycum)
            #plt.xlim(Xcum,Xcum)
            plt.grid()
            plt.minorticks_on()
            plt.xlabel('')

        return D,omegaD
    else:
        return D

def LS_window_white_noise(t, omega, y=0, dy=1,
                           generalized=False, subtract_mean=False,
                           random_state=None, N_mock=100,
                           hist=False, plot_hist=True,Nbins=200):
    """Use a monte carlo simulation to compute Lomb-Scargle white noise 
        significance for the given spectral window
    Parameters
    ----------
    The first set of parameters are passed to the lomb_scargle algorithm
    t : array_like
        sequence of times
    omega : array_like
        frequencies at which to evaluate p(omega)
    generalized : bool
        if True (default) use generalized lomb-scargle method
        otherwise, use classic lomb-scargle.
    subtract_mean : bool
        if True (default) subtract the sample mean from the data before
        computing the periodogram.  Only referenced if generalized is False
    Remaining parameters control the bootstrap
    N_mock : int
        number of simulations
    random_state : None, int, or RandomState object
        random seed, or random number generator
    Returns
    -------
    D : ndarray
        distribution of the height of the highest peak
    if hist=True:
        omegaD : ndarray
            distribution of the angular frecuencies corresponding to D
    """
    random_state = check_random_state(random_state)
    t = np.asarray(t)
    #dy = np.ones_like(y)

    D = np.zeros(N_mock)
    omegaD= np.zeros(N_mock)
    #PS_mock_all=[]
    #PS_mock_max=np.array([])
    #omega_mock_max=np.array([])
    for i in range(N_mock):
        #ind = random_state.randint(0, len(y), len(y))
        y = np.random.normal(y, dy , size=len(t))
        #print y 
        p = lomb_scargle(t, y, dy, omega,
                         generalized=generalized, subtract_mean=subtract_mean)
        #print p
        D[i] = p.max()
        omegaD[i]=omega[p.argmax()]
        
        
    #max_PS_mock=np.max(PS_mock_all,axis=0)
    if hist:
        
        if plot_hist:
            from matplotlib import pyplot as plt
            frecD=omegaD.copy()/(2*np.pi)
            
            plt.figure('white noise peak hist')
            plt.hist(frecD,normed=True, bins=Nbins)
            plt.hist(frecD,normed=True,histtype='step')

            plt.figure('white noise peak cumhist')
            Xcum=np.sort(D)
            Ycum=np.array(range(N_mock))/float(N_mock)
            plt.plot(Xcum,Ycum)
            #plt.xlim(Xcum,Xcum)
            plt.grid()
            plt.minorticks_on()
            plt.xlabel('')

        return D,omegaD
    else:
        return D
    
    
    
def LS_null_hypotesis(t, y, dy, omega, 
                           generalized=False, subtract_mean=False,
                           random_state=None, N_mock=1000,
                           hist=False, plot_hist=True,Nbins=200):
    """Use a monte carlo simulation to compute Lomb-Scargle null hypotesis periodogram
        shuffling the vector {t U y_obs}
    Parameters
    ----------
    The first set of parameters are passed to the lomb_scargle algorithm
    t : array_like
        sequence of times
    y : array_like
        sequence of observations
    dy : array_like
        sequence of observational errors
    omega : array_like
        frequencies at which to evaluate p(omega)
    generalized : bool
        if True (default) use generalized lomb-scargle method
        otherwise, use classic lomb-scargle.
    subtract_mean : bool
        if True (default) subtract the sample mean from the data before
        computing the periodogram.  Only referenced if generalized is False
    Remaining parameters control the bootstrap
    N_mock : int
        number of simulations
    random_state : None, int, or RandomState object
        random seed, or random number generator
    Returns
    -------
    D : ndarray
        distribution of the height of the highest peak
    if hist=True:
        omegaD : ndarray
            distribution of the angular frecuencies corresponding to D
    """
    #print(len(y))
    #print(len(dy))
    random_state = check_random_state(random_state)
    t = np.asarray(t)
    #dy = np.ones_like(y)
    y = np.asarray(y)
    dy = np.asarray(dy) + np.zeros_like(y)

    D = np.zeros(N_mock)
    omegaD= np.zeros(N_mock)

    mock_vector=np.append(y,t)
    mock_dy=dy.copy()
    
    for i in range(100):
        np.random.shuffle(mock_vector)
        print(mock_vector)
    for i in range(N_mock):
        #ind = random_state.randint(0, len(y), len(y))
        #y = np.random.normal(len(t))
        np.random.shuffle(mock_vector) 
        np.random.shuffle(mock_dy)   
        
        p = lomb_scargle(mock_vector[:len(t)], mock_vector[-len(t):], mock_dy, omega,\
                         generalized=generalized, subtract_mean=subtract_mean)
        D[i] = p.max()
        omegaD[i]=omega[p.argmax()]
        
        
    #max_PS_mock=np.max(PS_mock_all,axis=0)
    if hist:
        
        if plot_hist:
            from matplotlib import pyplot as plt
            frecD=omegaD.copy()/(2*np.pi)
            
            plt.figure('null hypothesis hist')
            plt.hist(frecD,normed=True, bins=Nbins)
            plt.hist(frecD,normed=True,histtype='step')

            plt.figure('null hypotesis cumhist')
            Xcum=np.sort(D)
            Ycum=np.array(range(N_mock))/float(N_mock)
            plt.plot(Xcum,Ycum)
            #plt.xlim(Xcum,Xcum)
            plt.grid()
            plt.minorticks_on()
            plt.xlabel('')

        return D,omegaD
    else:
        return D
    
def LS_A_phi_estmation(t, y, dy, omega, generalized=True,
                 subtract_mean=True, significance=None):
    """
    (Generalized) Lomb-Scargle Periodogram with Floating Mean
    Parameters
    ----------
    t : array_like
        sequence of times
    y : array_like
        sequence of observations
    dy : array_like
        sequence of observational errors
    omega : array_like
        frequencies at which to evaluate p(omega)
    generalized : bool
        if True (default) use generalized lomb-scargle method
        otherwise, use classic lomb-scargle.
    subtract_mean : bool
        if True (default) subtract the sample mean from the data before
        computing the periodogram.  Only referenced if generalized is False
    significance : None or float or ndarray
        if specified, then this is a list of significances to compute
        for the results.
    Returns
    -------
    p : array_like
        Lomb-Scargle power associated with each frequency omega
    z : array_like
        if significance is specified, this gives the levels corresponding
        to the desired significance (using the Scargle 1982 formalism)
    Notes
    -----
    The algorithm is based on reference [1]_.  The result for generalized=False
    is given by equation 4 of this work, while the result for generalized=True
    is given by equation 20.
    Note that the normalization used in this reference is different from that
    used in other places in the literature (e.g. [2]_).  For a discussion of
    normalization and false-alarm probability, see [1]_.
    To recover the normalization used in Scargle [3]_, the results should
    be multiplied by (N - 1) / 2 where N is the number of data points.
    References
    ----------
    .. [1] M. Zechmeister and M. Kurster, A&A 496, 577-584 (2009)
    .. [2] W. Press et al, Numerical Recipies in C (2002)
    .. [3] Scargle, J.D. 1982, ApJ 263:835-853
    """
    t = np.asarray(t)
    y = np.asarray(y)
    dy = np.asarray(dy) * np.ones_like(y)

    assert t.ndim == 1
    assert y.ndim == 1
    assert dy.ndim == 1
    assert t.shape == y.shape
    assert y.shape == dy.shape

    w = 1. / dy / dy
    w /= w.sum()

    # the generalized method takes care of offset automatically,
    # while the classic method requires centered data.
    if (not generalized) and subtract_mean:
        # subtract MLE for mean in the presence of noise.
        y = y - np.dot(w, y)

    omega = np.asarray(omega)
    shape = omega.shape
    omega = omega.ravel()[np.newaxis, :]

    t = t[:, np.newaxis]
    y = y[:, np.newaxis]
    dy = dy[:, np.newaxis]
    w = w[:, np.newaxis]

    sin_omega_t = np.sin(omega * t)
    cos_omega_t = np.cos(omega * t)

    # compute time-shift tau
    # S2 = np.dot(w.T, np.sin(2 * omega * t)
    S2 = 2 * np.dot(w.T, sin_omega_t * cos_omega_t)
    # C2 = np.dot(w.T, np.cos(2 * omega * t)
    C2 = 2 * np.dot(w.T, 0.5 - sin_omega_t ** 2)

    if generalized:
        S = np.dot(w.T, sin_omega_t)
        C = np.dot(w.T, cos_omega_t)

        S2 -= (2 * S * C)
        C2 -= (C * C - S * S)

    tan_2omega_tau = S2 / C2
    tau = np.arctan(tan_2omega_tau)
    tau *= 0.5
    tau /= omega

    # compute components needed for the fit
    omega_t_tau = omega * (t - tau)

    sin_omega_t_tau = np.sin(omega_t_tau)
    cos_omega_t_tau = np.cos(omega_t_tau)

    Y = np.dot(w.T, y)
    YY = np.dot(w.T, y * y) - Y * Y

    wy = w * y

    YCtau = np.dot(wy.T, cos_omega_t_tau)
    YStau = np.dot(wy.T, sin_omega_t_tau)
    CCtau = np.dot(w.T, cos_omega_t_tau * cos_omega_t_tau)
    SStau = np.dot(w.T, sin_omega_t_tau * sin_omega_t_tau)

    if generalized:
        Ctau = np.dot(w.T, cos_omega_t_tau)
        Stau = np.dot(w.T, sin_omega_t_tau)

        YCtau -= Y * Ctau
        YStau -= Y * Stau
        CCtau -= Ctau * Ctau
        SStau -= Stau * Stau

    p_omega = (YCtau * YCtau / CCtau + YStau * YStau / SStau) / YY
    p_omega = p_omega.reshape(shape)

    if significance is not None:
        N = t.size
        M = 2 * N
        z = (-2.0 / (N - 1.)
             * np.log(1 - (1 - np.asarray(significance)) ** (1. / M)))
        return p_omega, z
    else:
return p_omega
