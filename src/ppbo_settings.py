

class PPBO_settings:
    """
    Class that specifies the settings for Projective Preferential Bayesian Optimization
    """

    def __init__(self,
                 D,
                 bounds,
                 xi_acquisition_function,
                 theta_initial=[0.1,0.12,0.36],
                 user_feedback_grid_size=100,  
                 m=25,                           # number of pseudo comparisons for GP fitting
                 verbose=True,
                 EI_EXR_mc_samples=170,        
                 EI_EXR_BO_maxiter=25,
                 max_iter_fMAP_estimation=5000,
                 mustar_finding_trials=4):         
        
        """
        BASIC SETTINGS
        """
        self.verbose = verbose #Wheter or not to print what is happening under the hood
        self.user_feedback_grid_size = user_feedback_grid_size
        self.n_gausshermite_sample_points = 30 #How many sample points in GaussHermite quadrature for approximating the convolution in the likelihood?

        """
        THE DOMAIN OF THE PROBLEM 
        """
        self.D = D   #Problem dimension
        self.original_bounds = bounds   #((xmin,xmax),)*self.D #Boundaries of each variables as a sequence of tuplets
        
        """
        SETTINGS FOR THE OPTIMIZERS 
        """
        self.max_iter_fMAP_estimation = max_iter_fMAP_estimation
        self.fMAP_optimizer = 'trust-exact'   #scipy optimizer for f_MAP-estimation: trust-krylov or trust-exact
        self.mustar_finding_trials = mustar_finding_trials  #This depends on the noise level. #Good default values is 1.
         
        '''HYPERPARAMETER INITIAL VALUES'''
        self.theta_initial = theta_initial #Intial hyperparameters. Put None if you want keep default hyper-parameters.
       
        ''' PSEUDO-OBSERVATIONS '''
        self.n_pseudoobservations = m  #How many pseudoobservations. Note that Sigma condition number grows w.r.t. that!
        self.alpha_grid_distribution = 'TGN'   #evenly, cauchy or TGN (truncated generalized normal distribution)
        self.TGN_speed = 0.3 #a speed of transformation from uniform dist to normal dist if TGN is selected,  0.3-0.4
        
        ''' ACQUISITION STRATEGY '''
        #[PCD,EXT,RAND,EI,EI-FIXEDX,EXR,EI-EXT,EI-EXT-FAST,EI-VARMAX,EI-VARMAX-FAST]
        self.mc_samples = EI_EXR_mc_samples
        self.BO_maxiter = EI_EXR_BO_maxiter
        self.xi_acquisition_function = xi_acquisition_function
        if self.xi_acquisition_function == "PCD" or self.xi_acquisition_function == "EXT" :
            self.dim_query_prev_iter = self.D #That is, PCD/EXT starts from 1st dimension
            self.x_acquisition_function = 'exploit'
        elif self.xi_acquisition_function == "RAND":
            self.x_acquisition_function = 'random'
        elif self.xi_acquisition_function == 'EI' or self.xi_acquisition_function == 'EI-FIXEDX' or self.xi_acquisition_function == 'EXR':
            self.x_acquisition_function = 'none'
            #Initial nonzero dimensions of xi
            #self.xi_dims_prev_iter = list(range(self.D - 1))
            self.xi_dims_prev_iter = [0,1]
        elif self.xi_acquisition_function == 'EI-EXT' or self.xi_acquisition_function == 'EI-EXT-FAST':
            self.x_acquisition_function = 'exploit'
        elif self.xi_acquisition_function == 'EI-VARMAX' or self.xi_acquisition_function == 'EI-VARMAX-FAST':
            self.x_acquisition_function = 'varmax'
        else:
            print("Unknown acquisition function!")



