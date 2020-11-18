import os
import shutil
import time

from oct2py import Oct2Py

import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets.widgets import Layout

from joblib import Parallel, delayed
import multiprocessing
from joblib import wrap_non_picklable_objects

import pandas as pd
import numpy as np

from feedback_processing import FeedbackProcessing



class GUI_session:
    """
    Class for handling Graphical User Inteterface:
    """
    def __init__(self,PPBO_settings):
        """
        Basic settings
        """
        
        self.D = PPBO_settings.D   #Problem dimension
        self.bounds = PPBO_settings.original_bounds #Boundaries of each variables as a sequence of tuplets
        self.alpha_grid_distribution = PPBO_settings.alpha_grid_distribution
        
        self.user_feedback_grid_size = PPBO_settings.user_feedback_grid_size #How many possible points in the user feedback grid?
        self.n_irfs_periods = 20 #This must be same as in a model.mod file
        self.FP = FeedbackProcessing(self.D, self.user_feedback_grid_size,self.bounds,self.alpha_grid_distribution,PPBO_settings.TGN_speed)
              
        self.current_xi = None
        self.current_x = None
        self.current_xi_grid = None
        self.dsge_results = None
        self.user_feedback = None  #variable to store user feedback
        self.user_feedback_was_given = False  
        self.popup_slider_has_been_called = False
        
        self.results = pd.DataFrame(columns=(['alpha_xi_x' + str(i) for i in range(1,self.D+1)] 
                    + ['xi' + str(i) for i in range(1,self.D+1)]
                    + ['alpha_star']),dtype=np.float64)  #The output of the session is a dataframe containing user feedback
        
        self.engine = 'OCTAVE'
        
        ''' Prepare model files to temp '''
        self.path_to_dsge = os.getcwd() + '/dsge'  #CHECK THAT THIS IS CORRECT
        if os.path.exists(self.path_to_dsge + '/temp'):
            shutil.rmtree(self.path_to_dsge + '/temp')
        os.mkdir(self.path_to_dsge + '/temp')
        for i in range(self.user_feedback_grid_size):
            shutil.copy2('dsge/US_FU19_rep.mod', 'dsge/temp/US_FU19_rep_'+str(i)+'.mod')
        shutil.copy2('dsge/prior_function_US_FU19.m', 'dsge/temp/prior_function_US_FU19.m')

                
    
    ''' Auxiliary functions '''    
    def set_xi(self,xi):
        self.current_xi = xi
    def set_x(self,x):
        self.current_x = x
    def create_xi_grid(self):
        self.current_xi_grid = self.FP.xi_grid(xi=self.current_xi,x=self.current_x,
                                               alpha_grid_distribution='evenly',
                                               alpha_star=None,m=self.user_feedback_grid_size,
                                               is_scaled=False)
 

    
    def simulate_DSGE(self):
        """
        Generate IRFs given prior parameters
        """   

#        dsge_results = []         
#        for i in range(self.current_xi_grid.shape[0]):
#            conf = list(self.current_xi_grid[i,:])
#            print("lambda = "+str(conf))
#            if self.engine == 'OCTAVE':
#                self.octave.push('param0',conf[0])
#                self.octave.push('param1',conf[1])
#                self.octave.push('param2',conf[2])
#                self.octave.push('param3',conf[3])
#                self.octave.push('param4',conf[4])
#                self.octave.push('param5',conf[5])
#                self.octave.push('param6',conf[6])
#                self.octave.run('simulate_DSGE.m',verbose=False)
#                dsge_results.append({'irf_dy_eZ': self.octave.pull('irf_dy_eZ'),'irf_dc_eZ': self.octave.pull('irf_dc_eZ')})
#            else:
#                self.eng.eval('param0'+str(conf[0])+";", nargout=0)
#                self.eng.eval('param1'+str(conf[1])+";", nargout=0)
#                self.eng.eval('param2'+str(conf[2])+";", nargout=0)
#                self.eng.eval('param3'+str(conf[3])+";", nargout=0)
#                self.eng.eval('param4'+str(conf[4])+";", nargout=0)
#                self.eng.eval('param5'+str(conf[5])+";", nargout=0)
#                self.eng.eval('param6'+str(conf[6])+";", nargout=0)
#                self.eng.simulate_DSGE(nargout=0)  #simulate_DSGE is the filename of matlab-script
#                dsge_results.append({'irf_dy_eZ': self.eng.workspace['irf_dy_eZ'],'irf_dc_eZ': self.eng.workspace['irf_dc_eZ']})

         
        
        #Multi-procesessing loop for non-picklable objects
        @delayed
        @wrap_non_picklable_objects
        def func_async_wrapped(conf,i, *args):
            
            octave = Oct2Py()
            octave.addpath(r'/usr/lib/dynare/matlab')  #CHECK THAT THIS IS CORRECT
            octave.cd(self.path_to_dsge+'/temp') 
            octave.push('param0',conf[0])
            octave.push('param1',conf[1])
            octave.push('param2',conf[2])
            octave.push('param3',conf[3])
            octave.push('param4',conf[4])
            octave.push('param5',conf[5])
            octave.push('param6',conf[6])
            octave.eval('dynare US_FU19_rep_' +str(i)+ ' noclearall nolog',verbose=False)
            results = {'irf_dy_eZ': octave.pull('irf_dy_eZ'),'irf_dc_eZ': octave.pull('irf_dc_eZ'),
                       'irf_labobs_eZ': octave.pull('irf_labobs_eZ'),'irf_dw_eZ': octave.pull('irf_dw_eZ')}
            octave.exit()
            del octave
            return results                      
        self.dsge_results =Parallel(n_jobs=-2,backend='loky',temp_folder='tmp')(func_async_wrapped(list(self.current_xi_grid[i,:]),i) for i in range(self.current_xi_grid.shape[0]))
        
        
               
    def save_results(self):
        res = pd.DataFrame(columns=(['alpha_xi_x' + str(i) for i in range(1,self.D+1)] 
                    + ['xi' + str(i) for i in range(1,self.D+1)]
                    + ['alpha_star']),dtype=np.float64)    
        xi = self.current_xi
        x = self.current_x
        alpha_xi_x = self.user_feedback
        alpha_star = np.nanmin(alpha_xi_x[x==0]/xi[x==0])  #every component in alpha_xi_x[x==0]/xi[x==0] should be same
        new_row = list(alpha_xi_x) + list(xi) + [alpha_star]
        res.loc[0,:] = new_row
        self.results=self.results.append(res, ignore_index=True)

    
    def irf(self,ind,var):
        irfs = self.dsge_results[ind]
        if var=='dy':
            irf_ = irfs['irf_dy_eZ']
        elif var=='dc':
            irf_ = irfs['irf_dc_eZ']
        elif var=='labobs':
            irf_ = irfs['irf_labobs_eZ']
        elif var=='dw':
            irf_ = irfs['irf_dw_eZ']
        irf_ = np.array(irf_) #irf_.shape (414, 20)
        return irf_
    
    
    def prepare_app(self):
        """
        Prepares one iteration of the user session.
        Makes sure that configurations are set correctly.
        """
        self.create_xi_grid()
        self.simulate_DSGE()
        
        ''' --- GUI plot/slider --- '''        
        #https://pypi.org/project/jupyter-ui-poll/
#        plt.rcParams['figure.dpi'] = 127
#        fig = plt.figure()
#        fig.set_figheight(3.6)
#        ax1 = fig.add_subplot(221)
#        ax2 = fig.add_subplot(222, sharex=ax1, sharey=ax1)
#        ax3 = fig.add_subplot(223, sharex=ax1, sharey=ax1)
#        ax4 = fig.add_subplot(224, sharex=ax1, sharey=ax1)
#        fig.suptitle('IRFs to an shock to TFP', fontsize=12, fontweight='bold')
#        plt.subplots_adjust(left=0.07, bottom=0.19, hspace=0.42)
#        self.x_axis_points = list(range(1,self.n_irfs_periods+1))
#        ax3.set_xlabel('Period')
#        ax4.set_xlabel('Period')
#        ax1.title.set_text('dy')
#        ax2.title.set_text('dc')
#        ax3.title.set_text('labobs')
#        ax4.title.set_text('dw')
#        ax1.axhline(y=0, color='k', linestyle='dashed')
#        ax2.axhline(y=0, color='k', linestyle='dashed')
#        ax3.axhline(y=0, color='k', linestyle='dashed')
#        ax4.axhline(y=0, color='k', linestyle='dashed')
#        ax1.set_ylim([-0.1, 0.1])
#        ax2.set_ylim([-0.1, 0.1])
#        ax3.set_ylim([-0.1, 0.1])
#        ax4.set_ylim([-0.1, 0.1])
#        #fig.text(0.1, 0.01, "$\it{Please\ select\ the\ most\ realistic\ hyperparameter\ configuration. }$")
        
        plt.rcParams['figure.dpi'] = 135
        fig = plt.figure()
        fig.set_figheight(3.9)
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222, sharex=ax1, sharey=ax1)
        ax3 = fig.add_subplot(223, sharex=ax1, sharey=ax1)
        ax4 = fig.add_subplot(224, sharex=ax1, sharey=ax1)
        fig.suptitle('IRFs to an shock to TFP', fontsize=12, fontweight='bold')
        plt.subplots_adjust(left=0.07, bottom=0.19, hspace=0.42)
        self.x_axis_points = list(range(1,self.n_irfs_periods+1))
        ax3.set_xlabel('Period')
        ax4.set_xlabel('Period')
        ax1.title.set_text('dy')
        ax2.title.set_text('dc')
        ax3.title.set_text('labobs')
        ax4.title.set_text('dw')
        ax1.axhline(y=0, color='k', linestyle='dashed')
        ax2.axhline(y=0, color='k', linestyle='dashed')
        ax3.axhline(y=0, color='k', linestyle='dashed')
        ax4.axhline(y=0, color='k', linestyle='dashed')
        ax1.set_ylim([-0.1, 0.12])
        #fig.text(0.1, 0.01, "$\it{Please\ select\ the\ most\ realistic\ hyperparameter\ configuration. }$")
        
        
        l1 = ax1.errorbar(self.x_axis_points, np.mean(self.irf(0,'dy'),axis=0), np.std(self.irf(0,'dy'),axis=0), elinewidth=0.7, label='IRF_dy') 
        l2 = ax2.errorbar(self.x_axis_points, np.mean(self.irf(0,'dc'),axis=0), np.std(self.irf(0,'dc'),axis=0), elinewidth=0.7, label='IRF_dc')
        l3 = ax3.errorbar(self.x_axis_points, np.mean(self.irf(0,'labobs'),axis=0), np.std(self.irf(0,'labobs'),axis=0), elinewidth=0.7, label='IRF_labobs') 
        l4 = ax4.errorbar(self.x_axis_points, np.mean(self.irf(0,'dw'),axis=0), np.std(self.irf(0,'dw'),axis=0), elinewidth=0.7, label='IRF_dw')
        plt.draw()
        slider = widgets.IntSlider(min=1, max=self.user_feedback_grid_size, step=1, description='Slider: ', 
                                   value=1,continuous_update=False, readout=False,
                            layout=widgets.Layout(width='60%', height='80px',position='right'))
        button = widgets.Button(description='Confirm',icon='fa-check',button_style='success',layout=Layout(width='90px')) 
        def confirm(event):
            typed_value = int(slider.value)
            self.user_feedback = self.current_xi_grid[(int(typed_value)-1),:]
            self.user_feedback_was_given = True
            plt.close('all')
            
        button.on_click(confirm)
        plt.show(block=False)    
        return button,slider,fig, l1, l2, l3, l4
    
    def update_plot(self,l1,l2,l3,l4,fig,slider):
        param_ind = int(slider.value-1)
        update_errorbar(l1, self.x_axis_points, np.mean(self.irf(param_ind,'dy'),axis=0), xerr=None, yerr=np.std(self.irf(0,'dy'),axis=0)) 
        update_errorbar(l2, self.x_axis_points, np.mean(self.irf(param_ind,'dc'),axis=0), xerr=None, yerr=np.std(self.irf(0,'dc'),axis=0))
        update_errorbar(l3, self.x_axis_points, np.mean(self.irf(param_ind,'labobs'),axis=0), xerr=None, yerr=np.std(self.irf(0,'labobs'),axis=0)) 
        update_errorbar(l4, self.x_axis_points, np.mean(self.irf(param_ind,'dw'),axis=0), xerr=None, yerr=np.std(self.irf(0,'dw'),axis=0)) 
        fig.canvas.draw_idle()
        
    
def update_errorbar(errobj, x, y, xerr=None, yerr=None):
    ln, caps, bars = errobj


    if len(bars) == 2:
        assert xerr is not None and yerr is not None, "Your errorbar object has 2 dimension of error bars defined. You must provide xerr and yerr."
        barsx, barsy = bars  # bars always exist (?)
        try:  # caps are optional
            errx_top, errx_bot, erry_top, erry_bot = caps
        except ValueError:  # in case there is no caps
            pass

    elif len(bars) == 1:
        assert (xerr is     None and yerr is not None) or\
               (xerr is not None and yerr is     None),  \
               "Your errorbar object has 1 dimension of error bars defined. You must provide xerr or yerr."

        if xerr is not None:
            barsx, = bars  # bars always exist (?)
            try:
                errx_top, errx_bot = caps
            except ValueError:  # in case there is no caps
                pass
        else:
            barsy, = bars  # bars always exist (?)
            try:
                erry_top, erry_bot = caps
            except ValueError:  # in case there is no caps
                pass

    ln.set_data(x,y)

    try:
        errx_top.set_xdata(x + xerr)
        errx_bot.set_xdata(x - xerr)
        errx_top.set_ydata(y)
        errx_bot.set_ydata(y)
    except NameError:
        pass
    try:
        barsx.set_segments([np.array([[xt, y], [xb, y]]) for xt, xb, y in zip(x + xerr, x - xerr, y)])
    except NameError:
        pass

    try:
        erry_top.set_xdata(x)
        erry_bot.set_xdata(x)
        erry_top.set_ydata(y + yerr)
        erry_bot.set_ydata(y - yerr)
    except NameError:
        pass
    try:
        barsy.set_segments([np.array([[x, yt], [x, yb]]) for x, yt, yb in zip(x, y + yerr, y - yerr)])
    except NameError:
        pass



	
