function output_cell = IRF_moments(xparam1,M_,options_,oo_,estim_params_,bayestopt_,dataset_,dataset_info),
% function output_cell = IRF_moments(xparam1,M_,options_,oo_,estim_params_,bayestopt_,dataset_,dataset_info),
% Function called by execute_prior_posterior_function that generates
% variance decomposition for given set of parameters
% For input/output arguments, see the Dynare manual on posterior_function

% Copyright (C) 2016 Johannes Pfeifer
% 
%  This is free software: you can redistribute it and/or modify
%  it under the terms of the GNU General Public License as published by
%  the Free Software Foundation, either version 3 of the License, or
%  (at your option) any later version.
% 
%  It is distributed in the hope that it will be useful,
%  but WITHOUT ANY WARRANTY; without even the implied warranty of
%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%  GNU General Public License for more details.
% 
%  For a copy of the GNU General Public License,
%  see <http://www.gnu.org/licenses/>.
global options_ M_ oo_

M_ = set_all_parameters(xparam1,estim_params_,M_); %set parameters
options_.noprint=1;
options_.nocorr=1;
options_.order=1;
options_.irf=20;
options_.nograph=1;
var_list_=char('dy','dc','labobs','dw'); %define variable list
info = stoch_simul(var_list_); %run stoch_simul on the variable list
if ~info(1)
    output_cell{1,1}  = (oo_.irfs.dy_eZ);
    output_cell{1,2}  = (oo_.irfs.dc_eZ);
    output_cell{1,3}  = (oo_.irfs.labobs_eZ);
    output_cell{1,4}  = (oo_.irfs.dw_eZ);

else
    output_cell={[],[],[],[]};
end
