function err = mmi_Rerr(net, x, t)

% MMI_RERR Wrapper function for the mixture mutual information bound.
% FORMAT
% DESC wraps MMI to allow it to be minimised.
% ARG net : input network.
% ARG x : input data (not used).
% ARG t : output data (not used).
% 
% COPYRIGHT : Neil D. Lawrence and Mehdi Azzouzi, 1998, 1999
%
% SEEALSO : MMI, mmi_Rgrad

% ENSMLP
  
err = -mmi(net);