function g = mmi_Rgrad(net, x, t)

% MMI_RGRAD Wrapper function for the mixture mutual information gradient.
% FORMAT
% DESC wraps MMIGRAD to allow it to be used in minimisation.
% ARG net : input network.
% ARG x : input data (not used).
% ARG t : output data (not used).
% 
% COPYRIGHT : Neil D. Lawrence and Mehdi Azzouzi, 1998, 1999
%
% SEEALSO : MMIGRAD, mmi_Rerr

% ENSMLP
  
g = -mmigrad(net, 0);