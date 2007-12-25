function [l] = get_lambda(z)

% GET_LAMBDA Function to get lambda given the exp parameter z.
% FORMAT
% DESC returns lambda given the exponential parameter z.
% ARG z : the argument of the exponential.
% RETURN lambda : the output of the exponential.
%
% COPYRIGHT : Mehdi Azzouzi, 1998
%
% SEEALSO : get_pi
  
% ENSMLP
  
z = z.*(z<7e2)+(7e2*(z>=7e2));
l = exp(z);
