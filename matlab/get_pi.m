function [init] = get_pi(z)

% GET_LAMBDA Function to get pi given the exp parameter z.
% FORMAT
% DESC returns lambda given the exponential parameter z.
% ARG z : the argument of the exponential.
% RETURN pi : the output of the softmax.
%
% COPYRIGHT : Mehdi Azzouzi, 1998
%
% SEEALSO : get_lambda
  
% ENSMLP


z =z - max(z);
init = exp(z) ./ sum(exp(z));
