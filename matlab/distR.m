function [dist, vecQ, vecR, diff_vecQ_vecR] = distR(ensQ, smoothR, CovarQ, InverseR)
% DISTR Computes the distance with respect to R
% FORMAT
% DESC computes the Mahalanobis distance for R.
% ARG ensQ : ensemble learning distribution ~ N(vecQ, Q)
% ARG smoothR : smoothing distribution ~ N(vecR, R)
% RETURN dist : the mahalobis distrance.
%
% COPYRIGHT : Mehdi Azzouzi, 1998, 1999
%
% SEEALSO : distsum
  
% ENSMLP

% Extract the means of Q and R
vecQ = [ensQ.w1(:); ensQ.b1'; ensQ.w2(:); ensQ.b2'];
vecR = [smoothR.w1(:); smoothR.b1'; smoothR.w2(:); smoothR.b2'];
diff_vecQ_vecR = vecQ - vecR;

% Compute the distance with respect to R distribution
dist = (diff_vecQ_vecR)' * InverseR * (diff_vecQ_vecR);
