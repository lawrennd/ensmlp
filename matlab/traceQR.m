function [tr, CovarQ, CovarR, InverseR] = traceQR(ensQ, smoothR, CovarQ, InverseR)
% TRACEQR Computes the trace of (Q*inv(R))
% FORMAT 
% DESC Computes the trace of Q*inv(R) and returns some parameters needed
% in other codes.          
% ARG ensQ : is an ensemble learning distribution ~ N(vecQ, Q)
% ARG smoothR : is a smooth distribution ~ N(vecR, R)
% ARG CovarQ : 
% ARG InverseR :
% RETURN tr : trace of Q*inv(R).
%          See also:
% SEEALSO : mmi, mixensmixmstep
%          
% COPYRIGHT : Mehdi Azzouzi and Neil D Lawrence, 1998, 1999

% ENSMLP
%/~
% Extract the different covariances and their inverse
%[CuuQ, CvvQ, CuvQ, CovarQ] = enscovar(ensQ);
%[CuuR, CvvR, CuvR, CovarR] = smoothcovar(smoothR);
%[IuuR, IvvR, IuvR, InverseR] = smoothcovinv(smoothR);

% Extract the means of Q and R
%vecQ = [ensQ.w1(:); ensQ.b1'; ensQ.w2(:); ensQ.b2'];
%vecR = [smoothR.w1(:); smoothR.b1'; smoothR.w2(:); smoothR.b2'];
%~/
  
% Get the trace
switch smoothR.covstrct
 case 'diag'
  tr = sum(diag(CovarQ).*diag(InverseR));
 otherwise
  % Should be sum(sum(CovarQ.*InverseR)) for efficiency.
  tr = trace(CovarQ*InverseR);
end