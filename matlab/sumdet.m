function [suminv, suminv2, detprod] = sumdet(ensQ, smoothR, CovarQ, CovarR)

% SUMDET Computes the determinant of (I+Q*inv(R))
% FORMAT
% DESC returns three parameters: inv(Q + R), inv(I+Q*inv(R)) and det(I+Q*inv(R))
%     
% ARG ensQ : is an ensemble learning distribution ~ N(vecQ, Q)
% ARG smoothR : is a smooth distribution ~ N(vecR, R)
% ARG covarQ : 
% ARG covarR : 
% RETURN suminv : 
% RETURN suminv2 : 
% RETURN detprod : 
%
% COPYRIGHT : Neil D. Lawrence and Mehdi Azzouzi, 1998
%
% SEEALSO : mmi, mixensmixmstep
  
% ENSMLP
  
% Extract the different covariances and their inverse
% Get the identity matrix


switch smoothR.covstrct
 case 'diag'
  
  diag_covarR = diag(CovarR);
  diag_suminv = 1./(diag(CovarQ) + diag_covarR);
  diag_suminv2 = diag_suminv.*diag_covarR;
  suminv = sparse(diag(diag_suminv));
  suminv2 = sparse(diag(diag_suminv2));
  detprod = 1/prod(diag_suminv2);
  
 otherwise 
  Id = eye(size(CovarQ));
  
  % Might be computationnaly expensive for big network
  % In that case, we should consider each block of the matrices
  suminv = Id/(CovarQ + CovarR);
  suminv2 = CovarR*suminv;  
  detprod = 1/det(suminv2);
end
 
