function net = mixensrmnode(net, row, nodes);

% MIXENSRMNODE Remove a node from the network.
% FORMAT
% DESC removes a node from the network.
% ARG net : the network from which the nodes will be removed.
% ARG row : the row from which to remove the nodes.
% ARG nodes : the nodes to remove.
%
% SEEALSO : mixens
%
% COPYRIGHT : Neil D. Lawrence, 1999
  
% ENSMLP

for m = 1:net.M
  net.smooth(m) = smoothrmnode(net.smooth(m), row, nodes);
  net.ens(m) = ensrmnode(net.ens(m), row, nodes);
end
net.nwts = net.ens(1).nwts;
net.npars = net.ens(1).npars;
net.nhidden = net.ens(1).nhidden;
