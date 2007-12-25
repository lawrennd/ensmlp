function net = ensrmnode(net, row, nodes);

% ENSRMNODE Remove a node from the network.
% FORMAT
% DESC removes a node from the network.
% ARG net : the network from which the nodes will be removed.
% ARG row : the row from which to remove the nodes.
% ARG nodes : the nodes to remove.
%
% SEEALSO : ens, ensaddnode
%
% COPYRIGHT : Neil D. Lawrence, 1999
  
% ENSMLP
  
orignnodes = net.nin + net.nhidden + net.nout + 2;
orignwts = net.nwts;
switch row
case 1  
  nodenum = nodes;
  remainnodes = 1:net.nin;
  remainnodes(nodes) = []; 
  net.nin = net.nin-length(nodes);
  net.nwts = (net.nin + 1)*net.nhidden + (net.nhidden + 1)*net.nout;
  net.w1 = net.w1(remainnodes, :);
  
  switch net.covstrct

  case 'none'
    net.npars = net.nwts;

  otherwise
    net.d1 = net.d1(remainnodes, :);

    switch net.covstrct
    case 'diag';
      net.npars = 2*net.nwts; 
    otherwise
      % deal with mus
    end
  end

case 2 					% Hidden Layer
  nodenum = nodes + net.nin + 1;
  remainnodes = 1:net.nhidden;
  remainnodes(nodes) = []; 
  net.nhidden = net.nhidden-length(nodes);
  net.nwts = (net.nin + 1)*net.nhidden + (net.nhidden + 1)*net.nout;
  net.w2 = net.w2(remainnodes, :);
  net.w1 = net.w1(:, remainnodes);
  net.b1 = net.b1(:, remainnodes);
  switch net.covstrct

  case 'none'
    net.npars = net.nwts;

  otherwise
    net.d2 = net.d2(remainnodes, :);
    net.d1 = net.d1(:, remainnodes);
    net.db1 = net.db1(:, remainnodes);

    switch net.covstrct
    case 'diag';
      net.npars = 2*net.nwts; 
    otherwise
      % deal with mus
    end
  end
  
case 3 					% Output layer
  nodenum = nodes + net.nin + net.nhidden + 1;
  remainnodes = 1:net.nout;
  remainnodes(nodes) = []; 
  net.betaprior.a = net.betaprior.a(remainnodes);
  net.betaprior.b = net.betaprior.b(remainnodes);
  net.betaposterior.a = net.betaposterior.a(remainnodes);
  net.betaposterior.b = net.betaposterior.b(remainnodes);
  net.nout = net.nout-length(nodes);
  net.nwts = (net.nin + 1)*net.nhidden + (net.nhidden + 1)*net.nout;
  net.w2 = net.w2(remainnodes, :);
  net.b2 = net.b2(remainnodes, :);
  switch net.covstrct

  case 'none'
    net.npars = net.nwts;

  otherwise
    net.npars = net.nwts + (net.nin + 1)*net.nhidden*(net.t + 1) + ...
	(net.nhidden+ 1)*net.nout*(net.t + 1); 
    net.d2 = net.d2(remainnodes, :);
    net.db2 = net.db2(remainnodes, :);

    switch net.covstrct
    case 'diag';
      net.npars = 2*net.nwts; 
    otherwise
      % deal with mus
    end
  end
end  
remainnodes = 1:orignnodes;
remainnodes(nodenum) = [];
nin = net.nin;
nhidden = net.nhidden;
nout = net.nout;
nextra = nhidden + (nhidden + 1)*nout;
nwts = nin*nhidden + nextra;

switch net.alphaprior.type
 case 'single'
  % do nothing
 case {'group', 'ARD'}
  
  switch net.alphaprior.type
   case 'group' 
    indx = [ones(1, nin*nhidden), zeros(1, nextra)]';
  
   case 'ARD'
    indx = kron(ones(nhidden, 1), eye(nin));
    indx = [indx; zeros(nextra, nin)];
    switch row
     case 1
      net.alphaprior.a = net.alphaprior.a(remainnodes);
      net.alphaprior.b = net.alphaprior.b(remainnodes);
      net.alphaposterior.a = net.alphaposterior.a(remainnodes);
      net.alphaposterior.b = net.alphaposterior.b(remainnodes);
    end
    
  end

  extra = zeros(nwts, 3);
  
  mark1 = nin*nhidden;
  mark2 = mark1 + nhidden;
  extra(mark1 + 1:mark2, 1) = ones(nhidden,1);
  mark3 = mark2 + nhidden*nout;
  extra(mark2 + 1:mark3, 2) = ones(nhidden*nout,1);
  mark4 = mark3 + nout;
  extra(mark3 + 1:mark4, 3) = ones(nout,1);
  
  indx = [indx, extra];
  
  net.alphaprior.index = indx;
  net.alphaposterior.index = indx;
 
 case 'NRD'
  % 
  net.alphaprior.a = net.alphaprior.a(remainnodes);
  net.alphaprior.b = net.alphaprior.b(remainnodes);
  net.alphaprior.index = net.alphaprior.index(:, remainnodes);
  net.alphaposterior.a = net.alphaposterior.a(remainnodes);
  net.alphaposterior.b = net.alphaposterior.b(remainnodes);
  net.alphaposterior.index = net.alphaposterior.index(:, remainnodes);
  remainweights = 1:orignwts;
  weightsnum = find(sum(net.alphaprior.index, 2) == 1);
  remainweights(weightsnum) = [];  
  net.alphaprior.index = net.alphaprior.index(remainweights, :);
  net.alphaposterior.index  = net.alphaposterior.index(remainweights, :);
end
net = enshypermoments(net);