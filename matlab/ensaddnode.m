function net = ensaddnode(net, row);

% ENSADDNODE Add a node to a ENS structure.
% FORMAT
% DESC adds a node to an ENS structure.
% ARG net : the network to add a node to.
% ARG row : the row in which to add the node.
%
% COPYRIGHT : Neil D. Lawrence, 1999

% ENSMLP

orignnodes = net.nin + net.nhidden + net.nout + 2;
orignwts = net.nwts;
switch row
 case 1  
 
 case 2 					% Hidden Layer
  initfactor = 0.1;
  nodenum = net.nin + 2;
  remainnodes = 1:net.nhidden;
  remainnodes(nodes) = []; 
  net.nhidden =  net.nhidden+1;
  net.nwts = (net.nin + 1)*net.nhidden + (net.nhidden + 1)*net.nout;
  net.w2 = [initialise(1, net.nout, initfactor); net.w2];
  net.w1 = [initialise(net.nin, 1, initfactor) net.w1];
  net.b1 = [initialise(1, 1, initfactor) net.b1];
  switch net.covstrct
    
   case 'none'
    net.npars = net.nwts;
    
   otherwise
    net.d2 = [randn(1, net.nout, initfactor)*.1; net.w2];
    net.d1 = [randn(net.nin, 1, initfactor)*.1 net.w1];
    net.db1 = [randn(1, 1, initfactor)*.1 net.b1];
    
    switch net.covstrct
     case 'diag';
      net.npars = 2*net.nwts; 
     otherwise
      % deal with mus
    end
  end
end  

switch net.alphaprior.type
case 'single'
% do nothing
case 'group'
  % Weights and biases are grouped seperately
  error('Not yet implemented')

case 'ARD'
  % Automatic relevance prior
  error('Not yet implemented')

case 'NRD'
  % 
  numalphas = size(net.alphas, 1);
  net.alphas(nodenum+1:numalphas+1) = net.alphas(nodenum: ...
						 numalphas);
  net.alphas(nodnum)
  net.alphaprior.a(nodenum+1:numalphas+1) = net.alphaprior.a(nodenum: ...
						 numalphas);(remainnodes);
  net.alphaprior.b(nodenum+1:numalphas+1) = net.alphaprior.b(remainnodes);
  net.alphaprior.index(:, nodenum+1:numalphas+1) = net.alphaprior.index(:, remainnodes);
  net.alphaposterior.a(nodenum+1:numalphas+1) = net.alphaposterior.a(remainnodes);
  net.alphaposterior.b(nodenum+1:numalphas+1) = net.alphaposterior.b(remainnodes);
  net.alphaposterior.index(:, nodenum+1:numalphas+1) = net.alphaposterior.index(:, remainnodes);
  remainweights = 1:orignwts;
  weightsnum = find(sum(net.alphaprior.index, 2) == 1);
  remainweights(weightsnum) = [];  
  net.alphaprior.index = net.alphaprior.index(remainweights, :);
  net.alphaposterior.index  = net.alphaposterior.index(remainweights, :);
end
net = enshypermoments(net);
