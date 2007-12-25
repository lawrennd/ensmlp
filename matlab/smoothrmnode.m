function net = smoothrmnode(net, row, nodes);

% SMOOTHRMNODE Remove a node from the smoothing distribution.
% FORMAT
% DESC removes a node from the smoothing distribution when pruning is
% taking place.
% ARG net : the network to remove a node from.
% ARG row : the row from which to remove the node.
% ARG nodeNumbers : the node indices that are to be removed.
% RETURN net : the network with the node(s) removed.
%
% COPYRIGHT : Neil D. Lawrence, 1998, 1999
% 
% SEEALSO : mixensrmnode
  
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
  biastwo = net.nhidden +1;
  net.nhidden = net.nhidden-length(nodes);
  net.nwts = (net.nin + 1)*net.nhidden + (net.nhidden + 1)*net.nout;
  net.w2 = net.w2(remainnodes, :);
  net.w1 = net.w1(:, remainnodes);
  net.b1 = net.b1(:, remainnodes);
  switch net.covstrct

   case 'none'
    net.npars = net.nwts;

   otherwise
    net.d2 = net.d2([remainnodes biastwo], :);
    net.d1 = net.d1(:, remainnodes);

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

