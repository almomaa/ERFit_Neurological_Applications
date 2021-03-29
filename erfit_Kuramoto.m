function [vec, tol] = erfit_Kuramoto(xdot,X,I,varargin)

% y : the Ith node dynamic (\dot{x_i})
% X : state vectors of all nodes
% I : index of the target node
% tol : tolerence for the ER
% vec : The output with nonzero entries corrisponding to links

p = inputParser;
p.addRequired('xdot');
p.addRequired('X');
p.addRequired('I');

p.addParameter('alpha',0.95);
p.addParameter('numPerm',1000);

p.parse(xdot,X,I,varargin{:});
options = p.Results;

xdot = options.xdot;
X    = options.X;
I    = options.I;


[m,N] = size(X);

% Function Library (basis, expansion, dictionary, ....)
A = [ones(m,1) sin(bsxfun(@minus, X,X(:,I)))];



% Initialize the selected to include all nodes
IX = 1:N+1;
% note that in the current approach we skip the forward step in ER and we
% only apply the backward elimination. 
% This approach is expensive... but also more accurate


% registration function... the least square fitting
regFun = @(x) x*pinv(x)*xdot;



% Backward ER
val = -inf; ix = [];
while (val < 0) 
    IX(ix) = [];    % remove the weak function      
    D = inf(1,length(IX));
    for i=1:length(D)
        rem = setdiff(IX,IX(i));
        D(i) = cmiVP(regFun(A(:,IX)),xdot,regFun(A(:,rem)));
    end
    [val,ix] = min(D);    %find the weakest influencer
    if ix==1, break; end  %if the constant term is the weakest then break
                          %coz it indicate high degree of uncertinity
end

%Return binary row vector of size N, with 1 indicating index relevence and 0
%otherwise
vec = zeros(1,N);
vec(IX(2:end)-1) = D(2:end);


%find the fits of all the non-connections

for i = setdiff(1:N+1,[I+1,IX])
    vec(i-1) = cmiVP(regFun(A(:,[IX,i])),xdot,regFun(A(:,IX)));
end

P = zeros(1,options.numPerm);
for i=1:length(P)
    P(i) = cmiVP(xdot,xdot(randperm(m)),regFun(A(:,IX)));
end
P = sort(P);
tol = P(ceil(options.alpha*options.numPerm));
end


function [I_est] = cmiVP(x,y,z,varargin)

distInfo = {'minkowski',Inf}; K = 1;
if nargin > 3
    if isnumeric(varargin{1})
        K = varargin{1};
        
    elseif strcmp(varargin{1},'seuclidean') || ...
            strcmp(varargin{1},'mahalanobis') || ...
            strcmp(varargin{1},'minkowski')
        
        distInfo = {varargin{1},varargin{2}};
        if nargin > 5
            K = varargin{3};
        end
    else
       distInfo = varargin(1);
       if nargin > 4
           K = varargin{2};
       end
    end
end

% If the condition set $z$ is empty, then use the Mutual inforation
% estimator.
if isempty(z)
    [I_est] = miKSG(x,y,options);
    return
end
 
% To construct the Joint Space between all variables we have:
JS = cat(2,x,y,z);
% Find the K^th smallest distance in the joint space JS = (x,y,z)
D = pdist2(JS,JS,distInfo{:},'Smallest',K+1)';
epsilon = D(:,end);
% Instead of the above two lines, the one may use the knnsearch function,
% but we found the above implementation is faster.

% Find number of points from $(x,z), (y,z)$, and $(z,z)$ that lies withing the
% K^{th} nearest neighbor distance from each point of themself.
Dxz = pdist2([x,z],[x,z],distInfo{:});
nxz = sum(bsxfun(@lt,Dxz,epsilon),2) - 1;

Dyz = pdist2([y,z],[y,z],distInfo{:});
nyz = sum(bsxfun(@lt,Dyz,epsilon),2) - 1;

Dz = pdist2(z,z,distInfo{:});
nz = sum(bsxfun(@lt,Dz,epsilon),2) - 1;

% VP Estimation formula.
I_est = psi(K) - mean(psi(nxz+1)+psi(nyz+1)-psi(nz+1)); 
end

function I = miKSG(x,y,varargin)
% Initialize and verify inputs.
distInfo = {'minkowski',Inf}; k = 2;
if nargin > 2
    if isnumeric(varargin{1})
        k = varargin{1};
        
    elseif strcmp(varargin{1},'seuclidean') || ...
            strcmp(varargin{1},'mahalanobis') || ...
            strcmp(varargin{1},'minkowski')
        
        distInfo = {varargin{1},varargin{2}};
        if nargin > 4
            k = varargin{3};
        end
    else
       distInfo = varargin(1);
       if nargin > 3
           k = varargin{2};
       end
    end
end


% To construct the Joint Space between all variables we have:
JS = [x,y];  n = size(JS,1);


% Find the K^th smallest distance in the joint space JS
D = pdist2(JS,JS,distInfo{:},'Smallest',k+1)';
epsilon = D(:,end); %Set threshold value

% Find points on x with pairwise distance
% less than threshold value
Dx = pdist2(x,x,distInfo{:});
nx = sum(bsxfun(@lt,Dx,epsilon),2) - 1;

% Find points on y with pairwise distance
% less than threshold value
Dy = pdist2(y,y,distInfo{:});
ny = sum(bsxfun(@lt,Dy,epsilon),2) - 1;

% KSG Estimation formula.
I = psi(k) + psi(n) - mean(psi(nx+1)+psi(ny+1));
end
