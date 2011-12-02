function W = CFLML( varargin)
% Closed-Form Local Metric Learning, Jianbo YE(jbye@cs.hku.hk)
addpath('knnsearch');% knn lib
%% parse input and initialization

error(nargchk(2, 5, nargin));

X = varargin{1}; % dataset
[num, dim] = size(X);% size dimension


G = varargin{2};% label info
labels = unique(G);% label set
lnum = length(labels);% number of labels

if nargin >2 % projection dimension
    m = min(varargin{3}, dim);
else
    m = min(lnum,dim);
end

if nargin >3 % k-nearest neighbor
    kn = varargin{4};
else
    kn = m;
end

if nargin >4 % estimated metric
    M = varargin{5};
else
    M = eye(dim,dim);
end

%% matrix assembly

sigma = zeros(num,1); % estimated neighbor radius
for i=1:lnum
    labelbool = (G == labels(i));
    XL = X(labelbool,:)*M; 
    [idx, D] = knnsearch(XL,[], kn);
    sigma(labelbool) = 2 * mean(D,2).^2;
end
avgsigma = mean(sigma);

ME = zeros(dim,dim);
MC = zeros(dim,dim);
EM = M*M';
for i=1:num
    MDi = zeros(dim,dim);
    wDi = 0;
    MSi = zeros(dim,dim);
    wSi = 0;
    if (sigma(i) < avgsigma * 1E-3)
        continue;
    end
    
    for j=1:num
        if i==j
            continue;
        end
        vector_ij = X(i,:)-X(j,:);
        weight = exp(-(vector_ij*EM*vector_ij')/sigma(i));
        matrix = (vector_ij'*vector_ij);
        if (G(j)~=G(i))
            MDi = MDi + weight * matrix;
            wDi = wDi + weight;
        else
            MSi = MSi + weight * matrix;
            wSi = wSi + weight;
        end
    end
    
    if wDi/wSi < 1E-5
        continue;
    end
    
    weight = norm(MDi,2)/norm(MDi+MSi);
    
    %[V D] = eig(MDi/wDi - MSi/wSi);
    %D(D<0) = 0;
    %ME = ME + weight * V*D*V';
    ME = ME + MDi/wDi - MSi/wSi;
    MC = MC + weight * (MDi + MSi)/(wDi + wSi);
end

[W D] = eig(ME, MC);
[D, IDX] = sort(diag(D),'descend');
disp(D/D(1));%disp(sum(D(1:m)));
W = W(:,IDX(1:m));
end

