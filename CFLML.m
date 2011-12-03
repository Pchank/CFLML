function M = CFLML( varargin)
% Closed-Form Local Metric Learning, Jianbo YE(jbye@cs.hku.hk)
addpath('knnsearch');% knn lib
%% parse input and initialization

error(nargchk(2, 6, nargin));

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

if nargin >4 % estimated init metric
    M{1} = varargin{5};
else
    M{1} = eye(dim,dim);
end

if nargin >5 
    iteration = varargin{6};
else
    iteration = 1;
end
%% matrix assembly

sigma = zeros(num,1); % estimated neighbor radius of instance
prob = ones(num,1); % neighbor emphasis weight of instance
metric = ones(num,1); % metric label of instance
active = true(num,1); % active tag of instance
updated = true(num,1); % updated tag of instance


for count = 1:iteration
    % initialize neighbor radius    
    for iclass=1:lnum
        labelbool = (G == labels(iclass));
        XR = X(labelbool,:);
        for mc = 1:length(M)
            label2update = labelbool & (metric == mc) & updated;
            XQ = X(label2update,:);
            [notcareidx, D] = knnsearch(XQ*M{mc}, XR*M{mc}, kn);
            sigma(label2update) = 2 * mean(D,2).^2;
        end               
    end
    avgsigma = mean(sigma);
    
    ME = zeros(dim,dim);
    MC = zeros(dim,dim);
    
    
    allinstance = 1:num;

    for i=allinstance(active) % only to update active instance        
        MDi = zeros(dim,dim);
        wDi = 0;
        MSi = zeros(dim,dim);
        wSi = 0;
        
        EM = M{metric(i)}*M{metric(i)}'; % metric associate with instance i
        if (sigma(i) < avgsigma * 1E-5)
            active(i) = false; % neighbor shrink
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
        
        weight = wDi/(wDi+wSi);        
        %weight = norm(MDi,2)/norm(MDi+MSi);
        if (weight < 1E-5 || 1-weight < 1E-5) % inner point or noise point
            active(i) =false;
            continue;
        end
        
        if (weight > prob(i)) % non-interest metric for instance i
            updated = false;
            continue;
        else
            updated = true;
            prob(i) = weight;            
        end               
        
        
        
        %[V D] = eig(MDi/wDi - MSi/wSi);
        %D(D<0) = 0;
        %ME = ME + weight * V*D*V';
        ME = ME + weight * (MDi/wDi - MSi/wSi);
        MC = MC + weight * MSi/wSi; %(MDi + MSi)/(wDi + wSi);
    end
    disp('nca probability: ');
    disp(sum(prob));
    
    [W D] = eig(ME, MC);
    [D, IDX] = sort(diag(D),'descend');
    %disp('component intensity: ');
    %disp(D);%disp(sum(D(1:m)));
    M{end+1} = W(:,IDX(1:m));
end
end

