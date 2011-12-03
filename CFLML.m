function [M metric]= CFLML( varargin)
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
sigmanew = zeros(num,1); % estimated neighbor radius of instance w.r.t new metric
prob = ones(num,1); % neighbor emphasis weight of instance
metric = ones(num,1); % metric label of instance
active = true(num,1); % active tag of instance
updated = false(num,1); % updated tag of instance




for count = 1:iteration+1
    % initialize neighbor radius
    %gradientcount = 1;
    %for newton = 1:gradientcount
    ME = zeros(dim,dim);
    MC = zeros(dim,dim);   
    
    allinstance = 1:num;
%%     
    for iclass=1:lnum
        labelbool = (G == labels(iclass));
        XR = X(labelbool,:);                                 
        [notcareidx, D] = knnsearch(XR*M{end}, [], kn);
        sigmanew(labelbool) = 2 * mean(D,2).^2;
    end    
    avgsigmanew = mean(sigmanew);    

    for i=allinstance(active) % only to update active instance   

        if (sigmanew(i) < avgsigmanew * 1E-5)
            %active(i) = false;
            updated(i) = true;
            %prob(i) = 0;
            %metric(i) = length(M);
            continue;
        end
        
        wDi = 0;        
        wSi = 0;                
        
        EM = M{end}*M{end}'; % test new metric for instance i
        
        neighborweightupdate(EM, sigmanew(i)); % init
        
        weight = wDi/(wDi+wSi);
        

        
        if (weight > prob(i)) % non-interest metric for instance i
            updated(i) = false;
        else
            updated(i) = true;
            prob(i) = weight;
            metric(i) = length(M);            
        end
    end
%%    
                   
    sigma(updated) = sigmanew(updated);    
    avgsigma = mean(sigma);
    
    for i=allinstance(active)

        if (sigma(i) < avgsigma * 1E-5)
            active(i) = false; % neighbor shrink
            continue;
        end
        
        MDi = zeros(dim,dim);
        wDi = 0;
        MSi = zeros(dim,dim);
        wSi = 0;  
        EM = M{metric(i)}*M{metric(i)}';
        neighborupdate(EM, sigma(i));
        
        weight = wDi/(wDi+wSi);         

        if (weight < 1E-5 || 1-weight < 1E-5) % inner point or noise point            
            active(i) =false;
            continue;
        end
        
        ME = ME + weight * (MDi/wDi - MSi/wSi);
        MC = MC + weight * (MDi + MSi)/(wDi + wSi); 
        
    end
    
    strtmp=sprintf('log-nca: %.2f', sum(log(1-prob(active))));
    disp(strtmp);
    
    [W D] = eig(ME, MC);
    %dD = diag(D);
    %dt = 1/ max(-dD(dD<0));
    D(D<0) = 0;
    W= W*D; %% estimated optimal

    
    
    
    %end
    
    [D, IDX] = sort(diag(D),'descend');
    W = W(:,IDX);
    %disp('component intensity: ');
    %disp(D(D>0));%disp(sum(D(1:m)));
    if (count ~= iteration+1) 
        M{end+1} = W(:,1:m);
    end
end


    function neighborweightupdate(metrictest, sg)
        for j=1:num
            vector_ij = X(i,:)-X(j,:);
            weight = exp(-(vector_ij*metrictest*vector_ij')/sg);

            if (G(j)~=G(i))
                wDi = wDi + weight;
            else
                wSi = wSi + weight;
            end
        end
    end

    function neighborupdate(metrictest, sg)
        for j=1:num
            vector_ij = X(i,:)-X(j,:);
            weight = exp(-(vector_ij*metrictest*vector_ij')/sg);
            matrix = (vector_ij'*vector_ij);
            if (G(j)~=G(i))
                MDi = MDi + weight * matrix;
                wDi = wDi + weight;
            else
                MSi = MSi + weight * matrix;
                wSi = wSi + weight;
            end
        end        
    end
end

