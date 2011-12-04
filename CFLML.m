function [M MIDX X G]= CFLML( varargin)
% Closed-Form Local Metric Learning, Jianbo YE(jbye@cs.hku.hk)
addpath('knnsearch');% knn lib
addpath('count_unique');
%% parse input and initialization

error(nargchk(2, 6, nargin));

TotalData = varargin{1}; % dataset


TotalLabel = varargin{2};% label info
labels = unique(TotalLabel);% label set
lnum = length(labels);% number of labels

[X, Xidx] = cutset(TotalData, TotalLabel, .85);

G = TotalLabel(Xidx);
TotalData(Xidx,:) = []; TotalLabel(Xidx) = [];


[num, dim] = size(X);% size dimension

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
MIDX = ones(num,1); % metric label of instance
active = true(num,1); % active tag of instance
updated = false(num,1); % updated tag of instance


validtesterr_backtrace = 1;

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
        XR = X(labelbool,:)*M{end};
        
        XQ = XR(active(allinstance(labelbool)),:);
        [notcareidx, D] = knnsearch(XQ, XR , kn);
        sigmanew(labelbool&active) = 2 * mean(D,2).^2;
    end
    %avgsigmametric{count} = mean(sigmanew(active));
    
    for i=allinstance(active) % only to update active instance
        
        if (sigmanew(i) < 1E-10 )%< avgsigmametric{count} * 1E-5)
            updated(i) = true;
            prob(i) = 0;
            MIDX(i) = count;
            active(i) = false;
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
            MIDX(i) = count;
        end
        
        if (weight < 1E-10)
            active(i) = false;
        end
        
        %if (weight < 1E-5 || 1-weight < 1E-5) % inner point or noise point
        %    active(i) =false;
        %end
    end
    %%
    %[metrixidx, asscount] = count_unique(metric);
    %disp([metrixidx'; asscount']);
    
    
    sigma(updated) = sigmanew(updated);
    
    for i=allinstance(active)
        
        MDi = zeros(dim,dim);
        wDi = 0;
        MSi = zeros(dim,dim);
        wSi = 0;
        EM = M{MIDX(i)}*M{MIDX(i)}';
        neighborupdate(EM, sigma(i));
        
        weight = wDi/(wDi+wSi);
        
        ME = ME + weight * (MDi/wDi - MSi/wSi);
        MC = MC + weight * (MDi + MSi)/(wDi + wSi);
        
    end
    
    
    validclass = knnclsmm(TotalData, X, G, kn, MIDX, M);
    validtesterr = 1 - mean(validclass == TotalLabel);
    if (validtesterr >= validtesterr_backtrace && count ~=1) % overfitting!
        MIDX = MIDX_backtrace;
        break;
    else
        strtmp=sprintf('%.2f\t%.2f', sum(log(1-prob)), 100*validtesterr);
        disp(strtmp);
        MIDX_backtrace = MIDX;
        validtesterr_backtrace = validtesterr;
    end
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
        for j= 1:num
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

