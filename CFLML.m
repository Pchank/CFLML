function [M MIDX X G]= CFLML( varargin)
% Closed-Form Local Metric Learning, Jianbo YE(jbye@cs.hku.hk)
addpath('knnsearch');% knn lib
addpath('count_unique');
%% parse input and initialization

error(nargchk(2, 6, nargin));

TotalData = varargin{1}; % dataset
TotalLabel = varargin{2};% label info
dim = size(TotalData,2);

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

labels = unique(TotalLabel);% label set
lnum = length(labels);% number of labels

if iteration == 1 % no EM steps, no validation set
    [X, Xidx] = cutset(TotalData, TotalLabel, 1);    
else
    [X, Xidx] = cutset(TotalData, TotalLabel, .85);
end

G = TotalLabel(Xidx);
TotalData(Xidx,:) = []; TotalLabel(Xidx) = [];


num = size(X,1);% size dimension


sigma = zeros(num,1); % estimated neighbor radius of instance
sigmanew = zeros(num,1); % estimated neighbor radius of instance w.r.t new metric
prob = ones(num,1); % neighbor emphasis weight of instance
MIDX = ones(num,1); % metric label of instance
active = true(num,1); % active tag of instance
updated = false(num,1); % updated tag of instance
%XwDi = zeros(num,1);
%XwSi = zeros(num,1);

neighborsize = min(num,kn*100);

validtesterr_backtrace = 1;


strtmp = sprintf('EM-CFLML\tnca(log)\tvalid(%%)');
disp(strtmp);

XNR = knnsearch(X*M{1}, [] ,neighborsize);

for count = 1:iteration+1
    % initialize neighbor radius
    %gradientcount = 1;
    %for newton = 1:gradientcount
    
    ME = zeros(dim,dim);
    MC = zeros(dim,dim);
    
    allinstance = 1:num;
    
    Y{count} = X*M{count};
    
    
    for iclass=1:lnum
        labelbool = (G == labels(iclass));
        XR = Y{count}(labelbool,:);%X(labelbool,:)*M{end};        
        XQ = XR(active(allinstance(labelbool)),:);
        [dannii, D] = knnsearch(XQ, XR , kn+1); % degenerate preventation
        sigmanew(labelbool&active) = 2 * mean(D,2).^2;
    end

    %avgsigmanew = mean(sigmanew(active));
    
    for i=allinstance(active) % only to update active instance
        
        if (sigmanew(i) < 1E-10 )%< avgsigmametric{count} * 1E-5)
            updated(i) = true;
            prob(i) = 0;
            MIDX(i) = count;
            active(i) = false;
            continue;
        end       
        
        neighborweightupdate(count, sigmanew(i)); % init
        
        weight = wDi/(wDi+wSi);
        
        
        
        if (weight > prob(i)) % non-interest metric for instance i
            updated(i) = false;
        else
            updated(i) = true;
            prob(i) = weight;            
            MIDX(i) = count;
        end
        
        if (wDi < 1E-10 || weight < 1E-10)
            active(i) = false;
        end
        
    end

    %[metrixidx, asscount] = count_unique(metric);
    %disp([metrixidx'; asscount']);
    
    
    sigma(updated) = sigmanew(updated);
    %% matrix assembly   
    for i=allinstance(active)
        
        neighborupdate(MIDX(i), sigma(i));
        
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
        strtmp=sprintf('Iteration %d\t%.2f\t\t%.2f', count, sum(log(1-prob)), 100*validtesterr);
        disp(strtmp);
        MIDX_backtrace = MIDX;
        validtesterr_backtrace = validtesterr;
    end
    [W D] = eig(ME, MC);
    
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


    function neighborweightupdate(midx, sg)
        IDD = XNR(i,:);        
        YD = Y{midx}(IDD(G(IDD)~=G(i)),:);
        YS = Y{midx}(IDD(G(IDD)==G(i)),:);                
        
        YD = repmat(Y{midx}(i,:),size(YD,1),1) - YD;
        YS = repmat(Y{midx}(i,:),size(YS,1),1) - YS;
        wDi = sum(exp(-sum(YD.^2,2)/sg));
        wSi = sum(exp(-sum(YS.^2,2)/sg));       
    end

    function neighborupdate(midx, sg)
        IDD = XNR(i,:);
        DTAG = G(IDD)~=G(i);
        STAG = G(IDD)==G(i);
        YD = Y{midx}(IDD(DTAG),:);
        YS = Y{midx}(IDD(STAG),:);
        sizeD = size(YD,1);
        sizeS = size(YS,1);
        
        YD = repmat(Y{midx}(i,:),sizeD,1) - YD;
        YS = repmat(Y{midx}(i,:),sizeS,1) - YS;

        WMD = exp(-sum(YD.^2,2)/sg);
        WMS = exp(-sum(YS.^2,2)/sg);        
        wDi = sum(WMD);
        wSi = sum(WMS);
        
        XD = X(IDD(DTAG),:);
        XS = X(IDD(STAG),:);
        
        %ZD = (WMD/wDi)'*XD;
        ZS = (WMS/wSi)'*XS;

        XD = repmat(ZS,sizeD,1) - XD;
        XS = repmat(ZS,sizeS,1) - XS;
        
        %ZD = ZD - X(i,:);
        %ZS = ZS - X(i,:);
        
        %lDi = exp(-sum(ZD.^2)/sg);
        %lSi = exp(-sum(ZS.^2)/sg);
               

        MDi = XD'*(repmat(WMD, 1, dim).*XD);
        MSi = XS'*(repmat(WMS, 1, dim).*XS);
        %MDi = wDi*(ZD'*ZD);
        %MSi = wSi*(ZS'*ZS);
        
        

    end
end

