function [M MIDX X G]= CFLML( varargin)
% Multiple Closed-Form Local Metric Learning (in progress)
% Author: Jianbo YE(jbye@cs.hku.hk), Dept of Computer Science, HKU
% Examples


INNER_CUT = .1;%1E-3; % to cut inner point within class
MAX_TRACE_ITER = 3; % max trace iteration
PENDING_PROB = .99;


% KNN search is very crucial for the efficient implement of this algorithm
%addpath('knnsearch');% knn implement path
%addpath('count_unique');% useful routine to obtain count of unique;

% Parse input arguments
error(nargchk(2, 6, nargin));

TotalData = varargin{1}; % dataset
TotalLabel = varargin{2};% label info
dim = size(TotalData,2);
labels = unique(TotalLabel);% label set

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


% cut data into trainset and validset
if iteration <= 3 % EM steps less than 3, no validation set
    [X, Xidx] = cutset(TotalData, TotalLabel, 1);    
else
    [X, Xidx] = cutset(TotalData, TotalLabel, .85);
end

% initialization part starts here
G = TotalLabel(Xidx); %labels for trainset
TotalData(Xidx,:) = []; TotalLabel(Xidx) = [];


num = size(X,1);% size dimension
vnum = size(TotalData,1);


sigma = zeros(num,1); % estimated neighbor radius of instance
sigmanew = zeros(num,1); % estimated neighbor radius of instance w.r.t new metric
prob = ones(num,1); % neighbor emphasis weight of instance
MIDX = ones(num,1); % metric label of instance
active = true(num,1); % active tag of instance

% size of neighbor estimated. For large size of data, a small neighbor is
% enough, like 10 times k-nearests.
neighborsize = min([num,max(kn*100,700)]); 
%XwDi = zeros(num, neighborsize);
%XwSi = zeros(num, neighborsize);

% backtrace init
validtesterr_backtrace = 1;
stepsout_backtrace = MAX_TRACE_ITER; % max backtrace iteration
stepsout_count = 0;
validcandidateidx = cell(0);
validclasscandidate = zeros(vnum, 0);
probtrace = ones(num,1);




% apply init metric transform
Y{1} = X*M{1};

% bool check for each label
labelbool = false(num, lnum);
for iclass = 1:lnum
    labelbool(:,iclass) = (G == labels(iclass));
end
%labelcount = sum(labelbool);

% knnsearch to contruct neighbor
if neighborsize < num 
    fprintf('0\tNeighborhood Initialization ...');
    tStarted = tic;
    XNR = knnsearch(Y{1}, Y{1} ,neighborsize);
    YNR = knnsearch(TotalData*M{1},Y{1},neighborsize); % for valid evalution
    tElapsed = toc(tStarted);
    fprintf(1,'\t%.2fs\n',tElapsed);
else
    XNR = repmat(1:num, num, 1);
    YNR = repmat(1:num, vnum, 1);
end


% the same(or not) labels of neighbor
ST = G(XNR)==repmat(G, 1, neighborsize);

% status output
fprintf(1,'   NCA(log)\tVE(%%)\tT(s)\tA\tU\tP\n');



for count = 1:iteration+1
    tStart = tic;
    % initialize neighbor radius
    %gradientcount = 1;
    %for newton = 1:gradientcount
    % matrix to assembly
    ME = zeros(dim,dim);
    MC = zeros(dim,dim);

    
    allinstance = 1:num;        

    % compute k-neighbor radius w.r.t new metric    
    for iclass=1:lnum        
        XR = Y{count}(labelbool(:,iclass),:);
        XQ = XR(active(allinstance(labelbool(:,iclass))),:);

        [~, D] = knnsearch(XQ, XR , kn+2);                                       
        sigmanew(labelbool(:,iclass)&active) = 2 * median(D,2).^2;        
    end


    % test if the new metric is more fitted to certain instance
    updated = false(num,1); % updated tag of instance
    nonpending = true(num,1); % pending in matrix assembly
    prob_trace = prob;
    
    for i=allinstance(active) % only to update active instance
        
        if (sigmanew(i) < 1E-10 )
            prob(i) = 0;
            MIDX(i) = count;
            active(i) = false;
            continue;
        end       
       
        neighborweightupdate(count, sigmanew(i)); % init
                                
        
        if (wDi < prob(i)*(wDi+wSi)) % non-interest metric for instance i                    
            updated(i) = true;
            prob(i) = wDi/(wDi+wSi);
            MIDX(i) = count;
        end
        
        % remove inner point and noise point
        if (wDi <= INNER_CUT * (wDi+wSi))        
            active(i) = false;
        end
        
        if (prob(i) > PENDING_PROB) %noise point
            nonpending(i) = false;            
        end
        
    end
    
    
    %[metrixidx, asscount] = count_unique(metric);
    %disp([metrixidx'; asscount']);

    % metric validation start
    %[validclass, valididxmetric] = knnclsmm(TotalData, X, G, kn, MIDX, M, YNR);
    validclass = zeros(vnum,1);
    validclasscandidateweight = zeros(vnum, count);
    for i = 1:count        
        if i==count
            validcandidateidx{count}= knnsearch(TotalData*M{i}, X*M{i}, kn, YNR);                        
            validclasscandidate(:,count) = zeros(vnum,1);
            for j=1:vnum
                [classname numclass] = count_unique(G(validcandidateidx{count}(j,:)));
                [~, idxcls] = max(numclass);
                validclasscandidate(j,count) = classname(idxcls);        
            end
        end
        
        for j = 1:vnum                                    
            validclasscandidateweight(j,i) = sum(MIDX(validcandidateidx{i}(j,:)) == i);
        end
    end
    [~, valididxmetric] = max(validclasscandidateweight, [], 2);
    for j=1:vnum
        validclass(j) = validclasscandidate(j,valididxmetric(j));
    end   
    
    validcorrectbool = validclass == TotalLabel;
    validtesterr = 1 - mean(validcorrectbool);
    tElapsed = toc(tStart);
    fprintf(1,'%d  %.2E\t%.2f\t%.2f\t%d\t%d\t%d\n', ...
        count, sum(log(1-prob(active&nonpending))), 100*validtesterr, tElapsed, ...
        sum(active), sum(updated), sum(1-nonpending));

    if ... %(((sum(log(1-prob)) <= sum(log(1-probtrace)) && validtesterr == validtesterr_backtrace) ||...
            ((validtesterr >= validtesterr_backtrace) && count ~=1) % overfitting!
        stepsout_count = stepsout_count +1;
        MIDX = MIDX_backtrace;
        prob = prob_trace;
        if (stepsout_count >= stepsout_backtrace)                        
            break;
        end
    else        
        MIDX_backtrace = MIDX;
        validtesterr_backtrace = validtesterr;
        validdatatrace = TotalData(validcorrectbool,:);
        validmetrictrace = valididxmetric(validcorrectbool);
        validclasstrace = TotalLabel(validcorrectbool);
        stepsout_count = 0;
    end
    
    if (sum(updated) == 0)
        break;
    end        
    
    
    
    sigma(updated) = sigmanew(updated);
    
    %avgsigma = 2 * max(sigma(active))/9;
    
    
    
    % matrix assembly start  
    for i=allinstance(active&nonpending)
        % to prevent degenerate cases like []
        MDi = zeros(dim,dim);
        MSi = zeros(dim,dim);
        
        %neighborupdate(MIDX(i), sigma(i));
        neighborupdate(count,sigmanew(i));
        
        weight = wDi/(wDi+wSi);        
                
        % the coefficient is carefully designed to achieve most robustness.
        % performance and stability.
        ME = ME + prob(i)/weight * (MDi/(wDi+wSi) - (weight/wSi) * MSi);
        MC = MC + (prob(i) /(wDi + wSi)) * (MDi + MSi);
        
    end
    

    % closed-form estimated of the optimal metric
    [W D] = eig(ME, MC);

    % sort
    [D, IDX] = sort(diag(D),'descend');
    W = W(:,IDX);
    
    % estimated optimal coefficient should be carefully designed
    % disp(D);
    D(D<0) = 0;
    %disp(D);
    %D = D./(1 + (max(D)/10./D).^4);
    W= W*diag(D); 
    

    
    %disp('component intensity: ');
    %disp(D(D>0));%disp(sum(D(1:m)));
    if (count ~= iteration+1)
        M{end+1} = W(:,1:m);
        Y{end+1} = X*M{end};
    end        
    
end

    % append validset to reference set.
    M = M(1:length(M)-stepsout_count);
    X = [X; validdatatrace];    
    MIDX = [MIDX; validmetrictrace];
    G = [G;validclasstrace];
    
    

    function neighborweightupdate(midx, sg)
        IDD = XNR(i,:);        
        YD = Y{midx}(IDD(~ST(i,:)),:);
        YS = Y{midx}(IDD(ST(i,:)),:);     
      
        YD = repmat(Y{midx}(i,:),size(YD,1),1) - YD;
        YS = repmat(Y{midx}(i,:),size(YS,1),1) - YS;
        
        %wDi = sum(exp(-sum(YD.^2,2)/sg));
        %wSi = sum(exp(-sum(YS.^2,2)/sg));       
        wDi = sum(1./(1+4 * sum(YD.^2,2).^2/(sg^2)));
        wSi = sum(1./(1+4 * sum(YS.^2,2).^2/(sg^2)));
    end

    function neighborupdate(midx, sg)
        IDD = XNR(i,:);
        YD = Y{midx}(IDD(~ST(i,:)),:);
        YS = Y{midx}(IDD(ST(i,:)),:);
        
        sizeD = size(YD,1);
        sizeS = size(YS,1);
        
        YD = repmat(Y{midx}(i,:),sizeD,1) - YD;
        YS = repmat(Y{midx}(i,:),sizeS,1) - YS;

        %WMD = exp(-sum(YD.^2,2)/sg);
        %WMS = exp(-sum(YS.^2,2)/sg);
        WMD = 1./(1+4 * sum(YD.^2,2).^2/(sg^2));
        WMS = 1./(1+4 * sum(YS.^2,2).^2/(sg^2));
        
        wDi = sum(WMD);
        wSi = sum(WMS);
        
        XD = X(IDD(~ST(i,:)),:);
        XS = X(IDD(ST(i,:)),:);
        
        %ZD = (WMD/wDi)'*XD;
        ZS = (WMS/wSi)'*XS; %
        %ZS = X(i,:);

        XD = repmat(ZS,sizeD,1) - XD;
        XS = repmat(ZS,sizeS,1) - XS;

        MDi = MDi + XD' * (repmat(WMD, 1, dim).* XD);
        MSi = MSi + XS' * (repmat(WMS, 1, dim).* XS);        
       
    end
end

