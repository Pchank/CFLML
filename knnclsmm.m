function [C idxmetric] = knnclsmm(varargin)
test = varargin{1};
train = varargin{2};
group = varargin{3};
k = varargin{4};
MIDX = varargin{5}; 
M = varargin{6};
if nargin>6
    TNR = varargin{7};
end

% KNNCLSMM knn classifier for multiple metric

nt = size(test,1);
testclasscandidate = zeros(nt, length(M));
testclasscandidateweight = zeros(nt, length(M));
C = zeros(nt,1);

for i=1:length(M)
    if (nargin>6) 
        idx = knnsearch(test*M{i}, train*M{i}, k, TNR);
    else
        idx = knnsearch(test*M{i}, train*M{i}, k);
    end
    
    for j=1:nt
        [class numclass] = count_unique(group(idx(j,:)));
        [~, idxcls] = max(numclass);
        testclasscandidate(j,i) = class(idxcls);
        testclasscandidateweight(j,i) = sum(MIDX(idx(j,:)) == i);
    end
end

[~, idxmetric] = max(testclasscandidateweight, [], 2);
for j=1:nt
    C(j) = testclasscandidate(j,idxmetric(j));
end
end

