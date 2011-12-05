function [C idxmetric] = knnclsmm( test, train, group, k,  MIDX, M)
% KNNCLSMM knn classifier for multiple metric
addpath('knnsearch');
addpath('count_unique');

nt = size(test,1);
testclasscandidate = zeros(nt, length(M));
testclasscandidateweight = zeros(nt, length(M));
C = zeros(nt,1);

for i=1:length(M)
    idx = knnsearch(test*M{i}, train*M{i}, k);
    for j=1:nt
        [class numclass] = count_unique(group(idx(j,:)));
        [notcare, idxcls] = max(numclass);
        testclasscandidate(j,i) = class(idxcls);
        testclasscandidateweight(j,i) = sum(MIDX(idx(j,:)) == i);
    end
end

[notcare, idxmetric] = max(testclasscandidateweight, [], 2);
for j=1:nt
    C(j) = testclasscandidate(j,idxmetric(j));
end
end

