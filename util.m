[trainset, testset] = readset('data/bals.data');
numberoftestinstance = size(testset,1);
numberoftraininstance = size(trainset,1);
%% knn classification
%W = eye(size(trainset,2)-1);
tic;
W = CFLML(trainset(:,1:end-1),trainset(:,end), 3, 10);
W = W{end};


%dlmwrite('metric.mat',W);


knearest = 10;
testclass = knnclassify(testset(:,1:end-1)*W,trainset(:,1:end-1)*W ,trainset(:,end),knearest);

knnerr = 1 - sum(testclass == testset(:,end))/numberoftestinstance;
strtmp = sprintf('k:%d\terr:%.2f%%', knearest, 100*knnerr);
disp(strtmp);
toc;
