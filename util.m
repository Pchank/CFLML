[trainset, testset] = readset('data/wine.data');
numberoftestinstance = size(testset,1);
numberoftraininstance = size(trainset,1);
%% knn classification
dimension = size(trainset,2)-1;

tic;
%trainset = [trainset;testset]; %overfit
[M MIDX]= CFLML(trainset(:,1:end-1),trainset(:,end), 5, 10, eye(dimension), 1);
%dlmwrite('metric.mat',M);


knearest = 15;


testclass = knnclsmm(testset(:,1:end-1), trainset(:,1:end-1), ...
    trainset(:,end), knearest, MIDX, M);
%testclass = knnclassify(testset(:,1:end-1)*M{2}, trainset(:,1:end-1)*M{2}, trainset(:,end), knearest);

knnerr = 1 - sum(testclass == testset(:,end))/numberoftestinstance;
strtmp = sprintf('k:%d\terr:%.2f%%', knearest, 100*knnerr);
disp(strtmp);
toc;
