[trainset, testset] = readset('data/wine.data');
numberoftestinstance = size(testset,1);
numberoftraininstance = size(trainset,1);
%% knn classification
%W = eye(size(trainset,2)-1);
tic;
%W = CFLML(trainset(:,1:end-1),trainset(:,end), 3, 10);
%W = W{end};
toc;

%dlmwrite('metric.mat',W);
%W = W(:,1:3);

knearest = 10;
testclass = knnclassify(testset(:,1:end-1)*W,trainset(:,1:end-1)*W ,trainset(:,end),knearest);

knnerr = 1 - sum(testclass == testset(:,end))/numberoftestinstance;
disp(knnerr);

