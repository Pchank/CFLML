[trainset, testset] = readset('data/lett.data');
numberoftestinstance = size(testset,1);
numberoftraininstance = size(trainset,1);
%% knn classification
tic;
W = CFLML(trainset(:,1:end-1),trainset(:,end), 5, 1);
%W = eye(size(trainset,2)-1);
toc;

knearest = 1;
testclass = knnclassify(testset(:,1:end-1)*W,trainset(:,1:end-1)*W ,trainset(:,end),knearest);

knnerr = 1 - sum(testclass == testset(:,end))/numberoftestinstance;
disp(knnerr);