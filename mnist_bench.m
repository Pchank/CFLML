%% configuration
load('data/mnist.mat');
trainset = reshape(train_images, 28*28, 60000)';
trainlabel = train_labels;
testset  = reshape(test_images, 28*28, 10000)';
testlabel = test_labels;

projdim = 128;
knearest = 3;
EMitermax = 10;

repeattime = 1;

%% evaluation bench
tic;
knnerr = [];
eucknnerr = [];
for i = 1:repeattime;
    % data input and cut
    dimension = size(trainset,2);
    
    numberoftestinstance = size(testset,1);
    numberoftraininstance = size(trainset,1);
    %% knn classification
    % CFLML 
    %[M MIDX R RC]= CFLML(trainset, trainlabel, projdim, knearest, eye(dimension), EMitermax);    
    %testclass = knnclsmm(testset, R, RC, knearest, MIDX, M);
    %knnerr(end+1) = 1 - sum(testclass == testlabel)/numberoftestinstance;
    
    % Euclidean
    euctestclass = knnclassify(testset, trainset, trainlabel, knearest);      
    eucknnerr(end+1) = 1 - sum(euctestclass == testlabel)/numberoftestinstance;
end

strtmp = sprintf('k:%d\nEM-CFLML err:%.2f(%.2f)%%\nEuclidean err:%.2f(%.2f)%%', knearest, ...
    100*mean(knnerr), 100*std(knnerr), 100*mean(eucknnerr), 100*std(eucknnerr));
disp(strtmp);
toc;
