%% configuration
data = dlmread('data/lett.data');
projdim = 16;
knearest = 1;
EMitermax = 10;

repeattime = 1;

%% evaluation bench
tic;
knnerr = [];
eucknnerr = [];
for i = 1:repeattime;
    % data input and cut
    dataset = data(:,1:end-1);
    label = data(:,end);
    [trainset, trainidx] = cutset(dataset, label, .85);
    trainlabel = label(trainidx);
    dataset(trainidx,:) = [];
    label(trainidx) = [];
    testset = dataset;
    testlabel = label;
    dimension = size(trainset,2);
    
    numberoftestinstance = size(testset,1);
    numberoftraininstance = size(trainset,1);
    %% knn classification
    % CFLML 
    [M MIDX R RC]= CFLML(trainset, trainlabel, projdim, knearest, eye(dimension), EMitermax);    
    testclass = knnclsmm(testset, R, RC, knearest, MIDX, M);
    knnerr(end+1) = 1 - sum(testclass == testlabel)/numberoftestinstance;
    
    % Euclidean
    euctestclass = knnclassify(testset, trainset, trainlabel, knearest);      
    eucknnerr(end+1) = 1 - sum(euctestclass == testlabel)/numberoftestinstance;
    
end

strtmp = sprintf('k:%d\nEM-CFLML err:%.2f(%.2f)%%\nEuclidean err:%.2f(%.2f)%%', knearest, ...
    100*mean(knnerr), 100*std(knnerr), 100*mean(eucknnerr), 100*std(eucknnerr));
disp(strtmp);
toc;
