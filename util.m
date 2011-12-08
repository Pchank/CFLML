%% configuration
data = dlmread('data/bals.data');
projdim = 4;
knearest = 5;
EMitermax = 10;
trainprop = .85;

repeattime = 100;

%% evaluation bench
tic;
knnerr = [];
eucknnerr = [];
for i = 1:repeattime;
    % data input and cut
    dataset = data(:,1:end-1);
    label = data(:,end);
    [trainset, trainidx] = cutset(dataset, label, trainprop);
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

Mknnerr = mean(knnerr);
Vknnerr = sqrt((mean(knnerr.^2) - Mknnerr.^2)/(repeattime-1));
Meucknnerr = mean(eucknnerr);
Veucknnerr = sqrt((mean(eucknnerr.^2) - Meucknnerr.^2)/(repeattime-1));

strtmp = sprintf('k:%d\nEM-CFLML err:%.2f(%.2f)%%\nEuclidean err:%.2f(%.2f)%%', knearest, ...
    100*Mknnerr, 100*tinv(.99,repeattime)*Vknnerr, ...
    100*Meucknnerr, 100*tinv(.99,repeattime)*Veucknnerr);
disp(strtmp);
toc;
