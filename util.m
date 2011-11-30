%% initialization
clear;
addpath('knnsearch');
data=dlmread('data/iris.data'); % last attribute is the class number

classes=unique(data(:,end));
numberofclasses = length(classes);
numberofinstance = size(data,1);
dim = size(data,2) - 1;

%divide trainset and testset
trainset = zeros(0,dim+1);
testset = zeros(0,dim+1);
for i=1:numberofclasses
    classset = data(data(:,end)==classes(i),:);
    sizeofclassdata = size(classset,1);
    
    trainsampleidx = randsample(sizeofclassdata, floor(.7*sizeofclassdata));
    trainset = [trainset; classset(trainsampleidx,:)];
    classset(trainsampleidx,:) = [];
    testset  = [testset; classset];
end
numberoftestinstance = size(testset,1);
numberoftraininstance = size(trainset,1);

%% knn classification
knearest = 10;
testclass = knnclassify(testset(:,1:end-1),trainset(:,1:end-1),trainset(:,end),knearest);

knnerr = 1 - sum(testclass == testset(:,end))/numberoftestinstance;
disp(knnerr);




