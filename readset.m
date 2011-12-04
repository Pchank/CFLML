function [trainset, testset] = readset(name, prop)
% initialization
data=dlmread(name); % last attribute is the class number

classes=unique(data(:,end));
numberofclasses = length(classes);
%numberofinstance = size(data,1);
dimension = size(data,2) - 1;

%divide trainset and testset
trainset = zeros(0,dimension+1);
testset = zeros(0,dimension+1);
for i=1:numberofclasses
    classset = data(data(:,end)==classes(i),:);
    sizeofclassdata = size(classset,1);
    
    trainsampleidx = randsample(sizeofclassdata, floor(prop*sizeofclassdata));
    trainset = [trainset; classset(trainsampleidx,:)];
    classset(trainsampleidx,:) = [];
    testset  = [testset; classset];
end
end