function [trainset, trainsetidx] = cutset(data, label, prop)
% initialization

classes=unique(label);
numberofclasses = length(classes);
%numberofinstance = size(data,1);
dimension = size(data,2);

%divide trainset and testset
trainset = [];
trainsetidx = [];
idx = 1:size(data,1);

for i=1:numberofclasses
    classset = data(label==classes(i),:);
    classsetidx = idx(label==classes(i));    
    sizeofclassdata = size(classset,1);
    
    trainsampleidx = randsample(sizeofclassdata, floor(prop*sizeofclassdata));
    trainset = [trainset; classset(trainsampleidx,:)];
    trainsetidx = [trainsetidx; classsetidx(trainsampleidx)'];

end
end