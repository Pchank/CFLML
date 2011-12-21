%% configuration
restoredefaultpath;
setpaths;
data = dlmread('data/wine.data');
projdim = 4;
knearest = 9;
EMitermax = 20;
trainprop = .80;

repeattime = 1;

run = containers.Map(...
    { 'CFLML-1', ...
    'CFLML-3', ...
    'EM-CFLML', ...
    'Euclidean', ...
    'PCA', ...
    'LDA', ...
    'NCA', ...    
    'MCML', ...
    'Boost', ...
    'LMNN' ...
    }, ...
    { true, ... CFLML-1
    true, ... CFLML-3
    true, ... EM-CFLML
    true, ... Euclidean
    true, ... PCA
    true, ... LDA
    true, ...  NCA
    false, ... MCML (!featured)
    true, ... Boost
    true ... LMNN
    });
%% dimension estimation
% fprintf(1,'--------------------------------------\nDIM-EST\n');
% d1 = intrinsic_dim(data(:,1:end-1),'CorrDim');
% d2 = intrinsic_dim(data(:,1:end-1),'NearNbDim');
% d3 = intrinsic_dim(data(:,1:end-1),'GMST');
% d4 = intrinsic_dim(data(:,1:end-1),'EigValue');
% d5 = intrinsic_dim(data(:,1:end-1),'MLE');
% fprintf(1,'\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\n', d1,d2,d3,d4,d5);

%% evaluation bench
knnerr_1 = [];
knnerr_3 = [];
knnerr_em = [];
eucknnerr = [];
pcaknnerr = [];
ldaknnerr = [];
ncaknnerr = [];
mcmlknnerr = [];
boostknnerr = [];
lmnnknnerr = [];

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
    % CFLML-1
    if run('CFLML-1')
        fprintf(1,'--------------------------------------\nCFLML-1\n'); tic;
        [M MIDX R RC]= CFLML(trainset, trainlabel, projdim, knearest, eye(dimension), 1);
        testclass = knnclsmm(testset, R, RC, knearest, MIDX, M);
        knnerr_1(end+1) = 1 - sum(testclass == testlabel)/numberoftestinstance;
        toc;
    end
    % CFLML-3
    if run('CFLML-3')
        fprintf(1,'--------------------------------------\nCFLML-3\n'); tic;
        [M MIDX R RC]= CFLML(trainset, trainlabel, projdim, knearest, eye(dimension), 3);
        testclass = knnclsmm(testset, R, RC, knearest, MIDX, M);
        knnerr_3(end+1) = 1 - sum(testclass == testlabel)/numberoftestinstance;
        toc;
    end
    % EM-CFLML
    if run('EM-CFLML')
        fprintf(1,'--------------------------------------\nEM-CFLML\n'); tic;
        [M MIDX R RC]= CFLML(trainset, trainlabel, projdim, knearest, eye(dimension), EMitermax);
        testclass = knnclsmm(testset, R, RC, knearest, MIDX, M);
        knnerr_em(end+1) = 1 - sum(testclass == testlabel)/numberoftestinstance;
        toc;
    end        
    % Euclidean
    if run('Euclidean')
        fprintf(1,'--------------------------------------\nEuclidean\n');tic;
        euctestclass = knnclassify(testset, trainset, trainlabel, knearest);
        eucknnerr(end+1) = 1 - sum(euctestclass == testlabel)/numberoftestinstance;
        toc;
    end
    % PCA
    if run('PCA')
        fprintf(1,'--------------------------------------\nPCA\n');tic;
        [~, mapping] = compute_mapping(trainset,'PCA',projdim);
        pcatestclass = knnclassify(testset*mapping.M,trainset*mapping.M,trainlabel,knearest);
        pcaknnerr(end+1) = 1 - sum(pcatestclass == testlabel)/numberoftestinstance;
        toc;
    end
    % LDA
    if run('LDA')
        fprintf(1,'--------------------------------------\nLDA\n');tic;
        [~, mapping] = compute_mapping([trainlabel,trainset],'LDA',projdim);
        ldatestclass = knnclassify(testset*mapping.M,trainset*mapping.M,trainlabel,knearest);
        ldaknnerr(end+1) = 1 - sum(ldatestclass == testlabel)/numberoftestinstance;
        toc;
    end
    % NCA
    if run('NCA')
        fprintf(1,'--------------------------------------\nNCA\n');tic;
        [~, mapping] = compute_mapping([trainlabel,trainset],'NCA',projdim);
        ncatestclass = knnclassify(testset*mapping.M,trainset*mapping.M,trainlabel,knearest);
        ncaknnerr(end+1) = 1 - sum(ncatestclass == testlabel)/numberoftestinstance;
        toc;
    end
    % MCML
    if run('MCML')   
        fprintf(1,'--------------------------------------\nMCML\n');tic;
        [~, mapping] = compute_mapping([trainlabel,trainset],'MCML',projdim);
        mcmltestclass = knnclassify(testset*mapping.M,trainset*mapping.M,trainlabel,knearest);
        mcmlknnerr(end+1) = 1 - sum(mcmltestclass == testlabel)/numberoftestinstance;
        toc;        
    end
    % Boost
    if run('Boost')
        cd(BoostPath)
        fprintf(1,'--------------------------------------\nBoost\n');tic;
        trn.X = trainset';
        trn.y = trainlabel';
        tst.X = testset';
        tst.y = testlabel';
        boostknnerr(end+1) = test(trn,  tst);
        toc;
        cd(RootPath)
    end
    % LMNN
    if run('LMNN')
        cd(LMNNPath)
        restoredefaultpath;
        setpaths;
        fprintf(1,'--------------------------------------\nLMNN\n');tic;
        xTr = trainset';
        yTr = trainlabel';
        [L,~]=lmnn2(xTr,yTr,'quiet',3,'maxiter',1000,'validation',0.15,'checkup',0);
        cd(RootPath)
        restoredefaultpath;
        setpaths;        
        lmnntestclass = knnclassify(testset*L',trainset*L',trainlabel,3);
        lmnnknnerr(end+1) = 1 - sum(lmnntestclass == testlabel)/numberoftestinstance;
        toc;        
    end
end
%% evaluation output
fprintf(1,'--------------------------------------\n');tic;
% best of CFLML
knnerr = min([knnerr_1; knnerr_3; knnerr_em]);
Mknnerr = mean(knnerr);
Vknnerr = sqrt(repeattime*(mean(knnerr.^2) - Mknnerr.^2)/(repeattime-1));



Mknnerr_1 = mean(knnerr_1);
Vknnerr_1 = sqrt(repeattime*(mean(knnerr_1.^2) - Mknnerr_1.^2)/(repeattime-1));
Mknnerr_3 = mean(knnerr_3);
Vknnerr_3 = sqrt(repeattime*(mean(knnerr_3.^2) - Mknnerr_3.^2)/(repeattime-1));
Mknnerr_em = mean(knnerr_em);
Vknnerr_em = sqrt(repeattime*(mean(knnerr_em.^2) - Mknnerr_em.^2)/(repeattime-1));
Meucknnerr = mean(eucknnerr);
Veucknnerr = sqrt(repeattime*(mean(eucknnerr.^2) - Meucknnerr.^2)/(repeattime-1));
Mpcaknnerr = mean(pcaknnerr);
Vpcaknnerr = sqrt(repeattime*(mean(pcaknnerr.^2) - Mpcaknnerr.^2)/(repeattime-1));
Mldaknnerr = mean(ldaknnerr);
Vldaknnerr = sqrt(repeattime*(mean(ldaknnerr.^2) - Mldaknnerr.^2)/(repeattime-1));
Mncaknnerr = mean(ncaknnerr);
Vncaknnerr = sqrt(repeattime*(mean(ncaknnerr.^2) - Mncaknnerr.^2)/(repeattime-1));
Mmcmlknnerr = mean(mcmlknnerr);
Vmcmlknnerr = sqrt(repeattime*(mean(mcmlknnerr.^2) - Mmcmlknnerr.^2)/(repeattime-1));
Mboostknnerr = mean(boostknnerr);
Vboostknnerr = sqrt(repeattime*(mean(boostknnerr.^2) - Mboostknnerr.^2)/(repeattime-1));
Mlmnnknnerr = mean(lmnnknnerr);
Vlmnnknnerr = sqrt(repeattime*(mean(lmnnknnerr.^2) - Mlmnnknnerr.^2)/(repeattime-1));


fprintf([...
    'Euclidean err:%.2f(%.2f)%%\n',...
    'PCA err:%.2f(%.2f)%%\n',...
    'LDA err:%.2f(%.2f)%%\n',...
    'NCA err:%.2f(%.2f)%%\n',...
    'MCML err:%.2f(%.2f)%%\n',...
    'Boost err:%.2f(%.2f)%%\n',...
    'LMNN err:%.2f(%.2f)%%\n',...
    'CFLML-1 err:%.2f(%.2f)%%\n',...
    'CFLML-3 err:%.2f(%.2f)%%\n',...
    'EM-CFLML err:%.2f(%.2f)%%\n', ...
    'CFLML best:%.2f(%.2f)%%\n'], ...
    100*Meucknnerr, 100*Veucknnerr,...
    100*Mpcaknnerr, 100*Vpcaknnerr,...
    100*Mldaknnerr, 100*Vldaknnerr,...
    100*Mncaknnerr, 100*Vncaknnerr,...
    100*Mmcmlknnerr, 100*Vmcmlknnerr,...
    100*Mboostknnerr, 100*Vboostknnerr, ...
    100*Mlmnnknnerr, 100*Vlmnnknnerr,...
    100*Mknnerr_1, 100*Vknnerr_1,...
    100*Mknnerr_3, 100*Vknnerr_3,...
    100*Mknnerr_em, 100*Vknnerr_em, ...
    100*Mknnerr, 100*Vknnerr ...
    );
toc;
