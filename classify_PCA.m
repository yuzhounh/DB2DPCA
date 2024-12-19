function classify_PCA(database,classifier,nPC)
% classify by PCA, 10 fold CV

% 2013-11-04

load(database);

[height,width,nSub]=size(x);
x=reshape(x,height*width,nSub);
x=x'; % each row is an observation and each column is a feature

% kDSet=[1:height+width]; % d<=height*width, set of feature numbers
kDSet=[1:nPC];
nkD=length(kDSet);
nCV=10; % 10 CV
nRep=30; % repeat the experiment for 30 times

accuracy=zeros(nkD,nCV,nRep);
tic;
for iRep=1:nRep
    ixSub=randperm(nSub); % random order
    for iCV=1:nCV
        ix_test=ixSub(floor(nSub*(iCV-1)/nCV)+1:floor(nSub*iCV/nCV));
        ix_train=setdiff(ixSub,ix_test);
        
        nTrain=length(ix_train);
        nTest=length(ix_test);
        
        x_train=x(ix_train,:);
        x_test=x(ix_test,:);
        
        label_train=label(ix_train);
        label_test=label(ix_test);
        
        x_mean=mean(x_train);
        
        % substract the mean
        x_train=x_train-repmat(x_mean,[nTrain,1]);
        x_test=x_test-repmat(x_mean,[nTest,1]);
        
        % svd
        % X=U*S*V'
        [~,~,W]=svd(x_train,0);
        
        for ikD=1:nkD
            kD=kDSet(ikD);
            
            w=W(:,1:kD);
            
            % projection
            x_train_proj=x_train*w;
            x_test_proj=x_test*w;
            
            if strcmp(classifier,'NN')==1
                % nearest neighbor classifier
                dxx=pdist2(x_test_proj,x_train_proj);
                [~,indx]=sort(dxx,2);
                label_predict=label_train(indx(:,1));
            elseif strcmp(classifier,'SVM')==1
                % SVM
                [x_train_proj, x_test_proj]=norm_01(x_train_proj,x_test_proj); % normalzie to [0,1] before classification
                options='-t 0 -q';
                label_predict=c01_libsvm(x_train_proj,x_test_proj,label_train,options);
            elseif strcmp(classifier,'CRC')==1
                label_predict=CRC(x_train_proj,x_test_proj,label_train);
            end
            
            accuracy(ikD,iCV,iRep)=mean(label_predict==label_test);
        end
    end
    perct(toc,iRep,nRep);
end
time=toc/3600;

ix=strfind(database,'/');
db=database(ix(3)+1:end);
if ~exist(db,'dir')
    mkdir(db);
end
save(sprintf('%s/Acc_PCA_%s.mat',db,classifier),'accuracy','time');