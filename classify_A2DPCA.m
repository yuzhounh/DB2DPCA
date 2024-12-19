function classify_A2DPCA(database,classifier,nPC)
% classify by alternative 2DPCA, 10 fold CV
% 2DPCA from another direction

% 2013-11-04

load(database);

[height,width,nSub]=size(x);

% kHSet=[1:height]; % set of feature numbers
kHSet=[1:nPC];
nkH=length(kHSet);
nCV=10; % 10 CV
nRep=30; % repeat the experiment for 30 times

accuracy=zeros(nkH,nCV,nRep);
tic;
for iRep=1:nRep
    ixSub=randperm(nSub); % random order
    for iCV=1:nCV
        ix_test=ixSub(floor(nSub*(iCV-1)/nCV)+1:floor(nSub*iCV/nCV));
        ix_train=setdiff(ixSub,ix_test);
        
        nTrain=length(ix_train);
        nTest=length(ix_test);
        
        x_train=x(:,:,ix_train);
        x_test=x(:,:,ix_test);
        
        label_train=label(ix_train);
        label_test=label(ix_test);
        
        x_mean=mean(x_train,3); % mean
        
        x_train=x_train-repmat(x_mean,[1,1,nTrain]);
        x_test=x_test-repmat(x_mean,[1,1,nTest]);
        
        Cov=zeros(height);
        for iTrain=1:nTrain
            Cov=Cov+x_train(:,:,iTrain)*x_train(:,:,iTrain)';
        end
        [V,D]=eig(Cov);
        
        % Sort the eigen values in order.
        [~,ixEigen]=sort(abs(diag(D)),'descend');
        V=V(:,ixEigen);
        
        for ikH=1:nkH
            kH=kHSet(ikH);
            
            v=V(:,1:kH);
            
            % projection
            x_train_proj=zeros(kH,width,nTrain);
            for iTrain=1:nTrain
                x_train_proj(:,:,iTrain)=v'*x_train(:,:,iTrain);
            end
            
            x_test_proj=zeros(kH,width,nTest);
            for iTest=1:nTest
                x_test_proj(:,:,iTest)=v'*x_test(:,:,iTest);
            end
            
            % transfrom each sample into a vector
            x_train_proj=reshape(x_train_proj,kH*width,nTrain);
            x_test_proj=reshape(x_test_proj,kH*width,nTest);
            
            % by default, each row is a sample and each column is an
            % observation
            x_train_proj=x_train_proj';
            x_test_proj=x_test_proj';
            
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
            
            accuracy(ikH,iCV,iRep)=mean(label_predict==label_test);
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
save(sprintf('%s/Acc_A2DPCA_%s.mat',db,classifier),'accuracy','time');