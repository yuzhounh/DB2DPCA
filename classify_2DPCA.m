function classify_2DPCA(database,classifier,nPC)
% classify by 2DPCA, 10 fold CV

% 2013-11-04

load(database);
[height,width,nSub]=size(x);

% kWSet=[1:width]; % set of feature numbers
kWSet=[1:nPC];
nkW=length(kWSet);
nCV=10; % 10 CV
nRep=30; % repeat the experiment for 30 times

accuracy=zeros(nkW,nCV,nRep);
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
        
        Cov=zeros(width);
        for iTrain=1:nTrain
            Cov=Cov+x_train(:,:,iTrain)'*x_train(:,:,iTrain);
        end
        [U,D]=eig(Cov);
        
        % Sort the eigen values in order.
        [~,ixD]=sort(abs(diag(D)),'descend');
        U=U(:,ixD);
        
        for ikW=1:nkW
            kW=kWSet(ikW);
            
            u=U(:,1:kW);
            
            % projection
            x_train_proj=zeros(height,kW,nTrain);
            for iTrain=1:nTrain
                x_train_proj(:,:,iTrain)=x_train(:,:,iTrain)*u;
            end
            
            x_test_proj=zeros(height,kW,nTest);
            for iTest=1:nTest
                x_test_proj(:,:,iTest)=x_test(:,:,iTest)*u;
            end
            
            % transfrom each sample into a vector
            x_train_proj=reshape(x_train_proj,height*kW,nTrain);
            x_test_proj=reshape(x_test_proj,height*kW,nTest);
            
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
            
            accuracy(ikW,iCV,iRep)=mean(label_predict==label_test);
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
save(sprintf('%s/Acc_2DPCA_%s.mat',db,classifier),'accuracy','time');