function reco_PCA(database)
% reconstruction by PCA

% 2013-11-04

tic;
load(database);

[height,width,nSub]=size(x);
x=reshape(x,height*width,nSub);
x=x'; % each row is an observation and each column is a feature

% subtract the mean
x_mean=mean(x);
x_centered=x-repmat(x_mean,nSub,1);

% svd
[~,~,W]=svd(x_centered,0);

kDSet=[1:30]; % d=h*w, to indicate the selected number of features
nkD=length(kDSet);
err=zeros(nkD,1);
tic;
for ikD=1:nkD
    kD=kDSet(ikD);
    w=W(:,1:kD);
    
    % count the reconstruction error
    rsd=0;
    for iSub=1:nSub
        rsd=rsd+norm(w*w'*x_centered(iSub,:)'+x_mean'-x(iSub,:)','fro');
    end
    err(ikD,1)=rsd/nSub;
    
    perct(toc,ikD,nkD);
end
time=toc/3600;

ix=strfind(database,'/');
db=database(ix(3)+1:end);
if ~exist(db,'dir')
    mkdir(db);
end
save(sprintf('%s/Err_PCA.mat',db),'err','time');