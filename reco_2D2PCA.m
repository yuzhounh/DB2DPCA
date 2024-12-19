function reco_2D2PCA(database)
% reconstruction by 2D2PCA

% 2013-11-04

tic;
load(database);

[height,width,nSub]=size(x);
x_mean=mean(x,3);
x_centered=x-repmat(x_mean,[1,1,nSub]);

% one direction, width
Cov=zeros(width);
for iSub=1:nSub
    Cov=Cov+x_centered(:,:,iSub)'*x_centered(:,:,iSub);
end
Cov=Cov/nSub;
[U,D]=eig(Cov);
[~,ixEigen]=sort(abs(diag(D)),'descend');
U=U(:,ixEigen);

% another direction, height
Cov=zeros(height);
for iSub=1:nSub
    Cov=Cov+x_centered(:,:,iSub)*x_centered(:,:,iSub)';
end
Cov=Cov/nSub;
[V,D]=eig(Cov);
[~,ixEigen]=sort(abs(diag(D)),'descend');
V=V(:,ixEigen);

% select different number of features
% select different number of features
kDSet=[1:min(height,width)];
nkD=length(kDSet);

err=zeros(nkD,1);
tic;
for ikD=1:nkD
    kD=kDSet(ikD);
    
    u=U(:,1:kD);
    v=V(:,1:kD);
    
    % count the reconstruction error
    rsd=0;
    for iSub=1:nSub
        rsd=rsd+norm(v*v'*x_centered(:,:,iSub)*u*u'+x_mean-x(:,:,iSub),'fro');
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
save(sprintf('%s/Err_2D2PCA.mat',db),'err','time');