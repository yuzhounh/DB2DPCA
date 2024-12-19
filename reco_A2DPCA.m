function reco_A2DPCA(database)
% reconstruction by A2DPCA

% 2013-11-04

tic;
load(database);

[height,width,nSub]=size(x);
x_mean=mean(x,3);
x_centered=x-repmat(x_mean,[1,1,nSub]);

Cov=zeros(height);
for iSub=1:nSub
    Cov=x_centered(:,:,iSub)*x_centered(:,:,iSub)';
end
Cov=Cov/nSub;
[V,D]=eig(Cov);

% Sort the eigen values in order.
[~,indx]=sort(abs(diag(D)),'descend');
V=V(:,indx);

kHSet=[1:height];
nkH=length(kHSet);
err=zeros(nkH,1);
tic;
for ikH=1:nkH
    kH=kHSet(ikH);
    v=V(:,1:kH);
    
    % count the reconstruction error
    rsd=0;
    for iSub=1:nSub
        rsd=rsd+norm(v*v'*x_centered(:,:,iSub)+x_mean-x(:,:,iSub),'fro');
    end
    err(ikH,1)=rsd/nSub;
    perct(toc,ikH,nkH);
end
time=toc/3600;

ix=strfind(database,'/');
db=database(ix(3)+1:end);
if ~exist(db,'dir')
    mkdir(db);
end
save(sprintf('%s/Err_A2DPCA.mat',db),'err','time');