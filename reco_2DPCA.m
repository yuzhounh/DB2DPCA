function reco_2DPCA(database)
% reconstruction by 2DPCA

% 2013-11-04

tic;
load(database);

[height,width,nSub]=size(x);
x_mean=mean(x,3);
x_centered=x-repmat(x_mean,[1,1,nSub]);

Cov=zeros(width);
for iSub=1:nSub
    Cov=x_centered(:,:,iSub)'*x_centered(:,:,iSub);
end
Cov=Cov/nSub;
[U,D]=eig(Cov);

% Sort the eigen values in order.
[~,indx]=sort(abs(diag(D)),'descend');
U=U(:,indx);

kWSet=[1:width]; % width, a direction, to indicate the selected number of features
nkW=length(kWSet);
err=zeros(nkW,1);
tic;
for ikW=1:nkW
    kW=kWSet(ikW);
    u=U(:,1:kW);
    
    % count the reconstruction error
    rsd=0;
    for iSub=1:nSub
        rsd=rsd+norm(x_centered(:,:,iSub)*u*u'+x_mean-x(:,:,iSub),'fro');
    end
    err(ikW,1)=rsd/nSub;
    perct(toc,ikW,nkW);
end
time=toc/3600;

ix=strfind(database,'/');
db=database(ix(3)+1:end);
if ~exist(db,'dir')
    mkdir(db);
end
save(sprintf('%s/Err_2DPCA.mat',db),'err','time');