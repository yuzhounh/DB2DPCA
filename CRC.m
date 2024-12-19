function label_predict=CRC(x_train,x_test,label_train)
% for multiple classes
% L. Zhang et al., 2011, ICCV
% 2013-11-26 15:24:54

n=size(x_train,1);
m=size(x_test,1);

lam=n*1e-3/700; % as in the paper

p=(x_train*x_train'+lam*eye(n))\(x_train*x_test');

k=length(unique(label_train));
rsd=zeros(m,k);
for i=1:k
    ix=label_train==i;
    pi=p.*repmat(ix,1,m);
    rsd(:,i)=norm_row(x_train'*pi-x_test')./norm_row(pi);
end
[~,label_predict]=min(rsd,[],2);

function y=norm_row(x)
% calculate norm for each column of x
% 2013-3-12 8:09:54

for i=1:size(x,2)
    y(i)=norm(x(:,i));
end