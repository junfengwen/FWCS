function K = gausskernel(X1,X2,sigma)
	t1 = size(X1);
	t2 = size(X2);
	D = repmat(sum(X1.*X1,2),1,t2) + repmat(sum(X2.*X2,2)',t1,1) - 2*X1*X2';
	K = exp(-0.5*D/sigma^2);
end
