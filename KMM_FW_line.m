% Conditional gradient descent with line search
% See Bach et al. ICML 2012
% Last modified: Jan. 2015
% Author: Junfeng Wen (junfeng.wen@ualberta.ca), University of Alberta
function [weight, obj] = KMM_FW_line(Xtr, Xte, options)

DEFAULTS.T = 100;
DEFAULTS.kernel = @linearkernel;
DEFAULTS.kernel_param = 1;
DEFAULTS.tol = 1e-4;
DEFAULTS.verbose = 0;
DEFAULTS.method = 2;
% methods:
% 1 - repeat T iterations
% 2 - tolerance on objective changes
% 3 - stop when reaching targeted obj
if nargin < 1
    options = DEFAULTS;
else
    options = getOptions(options, DEFAULTS);
end

kernel = @(x1,x2)options.kernel(x1,x2,options.kernel_param);
T = options.T;
method = options.method;
tol = options.tol;

n = size(Xtr, 1);
m = size(Xte, 1);

% compute constants
Ktr_tr = kernel(Xtr, Xtr);
Ktr_te = kernel(Xtr, Xte);
dataExp = sum(Ktr_te, 2)./m;
sampleSum = zeros(n, 1); % inner product of training and gt
weight = zeros(n, 1);

track = [];

t = 0; obj = inf; diff = inf;
while (method == 1 && t < T) ||...
        (method == 2 && abs(diff) > tol) ||...
        (method == 3 && obj-options.target > tol)
    score = sampleSum - dataExp;
    [~, idx] = min(score);
    gt_gt1 = weight'*Ktr_tr(:,idx);
    if t == 0
        rho = 1;
        gt2 = weight'*Ktr_tr*weight;
        wK = weight'*Ktr_tr;
        gt_mu = sum(weight'*Ktr_te)/m;
    else
%         rho = 1/(t+1);
        gt2 = (1-rho)^2*gt2 +...
            2*rho*(1-rho)*wK(oldIdx) +...
            rho^2*Ktr_tr(oldIdx,oldIdx);
        wK = (1-rho).*wK + rho.*Ktr_tr(oldIdx,:);
        gt_mu = (1-rho)*gt_mu + rho*dataExp(oldIdx);
        rho = (gt2-gt_mu-gt_gt1+dataExp(idx)) /...
            (Ktr_tr(idx,idx)+gt2-2*gt_gt1);
    end
    weight = (1-rho).*weight;
    weight(idx) = weight(idx) + rho;
    sampleSum = (1-rho).*sampleSum + rho.*Ktr_tr(:,idx);
    oldIdx = idx;
    if method == 2
        oldObj = obj;
        obj = weight'*Ktr_tr*weight - 2*sum(weight'*Ktr_te)/m;
        diff = oldObj - obj;
    elseif method == 3
        obj = weight'*Ktr_tr*weight - 2*sum(weight'*Ktr_te)/m;
    end
    t = t + 1;
    if options.verbose
        track = [track, weight'*Ktr_tr*weight - 2*sum(weight'*Ktr_te)/m];
    end
end
if options.verbose
    plot(track);
end

if nargout > 2
    obj = weight'*Ktr_tr*weight - 2*sum(weight'*Ktr_te)/m;
end

%%% End of function %%%
end
