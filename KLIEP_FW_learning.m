% Solving KLIEP with Frank-Wolfe algorithm 
% Learning function
% Last modified: Jan. 2015
% Author: Junfeng Wen (junfeng.wen@ualberta.ca), University of Alberta
function [weight, obj, alpha] = KLIEP_FW_learning(Xtr, Xte, options)

method = options.method;
tol = options.tol;
T = options.T;
[n, d] = size(Xtr);
m = size(Xte, 1);
Ktr_te = gausskernel(Xtr,Xte,options.sigma);
Kte_te = gausskernel(Xte,Xte,options.sigma);
alpha = zeros(m, 1);
k_sum_tr = sum(Ktr_te, 1) + 1e-30; % add small value for stability
bounds = n./k_sum_tr';
[~, idx] = min(bounds);
alpha(idx) = bounds(idx); % initialization

track = [];

t = 0; obj = -inf; diff = inf;
while (method == 1 && t < T) ||...
        (method == 2 && abs(diff) > tol) ||...
        (method == 3 && options.target-obj > tol)
    if t == 0
        Kalpha = Kte_te*alpha;
    else
        Kalpha = (1 - stepSize).*Kalpha +...
            (stepSize*bounds(oldIdx)).*Kte_te(:,oldIdx);
    end
    g = Kte_te'*(1./Kalpha); % gradient of alpha, all positive
    if any(isnan(g)) || any(isinf(g))
        weight = ones(n,1); obj = -inf;
        if options.verbose
            warning('Sigma too small.');
        end
        return;
    end
    [~, idx] = max(g.*bounds); % maximizing obj
    if strcmp(options.stepSize, 'default')
        stepSize = 2/(t+2); % standard Frank-Wolfe
    elseif strcmp(options.stepSize, 'lineSearch')
        alphaK = Kalpha';
        stepSize = lineSearch; % line search
    end
    alpha = (1 - stepSize).*alpha;
    alpha(idx) = alpha(idx) + stepSize*bounds(idx);
    oldIdx = idx;
    if method == 2
        oldObj = obj;
        obj = sum(log(Kte_te*alpha))/m;
        diff = obj - oldObj; % maximization
    elseif method == 3
        obj = sum(log(Kte_te*alpha))/m;
    end
    if options.verbose > 1
        track = [track, sum(log(Kte_te*alpha))/m];
    end
    t = t + 1;
end
if options.verbose > 1
    plot(track);
end
weight = Ktr_te*alpha;

function middle = lineSearch
    % check the gradient on [0,1] instead of the objective
    % the gradient is decreasing in (0,1) with g(0)>0
    % try to find the step size rho s.t. g(rho) = 0
    % if g(1)>=0, then 1 is the step size
    % otherwise g(1)<0, use binary search
    lower = 0; upper = 1; prec = inf; middle = 1;
    sK = Kte_te(idx,:).*bounds(idx);
    % alphaK can be updated more efficiently in the main loop
    grad = sum((sK-alphaK)./((1-middle).*alphaK+middle.*sK));
    if grad >= 0
        return;
    end
    while prec > options.tol
        middle = (lower + upper)/2;
        grad = sum((sK-alphaK)./((1-middle).*alphaK+middle.*sK));
        if grad > 0
            lower = middle;
        else
            upper = middle;
        end
        prec = upper - lower;
    end
end

end
