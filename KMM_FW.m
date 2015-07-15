% Herding for KMM
% See Eq.(8) of the following paper
% http://event.cwi.nl/uai2010/papers/UAI2010_0238.pdf
% Last modified: Nov. 2014
% Author: Junfeng Wen (junfeng.wen@ualberta.ca), University of Alberta
function [weight, obj] = KMM_FW(Xtr, Xte, options)

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
sampleSum = zeros(n, 1);
weight = zeros(n, 1);

track = [];

% with replacement
t = 1; obj = inf; diff = inf;
while (method == 1 && t <= T) ||...
        (method == 2 && abs(diff) > tol) ||...
        (method == 3 && obj-options.target > tol)
    score = t*dataExp - sampleSum;
    [~, idx] = max(score);
    weight(idx) = weight(idx) + 1;
    sampleSum = sampleSum + Ktr_tr(:,idx);
    if method == 2
        oldObj = obj;
        obj = weight'*Ktr_tr*weight/(t^2) - 2*sum(weight'*Ktr_te)/(m*t);
        diff = oldObj - obj;
    elseif method == 3
        obj = weight'*Ktr_tr*weight/(t^2) - 2*sum(weight'*Ktr_te)/(m*t);
    end
    if options.verbose
        track = [track, weight'*Ktr_tr*weight/(t^2) - 2*sum(weight'*Ktr_te)/(m*t)];
    end
    t = t + 1;
end
weight = weight./(t-1);
if options.verbose
    plot(track);
end

if nargout > 2
    obj = weight'*Ktr_tr*weight - 2*sum(weight'*Ktr_te)/m;
end

%%% End of function %%%
end
