% Solving KLIEP with Frank-Wolfe algorithm 
% Last modified: Nov. 2014
% Author: Junfeng Wen (junfeng.wen@ualberta.ca), University of Alberta
function [weight, obj, alpha, sigmaChosen] = KLIEP_FW(Xtr, Xte, options)

DEFAULTS.T = 100;
DEFAULTS.nFold = 5;
DEFAULTS.tol = 1e-4;
DEFAULTS.verbose = 1;
DEFAULTS.stepSize = 'default';
DEFAULTS.method = 2;
% methods:
% 1 - repeat T iterations
% 2 - tolerance on objective changes
% 3 - stop when reaching targeted obj
if nargin < 3
    options = DEFAULTS;
else
    options = getOptions(options, DEFAULTS);
end

% Choose a proper sigma as did in the original paper
if ~isfield(options,'sigma')
    sigma = 10; score = -inf;
    m = size(Xte, 1);
    part = floor(m/options.nFold);
    for epsilon = log10(sigma)-1:-1:-3
        for i = 1:9
            newSigma = sigma - 10^epsilon;
            newScore = 0;
            options.sigma = newSigma;
            % Partition the dataset for CV score
            for r = 1:options.nFold
                Xtmp = Xte;
                Xtmp((r-1)*part+1:r*part,:) = [];
                [~, obj] = KLIEP_FW_learning(Xtr, Xtmp, options);
                newScore = newScore + obj/options.nFold;
                if obj == -inf
                    break;
                end
            end
            if newScore - score <= 0
                break;
            end
            sigma = newSigma; score = newScore;
            if options.verbose
                fprintf(1, '- score = %g, sigma = %g\n', score, sigma);
            end
        end
        if options.verbose
            fprintf(1, '- next digit\n');
        end
    end
    options.sigma = sigma;
    if options.verbose
        fprintf(1, '- final sigma = %g\n', sigma);
    end
end
sigmaChosen = options.sigma;

% Do the actual learning
[weight, obj, alpha] = KLIEP_FW_learning(Xtr, Xte, options);

% END OF FUNCTION
end
