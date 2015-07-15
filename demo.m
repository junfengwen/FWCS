% A toy demo for the paper
% Correcting Covariate Shift with the Frank-Wolfe Algorithm, IJCAI 2015
% Author: Junfeng Wen (junfeng.wen@ualberta.ca), University of Alberta
clear
clc

seed = 100;
rng(seed);
t = 2^7; % number of training or test points
noise = 0.1; % noise level of y

%% Data generation
f = @(x)(- x + x.^3 + 1);
x1 = randn(t,1)/2 + 0.5;
y1 = f(x1) + noise*randn(t,1);
x2 = 0.3*randn(t,1);
y2 = f(x2) + noise*randn(t,1);

%% KMM
options = [];
options.verbose = 0;
options.kernel = @gausskernel;
options.kernel_param = median(pdist(x1));
[w_KMM_FW, obj_KMM_FW] = KMM_FW(x1, x2, options);
[w_KMM_FW_line, obj_KMM_FW_line] = KMM_FW_line(x1, x2, options);

%% KLIEP
options = [];
options.verbose = 0;
options.stepSize = 'default';
[w_KLIEP_FW, obj_KLIEP_FW, alpha_KLIEP_FW] = KLIEP_FW(x1, x2, options);
w_KLIEP_FW = w_KLIEP_FW./sum(w_KLIEP_FW);

options.stepSize = 'lineSearch';
[w_KLIEP_FW_line, obj_KLIEP_FW_line, alpha_KLIEP_FW_line] = KLIEP_FW(x1, x2, options);
w_KLIEP_FW_line = w_KLIEP_FW_line./sum(w_KLIEP_FW_line);

%% Plot
weight = w_KLIEP_FW_line;
plotWeight(x1,x2,y1,y2,weight,f);
