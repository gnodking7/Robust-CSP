%%% Synthetic Dataset %%%
% Computes robust Common Spatial Filters (solves CSP-NRQ)
%   SCF for NEPv
%   Fixed-point iteration
% and standard Common Spatial Filters
% 
% Performs binary motor imagery classification tasks using filters
% and linear classifier

MISSRATE_SCF = []; MISSRATE_FIXED = [];
DELTA = [0.5, 1, 2, 4, 6, 8]; % perturbation levels, large value may lead to nonsmooth case 

for i = 1:100
    [V1, lam1, Sbar1, V2, lam2, Sbar2, Tst1, Tst2, Train1, Train2] = synth_datagen();   % Synthetic data
    x0 = randn(length(Sbar1), 1);   % random initial vector
    x0 = x0 / norm(x0);

    % Standard CSP
    [x_L, ~] = eigs(Sbar1, Sbar1 + Sbar2, 1, 'smallestreal');
    [x_R, ~] = eigs(Sbar2, Sbar1 + Sbar2, 1, 'smallestreal');

    MissRate = CSP_classifier(x_L, x_R, Train1, Train2, Tst1, Tst2);

    MissRate_SCF = []; MissRate_FIXED = [];

    % Robust CSP
    for j = 1:length(DELTA)
        delta = DELTA(j);
        fprintf('%%%%%%%%%% delta %d %%%%%%%%%% \n', delta)
        [x1, x2] = SCF(delta, Sbar1, lam1, V1, Sbar2, lam2, V2, x_L, x_R);
        [x_f1, x_f2] = Fixed(delta, Sbar1, lam1, V1, Sbar2, lam2, V2, x_L, x_R);
        MissRate_SCF = [MissRate_SCF; CSP_classifier(x1, x2, Train1, Train2, Tst1, Tst2)];
        MissRate_FIXED = [MissRate_FIXED; CSP_classifier(x_f1, x_f2, Train1, Train2, Tst1, Tst2)];
    end

    MissRate_SCF = [MissRate; MissRate_SCF]; 
    MissRate_FIXED = [MissRate; MissRate_FIXED];

    MISSRATE_SCF = [MISSRATE_SCF, MissRate_SCF];
    MISSRATE_FIXED = [MISSRATE_FIXED, MissRate_FIXED];
end

%% Plot boxplots
figure(1)
boxplot(1 - MISSRATE_SCF')
title('Algorithm 1', 'Fontsize', 20);
ylabel('classification rate', 'Fontsize', 20);
xticklabels(['non-rbst', string(DELTA)]);
ax = gca;
ax.XAxis.FontSize = 13;

figure(2)
boxplot(1 - MISSRATE_FIXED')
title('Fixed-point Iteration', 'Fontsize', 20);
ylabel('classification rate', 'Fontsize', 20);
xticklabels(['non-rbst', string(DELTA)]);
ax = gca;
ax.XAxis.FontSize = 13;


%%
% ****************************** AUX FUN ********************************
%

% ---------------------------------------------------- %
%   SCF: robust csp by scf
% ---------------------------------------------------- %
function [x1, x2] = SCF(delta, Sbar1, lam1, V1, Sbar2, lam2, V2, x_L, x_R)
    tol = 1e-8;
    % ------------------------------
    % --- Run situation 1 
    % ------------------------------
    SM_MAT = @(x) S_MAT(x, lam1, V1, delta, Sbar1, 0);
    SP_MAT = @(x) S_MAT(x, lam2, V2, delta, Sbar2, 1);
    [x1, ~, ~, ~, ~, ~] = rbstCSP_SCF(SM_MAT, SP_MAT, x_L, tol);
    x1 = x1 / sqrt(x1' * (SM_MAT(x1) + SP_MAT(x1)) * x1);
    % ------------------------------
    % --- Run situation 2 
    % ------------------------------
    SP_MAT = @(x) S_MAT(x, lam2, V2, delta, Sbar2, 0);
    SM_MAT = @(x) S_MAT(x, lam1, V1, delta, Sbar1, 1);
    [x2, ~, ~, ~, ~, ~] = rbstCSP_SCF(SP_MAT, SM_MAT, x_R, tol);
    x2 = x2 / sqrt(x2' * (SM_MAT(x2) + SP_MAT(x2)) * x2);
end

% ---------------------------------------------------- %
%   Fixed: robust csp by fixed-point iteration
% ---------------------------------------------------- %
function [x_f1, x_f2] = Fixed(delta, Sbar1, lam1, V1, Sbar2, lam2, V2, x_L, x_R)
    tol = 1e-8;
    % ------------------------------
    % --- Run situation 1 
    % ------------------------------
    SM_MAT = @(x) S_MAT(x, lam1, V1, delta, Sbar1, 0);
    SP_MAT = @(x) S_MAT(x, lam2, V2, delta, Sbar2, 1);
    [~, Q1, X1, ~, ~] = rbstCSP_Fixed(SM_MAT, SP_MAT, x_L, tol);
    [~, idx] = min(Q1);
    x_f1 = X1(:, idx);
    % ------------------------------
    % --- Run situation 2 
    % ------------------------------
    SP_MAT = @(x) S_MAT(x, lam2, V2, delta, Sbar2, 0);
    SM_MAT = @(x) S_MAT(x, lam1, V1, delta, Sbar1, 1);
    [~, Q2, X2, ~, ~] = rbstCSP_Fixed(SP_MAT, SM_MAT, x_R, tol);
    [~, idx] = min(Q2);
    x_f2 = X2(:, idx);
end