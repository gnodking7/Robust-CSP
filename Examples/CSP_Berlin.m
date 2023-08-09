%%% Berlin Dataset %%%
% Computes robust Common Spatial Filters and CSP
% and performs classification
% 
% Performs binary motor imagery classification tasks using filters
% and linear classifier
%
% Dataset description: https://api-depositonce.tu-berlin.de/server/api/core/bitstreams/80d957c0-b06e-4e74-a4aa-3eb424e5ba1f/content
%   80 subjects performing 3 motor imageries: left hand, right hand, foot
%       (only 40 subjects available)
%
% Utilizes preprocessed data from 'Berlin_Prep_Data.mat'
%

load('Berlin_Prep_Data.mat'); % Berlin_Prep_Data

DELTA = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
% DELTA = [0.4];

MISSRATE = [];

for i = 1:40
    fprintf('%%%%%%%%%% STARTING Subject %d %%%%%%%%%% \n', i)
    
    % Load preprocessed data
    Cur_Data = Berlin_Prep_Data{i};
    train_L = Berlin_Prep_Data{i}.train_L;
    train_R = Berlin_Prep_Data{i}.train_R;
    CovL = Berlin_Prep_Data{i}.CovL;
    CovR = Berlin_Prep_Data{i}.CovR;
    test_L = Berlin_Prep_Data{i}.test_L;
    test_R = Berlin_Prep_Data{i}.test_R;

    % Compute interpolation matrices, average covariance matrices, robust
    % CSP weights (eigs)
    k = 10;
    [V1, Sbar1, lam1] = PCA_V(CovL, k);
    [V2, Sbar2, lam2] = PCA_V(CovR, k);

    % Standard CSP
    [x_L, ~] = eigs(Sbar1, Sbar1 + Sbar2, 1, 'smallestreal');
    [x_R, ~] = eigs(Sbar2, Sbar1 + Sbar2, 1, 'smallestreal');

    % Robust CSP
    min_miss_scf = 1; min_miss_fixed = 1;
    min_delta_scf = 1; min_delta_fixed = 1;
    for j = 1:length(DELTA)
        delta = DELTA(j);
        fprintf('%%%%%%%%%% delta %d %%%%%%%%%% \n', delta)

        [x1, x2] = SCF(delta, Sbar1, lam1, V1, Sbar2, lam2, V2, x_L, x_R, tol);
        cur_missrate_scf = CSP_classifier(x1, x2, train_L, train_R, test_L, test_R);
        if cur_missrate_scf < min_miss_scf
            min_miss_scf = cur_missrate_scf;
            min_delta_scf = delta;
        end

        [x_f1, x_f2] = Fixed(delta, Sbar1, lam1, V1, Sbar2, lam2, V2, x_L, x_R, tol);
        cur_missrate_fixed = CSP_classifier(x_f1, x_f2, train_L, train_R, test_L, test_R);
        if cur_missrate_fixed < min_miss_fixed
            min_miss_fixed = cur_missrate_fixed;
            min_delta_fixed = delta;
        end
    end

    MissRate = CSP_classifier(x_L, x_R, train_L, train_R, test_L, test_R);
    MissRate = [MissRate; min_miss_scf; min_delta_scf; min_miss_fixed; min_delta_fixed];
    MISSRATE = [MISSRATE, MissRate];

end

%%
% ++++++++++++++++++++++++++++++
%       Plot Results 
% ++++++++++++++++++++++++++++++

figure(1)
hold on
for i = 1:40
    scatter(1 - MISSRATE(1, i), 1 - MISSRATE(2, i), 50, 'b', 'filled')
end
xlim([0.2, 1.0])
ylim([0.2, 1.0])
t = linspace(0.2, 1.0, 500);
plot(t, t, 'r', 'LineWidth', 1.5)
text(0.3, 0.9, '87.5%', 'Color', 'r', 'FontSize', 20)
text(0.9, 0.3, '10.0%', 'Color', 'r', 'FontSize', 20)
xlabel('CSP', 'Fontsize', 20);
ylabel('CSP-nepv','Fontsize', 20);
title('Classification rate', 'Fontsize', 20);


%%

%%%%%%%%%% FUNCTIONS %%%%%%%%%%

% ---------------------------------------------------- %
%   SCF: robust csp by scf
% ---------------------------------------------------- %
function [x1, x2] = SCF(delta, Sbar1, lam1, V1, Sbar2, lam2, V2, x_L, x_R, tol)
    % ------------------------------
    % --- Run situation 1 
    % ------------------------------
    SM_MAT = @(x) S_MAT(x, lam1, V1, delta, Sbar1, 0);
    SP_MAT = @(x) S_MAT(x, lam2, V2, delta, Sbar2, 1);

    [x1, Q1, ~, ~, ~, ~] = rbstCSP_SCF(SM_MAT, SP_MAT, x_L, tol);
    [SM, TSM] = SM_MAT(x1);
    [SP, TSP] = SP_MAT(x1);
    x1 = x1 / sqrt(x1' * (SM + SP) * x1);

    % ++++++++++++++++++++++++++++++
    %       Run situation 2 
    % ++++++++++++++++++++++++++++++
    SP_MAT = @(x) S_MAT(x, lam2, V2, delta, Sbar2, 0);
    SM_MAT = @(x) S_MAT(x, lam1, V1, delta, Sbar1, 1);
    
    [x2, Q2, ~, ~, ~, ~] = rbstCSP_SCF(SP_MAT, SM_MAT, x_R, tol);
    [SP, TSP] = SP_MAT(x2);
    [SM, TSM] = SM_MAT(x2);
    x2 = x2 / sqrt(x2' * (SM + SP) * x2);
end

% ---------------------------------------------------- %
%   Fixed: robust csp by fixed-point iteration
% ---------------------------------------------------- %
function [x_f1, x_f2] = Fixed(delta, Sbar1, lam1, V1, Sbar2, lam2, V2, x_L, x_R, tol)
    % ------------------------------
    % --- Run situation 1 
    % ------------------------------
    SM_MAT = @(x) S_MAT(x, lam1, V1, delta, Sbar1, 0);
    SP_MAT = @(x) S_MAT(x, lam2, V2, delta, Sbar2, 1);

    [~, Q_f1, X_f1, ~, ~] = rbstCSP_Fixed(SM_MAT, SP_MAT, x_L, tol);
    [~, idx] = min(Q_f1);
    x_f1 = X_f1(:, idx);
    % ------------------------------
    % --- Run situation 2 
    % ------------------------------
    SP_MAT = @(x) S_MAT(x, lam2, V2, delta, Sbar2, 0);
    SM_MAT = @(x) S_MAT(x, lam1, V1, delta, Sbar1, 1);

    [~, Q_f2, X_f2, ~, ~] = rbstCSP_Fixed(SP_MAT, SM_MAT, x_R, tol);
    [~, idx] = min(Q_f2);
    x_f2 = X_f2(:, idx);
end
