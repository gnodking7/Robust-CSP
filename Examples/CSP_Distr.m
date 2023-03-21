%%% Distraction Dataset %%%
% Computes robust Common Spatial Filters (solves CSP-NRQ) and CSP
% and performs classification
% 
% Performs binary motor imagery classification tasks using filters
% and linear classifier
%
% Dataset description: https://github.com/stephaniebrandl/bci-under-distraction
%   16 subjects performing 2 motor imageries: left hand, right hand
%
% Utilizes BBCI Toolbox: https://github.com/bbci/bbci_public

BAND = load('bands.mat');   % bandpass filter values for each subject
IVAL = load('ivals.mat');   % time interval for each subject
filtOrder = 3;  % for butter bandpass filter

DELTA = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

MISSRATE = cell(1, 16);

for i = 1:16
    fprintf('%%%%%%%%%% STARTING Subject %d %%%%%%%%%% \n', i)

    name = strcat('BCI_dist_prepro', string(i), '.mat');    % preprocessed data
    DATA = load(name);

    train_L = DATA.L; train_R = DATA.R;

    % Compute interpolation matrices, average covariance matrices, robust
    % CSP weights (eigs)
    k = 10;
    [V1, Sbar1, lam1] = PCA_V(DATA.CovL, k);
    [V2, Sbar2, lam2] = PCA_V(DATA.CovR, k);

    % Standard CSP
    [x_L, ~] = eigs(Sbar1, Sbar1 + Sbar2, 1, 'smallestreal');
    [x_R, ~] = eigs(Sbar2, Sbar1 + Sbar2, 1, 'smallestreal');

    % Robust CSP
    MissRate = [];
    conditions = {'clean', 'eyes', 'news', 'numbers', 'flicker', 'stimulation'};
    for j = 1:6
        fprintf('-- Conditions %d -- \n', j)
        test_L = DATA.FeedL{j}; test_R = DATA.FeedR{j};
        missrate_cond = [CSP_classifier(x_L, x_R, train_L, train_R, test_L, test_R)];
        min_miss = 1;
        min_delta = 2;
        for d = 1:length(DELTA)
            delta = DELTA(d);
            [x1, x2] = SCF(delta, Sbar1, lam1, V1, Sbar2, lam2, V2, x_L, x_R, tol);
            cur_missrate = CSP_classifier(x1, x2, train_L, train_R, test_L, test_R);
            if cur_missrate < min_miss
                min_miss = cur_missrate;
                min_delta = delta;
            end
        end
        missrate_cond = [missrate_cond; min_miss; min_delta];
        MissRate = [MissRate, missrate_cond];
    end

    MISSRATE{i} = MissRate;

end


%%
% ++++++++++++++++++++++++++++++
%       Plot Results 
% ++++++++++++++++++++++++++++++

figure(1)
hold on
for i = 1:16
    scatter(1 - MISSRATE{i}(1, 1), 1 - MISSRATE{i}(2, 1), 50, 'b', 'filled')
end
xlim([0.4, 1.0])
ylim([0.4, 1.0])
t = linspace(0.4, 1.0, 500);
plot(t, t, 'r', 'LineWidth', 1.5)
text(0.5, 0.9, '81.25%', 'Color', 'r', 'FontSize', 20)
text(0.9, 0.5, '6.25%', 'Color', 'r', 'FontSize', 20)
xlabel('CSP', 'Fontsize', 20);
ylabel('CSP-NRQ (Alg. 1)','Fontsize', 20);
title('Classification rate (Clean)', 'Fontsize', 20);

figure(2)
hold on
for i = 1:16
    scatter(1 - MISSRATE{i}(1, 2), 1 - MISSRATE{i}(2, 2), 50, 'b', 'filled')
end
xlim([0.4, 1.0])
ylim([0.4, 1.0])
t = linspace(0.4, 1.0, 500);
plot(t, t, 'r', 'LineWidth', 1.5)
text(0.5, 0.9, '75.0%', 'Color', 'r', 'FontSize', 20)
text(0.9, 0.5, '6.25%', 'Color', 'r', 'FontSize', 20)
xlabel('CSP', 'Fontsize', 20);
ylabel('CSP-NRQ (Alg. 1)','Fontsize', 20);
title('Classification rate (eyes)', 'Fontsize', 20);

figure(3)
hold on
for i = 1:16
    scatter(1 - MISSRATE{i}(1, 3), 1 - MISSRATE{i}(2, 3), 50, 'b', 'filled')
end
xlim([0.4, 1.0])
ylim([0.4, 1.0])
t = linspace(0.4, 1.0, 500);
plot(t, t, 'r', 'LineWidth', 1.5)
text(0.5, 0.9, '50.0%', 'Color', 'r', 'FontSize', 20)
text(0.9, 0.5, '12.5%', 'Color', 'r', 'FontSize', 20)
xlabel('CSP', 'Fontsize', 20);
ylabel('CSP-NRQ (Alg. 1)','Fontsize', 20);
title('Classification rate (news)', 'Fontsize', 20);

figure(4)
hold on
for i = 1:16
    scatter(1 - MISSRATE{i}(1, 4), 1 - MISSRATE{i}(2, 4), 50, 'b', 'filled')
end
xlim([0.4, 1.0])
ylim([0.4, 1.0])
t = linspace(0.4, 1.0, 500);
plot(t, t, 'r', 'LineWidth', 1.5)
text(0.5, 0.9, '81.25%', 'Color', 'r', 'FontSize', 20)
text(0.9, 0.5, '12.5%', 'Color', 'r', 'FontSize', 20)
xlabel('CSP', 'Fontsize', 20);
ylabel('CSP-NRQ (Alg. 1)','Fontsize', 20);
title('Classification rate (numbers)', 'Fontsize', 20);

figure(5)
hold on
for i = 1:16
    scatter(1 - MISSRATE{i}(1, 5), 1 - MISSRATE{i}(2, 5), 50, 'b', 'filled')
end
xlim([0.4, 1.0])
ylim([0.4, 1.0])
t = linspace(0.4, 1.0, 500);
plot(t, t, 'r', 'LineWidth', 1.5)
text(0.5, 0.9, '68.75%', 'Color', 'r', 'FontSize', 20)
text(0.9, 0.5, '6.25%', 'Color', 'r', 'FontSize', 20)
xlabel('CSP', 'Fontsize', 20);
ylabel('CSP-NRQ (Alg. 1)','Fontsize', 20);
title('Classification rate (flicker)', 'Fontsize', 20);

figure(6)
hold on
for i = 1:16
    scatter(1 - MISSRATE{i}(1, 6), 1 - MISSRATE{i}(2, 6), 50, 'b', 'filled')
end
xlim([0.4, 1.0])
ylim([0.4, 1.0])
t = linspace(0.4, 1.0, 500);
plot(t, t, 'r', 'LineWidth', 1.5)
text(0.5, 0.9, '68.75%', 'Color', 'r', 'FontSize', 20)
text(0.9, 0.5, '6.25%', 'Color', 'r', 'FontSize', 20)
xlabel('CSP', 'Fontsize', 20);
ylabel('CSP-NRQ (Alg. 1)','Fontsize', 20);
title('Classification rate (stimulation)', 'Fontsize', 20);


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
