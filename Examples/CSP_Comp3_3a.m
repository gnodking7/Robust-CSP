%%% Data set IIIa—BCI competition III %%%
% Computes robust Common Spatial Filters and 
% performs classification of motor imageries on BCI competition III - IIIa
% dataset
%
% Dataset description: https://www.bbci.de/competition/iii/desc_IIIa.pdf
%   three subjects performing 4 motor imageries: left hand, right hand,
%   foot, tongue
%
% Utilizes preprocessed data from 'Comp3_3a_Prep_Data.mat'
%

load('Comp3_3a_Prep_Data.mat'); % Comp3_3a_Prep_Data

DELTA = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5];

MISSRATE = [];
for i = 1:3
    fprintf('%%%%%%%%%% STARTING Subject %d %%%%%%%%%% \n', i)

    % Load preprocessed data
    Cur_Data = Comp3_3a_Prep_Data{i};
    train_L = Comp3_3a_Prep_Data{i}.train_L;
    train_R = Comp3_3a_Prep_Data{i}.train_R;
    CovL = Comp3_3a_Prep_Data{i}.CovL;
    CovR = Comp3_3a_Prep_Data{i}.CovR;
    test_L = Comp3_3a_Prep_Data{i}.test_L;
    test_R = Comp3_3a_Prep_Data{i}.test_R;

    % Compute interpolation matrices, average covariance matrices, robust
    % CSP weights (eigs)
    k = 10;
    [V1, Sbar1, lam1] = PCA_V(CovL, k);
    [V2, Sbar2, lam2] = PCA_V(CovR, k);

    % Standard CSP
    [x_L, ~] = eigs(Sbar1, Sbar1 + Sbar2, 1, 'smallestreal');
    [x_R, ~] = eigs(Sbar2, Sbar1 + Sbar2, 1, 'smallestreal');

    MissRate_SCF = [];
    % Robust CSP
    for j = 1:length(DELTA)
        delta = DELTA(j);
        fprintf('%%%%%%%%%% delta %d %%%%%%%%%% \n', delta)
        [x1, x2] = SCF(delta, Sbar1, lam1, V1, Sbar2, lam2, V2, x_L, x_R, tol);

        MissRate_SCF = [MissRate_SCF; CSP_classifier(x1, x2, train_L, train_R, test_L, test_R)];

    end

    MissRate = CSP_classifier(x_L, x_R, train_L, train_R, test_L, test_R);
    MISSRATE = [MISSRATE, [MissRate; MissRate_SCF]];

end

% MISSRATE is a matrix array of missrates; 
% each column corresponds a subject, first value is standard CSP followed
% by robust CSP for each delta

%% Plots of classification rate

figure(1)
plot(DELTA, 1 - MISSRATE(2:end, 1), '-+', 'LineWidth', 3, 'MarkerSize', 8);
yline(1 - MISSRATE(1, 1), '--', 'LineWidth', 2);
xlim([0.1 1.5]);
xticks([0.2 0.4 0.6 0.8 1 1.2 1.4]);
ax = gca;
ax.XAxis.FontSize = 13;
xlabel('\delta_c', 'Fontsize', 20);
ylabel('Classification rate', 'Fontsize', 20);
h_legend=legend('CSP-nepv', 'non-rbst', 'Location', 'southwest');
set(h_legend, 'FontSize', 20);
title('Subject l1', 'FontSize', 20);

figure(2)
plot(DELTA, 1 - MISSRATE(2:end, 2), '-+', 'LineWidth', 3, 'MarkerSize', 8);
yline(1 - MISSRATE(1, 2), '--', 'LineWidth', 2);
xlim([0.1 1.5]);
xticks([0.2 0.4 0.6 0.8 1 1.2 1.4]);
ax = gca;
ax.XAxis.FontSize = 13;
xlabel('\delta_c', 'Fontsize', 20);
ylabel('Classification rate', 'Fontsize', 20);
h_legend=legend('CSP-nepv', 'non-rbst', 'Location', 'southwest');
set(h_legend, 'FontSize', 20);
title('Subject k3', 'FontSize', 20);

figure(3)
plot(DELTA, 1 - MISSRATE(2:end, 3), '-+', 'LineWidth', 3, 'MarkerSize', 8);
yline(1 - MISSRATE(1, 3), '--', 'LineWidth', 2);
xlim([0.1 1.5]);
xticks([0.2 0.4 0.6 0.8 1 1.2 1.4]);
ax = gca;
ax.XAxis.FontSize = 13;
xlabel('\delta_c', 'Fontsize', 20);
ylabel('Classification rate', 'Fontsize', 20);
h_legend=legend('CSP-nepv', 'non-rbst', 'Location', 'southwest');
set(h_legend, 'FontSize', 20);
title('Subject k6', 'FontSize', 20);



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
