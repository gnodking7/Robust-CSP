%%% Berlin Dataset %%%
% Computes robust Common Spatial Filters (solves CSP-NRQ) and CSP
% and performs convergence analysis for subject #40
%
% Dataset description: https://api-depositonce.tu-berlin.de/server/api/core/bitstreams/80d957c0-b06e-4e74-a4aa-3eb424e5ba1f/content
%   80 subjects performing 3 motor imageries: left hand, right hand, foot
%       (only 40 subjects available)
%
% Utilizes preprocessed data from 'Berlin_Prep_Data.mat'
%

load('Berlin_Prep_Data.mat'); % Berlin_Prep_Data

% Load preprocessed data
i = 40;
Cur_Data = Berlin_Prep_Data{i};
train_L = Berlin_Prep_Data{i}.train_L;
train_R = Berlin_Prep_Data{i}.train_R;
CovL = Berlin_Prep_Data{i}.CovL;
CovR = Berlin_Prep_Data{i}.CovR;
test_L = Berlin_Prep_Data{i}.test_L;
test_R = Berlin_Prep_Data{i}.test_R;

delta = 0.4;

% Compute interpolation matrices, average covariance matrices, robust
% CSP weights (eigs)
k = 10;
[V1, Sbar1, lam1] = PCA_V(CovL, k);
[V2, Sbar2, lam2] = PCA_V(CovR, k);

% Standard CSP
[x_L, ~] = eigs(Sbar1, Sbar1 + Sbar2, 1, 'smallestreal');
[x_R, ~] = eigs(Sbar2, Sbar1 + Sbar2, 1, 'smallestreal');

%%
% ++++++++++++++++++++++++++++++
%       Run situation 1 
% ++++++++++++++++++++++++++++++
SM_MAT = @(x) S_MAT(x, lam1, V1, delta, Sbar1, 0);
SP_MAT = @(x) S_MAT(x, lam2, V2, delta, Sbar2, 1);

%% SCF for NEPv
[x1, Q1, X1, RESD1, iter1, Nls1] = rbstCSP_SCF(SM_MAT, SP_MAT, x_L, tol);

%% Fixed-point iteration
[x_f1, Q_f1, X_f1, RESD_f1, iter_f1] = rbstCSP_Fixed(SM_MAT, SP_MAT, x_L, tol);

%% Manopt (Conjugate Gradient)
[x_m1, Q_m1, iter_m1] = rbstCSP_Man_CG(SM_MAT, SP_MAT, x_L, tol);

%% Manopt (Trust Region)
[x_mm1, Q_mm1, iter_mm1] = rbstCSP_Man_TR(SM_MAT, SP_MAT, x_L, tol);

%%
% ++++++++++++++++++++++++++++++
%       Run situation 2 
% ++++++++++++++++++++++++++++++
SP_MAT = @(x) S_MAT(x, lam2, V2, delta, Sbar2, 0);
SM_MAT = @(x) S_MAT(x, lam1, V1, delta, Sbar1, 1);

%% SCF for NEPv
[x2, Q2, X2, RESD2, iter2, Nls2] = rbstCSP_SCF(SP_MAT, SM_MAT, x_R, tol);

%% Fixed-point iteration
[x_f2, Q_f2, X_f2, RESD_f2, iter_f2] = rbstCSP_Fixed(SP_MAT, SM_MAT, x_R, tol);

%% Manopt (Conjugate Gradient)
[x_m2, Q_m2, iter_m2] = rbstCSP_Man_CG(SP_MAT, SM_MAT, x_R, tol);

%% Manopt (Conjugate Gradient)
[x_mm2, Q_mm2, iter_mm2] = rbstCSP_Man_TR(SP_MAT, SM_MAT, x_R, tol);

%%
% ++++++++++++++++++++++++++++++
%       Plot Results 
% ++++++++++++++++++++++++++++++

% CSP-NRQ values
figure(1)
plot(Q_f1, '-+', 'LineWidth', 3, 'MarkerSize', 8);
hold on
plot(Q1,'-or', 'LineWidth', 3, 'MarkerSize', 8);
plot(Q_m1,'-xk', 'LineWidth', 3, 'MarkerSize', 8);
plot(Q_mm1,'-*g', 'LineWidth', 3, 'MarkerSize', 8);
xlabel('Iterations', 'Fontsize', 20);
ylabel('CSP-NRQ values q(x_k)','Fontsize',20);
h_legend=legend('Fixed-point itr', 'CSP-nepv', 'Manopt (CG)', 'Manopt (TR)', 'Location', 'southeast');
set(h_legend, 'FontSize', 20);
xlim([0, 30]);

figure(2)
plot(Q_f2, '-+', 'LineWidth', 3, 'MarkerSize', 8);
hold on
plot(Q2,'-or', 'LineWidth', 3, 'MarkerSize',8);
plot(Q_m2,'-xk', 'LineWidth', 3, 'MarkerSize',8);
plot(Q_mm2,'-*g', 'LineWidth', 3, 'MarkerSize',8);
xlabel('Iterations', 'Fontsize', 20);
ylabel('CSP-NRQ values q(x_k)','Fontsize',20);
h_legend=legend('Fixed-point itr', 'CSP-nepv', 'Manopt (CG)', 'Manopt (TR)', 'Location', 'southeast');
set(h_legend, 'FontSize', 20);
xlim([0, 30]);

% Relative errors
figure(3)
semilogy(abs(Q_f1-Q1(end)), '-+', 'LineWidth', 3, 'MarkerSize', 8);
hold on
semilogy(abs(Q1-Q1(end)), '-or', 'LineWidth', 3, 'MarkerSize', 8);
semilogy(abs(Q_m1-Q_m1(end)), '-xk', 'LineWidth', 3, 'MarkerSize', 8);
semilogy(abs(Q_mm1-Q_mm1(end)), '-*g', 'LineWidth', 3, 'MarkerSize', 8);
xlabel('iterations', 'Fontsize', 20);
ylabel('|q(x_k) -q(x_*)|', 'Fontsize', 20);
h_legend=legend('Fixed-point itr', 'CSP-nepv', 'Manopt (CG)', 'Manopt (TR)', 'Location', 'southeast');
set(h_legend, 'FontSize', 20);
xlim([0, 30]);

figure(4)
semilogy(abs(Q_f2-Q2(end)), '-+', 'LineWidth', 3, 'MarkerSize', 8);
hold on
semilogy(abs(Q2-Q2(end)), '-or', 'LineWidth', 3, 'MarkerSize', 8);
semilogy(abs(Q_m2-Q_m2(end)), '-xk', 'LineWidth', 3, 'MarkerSize', 8);
semilogy(abs(Q_mm2-Q_mm2(end)), '-*g', 'LineWidth', 3, 'MarkerSize', 8);
xlabel('iterations', 'Fontsize', 20);
ylabel('|q(x_k) -q(x_*)|', 'Fontsize', 20);
h_legend=legend('Fixed-point itr', 'CSP-nepv', 'Manopt (CG)', 'Manopt (TR)', 'Location', 'southeast');
set(h_legend, 'FontSize', 20);
xlim([0, 30]);
