%%% Synthetic Dataset %%%
% Computes robust Common Spatial Filters (solves CSP-NRQ)
%   SCF for NEPv
%   Fixed-point iteration
% and standard Common Spatial Filters
% 
% Displays convergence of Fixed-point iteration and SCF for NEPv

rng(3); 

% Generate Synthetic data
[V1, lam1, Sbar1, V2, lam2, Sbar2, Tst1, Tst2, Train1, Train2] = synth_datagen();
delta = 6; % perturbation level, large value may lead to nonsmooth case 

x0 = randn(length(Sbar1), 1);   % random initial vector
x0 = x0 / norm(x0);
tol = 1e-8;

%%
% ++++++++++++++++++++++++++++++
%       Run situation 1 
% ++++++++++++++++++++++++++++++
SM_MAT = @(x) S_MAT(x, lam1, V1, delta, Sbar1, 0);
SP_MAT = @(x) S_MAT(x, lam2, V2, delta, Sbar2, 1);

%% SCF for NEPv
[x1, Q1, X1, RESD1, iter1, Nls1] = rbstCSP_SCF(SM_MAT, SP_MAT, x0, tol);
% Display eig order
[SM, TSM] = SM_MAT(x1);
[SP, TSP] = SP_MAT(x1);
disp(['Eigenvalue associated with solution is: ', num2str(Q1(end))]);
disp([sort(eig(SM, SM + SP)), sort(eig(SM + TSM, SM + TSM + SP + TSP))])

%% Fixed-point iteration
[x_f1, Q_f1, X_f1, RESD_f1, iter_f1] = rbstCSP_Fixed(SM_MAT, SP_MAT, x0, tol);

%% Manopt (Conjugate Gradient)
[x_m1, Q_m1, iter_m1] = rbstCSP_Man_CG(SM_MAT, SP_MAT, x0, tol);

%% Manopt (Trust Region)
[x_mm1, Q_mm1, iter_mm1] = rbstCSP_Man_TR(SM_MAT, SP_MAT, x0, tol);

%%
% ++++++++++++++++++++++++++++++
%       Run situation 2 
% ++++++++++++++++++++++++++++++
SP_MAT = @(x) S_MAT(x, lam2, V2, delta, Sbar2, 0);
SM_MAT = @(x) S_MAT(x, lam1, V1, delta, Sbar1, 1);

%% SCF for NEPv
[x2, Q2, X2, RESD2, iter2, Nls2] = rbstCSP_SCF(SP_MAT, SM_MAT, x0, tol);
% Display eig order
[SP, TSP] = SP_MAT(x2);
[SM, TSM] = SM_MAT(x2);
disp(['Eigenvalue associated with solution is: ', num2str(Q2(end))]);
disp([sort(eig(SP, SM + SP)), sort(eig(SP + TSP, SM + TSM + SP + TSP))])

%% Fixed-point iteration
[x_f2, Q_f2, X_f2, RESD_f2, iter_f2] = rbstCSP_Fixed(SP_MAT, SM_MAT, x0, tol);

%% Manopt (Conjugate Gradient)
[x_m2, Q_m2, iter_m2] = rbstCSP_Man_CG(SP_MAT, SM_MAT, x0, tol);

%% Manopt (Conjugate Gradient)
[x_mm2, Q_mm2, iter_mm2] = rbstCSP_Man_TR(SP_MAT, SM_MAT, x0, tol);

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
h_legend=legend('Fixed-point itr', 'Alg.1', 'Manopt (CG)', 'Manopt (TR)', 'Location', 'southeast');
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
h_legend=legend('Fixed-point itr', 'Alg.1', 'Manopt (CG)', 'Manopt (TR)', 'Location', 'southeast');
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
h_legend=legend('Fixed-point itr', 'Alg.1', 'Manopt (CG)', 'Manopt (TR)', 'Location', 'southeast');
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
h_legend=legend('Fixed-point itr', 'Alg.1', 'Manopt (CG)', 'Manopt (TR)', 'Location', 'southeast');
set(h_legend, 'FontSize', 20);
xlim([0, 30]);
