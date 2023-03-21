% ---------------------------------------------------- %
%   synth_datagen: generates synthetic dataset
%       a signal x(t) at time t is generated as:
%       x(t) = A[sd(t);sn(t)]+\epsilon(t)
%           A is a random rotation matrix
%           sd(t) \in \R^2 discriminatory sources
%               N(0,[0.2,1.4]) for '-'
%               N(0,[1.8,0.6]) for '+'
%           sn(t) \in \R^8 nondiscriminatory sources
%               N(0,1) for both '-' and '+'
%           \epsilon(t) \in \R^10 artifacts
%               N(0,2) for training
%               N(03,30) for testing
%       200 time samples for a trial, i.e., Y \in \R^{2\times100}
%           50 trials, for each condition, for training and testing
% ---------------------------------------------------- %

function [V1, lam1, Sbar1, V2, lam2, Sbar2, Tst1, Tst2, Train1, Train2] = synth_datagen()
    % Output:
    %   V1, V2 - cell array of interpolation matrices
    %   lam1, lam2 - cell array of eigenvalues
    %   Sbar1, Sbar2 - average covariance matrix
    %   Tst1, Tst2 - cell array of test trials
    %   Train1, Train2 - cell array of train trials

    n_trials = 50;
    n_samples = 200;

    eps_tr = 2; eps_tst = 30;   % epsilons

    % Mixing matrix
    [A, ~] = qr(rand(10));
    % Create trials and covariance matrices
    for i = 1:n_trials
        % Trials
        Y1_tr = []; Y2_tr = [];
        Y1_tst = []; Y2_tst = [];
        for j = 1:n_samples
            s1 = [randn * sqrt(1.8); randn * sqrt(0.6); randn(8,1)];
            s2 = [randn * sqrt(0.2); randn * sqrt(1.4); randn(8,1)];

            Y1_tr = [Y1_tr, A * s1 + randn * sqrt(eps_tr)];
            Y2_tr = [Y2_tr, A * s2 + randn * sqrt(eps_tr)];

            Y1_tst = [Y1_tst, A * s1 + randn * sqrt(eps_tst)];
            Y2_tst = [Y2_tst, A * s2 + randn * sqrt(eps_tst)];
        end
        S1{i} = cov(Y1_tr');
        S2{i} = cov(Y2_tr');

        Train1{i} = Y1_tr; Train2{i} = Y2_tr;
        Tst1{i} = Y1_tst; Tst2{i} = Y2_tst;
    end
    [V1, Sbar1, lam1] = PCA_V(S1, 10);
    [V2, Sbar2, lam2] = PCA_V(S2, 10);
end
