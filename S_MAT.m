% ---------------------------------------------------- %
%   S_MAT: computes SM (SP) and tilde SM (tilde SP)
% ---------------------------------------------------- %

function [S, TS] = S_MAT(x, lam, V, delta, Sbar, ind)
    % Input:
    %   x - vector to evaluate at
    %   lam - cell array of weights (PCA eigs)
    %   V - cell array of interpolation matrices
    %   delta - radius of tolerance region
    %   Sbar - average covariance matrix
    %   ind - 0 or 1; 0 for G and 1 for H
    % Output:
    %   S - \Sigma(x)
    %   TS - \tilde{\Sigma}(x)

    [S, TS] = optmat_build(x, lam, V, delta);

    if ind == 0
        S = Sbar + S;
    else
        S = Sbar - S;
        TS = -TS;
    end

    % Modified Cholesky to obtain positive definite
    % Created by Bobby Cheng and Nick Higham
    S = (S + S') / 2;
    [L, D, P, ~] = modchol_ldlt(S); 
    S = P' * L * D * L' * P;
    S = (S + S') / 2;

end
