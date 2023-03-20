% ---------------------------------------------------- %
%   rbstCSP_Fixed: Fixed-point iteration for solving CSP-NRQ
% ---------------------------------------------------- %

function [x, Q, X, RESD, iter] = rbstCSP_Fixed(SM_MAT, SP_MAT, x0, tol)
    % Input:
    %   SM_MAT - function handle
    %            \Sigma(x) = SM_MAT(x)
    %            [\Sigma(x) \tilde{\Sigma}(x)] = SM_MAT(x)
    %   SP_MAT - function handle
    %            \Sigma(x) = SP_MAT(x)
    %            [\Sigma(x) \tilde{\Sigma}(x)] = SP_MAT(x)
    %   x0 - initial vector
    %   tol - tolerance
    % Output:
    %   x - final vector
    %   Q - array of objective values at each iterate
    %   X - matrix with columns corresponding to iterates
    %   RESD - relative NEPv residual norm at each iterate
    %   iter - number of iterations

    if nargin < 4
        tol = 1.0E-8; 
    end

    maxit = 200;
    Q = []; X = []; RESD = [];

    % +++++++++++++++++++++++++++
    % --- MAIN LOOP
    % +++++++++++++++++++++++++++
    x = x0;
    for i = 1:maxit
        % Evaluate ratio
        SM = SM_MAT(x);
        SP = SP_MAT(x);
        q = (x' * SM * x) / (x' * (SM + SP) * x);
        Q = [Q, q];
        X = [X, x];

        % Evaluate relative NEPv norm
        resd = norm(SM * x - q * (SM + SP) * x) / (norm(SM * x) + q * norm((SM + SP) * x));
        RESD = [RESD, resd];

        % Stopping condition
        if i > 1 && norm(Q(end-1) - 1) < tol * 1
            break
        end

        [new_x, ~] = eigs(SM, SM + SP, 1, 'smallestreal');
        x = new_x;
    end
    iter = i;
end