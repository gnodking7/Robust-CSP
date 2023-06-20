% ---------------------------------------------------- %
%   rbstCSP_Man: Manopt on Stiefel manifold
%           conjugate gradient
% ---------------------------------------------------- %
%
% Utilizes Manopt Toolbox: https://www.manopt.org

function [x, Q, iter] = rbstCSP_Man(SM_MAT, SP_MAT, x0, tol)
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
    %   iter - number of iterations

    n = length(x0);
    q_fun = @(x) (x' * SM_MAT(x) * x) / (x' * (SM_MAT(x) + SP_MAT(x)) * x);
    manifold = stiefelfactory(n, 1);
    problem.M = manifold;
    problem.cost = @(x) q_fun(x);
    problem.egrad = @(x) (2 / (x' * (SM_MAT(x) + SP_MAT(x)) * x)) * (SM_MAT(x) * x - q_fun(x) * (SM_MAT(x) + SP_MAT(x)) * x);
    options.verbosity = 0;
    options.tolgradnorm = tol;
    [x, ~, info, ~] = conjugategradient(problem, x0, options);
    Q = vertcat(info.cost);
    iter = vertcat(info.iter);
    iter = iter(end);

end
