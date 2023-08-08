% ---------------------------------------------------- %
%   rbstCSP_Man_TR: Manopt on Sphere manifold
%           Trust Region
% ---------------------------------------------------- %
%
% Utilizes Manopt Toolbox: https://www.manopt.org

function [x, Q, iter] = rbstCSP_Man_TR(SM_MAT, SP_MAT, x0, tol)
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
    nn = @(x) x' * (S_MAT_FIRST(SM_MAT, x) + S_MAT_FIRST(SP_MAT, x)) * x;
    nnx = @(x) (S_MAT_FIRST(SM_MAT, x) + S_MAT_FIRST(SP_MAT, x)) * x;
    q_fun = @(x) (x' * S_MAT_FIRST(SM_MAT, x) * x) / nn(x);
    q_til_fun = @(x) q_fun(x) / nn(x);
    q_til_der = @(x) (2 / (nn(x) ^ 2)) * (S_MAT_FIRST(SM_MAT, x) - 2 * q_til_fun(x) * nn(x) * nnx(x));
    manifold = stiefelfactory(n, 1);    % sphere manifold
    problem.M = manifold;
    problem.cost = @(x) q_fun(x);
    problem.egrad = @(x) (2 / nn(x)) * (S_MAT_FIRST(SM_MAT, x) * x - q_fun(x) * nnx(x));
    options.verbosity = 0;
    options.tolgradnorm = tol;
    [x, ~, info] = trustregions(problem, x0, options);
    Q = vertcat(info.cost);
    iter = vertcat(info.iter);
    iter = iter(end);

end



%%%%%%%%%% FUNCTIONS %%%%%%%%%%

% ---------------------------------------------------- %
%   S_MAT_FIRST: computes SM (SP)
% ---------------------------------------------------- %
function S = S_MAT_FIRST(S_MAT, x)
    % Input:
    %   S_MAT - function handle
    %            \Sigma(x) = S_MAT(x)
    %            [\Sigma(x) \tilde{\Sigma}(x)] = S_MAT(x)
    %   x - vector
    % Output:
    %   S - \Sigma(x)
    S = S_MAT(x);
end

% ---------------------------------------------------- %
%   S_MAT_SECOND: computes tilde SM (tilde SP)
% ---------------------------------------------------- %
function TS = S_MAT_SECOND(S_MAT, x)
    % Input:
    %   S_MAT - function handle
    %            \Sigma(x) = S_MAT(x)
    %            [\Sigma(x) \tilde{\Sigma}(x)] = S_MAT(x)
    %   x - vector
    % Output:
    %   TS - \tilde{\Sigma}(x)
    [~, TS] = S_MAT(x);
end
