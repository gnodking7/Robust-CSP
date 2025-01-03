% ---------------------------------------------------- %
%   optmat_build: Compute expansion of PCA interpolation
%                 and derivatives
%                 for generating the matrices 
%                 G(w) = Sigma + optmat_build(w, lam, V, delta)
%                 H(w) = Sigma - optmat_build(w, lam, V, delta)
% ---------------------------------------------------- %

function [P, dP] = optmat_build(x, lam, V, delta)
    % Input:
    %   x - vector to evaluate at
    %   lam - cell array of weights (PCA eigs)
    %   V - cell array of interpolation matrices
    %   delta - radius of tolerance region
    % Output:
    %   P - \sum_{i=1}^m\alpha_{c}^{(i)}(x)V_{c}^{(i)}
    %   dP - \sum_{i=1}^m\nabla\alpha_c^{(i)}(x)x^TV_c^{(i)}

    % Evaluate P
    P = zeros(size(V{1}));
    xtVx_fun = @(Vi) x'*Vi*x;
    xtVx = cellfun(xtVx_fun, V);
    nn = sqrt(xtVx.^2 * cell2mat(lam));
    
    for i = 1:length(V)
        alpha = delta * (lam{i} * xtVx(i)) / nn;
        P = P + alpha * V{i};
    end
    P = (P + P') / 2;
    
    % Evaluate dP
    S1 = 0; S2 = 0;
    for i = 1:length(V)
        S1 = S1 + (V{i}' * x) * (x' * V{i}) * lam{i};
        S2 = S2 + xtVx(i) * (V{i}' * x) * lam{i};
    end
    
    dP = 2 * delta * (S1 / nn - (S2 * S2') / (nn^3));
    
end
