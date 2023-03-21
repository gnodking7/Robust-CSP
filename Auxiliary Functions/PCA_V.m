% ---------------------------------------------------- %
%   PCA_V: computes interpolation matrices, average covariance matrices,
%          and eigenvalues from PCA on covariance matrices
% ---------------------------------------------------- %

function [V, Sbar, lam] = PCA_V(S, k)
    % Input:
    %   S - cell array of covariance matrices
    %   k - Number of interpolation matrices to compute
    % Output:
    %   V - cell array of interpolation matrices
    %   Sbar - Average covariance matrix
    %   lam - cell array of eigenvalues
    
    [n, ~] = size(S{1});   % Size of covariance matrices
    S_mat = cell2mat(S);  % Concatenate covariance matrices in cell S; n x (n x length(S))
    S_vec = reshape(S_mat, n * n, []);    % Vectorize covariance matrices; n^2 x length(S)
    Sbar = mean(S_vec, 2);    % Mean of vectors
    Sbar = reshape(Sbar, n, n);   % Average covariance matrix; n x n
    Sbar = (Sbar + Sbar') / 2;

    CovM = cov(S_vec');
    [VV, E] = eigs(CovM, k, 'largestreal');
    lam = diag(E);
    if nnz(lam <= 0) > 0
        disp('UH OH, there are non-positive eigenvalues')
    end
    lam = num2cell(lam);

    V = cell(1, k);
    for i=1:k
        cur = reshape(VV(:, i), n, n);  % Transform eigenvector to interpolation matrix
        cur = (cur' + cur) / 2;  % Symmetrize
        V{i} = cur;
    end
end
