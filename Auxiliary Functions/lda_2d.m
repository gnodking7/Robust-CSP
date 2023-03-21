% ---------------------------------------------------- %
%   lda_2d: linear discriminant analysis for 2-dim data
% ---------------------------------------------------- %

function [w, c1, c2] = lda_2d(T1, T2)
    % Input:
    %   T1, T2 - data matrix
    %       2 x m : 2 features, m samples
    % Output:
    %   w - weight vector
    %   c1, c2 - projected mean values

    S1 = cov(T1'); S2 = cov(T2');
    m1 = mean(T1, 2); m2 = mean(T2, 2);

    w = (S1 + S2) \ (m1 - m2);
    w = w / norm(w);
    c1 = w' * m1;
    c2 = w' * m2;
end
