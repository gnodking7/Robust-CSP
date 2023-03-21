% ---------------------------------------------------- %
%   Cov_Mats: computes trials and covariance matrices
%             for Left hand and Right hand motor imagery
% ---------------------------------------------------- %

function [L, R, CovL, CovR] = Cov_Mats(dat)
    % Input: 
    %   dat - in BBCI data structure form
    % Output:
    %   L, R - cell array of data matrices for Left hand and Right hand
    %   CovL, CovR - cell array of covariance matrices for Left hand and Right
    %   hand

    idx1 = find(dat.y(1, :));    %index of first motor
    idx2 = find(dat.y(2, :));    %index of second motor
    
    n1 = length(idx1);
    n2 = length(idx2);
    
    L = cell(1, n1); R = cell(1, n2);
    CovL = cell(1, n1); CovR = cell(1, n2);
        
    x1 = dat.x(:, :, idx1);
    x2 = dat.x(:, :, idx2);
    
    for j = 1:length(idx1)
        data = x1(:, :, j);
        L{j} = data';
        CovL{j} = cov(data);
    end
    
    for j = 1:length(idx2)
        data = x2(:, :, j);
        R{j} = data';
        CovR{j} = cov(data);
    end
end
