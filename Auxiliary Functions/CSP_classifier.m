% ---------------------------------------------------- %
%   CSP_classifier: a linear classifier for the 
%       log-variance features of the filtered signals
% ---------------------------------------------------- %

function MissRate = CSP_classifier(x1, x2, Train1, Train2, Tst1, Tst2)
    % Input:
    %   x1, x2 - spatial filter for each condition
    %   Train1, Train2 - cell array of train trials
    %   Tst1, Tst2 - cell array of test trials
    % Output:
    %   MissRate - percentage of miss classification

    Tr1 = logvar_feat(x1, x2, Train1);
    Tr2 = logvar_feat(x1, x2, Train2);
    Ts1 = logvar_feat(x1, x2, Tst1);
    Ts2 = logvar_feat(x1, x2, Tst2);

    % LDA classifier
    [w, c1, c2] = lda_2d(Tr1, Tr2);
    % For data f
    % if projected w' * f closer to c1 than c2, then indicates class 1,
    % otherwise class 2
    Tsw1 = w' * Ts1; Tsw2 = w' * Ts2;
    NumMiss = nnz(abs(Tsw1-c1) > abs(Tsw1-c2));
    NumMiss = NumMiss + nnz(abs(Tsw2-c2) > abs(Tsw2-c1));
    MissRate = NumMiss / (length(Tsw1)+length(Tsw2));

end
