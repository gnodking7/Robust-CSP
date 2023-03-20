% ---------------------------------------------------- %
%   logvar_feat: log-variance features of the filtered signals
% ---------------------------------------------------- %

function F = logvar_feat(x1, x2, T)
    % Input:
    %   x1, x2 - spatial filter for each condition
    %   T - cell array of trials
    % Output:
    %   F - log-variance features of the filtered signals
    %       2 x length(trial)

    P = [x1, x2];
    F = [];
    for i = 1:length(T)
        Y = T{i};
        YY = P' * Y;
        f = log(var(YY, 0, 2));
        F = [F, f];
    end

end