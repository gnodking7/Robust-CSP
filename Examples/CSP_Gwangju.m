%%% Gwangju Dataset %%%
% Computes robust Common Spatial Filters (solves CSP-NRQ) and CSP
% and performs classification
% 
% Performs binary motor imagery classification tasks using filters
% and linear classifier
%
% Dataset: http://gigadb.org/dataset/100295
%   52 subjects performing 2 motor imageries: left hand, right hand, 
%
% Utilizes BBCI Toolbox: https://github.com/bbci/bbci_public

band = [8 30];
filtOrder = 4;
[filt_b, filt_a] = butter(filtOrder, band/512 * 2);

DELTA = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
MISSRATE = cell(1, 52);

for i = 35:52
    fprintf('%%%%%%%%%% STARTING Subject %d %%%%%%%%%% \n',i)

    %Load file
    if i < 10
        name = strcat('s0',string(i),'.mat');
    else
        name = strcat('s',string(i),'.mat');
    end
    DATA = load(name); %loads eeg

    %% Preprocessing Steps
    EVENT = find(DATA.eeg.imagery_event==1);    % marking for trial onset

    % Choose time segment 0.5 to 2.5 seconds after onset
    LEFT = []; RIGHT = [];
    for ind = EVENT
        LEFT = [LEFT, DATA.eeg.imagery_left(1:64, ind+256:ind+1280)];
        RIGHT = [RIGHT, DATA.eeg.imagery_right(1:64, ind+256:ind+1280)];
    end
    LEFT = reshape(LEFT, 64, [], length(EVENT));
    RIGHT = reshape(RIGHT, 64, [], length(EVENT));
    LEFT = permute(LEFT, [2 1 3]);
    RIGHT = permute(RIGHT, [2 1 3]);

    % Throw out bad trials
    LEFT(:, :, DATA.eeg.bad_trial_indices.bad_trial_idx_voltage{1}) = [];
    RIGHT(:, :, DATA.eeg.bad_trial_indices.bad_trial_idx_voltage{2}) = [];
    LEFT(:, :, DATA.eeg.bad_trial_indices.bad_trial_idx_mi{1}) = [];
    RIGHT(:, :, DATA.eeg.bad_trial_indices.bad_trial_idx_mi{2}) = [];

    % Create necessary structures for BBCI
    EEG.clab = cellstr(string(1:1:64));
    EEG.fs = DATA.eeg.srate;
    EEG.x = cat(3, LEFT, RIGHT);
    EEG.x = double(EEG.x);
    [~, ~, left_len] = size(LEFT);
    [~, ~, right_len] = size(RIGHT);
    EEG.y = [[ones(1, left_len); zeros(1, left_len)], [zeros(1, right_len); ones(1, right_len)]];
    EEG.className = {'left' 'right'};

    % Filter
    EEG = proc_filtfilt(EEG, filt_b, filt_a);

    % Randomly split number of trials into 10 subsets
    L_rnd_idx = randperm(left_len);
    R_rnd_idx = randperm(right_len);
    tmp_L = fix(left_len / 10) * ones(1, 10);
    tmp_R = fix(right_len / 10) * ones(1, 10);
    for r = 1:rem(left_len, 10)
        tmp_L(r) = tmp_L(r) + 1;
    end
    for r = 1:rem(right_len, 10)
        tmp_R(r) = tmp_R(r) + 1;
    end
    L_C = mat2cell(L_rnd_idx', tmp_L);
    R_C = mat2cell(R_rnd_idx', tmp_R);

    test_idx = nchoosek(1:10, 3); %each row an index of testing set

    MissRate = [];

    for k = 1:10
        fprintf('********** CV number %d ********** \n', k)

        IDX = test_idx(k, :);
        TR_IDX = []; TST_IDX = [];
        
        tst_idx = 1:1:10;

        for m = 1:3
            TR_IDX = [TR_IDX; L_C{IDX(m)}];
            TR_IDX = [TR_IDX; left_len + R_C{IDX(m)}];
            tst_idx(tst_idx==IDX(m)) = [];
        end
        for m = 1:7
            TST_IDX = [TST_IDX; L_C{tst_idx(m)}];
            TST_IDX = [TST_IDX; left_len + R_C{tst_idx(m)}];
        end

        TR_EPO.x = EEG.x(:, :, TR_IDX);
        TST_EPO.x = EEG.x(:, :, TST_IDX);
        TR_EPO.y = EEG.y(:, TR_IDX);
        TST_EPO.y = EEG.y(:, TST_IDX);

        % Extract datasets and covariance matrices
        [train_L, train_R, CovL, CovR] = Cov_Mats(TR_EPO);
        [test_L, test_R, ~, ~] = Cov_Mats(TST_EPO);

        % Compute interpolation matrices, average covariance matrices, robust
        % CSP weights (eigs)
        [V1, Sbar1, lam1] = PCA_V(CovL, 10);
        [V2, Sbar2, lam2] = PCA_V(CovR, 10);
    
        % Standard CSP
        [x_L, ~] = eigs(Sbar1, Sbar1 + Sbar2, 1, 'smallestreal');
        [x_R, ~] = eigs(Sbar2, Sbar1 + Sbar2, 1, 'smallestreal');

        % Robust CSP
        min_miss = 1;
        min_delta = 2;
        for j = 1:length(DELTA)
            delta = DELTA(j);
            [x1, x2] = SCF(delta, Sbar1, lam1, V1, Sbar2, lam2, V2, x_L, x_R, tol);
            cur_missrate = CSP_classifier(x1, x2, train_L, train_R, test_L, test_R);
            if cur_missrate < min_miss
                min_miss = cur_missrate;
                min_delta = delta;
            end
        end
        MissRate = [MissRate, [CSP_classifier(x_L, x_R, train_L, train_R, test_L, test_R); min_miss; min_delta]];
    end
    MISSRATE{i} = MissRate;

end

%%
% ++++++++++++++++++++++++++++++
%       Plot Results 
% ++++++++++++++++++++++++++++++

figure(1)
hold on
for i = 1:28
scatter(1 - mean(GWANGJU_MISSRATE{i}(1, :)), 1 - mean(GWANGJU_MISSRATE{i}(2, :)), 50, 'b', 'filled')
end
xlim([0.4, 1.0])
ylim([0.4, 1.0])
t = linspace(0.4, 1.0, 500);
plot(t, t, 'r', 'LineWidth', 1.5)


%%

%%%%%%%%%% FUNCTIONS %%%%%%%%%%

% ---------------------------------------------------- %
%   SCF: robust csp by scf
% ---------------------------------------------------- %
function [x1, x2] = SCF(delta, Sbar1, lam1, V1, Sbar2, lam2, V2, x_L, x_R, tol)
    % ------------------------------
    % --- Run situation 1 
    % ------------------------------
    SM_MAT = @(x) S_MAT(x, lam1, V1, delta, Sbar1, 0);
    SP_MAT = @(x) S_MAT(x, lam2, V2, delta, Sbar2, 1);

    [x1, Q1, ~, ~, ~, ~] = rbstCSP_SCF(SM_MAT, SP_MAT, x_L, tol);
    [SM, TSM] = SM_MAT(x1);
    [SP, TSP] = SP_MAT(x1);
    x1 = x1 / sqrt(x1' * (SM + SP) * x1);

    % ++++++++++++++++++++++++++++++
    %       Run situation 2 
    % ++++++++++++++++++++++++++++++
    SP_MAT = @(x) S_MAT(x, lam2, V2, delta, Sbar2, 0);
    SM_MAT = @(x) S_MAT(x, lam1, V1, delta, Sbar1, 1);
    
    [x2, Q2, ~, ~, ~, ~] = rbstCSP_SCF(SP_MAT, SM_MAT, x_R, tol);
    [SP, TSP] = SP_MAT(x2);
    [SM, TSM] = SM_MAT(x2);
    x2 = x2 / sqrt(x2' * (SM + SP) * x2);
end
