%%% Data set IIIa—BCI competition III %%%
% Computes robust Common Spatial Filters (solves CSP-NRQ) and 
% performs classification of motor imageries on BCI competition III - IIIa
% dataset
%
% Dataset description: https://www.bbci.de/competition/iii/desc_IIIa.pdf
%   three subjects performing 4 motor imageries: left hand, right hand,
%   foot, tongue
%
% Utilizes BBCI Toolbox: https://github.com/bbci/bbci_public

subjects = {'l1', 'k3', 'k6'};

x0 = randn(length(Sbar1), 1);   % random initial vector
x0 = x0 / norm(x0);
tol = 1e-8;

DELTA = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5];

MISSRATE = [];
for i = 1:3
    fprintf('%%%%%%%%%% STARTING Subject %d %%%%%%%%%% \n', i)

    % Load file -> gives s and HDR
    % Load true label
    name = strcat(subjects{i}, 'b.mat');
    load(name) %s and HDR
    label_name = strcat('true_label_', subjects{i}, '.txt');
    true_label = importdata(label_name);

    % Get prprocessed data structures
    [train_CNT, test_CNT] = preprocesser(s, HDR, true_label);

    % Extract datasets and covariance matrices
    [train_L, train_R, CovL, CovR] = Cov_Mats(train_CNT);
    [test_L, test_R, ~, ~] = Cov_Mats(test_CNT);

    % Compute interpolation matrices, average covariance matrices, robust
    % CSP weights (eigs)
    k = 10;
    [V1, Sbar1, lam1] = PCA_V(CovL, k);
    [V2, Sbar2, lam2] = PCA_V(CovR, k);

    % Standard CSP
    [x_L, ~] = eigs(Sbar1, Sbar1 + Sbar2, 1, 'smallestreal');
    [x_R, ~] = eigs(Sbar2, Sbar1 + Sbar2, 1, 'smallestreal');

    MissRate_SCF = [];
    % Robust CSP
    for j = 1:length(DELTA)
        delta = DELTA(j);
        fprintf('%%%%%%%%%% delta %d %%%%%%%%%% \n', delta)
        [x1, x2] = SCF(delta, Sbar1, lam1, V1, Sbar2, lam2, V2, x_L, x_R, tol);

        MissRate_SCF = [MissRate_SCF; CSP_classifier(x1, x2, train_L, train_R, test_L, test_R)];

    end

    MissRate = CSP_classifier(x_L, x_R, train_L, train_R, test_L, test_R);
    MISSRATE = [MISSRATE, [MissRate; MissRate_SCF]];

end

% MISSRATE is a matrix array of missrates; 
% each column corresponds a subject, first value is standard CSP followed
% by robust CSP for each delta

%% Plots of classification rate

figure(1)
plot(1 - MISSRATE(2:end, 1), '-+', 'LineWidth', 3, 'MarkerSize', 8);
yline(1 - MISSRATE(1, 1), '--', 'LineWidth', 2)
xticklabels(string(DELTA));
ax = gca;
ax.XAxis.FontSize = 13;
xlabel('\delta_c', 'Fontsize', 20);
ylabel('classification rate', 'Fontsize', 20);
h_legend=legend('Alg.1', 'non-rbst', 'Location', 'northwest');
set(h_legend, 'FontSize', 20);
title('Subject l1', 'FontSize', 20);

figure(2)
plot(1 - MISSRATE(2:end, 2), '-+', 'LineWidth', 3, 'MarkerSize', 8);
yline(1 - MISSRATE(1, 2), '--', 'LineWidth', 2)
xticklabels(string(DELTA));
ax = gca;
ax.XAxis.FontSize = 13;
xlabel('\delta_c', 'Fontsize', 20);
ylabel('classification rate', 'Fontsize', 20);
h_legend=legend('Alg.1', 'non-rbst', 'Location', 'northwest');
set(h_legend, 'FontSize', 20);
title('Subject k3', 'FontSize', 20);

figure(3)
plot(1 - MISSRATE(2:end, 3), '-+', 'LineWidth', 3, 'MarkerSize', 8);
yline(1 - MISSRATE(1, 3), '--', 'LineWidth', 2)
xticklabels(string(DELTA));
ax = gca;
ax.XAxis.FontSize = 13;
xlabel('\delta_c', 'Fontsize', 20);
ylabel('classification rate', 'Fontsize', 20);
h_legend=legend('Alg.1', 'non-rbst', 'Location', 'northwest');
set(h_legend, 'FontSize', 20);
title('Subject k6', 'FontSize', 20);



%%

%%%%%%%%%% FUNCTIONS %%%%%%%%%%

% ---------------------------------------------------- %
%   preprocesser: preprocess EEG signals
% ---------------------------------------------------- %
function [train_CNT, test_CNT] = preprocesser(s, h, label)
    % Event types are:
    %   768 - beginning of trial
    % 	785 - beep, fixation cross
    % 	786 - same as 785
    % 	771, 772, 770, 769 - trigger (four classes: 3, 4, 2, 1)
    % 		1 - left, 2 - right
    % 	783 - unknown class (for testing)
    % 	1023 - artifact indication
    %
    % Input:
    %   s - raw data
    %   h - markers data
    %   label - true labels
    % Output:
    %   train_CNT, test_CNT - segmented and filtered BBCI data structure
    
    % Add/create necessary structures (for BBCI Toolbox)
    aa = cellstr(string(1:1:60));
    new_cnt.clab = aa;
    new_cnt.fs = 250;
    new_cnt.T = length(s);
    new_cnt.x = s;
    
    mrk.time = h.EVENT.POS'; % Time when event occurs
    type = h.EVENT.TYP; % Numbered types of event
    tmp_type = type;
    % Discard non-motory imagery events
    tmp_type(tmp_type==768) = [];
    tmp_type(tmp_type==785) = [];
    tmp_type(tmp_type==786) = [];
    tmp_type(tmp_type==1023) = [];
    ind = find(tmp_type==783); %index of unknown trials (used for testing)
    counter = 1;
    for i = 1:length(type)
        if type(i) == 783
            type(i) = label(ind(counter)) + 668; %replace 783 (testing) with new type values
            counter = counter + 1;
        end
    end
    mrk.event.desc = type;
    train_mrk = mrk_defineClasses(mrk, {769 770; 'left' 'right'});  % BBCI Toolbox function
    test_mrk = mrk_defineClasses(mrk, {669 670; 'left' 'right'});
    
    %Segment and filter
    train_epo = proc_segmentation(new_cnt, train_mrk, [0 4000]); % BBCI Toolbox function; this chooses epoch length incorrectly; but still same size structure
    test_epo = proc_segmentation(new_cnt, test_mrk, [0 4000]);
    ival = [0 750]; % Use EEG signals up to 3 seconds after trigger

    train_nMarkers = length(train_mrk.time);
    train_len_sa = diff(ival);
    train_pos_zero = train_mrk.time;
    train_pos_end = train_pos_zero + ival(2);
    train_IV = [-train_len_sa + 1:0]' * ones(1, train_nMarkers) + ones(train_len_sa, 1) * train_pos_end;
    train_epo.x = reshape(new_cnt.x(train_IV, :), [train_len_sa train_nMarkers 60]);
    train_epo.x = permute(train_epo.x, [1 3 2]);
    
    test_nMarkers = length(test_mrk.time);
    test_len_sa = diff(ival);
    test_pos_zero = test_mrk.time;
    test_pos_end = test_pos_zero + ival(2);
    test_IV = [-test_len_sa+1:0]' * ones(1, test_nMarkers) + ones(test_len_sa, 1) * test_pos_end;
    test_epo.x = reshape(new_cnt.x(test_IV, :), [test_len_sa test_nMarkers 60]);
    test_epo.x = permute(test_epo.x, [1 3 2]);
    
    % Bandpass filter
    band = [8 30];
    filtOrder = 5;
    [filt_b, filt_a] = butter(filtOrder, band/new_cnt.fs * 2);
    
    train_CNT = proc_filtfilt(train_epo, filt_b, filt_a); % BBCI Toolbox function
    test_CNT = proc_filtfilt(train_epo, filt_b, filt_a);
end

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
