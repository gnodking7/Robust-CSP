%%% Data set IIIa—BCI competition III %%%
% Preprocesses the dataset and saves it
%
% Dataset description: https://www.bbci.de/competition/iii/desc_IIIa.pdf
%   three subjects performing 4 motor imageries: left hand, right hand,
%   foot, tongue
%
% Utilizes BBCI Toolbox: https://github.com/bbci/bbci_public

subjects = {'l1', 'k3', 'k6'};

Comp3_3a_Prep_Data = {};

for i = 1:3
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

    % Assign preprocessed data
    Comp3_3a_Prep_Data{i}.train_L = train_L;
    Comp3_3a_Prep_Data{i}.train_R = train_R;
    Comp3_3a_Prep_Data{i}.CovL = CovL;
    Comp3_3a_Prep_Data{i}.CovR = CovR;
    Comp3_3a_Prep_Data{i}.test_L = test_L;
    Comp3_3a_Prep_Data{i}.test_R = test_R;

end

% Save preprocessed data
save('Comp3_3a_Prep_Data.mat', 'Comp3_3a_Prep_Data')

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
