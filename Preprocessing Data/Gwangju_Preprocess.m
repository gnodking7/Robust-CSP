%%% Gwangju Dataset %%%
% Preprocesses the dataset and saves it
%
% Dataset: http://gigadb.org/dataset/100295
%   52 subjects performing 2 motor imageries: left hand, right hand, 
%
% Utilizes BBCI Toolbox: https://github.com/bbci/bbci_public

band = [8 30];
filtOrder = 4;
[filt_b, filt_a] = butter(filtOrder, band/512 * 2);

Gwangju_Prep_Data = {};

for i = 1:52

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

    % Assign preprocessed data
    Gwangju_Prep_Data{i}.EEG = EEG;

end

% Save preprocessed data
save('Gwangju_Prep_Data.mat', 'Gwangju_Prep_Data', '-v7.3')
