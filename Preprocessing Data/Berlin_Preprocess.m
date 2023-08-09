%%% Berlin Dataset %%%
% Preprocesses the dataset and saves it
%
% Dataset description: https://api-depositonce.tu-berlin.de/server/api/core/bitstreams/80d957c0-b06e-4e74-a4aa-3eb424e5ba1f/content
%   80 subjects performing 3 motor imageries: left hand, right hand, foot
%       (only 40 subjects available)
%
% Utilizes BBCI Toolbox: https://github.com/bbci/bbci_public

filtOrder = 2;  % for butter bandpass filter
band = [8 30];
[filt_b, filt_a] = butter(filtOrder, band / 100 * 2);   % filter interval

Berlin_Prep_Data = {};

for i = 1:40
    %load raw EEG data
    if i < 10
        name_tr = strcat('lap_imag_arrow0', string(i), '.mat');
        name_tst = strcat('lap_imag_fbarrow0', string(i), '.mat');
    else
        name_tr = strcat('lap_imag_arrow', string(i), '.mat');
        name_tst = strcat('lap_imag_fbarrow', string(i), '.mat');
    end
    DATA_tr = load(name_tr); DATA_tst = load(name_tst);
    tr_epo = DATA_tr.epo; tst_epo = DATA_tst.epo;
    

    %% Preprocessing Steps
    chn = [14:1:30, 33:1:39, 42:1:49, 51:1:57, 59:1:66, 68:1:74, 76:1:107]; %selecting 86 channels surrounding motor cortex
    tr_epo.x = tr_epo.x(:, chn, :);
    tst_epo.x = tst_epo.x(:, chn, :);

    % filter training and testing
    cnt{1} = proc_filtfilt(tr_epo, filt_b, filt_a);
    cnt{2} = proc_filtfilt(tst_epo, filt_b, filt_a);

    % Extract datasets and covariance matrices
    [train_L, train_R, CovL, CovR] = Cov_Mats(cnt{1});
    [test_L, test_R, ~, ~] = Cov_Mats(cnt{2});

    % Assign preprocessed data
    Berlin_Prep_Data{i}.train_L = train_L;
    Berlin_Prep_Data{i}.train_R = train_R;
    Berlin_Prep_Data{i}.CovL = CovL;
    Berlin_Prep_Data{i}.CovR = CovR;
    Berlin_Prep_Data{i}.test_L = test_L;
    Berlin_Prep_Data{i}.test_R = test_R;

end

% Save preprocessed data
save('Berlin_Prep_Data.mat', 'Berlin_Prep_Data', '-v7.3')
