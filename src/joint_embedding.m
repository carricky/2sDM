%% Emotion data
cut = 1; % whether to cut the first fixation
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/network_label_259.mat')

% LR data
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signal_LR_EMOTION_NOGSR.mat')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/true_label_EMOTION_RL.mat')

missing_nodes = [249, 239, 243, 129, 266, 109, 115, 118, 250];
signal_LR(missing_nodes, :, :) = [];

true_label_LR = true_label;

% RL data
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signal_RL_EMOTION_NOGSR.mat')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/true_label_EMOTION_RL.mat')
true_label_RL = true_label;
signal_RL(missing_nodes, :, :) = [];

% cut the beginning period
if cut == 1
    emo_data = cat(2, signal_LR(:, 10:end-5, :), signal_RL(:, 10:end-5, :));
    emo_true_label = [true_label_LR(10:end-5); true_label_RL(10:end-5)];
else
% don't cut the beginning period
    emo_data = cat(2, signal_LR(:, :, :), signal_RL(:, :, :));
    emo_true_label = [true_label_LR; true_label_RL];
end

%% WM data
cut = 1; % whether to cut the first fixation
% LR data
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signal_LR_WM_NOGSR.mat')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/true_label_WM_LR.mat')

missing_nodes = [249, 239, 243, 129, 266, 109, 115, 118, 250];
signal_LR(missing_nodes, :, :) = [];

% get the visual roi
%     signal_LR = signal_LR(network_label, :, :);

true_label_LR = true_label;
task_order_LR = 'Tl Bd Fa Tl Bd Pl Fa Pl';

% RL data
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signal_RL_WM_NOGSR.mat')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/true_label_WM_RL.mat')
true_label_RL = true_label;
signal_RL(missing_nodes, :, :) = [];

% cut the beginning period
if cut == 1
    wm_data = cat(2, signal_LR(:, 19:end-5, :), signal_RL(:, 19:end-5, :));
    wm_true_label = [true_label_LR(19:end-5); true_label_RL(19:end-5)];
else
% don't cut the beginning period
    wm_data = cat(2, signal_LR(:, :, :), signal_RL(:, :, :));
    wm_true_label = [true_label_LR; true_label_RL];
end

%% Rest data
cut = 1; % whether to cut the first fixation
% LR data
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signal_LR_REST_NOGSR.mat')

missing_nodes = [249, 239, 243, 129, 266, 109, 115, 118, 250];
signal_LR(missing_nodes, :, :) = [];

% get the visual roi
%     signal_LR = signal_LR(network_label, :, :);

% % RL data
% load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signal_RL_WM_NOGSR.mat')
% load('/Users/siyuangao/Working_Space/fmri/data/HCP515/true_label_WM_RL.mat')
% true_label_RL = true_label;
% signal_RL(missing_nodes, :, :) = [];

% cut the beginning period
if cut == 1
    rest_data = signal_LR(:, 19:end-5, :);
else
% don't cut the beginning period
    rest_data = signal_LR(:, :, :);
end

%% combine data
data = cat(2, wm_data, emo_data);
data(:, :, 515) = [];
data = cat(2, data, rest_data);
wm_true_label(wm_true_label==2) = 4;
wm_true_label(wm_true_label==3) = 5;
true_label = [wm_true_label; emo_true_label; 6*ones(size(rest_data,2), 1)];
