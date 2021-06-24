% clc;
% clear all;
addpath('/Users/siyuangao/Working_Space/fmri/code_siyuan/diffusion_maps-master/')
%% WM data
cut = 1; % whether to cut the first fixation

% LR data
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/signal_LR_WM_NOGSR.mat')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/true_label/with_cue/wm_true_label_lr.mat')
true_label_LR = [zeros(1, 8), wm_true_label_lr(1:end-8)];

% RL data
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/signal_RL_WM_NOGSR.mat')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/true_label/with_cue/wm_true_label_rl.mat')
true_label_RL = [zeros(1, 8), wm_true_label_rl(1:end-8)];

% cut the beginning period
if cut == 1
    data = cat(2, signal_LR(:, 10:end-10, :), signal_RL(:, 10:end-10, :));
    true_label_all = [true_label_LR(10:end-10), true_label_RL(10:end-10)];
else
% don't cut the beginning period
    data = cat(2, signal_LR(:, :, :), signal_RL(:, :, :));
    true_label_all = [true_label_LR, true_label_RL];
end

%% EMOTION data
cut = 1; % whether to cut the first fixation

% LR data
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/signal_LR_EMOTION_NOGSR.mat')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/true_label/with_cue/emotion_true_label_lr.mat')
true_label_LR = [zeros(1, 8), emotion_true_label_lr(1:end-8)];

% RL data
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/signal_RL_EMOTION_NOGSR.mat')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/true_label/with_cue/emotion_true_label_rl.mat')
true_label_RL = [zeros(1, 8), emotion_true_label_rl(1:end-8)];

true_label_LR(true_label_LR>0.5)=max(true_label_all)+true_label_LR(true_label_LR>0.5);
true_label_RL(true_label_RL>0.5)=max(true_label_all)+true_label_RL(true_label_RL>0.5);

% cut the beginning period
if cut == 1
    data = cat(2, data, cat(2, signal_LR(:, 10:end-10, :), signal_RL(:, 10:end-10, :)));
    true_label_all = [true_label_all,true_label_LR(10:end-10), true_label_RL(10:end-10)];
else
% don't cut the beginning period
    data = cat(2, data, cat(2, signal_LR, signal_RL));
    true_label_all = [true_label_all,true_label_LR, true_label_RL];
end
%% GAMBLING data
cut = 1; % whether to cut the first fixation

% LR data
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/signal_LR_GAMBLING_NOGSR.mat')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/true_label/with_cue/gambling_true_label_lr.mat')
true_label_LR = [zeros(1, 8), gambling_true_label_lr(1:end-8)];

% RL data
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/signal_RL_GAMBLING_NOGSR.mat')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/true_label/with_cue/gambling_true_label_rl.mat')
true_label_RL = [zeros(1, 8), gambling_true_label_rl(1:end-8)];

true_label_LR(true_label_LR>0.5)=max(true_label_all)+true_label_LR(true_label_LR>0.5);
true_label_RL(true_label_RL>0.5)=max(true_label_all)+true_label_RL(true_label_RL>0.5);

% cut the beginning period
if cut == 1
    data = cat(2, data, cat(2, signal_LR(:, 10:end-10, :), signal_RL(:, 10:end-10, :)));
    true_label_all = [true_label_all,true_label_LR(10:end-10), true_label_RL(10:end-10)];
else
% don't cut the beginning period
    data = cat(2, data, cat(2, signal_LR, signal_RL));
    true_label_all = [true_label_all,true_label_LR, true_label_RL];
end

%% MOTOR data
cut = 1; % whether to cut the first fixation

% LR data
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/signal_LR_MOTOR_NOGSR.mat')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/true_label/with_cue/motor_true_label_lr.mat')
true_label_LR = [zeros(1, 8), motor_true_label_lr(1:end-8)];

% RL data
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/signal_RL_MOTOR_NOGSR.mat')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/true_label/with_cue/motor_true_label_rl.mat')
true_label_RL = [zeros(1, 8), motor_true_label_rl(1:end-8)];

true_label_LR(true_label_LR>0.5)=max(true_label_all)+true_label_LR(true_label_LR>0.5);
true_label_RL(true_label_RL>0.5)=max(true_label_all)+true_label_RL(true_label_RL>0.5);

% cut the beginning period
if cut == 1
    data = cat(2, data, cat(2, signal_LR(:, 10:end-10, :), signal_RL(:, 10:end-10, :)));
    true_label_all = [true_label_all,true_label_LR(10:end-10), true_label_RL(10:end-10)];
else
% don't cut the beginning period
    data = cat(2, data, cat(2, signal_LR, signal_RL));
    true_label_all = [true_label_all,true_label_LR, true_label_RL];
end

%% SOCIAL data
cut = 1; % whether to cut the first fixation

% LR data
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/signal_LR_SOCIAL_NOGSR.mat')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/true_label/with_cue/social_true_label_lr.mat')
true_label_LR = [zeros(1, 8), social_true_label_lr(1:end-8)];

% RL data
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/signal_RL_SOCIAL_NOGSR.mat')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/true_label/with_cue/social_true_label_rl.mat')
true_label_RL = [zeros(1, 8), social_true_label_rl(1:end-8)];

true_label_LR(true_label_LR>0.5)=max(true_label_all)+true_label_LR(true_label_LR>0.5);
true_label_RL(true_label_RL>0.5)=max(true_label_all)+true_label_RL(true_label_RL>0.5);

% cut the beginning period
if cut == 1
    data = cat(2, data, cat(2, signal_LR(:, 10:end-10, :), signal_RL(:, 10:end-10, :)));
    true_label_all = [true_label_all,true_label_LR(10:end-10), true_label_RL(10:end-10)];
else
% don't cut the beginning period
    data = cat(2, data, cat(2, signal_LR, signal_RL));
    true_label_all = [true_label_all,true_label_LR, true_label_RL];
end

%% RELATIONAL data
cut = 1; % whether to cut the first fixation

% LR data
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/signal_LR_RELATIONAL_NOGSR.mat')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/true_label/with_cue/relational_true_label_lr.mat')
true_label_LR = [zeros(1, 8), relational_true_label_lr(1:end-8)];

% RL data
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/signal_RL_RELATIONAL_NOGSR.mat')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/true_label/with_cue/relational_true_label_rl.mat')
true_label_RL = [zeros(1, 8), relational_true_label_rl(1:end-8)];

true_label_LR(true_label_LR>0.5)=max(true_label_all)+true_label_LR(true_label_LR>0.5);
true_label_RL(true_label_RL>0.5)=max(true_label_all)+true_label_RL(true_label_RL>0.5);

% cut the beginning period
if cut == 1
    data = cat(2, data, cat(2, signal_LR(:, 10:end-10, :), signal_RL(:, 10:end-10, :)));
    true_label_all = [true_label_all,true_label_LR(10:end-10), true_label_RL(10:end-10)];
else
% don't cut the beginning period
    data = cat(2, data, cat(2, signal_LR, signal_RL));
    true_label_all = [true_label_all,true_label_LR, true_label_RL];
end

%% REST data
cut = 1; % whether to cut the first fixation

% LR data
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/signal_LR_REST_NOGSR.mat')
label = max(true_label_all)+1;
true_label_LR = label*ones(1, 1200);

% RL data
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/signal_RL_REST_NOGSR.mat')
true_label_RL = label*ones(1, 1200);

% cut the beginning period
if cut == 1
    data = cat(2, data, cat(2, signal_LR(:, :, :), signal_RL(:, :, :)));
    true_label_all = [true_label_all,true_label_LR(10:end-10), true_label_RL(10:end-10)];
else
% don't cut the beginning period
    data = cat(2, data, cat(2, signal_LR, signal_RL));
    true_label_all = [true_label_all,true_label_LR, true_label_RL];
end

%% REST2 data
% cut = 1; % whether to cut the first fixation
% 
% % LR data
% load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/signal_LR_REST2_NOGSR.mat')
% label = max(true_label_all)+1;
% true_label_LR = label*ones(1, 1200);
% 
% % RL data
% load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/signal_RL_REST2_NOGSR.mat')
% true_label_RL = label*ones(1, 1200);
% 
% % cut the beginning period
% if cut == 1
%     data = cat(2, data, cat(2, signal_LR(:, 10:end-10, :), signal_RL(:, 10:end-10, :)));
%     true_label_all = [true_label_all,true_label_LR(10:end-10), true_label_RL(10:end-10)];
% else
% % don't cut the beginning period
%     data = cat(2, data, cat(2, signal_LR, signal_RL));
%     true_label_all = [true_label_all,true_label_LR, true_label_RL];
% end

%% transform cue's label
true_label_all(true_label_all>0.5) = true_label_all(true_label_all>0.5)+1;
true_label_all(true_label_all==0.5) = 1;

%% remove subs with unaligned subjects and regions
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/good_id_pos_relational.mat')
missing_nodes = [249, 239, 243, 129, 266, 109, 115, 118, 250];
data = data(:, :, good_id_pos);
data(missing_nodes, :, :) = [];

%% zscore the data
n_subs = size(data, 3);
for i = 1 : n_subs
    data(:, :, i) = zscore(data(:, :, i), 0, 1);
end
 
