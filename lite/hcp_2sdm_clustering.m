%% add path
addpath('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/src')

%% load data
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/data.mat')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/true_label/true_label_all_withrest_coarse.mat')
load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/cmap10.mat')
load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/cmap4_2.mat')
load('/Users/siyuangao/Working_Space/fmri/data/parcellation_and_network/map259.mat')

% choose less subjects for quicker computation
data = data(:, :, 1:30);

% % zscore the region
% for i_sub = 1 : size(data,3)
%     data(:, :, i_sub) = zscore(data(:, :, i_sub), 0, 2);
% end

task_endtime = [0,772,1086,1554,2084,2594,3020,4176];
% task_length = [772,314,468,530,510,426]
test_range = 3021:5382;
train_range = 1:3020;

test_data = data(:, test_range, :);
num_t_test = size(test_data, 2); % length of testing data

train_data = data(:, train_range, :); % task data to generate manifold
num_s = size(train_data, 3); % number of subjects
num_t_train = size(train_data, 2); % length of training data
num_r = size(train_data, 1); % number of regions
num_t_all = num_t_test+num_t_train; % total length of time


%% config the parameters
k = 500;
configAffParams1.dist_type = 'euclidean';
configAffParams1.kNN = k;
configAffParams1.self_tune = 0;

configDiffParams1.t = 1;
configDiffParams1.normalization='lb';

configAffParams2 = configAffParams1;
configDiffParams2 = configDiffParams1;

n_d = 7;

%% generate training embedding
% embed is the n_t*(n_sub*n_d) first round flatten matrix, dm is the final embedding
[dm, ~, embed, lambda1, lambda2, sigma1, sigma2] = calc2sDM(train_data,...
    n_d, configAffParams1, configAffParams2, configDiffParams1, configDiffParams2);

c = true_label_all(train_range);
vmin = -0.5;
vmax = 9.5;

figure;subplot(2,2,1);scatter3(dm(1, :), -dm(2, :), dm(3, :), 20, c, 'filled');
colormap(cmap10)
caxis([vmin,vmax])

subplot(2,2,2);scatter3(dm(1, :), -dm(2, :), dm(4, :), 20, c, 'filled');
colormap(cmap10)
caxis([vmin, vmax])

%% kmeans part
rng(665)
n_cluster = 4;
kmeans_idx = kmeans(dm(1:4, :)', n_cluster, 'Replicates', 100);
kmeans_idx(kmeans_idx==1)=8;
kmeans_idx(kmeans_idx==2)=6;
kmeans_idx(kmeans_idx==3)=5;
kmeans_idx(kmeans_idx==4)=7;

% kmeans_idx(kmeans_idx==1)=6;
% kmeans_idx(kmeans_idx==2)=8;
% kmeans_idx(kmeans_idx==3)=7;
% kmeans_idx(kmeans_idx==4)=5;
kmeans_idx = kmeans_idx-4;

figure;
scatter3(dm(1, :), -dm(2, :), dm(4, :), 20, kmeans_idx, 'filled');
colormap(cmap4)
caxis([0.5, 5.5])