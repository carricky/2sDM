%% load data
% HCP
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/data.mat')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/true_label/true_label_all_with_rest.mat')
load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/cmap18.mat')

% choose less subjects for quicker computation
train_data = data(:, :, 1:10);
% leftout WM task
train_idx = 773:3020;

train_data = train_data(:, train_idx, :);
n_sub = size(train_data, 3); % num of subs
n_tp_train = size(train_data, 2); % num of time points
n_rg = size(train_data, 1); % num of regions

tar_data = data(:, 1:772, 11);
n_tp_test = size(tar_data, 2);

% UCLA
% ref_sub = 122;
% % rest
% % load('/Users/siyuangao/Working_Space/fmri/data/UCLA/rest199.mat')
% % missing_nodes = [249, 239, 243, 129, 266, 109, 115, 118, 250];
% % all_signal(missing_nodes, :, :) = [];
% tar_data = all_signal(:, :, ref_sub);
% tar_data = zscore(tar_data, 0, 1);
% n_tp_test = size(tar_data, 2);

% task
% load('/Users/siyuangao/Working_Space/fmri/data/UCLA/all_task.mat')
% load('/Users/siyuangao/Working_Space/fmri/data/UCLA/all_label.mat')
% missing_nodes = [249, 239, 243, 129, 266, 109, 115, 118, 250];
% all_task(missing_nodes, :, :) = [];
% test_task = 4;
% tar_data = all_task(:, label==test_task, 1);
% n_tp_test = sum(label==test_task);

% final config
rng(665)
n_tp_total = n_tp_train+n_tp_test;
train_idx = 1 : n_tp_train;
test_idx = n_tp_train+1:n_tp_total;

%% config the parameters
k = 500;
configAffParams1.dist_type = 'euclidean';
configAffParams1.kNN = k;
configAffParams1.self_tune = 0;

configDiffParams1.t = 1;
configDiffParams1.normalization='lb';

configAffParams2 = configAffParams1;
configDiffParams2 = configDiffParams1;

n_dim = 7;

%% 2sDM for training data (can skip if run multiple times)
[dm_train, ~, embed, lambda1, lambda2, sigma1, sigma2] = calc2sDM(train_data, n_dim, configAffParams1, configAffParams2, configDiffParams1, configDiffParams2);
k = 4;
IDX = kmeans(dm_train(:, :)', k);

%% extend new time points for any subject
% compute subject-wise filter
filter_set = zeros(n_rg, n_dim, n_sub);
for i_sub = 1 : n_sub
    fprintf('computing filter for %dth sub\n', i_sub)
    % zscore by region for region embedding
    train_data_temp = train_data(:, :, i_sub)';
    train_data_temp = zscore(train_data_temp);
    
    K_temp = calcAffinityMat(train_data_temp, configAffParams1);
    dm_temp = calcDiffusionMap(K_temp, configDiffParams1);
    dm_temp = dm_temp(1:n_dim, :)';
    filter_set(:, :, i_sub) = dm_temp;
end
% plot any filter/region embedding/gradient
figure;
load('/Users/siyuangao/Working_Space/fmri/data/map259.mat')
subplot(2,1,1);
scatter3(filter_set(:, 1, 1), filter_set(:, 2, 1), filter_set(:, 3, 1), 20, map259, 'filled');
colormap(gca, cmap)
subplot(2,1,2);
imagesc(filter_set(:, :, 1));
colormap(gca, parula)

%% simulate time points for target subject
tar_data = zscore(tar_data, 0, 2); % zscore by region
filter_data = zeros(n_tp_total, n_dim, n_sub);
for i_sub = 1 : n_sub
    fprintf('filter data for %dth sub\n', i_sub)
    % filter data
    train_data_temp = train_data(:, :, i_sub);
    filter_data(train_idx, :, i_sub) = train_data_temp' * filter_set(:, :, i_sub);
    filter_data(test_idx, :, i_sub) = tar_data' * filter_set(:, :, i_sub);
end
filter_data = reshape(filter_data, n_tp_total, n_dim*n_sub);
filter_data = filter_data';
[K, ~] = calcAffinityMat(filter_data, configAffParams1);
K = K(n_tp_train+1:n_tp_total, 1:n_tp_train);
K = K./sum(K, 2);
psi2 = K*dm_train'./lambda2';
dm_all = [dm_train, psi2'];
% plot
s = [10*ones(1, n_tp_train),40*ones(1, n_tp_test)];
% c = [true_label_all(train_idx), 17*ones(1, n_tp_test)];
% c = [IDX', 17*ones(1, n_tp_test)];
c = [true_label_all(773:3020), true_label_all(1:772)];
vmin = -0.5;
vmax = 17.5;
figure;
subplot(2,2,1);
scatter3(dm_all(1, :), dm_all(2, :), dm_all(3, :), s, c, 'filled');
colormap(cmap)
caxis([vmin,vmax])
title('concatenated embed, 123coord')
subplot(2,2,2);
scatter3(dm_all(2, :), dm_all(3, :), dm_all(4, :), s, c, 'filled');
colormap(cmap)
caxis([vmin,vmax])
title('concatenatedembed, 234coord')

[D_ext, I_ext] = pdist2(dm_all(:, train_idx)', dm_all(:, test_idx)', 'Euclidean', 'Smallest', 10);
IDX_list = zeros(size(I_ext, 2), 1);
for i = 1 : size(I_ext, 2)
    IDX_temp = mode(IDX(I_ext(:, i)));
    IDX_list(i) = IDX_temp;
end
for i = 1 : 4
    dwell_time_list(ref_sub, i) = (sum(IDX_list==i)/numel(IDX_list));
end




% embed using simulated group data, only testing points
[K_temp, ~, ~] = calcAffinityMat(filter_data(:, test_idx), configAffParams2);
[dm_new, ~, ~, ~, ~, ~] = calcDiffusionMap(K_temp, configDiffParams2);
% plot
s = 20;
c = true_label_all(test_idx);
vmin = -0.5;
vmax = 17.5;
figure;
subplot(2,2,1);
scatter3(dm_new(1, :), dm_new(2, :), dm_new(3, :), s, c, 'filled');
colormap(cmap)
caxis([vmin,vmax])
title('simulated group embed, 123coord')
subplot(2,2,2);
scatter3(dm_new(2, :), dm_new(3, :), dm_new(4, :), s, c, 'filled');
colormap(cmap)
caxis([vmin,vmax])
title('simulated group embed, 234coord')

% embed using single data, only testing points
[K_temp2, ~, ~] = calcAffinityMat(tar_data, configAffParams2);
[dm_new, ~, ~, ~, ~, ~] = calcDiffusionMap(K_temp2, configDiffParams2);
% plot
s = 20;
c = true_label_all(test_idx);
vmin = -0.5;
vmax = 17.5;
subplot(2,2,3);
scatter3(dm_new(1, :), dm_new(2, :), dm_new(3, :), s, c, 'filled');
colormap(cmap)
caxis([vmin,vmax])
title('single sub embed, 123coord')
subplot(2,2,4);
scatter3(dm_new(2, :), dm_new(3, :), dm_new(4, :), s, c, 'filled');
colormap(cmap)
caxis([vmin,vmax])
title('single sub embed, 234coord')
%% generate training embedding for comparison (optional)
% [dm, ~, embed, lambda1, lambda2, sigma1, sigma2] = calc2sDM(train_data, n_dim, configAffParams1, configAffParams2, configDiffParams1, configDiffParams2);
% 
% % plot
% s = 20;
% c = true_label_all(test_idx);
% vmin = -0.5;
% vmax = 17.5;
% 
% figure;scatter3(dm(1, :), dm(2, :), dm(3, :), s, c, 'filled');
% colormap(cmap)
% caxis([vmin,vmax])
% 
% figure;scatter3(dm(2, :), dm(3, :), dm(4, :), s, c, 'filled');
% colormap(cmap)
% caxis([vmin,vmax])