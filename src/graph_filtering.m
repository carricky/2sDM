%% load data
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/data.mat')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/true_label/true_label_all_with_rest.mat')
load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/cmap18.mat')

% choose less subjects for quicker computation
train_data = data(:, :, 1:10);

% leftout WM task
train_idx = 775:3020;
test_idx = 1:774;

n_tp_test = numel(test_idx);
test_data = data(:, test_idx, :);

train_data = train_data(:, train_idx, :);
n_sub = size(train_data, 3); % num of subs
n_tp_train = size(train_data, 2); % num of time points
n_rg = size(train_data, 1); % num of regions

n_tp_total = n_tp_train + n_tp_test;
%% config the parameters
k = 50;
configAffParams1.dist_type = 'euclidean';
configAffParams1.kNN = k;
configAffParams1.self_tune = 0;

configDiffParams1.t = 1;
configDiffParams1.normalization='lb';

configAffParams2 = configAffParams1;
configDiffParams2 = configDiffParams1;

n_dim = 7;

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
subplot(2,1,1);
scatter3(filter_set(:, 1, 1), filter_set(:, 2, 1), filter_set(:, 3, 1), s, 'filled');
subplot(2,1,2);
imagesc(filter_set(:, :, 1));


% simulate time points for target subject
tar_sub = 1;
tar_data = test_data(:, :, tar_sub);
tar_data = zscore(tar_data, 0, 2); % zscore by region
embed_cat = zeros(n_tp_total, n_dim, n_sub);
for i_sub = 1 : n_sub
    fprintf('computing embedding for %dth sub\n', i_sub)
    % train embedding
    K_temp = calcAffinityMat(train_data(:, :, i_sub), configAffParams1);
    dm_temp = calcDiffusionMap(K_temp, configDiffParams1);
    dm_temp = dm_temp(1:n_dim, :)';
    embed_cat(train_idx, :, i_sub) = dm_temp;
    % test embedding
    embed_cat(test_idx, :, i_sub) = tar_data' * filter_set(:, :, i_sub);
end
embed_cat = reshape(embed_cat, n_tp_total, n_dim*n_sub);
embed_cat = embed_cat';

% [K_temp, ~, ~] = calcAffinityMat(embed_cat, configAffParams2);
% [dm_new, ~, ~, ~, ~, ~] = calcDiffusionMap(K_temp, configDiffParams2);
% 
% % plot
% s = 20;
% c = true_label_all([train_idx, test_idx]);
% vmin = -0.5;
% vmax = 17.5;
% 
% figure;scatter3(dm_new(1, :), dm_new(2, :), dm_new(3, :), s, c, 'filled');
% colormap(cmap)
% caxis([vmin,vmax])
% 
% figure;scatter3(dm_new(2, :), dm_new(3, :), dm_new(4, :), s, c, 'filled');
% colormap(cmap)
% caxis([vmin,vmax])

% embed using simulated group data, only testing points
[K_temp, ~, ~] = calcAffinityMat(embed_cat(:, test_idx), configAffParams2);
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
[K_temp2, ~, ~] = calcAffinityMat(tar_data(:, test_idx), configAffParams2);
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