load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/data.mat')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/true_label/true_label_all_with_rest.mat')
load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/cmap18.mat')

% choose less subjects for quicker computation
data = data(:, :, 1:30);

% task_endtime = [772,1086,1554,2084,2594,3020]
% task_length = [772,314,468,530,510,426]

test_range = 1:386;
train_range = [387:3020];

test_data = data(:, test_range, :);
num_t_test = size(test_data, 2);

train_data = data(:, train_range, :);
num_s = size(train_data, 3);
num_t_train = size(train_data, 2);
num_r = size(train_data, 1);
num_t_all = num_t_test+num_t_train;

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
[dm, ~, embed, lambda1, lambda2, sigma1, sigma2] = calc2sDM(train_data, n_d, configAffParams1, configAffParams2, configDiffParams1, configDiffParams2);

s = 20*ones(num_t_train, 1);
c = true_label_all(train_range);
vmin = -0.5;
vmax = 17.5;

figure;subplot(2,2,1);scatter3(dm(1, :), dm(2, :), dm(3, :), s, c, 'filled');
colormap(cmap)
caxis([vmin,vmax])

subplot(2,2,2);scatter3(dm(1, :), dm(2, :), dm(4, :), s, c, 'filled');
colormap(cmap)
caxis([vmin,vmax])

%% generate extension embedding
sub = 1;

configAffParams3 = configAffParams1;
configAffParams3.kNN = 500;
configAffParams3.sig = sigma1(sub);

data_ind = zeros(num_r, num_t_all);
data_ind(:, train_range) = train_data(:, :, sub);
data_ind(:, test_range) = test_data(:, :, sub);

[K, ~] = calcAffinityMat(data_ind, configAffParams3);
K = K(test_range, train_range);
K_new = K./sum(K, 2);
psi1 = K_new*dm'./lambda2';

dm_all = zeros(n_d, num_t_all);

dm_all(:, train_range) = dm;
dm_all(:, test_range) = psi1';

% plot
s = 20*ones(num_t_all, 1);
c = true_label_all(1:num_t_all);
vmin = -0.5;
vmax = 17.5;

c(train_range) = 17;

subplot(2,2,3);scatter3(dm_all(1, :), dm_all(2, :), dm_all(3, :), s, c, 'filled');
colormap(cmap)
caxis([vmin,vmax])

subplot(2,2,4);scatter3(dm_all(1, :), dm_all(2, :), dm_all(4, :), s, c, 'filled');
colormap(cmap)
caxis([vmin,vmax])



