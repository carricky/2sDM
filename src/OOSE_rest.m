load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/data.mat')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/true_label/true_label_all_with_rest.mat')
load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/cmap18.mat')

% choose less subjects for quicker computation
data = data(:, :, :);


end_time = 5420;
% leftout Rest task
rest_data = data(:, 3021:end_time, :);
num_t_rest = size(rest_data, 2);

data = data(:, 1:3020, :);
num_s = size(data, 3);
num_t = size(data, 2);
num_r = size(data, 1);

%% config the parameters
sub = 1; % sub for reference
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
[dm, ~, embed, lambda1, lambda2, sigma1, sigma2] = calc2sDM(data, n_d, configAffParams1, configAffParams2, configDiffParams1, configDiffParams2);

s = 20*ones(num_t, 1);
c = true_label_all(1:3020);
vmin = -0.5;
vmax = 17.5;

figure;subplot(2,2,1);scatter3(dm(1, :), dm(2, :), dm(3, :), s, c, 'filled');
colormap(cmap)
caxis([vmin,vmax])

subplot(2,2,2);scatter3(dm(1, :), dm(2, :), dm(4, :), s, c, 'filled');
colormap(cmap)
caxis([vmin,vmax])

%% generate extension embedding

configAffParams3 = configAffParams1;
configAffParams3.kNN = 700;
configAffParams3.sig = sigma1(sub);

data_ind = [data(:, :, sub), rest_data(:, :, sub)];
[K, ~] = calcAffinityMat(data_ind, configAffParams3);
K = K(3021:end_time, 1:3020);
K_new = K./sum(K, 2);
psi1 = K_new*dm'./lambda2';

dm_all = [dm, psi1'];

% plot
s = 20*ones(end_time, 1);
c = true_label_all(1:end_time);
vmin = -0.5;
vmax = 17.5;

% c(1:3020) = 17;

subplot(2,2,3);scatter3(dm_all(1, :), dm_all(2, :), dm_all(3, :), s, c, 'filled');
colormap(cmap)
caxis([vmin,vmax])

subplot(2,2,4);scatter3(dm_all(1, :), dm_all(2, :), dm_all(4, :), s, c, 'filled');
colormap(cmap)
caxis([vmin,vmax])



