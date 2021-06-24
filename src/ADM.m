%% load data
% load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/data.mat')
% load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/true_label/true_label_all_with_rest.mat')
% load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/cmap18.mat')

% leftout WM task
time_idx = 1:774;

%% config the parameters
k = 400;
configAffParams1.dist_type = 'euclidean';
configAffParams1.kNN = k;
configAffParams1.self_tune = 0;

configDiffParams1.t = 1;
configDiffParams1.normalization='lb';

n_dim = 7;

%% ADM
sub1 = 1;
sub2 = 2;

data_temp = data(:, time_idx, sub1);
K_1 = calcAffinityMat(data_temp, configAffParams1);
data_temp = data(:, time_idx, sub2);
K_2 = calcAffinityMat(data_temp, configAffParams1);
K_adm = K_1*K_2' + K_2*K_1';
dm_1 = calcDiffusionMap(K_1, configDiffParams1);
dm_1 = dm_1(1:n_dim, :);

dm_2 = calcDiffusionMap(K_2, configDiffParams1);
dm_2 = dm_2(1:n_dim, :);

dm_adm = calcDiffusionMap(K_adm, configDiffParams1);
dm_adm = dm_adm(1:n_dim, :);



% plot
s = 20;
c = true_label_all(time_idx);
vmin = -0.5;
vmax = 17.5;
figure;
subplot(2,2,1);
scatter3(dm_1(1, :), dm_1(2, :), dm_1(3, :), s, c, 'filled');
colormap(cmap)
caxis([vmin,vmax])

subplot(2,2,2);
scatter3(dm_2(1, :), dm_2(2, :), dm_2(3, :), s, c, 'filled');
colormap(cmap)
caxis([vmin,vmax])

subplot(2,2,3);
scatter3(dm_adm(1, :), dm_adm(2, :), dm_adm(3, :), s, c, 'filled');
colormap(cmap)
caxis([vmin,vmax])

subplot(2,2,4);
scatter3(dm_adm(2, :), dm_adm(3, :), dm_adm(4, :), s, c, 'filled');
colormap(cmap)
caxis([vmin,vmax])

