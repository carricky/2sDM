load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/data.mat')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/true_label/true_label_all_withrest_coarse.mat')
load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/cmap10.mat')
load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/cmap4_2.mat')

% choose less subjects for quicker computation
data = data(:, 1:3020, 1:390);

n_sub = size(data, 3);
n_tp = size(data, 2);
n_region = size(data, 1);

label = true_label_all(1:3020);

%% config the parameters
n_dim = 3;

%% generate training embedding
mds_coord = calc2sMDS(data, n_dim);

%% plot
figure;
scatter3(mds_coord(:, 1), mds_coord(:, 2), mds_coord(:, 3), 40, label, 'filled');
colormap(cmap10)
caxis([-0.5,9.5])

%% kmeans clustering
kmeans_idx = kmeans(mds_coord, 4, 'Replicates', 100);

figure;
scatter3(mds_coord(:, 1), mds_coord(:, 2), mds_coord(:, 3), 40, kmeans_idx, 'filled');
colormap(cmap4)
caxis([0.5, 5.5])