load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/data.mat')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/true_label/true_label_all_withrest_coarse.mat')
load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/cmap10.mat')
load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/cmap4_2.mat')
load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/output/integration/hcp_sub30.mat')

% choose less subjects for quicker computation
data = data(:, 1:3020, 1:30);

n_sub = size(data, 3);
n_tp = size(data, 2);
n_region = size(data, 1);

label = true_label_all(1:3020);

%% config the parameters
n_dim = 7;

%% 2sPCA
embed = zeros(n_tp, n_sub*n_dim);
for i_sub = 1 : n_sub
    disp(i_sub/n_sub)
    data_ind = data(:, :, i_sub);
    [~, score] = pca(data_ind');
    embed(:, (i_sub-1)*n_dim+1:(i_sub)*n_dim) = score(:, 1:n_dim);
end

% second round of PCA computation
[~, score] = pca(embed);

%% plot
c = true_label_all(1:n_tp);
vmin = -0.5;
vmax = 9.5;
figure;scatter3(score(:, 1), score(:, 2), score(:, 3), 20, c, 'filled');
colormap(cmap10)
caxis([vmin,vmax])

%% kmeans clustering
kmeans_idx = kmeans(score(:, 1:3), 4, 'Replicates', 100);

figure;
scatter3(score(:, 1), score(:, 2), score(:, 3), 40, kmeans_idx, 'filled');
colormap(cmap4)
caxis([0.5, 5.5])

% pc of each cluster
pc_global_temp = pc_global(pc_global~=0);
kmeans_idx_temp = kmeans_idx(pc_global~=0);

kmeans_idx_temp(kmeans_idx_temp==1)=6;
kmeans_idx_temp(kmeans_idx_temp==2)=8;
kmeans_idx_temp(kmeans_idx_temp==3)=5;
kmeans_idx_temp(kmeans_idx_temp==4)=7;
kmeans_idx_temp = kmeans_idx_temp-4;

% box plot
figure;
boxplot(pc_global_temp, kmeans_idx_temp, 'Colors', cmap4(1:4, :))

figure;
boxplot(pc_global_temp, kmeans_idx, 'Colors', cmap4(1:4, :))
% boxplot({pc_global_temp(kmeans_idx_temp==1)';...
%     pc_global_temp(kmeans_idx_temp==2)'; pc_global_temp(kmeans_idx_temp==3)'; ...
%     pc_global_temp(kmeans_idx_temp==4)'})


c = true_label_all(1:3020);

figure;
scatter3(score(pc_global~=0, 1), score(pc_global~=0, 2), score(pc_global~=0, 3), 40, c(pc_global~=0), 'filled');
colormap(cmap10)
caxis([-0.5, 9.5])

figure;
scatter(-score(pc_global~=0, 1), -score(pc_global~=0, 2), 20, pc_global(pc_global~=0), 'filled');


%% correlation with pc_global
corr(pc_global(pc_global~=0)', score(pc_global~=0, 1:3))