load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/data.mat')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/true_label/true_label_all_withrest_coarse.mat')
load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/cmap10.mat')
load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/cmap4_2.mat')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/WM_accuracy_RT390.mat')

% choose less subjects for quicker computation
data = data(:, 1:3020, 1:390);

n_sub = size(data, 3);
n_tp_all = size(data, 2);
n_region = size(data, 1);

%% config the parameters
k = 500;
configAffParams1.dist_type = 'euclidean';
configAffParams1.kNN = k;
configAffParams1.self_tune = 0;

configDiffParams1.t = 1;
configDiffParams1.normalization='lb';

configAffParams2 = configAffParams1;
configDiffParams2 = configDiffParams1;
configDiffParams2.maxInd = 12;

%% generate embedding with different number of dimensions
n_dim_list = [3, 5, 7, 9, 11];
dm_all = cell(numel(n_dim_list), 1);
for i_dim = 1 : numel(n_dim_list)
    dm_all{i_dim} = calc2sDM(data, n_dim_list(i_dim), configAffParams1,...
        configAffParams2, configDiffParams1, configDiffParams2);
end

%% calculate similarities between different embeddings
n_dim = 7;
cord_similiarity = zeros(n_dim, numel(n_dim_list));
for i = 1 : 5
    cord_similiarity(:, i) = max(abs(corr(dm_all{3}(1:n_dim, :)', dm_all{i}(1:n_dim, :)')), [], 2);
end
    

figure;
plot(cord_similiarity(1:n_dim, :), '-*')
ylim([0, 1])
legend({'d1=3', 'd1=5', 'd1=7', 'd1=9', 'd1=11'})
xlabel('coordinate')
ylabel('correlation')

%% plot certain embedding
i_d1 = 1;
figure;
c = true_label_all(1:3020);
scatter3(dm_all{i_d1}(1, :), dm_whole(2, :), dm_whole(3, :), 20, c, 'filled');
colormap(cmap10)
caxis([-0.5,9.5])