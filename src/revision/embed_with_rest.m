% load data
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/data.mat')
% load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/true_label/true_label_all_with_rest.mat')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/true_label/true_label_all_withrest_coarse.mat')
load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/cmap10.mat')

addpath('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/BrainSync/');
addpath('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils')

% choose less subjects for quicker computation
data = data(:, :, 1:390);
test_range = 3021:4220;
train_range = 1:3020;
num_t = max(test_range);
num_s = size(data, 3);
data = data(:, 1:num_t, :);

%% BrainSync data
ref_sub = 1;
for i = 1 : num_s
    disp(i)
    if i ~= ref_sub
        [Y2, R] = brainSync(data(:, test_range, ref_sub)', data(:, test_range, i)');
        data(:, test_range, i) = Y2';
    end
end

%% config the parameters
k = 1100;
configAffParams1.dist_type = 'euclidean';
configAffParams1.kNN = k;
configAffParams1.self_tune = 0;

configDiffParams1.t = 1;
configDiffParams1.normalization='lb';

configAffParams2 = configAffParams1;
configDiffParams2 = configDiffParams1;

n_d = 7;

%% generate training embedding
[dm, ~, embed, lambda1, lambda2, sigma1, sigma2] = calc2sDM(data, n_d,...
    configAffParams1, configAffParams2, configDiffParams1, configDiffParams2); 

%% plot
c = true_label_all(1:num_t);
vmin = -0.5;
vmax = 9.5;

figure;
scatter3(dm(1, train_range), dm(2, train_range), dm(3, train_range),...
    40, c(train_range), 'filled');
hold on;
scatter3(dm(1, test_range), dm(2, test_range), dm(3, test_range), 40,...
    c(test_range), 'filled', 'MarkerFaceAlpha',.8);
colormap(cmap10)
caxis([vmin,vmax])
hold off;

% figure;scatter3(dm(1, :), dm(2, :), dm(4, :), s, c, 'filled');
% colormap(cmap10)
% caxis([vmin,vmax])
