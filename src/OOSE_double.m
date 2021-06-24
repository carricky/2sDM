addpath('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/')

%% load in the data
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/data.mat')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/true_label/true_label_all_withrest_coarse.mat')
load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/cmap10.mat')

% choose less subjects for quicker computation
data = data(:, :, 1:30);

%% set the range of task to interpolate
% task = 1; % change this
% 
% task_endtime = [0, 4772,1086,1554,2084,2594,3020];
% task_length = [772,314,468,530,510,426];
% test_range = task_endtime(task)+1:task_endtime(task)+task_length(task)/2;
% train_range = [1:task_endtime(task),...
%     (task_endtime(task)+task_length(task)/2+1):3020];

train_range = 1:3020;

%% set the data
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
[dm, ~, embed, lambda1, lambda2, sigma1, sigma2] = calc2sDM(train_data, n_d,...
    configAffParams1, configAffParams2, configDiffParams1, configDiffParams2);

s = 20*ones(num_t_train, 1);
c = true_label_all(train_range);
vmin = -0.5;
vmax = 9.5;

figure;subplot(2,2,1);scatter3(dm(1, :), -dm(2, :), dm(3, :), s, c, 'filled');
colormap(cmap10)
caxis([vmin,vmax])

subplot(2,2,2);scatter3(dm(1, :), dm(2, :), dm(4, :), s, c, 'filled');
colormap(cmap10)
caxis([vmin,vmax])

%% generate extension embedding
psi1 = zeros(num_t_test, n_d*num_s);
configAffParams3 = configAffParams1;
configAffParams3.kNN = 500;
for i = 1 : num_s
    disp(i)
    %     [K] = calcAffinityMat2(wm_data(:,:,i), data(:,:,i), k, sigma1(i));
    data_ind = zeros(num_r, num_t_all);
    data_ind(:, train_range) = train_data(:, :, i);
    data_ind(:, test_range) = test_data(:, :, i);
    configAffParams3.sig = sigma1(i);
    K = calcAffinityMat(data_ind, configAffParams3);
    K = K(test_range, train_range);
    K = K./sum(K, 2);
    psi1(:, (i-1)*n_d+1:i*n_d) = K*embed(:, (i-1)*n_d+1:i*n_d)./lambda1(:, i)';
end

% second round embedding

embed_all = zeros(n_d*num_s, num_t_all);
embed_all(:, train_range) = embed';
embed_all(:, test_range) = psi1';
configAffParams3.sig = sigma2;
[K, ~] = calcAffinityMat(embed_all, configAffParams3);
K = K(test_range, train_range);
K = K./sum(K, 2);
psi2 = K*dm'./lambda2';

dm_all = zeros(n_d, num_t_all);
dm_all(:, train_range) = dm;
dm_all(:, test_range) = psi2';

% plot
s = 20*ones(num_t_all, 1);
c = true_label_all(1:num_t_all);
vmin = -0.5;
vmax = 9.5;

c(train_range) = 9;

subplot(2,2,3);scatter3(dm_all(1, :), dm_all(2, :), dm_all(3, :), s, c, 'filled');
colormap(cmap10)
caxis([vmin,vmax])

subplot(2,2,4);scatter3(dm_all(1, :), dm_all(2, :), dm_all(4, :), s, c, 'filled');
colormap(cmap10)
caxis([vmin,vmax])

% save(['dm_all_leftout', num2str(task), '.mat'], 'dm_all', 'train_range', 'test_range')



%% plot results from all the tasks
figure;
num_t_all = 3020;

vmin = -0.5;
vmax = 17.5;
for task = 1 : 6
    s = 20*ones(num_t_all, 1);
    c = true_label_all(1:num_t_all);
    load(['dm_all_leftout',num2str(task),'.mat'])
%     load(['dm_all_bothsession_leftout',num2str(task),'.mat'])
    subplot(6,2,2*(task)-1); scatter3(dm_all(1, train_range), dm_all(2, train_range), dm_all(4, train_range), s(train_range), c(train_range), 'filled');
    c(train_range) = 17;
    subplot(6,2,2*(task)); scatter3(dm_all(1, :), dm_all(2, :), dm_all(4, :), s, c, 'filled');
    colormap(cmap18)
    caxis([vmin,vmax])
end

figure;

for task = 1:6
    s = 50*ones(num_t_all, 1);
    c = true_label_all(1:num_t_all);
    load(['dm_all_leftout',num2str(task),'.mat'])
    c(:) = 17;
    compare_range = (test_range(end)+1):(test_range(end)+numel(test_range));
    c(compare_range) = true_label_all(compare_range);
    %     subplot(6,2,2*task-1); scatter3(dm_all(1, :), dm_all(2, :), dm_all(4, :), s, c, 'filled');
    figure;subplot(1,2,1);scatter3(dm_all(1, :), dm_all(2, :), dm_all(4, :), s, c, 'filled');
    colormap(cmap)
    caxis([vmin,vmax])
    title('grounth truth')
    view([-38, 6])
    
    c = true_label_all(1:num_t_all);
    c(train_range) = 17;
%     subplot(6,2,2*task); scatter3(dm_all(1, :), dm_all(2, :), dm_all(4, :), s, c, 'filled');
    subplot(1,2,2);scatter3(dm_all(1, :), dm_all(2, :), dm_all(4, :), s, c, 'filled');
    colormap(cmap)
    caxis([vmin,vmax])
    title('OOSE')
    view([-38, 6])
end


% compare bothsession leftout with ground truth
for task = 1:6
    s = 50*ones(num_t_all, 1);
    c = true_label_all(1:num_t_all);
    load(['dm_all_bothsession_leftout',num2str(task),'.mat'])
    c(:) = 17;
    c(test_range) = true_label_all(test_range);
    figure;subplot(1,2,1);scatter3(dm(1, :), dm(2, :), dm(4, :), s, c, 'filled');
    colormap(cmap)
    caxis([vmin,vmax])
    
    title('grounth truth')
    view([-38, 6])

%     subplot(6,2,2*task); scatter3(dm_all(1, :), dm_all(2, :), dm_all(4, :), s, c, 'filled');
    subplot(1,2,2);scatter3(dm_all(1, :), dm_all(2, :), dm_all(4, :), s, c, 'filled');
    colormap(cmap)
    caxis([vmin,vmax])
    title('OOSE')
    view([-38, 6])
end


for task = 1:6
    % calculate correlation with ground truth
    load(['dm_all_leftout',num2str(task),'.mat'])
%     load(['dm_all_bothsession_leftout',num2str(task),'.mat'])
    
    temp = abs(diag(corr(dm(1:4, test_range)', dm_all(1:4, test_range)')));
    disp(mean(temp([1,2,4])))
%     disp(temp)
    
%     compare_range = (test_range(end)+1):(test_range(end)+numel(test_range));
%     temp = abs(diag(corr(dm_all(1:4, compare_range)', dm_all(1:4, test_range)')));
%     disp(mean(temp([1,2,4])))
%     disp(temp)
end