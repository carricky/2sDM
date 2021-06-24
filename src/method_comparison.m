%% load the signals and true labels
% generate_data;
addpath('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils')

load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/data.mat')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/true_label/true_label_all_withrest_coarse.mat')
load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/cmap10.mat')
data = data(:, 1:3020, 1:390);

%% config the parameters
configAffParams1.dist_type = 'euclidean';
configAffParams1.kNN = 500;
configAffParams1.self_tune = 0;

configDiffParams1.t = 1;
configDiffParams1.normalization='lb';

n_dim = 7;

%% parse parameters
n_rg = size(data, 1);
n_tp = size(data, 2);
n_sub = size(data, 3);

%% one step embedding
[K, ~] = calcAffinityMat(reshape(permute(data, [1, 3, 2]), [], n_tp),...
    configAffParams1);
[dm, ~, ~, ~, ~, ~] = calcDiffusionMap(K, configDiffParams1);


% visualize the first three non-trivial dimensions
s = 20*ones(n_tp, 1);
c = true_label_all(1:n_tp);
vmin = -0.5;
vmax = 9.5;

figure;scatter3(dm(1, :), dm(2, :), dm(3, :), s, c, 'filled');
colormap(cmap10)
caxis([vmin,vmax])

figure;scatter3(dm(3, :), dm(4, :), dm(5, :), s, c, 'filled');
colormap(cmap10)
caxis([vmin,vmax])


%% mean embedding
mean_data = mean(data, 3);
[K, ~] = calcAffinityMat(mean_data, configAffParams1);
[dm, ~, ~, ~, ~, ~] = calcDiffusionMap(K, configDiffParams1);


% visualize the first three non-trivial dimensions
s = 20*ones(n_tp, 1);
c = true_label_all(1:n_tp);
vmin = -0.5;
vmax = 9.5;

figure;scatter3(dm(1, :), dm(2, :), dm(3, :), s, c, 'filled');
colormap(cmap10)
caxis([vmin,vmax])

figure;scatter3(dm(3, :), dm(4, :), dm(5, :), s, c, 'filled');
colormap(cmap10)
caxis([vmin,vmax])

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

c = true_label_all(1:n_tp);
vmin = -0.5;
vmax = 9.5;

figure;scatter3(score(:, 1), score(:, 2), score(:, 3), 20, c, 'filled');
colormap(cmap10)
caxis([vmin,vmax])

%% trajectory
score = score(:, 1:7)';
figure;
task_length = [405, 176, 253, 284, 274, 232];
task_name = {'WM', 'Emotion', 'Gambling', 'Motor', 'Social', 'Relational'};
task_length = task_length-19;
task_length = task_length*2;
begin = 1;
colors=['b','y','g','m','r','c'];

% each task separately
for i = 1 : 6
    subplot(3,3,i);
    %     for j = begin:task_length(i)+begin-1
%     plot3(dm(1, begin:task_length(i)+begin-1), dm(2, begin:task_length(i)+begin-1), dm(4, begin:task_length(i)+begin-1));
    plot3(score(1, begin:task_length(i)+begin-1), score(2, begin:task_length(i)+begin-1), score(3, begin:task_length(i)+begin-1));
    %     hold on
    title(task_name{i})
    begin = task_length(i)+begin;
end

%% average within same task block
true_label_all = true_label_all(1:3020);

% get task start and end time stamp
task_start = {};
task_end = {};
for i = 2 : 8
    temp = find(true_label_all == i);
    task_start_temp = temp(1);
    task_end_temp = [];
    for j = 2 : length(temp)
        if temp(j) ~= temp(j-1)+1
            task_start_temp = [task_start_temp, temp(j)];
            task_end_temp = [task_end_temp, temp(j-1)];
        end
    end
    task_end_temp = [task_end_temp, temp(end)];
    task_start{i-1} = task_start_temp;
    task_end{i-1} = task_end_temp;
end

dim = [1,2,3];

% get the average trajectory
figure;
hold on
for i = 2 : 8
% for i = [2,3]
    temp_sum = 0;
    count2 = 0;
    for j = 1 : length(task_start{i-1})
        s = task_start{i-1}(j);
        e = task_end{i-1}(j);
        if size(temp_sum, 2) ~= 1 && size(temp_sum, 2) > e-s+1
            temp_sum = temp_sum(:, 1:e-s+1);
        end
        if size(temp_sum, 2) ~= 1 && size(temp_sum, 2) < e-s+1
            e = e - (e-s+1-size(temp_sum,2));
        end
        temp_sum = temp_sum + score(:, s:e);
        count2 = count2+1;
    end
    temp_sum = temp_sum / count2;
    
    %     plot3(temp_sum(dim(1), :), -temp_sum(dim(2), :), temp_sum(dim(3), :), '-o', 'MarkerFaceColor',cmap10(i+1,:), 'color', cmap10(i+1,:), 'linewidth', 2, 'MarkerSize',8);
    plot3(temp_sum(dim(1), :), temp_sum(dim(2), :), temp_sum(dim(3), :),...
        'color', cmap10(i+1,:), 'linewidth', 8);
end

vmin = -0.5;
vmax = 9.5;
f = scatter3(score(dim(1), :), score(dim(2), :), score(dim(3), :), 15, true_label_all, 'filled');
f.MarkerFaceAlpha = 0.5;
colormap(cmap10)
caxis([vmin,vmax])
grid on
hold off
