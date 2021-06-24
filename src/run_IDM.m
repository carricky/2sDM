%% config data
dataset = 'HCP';
if strcmp(dataset, 'HCP')
    % HCP
%     load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/data.mat')
%     load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/true_label/true_label_all_with_rest.mat')
    load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/cmap18.mat')
    sub = 1;
    n_iter = 1;
    n_dim = 7;
    %     time_idx = [1:1553, 3021:3620];
    %     time_idx = 1 : 5382;
    time_idx = 1 : 3020;
    k = 300;
    c = true_label_all(time_idx);
    s = 40;
    vmin = -0.5;
    vmax = 17.5;
    
elseif strcmp(dataset, 'UCLA')
    % UCLA
    load('/Users/siyuangao/Working_Space/fmri/data/UCLA/all_task.mat')
    load('/Users/siyuangao/Working_Space/fmri/data/UCLA/all_label.mat')
    sub = 2;
    n_iter = 5;
    n_dim = 20;
    k = 20;
    % time_idx = [1:1553, 3021:3620];
    time_idx = 1 : 262;
    
    s = 40;
    c = label(time_idx);
end


configAffParams.dist_type = 'euclidean';
configAffParams.kNN = k;
configAffParams.self_tune = 0;

configDiffParams.t = 1;
configDiffParams.normalization='lb';
configDiffParams.maxInd = n_dim+1;

[dm, K] = IDM(data(:, time_idx, sub), n_dim, n_iter, configAffParams, configDiffParams, 0);

for i_iter = 1 : n_iter
    figure(1);
    subplot(ceil(sqrt(n_iter)), ceil(sqrt(n_iter)), i_iter);
    scatter3(dm(:, 1, i_iter), dm(:, 2, i_iter), dm(:, 3, i_iter), s, c, 'filled');
    colormap(cmap)
    caxis([vmin, vmax])
    
    figure(2);
    subplot(ceil(sqrt(n_iter)), ceil(sqrt(n_iter)), i_iter);
    imagesc(K(:, :, i_iter))
    
    figure(3);
    subplot(ceil(sqrt(n_iter)), ceil(sqrt(n_iter)), i_iter);
    imagesc(dm(:, :, i_iter))
end
%% config task description
% num_task = 9;
% task_set = cell(num_task,1);
% task_set{1} = [1,2]+1;
% task_set{2} = [3,4]+1;
% task_set{3} = [5,6]+1;
% task_set{4} = [7,8,9,10,11]+1;
% task_set{5} = [12,13]+1;
% task_set{6} = [14,15]+1;
% task_set{7} = [0];
% task_set{8} = [16]+1;
% task_set{9} = [1];
% 
% task_name = cell(num_task,1);
% task_name{1} = "WM";
% task_name{2} = "Emotion";
% task_name{3} = "Gambling";
% task_name{4} = "Motor";
% task_name{5} = "Social";
% task_name{6} = "Relational";
% task_name{7} = "Fixation";
% task_name{8} = "Rest";
% task_name{9} = "Cue";
% 
%% some cool animation
% figure;scatter3(dm(1, :), dm(2, :), dm(3, :), s, c, 'filled');
% % figure;scatter3(dm(3, :), dm(4, :), dm(5, :), s, c, 'filled');
% colormap(cmap)
% pausetime = 4;
% caxis([vmin,vmax])
% while 1
%     for i = 1 : num_task
%         temp_map = 0.5*ones(size(cmap));
%         %         temp_map(1,:) = cmap(1,:);
%         %         temp_map(18,:) = [0.9,0.9,0.9];
%         if i==8
%             temp_map(task_set{i}+1,:) = [0.9,0.9,0.9];
%         else
%             temp_map(task_set{i}+1,:) = cmap(task_set{i}+1, :);
%         end
%         colormap(temp_map)
%         caxis([vmin,vmax])
%         title(task_name{i})
%         pause(pausetime)
%     end
% end