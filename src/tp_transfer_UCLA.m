% load network definition
load('/Users/siyuangao/Working_Space/fmri/data/parcellation_and_network/map268.mat')
map268(map268==6) = 5;
map268(map268==7) = 5;
network_name = {'MF', 'FP', 'DMN', 'Motor', 'Visual', 'Subcortical', 'Cerebellum'};

% load data
addpath('/Users/siyuangao/Working_Space/fmri/data/UCLA')
[ucla_task, ~] = data_label_generation(1);
% missing_nodes = [249, 239, 243, 129, 266, 109, 115, 118, 250];
% ucla_task(missing_nodes, :, :) = [];

% load labels, subject type and specific colormap
load('/Users/siyuangao/Working_Space/fmri/data/UCLA/all_labels.mat')
load('/Users/siyuangao/Working_Space/fmri/data/UCLA/idx199.mat')
load('/Users/siyuangao/Working_Space/fmri/data/UCLA/cmap12.mat')

% parse the size
n_rg = size(ucla_task, 1);

% set the subject and time
sub_idx_train = idx_control(1:30);
sub_idx_test = idx_control(31:40);
% sub_idx_test = idx_schz(31:40);
n_sub_train = numel(sub_idx_train);
n_sub_test = numel(sub_idx_train);
tp_idx_train = 1 : 1009;
n_tp_train = size(ucla_task, 2);

% final config
rng(665)
vmin = -0.5;
vmax = 11.5;

%% config the parameters
k = 150;
configAffParams1.dist_type = 'euclidean';
configAffParams1.kNN = k;
configAffParams1.self_tune = 0;
configDiffParams1.t = 1;
configDiffParams1.normalization='lb';
configAffParams2 = configAffParams1;
configDiffParams2 = configDiffParams1;
n_dim = 7;

%% 2sDM for training subjects
[dm_train, ~, embed, lambda1, lambda2, sigma1, sigma2] = calc2sDM(ucla_task(:, :, ...
    sub_idx_train), n_dim, configAffParams1, configAffParams2, ...
    configDiffParams1, configDiffParams2);

figure;
scatter3(dm_train(1, :), dm_train(2, :), dm_train(3, :), 40, all_labels, 'filled');
colormap(cmap12)
caxis([vmin, vmax])
title('training embedding')
%% 2sDM for training subjects
for ref_sub = sub_idx_test(1:5)
    disp(ref_sub)
    % rest
    %     tar_data = all_signal(:, :, ref_sub);
    %     tar_data = zscore(tar_data, 0, 1); % zscore by time dimension
    %     n_tp_test = size(tar_data, 2);
    
    % task
    tar_data = ucla_task(:, :, ref_sub);
    n_tp_test = size(tar_data, 2);
    n_tp_total = n_tp_train+n_tp_test;
    train_label = all_labels(tp_idx_train);
    %     test_label = 11*ones(n_tp_test,1);
    test_label = train_label;
    tp_idx_train = 1 : n_tp_train;
    tp_idx_test = n_tp_train+1:n_tp_total;
    
    n_dim_f = 268;
    %% simulate time points for target subject
    % tar_data = zscore(tar_data, 0, 2); % zscore by region
    
    x_tilda_all = zeros(n_tp_total, n_dim_f, n_sub_train);
    for i_sub = sub_idx_train
        %         fprintf('filter data for %dth sub\n', i_sub)
        % filter data
        x = ucla_task(:, :, i_sub);
        %         x_tilda_all(tp_idx_train, :, i_sub) = x' * V_set(:, :, i_sub);
        %         x_tilda_all(tp_idx_test, :, i_sub) = tar_data' * V_set(:, :, i_sub);
        x_tilda_all(tp_idx_train, :, i_sub) = x';
        x_tilda_all(tp_idx_test, :, i_sub) = tar_data';
    end
    x_tilda_all = reshape(x_tilda_all, n_tp_total, n_dim_f*n_sub_train);
    x_tilda_all = x_tilda_all';
    
    % Laplacian pyramid extension
    configAffParams3 = configAffParams1;
    configAffParams3.kNN = 50;
    configAffParams3.self_tune = 0;
    configAffParams3.n_sub = n_sub_train;
    dm_all = zeros(n_tp_total, n_dim);
    % compute dXX only once to speed up
    [dm_all(tp_idx_test, 1), l(1), dXX, dXY, indsXX, indsXY] = ...
        calcLaplcianPyramidExtension(configAffParams3, x_tilda_all(:, ...
        tp_idx_train), (dm_train(1,:)./lambda2(1))', x_tilda_all(:, tp_idx_test));
    for i = 2 : n_dim
        [dm_all(tp_idx_test, i), l(i)] = calcLaplcianPyramidExtension(...
            configAffParams3, x_tilda_all(:, tp_idx_train), ...
            (dm_train(i,:)./lambda2(i))', x_tilda_all(:, tp_idx_test), dXX, dXY, indsXX, indsXY);
    end
    dm_all = dm_all';
    dm_all(:, tp_idx_train) = dm_train;
    dm_all(:, tp_idx_test) = dm_all(:, tp_idx_test) .* lambda2; % don't forget about the lambda
    
    % plot
%     s = zeros(n_tp_total, 1);
%     c = zeros(n_tp_total, 1);
%     s(tp_idx_train) = 10;
%     s(tp_idx_test) = 30;
%     c(tp_idx_train) = train_label;
%     c(tp_idx_test) = test_label;
%     figure;
%     subplot(2,2,1);
%     scatter3(dm_all(1, :), dm_all(2, :), dm_all(3, :), s, c, 'filled');
%     colormap(cmap12)
%     caxis([vmin,vmax])
%     title('concatenated embed, testing enlarged, 123coord')
%     subplot(2,2,2);
%     scatter3(dm_all(1, :), dm_all(3, :), dm_all(4, :), s, c, 'filled');
%     colormap(cmap12)
%     caxis([vmin,vmax])
%     title('concatenatedembed, testing enlarged, 124coord')
%     s(tp_idx_train) = 30;
%     s(tp_idx_test) = 10;
%     subplot(2,2,3);
%     scatter3(dm_all(1, :), dm_all(2, :), dm_all(3, :), s, c, 'filled');
%     colormap(cmap12)
%     caxis([vmin,vmax])
%     title('concatenatedembed, training enlarged, 123coord')
%     subplot(2,2,4);
%     scatter3(dm_all(1, :), dm_all(3, :), dm_all(4, :), s, c, 'filled');
%     colormap(cmap12)
%     caxis([vmin,vmax])
%     title('concatenatedembed, training enlarged, 124coord')
    
    figure;
    %     scatter3(dm_all(1, tp_idx_test), dm_all(2, tp_idx_test), dm_all(3, tp_idx_test), 40, c(tp_idx_test), 'filled');
    scatter(dm_all(1, tp_idx_test), dm_all(2, tp_idx_test), 40, c(tp_idx_test), 'filled');
    colormap(cmap12)
    caxis([vmin,vmax])
    xlim([-0.05, 0.05])
    ylim([-0.04, 0.04])
    title(sprintf('sub%d testing embedding', ref_sub))
    % figure 2
    % figure;
    % c_train = c(tp_idx_train);
    % c_train(c_train<2 | c_train>3) = 17;
    % subplot(2,1,1);
    % scatter3(dm_all(1, tp_idx_train), dm_all(2, tp_idx_train), dm_all(4, tp_idx_train), 30, c_train, 'filled');
    % colormap(cmap12)
    % caxis([vmin,vmax])
    % title('training embedding, 123coord')
    %
    % subplot(2,1,2);
    % c_test = c(tp_idx_test);
    % c_test(c_test<2 | c_test>3) = 17;
    % scatter3(dm_all(1, tp_idx_test), dm_all(2, tp_idx_test), dm_all(4, tp_idx_test), 30, c_test, 'filled');
    % colormap(cmap12)
    % caxis([vmin,vmax])
    % title('test embedding, 123coord')
    
    % plot trajectory
    % figure;
    % for i = 2:3
    %     temp_sum = 0;
    %     count2 = 0;
    %     for j = 1 : length(task_start{i-1})
    %         s = task_start{i-1}(j);
    %         if find(tp_idx_test==s)
    %             e = task_end{i-1}(j);
    %             if size(temp_sum, 2) ~= 1 && size(temp_sum, 2) > e-s+1
    %                 temp_sum = temp_sum(:, 1:e-s+1);
    %             end
    %             if size(temp_sum, 2) ~= 1 && size(temp_sum, 2) < e-s+1
    %                 e = e - (e-s+1-size(temp_sum,2));
    %             end
    %             temp_sum = temp_sum + dm_all(:, s:e);
    %             count2 = count2+1;
    %         end
    %     end
    %     temp_sum = temp_sum / count2;
    %
    %     plot3(temp_sum(1, :), temp_sum(2, :), temp_sum(3, :), 'linewidth', 5, 'color', cmap(i+1,:));
    %     hold on
    % end
    
    % dynamic trajectory
    %     s(tp_idx_train) = 10;
    %     s(tp_idx_test) = 30;
    %     c(tp_idx_train) = train_label;
    %     c(tp_idx_test) = test_label;
    %     figure;
    %     % scatter3(dm_all(1, :), dm_all(2, :), dm_all(3, :), s, c, 'filled');
    %     scatter3(dm_all(1, tp_idx_train), dm_all(2, tp_idx_train), dm_all(3, tp_idx_train), s(tp_idx_train), IDX, 'filled');
    %     colormap(cmap12)
    %     caxis([vmin,vmax])
    %     hold on
    %     for i =  1 : n_tp_test
    %         %     plot3(dm_all(1, tp_idx_test(1:end)), dm_all(2, tp_idx_test(1:end)), dm_all(3, tp_idx_test(1:end)), 'linewidth', 5, 'color', cmap(18,:));
    %         plot3(dm_all(1, tp_idx_test(1:i)), dm_all(2, tp_idx_test(1:i)), dm_all(3, tp_idx_test(1:i)), 'linewidth', 5, 'color', cmap(18,:));
    %         pause(0.1)
    %     end
    %     hold off
    
    % figure;
    % plot3(temp_sum(1, :), temp_sum(2, :), temp_sum(3, :), 'linewidth', 5, 'color', cmap(i+1,:));
    
end
