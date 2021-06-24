%% training phase
%% train data
% HCP
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/data.mat')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/true_label/true_label_all_with_rest.mat')
load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/cmap18.mat')

% choose less subjects for quicker computation

% leftout WM task
% train_idx = [1:386, 773:929, 1087:1320, 1555:1819, 2085:2339, 2595:2807];
% test_idx = [387:772, 930:1086, 1321:1554, 1820:2084, 2340:2594, 2808:3020];
train_idx = 1 : 3020;
train_data = data(:, train_idx, 1:30);
n_sub = size(train_data, 3); % num of subs
n_tp_train = size(train_data, 2); % num of time points
n_rg = size(train_data, 1); % num of regions

% ref_sub = 33;
% tar_data = data(:, test_idx, ref_sub);
% n_tp_test = size(tar_data, 2);

% final config
rng(665) % for reproducible K-means result
vmin = -0.5;
vmax = 17.5;

%% config the parameters
k = 500;
configAffParams1.dist_type = 'euclidean';
configAffParams1.kNN = k;
configAffParams1.self_tune = 10;
configDiffParams1.t = 1;
configDiffParams1.normalization='lb';
configAffParams2 = configAffParams1;
configDiffParams2 = configDiffParams1;
n_dim = 7;
n_dim_f = 10;
%% 2sDM for training data (can skip if run multiple times)
[dm_train, ~, embed, lambda1, lambda2, sigma1, sigma2] = calc2sDM(train_data, n_dim, configAffParams1, configAffParams2, configDiffParams1, configDiffParams2);
k = 4;
IDX = kmeans(dm_train(:, :)', k, 'Replicates', 100);

figure;
imagesc(dm_train);

figure;
scatter3(dm_train(1, :), dm_train(2, :), dm_train(3, :), 40, IDX, 'filled');
% hold on;
% sc = scatter3(dm_train(1, :), dm_train(2, :), dm_train(3, :), 20, true_label_all(train_idx), 'filled');
% sc.MarkerFaceAlpha = 0.5;
% colormap(cmap)
% caxis([vmin,vmax])
% hold off;

figure;
% subplot(2, 2, 1);
scatter3(dm_train(1, :), dm_train(2, :), dm_train(3, :), 20, IDX, 'filled');
subplot(2, 2, 2);
scatter3(dm_train(2, :), dm_train(3, :), dm_train(4, :), 20, IDX, 'filled');
subplot(2, 2, 3);
scatter3(dm_train(1, :), dm_train(3, :), dm_train(7, :), 20, IDX, 'filled');

figure;
% subplot(2, 2, 1);
scatter3(dm_train(1, :), dm_train(2, :), dm_train(3, :), 20, train_label, 'filled');
colormap(cmap)
caxis([vmin,vmax])
subplot(2, 2, 2);
scatter3(dm_train(2, :), dm_train(3, :), dm_train(4, :), 20, train_label, 'filled');
colormap(cmap)
caxis([vmin,vmax])
subplot(2, 2, 3);
scatter3(dm_train(1, :), dm_train(3, :), dm_train(7, :), 20, train_label, 'filled');
colormap(cmap)
caxis([vmin,vmax])

%% Learn GSP basis
% compute subject-wise filter
V_set = getFrequencyVector(train_data, n_dim_f);

% plot any filter/region embedding/gradient
figure;
load('/Users/siyuangao/Working_Space/fmri/data/map259.mat')
subplot(2,1,1);
scatter3(V_set(:, 1, 1), V_set(:, 2, 1), V_set(:, 3, 1), 20, map259, 'filled');
colormap(gca, cmap)
subplot(2,1,2);
imagesc(V_set(:, :, 1));
colormap(gca, parula)


%% testing phase
change_time = zeros(1, n_tp_test);
%% test data
% HCP WM
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/WM_accuracy390.mat')
[~, wm_idx] = sort(WM_accuracy);
% idx_subset = [wm_idx(1:5); wm_idx(end-4:end)]';
idx_subset = 20;
for ref_sub = idx_subset
    ref_sub
   
    % task data
    test_idx = 1 : 772;
    tar_data = data(:, test_idx, ref_sub);
% tar_data = data(:, 773:1086, ref_sub);
%     tar_data = data(:, 2595:3020, ref_sub);
    n_tp_test = size(tar_data, 2);
    n_tp_total = n_tp_train+n_tp_test;
    
    train_label = true_label_all(train_idx);
    %     test_label = 17*ones(n_tp_test,1);
    test_label = true_label_all(test_idx);
    train_idx = 1 : n_tp_train;
    test_idx = n_tp_train+1:n_tp_total;
    

    %% simulate time points for target subject
    % tar_data = zscore(tar_data, 0, 2); % zscore by region
    n_dim_f = n_rg;
    x_tilda_all = zeros(n_tp_total, n_dim_f, n_sub);
    for i_sub = 1 : n_sub
%         fprintf('filter data for %dth sub\n', i_sub)
        % filter data
        x = train_data(:, :, i_sub);
%         x_tilda_all(train_idx, :, i_sub) = x' * V_set(:, :, i_sub);
%         x_tilda_all(test_idx, :, i_sub) = tar_data' * V_set(:, :, i_sub);
        x_tilda_all(train_idx, :, i_sub) = x';
        x_tilda_all(test_idx, :, i_sub) = tar_data';
    end
    x_tilda_all = reshape(x_tilda_all, n_tp_total, n_dim_f*n_sub);
    x_tilda_all = x_tilda_all';
    %compare
    % x_tilda_all = zeros(n_tp_total, n_rg, n_sub);
    % for i_sub = 1 : n_sub
    %     fprintf('filter data for %dth sub\n', i_sub)
    %     % filter data
    %     x = train_data(:, :, i_sub);
    %     x_tilda_all(train_idx, :, i_sub) = x';
    %     x_tilda_all(test_idx, :, i_sub) = tar_data';
    % end
    % x_tilda_all = reshape(x_tilda_all, n_tp_total, n_rg*n_sub);
    % x_tilda_all = x_tilda_all';
    % Laplacian pyramid extension
    configAffParams3 = configAffParams1;
    configAffParams3.kNN = 200;
    configAffParams3.self_tune = 10;
    dm_all = zeros(n_tp_total, n_dim);
    % compute only once dXX to speed up
    [dm_all(test_idx, 1), l(1), dXX, dXY, indsXX, indsXY] = calcLaplcianPyramidExtension(configAffParams3, x_tilda_all(:, train_idx), (dm_train(1,:)./lambda2(1))', x_tilda_all(:, test_idx));
    for i = 2 : n_dim
        [dm_all(test_idx, i), l(i)] = calcLaplcianPyramidExtension(configAffParams3, x_tilda_all(:, train_idx), (dm_train(i,:)./lambda2(i))', x_tilda_all(:, test_idx), dXX, dXY, indsXX, indsXY);
    end
    dm_all = dm_all';
    dm_all(:, train_idx) = dm_train;
    dm_all(:, test_idx) = dm_all(:, test_idx) .* lambda2; % don't forget about the lambda
    
    % plot
    % figure 1
    s = zeros(n_tp_total, 1);
    c = zeros(n_tp_total, 1);
    s(train_idx) = 10;
    s(test_idx) = 30;
    % c = [IDX', 17*ones(1, n_tp_test)];
    c(train_idx) = train_label;
    c(test_idx) = test_label;
    
    figure;
    subplot(2,2,1);
    scatter3(dm_all(1, :), dm_all(2, :), dm_all(3, :), s, c, 'filled');
    colormap(cmap)
    caxis([vmin,vmax])
    title('concatenated embed, testing enlarged, 123coord')
    subplot(2,2,2);
    scatter3(dm_all(1, :), dm_all(3, :), dm_all(4, :), s, c, 'filled');
    colormap(cmap)
    caxis([vmin,vmax])
    title('concatenatedembed, testing enlarged, 124coord')
    s(train_idx) = 30;
    s(test_idx) = 10;
    subplot(2,2,3);
    scatter3(dm_all(1, :), dm_all(2, :), dm_all(3, :), s, c, 'filled');
    colormap(cmap)
    caxis([vmin,vmax])
    title('concatenatedembed, training enlarged, 123coord')
    subplot(2,2,4);
    scatter3(dm_all(1, :), dm_all(3, :), dm_all(4, :), s, c, 'filled');
    colormap(cmap)
    caxis([vmin,vmax])
    title('concatenatedembed, training enlarged, 124coord')
    
    % figure 2
    % figure;
    % c_train = c(train_idx);
    % c_train(c_train<2 | c_train>3) = 17;
    % subplot(2,1,1);
    % scatter3(dm_all(1, train_idx), dm_all(2, train_idx), dm_all(4, train_idx), 30, c_train, 'filled');
    % colormap(cmap)
    % caxis([vmin,vmax])
    % title('training embedding, 123coord')
    %
    % subplot(2,1,2);
    % c_test = c(test_idx);
    % c_test(c_test<2 | c_test>3) = 17;
    % scatter3(dm_all(1, test_idx), dm_all(2, test_idx), dm_all(4, test_idx), 30, c_test, 'filled');
    % colormap(cmap)
    % caxis([vmin,vmax])
    % title('test embedding, 123coord')
    
    % plot trajectory
    % figure;
    % for i = 2:3
    %     temp_sum = 0;
    %     count2 = 0;
    %     for j = 1 : length(task_start{i-1})
    %         s = task_start{i-1}(j);
    %         if find(test_idx==s)
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
    s(train_idx) = 10;
    s(test_idx) = 30;
    c(train_idx) = train_label;
    c(test_idx) = test_label;
%     figure;
%     % scatter3(dm_all(1, :), dm_all(2, :), dm_all(3, :), s, c, 'filled');
%     scatter3(dm_all(1, train_idx), dm_all(2, train_idx), dm_all(3, train_idx), s(train_idx), IDX, 'filled');
%     colormap(cmap)
%     caxis([vmin,vmax])
%     hold on
%     for i =  1 : n_tp_test
%         %     plot3(dm_all(1, test_idx(1:end)), dm_all(2, test_idx(1:end)), dm_all(3, test_idx(1:end)), 'linewidth', 5, 'color', cmap(18,:));
%         plot3(dm_all(1, test_idx(1:i)), dm_all(2, test_idx(1:i)), dm_all(3, test_idx(1:i)), 'linewidth', 5, 'color', cmap(18,:));
%         pause(0.1)
%     end
%     hold off
    
    % figure;
    % plot3(temp_sum(1, :), temp_sum(2, :), temp_sum(3, :), 'linewidth', 5, 'color', cmap(i+1,:));
    
    % calculate dwell time
    [D_ext, I_ext] = pdist2(dm_all(:, train_idx)', dm_all(:, test_idx)', 'Euclidean', 'Smallest', 5);
    IDX_list = zeros(size(I_ext, 2), 1);
    for i = 1 : size(I_ext, 2)
        IDX_temp = mode(IDX(I_ext(:, i)));
        IDX_list(i) = IDX_temp;
    end
    for i = 1 : 4
        %         idx_temp = 1:386; % WM
        %         idx_temp = 387:543; % EMO
        %         idx_temp = 544:777; % GAM
        %     idx_temp = 778:1042; % MOT
        %         idx_temp = 1043:1297; % SOC
        %     idx_temp = 1298:1510; % REL
        %     dwell_time_list(ref_sub, i) = sum(IDX_list(idx_temp)==i)/numel(IDX_list(idx_temp));
        dwell_time_list(ref_sub, i) = sum(IDX_list==i)/numel(IDX_list);
    end
    
    for i = 2 : n_tp_test
        if IDX_list(i) ~= IDX_list(i-1)
            change_time(ref_sub) = change_time(ref_sub) + 1;
        end
    end 
        
    % figure;
    % scatter3(dm_all(1, train_idx), dm_all(2, train_idx), dm_all(3, train_idx), s(train_idx), IDX, 'filled');
    % colormap(cmap)
    % caxis([vmin,vmax])
end

stat = [mean(dwell_time_list(idx_subset(1:5),:));
mean(dwell_time_list(idx_subset(6:10),:))];
figure;bar(stat')
ylim([0,0.6])


%% embed using simulated group data, only testing points
% [K_temp, ~, ~] = calcAffinityMat(x_tilda_all(:, test_idx), configAffParams2);
% [dm_new, ~, ~, ~, ~, ~] = calcDiffusionMap(K_temp, configDiffParams2);
% % plot
% s = 20;
% c = true_label_all(test_idx);
% vmin = -0.5;
% vmax = 17.5;
% figure;
% subplot(2,2,1);
% scatter3(dm_new(1, :), dm_new(2, :), dm_new(3, :), s, c, 'filled');
% colormap(cmap)
% caxis([vmin,vmax])
% title('simulated group embed, 123coord')
% subplot(2,2,2);
% scatter3(dm_new(2, :), dm_new(3, :), dm_new(4, :), s, c, 'filled');
% colormap(cmap)
% caxis([vmin,vmax])
% title('simulated group embed, 234coord')
% 
% % embed using single data, only testing points
% [K_temp2, ~, ~] = calcAffinityMat(tar_data, configAffParams2);
% [dm_new, ~, ~, ~, ~, ~] = calcDiffusionMap(K_temp2, configDiffParams2);
% % plot
% s = 20;
% c = true_label_all(test_idx);
% vmin = -0.5;
% vmax = 17.5;
% subplot(2,2,3);
% scatter3(dm_new(1, :), dm_new(2, :), dm_new(3, :), s, c, 'filled');
% colormap(cmap)
% caxis([vmin,vmax])
% title('single sub embed, 123coord')
% subplot(2,2,4);
% scatter3(dm_new(2, :), dm_new(3, :), dm_new(4, :), s, c, 'filled');
% colormap(cmap)
% caxis([vmin,vmax])
% title('single sub embed, 234coord')

%% generate training embedding for comparison (optional)
% [dm, ~, embed, lambda1, lambda2, sigma1, sigma2] = calc2sDM(train_data, n_dim, configAffParams1, configAffParams2, configDiffParams1, configDiffParams2);
% 
% % plot
% s = 20;
% c = true_label_all(test_idx);
% vmin = -0.5;
% vmax = 17.5;
% 
% figure;scatter3(dm(1, :), dm(2, :), dm(3, :), s, c, 'filled');
% colormap(cmap)
% caxis([vmin,vmax])
% 
% figure;scatter3(dm(2, :), dm(3, :), dm(4, :), s, c, 'filled');
% colormap(cmap)
% caxis([vmin,vmax])