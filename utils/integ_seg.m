%% dynamic
addpath('/Users/siyuangao/Working_Space/fmri/bctnet/BCT/2017_01_15_BCT/')
load('/Users/siyuangao/Working_Space/fmri/data/parcellation_and_network/map259.mat')
map259(map259==6) = 5;
map259(map259==7) = 5;
network_name = {'MF', 'FP', 'DMN', 'Motor', 'Visual', 'Subcortical', 'Cerebellum'};
%% HCP static conn
% load('/Users/siyuangao/Working_Space/fmri/data/HCP515/all_mats.mat')
% load('/Users/siyuangao/Working_Space/fmri/data/map268.mat')
% for sub = 1 : 30
%     for i = 1 : 9
%         %         c_signal = train_data(:, IDX==i, :);
%         %         [c_conn, p]= corr(train_data(:, IDX==i, sub)');
%         c_conn = all_mats(:, :, sub, i)';
%         
% %         c_conn(c_conn<0) = 0;
%         [P_pos,P_neg] = participation_coef_sign(c_conn,map268);
%         pc(sub, i) = mean(P_pos);
%         
% %         c_conn(p>0.01) = 0;
% %         c_conn(c_conn<0) = 0;
% %         P = participation_coef(c_conn, map259, 0);
% %         pc(sub, i) = mean(P);
% %         subplot(2,2,i);
% %         imagesc(c_conn);
%     end
% end
% 
% mean(pc)
% std(pc)

%% HCP all task simple Pearson correlation
% load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/data_nozscore.mat')
% tp_idx = 10:3020-10;
% sub_idx = 1 : 30;
% pc = zeros(numel(tp_idx), numel(sub_idx));
% load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/true_label/true_label_all_with_rest.mat')
% for sub = sub_idx
%     sub
%     for i = tp_idx
%         temp_conn = corr(data(:, i-9:i+9, sub)');
%         [P_pos,P_neg] = participation_coef_sign(temp_conn, map259);
%         pc(i, sub) = mean(P_pos);
%     end
% end
% pc_mean = mean(pc, 2);
% pc_std = std(pc, 0, 2);
% figure;
% plot(pc_mean(tp_idx))
% figure;
% imagesc(true_label_all(1:386))
% % hold off
% alpha(.5)

%% HCP all task MTD dynamic conn and participation coefficient calculated
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/data.mat')

sub_idx = 1 : 390;
tp_idx = 1 : 3020;
n_rg = size(data, 1);
pc = zeros(n_rg, numel(tp_idx), numel(sub_idx));
mc = zeros(n_rg, numel(tp_idx), numel(sub_idx));
% create temporal label
task_idx = [[1, 386];[387, 772];[773, 929];[930, 1086];[1087, 1320];
    [1321, 1554];[1555, 1819];[1820, 2084];[2085, 2339];[2340, 2594];
    [2595, 2807];[2808, 3020]];
temporal_label = zeros(1, 3020);
for i_task = 1 : size(task_idx, 1)
    temporal_label(task_idx(i_task, 1):task_idx(i_task, 2)) = i_task;
end
% tic
% [pc_global6, ~, pc_region6, ~] = computeDynamicNetworkMeasure(data(:, tp_idx, sub_idx), map259, 15, temporal_label);
% toc
[pc_global, pc_region] = computeDynamicNetworkMeasureDynamicCommunity(data(:, tp_idx, sub_idx), 15, temporal_label);

%% correlate embedding with participation coefficient
corr(pc_global(pc_global~=0)', dm(:, pc_global~=0)')

figure;scatter(dm(1, pc_global~=0), -dm(2, pc_global~=0), 20, pc_global(pc_global~=0), 'filled');

% caxis([0.44,0.55])

%% boxplot by brain states
figure;
h = boxplot(pc_global(pc_global~=0), kmeans_idx(pc_global~=0), 'Colors', cmap4(1:4, :))
set(h,{'linew'},{2})

%% plot by subnetwork and the whole brain
% manifold plot
count = 1;
figure;
for i_network = [1,2,3,4,5,8,9]
    subplot(2,4,count)
    pc_mean = mean(pc_region(map259==i_network, :));
    network_plot_idx = (pc_mean~=0);
%     scatter3(dm_train(1, network_plot_idx), dm_train(2, network_plot_idx), dm_train(3, network_plot_idx), 25, pc_mean(network_plot_idx), 'filled');
    scatter(dm_train(1, network_plot_idx), dm_train(2, network_plot_idx), 15, pc_mean(network_plot_idx), 'filled');
    colormap('jet')
    title(network_name{count})
%     caxis([0.85, 0.88])
    corr(dm_train(:, network_plot_idx)', pc_mean(network_plot_idx)')
    count = count + 1;
end
network_plot_idx = (pc_global~=0);
subplot(2,4,count)
scatter(dm_train(1, network_plot_idx), dm_train(2, network_plot_idx), 15, pc_global(network_plot_idx), 'filled');
title('Whole Brain')
corr(dm_train(:, network_plot_idx)', pc_global(network_plot_idx)')

% box plot
count = 1;
figure;
for i_network = [1,2,3,4,5,8,9]
    subplot(2,4,count)
    pc_mean = mean(pc_region(map259==i_network, :));
    boxplot(pc_mean(plot_idx), IDX(plot_idx))
    title(network_name{count})
%     caxis([0.85, 0.88])
    count = count + 1;
end
network_plot_idx = (pc_global~=0);
subplot(2,4,count)
boxplot(pc_global(network_plot_idx), IDX(network_plot_idx))
title('Whole Brain')

