% load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/output/taskwrelation_nn500.mat')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/true_label/true_label_all_withrest_coarse.mat')
load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/cmap10.mat')

%% get task start and end time stamp
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

%% static trajectory no task block
dim = [1,2,4];

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
        temp_sum = temp_sum + dm(:, s:e);
        count2 = count2+1;
    end
    temp_sum = temp_sum / count2;
    
    %     plot3(temp_sum(dim(1), :), -temp_sum(dim(2), :), temp_sum(dim(3), :), '-o', 'MarkerFaceColor',cmap10(i+1,:), 'color', cmap10(i+1,:), 'linewidth', 2, 'MarkerSize',8);
    plot3(temp_sum(dim(1), :), -temp_sum(dim(2), :), temp_sum(dim(3), :),...
        'color', cmap10(i+1,:), 'linewidth', 8);
end

vmin = 0.5;
vmax = 5.5;
f = scatter3(dm(dim(1), :), -dm(dim(2), :), dm(dim(3), :), 20, kmeans_idx, 'filled');
f.MarkerFaceAlpha = 0.5;
colormap(cmap4)
caxis([vmin,vmax])
grid on
hold off
zlim([-0.035, 0.02])
ax = gca;
ax.FontSize = 16;

%% static trajectory
dim = [1,2,4];

% get the average trajectory
figure;
hold on
% for i = 2 : 16
for i = [2,3]
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
        temp_sum = temp_sum + dm(:, s:e);
        count2 = count2+1;
    end
    temp_sum = temp_sum / count2;

    plot3(temp_sum(dim(1), :), -temp_sum(dim(2), :), temp_sum(dim(3), :),...
        'linewidth', 20, 'color', cmap10(i+1,:));
    
end

f = scatter3(dm(dim(1), :), -dm(dim(2), :), dm(dim(3), :), 20, kmeans_idx, 'filled');
f.MarkerFaceAlpha = 0.5;
colormap(cmap4)
caxis([0.5, 5.5])
zlim([-0.035, 0.02])
ax = gca;
ax.FontSize = 16;

grid on
hold off
