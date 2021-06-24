load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/cmap10.mat')

%% 2sDM scatter plot
figure;
plot_order = [1,2,3,4,5,6,8,7,9];
for i = 0:8
    subplot(3,3,plot_order(i+1));
    scatter(dm(1, c==i), -dm(2, c==i), 20, c(c==i), 'filled');
    colormap(cmap10)
    caxis([vmin, vmax])
    xlim([-0.02, 0.03])
    ylim([-0.03, 0.03])
end

%% 2sDM temporal gradient plot
figure;
for i = 0:1
    subplot(3,3,i+1);
    scatter(dm(1, c==i), -dm(2, c==i), 20, c(c==i), 'filled');
    colormap(cmap10)
    caxis([vmin, vmax])
    xlim([-0.02, 0.03])
    ylim([-0.03, 0.03])
end

plot_order = [2,3,4,5,7,6,8]+1;
for i = 2:8
    temporal_idx = find(c==i);
    temporal_reidx = ones(numel(temporal_idx), 1);
    count = 2;
    start_idx = 1;
    for j = 2 : numel(temporal_idx)
        if temporal_idx(j)==(temporal_idx(j-1)+1)
            temporal_reidx(j) = count;
            count = count + 1;
        else
            end_idx = j-1;
            temporal_reidx(start_idx:end_idx) = temporal_reidx(start_idx:end_idx) / (end_idx-start_idx+1);
            start_idx = j;
            count = 1;
            temporal_reidx(j) = count;
            count = count + 1;
        end
    end
    end_idx = numel(temporal_idx);
    temporal_reidx(start_idx:end_idx) = temporal_reidx(start_idx:end_idx) / (end_idx-start_idx+1);
    
    current_color = cmap10(i+1, :);
    color_gradient = 0.95-temporal_reidx.*(0.95-current_color);
    
    subplot(3,3,plot_order(i-1));
    scatter(dm(1, temporal_idx), -dm(2, temporal_idx), 20, color_gradient, 'filled');
    xlim([-0.02, 0.03])
    ylim([-0.03, 0.03])
end

%% 2sPCA scatter plot
figure;
plot_order = [1,2,3,4,5,6,8,7,9];
for i = 0:8
    subplot(3,3,plot_order(i+1));
    scatter(score(c==i, 1), score(c==i, 2), 20, c(c==i), 'filled');
    colormap(cmap10)
    caxis([vmin, vmax])
    xlim([-60, 60])
    ylim([-40, 40])
end

%% 2sPCA temporal gradient plot
figure;
for i = 0:1
    subplot(3,3,i+1);
    scatter(score(c==i, 1), score(c==i, 2), 20, c(c==i), 'filled');
    colormap(cmap10)
    caxis([vmin, vmax])
    xlim([-60, 60])
    ylim([-40, 40])
end

plot_order = [2,3,4,5,7,6,8]+1;
for i = 2:8
    temporal_idx = find(c==i);
    temporal_reidx = ones(numel(temporal_idx), 1);
    count = 2;
    start_idx = 1;
    for j = 2 : numel(temporal_idx)
        if temporal_idx(j)==(temporal_idx(j-1)+1)
            temporal_reidx(j) = count;
            count = count + 1;
        else
            end_idx = j-1;
            temporal_reidx(start_idx:end_idx) = temporal_reidx(start_idx:end_idx) / (end_idx-start_idx+1);
            start_idx = j;
            count = 1;
            temporal_reidx(j) = count;
            count = count + 1;
        end
    end
    end_idx = numel(temporal_idx);
    temporal_reidx(start_idx:end_idx) = temporal_reidx(start_idx:end_idx) / (end_idx-start_idx+1);
    
    current_color = cmap10(i+1, :);
    color_gradient = 0.95-temporal_reidx.*(0.95-current_color);
    
    subplot(3,3,plot_order(i-1));
    scatter(score(temporal_idx, 1), score(temporal_idx, 2), 20, color_gradient, 'filled');
    xlim([-60, 60])
    ylim([-40, 40])
end

%% 2sDM trajectory plot
dim = [1,2,4];

% get the average trajectory
figure;
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
    subplot(3, 3, i+1)
    %     plot3(temp_sum(dim(1), :), -temp_sum(dim(2), :), temp_sum(dim(3), :), '-o', 'MarkerFaceColor',cmap10(i+1,:), 'color', cmap10(i+1,:), 'linewidth', 2, 'MarkerSize',8);
    plot(temp_sum(dim(1), :), -temp_sum(dim(2), :), 'color', cmap10(i+1,:), 'linewidth', 8);
    colormap(cmap10)
    caxis([vmin, vmax])
    xlim([-0.02, 0.03])
    ylim([-0.03, 0.03])
end

%% 2sPCA trajectory plot
dim = [1,2,4];
% score = score';
% get the average trajectory
figure;
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
    subplot(3, 3, i+1)
    %     plot3(temp_sum(dim(1), :), -temp_sum(dim(2), :), temp_sum(dim(3), :), '-o', 'MarkerFaceColor',cmap10(i+1,:), 'color', cmap10(i+1,:), 'linewidth', 2, 'MarkerSize',8);
    plot(-temp_sum(dim(1), :), -temp_sum(dim(2), :), 'color', cmap10(i+1,:), 'linewidth', 8);
    colormap(cmap10)
    caxis([vmin, vmax])
    xlim([-50, 60])
    ylim([-35, 40])
end

%% individual trajectory 16 category
task_start = {};
task_end = {};
for i = 2:16
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


dim = [1,2,4];

% 2sDM
figure;
for i = 2 : 16
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
    subplot(4, 4, i)
    plot(temp_sum(dim(1), :), -temp_sum(dim(2), :), 'linewidth', 7, 'color', cmap18(i+1,:));
    
    xlim([-0.02, 0.03])
    ylim([-0.03, 0.03])
end

% 2sDM
figure;
for i = 2 : 16
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
    subplot(4, 4, i)
    plot(-temp_sum(dim(1), :), -temp_sum(dim(2), :), 'linewidth', 7, 'color', cmap18(i+1,:));
    
    xlim([-50, 60])
    ylim([-35, 40])
end