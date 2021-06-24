% load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/output/taskwrelation_nn500.mat')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/true_label/true_label_all_with_rest.mat')
load('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/cmap18.mat')

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
    plot(dm(1, begin:task_length(i)+begin-1), dm(2, begin:task_length(i)+begin-1));
    %     hold on
    title(task_name{i})
    begin = task_length(i)+begin;
end

%% get task start and end time stamp
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

%% static trajectory
dim = [1,2,4];

% get the average trajectory
figure;
% hold on
for i = 2 : 16
% for i = [2,3,7,9]
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

% vmin = 0.5;
% vmax = 6.5;
% f = scatter3(dm(dim(1), :), -dm(dim(2), :), dm(dim(3), :), 80, kmeans_idx, 'filled');
% f.MarkerFaceAlpha = 0.3;
% colormap(cmap6)
% caxis([vmin,vmax])
hold off

%% dynamic trajectory
% get the average trajectory
IDX = kmeans(dm(:, :)', 4);

task_label = [2, 4, 6, 8, 13, 15];
all_traj = cell(15, 1);
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
    all_traj{i-1} = temp_sum(dim, :);
end

% plot the dynamic trajectories
% max_length = 40;
% counter = 0;
% figure;
% for repeat = 1 : 100
%     
%     s = 40*ones(3020, 1);
%     c = true_label_all(1:3020);
%     vmin = 0.5;
%     vmax = 4.5;
%     a = scatter3(dm(dim(1), :), dm(dim(2), :), dm(dim(3), :), s, IDX, 'filled');
%     colormap(cmap6(1:4,:))
%     caxis([vmin,vmax])
%     
%     hold on
%     for i = 1 : max_length
%         for j = 1 : 15
%             end_point = min(size(all_traj{j}, 2), i);
%             plot3(all_traj{j}(1, 1:end_point), all_traj{j}(2, 1:end_point), all_traj{j}(3, 1:end_point), 'linewidth', 5, 'color', cmap18(j+2,:));
%             view(-37+counter, 29)
%             counter=counter+0.1;
%         end
%         
%         pause(0.05)
%     end
%     for i = 1 : max_length
%         view(-37+counter, 29)
%         counter = counter+0.2;
%         pause(0.1)
%     end
% %     pause(1)
%     hold off
% end


%% rearrange rows by task block type
% task_start = {};
% task_end = {};
% for i = 0:16
%     temp = find(true_label_all == i);
%     task_start_temp = temp(1);
%     task_end_temp = [];
%     for j = 2 : length(temp)
%         if temp(j) ~= temp(j-1)+1
%             task_start_temp = [task_start_temp, temp(j)];
%             task_end_temp = [task_end_temp, temp(j-1)];
%         end
%     end
%     task_end_temp = [task_end_temp, temp(end)];
%     task_start{i+1} = task_start_temp;
%     task_end{i+1} = task_end_temp;
% end
% 
% new_dm = [];
% for i = 0 : 16
%     for j = 1 : length(task_start{i+1})
%         s = task_start{i+1}(j);
%         e = task_end{i+1}(j);
%         new_dm = [new_dm,dm(:, s:e)];
%     end
% end
% 
%% calculate each task block's length
% task_length = [];
% task_pos = [0];
% for i = 1:17
%     task_start_temp = task_start{i};
%     task_end_temp = task_end{i};
%     length_temp = 0;
%     for j = 1 : length(task_start_temp)
%         length_temp = length_temp + task_end_temp(j)-task_start_temp(j) + 1;
%     end
%     task_length = [task_length;length_temp];
%     task_pos = [task_pos; length_temp + task_pos(i)];
% end
% task_pos = task_pos(2:end);

%% RS trajectory
dim = [1,2,4];
% plot the original manifold as reference
s = 3021;
% e = 4220;
e = 3270;

figure;

plot3(dm_all(dim(1), s:e), dm_all(dim(2), s:e), dm_all(dim(3), s:e), 'linewidth', 5);
hold on


s = 5*ones(3020, 1);
c = true_label_all(1:3020);
vmin = -0.5;
vmax = 17.5;
a = scatter3(dm(dim(1), :), dm(dim(2), :), dm(dim(3), :), s, c, 'filled');
colormap(cmap18)
caxis([vmin,vmax])
hold off

