addpath('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/');
% 
% % load the data
% dir = 'xxx'; % path to the data
% load(dir);

%% load the signals and true labels
% generate_data;

%% config the parameters
configAffParams1.dist_type = 'euclidean';
configAffParams1.kNN = 500;
configAffParams1.self_tune = 0;

configDiffParams1.t = 1;
configDiffParams1.normalization='lb';

configAffParams2 = configAffParams1;
configDiffParams2 = configDiffParams1;

n_d = 7;



%% run the code

[dm, K] = calc2sDM(data, n_d, configAffParams1, configAffParams2, configDiffParams1, configDiffParams2);

% visualize the first three non-trivial dimensions
% parse parameters
n_frames = size(dm, 2);

s = 20*ones(n_frames, 1);
c = true_label_all(1:n_frames);
vmin = -0.5;
vmax = 17.5;

figure;scatter3(dm(1, :), dm(2, :), dm(3, :), s, c, 'filled');
colormap(cmap)
caxis([vmin,vmax])

figure;scatter3(dm(3, :), dm(4, :), dm(5, :), s, c, 'filled');
colormap(cmap)
caxis([vmin,vmax])

% kmeans
IDX = kmeans(dm([1,2,4], :)', 4);
figure;scatter3(dm(1, :), dm(2, :), dm(4, :), s, IDX, 'filled');
colormap(cmap)
caxis([vmin,vmax])

%% config task description
num_task = 9;
task_set = cell(num_task,1);
task_set{1} = [1,2]+1; %+1 because of the cue
task_set{2} = [3,4]+1;
task_set{3} = [5,6]+1;
task_set{4} = [7,8,9,10,11]+1;
task_set{5} = [12,13]+1;
task_set{6} = [14,15]+1;
task_set{7} = [0];
task_set{8} = [1];
task_set{9} = [17];

task_name = cell(num_task,1);
task_name{1} = "WM";
task_name{2} = "Emotion";
task_name{3} = "Gambling";
task_name{4} = "Motor";
task_name{5} = "Social";
task_name{6} = "Relational";
task_name{7} = "Fixation";
task_name{8} = "Cue";
task_name{9} = "Rest";

%% some cool animation
figure;scatter3(dm(1, :), dm(2, :), dm(3, :), s, c, 'filled');
% figure;scatter3(dm(3, :), dm(4, :), dm(5, :), s, c, 'filled');
colormap(cmap)
pausetime = 3;

caxis([vmin,vmax])
while 1
    for i = 1 : num_task
        temp_map = 0.8*ones(size(cmap));
%         temp_map(1,:) = cmap(1,:);
        temp_map(task_set{i}+1,:) = cmap(task_set{i}+1, :);
        colormap(temp_map)
        caxis([vmin,vmax])
        title(task_name{i})
        pause(pausetime)
    end
end

%% some movie

for i =  1:5
    frame_list = [];
    for j = 1 : numel(task_set{i})
        frame_list = [frame_list,find(true_label_all==task_set{i}(j))];
    end
    frame_list = sort(frame_list);
    frame_index = zeros(size(dm,2), 1);
    frame_index(frame_list) = 1;
    F(sum(frame_index)) = struct('cdata',[],'colormap',[]);
    
    figure;
    scatter3(dm(1,~frame_index),dm(2,~frame_index), dm(3,~frame_index),s(~frame_index), c(~frame_index), 'filled')
    title(task_name{i})
    temp_map = 0.8*ones(size(cmap));
    temp_map(task_set{i}+1,:) = cmap(task_set{i}+1,:);
    colormap(temp_map)
    hold on;
    for j = 1 : sum(frame_index)
        scatter3(dm(1,frame_list(j)),dm(2,frame_list(j)),dm(3,frame_list(j)), s(frame_list(j)), c(frame_list(j)), 'filled')
        
        caxis([vmin, vmax])
        
        %     view(103, 24)
        drawnow
        F(j) = getframe;
        %     frame = getframe(fig);
        %     im{i} = frame2im(frame);
        %     gif
        
    end
end

%% test different ts
for t = 1:1:10
    configDiffParams1.t = t;
    configDiffParams2 = configDiffParams1;
    
    [dm, K] = calc2sDM(data(:,1:3020,1:30), n_d, configAffParams1, configAffParams2, configDiffParams1, configDiffParams2);
    
    % visualize the first three non-trivial dimensions
    % parse parameters
    n_frames = size(dm, 2);
    
    s = 20*ones(n_frames, 1);
    c = true_label_all(1:n_frames);
    vmin = -0.5;
    vmax = 17.5;
    
    figure;scatter3(dm(1, :), dm(2, :), dm(3, :), s, c, 'filled');
    colormap(cmap)
    caxis([vmin,vmax])
    title(t)
    figure;imagesc(dm)
    title(t)
end



