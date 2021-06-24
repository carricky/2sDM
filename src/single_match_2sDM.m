%% seperate the resting state data
% sub = 61; %gf=7
% sub = 13; %gf=24
sub = 1;
rest_data = data(:, end-2362+1:end, sub);
data(:, end-2362+1:end, :) = [];
true_label_all(end-2362+1:end) = [];

%% remove first wm LR
wm_data = data(:, 1:387, 1);
data(:, 1:387, :) = [];

%% embed task part
configAffParams1.dist_type = 'euclidean';
configAffParams1.kNN = 500;
configAffParams1.self_tune = 0;

configDiffParams1.t = 1;
configDiffParams1.normalization='lb';

n_frames = size(data, 2);
n_subs = size(data, 3);
n_d = 5;
configDiffParams1.maxInd = n_d+1;

embed = zeros(n_frames, (n_subs-1)*n_d);
count = 0;
for i_sub = 1 : n_subs
    if i_sub ~= sub
        count = count + 1;
        disp(count/(n_subs-1))
        data_ind = data(:, :, i_sub);
        
        [K, ~] = calcAffinityMat(data_ind, configAffParams1);
        [diffusion_map, ~, ~, ~, ~, ~] = calcDiffusionMap(K, configDiffParams1);
        
        embed(:, (count-1)*n_d+1:(count)*n_d) = diffusion_map';
    end
end
[K, ~] = calcAffinityMat(embed', configAffParams1);


%% embed resting state part
rest_len = 600;
X_temp = data(:, :, sub);
X_temp = [X_temp, rest_data(:, 1:rest_len)];

configAffParams2 = configAffParams1;
configDiffParams2 = configDiffParams1;

[K_temp, ~] = calcAffinityMat(X_temp, configAffParams1);
% K_new = [[K,K_temp(1:end-2362, end-2362+1:end)];K_temp(end-2362+1:end, :)];
K_new = [[K,K_temp(1:end-rest_len, end-rest_len+1:end)];K_temp(end-rest_len+1:end, :)];


[dm, ~, ~, ~, ~, ~] = calcDiffusionMap(K_new, configDiffParams2);

%% embed Language part
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/signal_LR_LANGUAGE_NOGSR.mat')
load('/Users/siyuangao/Working_Space/fmri/data/HCP515/signals/signal_RL_LANGUAGE_NOGSR.mat')
lan_data = signal_LR(:, :, 4);
lan_data = cat(2, lan_data, signal_RL(:, :, 4));
missing_nodes = [249, 239, 243, 129, 266, 109, 115, 118, 250];
lan_data(missing_nodes, :, :, :) = [];
lan_data(:, missing_nodes, :, :) = [];
lan_data = zscore(lan_data, 0, 1);

X_temp = data(:, :, 4);
X_temp = [X_temp, lan_data];

configAffParams2 = configAffParams1;
configDiffParams2 = configDiffParams1;

[K_temp, ~] = calcAffinityMat(X_temp, configAffParams1);
K_new = [[K,K_temp(1:end-623, end-623+1:end)];K_temp(end-623+1:end, :)];
[dm, ~, ~, ~, ~, ~] = calcDiffusionMap(K_new, configDiffParams2);

n_frames=size(dm, 2);
s = 25*ones(n_frames, 1);
c = [true_label_all,17*ones(1,n_frames-3020)];
vmin = -0.5;
vmax = 17.5;

figure;scatter3(dm(1, :), dm(2, :), dm(3, :), s, c, 'filled');
colormap(cmap)
caxis([vmin,vmax])

%% embed Working Memory part

% wm_data = data(:, 1:387, 1);

X_temp = data(:, :, 1);
X_temp = [X_temp, wm_data];

configAffParams2 = configAffParams1;
configDiffParams2 = configDiffParams1;

[K_temp, ~] = calcAffinityMat(X_temp, configAffParams1);
K_new = [[K,K_temp(1:end-387, end-387+1:end)];K_temp(end-387+1:end, :)];
[dm, ~, ~, ~, ~, ~] = calcDiffusionMap(K_new, configDiffParams2);

n_frames=size(dm, 2);
s = 25*ones(n_frames, 1);
% c = [true_label_all,17*ones(1,n_frames-3020)];
c = [true_label_all(388:end), 17*ones(1, 387)];
% c = [true_label_all(775:end), true_label_all(1:387)];
% c = [17*ones(size(true_label_all(775:end))), true_label_all(1:387)];
vmin = -0.5;
vmax = 17.5;

figure;scatter3(dm(1, 388:end), dm(2, 388:end), dm(3, 388:end), s(388:end), c, 'filled');
figure;scatter3(dm(1, :), dm(2, :), dm(3, :), s(:), c, 'filled');
colormap(cmap)
caxis([vmin,vmax])

%% plot it
n_frames = size(dm, 2);
s = 25*ones(n_frames, 1);
c = [true_label_all,17*ones(1,n_frames-3020)];
vmin = -0.5;
vmax = 17.5;

figure;scatter3(dm(1, :), dm(2, :), dm(3, :), s, c, 'filled');
colormap(cmap)
caxis([vmin,vmax])

% figure;scatter3(dm(3, :), dm(4, :), dm(5, :), s, c, 'filled');
% colormap(cmap)
% caxis([vmin,vmax])
% 
% 
% task_ind = 1:n_frames-2362;
% figure;scatter3(dm(1, task_ind), dm(2, task_ind), dm(3, task_ind), s(task_ind), c(task_ind), 'filled');
% colormap(cmap)
% caxis([vmin,vmax])
% 
% figure;scatter3(dm(3, :), dm(4, :), dm(5, :), s, c, 'filled');
% colormap(cmap)
% caxis([vmin,vmax])
% 

%% config task description
num_task = 9;
task_set = cell(num_task,1);
task_set{1} = [1,2]+1;
task_set{2} = [3,4]+1;
task_set{3} = [5,6]+1;
task_set{4} = [7,8,9,10,11]+1;
task_set{5} = [12,13]+1;
task_set{6} = [14,15]+1;
task_set{7} = [0];
task_set{8} = [16]+1;
task_set{9} = [1];

task_name = cell(num_task,1);
task_name{1} = "WM";
task_name{2} = "Emotion";
task_name{3} = "Gambling";
task_name{4} = "Motor";
task_name{5} = "Social";
task_name{6} = "Relational";
task_name{7} = "Fixation";
task_name{8} = "Rest";
task_name{9} = "Cue";

%% some cool animation
figure;scatter3(dm(1, :), dm(2, :), dm(3, :), s, c, 'filled');
% figure;scatter3(dm(3, :), dm(4, :), dm(5, :), s, c, 'filled');
colormap(cmap)
pausetime = 4;
caxis([vmin,vmax])
while 1
    for i = 1 : num_task
        temp_map = 0.3*ones(size(cmap));
%         temp_map(1,:) = cmap(1,:);
        temp_map(18,:) = [0.9,0.9,0.9];
        temp_map(task_set{i}+1,:) = cmap(task_set{i}+1, :);
        
        colormap(temp_map)
        caxis([vmin,vmax])
        title(task_name{i})
        pause(pausetime)
    end
end

%% different subjects' embedding
figure;
n_frames = size(dm, 2);
s = 15*ones(n_frames, 1);
c = [true_label_all,17*ones(1,n_frames-3020)];
count = 1;
for sub = [13, 17, 35, 55, 140, 255, 300, 339,  172, 102, 216, 61, 116, 157, 306, 307]
    load(['/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/output/taskandrest600_nn500_sub',num2str(sub),'.mat'])
    subplot(4,4,count);
    scatter3(dm(1, :), dm(2, :), dm(3, :), s, c, 'filled');
    colormap(cmap)
    caxis([vmin,vmax])
    title(['gF=', num2str(all_behav(sub))])
    count = count + 1;
end

%% different k's embedding
figure;
count = 1;
n_frames = size(dm, 2);
s = 15*ones(n_frames, 1);
c = [true_label_all,17*ones(1,n_frames-3020)];
for k = 300:200:1500
    load(['/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/output/test_knn/task_nn', num2str(k) ,'.mat'])
    subplot(2,4,count);
    scatter3(dm(1, :), dm(2, :), dm(3, :), s, c, 'filled');
    colormap(cmap)
    caxis([vmin,vmax])
    title(['k=', num2str(k)])
    count = count + 1;
end
    
