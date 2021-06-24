addpath('/Users/siyuangao/Working_Space/fmri/code_siyuan/2018summer/2sDM/utils/');
% 
% % load the data
% dir = 'xxx'; % path to the data
% load(dir);

%% load the signals and true labels
range = 1:3542;

%% config the parameters
max_corr = 0;
max_emotion = 0;
max_t = 0;
max_k = 0;
% for k = 300:200:1000
%     for t = 1:1:7
k=900
t=7
        configAffParams1.dist_type = 'euclidean';
        configAffParams1.kNN = k;
        configAffParams1.self_tune = 0;
        
        configDiffParams1.t = t;
        configDiffParams1.normalization='lb';
        
        configAffParams2 = configAffParams1;
        configDiffParams2 = configDiffParams1;
        
        n_d = 7;
        
        %% parse parameters
        n_regions = size(data, 1);
        n_frames = size(data, 2);
        n_subs = size(data, 3);
        
        %% run the code
        
        [dm, K] = calc2sDM(data(:, range,:), n_d, configAffParams1, configAffParams2, configDiffParams1, configDiffParams2);
        
        
        % visualize the first three non-trivial dimensions
        %     figure;
        count=1;
        
        for i = [1,2,3, 26,27, 28, 29, 30]
            s = 20*ones(length(range)-3, 1);
            c = emotion(range,i);
            c = c(1:end-3);
            c = [1*ones(451-4,1);2*ones(441-8,1);3*ones(438-8,1);4*ones(488-8,1);5*ones(462-8,1);6*ones(439-8,1);7*ones(542-8,1);8*ones(338-4-1-3,1)];
            c(c~=0)=1;
            
            vmin = -0.5;
            vmax = 17.5;
            
            %         subplot(3,3,count);scatter3(dm(1, 4:end), dm(2, 4:end), dm(3, 4:end), s, c, 'filled');
            % figure;imagesc(K)
            count = count + 1;
            for j = 1:5
                if abs(corr(c, dm(j,4:end)'))>max_corr   
                    max_corr = abs(corr(c, dm(j,4:end)'));
                    max_emotion = i;
                    max_t = t;
                    max_k = k;
                end
            end
        end
        max_corr
%     end
% end
