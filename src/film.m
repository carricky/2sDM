s = true_label;
c = 30*ones(n_frames, 1);


fig = figure

F(n_frames) = struct('cdata',[],'colormap',[]);
frames_shown = n_frames;
% gif('WM_traj.gif')
for i = 1 : n_frames
%     c = 10:40/i:50;
    if i<=frames_shown
        scatter3(diffusion_map(1,1:i),diffusion_map(2,1:i),diffusion_map(3,1:i), c(1:i), s(1:i), 'filled')
    else
        scatter3(diffusion_map(1,i-frames_shown:i),diffusion_map(2,i-frames_shown :i),diffusion_map(3,i-frames_shown :i), c(i-frames_shown :i), s(i-frames_shown :i), 'filled')
    end
    %xlim([-0.04, 0.02])
    %ylim([-0.06, 0.04])
    %zlim([-0.03, 0.04])
    caxis([0, 3])
    %view(14, 12)
    view(103, 24)
%     view(73, 16)
    drawnow
    F(i) = getframe;
%     frame = getframe(fig);
%     im{i} = frame2im(frame);
        %     gif
    
end

% [A,map] = rgb2ind(im{1},256);
% imwrite(A,map,'F_low.gif','gif','LoopCount',Inf,'DelayTime',1);