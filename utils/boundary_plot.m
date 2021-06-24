s = 20*ones(num_t_train, 1);
c = true_label_all(train_range);
vmin = -0.5;
vmax = 17.5;
fa = 0.1;
figure;
dim = [1,2,4];
P = dm(dim, 1:772)';
k = boundary(P);
hold on
trisurf(k,P(:,1),P(:,2),P(:,3),'Facecolor',cmap(3,:),'FaceAlpha',fa)

P = dm(dim, 773:1086)';
k = boundary(P);
trisurf(k,P(:,1),P(:,2),P(:,3),'Facecolor',cmap(5,:),'FaceAlpha',fa)

P = dm(dim, 1087:1554)';
k = boundary(P);
trisurf(k,P(:,1),P(:,2),P(:,3),'Facecolor',cmap(7,:),'FaceAlpha',fa)

P = dm(dim, 1555:2084)';
k = boundary(P);
trisurf(k,P(:,1),P(:,2),P(:,3),'Facecolor',cmap(9,:),'FaceAlpha',fa)

P = dm(dim, 2085:2594)';
k = boundary(P);
trisurf(k,P(:,1),P(:,2),P(:,3),'Facecolor',cmap(13,:),'FaceAlpha',fa)

P = dm(dim, 2595:3020)';
k = boundary(P);
trisurf(k,P(:,1),P(:,2),P(:,3),'Facecolor',cmap(15,:),'FaceAlpha',fa)


scatter3(dm(dim(1), :), dm(dim(2), :), dm(dim(3), :), s, c, 'filled');
colormap(cmap)
caxis([vmin,vmax])