function [ output_args ] = visualdata( data )
clf;
figure(1);
scatter(data.landmarks(:,1),data.landmarks(:,2));
hold on; scatter(data.poses(:,1),data.poses(:,2));
legend('landmarks','poses');
axis('equal');

figure(2);
subplot(1,4,1);
spy(data.G_full'*data.G_full);
title('G^T*G: total information');
subplot(1,4,2);
spy(data.G.H'*data.G.H);
title('H^T*H: sensor information');
subplot(1,4,3);
spy(data.G.D'*data.G.D);
title('D^T*D: odometry information');
subplot(1,4,4);
spy(data.G.L'*data.G.L);
title('L^T*L: prior information');
waitforbuttonpress;
end

