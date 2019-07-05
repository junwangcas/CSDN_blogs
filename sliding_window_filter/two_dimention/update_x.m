function [data] = update_x(data,delta_x)
size_pose = size(data.poses,1);
size_landmark = size(data.landmarks,1);
delta_pose = delta_x(1:size_pose*2);
delta_landmark = delta_x(size_pose*2+1:end);

data.poses = data.poses + reshape(delta_pose',[2,size_pose])';
data.landmarks = data.landmarks + reshape(delta_landmark',[2,size_landmark])';
end

