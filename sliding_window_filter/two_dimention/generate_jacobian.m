function [data] = generate_jacobian(data)
% generate jacobian functions;
%% prepare jacobian on gz;
num_error = size(data.observations,1)*2 +size(data.odoms,1)*2 + 2;
num_variables = size(data.poses,1)*2 + size(data.landmarks,1)*2;
data.G.H = zeros(num_error, num_variables); 
pos_obs_past = 0;
for i = 1:size(data.observations,1)
    obs = data.observations(i,:);
    z = obs(3:4);
    h_x = data.landmarks(obs(2),:) - data.poses(obs(1),:);
    
    pos_pose = [(obs(1) - 1)*2 + 1:(obs(1) - 1)*2 + 2];
    pos_landmark = [size(data.poses,1)*2 + (obs(2) - 1)*2 + 1: size(data.poses,1)*2 + (obs(2) - 1)*2 + 2];
    pos_obs = [pos_obs_past + 1:pos_obs_past + 2];
    
    data.G.H(pos_obs, pos_pose) = [1,0;0,1];
    data.G.H(pos_obs, pos_landmark) = [-1,0;0,-1];
    pos_obs_past = pos_obs_past + 2;
end

%% on gf;
data.G.D = zeros(num_error, num_variables);
for i = 1:size(data.odoms,1)
    odom = data.odoms(i,:);
    f_x = data.poses(i+1,:) - data.poses(i,:);
    pos_pose1 = [(i-1)*2 + 1:(i-1)*2 + 2];
    pos_pose2 = [(i)*2 + 1:(i)*2 + 2];
    pos_obs = [pos_obs_past + 1:pos_obs_past + 2];
    data.G.D(pos_obs,pos_pose1) = [1,0;0,1];
    data.G.D(pos_obs,pos_pose2) = [-1,0;0,-1];
    pos_obs_past = pos_obs_past + 2;
end

%% on first pose
data.G.L = zeros(num_error, num_variables);
pos_pose = [1:2];
pos_obs = [pos_obs_past + 1:pos_obs_past + 2];
data.G.L(pos_obs,pos_pose) = [-1,0;0,-1];
end

