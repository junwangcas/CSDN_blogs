function [ data ] = reset_statevector( data )
% 
for i = 1:size(data.poses,1)
    data.poses(i,:) = [0,0];
end

for i = 1:size(data.landmarks,1)
    data.landmarks(i,:) = [0,0];
end
end

