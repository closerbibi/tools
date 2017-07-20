function [pc, I] = processDepthImage_frompcdtoHHA(num)
% function [pc, N, yDir, h, pcRot, NRot] = processDepthImage(z, missingMask, C)
% Input: 
%   z is in metres 
%   C is the camera matrix

% AUTORIGHTS

tic;


% example: b = readNPY('a.npy');
pcdpath_r3 = '../parse-pcd/output1000k/';
pcN3 = readNPY(sprintf('%s/image%04d_pc.npy',pcdpath_r3,num));

% original pc & N : 427x561x3
pc = pcN3(:,1:3)*100;
N = pcN3(:,4:6);

% new pc: (x, z, -y), z is depth
h = pc(:,3);
yMin = prctile(h(:), 0); if(yMin > -90.) yMin = -130.; end
h = h-yMin;

%% estimate angle
yDir = [0 0 1];
angl = acosd(min(1,max(-1,sum(bsxfun(@times, N, yDir), 3))));

%% making HHA
% Making the minimum depth to be 100, to prevent large values for disparity!!!
tmppc(:,2) = max(pc(:,2), 1.); 
I(:,1) = 31000./tmppc(:,2); 
I(:,2) = h;
I(:,3) = (angl(:,3)+128.-90.); %Keeping some slack

toc;

end
