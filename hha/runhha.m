addpath('npy-matlab')
for num = 1:1449
    disp(num)
    [pc, I] = processDepthImage_frompcdtoHHA(num);
    % writeNPY(a, 'a.npy');
    hhasavepath = '../../data/hhaupsample';
    writeNPY(pc, sprintf('%s/image%04d_pc.npy',hhasavepath,num))
    writeNPY(I, sprintf('%s/image%04d_hha.npy',hhasavepath,num))
end    