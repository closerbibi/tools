
for i = 0:706
    disp(i)
    tic;
    img_path = '/media/disk3/data/scannet';
    ID = sprintf('scene%04d_00',i);
    data_path = sprintf('%s/%s',img_path,ID);
    filename = sprintf('%s/%s_vh_clean.ply',data_path, ID);
    outf = sprintf('%s/%s_vh_clean_align.ply',data_path, ID);

    ptCloud = pcread(filename);
%     R = findalignR(ptCloud.Location);
%     showalign(ptCloud.Location, [], R);
%     save(sprintf('rotation_m/rotation%04d.mat',i),'R')
    R = load(sprintf('rotation_m/rotation%04d.mat',i));
    
    pc = (ptCloud.Location)*R.R;
    ptCloud_align = pointCloud(pc);
    ptCloud_align.Color = ptCloud.Color;
    pcwrite(ptCloud_align,outf,'PLYFormat','ascii');
    toc;
end