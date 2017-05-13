addpath('./jsonlab')
addpath('./mBB')
addpath('~/workspace/bin/faster-rcnn/tools/') % to add writePC2XYZfile.m to path (PC, path, output name)
addpath('readData/')
%% bed=157, chair=5, table=19, sofa=83, toilet=124
load('~/workspace/data/nyu_depth_v2_labeled.mat','labels')
% count = 1;

%% setting dictionary for classname 
keyset = [   5,       19,     83,      124,    157];
valset = {'chair', 'table', 'sofa', 'toilet', 'bed'};
mapObj = containers.Map(keyset,valset);

relative_idx = fopen('relative_num.txt','w');
fprintf(relative_idx, 'original_number new_number\n');

for imagenum = 1:1449
    %target=find(labels(:,:,imagenum)==157 | labels(:,:,imagenum)==5 | labels(:,:,imagenum)==19 | labels(:,:,imagenum)==83 | labels(:,:,imagenum)==124);
    %if isempty(target)
    %    continue
    %end
    imagenum
    if ~exist('alignData','dir')
        mkdir('alignData')
    end
    scenefile = sprintf('image%04d',imagenum);
    pathtoscenefile = sprintf('alignData_with_nan_5_classes/%s',scenefile);
    if ~exist(pathtoscenefile,'dir')
        mkdir(pathtoscenefile)
    end    
    thispath =sprintf('/home/closerbibi/workspace/data/NYUdata/NYU%04d',imagenum);
    [data,bb3d] = readframeSUNRGBD(thispath,'');
% % %     keepgoing = 0;
% % %     for keep1 = 1:length(bb3d)
% % %         for keep2 = 1:length(valset)
% % %             if strcmp(valset{keep2},bb3d(keep1).classname)
% % %                 keepgoing = 1;
% % %             end
% % %         end
% % %     end
% % %     if keepgoing == 0
% % %         continue
% % %         %imshow(labels(:,:,9)==5); %show label, for debug
% % %     end
    [~,points3d,~,~]=read3dPoints(data);
%     save(sprintf('%s/pc.mat',pathtoscenefile),'points3d','-v7.3')
    %%% closed
    %writePC2XYZfile(points3d(1:10:end,:), pathtoscenefile, 'pc')
%%%    annofile = sprintf('~/bin/faster-rcnn/data/DIRE/Annotations/picture_%06d.txt',count);
%%%    AnnotationID = fopen(annofile,'w');  %% not align with the point cloud

    cls_idx = 1;
    for i=1:length(bb3d)
        clsn = bb3d(i).classname;
        if strcmp(clsn, 'bed')|| strcmp(clsn, 'chair')|| strcmp(clsn, 'table')|| ...
                strcmp(clsn, 'sofa')|| strcmp(clsn, 'toilet')
            corners = get_corners_of_bb3d(bb3d(i));
            xmin(cls_idx) = min(corners(1:4,1));
            xmax(cls_idx) = max(corners(1:4,1));
            ymin(cls_idx) = min(corners(1:4,2));
            ymax(cls_idx) = max(corners(1:4,2));

            zmin(cls_idx) = min(corners(1:8,3));
            zmax(cls_idx) = max(corners(1:8,3));

            clss{cls_idx} = clsn;
            %% write box to xyz file and log the original file
            %% closed
            %writePC2XYZfile(corners,pathtoscenefile,...
            %    sprintf('box_%03d_%s',i,bb3d(i).classname))
            cls_idx = cls_idx +1;
        end
    end
    if cls_idx == 1
            save(sprintf('%s/annotation_pc.mat',pathtoscenefile),'points3d')
            clear clss
            clear xmin
            clear xmax
            clear ymin
            clear ymax
            clear zmin
            clear zmax
            continue
    end
    save(sprintf('%s/annotation_pc.mat',pathtoscenefile),'points3d','xmin','xmax','ymin','ymax', 'zmin', 'zmax', 'clss')
    clear clss
    clear xmin
    clear xmax
    clear ymin
    clear ymax
    clear zmin
    clear zmax
    %%%    fclose(AnnotationID);
%     fprintf(relative_idx, '%d %d\n',imagenum,count);
%     count = count+1;
end
fclose(relative_idx);
clear all
