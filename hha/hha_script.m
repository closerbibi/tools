
% ar = {'depth0001', C, '.', [], []};
% saveHHA(ar{:});

imList = dir('depth');
C = load('cameraIntrinsics.mat');
C = C.C;
outDir = 'results';

parfor i = 3:length(imList),
%for i = 3:length(imList),
    args{i} = {imList(i).name, C, outDir, [], []}; 
    saveHHA(args{i}{:});
end

