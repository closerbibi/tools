function I = getImage(imName, typ)
% function I = getImage(imName, typ)
%  Input:
%   typ   one of ['images', 'depth']

% AUTORIGHTS

  %paths = benchmarkPaths();
  dataDir = fullfile(pwd(), '.');
%   I = imread(fullfile(dataDir, typ, strcat(imName, '.png')));
  I = imread(fullfile(dataDir, typ, strcat(imName)));
end
