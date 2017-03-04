function writePC2XYZfile(PC,path,name)
    file = fullfile(path,sprintf('%s.xyz',name)); 
    xyzfileID = fopen(file,'w');
    fprintf(xyzfileID,'%d\n',length(PC));
    for j = 1:length(PC)
        fprintf(xyzfileID,'%f %f %f\n',PC(j,:));
    end
    fclose(xyzfileID);