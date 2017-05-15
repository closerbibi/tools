function draw(prec, rec, ap, cum_fp, fname, IOU, target_class)
% plot precision/recall
    figure
    plot(rec*100,prec*100,'-');
    axis([0 100 0 100])
    grid;
    xlabel 'recall(%)'
    ylabel 'precision(%)'
    aa= strsplit(fname,'_');
    title(sprintf('%s %s %s,IOU= %.2f,AP = %.3f %%',aa{1:3},IOU, ap*100));
% % %     if target_class == 'chair'
% % %         set(12, 'Color', [.988, .988, .988])
% % %     end
% % %     
% % %     pause(0.1) %let's ui rendering catch up
% % %     average_precision_image = frame2im(getframe(12));
% % %     % getframe() is unreliable. Depending on the rendering settings, it will
% % %     % grab foreground windows instead of the figure in question. It could also
% % %     % return a partial image.
% % %     imwrite(average_precision_image, 'visualizations/average_precision.png')
    
%     figure
%     plot(cum_fp,rec,'-')
%     axis([0 300 0 1])
%     grid;
%     xlabel 'False positives'
%     ylabel 'Number of correct detections (recall)'
%     title(sprintf('%s %s %s',aa{1:3}));
