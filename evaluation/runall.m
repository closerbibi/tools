target_classes = {'chair'}%,'table','sofa','bed','toilet'}
for kkk = 1%:5
    target_class = target_classes{kkk};
    % test
    %fpath = sprintf('../../results/test/comp4-27463_det_test_%s.txt',target_class);
    % train
    fpath = sprintf('../../results/train/comp4-10143_det_train_%s.txt',target_class);
    label_parent_dir = '../../data/DIRE/Annotations';
    gt_file_list = dir(label_parent_dir);
    
    file = 'rcnn_result.mat';
    if ~exist(file)
        [bboxes, confidences, image_ids] = fetch_result(fpath);
        save(file,'bboxes','confidences','image_ids')
    else
        load(file)
    end
    
%     %% trim low confidence
%     target_id = find(confidences>0.2);
%     bboxes = bboxes(target_id);
%     confidences = confidences(target_id);
%     image_ids = image_ids(target_id);
    %%
    unique_image = unique(image_ids);

    all_tp=[]; all_fp=[]; all_box_num = 0; all_gt_box_num =0;
    for i = 1:length(unique_image)
        ids = find(image_ids==unique_image(i));
        label_path = fullfile(label_parent_dir,sprintf('picture_%06d.txt',unique_image(i)));
        [gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections, obj_count] = ...
            evaluate_detections(bboxes(ids,:), confidences(ids,:), image_ids(ids,:), label_path, 0, target_class);
%         all_tp = [all_tp;tp];
%         all_fp = [all_fp;fp];
%         all_box_num = all_box_num + length(tp);
%         all_gt_box_num = all_gt_box_num + obj_count;
        
        con_idx=find(confidences(ids,:)>0.0);
%         if ~isempty(tp)
            all_tp = [all_tp;tp(con_idx)];
            all_fp = [all_fp;fp(con_idx)];
            all_box_num = all_box_num + length(tp(con_idx));
            all_gt_box_num = all_gt_box_num + obj_count;
%         end
    end
    
    [prec, rec, ap, cum_tp, cum_fp]=compute_cu_pr(all_tp,all_fp,all_gt_box_num,confidences);
    draw(prec, rec, ap, cum_fp)
    
    
    disp(target_class)
    precision=sum(all_tp)/all_box_num;
    disp(sprintf('precision: %d/%d = %.01f%% \n',sum(all_tp),all_box_num,precision*100));

    recall=sum(all_tp)/all_gt_box_num;
    disp(sprintf('recall: %d/%d = %.01f%% \n',sum(all_tp),all_gt_box_num,recall*100));
end
