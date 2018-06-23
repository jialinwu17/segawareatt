%addpath('/home/adam/final_code/deeplab-public/matlab/my_script');
SetupEnv;

if strcmp(feature_type,'crf') 

disp('ok! generating preds from crfs');
post_folder = sprintf('post_densecrf_W%d_XStd%d_RStd%d_PosW%d_PosXStd%d', bi_w, bi_x_std, bi_r_std, pos_w, pos_x_std);
map_folder = fullfile('/data/jialinwu/segaware/scripts/segaware/features/', model_name, testset, feature_type, post_folder); 
%map_folder = fullfile('/data/jialinwu/segaware/scripts/segaware/features/', model_name, testset, feature_type); 
save_root_folder = fullfile('/workspace/jialinwu/segaware/scripts/segaware/results/', model_name, testset, feature_type);

map_dir = dir(fullfile(map_folder, '*.bin'));

fprintf(1,' saving to %s\n', save_root_folder);

seg_res_dir = [save_root_folder '/results/VOC2012/'];

%save_result_folder = fullfile(seg_res_dir, 'Segmentation', [id '_' testset '_cls']);
save_result_folder = fullfile(seg_res_dir, 'Segmentation', [id '_' testset(5:end) '_cls']);

if ~exist(save_result_folder, 'dir')
    mkdir(save_result_folder);
end

for i = 1 : numel(map_dir)
    fprintf(1, 'generating preds from crf for image %d (%d)...\n', i, numel(map_dir));
    map = LoadBinFile(fullfile(map_folder, map_dir(i).name), 'int16');

    img_fn = map_dir(i).name(1:end-4);
    imwrite(uint8(map), colormap, fullfile(save_result_folder, [img_fn, '.png']));
end

else %if (strcmp(feature_type,'fc8') || strcmp(feature_type,'fc8_safe'))


disp('ok! generating preds');
mat_folder  = fullfile('../../features/', model_name, testset, feature_type);

img_folder  = '/workspace/jialinwu/data/VOCdevkit0712/VOC2012/JPEGImages';
post_folder = 'none';
save_root_folder = fullfile('../../results/', model_name, testset, feature_type, post_folder);

fprintf(1,' saving to %s\n', save_root_folder);

seg_res_dir = [save_root_folder '/results/VOC2012/'];

%if (strcmp(testset,'voc_val'))
%save_result_folder = fullfile(seg_res_dir, 'Segmentation', [id '_' 'val' '_cls']);
%elseif (strcmp(testset,'voc_test'))
%save_result_folder = fullfile(seg_res_dir, 'Segmentation', [id '_' 'test' '_cls']);
%else
%save_result_folder = fullfile(seg_res_dir, 'Segmentation', [id '_' testset '_cls']);
%end
save_result_folder = fullfile(seg_res_dir, 'Segmentation', [id '_' testset(5:end) '_cls']);

if ~exist(save_result_folder, 'dir')
    mkdir(save_result_folder);
end

mat_dir = dir(fullfile(mat_folder, '*.mat'));
fprintf(1, 'found %d mats in %s\n', numel(mat_dir), mat_folder);
%parpool('local',8);
%parfor i = 1 : numel(mat_dir)
overwriteResults = 1;
for i = 1 : numel(mat_dir)
    fprintf(1, 'generating preds for image %d (%d)...\n', i, numel(mat_dir));

    img_fn = mat_dir(i).name(1:end-4);
    img_fn = strrep(img_fn, '_blob_0', '');
    img_fn = strrep(img_fn, '_blob_1', '');
if overwriteResults || (exist(fullfile(save_result_folder, [img_fn, '.png']), 'file') ~= 2)
    data = load(fullfile(mat_folder, mat_dir(i).name));
    data = data.data;
    data = permute(data, [2 1 3]);
    % Transform data to probability
    data = exp(data);
    data = bsxfun(@rdivide, data, sum(data, 3));
    
    data_size = size(data,1);
    
    fac = data_size / 513;
    
    img = imread(fullfile(img_folder, [img_fn, '.jpg']));
    %img = imresize(img_ori, fac);
    img_row = size(img, 1);
    img_col = size(img, 2);
    %img_row_ori = size(img_ori, 1);
    %img_col_ori = size(img_ori, 2);
    size_limit = 481;
    data = data(1:min(img_row,data_size) , 1:min(img_col, data_size), :);
    actual_data_size = size(data);
    if img_row >= size_limit || img_col >= size_limit 
        new_data = zeros(img_row,img_col, 21);
        new_data(:,:,1) = 1;
        new_data( ceil((img_row - actual_data_size(1))/2)+1: ...
            actual_data_size(1) + ceil((img_row - actual_data_size(1))/2),...
            ceil((img_col - actual_data_size(2))/2)+1: ...
            actual_data_size(2) + ceil((img_col - actual_data_size(2))/2),:) = data;
        data = new_data;
        
    end

    
    
    %new_data = zeros(img_row_ori, img_col_ori, 3);
    %for kk = 1:21
    %    ori_data =  uint8(squeeze(data(:,:,kk))*255);
    %    resized_data = imresize(ori_data, [img_row_ori, img_col_ori]);
    %    %new_data = single(medfilt2(ori_data))/255;
    %    new_data(:,:,kk) = resized_data;
    %    %new_data = single(imdilate(ori_data, offsetstrel('ball',3,3)))/255;
    %    %data(10 : (end - 10), 10 : (end - 10),kk) =  new_data(10 : (end - 10), 10 : (end - 10)); 
    %end
    
    
    
    [~,classes] = max(data,[],3);
    classes = classes-1;
    %imagesc(classes)
    classes = uint8(classes);
    %classes = imresize(classes, colormap, 1/fac);
    imwrite(classes,colormap, fullfile(save_result_folder, [img_fn, '.png']));
end
end

end
