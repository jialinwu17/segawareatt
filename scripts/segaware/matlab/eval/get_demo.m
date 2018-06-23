dir_att = '../../results/res_att/voc_test/mycrf/none/results/VOC2012/Segmentation/comp6_test_cls/';
dir_ori = '../../results/res0/voc_test/mycrf/none/results/VOC2012/Segmentation/comp6_test_cls/';
dst = '../../results/resresatt/voc_test/mycrf/none/results/VOC2012/Segmentation/comp6_test_cls/';
tmp_att = dir( dir_att);
tmp_ori = dir( dir_ori);
c= 0;
b = zeros(-3+size(tmp_att, 1) + 1, 1);
load('pascal_seg_colormap.mat')
name = cell(1,1);
%name{1} = '2008_000030.';
name{1} = '2008_000068.';
%name{3} = '2008_001953.';
%name{4} = '2008_002905.';
%name{5} = '2008_003810.';
%name{6} = '2008_004104.';
%name{7} = '2008_005661.';

for i = 1:size(name, 1)
    im_ori = [dir_ori, name{i},'png'];
    img_ori = imread(im_ori);
    im_att = [dir_att, name{i},'png'];
    img_att = imread(im_att);
    img = imread(['/workspace/jialinwu/data/VOCdevkit0712/VOC2012/JPEGImages/', name{i}, 'jpg']);
    h1 = showmasks(img, img_ori, colormap(2:21,:));
    saveas(h1, ['demo/',name{i}(1:end-1), 'ori.png']);
    h2 = showmasks(img, img_att, colormap(2:21,:));
    saveas(h2, ['demo/',name{i}(1:end-1), 'att.png']);

end
