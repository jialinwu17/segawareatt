function hs = showmasks(im, all_masks , CP)
  % Display the specified annotations.
  %
  % USAGE
  %  hs = coco.showAnns( anns )
  %
  % INPUTS
  %  anns       - annotations to display
  %
  % OUTPUTS
  %  hs         - handles to segment graphic objects

  
  figure;
  imshow(im);
  set(gca,'position',[0 0 1 1])
  axis('image');
  set(gca,'XTick',[],'YTick',[]);
  hold on;
  k=0;
  for c=1:20
    masks = (all_masks == c);
    sum(masks(:))
    if c == 9 
        masks(1:200,:) = 0;
    end
    if sum(masks(:))<1000
        masks = 0;
    end
    C=CP(c, :);
    cur_box = [1,1,size(im,2),size(im,1)];
    cur_mask =masks;
    M = masks;
    M = M>0.5;
    k=k+1;
    T=M;
    lw=6;
    T(lw:end,:)=T(lw:end, :) & M(1:end-lw+1, :);
    T(1:end-lw+1, :)=T(1:end-lw+1, :) & M(lw:end, :);
    T(:, 1:end-lw+1)=T(:, 1:end-lw+1) & M(:, lw:end);
    T(:, lw:end)=T(:, lw:end) & M(:, 1:end-lw+1);
    hss = imagesc(cat(3, M*C(1) ,M*C(2), M*C(3)),'Alphadata',M-T*.5);
   
  end
  hs = gcf;
  
end
