function  [fhand,test]=appendimages(image1, image2,correspond1,correspond2)
%  The code wriiten by Armin Moghimi (2022)
% Reference:
% Armin Moghimi, Turgay Celik, Ali Mohammadzadeh (2022)
% "Tensor-based keypoint detection and switching regression model for
% relative radiometric normalization of bitemporal multispectral images,"
% International Journal of Remote Sensing, 43:11, 3927-3956,
% DOI: 10.1080/01431161.2022.2102951
rows1 = size(image1,1);
rows2 = size(image2,1);

col1=size(image1,2);
col2=size(image2,2);

if (rows1 < rows2)
     image1(rows1+1:rows2,1:col1,:) = 0;
elseif(rows1 >rows2)
     image2(rows2+1:rows1,1:col2,:) = 0;
end

temp1=size(image1,3);
temp2=size(image2,3);
if(temp1==1 && temp2==3)
    image2=rgb2gray(image2);
elseif(temp1==3 && temp2==1)
    image1=rgb2gray(image1);
end
im3 = [image1 image2];

colormap = {'b','r','m','y','g','c'};
figure,imshow(im3,[])
title(['Left is the image $\textbf{MS}_S$ --- the number of pairs ',num2str(size(correspond1,1)),' --- Right is the image $\textbf{MS}_R$'], 'fontsize',19, 'Interpreter', 'latex');

% title('Good Matches & Object detection')
hold on;
cols1 = size(image1,2);
for i = 1: size(correspond1,1)
    num=1;
    if(num==1)%red
              plot([correspond1(i,1) correspond2(i,1)+cols1], ...
             [correspond1(i,2) correspond2(i,2)],colormap{mod(i,6)+1},'Marker','o','Markersize',2.5,'LineWidth',1.5);
    elseif(num==2)%green
        line([correspond1(i,1) correspond2(i,1)+cols1], ...
             [correspond1(i,2) correspond2(i,2)], 'Color', 'g','LineWidth',1.5); 
    elseif(num==3)%blue
        line([correspond1(i,1) correspond2(i,1)+cols1], ...
             [correspond1(i,2) correspond2(i,2)], 'Color', 'b','LineWidth',1.5); 
    end
end
end






