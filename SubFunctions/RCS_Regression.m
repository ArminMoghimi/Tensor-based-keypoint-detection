function[im_new,R_2,B,M,ptsObj1,ptsScene1]=RCS_Regression(ptsScene1,ptsObj1,imgObj,imgScene)


[m1 m2 m3]=size(imgObj);   

for k=1:m3
for i=1:max(size(ptsScene1))
R_DN(i,k)=imgScene(ptsScene1(i,2),ptsScene1(i,1),k);
end
end


for k=1:m3
for i=1:max(size(ptsObj1))
S_DN(i,k)=imgObj(ptsObj1(i,2),ptsObj1(i,1),k);
end
end

imgObj=reshape(imgObj,m1,m2,m3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Linear Regression based on the RCS
for i=1:m3
    mdl =fitlm((nonzeros(S_DN(:,i))),(nonzeros(R_DN(:,i))),'interactions','RobustOpts','off');
    brob=table2array(mdl.Coefficients(:,1));
R_2(:,i)=mdl.Rsquared.Ordinary;
M(1,i)=brob(2);B(1,i)=brob(1);
im_new(:,:,i)=abs(brob(1)+brob(2)*double(imgObj(:,:,i)));
end