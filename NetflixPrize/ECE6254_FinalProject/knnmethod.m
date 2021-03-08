clear;clc;close all;
load('movieData')
load('userData.mat')
load('ratings.mat')
load('mydata.mat')

Xtrain = data(1:64000,:);
Ytrain = labels(1:64000,:);
Xtest = data(64001:80000,:);
Ytest = labels(64001:80000,:);

% data = zeros(80000,22);
% labels = zeros(80000,1);
% [row,col]=find(ratings>0);
% 
% maxAge = 73;
% for i = 1:length(row)
%     age = userData{row(i),2}./maxAge;
%     if userData{row(i),3} == 'M'
%         gender = 1; %male
%     else
%         gender = 0; %female
%     end
%     zip = userData{row(i),3}./99835;
%     data(i,:) = [age,gender,zip,movieData{col(i),end-18:end}];
%     labels(i,1) = ratings(row(i),col(i));
% end

%% 
r = 16000;
[idx,d]=knnsearch(Xtrain,Xtest,'k',10);
fk_test = zeros(1,r);
for ndx = 1:r
    fk_test(ndx) = mean(Ytrain(idx(ndx,:)));
end
sum((fk_test'-Ytest).^2)./16000


%%
% [row,col]=size(ratings);
% for i = 1:row
%     for j = 1:col
%         if ratings(i,j)>0
%             data(end+1,:) = [userData{i,[2,5]},movieData{j,end-18:end}];
%             labels(end+1) = ratings(i,j);
%         end
%     end
% end

% bestK = 10;
% [D, I] = pdist2(Xtrain,Xtest,'euclidean','Smallest',bestK);
% 
% r = 16000;
% fk_test = zeros(1,r);
% for ndx = 1:r
%     fk_test(ndx) = mean(Ytrain(I(:,ndx)));
% end
% sum((fk_test'-Ytest).^2)./16000