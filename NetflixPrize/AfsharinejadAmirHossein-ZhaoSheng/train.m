% train table
clear; clc;
load('movieData.mat')
load('ratings.mat')
load('userData.mat');

tic

%convert movie data 
year = zeros(1682,1);
for i = 1:1682 
    yearStr = movieData{i,3}; 
    if ~isempty(yearStr)
        year(i) = str2num(yearStr(end-3:end)); 
    end
end
year0to40 = double(year<=1940);
year41to70= double(year>=1941&year<=1970);
year71to90= double(year>=1971&year<=1990);
year91to00= double(year>=1991&year<=2000);
year = [year0to40,year41to70,year71to90,year91to00];
gen = cell2mat(movieData(:,5:end));
myMovieData = [year, gen];

%convert user data
age = cell2mat(userData(:,2)); 
age1to17 = double(age<=17);
age18to30 = double(age>=18&age<=30);
age31to50 = double(age>=31&age<=50);
age50to80 = double(age>=51);
age = [age1to17,age18to30,age31to50,age50to80];
gender = [double(strcmp(userData(:,3),'M')), double(strcmp(userData(:,3),'F'))];
zipcode = cell2mat(userData(:,5));
zipcode(isnan(zipcode))=0; % convert NaN zipcode to 0

%21 unique occupations
uniqueOccupation = unique(userData(:,4));
occupation = zeros(943,21);
for i = 1:943 
    occupation(i,:) = strcmp(userData(i,4),uniqueOccupation)';
end

myUserData = [age,gender,occupation]; %zipcode

%% Movie-Movie Similarity Matrix
genre= [ ratings; [7.*year,5.*gen]'];
%
movie_sim=zeros(1682,1682);
for i=1:1682
    for j=i+1:1682
        movie_sim(i,j)=(genre(:,i)'*genre(:,j))/norm(genre(:,i),2)/norm(genre(:,j),2);
        movie_sim(j,i) = movie_sim(i,j);
    end
    movie_sim(i,i)=0;
end
%% Movie Prediction
[row,col]=find(ratings==0);
m_prediction=zeros(943,1682);
for o=1:length(row)
o;
[nearest, nearest_ind]=sort(movie_sim(:,col(o)),'descend');

z = 1;
pred_num = 0;
pred_den = 1e-6;
k_cnt = 0;
while z<=1682 & k_cnt<=16
    if ratings(row(o),nearest_ind(z))~=0
        pred_num=pred_num + nearest(z).*ratings(row(o),nearest_ind(z));
        pred_den=pred_den + nearest(z);
        k_cnt = k_cnt + 1;
    end
    z = z+1;
end
if (pred_num == 0)
    m_prediction(row(o),col(o)) = 3;
else
    m_prediction(row(o),col(o)) = pred_num./pred_den;
end
end
% z,k_cnt

%% User-User Similarity Matrix
user= [ ratings, [3.*age,4.*gender,5.*occupation]];

user_sim=zeros(943,943);
for i=1:943
    for j=i+1:943
        user_sim(i,j)=(user(i,:)*user(j,:)')/norm(user(i,:),2)/norm(user(i,:),2);
        user_sim(j,i) = user_sim(i,j);
    end
    user_sim(i,i)=0;
end

%% User Prediction
[row,col]=find(ratings==0);
u_prediction=zeros(943,1682);
for o=1:length(row)
o;
[nearest, nearest_ind]=sort(user_sim(row(o),:),'descend');

z = 1;
pred_num = 0;
pred_den = 1e-6;
k_cnt = 0;
while z<=943 & k_cnt<=16
    if ratings(nearest_ind(z),col(o))~=0
        pred_num=pred_num + nearest(z).*ratings(nearest_ind(z),col(o));
        pred_den=pred_den + nearest(z);
        k_cnt = k_cnt + 1;
    end
    z = z+1;
end
if (pred_num==0)
    u_prediction(row(o),col(o)) = 3;
else
    u_prediction(row(o),col(o)) = pred_num./pred_den;
end
end

toc
%% Error Testing
norm((m_prediction - ratings) .* (ratings > 0), 'fro') / sqrt(nnz(ratings > 0))
norm((u_prediction - ratings) .* (ratings > 0), 'fro') / sqrt(nnz(ratings > 0))

predictedRatings = 0.7*m_prediction+0.3*u_prediction;
predictedRatings = predictedRatings + ratings;
predictedRatings(predictedRatings>5)=5;
predictedRatings(predictedRatings<1)=1;
norm((predictedRatings - ratings) .* (ratings > 0), 'fro') / sqrt(nnz(ratings > 0))