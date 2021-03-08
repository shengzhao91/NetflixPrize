clear,clc
load('movieData')
load('userData.mat')

%% convert movie data
year = zeros(1682,1);
for i = 1:1682
    yearStr = movieData{i,3};
    if ~isempty(yearStr)
        year(i) = str2num(yearStr(end-3:end));
    end
end
genre = cell2mat(movieData(:,5:end));
myMovieData = [year, genre];

%% convert user data
age = cell2mat(userData(:,2));
gender = double(strcmp(userData(:,3),'M')); % 1-male, 0-female
zipcode = cell2mat(userData(:,5));
zipcode(isnan(zipcode))=0;   % convert NaN zipcode to 0

% 21 unique occupations
uniqueOccupation = unique(userData(:,4));
occupation = zeros(943,21);
for i = 1:943
    occupation(i,:) = strcmp(userData(i,4),uniqueOccupation)';
end

myUserData = [age,gender,occupation,zipcode];

%% my dataset 80000x(20+24)
load('ratings.mat')

data = zeros(80000,44);
labels = zeros(80000,1);
[row,col]=find(ratings>0);

for i = 1:80000
    data(i,:) = [myUserData(row(i),:),myMovieData(col(i),:)];
    labels(i,1) = ratings(row(i),col(i));
end

save('mydata.mat','myMovieData','myUserData','data','labels');