clear;clc;close all;

% Use real data:
load ('ratings');

% Global SVD Test:
lowRank = [1, 3, 5, 7, 10];
for l=1:size(lowRank, 2)
    [U, V] = myRecommender(ratings, lowRank(l));
    
    trainRMSE = norm((U*V' - ratings) .* (ratings > 0), 'fro') / sqrt(nnz(ratings > 0))
end
