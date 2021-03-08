clear;

% Use real data:
load ('movie_data');
rateMatrix = train;
testMatrix = test;

% Global SVD Test:
lowRank = [1, 3, 5, 7, 10];
for l=1:size(lowRank, 2)
    [U, V] = myRecommender(rateMatrix, lowRank(l));
    
    trainRMSE = norm((U*V' - rateMatrix) .* (rateMatrix > 0), 'fro') / sqrt(nnz(rateMatrix > 0));
    testRMSE = norm((U*V' - testMatrix) .* (testMatrix > 0), 'fro') / sqrt(nnz(testMatrix > 0));
    
    fprintf('SVD-%d\t%.4f\t%.4f\n', lowRank(l), trainRMSE, testRMSE);
end
