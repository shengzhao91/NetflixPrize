function [ U, V ] = myRecommender( rateMatrix, lowRank )
    % Parameters
    maxIter = 150; % Choose your own.
    learningRate = 0.0005; % Choose your own.
    regularizer = 0.01; % Choose your own.
    
    % Random initialization:
    [n1, n2] = size(rateMatrix);
    U = rand(n1, lowRank) / lowRank;
    V = rand(n2, lowRank) / lowRank;

    % Gradient Descent:
    iter = 1;
    trainRMSE_old = norm((U*V' - rateMatrix) .* (rateMatrix > 0), 'fro') / sqrt(nnz(rateMatrix > 0));;
    while iter <= maxIter
        errMatrix = ( rateMatrix - U*V') .* (rateMatrix > 0);                
        U = U - learningRate*(-2*errMatrix*V + regularizer*2*U); % update U
        V = V - learningRate*(-2*errMatrix.'*U + regularizer*2*V); % update V
        
        trainRMSE_new = norm((U*V' - rateMatrix) .* (rateMatrix > 0), 'fro') / sqrt(nnz(rateMatrix > 0));
        if trainRMSE_old - trainRMSE_new < 1e-6 
            % fprintf('convergence at iteration %d ...\n', iter);
            break;  % convergence            
        end
        trainRMSE_old = trainRMSE_new;
        iter = iter + 1;
    end
    
end