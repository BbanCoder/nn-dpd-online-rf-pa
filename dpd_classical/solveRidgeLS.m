function coef = solveRidgeLS(Phi, target, lambda)
%SOLVERIDGELS Ridge least-squares with column scaling and QR solve.
%   Normalizes each regressor column to unit norm, solves the augmented
%   system [Phi; sqrt(lambda)*I] c = [target; 0] via QR (backslash), then
%   rescales the coefficients. Avoids forming Phi'*Phi, whose condition
%   number is the square of cond(Phi) and triggers near-singular warnings
%   for high-order polynomial regressors.
colNorms = sqrt(sum(abs(Phi).^2, 1)).';
colNorms(colNorms < eps) = 1;
PhiN = Phi ./ colNorms.';
K = size(Phi, 2);
A = [PhiN; sqrt(lambda) * eye(K)];
b = [target; zeros(K, 1)];
coef = (A \ b) ./ colNorms;
end
