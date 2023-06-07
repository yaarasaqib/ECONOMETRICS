
% HW 1 
% part 2 (a)

clearvars
clc

% Given data
y_data = [16; 5; 10; 15; 13; 22];
x_data = [4; 1; 2; 3; 3; 4];

% Create the X matrix
X = [ones(6, 1), x_data];

% Calculate y'y
yTy = y_data' * y_data;

% Calculate X'X
XTX = X' * X;

% Calculate X'y
XTy = X' * y_data;

% Display the results
disp("y'y:");
disp(yTy);
disp("X'X:");
disp(XTX);
disp("X'y:");
disp(XTy);

%% part b

% Calculate hatbeta using the least squares method
hatbeta = inv(X' * X) * X' * y_data;

% Display the vector of estimated regression coefficients
disp("Estimated regression coefficients (hatbeta):");
disp(hatbeta);

% Interpretation of hat beta_1 and hat beta_2
disp("Interpretation:");
disp("hat beta_1 represents the estimated intercept term in the linear regression model.");
disp("hat beta_2 represents the estimated coefficient for the independent variable x in the linear regression model.");

%% part c

% Calculate the predicted values
Y_hat = X * hatbeta;

% Calculate the residuals
hat_epsilon = y_data - Y_hat;

% Display the vector of residuals
disp("Vector of Residuals (hat epsilon):");
disp(hat_epsilon);

%% part d

% Calculate the sum of squares due to regression (SSR)
Y_mean = mean(y_data);
SSR = sum((Y_hat - Y_mean).^2);

% Display the sum of squares due to regression (SSR)
disp("Sum of Squares due to Regression (SSR):");
disp(SSR);

%% part e & f

% Calculate the sum of squares due to error (SSE)
SSE = sum((y_data - Y_hat).^2);

% Calculate the covariance matrix of hat beta
n = size(X, 1);
k = size(X, 2)
covariance_matrix = (SSE / (n - k)) * inv(X' * X);

% Display the covariance matrix of hat beta
disp("Covariance Matrix of hat beta:");
disp(covariance_matrix);

%% part g

% Extract the diagonal elements for standard errors
standard_errors = sqrt(diag(covariance_matrix));

% Display the standard errors of hat beta_1 and hat beta_2
disp("Standard Errors:");
disp("Standard Error of hat beta_1:");
disp(standard_errors(1));
disp("Standard Error of hat beta_2:");
disp(standard_errors(2));

%% part h

% Calculate the projection matrix (P)
P = X*inv(X'*X)*X' ;

% Calculate the residual generator matrix (M)
n = size(X, 1);
M = eye(n) - P;

% Display the projection matrix (P)
disp("Projection Matrix (P):");
disp(P);

% Display the residual generator matrix (M)
disp("Residual Generator Matrix (M):");
disp(M);	

%% part i  


% Calculate the total sum of squares (SST)
SST = sum((y_data- Y_mean).^2);

% Calculate the standard error of hat sigma
n = size(X, 1);
p = size(X, 2);
se_hat_sigma = sqrt(SSE / (n - p));

% Calculate the coefficient of determination (R^2)
R_squared = 1 - (SSE / SST);

% Display the standard error of hat sigma and R^2
disp("Standard Error of hat sigma:");
disp(se_hat_sigma);
disp("Coefficient of Determination (R^2):");
disp(R_squared);

%% Qsn 5


% Step 1: Load the data from the 'time.xlsx' file
clc
clearvars

% Step 1: Load the data from the 'time.xlsx' file
filepath = 'C:/Users/hp/OneDrive/Desktop/Summer_Term/Econometrics1 (ECO341A)/HW/time.xlsx';
data = readtable(filepath);


% Extract the variables
Y1 = data.y1;
X2 = data.x2;
X3 = data.x3;


% Create the design matrix

designMatrix = [X2, X3];

% Perform the linear regression
lm = fitlm(designMatrix, Y1);

% Extract the regression coefficient, R^2, and standard error of regression
coefficients = lm.Coefficients.Estimate;
R2 = lm.Rsquared.Ordinary;
stdError = lm.RMSE;

% Display the results
disp('variance Cov matrix:');
lm.CoefficientCovariance
disp('Regression Coefficients:');
disp(coefficients');
disp('R^2:');
disp(R2);
disp('Standard Error of Regression:');
disp(stdError);
%% Regress y3 on intercept x2 and x3. part(b)
% Find the regression coefficient R^2 and standard error of regression     

% Extract the variables`
Y3 = data.y3;
X2 = data.x2;
X3 = data.x3;

% Create the design matrix
designMatrix = [X2, X3];

% Perform the linear regression
lm = fitlm(designMatrix, Y3);

% Extract the regression coefficient, R^2, and standard error of regression
coefficients = lm.Coefficients.Estimate;
R2 = lm.Rsquared.Ordinary;
stdError = lm.RMSE;

% Display the results
disp('variance Cov matrix:');
lm.CoefficientCovariance
disp('Regression Coefficients:');
disp(coefficients');
disp('R^2:');
disp(R2);
disp('Standard Error of Regression:');
disp(stdError);

%% part c

year = data.Year;

% Create the design matrix
Ct = 1 - (year >= 1939 & year <= 1945);
Dt = (year >= 1939 & year <= 1945);
designMatrix = [Ct, Dt, X2, X3];

% Perform the linear regression
lm = fitlm(designMatrix, Y1, 'Intercept', false);

% Extract the regression coefficients, standard errors, R^2, and standard error of regression
coefficients = lm.Coefficients.Estimate;
standardErrors = lm.Coefficients.SE;
R2 = lm.Rsquared.Ordinary;
stdError = lm.RMSE;

% Estimate the variance-covariance matrix
varCovMatrix = lm.CoefficientCovariance;

% Display the results
disp('Regression Coefficients:');
disp(coefficients');
disp('Standard Errors:');
disp(standardErrors');
disp('R^2:');
disp(R2);
disp('Standard Error of Regression:');
disp(stdError);
disp('Variance-Covariance Matrix:');
disp(varCovMatrix);

%% part d


% Extract the variables
year = data.Year;

% Create the design matrix
Ct = 1 - (year >= 1939 & year <= 1945);
Dt = (year >= 1939 & year <= 1945);
designMatrix = [Ct, Dt, data.x2, data.x3];

% Perform the linear regression
lm = fitlm(designMatrix, Y1, 'Intercept', false);

% Extract the estimated coefficients and variance-covariance matrix
coefficients = lm.Coefficients.Estimate;
varCovMatrix = lm.CoefficientCovariance;

% Calculate hat delta
hat_delta = coefficients(2) - coefficients(1);

% Calculate the variance of hat delta
var_hat_delta = varCovMatrix(2,2) + varCovMatrix(1,1) - 2*varCovMatrix(1,2);

% Display the results
disp('Estimated Coefficients:');
disp(coefficients');
disp('Variance of hat delta:');
disp(var_hat_delta);

%% part e
% Create the design matrix
designMatrix = [Dt, X2, X3];

% Perform the linear regression
lm = fitlm(designMatrix, Y1);

% Extract the regression coefficients, standard errors, R^2, and standard error of regression
coefficients = lm.Coefficients.Estimate;
standardErrors = lm.Coefficients.SE;
R2 = lm.Rsquared.Ordinary;
stdError = lm.RMSE;

% Estimate the variance-covariance matrix
varCovMatrix = lm.CoefficientCovariance;

% Display the results
disp('Regression Coefficients:');
disp(coefficients');
disp('Standard Errors:');
disp(standardErrors');
disp('R^2:');
disp(R2);
disp('Standard Error of Regression:');
disp(stdError);
disp('Variance-Covariance Matrix:');
disp(varCovMatrix);

%% part f



Y2= data.y2 ;
% Create the variables
X4 = X3.*Dt;
X5 = X3 .* Ct;

% Create the design matrix
designMatrix = [Dt, X2, X3, X4];

% Perform the linear regression
lm = fitlm(designMatrix, Y2);

% Extract the regression coefficients, standard errors, R^2, and standard error of regression
coefficients = lm.Coefficients.Estimate;
standardErrors = lm.Coefficients.SE;
R2 = lm.Rsquared.Ordinary;
stdError = lm.RMSE;

% Display the results
disp('Regression Coefficients:');
disp(coefficients');
disp('Standard Errors:');
disp(standardErrors');
disp('R^2:');
disp(R2);
disp('Standard Error of Regression:');
disp(stdError);

%% part g

Ct = 1 - (year >= 1939 & year <= 1945);
Dt = (year >= 1939 & year <= 1945);
X2 = data.x2 ;
X4 = X3.*Dt;
X5 = X3 .* Ct;



% Create the design matrix
Du = [Ct, Dt, X2, X4, X5]; % Unrestricted
Dr = [Ct, Dt, X2]; % restricted
% Perform the linear regression

n  = length(X2) ;
k =  size(Du,2)  ;
m =  size(Dr,2)  ;

lmU = fitlm(Du, Y2,'Intercept',false);
lmR = fitlm(Dr, Y2,'Intercept',false);

Ru = lmU.Rsquared.Ordinary
Rr = lmR.Rsquared.Ordinary


Fc = ((Ru-Rr)/(k-m))/((1-Ru)/(n-k))

% Significance level
alpha = 0.05; % 95% confidence interval

% Calculate the critical F value
f_critical = finv(1 - alpha, m, n);



% Extract the regression coefficients, standard errors, R^2, and standard error of regression
coefficients = lm.Coefficients.Estimate;
standardErrors = lm.Coefficients.SE;
R2 = lm.Rsquared.Ordinary;
stdError = lm.RMSE;

% Display the results
disp('Regression Coefficients:');
disp(coefficients');
disp('Standard Errors:');
disp(standardErrors');
disp('R^2:');
disp(R2);
disp('Standard Error of Regression:');
disp(stdError);

% Display the hypothesis test results
disp('Hypothesis Test Results:');
disp('Null Hypothesis: beta4 = 0');
disp(['test_stat: ', num2str(Fc)]);
disp(['f_critical: ', num2str(f_critical)]);
disp('Reject Null Hypothesis (alpha = 0.05) if test_stat > f_critical');


%% part h
X6 = X2 .* Dt;

% Create the design matrix
designMatrix = [Dt, X2, X3, X4, X6];

% Perform the linear regression
lm = fitlm(designMatrix, Y3);

% Extract the regression coefficients, standard errors, R^2, and standard
% error of regression
coefficients = lm.Coefficients.Estimate;
standardErrors = lm.Coefficients.SE;
R2 = lm.Rsquared.Ordinary;
stdError = lm.RMSE;

% Display the results
disp('Regression Coefficients:');
disp(coefficients');
disp('Standard Errors:');
disp(standardErrors');
disp('R^2:');
disp(R2);
disp('Standard Error of Regression:');
disp(stdError);


