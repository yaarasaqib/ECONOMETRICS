	
%% Homework 3   | Codes
    %% Qsn 5 a
    data = xlsread('C:\Users\hp\OneDrive\Desktop\Summer_Term\Econometrics1 (ECO341A)\MATLAB_Practice\Advertising.xlsx');
	x = data(:, 1);  % Input variable (number of hours of staff time)
	y = data(:, 2);  % Output variable (amount of billings)
	
	model = fitlm(x, y);
	
	beta = model.Coefficients.Estimate;  % OLS estimates of beta
	se_beta = model.Coefficients.SE;     % Standard errors of beta
	
	residuals = model.Residuals.Raw;
	
	disp("OLS Estimates of Beta:");
	disp(beta);
	disp("Standard Errors of Beta:");
	disp(se_beta);
	disp("Residuals:");
   disp(residuals);
   %% Part b
   	model = fitlm(x, y);
	
	residuals = model.Residuals.Raw;
	
	index = 1:length(residuals);  % Observation index
	
	plot(index, residuals, 'o');
	xlabel('Observation Index');
	ylabel('Residuals');
	title('Residuals vs Observation Index');
    %% Part c
    	% Assuming you already have residuals from previous steps
	test_statistic = sum(diff(residuals).^2) / sum(residuals.^2);
	
	% Get the number of observations and regressors (including the intercept)
	n = length(residuals);
	k = 2;
	
	% Calculate the degrees of freedom for the test
	df = n - k;
	
	% Compute the critical values for alpha = 0.05
	lower_critical_value = 1.535;  % For positive autocorrelation
	upper_critical_value = 2.465;  % For negative autocorrelation
	
	% Calculate the p-value for the test
	p_value = 2 * (1 - normcdf(abs(test_statistic), 0, 2));
	
	% Display the test statistic and p-value
	disp("Durbin-Watson Test Statistic:");
	disp(test_statistic);
	disp("P-Value:");
	disp(p_value);
	
	% Make the decision and conclusion
	alpha = 0.05;
	if p_value < alpha
	    disp("Conclusion: There is evidence of positive autocorrelation.");
	else
	    disp("Conclusion: There is no significant evidence of positive autocorrelation.");
	end

%% 6 a

	% Read the Excel file
	filename = 'C:\Users\hp\OneDrive\Desktop\Summer_Term\Econometrics1 (ECO341A)\MATLAB_Practice\TransportChoiceDataset.xlsx';
	sheet = 1; % Assuming the data is on the first sheet
	data = xlsread(filename, sheet);
	
	% Extract variables
	depend = data(:, 6);  % Column 6
	dcost = data(:, 1);   % Column 1
	cars = data(:, 2);    % Column 2
	dovtt = data(:, 3);   % Column 3
	divtt = data(:, 4);   % Column 4
	
	% Descriptive summary for continuous variables
	mean_dcost = mean(dcost);
	std_dcost = std(dcost);
	mean_cars = mean(cars);
	std_cars = std(cars);
	mean_dovtt = mean(dovtt);
	std_dovtt = std(dovtt);
	mean_divtt = mean(divtt);
	std_divtt = std(divtt);
	
	fprintf('Descriptive Summary for Continuous Variables:\n');
	fprintf('---------------------------------------------\n');
	fprintf('Variable   Mean     Standard Deviation\n');
	fprintf('------------------------------------\n');
	fprintf('dcost      %.2f     %.2f\n', mean_dcost, std_dcost);
	fprintf('cars       %.2f     %.2f\n', mean_cars, std_cars);
	fprintf('dovtt      %.2f     %.2f\n', mean_dovtt, std_dovtt);
	fprintf('divtt      %.2f     %.2f\n', mean_divtt, std_divtt);
	fprintf('------------------------------------\n');
	
	% Descriptive summary for discrete variable
	count_depend = histcounts(depend, [0 0.5 1.5]);
	percentage_depend = count_depend / numel(depend) * 100;
	
	fprintf('Descriptive Summary for Discrete Variable:\n');
	fprintf('-----------------------------------------\n');
	fprintf('Variable    Count    Percentage\n');
	fprintf('--------------------------------\n');
	fprintf('0           %d       %.2f%%\n', count_depend(1), percentage_depend(1));
	fprintf('1           %d       %.2f%%\n', count_depend(2), percentage_depend(2));
	fprintf('--------------------------------\n');

%% Qsn  6 b
	
    % For Probit Model
    filename = 'C:\Users\hp\OneDrive\Desktop\Summer_Term\Econometrics1 (ECO341A)\MATLAB_Practice\TransportChoiceDataset.xlsx';
	sheet = 1; % Assuming the data is on the first sheet
	
	data = xlsread(filename, sheet);
	
	% Extract variables from the data
	depend = data(:, 6); % Dependent variable (1 if automobile is chosen, 0 if transit is chosen)
	dcost = data(:, 1); % Transit fare minus automobile travel cost ($)
	cars = data(:, 2); % Number of cars owned by the traveler's household
	dovtt = data(:, 3); % Transit out-of-vehicle travel time minus automobile out-of-vehicle travel time (minutes)
	divtt = data(:, 4); % Transit in-vehicle travel time minus automobile in-vehicle travel time (minutes)
	
	% Create a table for regression
	tbl = table(depend, dcost, cars, dovtt, divtt, 'VariableNames', {'depend', 'dcost', 'cars', 'dovtt', 'divtt'});
	
	% Probit regression
	probitModel = fitglm(tbl, 'depend ~ dcost + cars + dovtt + divtt', 'Distribution', 'binomial', 'Link', 'probit');
	
	% Extract coefficients and standard errors
	coefficients = table2array(probitModel.Coefficients(:, 1:2));
	coefficients(:, 2) = coefficients(:, 2) * sqrt(probitModel.Dispersion);
	
	% Create a table for coefficients and standard errors
	variableNames = {'Intercept', 'dcost', 'cars', 'dovtt', 'divtt'};
	tableCoefficients = table(coefficients(:, 1), coefficients(:, 2), 'VariableNames', {'Coefficients', 'Standard_Errors'}, 'RowNames', variableNames);
	
	% Display the table
	disp(tableCoefficients);

    % For Logit Model
    	filename = 'C:\Users\hp\OneDrive\Desktop\Summer_Term\Econometrics1 (ECO341A)\MATLAB_Practice\TransportChoiceDataset.xlsx';
	sheet = 1; % Assuming the data is on the first sheet
	
	data = xlsread(filename, sheet);
	
	% Extract variables from the data
	depend = data(:, 6); % Dependent variable (1 if automobile is chosen, 0 if transit is chosen)
	dcost = data(:, 1); % Transit fare minus automobile travel cost ($)
	cars = data(:, 2); % Number of cars owned by the traveler's household
	dovtt = data(:, 3); % Transit out-of-vehicle travel time minus automobile out-of-vehicle travel time (minutes)
	divtt = data(:, 4); % Transit in-vehicle travel time minus automobile in-vehicle travel time (minutes)
	
	% Create a table for regression
	tbl = table(depend, dcost, cars, dovtt, divtt, 'VariableNames', {'depend', 'dcost', 'cars', 'dovtt', 'divtt'});
	
	% Logit regression
	logitModel = fitglm(tbl, 'depend ~ dcost + cars + dovtt + divtt', 'Distribution', 'binomial', 'Link', 'logit');
	
	% Extract coefficients and standard errors
	coefficients = table2array(logitModel.Coefficients(:, 1:2));
	coefficients(:, 2) = coefficients(:, 2) * sqrt(logitModel.Dispersion);
	
	% Create a table for coefficients and standard errors
	variableNames = {'Intercept', 'dcost', 'cars', 'dovtt', 'divtt'};
	tableCoefficients = table(coefficients(:, 1), coefficients(:, 2), 'VariableNames', {'Coefficients', 'Standard_Errors'}, 'RowNames', variableNames);
	
	% Display the table
	disp(tableCoefficients);

    %% Qsn 6c
    
% Calculate predicted probabilities
X = table2array(tbl(:, 2:end));
predictedProbabilities = normcdf(probitModel.Fitted.LinearPredictor);

% Calculate log-likelihood
logLikelihood = sum(tbl.depend .* log(predictedProbabilities) + (1 - tbl.depend) .* log(1 - predictedProbabilities));

% Get the number of parameters in the model
numParameters = numel(probitModel.CoefficientNames) - 1;

% Calculate AIC and BIC
AIC = -2 * logLikelihood + 2 * numParameters;
BIC = -2 * logLikelihood + numParameters * log(size(tbl, 1));

% Calculate Hit-rate
predictedLabels = double(predictedProbabilities >= 0.5);
hitRate = sum(predictedLabels == tbl.depend) / numel(tbl.depend) * 100;

% Display the results
fprintf('Probit Model Results:\n');
fprintf('======================\n');
fprintf('Sum of Log-Likelihood: %.3f\n', logLikelihood);
fprintf('AIC: %.3f\n', AIC);
fprintf('BIC: %.3f\n', BIC);
fprintf('Hit-rate: %.2f%%\n', hitRate);

% For Logit Model 
% Calculate predicted probabilities
X = table2array(tbl(:, 2:end));
predictedProbabilities = predict(logitModel, tbl);

% Calculate log-likelihood
logLikelihood = sum(tbl.depend .* log(predictedProbabilities) + (1 - tbl.depend) .* log(1 - predictedProbabilities));

% Get the number of parameters in the model
numParameters = numel(logitModel.CoefficientNames) - 1;

% Calculate AIC and BIC
AIC = -2 * logLikelihood + 2 * numParameters;
BIC = -2 * logLikelihood + numParameters * log(size(tbl, 1));

% Calculate Hit-rate
predictedLabels = double(predictedProbabilities >= 0.5);
hitRate = sum(predictedLabels == tbl.depend) / numel(tbl.depend) * 100;

% Display the results
fprintf('Logit Model Results:\n');
fprintf('=====================\n');
fprintf('Sum of Log-Likelihood: %.3f\n', logLikelihood);
fprintf('AIC: %.3f\n', AIC);
fprintf('BIC: %.3f\n', BIC);
fprintf('Hit-rate: %.2f%%\n', hitRate);


