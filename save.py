from sklearn.model_selection import RandomizedSearchCV, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X = balanced_df_train.drop(["Id", "Outcome"], axis=1)  # Features
y = balanced_df_train['Outcome']  # Target
# Get the features from test set
X_test = df_test.drop(['Id'], axis = 1)

# Define the hyperparameter grid for RandomizedSearchCV
hyperparam_grid = {
    'pca__n_components': [50, 100, 200, 500],
    'regressor__C': [0.01, 0.1, 1, 10, 100]
}

# Define the outer and inner cross-validation strategies
outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Create RandomizedSearchCV object
random_search = RandomizedSearchCV(estimator=PCA_pipes,
                                   param_distributions=hyperparam_grid,
                                   cv=inner_cv)

# Perform nested cross-validation
nested_test_score = cross_validate(random_search, X, y, cv = outer_cv, 
                            scoring = ('f1_macro', 'roc_auc_ovr'), return_estimator = True)

print(f'Macro F1 score: {nested_test_score["test_f1_macro"].mean()}')
print(f'ROC AUC score OvR averaged over all folds: {nested_test_score["test_roc_auc_ovr"].mean()}')

#---------------------------------------------------------------------------------------------------------------------
# rain a classifier on all the train data using the best hyperparameters and make predictions on the test set
#---------------------------------------------------------------------------------------------------------------------

# First we need to find those hyperparameters values that are optimal

# Get a list of the best parameters dictionaries across all folds
best_params_list = [estimator.best_params_ for estimator in nested_test_score['estimator']]

# Across every fold, calculate the mean of each hyperparameter value
mean_params = {}
for param_name in hyperparam_grid:
    param_values = [params[param_name] for params in best_params_list]
    mean_value = np.mean(param_values)
    mean_params[param_name] = mean_value

for i, estimator in enumerate(nested_test_score['estimator']):
    print(f"Parameters for fold {i+1}:")
    print(estimator.best_params_)

print(f"Average hyperparameters values: {mean_params}")

# Set the optimal pipeline with the optimal hyperparameter values

optimal_pipes = Pipeline([("simpleImputer", SimpleImputer(missing_values=np.nan, strategy='mean')), 
                             ("scaler", StandardScaler()),
                             ("PCA", PCA(n_components = 160)),
                             ("regressor", LogisticRegression(penalty='l2', C = 2.143))])

optimal_pipes.fit(X, y)
y_pred = optimal_pipes.predict(X_test)

output_df = pd.DataFrame()
output_df["Id"] = X_test["Id"].tolist()
output_df["Outcome"] = optimal_pipes.tolist()
output_df.to_csv("Optimal_predictions.csv", index=False)