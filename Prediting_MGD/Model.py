import pandas as pd 

file_path = ("/home/master1/Git/Forecasting_MGD/MergedNewFiles (1)(1).csv")

# Loading the CSV file again into a Pandas DataFrame
data = pd.read_csv(file_path)


from sklearn.model_selection import train_test_split

# Selecting the features and target variable
features = ['TIME','MONTH', 'DAY_OF_WEEK', 'DAY_OF_MONTH', 'HOUR', 'TEMPERATURE', 'PRECIPITATION']
target = 'MGD'

# Splitting the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

# Displaying the shape of the training and testing sets
X_train.shape, X_test.shape, y_train.shape, y_test.shape


from sklearn.ensemble import RandomForestRegressor

# Initialize and train the Random Forest model
random_forest_model = RandomForestRegressor(random_state=42)
random_forest_model.fit(X_train, y_train)

# Predict on the testing set
random_forest_predictions = random_forest_model.predict(X_test)

# # Calculate evaluation metrics for the Random Forest model
# random_forest_rmse = np.sqrt(mean_squared_error(y_test, random_forest_predictions))
# random_forest_mae = mean_absolute_error(y_test, random_forest_predictions)
# random_forest_r2 = r2_score(y_test, random_forest_predictions)

# # # Collecting the metrics
# random_forest_metrics = {
#     'RMSE': random_forest_rmse,
#     'MAE': random_forest_mae,
#     'R-squared': random_forest_r2
# }

# random_forest_metrics


from joblib import dump

# Save the Random Forest model
dump(random_forest_model, 'random_forest_model.joblib')
random_forest_predictions = random_forest_model.predict(X_test) 