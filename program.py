# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Loading datasets
translatoare_df = pd.read_csv('translatoareteleviziune.csv')
statii_df = pd.read_csv('statiiteleviziune.csv')
inflatie_df = pd.read_csv('ratainflatiei.csv')
abonamente_df = pd.read_csv('abonamenteteleviziune.csv')

# Cleaning column names
translatoare_df.columns = translatoare_df.columns.str.strip()
translatoare_df = translatoare_df.iloc[:, :2]
inflatie_df.columns = inflatie_df.columns.str.strip()
abonamente_df.columns = abonamente_df.columns.str.strip()
statii_df.columns = statii_df.columns.str.strip()

# Merging datasets based on 'Anul' column
merged_df = pd.merge(translatoare_df, statii_df, on='Anul', suffixes=('_translatoare', '_statii'))
merged_df = pd.merge(merged_df, inflatie_df, on='Anul')
merged_df = pd.merge(merged_df, abonamente_df, on='Anul', suffixes=('_inflatie', '_abonamente'))

# Sorting DataFrame by 'Anul' column to maintain temporal order
merged_df = merged_df.sort_values(by='Anul')

# Splitting the data into training and testing sets
train_index = int(0.8 * len(merged_df))
X_train = merged_df.iloc[:train_index].drop('Rata valoare', axis=1)
y_train = merged_df.iloc[:train_index]['Rata valoare']
X_test = merged_df.iloc[train_index:].drop('Rata valoare', axis=1)
y_test = merged_df.iloc[train_index:]['Rata valoare']

# Training a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)

# Training a Random Forest Regressor model with hyperparameter optimization
rf_model = RandomForestRegressor()
param_grid = {
    'n_estimators': [100, 200, 300],  
    'max_depth': [10, 20, 30], 
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
print("Best Hyperparameters:", grid_search.best_params_)
y_pred_rf = grid_search.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
print("Mean Squared Error (Random Forest):", mse_rf)
print("Mean Absolute Error (Random Forest):", mae_rf)
