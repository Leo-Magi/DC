import time
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

data = pd.read_csv('SH_Zht_PCA.csv',header=None)

X = data.drop(columns=[data.columns[0], data.columns[1]])
y = data[data.columns[1]]
print(X)
print(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=429431)

param_grid = {
    'n_estimators': [ 130,140,150,160],
    'max_depth': [3,5,7,10,15,20],
    'min_samples_split': [2,3,4],
    'min_samples_leaf': [1],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True]
}

grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=429431),
                           param_grid=param_grid,
                           cv=5,
                           scoring='r2',
                           verbose=1,
                           n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
y_pred_train = best_rf.predict(X_train)
y_pred_test = best_rf.predict(X_test)
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

plt.figure(figsize=(10, 7))
plt.scatter(y_train, y_pred_train, color='red', label='Training set', alpha=0.5)
plt.scatter(y_test, y_pred_test, color='blue', label='Test set', alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual',fontsize=17, fontfamily='Times New Roman', weight='bold')
plt.ylabel('Predicted',fontsize=17, fontfamily='Times New Roman',weight='bold')
title_font = {'fontsize': 12, 'fontfamily': 'Times New Roman','weight': 'bold'}

plt.legend(prop={'size': 14, 'family': 'Times New Roman'})
plt.xticks(fontsize=14, fontfamily='Times New Roman', weight='bold')
plt.yticks(fontsize=14, fontfamily='Times New Roman', weight='bold')
plt.savefig('Aiye_total phenol.tif')
plt.show()

print("Best parameters:", grid_search.best_params_)

mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

rmse_train = sqrt(mse_train)
rmse_test = sqrt(mse_test)

mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print("Training MSE:", mse_train)
print("Training RMSE:", rmse_train)
print("Training MAE:", mae_train)
print("Training R²:", r2_train)

print("Test MSE:", mse_test)
print("Test RMSE:", rmse_test)
print("Test MAE:", mae_test)
print("Test R²:", r2_test)