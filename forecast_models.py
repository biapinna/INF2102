# Importações
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet, BayesianRidge, LinearRegression, Lasso
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Função para calcular o MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Função para ajustar o modelo ARIMA ou SARIMA
def fit_arima_model(x, y, seasonal=True):
    autoarima_results = pm.auto_arima(y, seasonal=seasonal)
    order, seasonal_order = autoarima_results.get_params()['order'], autoarima_results.get_params()['seasonal_order']
    
    model = SARIMAX(y, order=order, seasonal_order=seasonal_order)
    results = model.fit(disp=False)
    y_pred = results.predict(start=0, end=len(y) - 1)
    mape = mean_absolute_percentage_error(y, y_pred)
    
    return mape, order, seasonal_order

# Função para realizar a validação cruzada e calcular métricas
def cross_val_metrics(model, x, y, encoded_break_variable=None, arima=False, seasonal=True):
    data = pd.DataFrame({'target': y, 'break_variable': encoded_break_variable})
    data['y_'] = np.nan

    sampling = StratifiedKFold() if encoded_break_variable is not None else None
    for train, test in sampling.split(x, encoded_break_variable) if sampling else [(range(len(x)), range(len(x)))]:
        x_train, y_train = x.iloc[train], y.iloc[train]
        x_test = x.iloc[test]
        
        if arima:
            mape, order, seasonal_order = fit_arima_model(pd.concat([x_train, x_test]), pd.concat([y_train, y.iloc[test]]), seasonal=seasonal)
        else:
            model.fit(x_train, y_train)
            y_ = model.predict(x_test)
            data['y_'].iloc[test] = y_

    if not arima:
        mae = mean_absolute_error(data['target'], data['y_'])
        mse = mean_squared_error(data['target'], data['y_'])
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(data['target'], data['y_'])
    else:
        mae, mse, rmse = None, None, None

    return mae, mse, rmse, mape

# Função para gerar o pipeline do modelo
def pipeline_generator(model, numeric_features, categorical_features, is_arima=False):
    if is_arima:
        pipeline = TransformedTargetRegressor(regressor=model, transformer=None)
    else:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")), 
                ("scaler", StandardScaler())
            ]
        )
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")), 
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )
        pipeline = Pipeline(
            steps=[
                ("preprocessing", preprocessor),
                ("model", model)
            ]
        )
        pipeline = TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())

    return pipeline

# Função para treinar modelos, calcular métricas e retornar resultados
def train_and_evaluate_models(models, data, numeric_features, categorical_features, target, break_variable=None):
    metrics_df = pd.DataFrame(columns=['Model', 'MAE', 'MSE', 'RMSE', 'MAPE'])

    for key, model_class in models.items():
        model = model_class()

        pipeline = pipeline_generator(model, numeric_features, categorical_features)
        encoded_break_variable = LabelEncoder().fit_transform(data[break_variable]) if break_variable else None
        
        mae, mse, rmse, mape = cross_val_metrics(pipeline, data[numeric_features + categorical_features], data[target], encoded_break_variable)
        
        metrics_df = metrics_df.append({
            'Model': key,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape
        }, ignore_index=True)
    
    return metrics_df

# Função para plotar o gráfico comparativo do MAPE dos modelos com orientação horizontal
def plot_mape_comparison(metrics_df):
    plt.figure(figsize=(12, 6))
    sns.barplot(y='Model', x='MAPE', data=metrics_df.sort_values(by='MAPE'), orient='h')
    plt.title('MAPE Comparison Across Models')
    plt.ylabel('Model')
    plt.xlabel('MAPE')
    plt.show()
    

# Função para plotar o gráfico de erro do melhor modelo
def plot_best_model_error(metrics_df):
    best_model_name = metrics_df.iloc[metrics_df['MAPE'].idxmin()]['Model']
    errors = metrics_df.loc[metrics_df['Model'] == best_model_name, ['MAE', 'MSE', 'RMSE', 'MAPE']].squeeze()
    
    plt.figure(figsize=(10, 6))
    errors.plot(kind='bar', color=['blue', 'green', 'orange', 'red'])
    plt.title(f'Error Metrics for the Best Model: {best_model_name}')
    plt.ylabel('Error Value')
    plt.xlabel('Metric')
    plt.show()

# Função para plotar o gráfico da previsão do consumo por break_variable
def plot_target_by_break_variable(data, model, numeric_features, categorical_features, target, break_variable):
    unique_values = data[break_variable].unique()

    plt.figure(figsize=(16, 8 * len(unique_values)))

    for i, value in enumerate(unique_values, start=1):
        subset_data = data[data[break_variable] == value]
        
        if isinstance(model, SARIMAX):
            order = model.order
            seasonal_order = model.seasonal_order
            fitted_model = SARIMAX(endog=subset_data[target], order=order, seasonal_order=seasonal_order)
            results = fitted_model.fit(disp=False)
            subset_data['y_'] = results.predict(start=0, end=len(subset_data) - 1)
        else:
            model.fit(subset_data[numeric_features + categorical_features], subset_data[target])
            subset_data['y_'] = model.predict(subset_data[numeric_features + categorical_features])

        subset_data['date'] = pd.to_datetime(subset_data['timestamp'], unit='ms')
        
        plt.subplot(len(unique_values), 1, i)
        plt.plot(subset_data['date'], subset_data[target], label='True', linestyle='--')
        plt.plot(subset_data['date'], subset_data['y_'], label='Predicted')
        plt.title(f'{break_variable}={value}')
        plt.xlabel('Date')
        plt.ylabel(target)
        plt.legend()

    plt.tight_layout()
    plt.show()


