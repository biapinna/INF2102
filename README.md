# Instalação e Execução

## Requisitos Prévios:

Python 3.x instalado

Bibliotecas necessárias (instaladas via pip install -r requirements.txt)

## Passos:

Faça o download ou clone o repositório para o seu ambiente local.

Instale as dependências.

    pip install -r requirements.txt

Execute o script de exemplo.

# Contato

e-mail: pinnarbeatriz@tecgraf.puc-rio.br

# Manual

A seguir estão as descrições detalhadas das funções definidas no script forecast_models.py:

1) **mean_absolute_percentage_error(y_true, y_pred)**

* Descrição: Calcula o erro percentual absoluto médio (MAPE) entre os valores reais (y_true) e os valores previstos (y_pred).

* Parâmetros de Entrada:

   - y_true: Array ou lista com os valores reais.
   - y_pred: Array ou lista com os valores previstos.

* Saída: Retorna o MAPE, uma métrica de erro percentual.

2) **fit_arima_model(x, y, seasonal=True)**

* Descrição: Ajusta automaticamente um modelo ARIMA ou SARIMA à série temporal fornecida (y) utilizando a biblioteca pmdarima. Retorna métricas de desempenho, incluindo MAPE.

* Parâmetros de Entrada:

    - x: Variável independente (pode ser uma série temporal).
    - y: Variável dependente (série temporal a ser prevista).
    - seasonal: Booleano indicando se deve considerar sazonalidade (padrão: True).

* Saída: Retorna MAPE, ordem do modelo ARIMA e ordem sazonal (se seasonal=True).

3) **cross_val_metrics(model, x, y, encoded_break_variable=None, arima=False, seasonal=True)**

* Descrição: Realiza validação cruzada para avaliar modelos, calculando métricas como MAE, MSE, RMSE e MAPE. Pode ser usado tanto para modelos de aprendizado de máquina quanto para ARIMA.

* Parâmetros de Entrada:
    - model: Modelo de aprendizado de máquina ou None para ARIMA.
    - x: Variáveis independentes.
    - y: Variável dependente.
    - encoded_break_variable: Variável de quebra codificada (para validação estratificada).
    - arima: Booleano indicando se o modelo é ARIMA (padrão: False).
    - seasonal: Booleano indicando se deve considerar sazonalidade (padrão: True).

* Saída: Retorna MAE, MSE, RMSE e MAPE (ou None para ARIMA).

4) **pipeline_generator(model, numeric_features, categorical_features, is_arima=False)**

* Descrição: Gera e retorna um pipeline de pré-processamento e modelo. O pipeline pode ser configurado para modelos ARIMA ou de aprendizado de máquina.

* Parâmetros de Entrada:
    - model: Modelo.
    - numeric_features: Lista de características numéricas.
    - categorical_features: Lista de características categóricas.
    - is_arima: Booleano indicando se é um modelo ARIMA (padrão: False).

* Saída: Retorna o pipeline configurado.

5) **train_and_evaluate_models(models, data, numeric_features, categorical_features, target, break_variable=None)**
   
* Descrição: Treina e avalia vários modelos de aprendizado de máquina utilizando validação cruzada. Retorna um DataFrame contendo métricas de desempenho para cada modelo.

* Parâmetros de Entrada:
    - models: Dicionário de modelos de aprendizado de máquina.
    - data: DataFrame contendo os dados.
    - numeric_features: Lista de características numéricas.
    - categorical_features: Lista de características categóricas.
    - target: Variável dependente.
    - break_variable: Variável de quebra para validação estratificada.


6) **plot_mape_comparison(metrics_df)**

* Descrição: Gera um gráfico comparativo horizontal de MAPE entre diferentes modelos. Facilita a visualização e comparação do desempenho dos modelos.

* Parâmetros de Entrada:
    - metrics_df: DataFrame contendo métricas de desempenho.

7) **plot_best_model_error(metrics_df)**

* Descrição: Gera um gráfico de barras com os erros (MAE, MSE, RMSE, MAPE) do melhor modelo identificado. Ajuda na visualização dos diferentes aspectos do desempenho do modelo.

* Parâmetros de Entrada:
    - metrics_df: DataFrame contendo métricas de desempenho.

8) **plot_target_by_break_variable(data, model, numeric_features, categorical_features, target, break_variable)**

* Descrição: Plota gráficos de previsão de consumo por variável de quebra para um modelo específico. Útil para entender o comportamento do modelo em diferentes cenários.

* Parâmetros de Entrada:
    - data: DataFrame contendo os dados.
    - model: Modelo treinado.
    - numeric_features: Lista de características numéricas.
    - categorical_features: Lista de características categóricas.
    - target: Variável dependente.
    - break_variable: Variável de quebra.

