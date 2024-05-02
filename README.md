## Link Oficial

[MLflow | MLflow](https://mlflow.org/)

## Usuários Alvo

**Cientistas de Dados** utilizam o MLflow para:

- Rastreamento de experimentos e persistência de testes de hipóteses.
- Estruturação de código para melhor reprodutibilidade.
- Embalar modelos e gerenciamento de dependências.
- Avaliação dos limites de seleção de ajuste de hiperparâmetros.
- Comparação dos resultados do re-treinamento de modelos ao longo do tempo.
- Revisão e seleção de modelos ótimos para implantação.

**Profissionais de MLOps** utilizam o MLflow para:

- Gerenciar os ciclos de vida de modelos treinados, tanto pré quanto pós implantação.
- Implantar modelos de forma segura em ambientes de produção.
- Auditar e revisar modelos candidatos antes da implantação.
- Gerenciar dependências de implantação.

**Gerentes de Ciência de Dados** interagem com o MLflow por:

- Revisão dos resultados de experimentação e atividades de modelagem.
- Colaboração com equipes para garantir que os objetivos de modelagem estejam alinhados com os objetivos de negócios.

**Usuários de Engenharia de Prompt** usam o MLflow para:

- Avaliação e experimentação com grandes modelos de linguagem.
- Criação de prompts personalizados e persistência de suas criações de candidatos.
- Decidindo sobre o melhor modelo base adequado para suas necessidades específicas de projeto.

## Instalação

```bash
# criar um ambiente virtual do python
python -m venv .venv
source .venv/bin/activate

# gerar o arquivo requirements
echo "mlflow==2.12.1" >> requirements.txt

# instalação
pip install -r requirements.txt

# execução do server
**mlflow server --host 127.0.0.1 --port 8080**
```

O ML Flow irá iniciar no [localhost](http://localhost) porta 8080.

## Primeiro uso do MLFlow

Abaixo um código Python com o treinamento de um modelo, baseado no dataset iris e o uso do MLflow para registrar os logs e registrar o experimento.

Crie um arquivo dentro do seu diretório, pode ser `01_learning.py`

```python
import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

# Train the model
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)

# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment("MLflow Quickstart")

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Log the loss metric
    mlflow.log_metric("accuracy", accuracy)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic LR model for iris data")

    # Infer the model signature
    signature = infer_signature(X_train, lr.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="iris_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-quickstart",
    )
		
		# Load the model back for predictions as a generic Python Function model
		loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
		
		predictions = loaded_model.predict(X_test)
		
		iris_feature_names = datasets.load_iris().feature_names
		
		result = pd.DataFrame(X_test, columns=iris_feature_names)
		result["actual_class"] = y_test
		result["predicted_class"] = predictions
		
		result[:4]
```

Execute o código acima.

```bash
python 01_learning.py
```

Abra a UI do MLFlow Para verificar os registros dos experimentos.