from typing import Union

import numpy as np
from fastapi import FastAPI
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

app = FastAPI()

# Путь к файлам
data_path_train = "C:\\Users\\User\\OneDrive\\Desktop\\Анализ больших данных\\Практика2\\data\\train.csv"

# Загрузка данных
train = pd.read_csv(data_path_train)

# Информация о датасете
@app.get("/info")
def read_info():
    return {
        "Количество строк": int(train.shape[0]),
        "Количество колонок": int(train.shape[1]),
        "Размер датасета": int(train.memory_usage(deep=True).sum())
    }

# Отображение всего датасета
@app.get("/dataset")
def read_dataset():
    return train.to_dict(orient="records")

# Анализ числовых признаков
@app.get("/numerical_analysis")
def read_numerical_analysis():
    numerical_features = train.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numerical_stats = train[numerical_features].agg(['min', 'median', 'mean', 'max']).T
    numerical_stats["25%"] = train[numerical_features].apply(lambda x: np.percentile(x.dropna(), 25))
    numerical_stats["75%"] = train[numerical_features].apply(lambda x: np.percentile(x.dropna(), 75))
    return numerical_stats

# Анализ числовых признаков
@app.get("/categorical_analysis")
def read_categorical_analysis():
    categorical_features = train.select_dtypes(include=['object']).columns.tolist()
    categorical_modes = train[categorical_features].mode().iloc[0] if not train[
        categorical_features].mode().empty else None
    categorical_frequencies = train[categorical_features].apply(
        lambda x: x.value_counts().iloc[0] if not x.value_counts().empty else 0)
    categorical_stats = pd.DataFrame({"Mode": categorical_modes, "Frequency": categorical_frequencies})
    return categorical_stats

# Результаты моделей
@app.get("/selected_features")
def read_model_results():
    selected_features = ["gravity", "ph"]

    return selected_features
