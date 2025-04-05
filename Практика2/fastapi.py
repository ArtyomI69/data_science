from typing import Union
from fastapi import Query
import numpy as np
from fastapi import FastAPI
from matplotlib.colors import ListedColormap
from pydantic import BaseModel
import pandas as pd
import matplotlib.pyplot as plt
from fastapi.responses import JSONResponse
import base64
import io
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import base64
from io import BytesIO
from PIL import Image


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
def read_dataset(limit: Union[int, None] = Query(default=None, ge=1)):
    if limit is not None:
        return train.head(limit).to_dict(orient="records")
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
class ModelResultsResponse(BaseModel):
    results: dict
    X_selected_scaled: list
    y_selected: list

@app.get("/model-results", response_model=ModelResultsResponse)
def get_model_results():
    selected_features = ["gravity", "ph"]
    X_selected = train[selected_features].values
    y_selected = train["target"].values

    scaler = StandardScaler()
    X_selected_scaled = scaler.fit_transform(X_selected)
    X_train, X_val, y_train, y_val = train_test_split(X_selected_scaled, y_selected, test_size=0.2, random_state=42)

    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]) if hasattr(model, 'predict_proba') else "N/A"
        results[name] = [acc, prec, rec, f1, auc]

    return {
        "results": results,
        "X_selected_scaled": X_selected_scaled.tolist(),
        "y_selected": y_selected.tolist()
    }

def generate_plot_image(model, X, y, feature_names):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA"])
    cmap_bold = ListedColormap(["#FF0000", "#00AA00"])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor="k", s=20)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_title(f"Границы классов для {feature_names[0]} и {feature_names[1]}")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

@app.get("/model-visualization")
def model_visualization():
    data = pd.read_csv("C:\\Users\\User\\OneDrive\\Desktop\\Анализ больших данных\\Практика2\\data\\train.csv")
    selected_features = ["gravity", "ph"]
    X = data[selected_features].values
    y = data["target"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(),
    }

    images = {}
    for name, model in models.items():
        model.fit(X_scaled, y)
        img_base64 = generate_plot_image(model, X_scaled, y, selected_features)
        images[name] = img_base64

    return JSONResponse(content=images)

# Добавтиь один изменяемый параметр. На клиенте что то меняем