import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from collections import Counter
from matplotlib.colors import ListedColormap
from sklearn.metrics import roc_curve
import requests

# Структурируем страницу с вкладками
st.title("Анализ данных и машинное обучение")

# Вкладки
tab_program, tab_dataset, tab_eda, tab_results, tab_visualization = st.tabs(
    ["Об программе", "Датасет", "EDA", "Результаты моделей", "Визуализация моделей"]
)

# Раздел "Об программе"
with tab_program:
    st.header("Об программе")
    st.write("""
        Этот проект направлен на решение задачи классификации с использованием различных моделей машинного обучения.
        Применяем методы KNN, логистической регрессии и SVM для предсказания целевой переменной на основе выбранных признаков.
        Данный проект также включает в себя этапы предобработки данных, анализа и визуализации результатов.
        Цель программы — обучение моделей, оценка их производительности и визуализация границ классификации.
    """)

# Раздел "Датасет"
with tab_dataset:
    st.header("Датасет")

    # Путь к файлам
    data_path_train = "C:\\Users\\User\\OneDrive\\Desktop\\Анализ больших данных\\Практика1\\data\\train.csv"

    # Загрузка данных
    train = pd.read_csv(data_path_train)

    # Информация о датасете
    info_response = requests.get("http://127.0.0.1:8000/info")
    info = info_response.json()
    st.write("### Общая информация о датасете:")
    st.write(f"Количество строк в train.csv: {info['Количество строк']}")
    st.write(f"Количество колонок в train.csv: {info['Количество колонок']}")
    st.write(f"Размер датасета train.csv (в байтах): {info['Размер датасета']} байт")

    # Описание каждого поля
    st.write("### Описание полей:")
    description = {
        "gravity": "Гравитация (числовой признак)",
        "ph": "pH (числовой признак)",
        "osmo": "Осмос (целочисленный признак)",
        "cond": "Проводимость (числовой признак)",
        "urea": "Мочевина (целочисленный признак)",
        "calc": "Кальций (числовой признак)",
        "target": "Целевая переменная (бинарная классификация)",
    }
    for col, desc in description.items():
        st.write(f"**{col}**: {desc}")
    
    # Отображение всего датасета
    st.write("### Полный датасет train.csv:")
    train_response = requests.get("http://127.0.0.1:8000/dataset")
    train_data = train_response.json()
    train_df = pd.DataFrame(train_data)
    st.dataframe(train_df)

# Раздел "EDA"
with tab_eda:
    st.header("EDA (Exploratory Data Analysis)")
    st.write("""
        В этом разделе проводим анализ данных для выявления статистических характеристик, таких как минимальные, 
        максимальные значения, медиана, среднее, а также квартильные значения для числовых признаков.
        Также мы исследуем категориальные признаки, определяя их наиболее часто встречающиеся значения.
    """)

    st.write("### Анализ числовых признаков:")
    numerical_analysis_response = requests.get("http://127.0.0.1:8000/numerical_analysis")
    numerical_analysis_data = numerical_analysis_response.json()
    if numerical_analysis_data:
        st.dataframe(numerical_analysis_data)
    else:
        st.write("Числовые признаки отсутствуют.")

    st.write("### Анализ категориальных признаков:")
    categorical_analysis_response = requests.get("http://127.0.0.1:8000/categorical_analysis")
    categorical_analysis_data = categorical_analysis_response.json()
    if len(categorical_analysis_data["Frequency"]):
        st.dataframe(categorical_analysis_data)
    else:
        st.write("Категориальные признаки отсутствуют.")

# Раздел "Результаты моделей с выбором лучшей"
with tab_results:
    st.header("Результаты моделей")

    train_response = requests.get("http://127.0.0.1:8000/dataset")
    train_data = train_response.json()
    train_df = pd.DataFrame(train_data)

    selected_features_response = requests.get("http://127.0.0.1:8000/selected_features")
    selected_features = selected_features_response.json()

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

    results_df = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"])
    st.dataframe(results_df)

# Раздел "Визуализация моделей"
with tab_visualization:
    st.header("Визуализация моделей")
    st.write("""
        В этом разделе визуализируем границы решений для каждой модели.
        Границы разделения отображаются для выбранных признаков, что позволяет лучше понять, как каждая модель классифицирует данные.
    """)

    # Визуализация границ классов
    def plot_decision_boundary(model, X, y, feature_names):
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
        st.pyplot(fig)
        plt.close()

    for name in models:
        st.write(f"### Визуализация границ классов: {name}")
        plot_decision_boundary(models[name], X_selected_scaled, y_selected, selected_features)
