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
import base64
from PIL import Image
from io import BytesIO


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

    # Добавляем инпут для лимита
    limit = st.number_input(
        "Введите количество строк для отображения:",
        min_value=1,
        value=1,
        step=1,
        help="Введите число больше 0"
    )

    # Добавляем кнопку для применения лимита
    if st.button("Применить лимит"):
        train_response = requests.get(f"http://127.0.0.1:8000/dataset?limit={limit}")
    else:
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

    response = requests.get("http://localhost:8000/model-results")
    data = response.json()

    results = data["results"]
    X_selected_scaled = np.array(data["X_selected_scaled"])
    y_selected = np.array(data["y_selected"])
    selected_features = ["gravity", "ph"]

    results_df = pd.DataFrame(results, index=["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"])
    st.dataframe(results_df)

# Раздел "Визуализация моделей"
with tab_visualization:
    st.header("Визуализация моделей")
    st.write("""
        В этом разделе визуализируем границы решений для каждой модели.
        Границы разделения отображаются для выбранных признаков, что позволяет лучше понять, как каждая модель классифицирует данные.
    """)

    # Получаем изображения от FastAPI
    response = requests.get("http://localhost:8000/model-visualization")
    images = response.json()

    for model_name, image_base64 in images.items():
        st.subheader(f"Границы классов: {model_name}")
        img_bytes = base64.b64decode(image_base64)
        image = Image.open(BytesIO(img_bytes))
        st.image(image, use_container_width=True)