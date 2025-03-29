import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from matplotlib.colors import ListedColormap

# Загрузка данных
print("\n Загружаем данные ")
try:
    train = pd.read_csv("./train.csv")
    test = pd.read_csv("./test.csv")
    print(f"Размер train: {train.shape}, Размер test: {test.shape}")
except Exception as e:
    print(f"Ошибка при загрузке данных: {e}")
    exit()

print("\n Информация о наборе данных ")
print(train.info() if not train.empty else "Данные train отсутствуют.")

print("\n Проверка пропущенных значений ")
if not train.empty:
    missing_values = train.isnull().sum()
    print(missing_values if missing_values.sum() > 0 else "Пропусков нет.")
else:
    print("Данные train отсутствуют.")

print("\n Анализ числовых данных ")
numerical_features = train.select_dtypes(include=['float64', 'int64']).columns.tolist()
if numerical_features:
    numerical_stats = train[numerical_features].describe().T
    print(numerical_stats)
else:
    print("Числовые признаки не найдены.")

print("\n Анализ категориальных данных ")
categorical_features = train.select_dtypes(include=['object']).columns.tolist()
if categorical_features:
    categorical_modes = train[categorical_features].mode().iloc[0] if not train[categorical_features].mode().empty else None
    categorical_frequencies = train[categorical_features].apply(lambda x: x.value_counts().iloc[0] if not x.value_counts().empty else 0)
    print(pd.DataFrame({"Наиболее частое значение": categorical_modes, "Частота": categorical_frequencies}))
else:
    print("Категориальные признаки не найдены.")

print("\n Обработка пропущенных значений ")
if not train.empty:
    train.fillna(train.median(), inplace=True)
    print("Пропуски заменены на медианные значения.")
else:
    print("Данные train отсутствуют.")

print("\n Выбор признаков ")
selected_features = ["gravity", "ph"]
if all(feature in train.columns for feature in selected_features):
    X_selected = train[selected_features].values
    y_selected = train["target"].values
else:
    print("Ошибка: Один или несколько выбранных признаков отсутствуют в данных.")
    exit()

print("\n Масштабирование признаков ")
scaler = StandardScaler()
X_selected_scaled = scaler.fit_transform(X_selected)
print("Данные нормализованы.")

print("\n Разделение данных на train/test ")
X_train, X_val, y_train, y_val = train_test_split(X_selected_scaled, y_selected, test_size=0.2, random_state=42)
print("Данные успешно разделены.")

print("\n  Обучение моделей ")
models = {
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Логистическая регрессия": LogisticRegression(),
}

results = {}
for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else np.zeros_like(y_pred, dtype=float)

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_prob) if hasattr(model, 'predict_proba') else "N/A"

        results[name] = [acc, prec, rec, f1, auc]
        print(f"\n{name} -> Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-score: {f1:.4f}, AUC: {auc}")
    except Exception as e:
        print(f"Ошибка во время обучения {name}: {e}")

print("\n Лучшая модель ")
if results:
    best_model = max(results, key=lambda x: results[x][-1] if isinstance(results[x][-1], float) else results[x][-2])
    print(f"Наиболее эффективная модель: {best_model}")
else:
    print("Не удалось определить лучшую модель.")

def plot_decision_boundary(model, X, y, feature_names):
    try:
        h = 0.02  # Шаг сетки
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))

        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA"])
        cmap_bold = ListedColormap(["#FF0000", "#AAFFAA"])

        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor="k", s=20)
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
        plt.title(f"Границы классов для {feature_names[0]} и {feature_names[1]}")
        plt.show()
    except Exception as e:
        print(f"Ошибка при построении границ классов: {e}")

knn_selected = KNeighborsClassifier(n_neighbors=7)
knn_selected.fit(X_selected_scaled, y_selected)
plot_decision_boundary(knn_selected, X_selected_scaled, y_selected, selected_features)

# KNN лучшая модель по параметру n
# Practice1: Проект Streamlit (в telegram есть пример). Берём пример и переделываем под лабу3.
# Practice2: Отделяем Streamlit от FastApi

# 1 KNN -> Accuracy: 0.6386, Precision: 0.6176, Recall: 0.5526, F1-score: 0.5833, AUC: 0.6318713450292399
# 2 KNN -> Accuracy: 0.6627, Precision: 0.7083, Recall: 0.4474, F1-score: 0.5484, AUC: 0.697076023391813
# 3 KNN -> Accuracy: 0.6506, Precision: 0.6154, Recall: 0.6316, F1-score: 0.6234, AUC: 0.6546783625730994
# 4 KNN -> Accuracy: 0.6627, Precision: 0.6923, Recall: 0.4737, F1-score: 0.5625, AUC: 0.6640350877192982
# 5 KNN -> Accuracy: 0.6747, Precision: 0.6571, Recall: 0.6053, F1-score: 0.6301, AUC: 0.6856725146198831
# 6 KNN -> Accuracy: 0.6747, Precision: 0.7037, Recall: 0.5000, F1-score: 0.5846, AUC: 0.6868421052631578
# 7 KNN -> Accuracy: 0.6506, Precision: 0.6286, Recall: 0.5789, F1-score: 0.6027, AUC: 0.6874269005847954