import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import ttest_ind, pearsonr

# 1. Загрузка данных
print("\n================= 1. Загрузка данных =================")
df = pd.read_csv(
    "./data/move.csv")
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', '{:.3f}'.format)
print(df.info())
print(df.head())

# 2. Анализ данных
print("\n================= 2. Описательная статистика =================")
print(df.describe())

# 3. Обработка пропусков и выбросов
print("\n================= 3. Обработка пропущенных значений =================")
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

df[num_cols] = df[num_cols].fillna(df[num_cols].median())
print("Пропущенные значения обработаны: числовые - медианой.")

# 3.1. Анализ категориальных признаков
print("\n================= 3.1. Анализ категориальных признаков =================")
categorical_cols = ['metro', 'way', 'provider']

# Вывод количества уникальных значений
for col in categorical_cols:
    print(f"{col}: {df[col].nunique()} уникальных значений")
    print(df[col].value_counts(), "\n")

# 4. Удаление выбросов
print("\n================= 4. Удаление выбросов =================")
q1 = df[num_cols].quantile(0.25)
q3 = df[num_cols].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
df = df[~((df[num_cols] < lower_bound) | (df[num_cols] > upper_bound)).any(axis=1)]
print(f"Размер данных после удаления выбросов: {df.shape}")

# 5. Проверка гипотез
print("\n================= 5. Проверка гипотез =================")

# Гипотеза 1: Среднее значение цены ниже для высоких этажей
median_storey = df['storey'].median()
low_storey = df[df['storey'] <= median_storey]['price']
high_storey = df[df['storey'] > median_storey]['price']
t_stat, p_value = ttest_ind(low_storey, high_storey, equal_var=False)
print(f"Гипотеза 1 - Квартиры на верхних этажах дороже:")
print(f"t-статистика: {t_stat:.4f}, p-значение: {p_value:.4f}")
if p_value < 0.05:
    print("Отвергаем нулевую гипотезу: средние значения различны.")
else:
    print("Не отвергаем нулевую гипотезу: значимых различий нет.")

# Гипотеза 2: Корреляция между количеством комнат и ценой жилья
corr, p_corr = pearsonr(df['total_area'], df['price'])
print("\nГипотеза 2 - Корреляция между площадью и ценой:")
print(f"Коэффициент корреляции: {corr:.4f}, p-значение: {p_corr:.4f}")
if p_corr < 0.05:
    print("Отвергаем нулевую гипотезу: существует значимая корреляция.")
else:
    print("Не отвергаем нулевую гипотезу: значимой связи нет.")

# 6. Разделение на train/test
print("\n================= 6. Разделение данных =================")
# Исключение категориальных переменных
df = df.drop(columns=["way", "metro", "provider", "Unnamed: 0"])  # Убираем ненужные столбцы

# Заполнение пропущенных значений медианой
df.fillna(df.median(), inplace=True)


target_column = 'price'  # Целевая переменная
X = df.drop(columns=[target_column])
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Данные разделены: train и test.")

# 7. Нормализация данных
print("\n================= 7. Нормализация данных =================")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Нормализация выполнена.")

# 8. Обучение моделей
print("\n================= 8. Обучение моделей =================")
models = {
    "KNN Regressor": KNeighborsRegressor(n_neighbors=5)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    results[name] = [mae, mse, rmse, r2]
    print(f"{name}:\n MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}\n")

    # Визуализация предсказанных vs реальных значений с гладкой апроксимацией
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', label='data')
    sorted_indices = np.argsort(y_test)
    y_pred_sorted = np.array(y_pred)[sorted_indices]
    y_test_sorted = np.array(y_test)[sorted_indices]

    # Используем скользящее среднее для сглаживания линии
    window_size = 5
    y_pred_smoothed = np.convolve(y_pred_sorted, np.ones(window_size) / window_size, mode='valid')
    y_test_smoothed = np.convolve(y_test_sorted, np.ones(window_size) / window_size, mode='valid')

    plt.plot(y_test_smoothed, y_pred_smoothed, color='green', label='prediction')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='ideal')
    plt.xlabel("Реальные значения")
    plt.ylabel("Предсказанные значения")
    plt.title(f"{name} - Апроксимирующая кривая и данные")
    plt.legend()
    plt.show()
