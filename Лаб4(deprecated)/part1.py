import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import ttest_ind, pearsonr

# Реализация собственного KNN Regressor
class CustomKNNRegressor:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = np.array(y)

    def predict(self, X):
        y_pred = []
        for x in X:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            neighbors_idx = np.argsort(distances)[:self.n_neighbors]
            y_pred.append(np.mean(self.y_train[neighbors_idx]))
        return np.array(y_pred)


# Реализация собственной линейной регрессии
class CustomLinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        theta_best = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.bias = theta_best[0]
        self.weights = theta_best[1:]

    def predict(self, X):
        return X.dot(self.weights) + self.bias


# Реализация собственного LASSO Regressor
class CustomLassoRegression:
    def __init__(self, alpha=0.1, iterations=1000, learning_rate=0.01):
        self.alpha = alpha
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for _ in range(self.iterations):
            y_pred = X.dot(self.weights) + self.bias
            dw = (-2 / m) * X.T.dot(y - y_pred) + self.alpha * np.sign(self.weights)
            db = (-2 / m) * np.sum(y - y_pred)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return X.dot(self.weights) + self.bias


# 1. Загрузка данных
print("\n================= 1. Загрузка данных =================")
df = pd.read_csv("./move.csv")
df = df.drop(columns=['Unnamed: 0'])
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
target_column = 'price'  # Целевая переменная
X = df.drop(columns=[target_column])
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Данные разделены: train и test.")

# 7. Нормализация данных
print("\n================= 7. Нормализация данных =================")
# Выделяем числовые и категориальные признаки
num_features = ['fee_percent', 'storey', 'minutes', 'storeys', 'living_area', 'kitchen_area', 'total_area']
cat_features = ['metro', 'way', 'provider']

# Применяем One-Hot Encoding к категориальным признакам
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_cat_train = encoder.fit_transform(X_train[cat_features])
X_cat_test = encoder.transform(X_test[cat_features])

# Масштабируем числовые признаки
scaler = StandardScaler()
X_num_train = scaler.fit_transform(X_train[num_features])
X_num_test = scaler.transform(X_test[num_features])

# Объединяем обработанные признаки
X_train = np.hstack([X_num_train, X_cat_train])
X_test = np.hstack([X_num_test, X_cat_test])

print("Нормализация выполнена.")

# 8. Обучение моделей
print("\n================= 8. Обучение моделей =================")
models = {
    "Custom Linear Regression": CustomLinearRegression(),
    "Linear Regression": LinearRegression(),
    "Custom KNN Regressor": CustomKNNRegressor(n_neighbors=5),
    "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
    "Custom LASSO Regressor": CustomLassoRegression(alpha=0.1),
    "Lasso": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5)
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

# 9. Выбор лучшей модели
best_model = max(results, key=lambda x: results[x][-1])
print(f"Лучшая модель: {best_model}")