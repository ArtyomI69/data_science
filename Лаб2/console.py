# Импорт библиотек
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# 1. Загрузка данных
df = sns.load_dataset('mpg')

# 2. Количество строк и столбцов
print(f"Строки: {df.shape[0]}, Столбцы: {df.shape[1]}")
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', '{:.3f}'.format)

# 3. Разведочный анализ
# a. Числовые переменные
numeric_cols = df.select_dtypes(include=['int', 'float']).columns
numeric_stats = pd.DataFrame({
    'Доля пропусков': df[numeric_cols].isna().mean(),
    'Мин': df[numeric_cols].min(),
    'Макс': df[numeric_cols].max(),
    'Среднее': df[numeric_cols].mean(),
    'Медиана': df[numeric_cols].median(),
    'Дисперсия': df[numeric_cols].var(),
    '0.1 квантиль': df[numeric_cols].quantile(0.1),
    '0.9 квантиль': df[numeric_cols].quantile(0.9),
    '1-й квартиль': df[numeric_cols].quantile(0.25),
    '3-й квартиль': df[numeric_cols].quantile(0.75)
})

# b. Категориальные переменные
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
categorical_stats = pd.DataFrame({
    'Доля пропусков': df[categorical_cols].isna().mean(),
    'Уникальные значения': df[categorical_cols].nunique(),
    'Мода': df[categorical_cols].mode().iloc[0]
})

print("Числовые переменные:")
print(numeric_stats)
print("\nКатегориальные переменные:")
print(categorical_stats)

# 4. Статистические гипотезы
# Гипотеза 1: Средний mpg для автомобилей с 4 и 6 цилиндрами различается (t-тест)
cyl_4 = df[df['cylinders'] == 4]['mpg']
cyl_6 = df[df['cylinders'] == 6]['mpg']
t_stat, p_value = stats.ttest_ind(cyl_4, cyl_6, equal_var=False)
print(f"\nГипотеза 1: p-value = {p_value:.4f}")

# Гипотеза 2: Корреляция между horsepower и mpg
# Синхронное удаление пропусков в обоих столбцах
df_clean = df[['horsepower', 'mpg']].dropna()
corr, p_value = stats.spearmanr(df_clean['horsepower'], df_clean['mpg'])
print(f"Гипотеза 2: Коэффициент Спирмена = {corr:.2f}, p-value = {p_value:.4f}")

# 5. Кодирование категориальных переменных
encoder = OneHotEncoder()
encoded_origin = encoder.fit_transform(df[['origin']]).toarray()
df_encoded = pd.concat([df, pd.DataFrame(encoded_origin, columns=encoder.get_feature_names_out(['origin']))], axis=1)

# 6. Таблица корреляции
target = 'mpg'
features = df_encoded.select_dtypes(include=['int', 'float']).columns.drop(target)
correlation = df_encoded[features].corrwith(df_encoded[target])
print("\nКорреляция с целевой переменной (mpg):")
print(correlation.sort_values(ascending=False))

# 7. Реализация градиентного спуска (исправленная версия)

# Синхронное удаление пропусков для X и y
data = df[['horsepower', 'mpg']].dropna()  # Важно: удаляем пропуски совместно!
X = data['horsepower'].values.reshape(-1, 1)  # Преобразуем в 2D массив
y = data['mpg'].values

# Стандартизация данных
X = (X - X.mean()) / X.std()
y = (y - y.mean()) / y.std()

# Проверка размерностей
print(f"Размер X: {X.shape}, Размер y: {y.shape}")  # Должно быть (392, 1) и (392,)

# Стохастический градиентный спуск (исправленная версия)
def stochastic_gradient_descent(X, y, lr=0.01, epochs=100):
    theta = np.random.randn(2)
    X_b = np.c_[np.ones((len(X), 1)), X]  # Добавляем столбец единиц
    for epoch in range(epochs):
        for i in range(len(X_b)):
            xi = X_b[i:i+1]
            yi = y[i]
            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
            theta -= lr * gradients
    return theta

# Обычный градиентный спуск (исправленная версия)
def gradient_descent(X, y, lr=0.01, epochs=100):
    theta = np.random.randn(2)
    X_b = np.c_[np.ones((len(X), 1)), X]  # Добавляем столбец единиц
    for epoch in range(epochs):
        gradients = 2/len(X) * X_b.T.dot(X_b.dot(theta) - y)
        theta -= lr * gradients
    return theta

# Запуск алгоритмов
theta_stochastic = stochastic_gradient_descent(X, y)
theta_standard = gradient_descent(X, y)
print(f"\nКоэффициенты SGD: {theta_stochastic}")
print(f"Коэффициенты GD: {theta_standard}")