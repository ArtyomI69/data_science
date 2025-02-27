import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import OneHotEncoder


def load_data():
    return sns.load_dataset('mpg')


def visualize_data_shape(df):
    plt.figure(figsize=(6, 3))
    plt.text(0.1, 0.6, f"Количество наблюдений: {df.shape[0]}", fontsize=12)
    plt.text(0.1, 0.3, f"Количество признаков: {df.shape[1]}", fontsize=12)
    plt.axis('off')
    plt.title("Размерность данных")
    plt.show()


def plot_numeric_distributions(df):
    numeric_cols = df.select_dtypes(include=['int', 'float']).columns.drop('mpg')
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(3, 3, i)
        sns.histplot(df[col], kde=True)
        plt.title(f"Распределение {col}")
    plt.tight_layout()
    plt.show()


def plot_categorical_distributions(df):
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(categorical_cols, 1):
        plt.subplot(1, 3, i)
        df[col].value_counts().nlargest(25).plot(kind='bar')
        plt.title(f"Распределение {col}")
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def run_hypothesis(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='cylinders', y='mpg', data=df[df['cylinders'].isin([4, 6])])
    plt.title("Сравнение топливной эффективности\n4-х и 6-цилиндровых автомобилей")
    plt.show()


def plot_correlation(df):
    df_clean = df[['horsepower', 'mpg']].dropna()
    plt.figure(figsize=(10, 6))
    sns.regplot(x='horsepower', y='mpg', data=df_clean, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
    plt.title("Зависимость топливной эффективности от мощности двигателя")
    plt.show()


def encode_categorical(df):
    encoder = OneHotEncoder()
    encoded_origin = encoder.fit_transform(df[['origin']]).toarray()
    return pd.concat([df, pd.DataFrame(encoded_origin, columns=encoder.get_feature_names_out(['origin']))], axis=1)


def plot_correlation_matrix(df):
    plt.figure(figsize=(12, 8))
    corr_matrix = df.select_dtypes(include=['number']).corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={'label': 'Коэффициент корреляции'})
    plt.title("Матрица корреляций признаков")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()


def stochastic_gradient_descent(X, y, lr=0.01, epochs=100):
    theta = np.random.randn(2)
    X_b = np.c_[np.ones((len(X), 1)), X]
    for epoch in range(epochs):
        for i in range(len(X_b)):
            xi = X_b[i:i + 1]
            yi = y[i:i + 1]
            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
            theta -= lr * gradients
    return theta


def gradient_descent(X, y, lr=0.01, epochs=100):
    theta = np.random.randn(2)
    X_b = np.c_[np.ones((len(X), 1)), X]
    for epoch in range(epochs):
        gradients = 2 / len(X) * X_b.T.dot(X_b.dot(theta) - y)
        theta -= lr * gradients
    return theta


def plot_gradient_descent(X, y, theta, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.7)
    x_values = np.array([X.min(), X.max()])
    y_values = theta[0] + theta[1] * x_values
    plt.plot(x_values, y_values, color='red')
    plt.xlabel('Мощность двигателя (стандартизованная)')
    plt.ylabel('MPG (стандартизованный)')
    plt.title(title)
    plt.show()


def main():
    df = load_data()
    visualize_data_shape(df)
    plot_numeric_distributions(df)
    plot_categorical_distributions(df)
    run_hypothesis(df)
    plot_correlation(df)
    df_encoded = encode_categorical(df)
    plot_correlation_matrix(df_encoded)

    data = df[['horsepower', 'mpg']].dropna()
    X = data[['horsepower']].values
    y = data['mpg'].values
    X_normalized = (X - X.mean()) / X.std()
    y_normalized = (y - y.mean()) / y.std()

    theta_stochastic = stochastic_gradient_descent(X_normalized, y_normalized)
    theta_standard = gradient_descent(X_normalized, y_normalized)

    plot_gradient_descent(X_normalized, y_normalized, theta_stochastic, "Стохастический градиентный спуск")
    plot_gradient_descent(X_normalized, y_normalized, theta_standard, "Обычный градиентный спуск")


if __name__ == "__main__":
    main()