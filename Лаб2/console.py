import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import OneHotEncoder


def load_data():
    return sns.load_dataset('mpg')


def print_shape(df):
    print(f"Строки: {df.shape[0]}, Столбцы: {df.shape[1]}")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.float_format', '{:.3f}'.format)


def analyze_numeric(df):
    numeric_cols = df.select_dtypes(include=['int', 'float']).columns
    return pd.DataFrame({
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


def analyze_categorical(df):
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    return pd.DataFrame({
        'Доля пропусков': df[categorical_cols].isna().mean(),
        'Уникальные значения': df[categorical_cols].nunique(),
        'Мода': df[categorical_cols].mode().iloc[0]
    })


def t_test(df, col, group1, group2, target):
    group1_data = df[df[col] == group1][target]
    group2_data = df[df[col] == group2][target]
    t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False)
    return p_value


def spearman_correlation(df, col1, col2):
    df_clean = df[[col1, col2]].dropna()
    return stats.spearmanr(df_clean[col1], df_clean[col2])


def encode_categorical(df, col):
    encoder = OneHotEncoder()
    encoded = encoder.fit_transform(df[[col]]).toarray()
    return pd.concat([df, pd.DataFrame(encoded, columns=encoder.get_feature_names_out([col]))], axis=1)


def gradient_descent(X, y, lr=0.01, epochs=100, stochastic=False):
    theta = np.random.randn(2)
    X_b = np.c_[np.ones((len(X), 1)), X]
    if stochastic:
        for epoch in range(epochs):
            for i in range(len(X_b)):
                xi = X_b[i:i + 1]
                yi = y[i]
                gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
                theta -= lr * gradients
    else:
        for epoch in range(epochs):
            gradients = 2 / len(X) * X_b.T.dot(X_b.dot(theta) - y)
            theta -= lr * gradients
    return theta


def main():
    df = load_data()
    print_shape(df)
    print("Числовые переменные:")
    print(analyze_numeric(df))
    print("\nКатегориальные переменные:")
    print(analyze_categorical(df))

    p_value_ttest = t_test(df, 'cylinders', 4, 6, 'mpg')
    print(f"\nГипотеза 1: p-value = {p_value_ttest:.4f}")

    corr, p_value_corr = spearman_correlation(df, 'horsepower', 'mpg')
    print(f"Гипотеза 2: Коэффициент Спирмена = {corr:.2f}, p-value = {p_value_corr:.4f}")

    df_encoded = encode_categorical(df, 'origin')
    target = 'mpg'
    features = df_encoded.select_dtypes(include=['int', 'float']).columns.drop(target)
    correlation = df_encoded[features].corrwith(df_encoded[target])
    print("\nКорреляция с целевой переменной (mpg):")
    print(correlation.sort_values(ascending=False))

    data = df[['horsepower', 'mpg']].dropna()
    X = (data['horsepower'].values.reshape(-1, 1) - data['horsepower'].mean()) / data['horsepower'].std()
    y = (data['mpg'].values - data['mpg'].mean()) / data['mpg'].std()

    theta_sgd = gradient_descent(X, y, stochastic=True)
    theta_gd = gradient_descent(X, y)
    print(f"\nКоэффициенты SGD: {theta_sgd}")
    print(f"Коэффициенты GD: {theta_gd}")


if __name__ == "__main__":
    main()