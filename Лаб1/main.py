import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data_to_db(csv_file, db_file):
    df = pd.read_csv(csv_file)
    with sqlite3.connect(db_file) as conn:
        df.to_sql('NashvilleHousing', conn, if_exists='replace', index=False)
        return conn


def analyze_db_structure(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Таблицы в базе данных:")
    for table in tables:
        print(table[0])

    for table in tables:
        table_name = table[0]
        print(f"\nСтруктура таблицы {table_name}:")
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        for column in columns:
            print(f"Поле: {column[1]}, Тип: {column[2]}")


def fetch_data(conn, query):
    return pd.read_sql(query, conn)


def clean_data(df):
    try:
        df['SalePrice'] = df['SalePrice'].replace('[\$,]', '', regex=True).astype(float)
    except Exception as e:
        print(f"Ошибка обработки данных: {e}")
    return df


def plot_histogram(data, column, bins, color, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    plt.hist(data[column], bins=bins, color=color, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.show()


def plot_scatter(data, x_col, y_col, c_col, cmap, title, xlabel, ylabel):
    plt.figure(figsize=(12, 8))
    plt.scatter(data[x_col], data[y_col], c=data[c_col], cmap=cmap, alpha=0.6)
    plt.colorbar(label='Количество спален')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


if __name__ == "__main__":
    db_connection = load_data_to_db('Nashville Housing.csv', 'nashville_housing.db')
    analyze_db_structure(db_connection)

    df_analysis = fetch_data(db_connection, 'SELECT * FROM NashvilleHousing LIMIT 1000')
    df_analysis1 = fetch_data(db_connection, 'SELECT * FROM NashvilleHousing')

    df_analysis = clean_data(df_analysis)

    plot_histogram(df_analysis, 'SalePrice', bins=30, color='skyblue', title='Распределение цен продажи (SalePrice)',
                   xlabel='Цена', ylabel='Частота')
    plot_histogram(df_analysis1, 'YearBuilt', bins=30, color='salmon', title='Распределение года постройки (YearBuilt)',
                   xlabel='Год', ylabel='Частота')
    plot_scatter(df_analysis, 'YearBuilt', 'SalePrice', 'Bedrooms', cmap='viridis',
                 title='Цена продажи vs Год постройки (по количеству спален)', xlabel='Год постройки',
                 ylabel='Цена продажи')
