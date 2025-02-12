import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# Создание базы данных и подключение
conn = sqlite3.connect('nashville_housing.db')
cursor = conn.cursor()

# Чтение CSV
df = pd.read_csv('Nashville Housing.csv')

# Загрузка в SQLite
df.to_sql('NashvilleHousing', conn, if_exists='replace', index=False)

# Чтение данных из SQLite в DataFrame
df_analysis = pd.read_sql('SELECT * FROM NashvilleHousing', conn)

# Получение списка таблиц
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Таблицы в базе данных:")
for table in tables:
    print(table[0])

# Анализ структуры каждой таблицы
for table in tables:
    table_name = table[0]
    print(f"\nСтруктура таблицы {table_name}:")
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    for column in columns:
        print(f"Поле: {column[1]}, Тип: {column[2]}")

# Одномерный анализ
# Гистограмма 1: Распределение SalePrice
plt.figure(figsize=(10, 6))
plt.hist(df_analysis['SalePrice'], bins=30, color='skyblue', edgecolor='black')
plt.title('Распределение цен продажи (SalePrice)')
plt.xlabel('Цена')
plt.ylabel('Частота')
plt.show()

# Гистограмма 2: Распределение YearBuilt
plt.figure(figsize=(10, 6))
plt.hist(df_analysis['YearBuilt'], bins=30, color='salmon', edgecolor='black')
plt.title('Распределение года постройки (YearBuilt)')
plt.xlabel('Год')
plt.ylabel('Частота')
plt.show()

# Многомерный анализ
# График: Зависимость SalePrice от YearBuilt и Bedrooms
plt.figure(figsize=(12, 8))
plt.scatter(
    df_analysis['YearBuilt'],
    df_analysis['SalePrice'],
    c=df_analysis['Bedrooms'],
    cmap='viridis',
    alpha=0.6
)
plt.colorbar(label='Количество спален')
plt.title('Цена продажи vs Год постройки (по количеству спален)')
plt.xlabel('Год постройки')
plt.ylabel('Цена продажи')
plt.show()
