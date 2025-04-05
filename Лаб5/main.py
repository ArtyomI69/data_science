import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from joblib import dump, load
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

# 1. Загрузка данных
df = pd.read_csv("./train.csv")
print("Снижение размерности")
print("[1] Данные загружены успешно!\n")
print("Пример данных:")
print(df.head())
print(df.info())

# 2. Анализ данных
print("[2] Анализ данных (EDA):")

# Разделение на числовые и категориальные признаки
num_features = df.select_dtypes(include=['int64', 'float64']).columns
cat_features = df.select_dtypes(include=['object', 'category']).columns

print("Числовые признаки:")
print(df[num_features].describe())

print("Категориальные признаки:")
if len(cat_features) > 0:
    for col in cat_features:
        print(f"{col}:")
        print(df[col].value_counts())
    print("Вывод: В датасете присутствуют категориальные признаки. Они могут требовать кодирования перед модельным анализом.")
else:
    print("Категориальные признаки отсутствуют.")




# Проверка на пропущенные значения
print("\nПропущенные значения:")
print(df.isnull().sum())

# 3. Нормализация данных
print("\n[3] Нормализация данных...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop(columns=["id","target"],errors='ignore'))
print("Данные успешно нормализованы!\n")

# 4. Снижение размерности - Kernel PCA с оценкой качества
print("[4] Применение Kernel PCA с оценкой качества...")
kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']
kpca_results = {}

for kernel in kernels:
    print(f"\nKernel: {kernel}", end=', ')
    kpca = KernelPCA(n_components=2, kernel=kernel)
    X_kpca = kpca.fit_transform(X_scaled)
    kpca_results[kernel] = X_kpca

    # Silhouette Score
    kmeans_tmp = KMeans(n_clusters=2, random_state=42)
    labels_tmp = kmeans_tmp.fit_predict(X_kpca)
    sil_score = silhouette_score(X_kpca, labels_tmp)

    # Среднее расстояние между центрами кластеров
    centroids = []
    for label in np.unique(labels_tmp):
        cluster_points = X_kpca[labels_tmp == label]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)
    avg_distance = np.linalg.norm(centroids[0] - centroids[1])

    print(f"Silhouette Score: {sil_score:.6f}, Avg Distance Between Classes: {avg_distance:.6f}")

    # Визуализация
    plt.scatter(X_kpca[:, 0], X_kpca[:, 1], alpha=0.5)
    plt.title(f'Kernel PCA с ядром: {kernel}')
    plt.show()


# 5. Оценка дисперсии: Original, Transformed и Lost
print("\n[5] Оценка дисперсии: Original, Transformed и Lost")
original_variance = np.sum(np.var(X_scaled, axis=0))
transformed_variance = np.sum(np.var(kpca_results['linear'], axis=0))  # линейное ядро как приближённый PCA
lost_variance = original_variance - transformed_variance

print(f"Original Variance: {original_variance}")
print(f"Transformed Variance: {transformed_variance}")
print(f"Lost Variance: {lost_variance}")


# 6. Сравнение с t-SNE
print("\n[6] Применение t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.5)
plt.title('Визуализация t-SNE')
plt.show()
print("t-SNE завершён!\n")
