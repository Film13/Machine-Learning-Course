import matplotlib.pyplot as plt
from data_loader import load_and_explore_data
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def run_clustering(data):
    X = data[['Age', 'Size', 'Breed']]

    # Encoding และ Scaling สำหรับ Clustering
    X_encoded = OneHotEncoder(handle_unknown='ignore').fit_transform(X).toarray()
    X_scaled = StandardScaler().fit_transform(X_encoded)

    # K-Means
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # PCA ลดมิติเพื่อ plot กราฟ 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)
    plt.title('K-Means Clustering of Cats Dataset')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig('clustering_output.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    dataset = load_and_explore_data()
    dataset = dataset[['Age', 'Size', 'Breed']].dropna()
    run_clustering(dataset)