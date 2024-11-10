from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class ComputerAssistedClustering:
    def __init__(
        self, algorithm="kmeans", num_clusters=10, random_state=42, **kwargs
    ):
        """
        Initializes the CAC model with a specified clustering algorithm.

        :param algorithm: Clustering algorithm to use ('kmeans', 'agglomerative', 'dbscan')
        :param num_clusters: Number of clusters (for algorithms that require it)
        :param random_state: Seed for reproducibility
        :param kwargs: Additional parameters for the clustering algorithm
        """
        self.algorithm = algorithm
        self.num_clusters = num_clusters
        self.random_state = random_state
        self.kwargs = kwargs
        self.vectorizer = TfidfVectorizer()
        self.model = None
        self.labels = None
        self.features = None

    def fit(self, documents):
        """
        Fits the clustering model to the documents.

        :param documents: List of preprocessed documents (list of tokens as strings)
        """
        self.features = self.vectorizer.fit_transform(documents)
        if self.algorithm == "kmeans":
            self.model = KMeans(
                n_clusters=self.num_clusters,
                random_state=self.random_state,
                **self.kwargs,
            )
        elif self.algorithm == "agglomerative":
            self.model = AgglomerativeClustering(
                n_clusters=self.num_clusters, **self.kwargs
            )
        elif self.algorithm == "dbscan":
            self.model = DBSCAN(**self.kwargs)
        else:
            raise ValueError(
                "Unsupported algorithm. Choose from 'kmeans', 'agglomerative', 'dbscan'."
            )
        self.labels = self.model.fit_predict(self.features)

    def get_clusters(self):
        """
        Retrieves the cluster labels for each document.

        :return: Array of cluster labels
        """
        return self.labels

    def visualize_clusters(self, top_n_components=2):
        """
        Visualizes the clusters using PCA for dimensionality reduction.

        :param top_n_components: Number of principal components for visualization
        """
        pca = PCA(
            n_components=top_n_components, random_state=self.random_state
        )
        reduced_features = pca.fit_transform(self.features.toarray())
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(
            reduced_features[:, 0],
            reduced_features[:, 1],
            c=self.labels,
            cmap="viridis",
        )
        plt.legend(*scatter.legend_elements(), title="Clusters")
        plt.title(f"Clusters Visualized with {self.algorithm.capitalize()}")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.show()

    def __call__(self, documents):
        """
        Makes the ComputerAssistedClustering instance callable. Performs clustering on the provided documents.

        :param documents: List of preprocessed documents (list of tokens as strings)
        :return: Dictionary containing clustering results
        """
        self.fit(documents)
        clusters = self.get_clusters()
        result = {
            "algorithm": self.algorithm,
            "num_clusters": (
                self.num_clusters
                if self.algorithm in ["kmeans", "agglomerative"]
                else "N/A"
            ),
            "labels": (
                clusters.tolist() if hasattr(clusters, "tolist") else clusters
            ),
            "additional_parameters": self.kwargs,
        }
        return result
