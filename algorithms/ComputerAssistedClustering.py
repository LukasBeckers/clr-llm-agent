from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Union
import copy


class ComputerAssistedClustering:
    def __init__(
        self,
        algorithm: str = "kmeans",
        num_clusters: int = 10,
        random_state: int = 42,
        **kwargs: Any
    ) -> None:
        """
        Initializes the CAC model with a specified clustering algorithm.

        :param algorithm: Clustering algorithm to use ('kmeans', 'agglomerative', 'dbscan')
        :param num_clusters: Number of clusters (for algorithms that require it)
        :param random_state: Seed for reproducibility
        :param kwargs: Additional parameters for the clustering algorithm
        """
        self.algorithm: str = algorithm
        self.num_clusters: int = num_clusters
        self.random_state: int = random_state
        self.kwargs: Dict[str, Any] = kwargs
        self.vectorizer: TfidfVectorizer = TfidfVectorizer()
        self.model: Optional[Union[KMeans, AgglomerativeClustering, DBSCAN]] = None
        self.labels: Optional[List[int]] = None
        self.features: Optional[Any] = None

    def fit(self, documents: List[Dict[str, Any]]) -> None:
        """
        Fits the clustering model to the documents.

        :param documents: List of document dictionaries.
        """
        # Create a deep copy to ensure original documents are not modified
        documents_copy = copy.deepcopy(documents)

        # Extract "Abstract Normalized" from each document
        abstracts: List[str] = []
        for doc in documents_copy:
            abstract_normalized = doc.get("Abstract Normalized")
            if not isinstance(abstract_normalized, list) or not all(
                isinstance(token, str) for token in abstract_normalized
            ):
                raise ValueError(
                    "Each document must contain an 'Abstract Normalized' field as a list of strings."
                )
            # Join tokens to form a single string for vectorization
            abstract_str = " ".join(abstract_normalized)
            abstracts.append(abstract_str)

        # Vectorize the abstracts
        self.features = self.vectorizer.fit_transform(abstracts)

        # Initialize the clustering model based on the selected algorithm
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

        # Fit the model and obtain cluster labels
        self.labels = self.model.fit_predict(self.features)

    def get_clusters(self) -> Optional[List[int]]:
        """
        Retrieves the cluster labels for each document.

        :return: List of cluster labels or None if not fitted.
        """
        return self.labels

    def visualize_clusters(self, top_n_components: int = 2) -> None:
        """
        Visualizes the clusters using PCA for dimensionality reduction.

        :param top_n_components: Number of principal components for visualization
        """
        if self.features is None or self.labels is None:
            raise ValueError("The model has not been fitted yet.")

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
            alpha=0.6,
            edgecolors="w",
            linewidth=0.5,
        )
        plt.legend(*scatter.legend_elements(), title="Clusters")
        plt.title(f"Clusters Visualized with {self.algorithm.capitalize()}")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.grid(True)
        plt.show()

    def __call__(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Makes the ComputerAssistedClustering instance callable. Performs clustering on the provided documents.

        :param documents: List of document dictionaries.
        :return: Dictionary containing clustering results.
        """
        self.fit(documents)
        clusters = self.get_clusters()
        result: Dict[str, Any] = {
            "algorithm": self.algorithm,
            "num_clusters": (
                self.num_clusters
                if self.algorithm in ["kmeans", "agglomerative"]
                else "N/A"
            ),
            "labels": clusters if clusters is None else list(clusters),
            "additional_parameters": self.kwargs,
        }
        return result
