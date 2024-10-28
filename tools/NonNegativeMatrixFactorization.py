from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


class NonNegativeMatrixFactorization:
    def __init__(self, num_topics=10, random_state=42, max_iter=200):
        """
        Initializes the NMF model.

        :param num_topics: Number of topics to extract
        :param random_state: Seed for reproducibility
        :param max_iter: Maximum number of iterations during training
        """
        self.num_topics = num_topics
        self.random_state = random_state
        self.max_iter = max_iter
        self.vectorizer = TfidfVectorizer()
        self.model = NMF(n_components=self.num_topics,
                         random_state=self.random_state,
                         max_iter=self.max_iter)
        self.feature_names = None

    def fit(self, documents):
        """
        Fits the NMF model to the documents.

        :param documents: List of preprocessed documents (list of tokens as strings)
        """
        # Join tokens back to strings for TfidfVectorizer
        joined_docs = [' '.join(doc) for doc in documents]
        tfidf = self.vectorizer.fit_transform(joined_docs)
        self.model.fit(tfidf)
        self.feature_names = self.vectorizer.get_feature_names_out()
    
    def get_topics(self, top_n=10):
        """
        Retrieves the topics.

        :param top_n: Number of top words per topic
        :return: List of topics with top words
        """
        topics = []
        for idx, topic in enumerate(self.model.components_):
            top_features = [self.feature_names[i] for i in topic.argsort()[:-top_n - 1:-1]]
            topics.append((idx, top_features))
        return topics
    
    def get_document_topics(self, document):
        """
        Gets the topic distribution for a single document.

        :param document: Preprocessed document (list of tokens)
        :return: List of (topic_id, weight) tuples
        """
        joined_doc = ' '.join(document)
        tfidf = self.vectorizer.transform([joined_doc])
        topic_distribution = self.model.transform(tfidf)[0]
        return list(enumerate(topic_distribution))
