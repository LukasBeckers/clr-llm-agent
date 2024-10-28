import tomotopy as tp
import gensim
from gensim import corpora
from gensim.models import LdaModel

class CorrelatedTopicModel:
    def __init__(self, num_topics=10, alpha='symmetric', eta='auto', seed=42):
        """
        Initializes the CTM model using tomotopy.

        :param num_topics: Number of topics to extract
        :param alpha: Hyperparameter for topic distribution
        :param eta: Hyperparameter for word distribution
        :param seed: Seed for reproducibility
        """
        self.num_topics = num_topics
        self.alpha = alpha
        self.eta = eta
        self.seed = seed
        self.model = tp.CTModel(k=self.num_topics, alpha=self.alpha, eta=self.eta, seed=self.seed)
        self.dictionary = None

    def fit(self, documents, iterations=1000):
        """
        Fits the CTM model to the documents.

        :param documents: List of preprocessed documents (list of tokens)
        :param iterations: Number of training iterations
        """
        self.model.reset()
        for doc in documents:
            self.model.add_doc(doc)
        self.model.train(iterations)
    
    def get_topics(self, top_n=10):
        """
        Retrieves the topics.

        :param top_n: Number of top words per topic
        :return: List of topics with top words
        """
        topics = []
        for i in range(self.num_topics):
            words = self.model.get_topic_words(i, top_n)
            topics.append((i, [word for word, _ in words]))
        return topics
    
    def get_document_topics(self, document, top_n=10):
        """
        Gets the topic distribution for a single document.

        :param document: Preprocessed document (list of tokens)
        :param top_n: Number of top topics to return
        :return: List of (topic_id, probability) tuples
        """
        doc = self.model.make_doc(document)
        self.model.add_doc(doc)
        self.model.train(0)  # Infer without training
        topic_dist = doc.get_topic_dist()
        top_topics = sorted(topic_dist, key=lambda x: x[1], reverse=True)[:top_n]
        return top_topics

