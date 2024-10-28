from gensim.models import HdpModel
import gensim
from gensim import corpora
from gensim.models import LdaModel


class HierarchicalDirichletProcess:
    def __init__(self, random_state=42):
        """
        Initializes the HDP model.

        :param random_state: Seed for reproducibility
        """
        self.random_state = random_state
        self.dictionary = None
        self.corpus = None
        self.model = None

    def fit(self, documents):
        """
        Fits the HDP model to the documents.

        :param documents: List of preprocessed documents (list of tokens)
        """
        self.dictionary = corpora.Dictionary(documents)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in documents]
        self.model = HdpModel(corpus=self.corpus,
                              id2word=self.dictionary,
                              random_state=self.random_state)
    
    def get_topics(self, top_n=10):
        """
        Retrieves the topics.

        :param top_n: Number of top words per topic
        :return: List of topics with top words
        """
        return self.model.print_topics(num_topics=-1, num_words=top_n)
    
    def get_document_topics(self, document):
        """
        Gets the topic distribution for a single document.

        :param document: Preprocessed document (list of tokens)
        :return: List of (topic_id, probability) tuples
        """
        bow = self.dictionary.doc2bow(document)
        return self.model[bow]
