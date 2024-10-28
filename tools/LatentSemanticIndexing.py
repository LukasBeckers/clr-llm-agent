from gensim.models import LsiModel
import gensim
from gensim import corpora
from gensim.models import LdaModel


class LatentSemanticIndexing:
    def __init__(self, num_topics=10):
        """
        Initializes the LSI model.

        :param num_topics: Number of topics to extract
        """
        self.num_topics = num_topics
        self.dictionary = None
        self.corpus = None
        self.model = None

    def fit(self, documents):
        """
        Fits the LSI model to the documents.

        :param documents: List of preprocessed documents (list of tokens)
        """
        self.dictionary = corpora.Dictionary(documents)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in documents]
        self.model = LsiModel(corpus=self.corpus,
                              id2word=self.dictionary,
                              num_topics=self.num_topics)
    
    def get_topics(self):
        """
        Retrieves the topics.

        :return: List of topics with top words
        """
        return self.model.print_topics(num_words=10)
    
    def get_document_topics(self, document):
        """
        Gets the topic distribution for a single document.

        :param document: Preprocessed document (list of tokens)
        :return: List of (topic_id, probability) tuples
        """
        bow = self.dictionary.doc2bow(document)
        return self.model[bow]
