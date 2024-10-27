class ProbabilisticLatentSemanticAnalysis:
    def __init__(self, num_topics=10, passes=10, random_state=42):
        """
        Initializes the PLSA model.

        :param num_topics: Number of topics to extract
        :param passes: Number of passes through the corpus during training
        :param random_state: Seed for reproducibility
        """
        self.num_topics = num_topics
        self.passes = passes
        self.random_state = random_state
        self.dictionary = None
        self.corpus = None
        self.model = None

    def fit(self, documents):
        """
        Fits the PLSA model to the documents.

        :param documents: List of preprocessed documents (list of tokens)
        """
        self.dictionary = corpora.Dictionary(documents)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in documents]
        # In Gensim, setting alpha='asymmetric' makes it behave more like PLSA
        self.model = LdaModel(corpus=self.corpus,
                              id2word=self.dictionary,
                              num_topics=self.num_topics,
                              passes=self.passes,
                              random_state=self.random_state,
                              alpha='asymmetric')
    
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
        return self.model.get_document_topics(bow)
