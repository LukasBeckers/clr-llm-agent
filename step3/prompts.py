algoritm_documentation = """
Unsupervised Techniques
1. Latent Dirichlet Allocation (LDA)

Type: Unsupervised Topic Modeling

Description:
LDA is a generative probabilistic model that assumes documents are mixtures of topics, where each topic is characterized by a distribution of words. It identifies latent topics within a corpus by analyzing the co-occurrence patterns of words across documents.

When to Use LDA:

    When you need to discover hidden thematic structures in a large collection of documents.
    When performing exploratory analysis to understand the main topics within a corpus.
    When summarizing large datasets by grouping similar documents based on topic distribution.

Types of Questions LDA Can Investigate:

    What are the predominant themes in a set of research articles on innovation?
    How do topics evolve over time within a specific academic field?
    Which topics are most associated with high-impact journals?

Example Use Case:

    Analyzing abstracts of scientific papers to identify key research areas and their interrelationships.

2. Latent Semantic Indexing/Analysis (LSI/LSA)

Type: Unsupervised Topic Modeling and Dimensionality Reduction

Description:
LSI/LSA reduces the dimensionality of the term-document matrix using Singular Value Decomposition (SVD) to uncover the latent semantic structure in the data. It captures the underlying relationships between terms and documents by identifying patterns in the usage of words.

When to Use LSI/LSA:

    When you aim to improve information retrieval by understanding the semantic relationships between terms.
    When seeking to reduce noise and dimensionality in text data for better visualization or further analysis.
    When addressing synonymy and polysemy issues in text corpora.

Types of Questions LSI/LSA Can Investigate:

    How are different research topics semantically related within a field?
    What are the core concepts that unify a diverse set of documents?
    Can we identify subfields or specialized areas within a broader discipline?

Example Use Case:

    Organizing a collection of business articles into coherent topics for easier navigation and analysis.

3. Probabilistic Latent Semantic Analysis (PLSA)

Type: Unsupervised Topic Modeling

Description:
PLSA is a statistical technique similar to LDA that models the relationship between documents and words by associating each word with a latent topic. Unlike LDA, PLSA does not assume a Dirichlet prior for the distribution of topics in documents.

When to Use PLSA:

    When you require a probabilistic framework to model topics without the need for prior distribution assumptions.
    When working with smaller datasets where the flexibility of PLSA can capture nuanced topic structures.
    When conducting detailed semantic analysis where topic dependencies are of interest.

Types of Questions PLSA Can Investigate:

    What probabilistic topics emerge from a collection of legal documents?
    How do specific topics correlate with certain types of legal cases?
    Can PLSA reveal complex thematic relationships in a specialized corpus?

Example Use Case:

    Identifying legal themes and their prevalence across different case types in a law review database.

4. Non-negative Matrix Factorization (NMF)

Type: Unsupervised Topic Modeling and Dimensionality Reduction

Description:
NMF factorizes the term-document matrix into two non-negative matrices, representing topics and their associated words. This additive, part-based representation emphasizes the presence of features without allowing for negative values, making the topics more interpretable.

When to Use NMF:

    When interpretability of topics is a priority, as NMF tends to produce more coherent and distinct topics.
    When dealing with data where the assumption of additive combinations of topics is valid.
    When requiring a method that naturally enforces non-negativity constraints, which can be beneficial for certain types of data.

Types of Questions NMF Can Investigate:

    What clear and distinct topics can be extracted from a set of marketing reports?
    How can we decompose customer feedback into actionable thematic areas?
    Which topics are most strongly associated with customer satisfaction in service reviews?

Example Use Case:

    Analyzing customer reviews to extract key service attributes and sentiment-related topics for improving service quality.

5. Correlated Topic Models (CTM)

Type: Unsupervised Topic Modeling

Description:
CTM extends LDA by allowing topics to be correlated rather than assuming independence. It captures the co-occurrence of topics within documents, providing a more nuanced understanding of topic relationships.

When to Use CTM:

    When you suspect that topics are not independent and may influence each other within documents.
    When analyzing complex corpora where thematic overlap is common.
    When needing a more sophisticated model to capture topic dependencies and correlations.

Types of Questions CTM Can Investigate:

    How do topics related to technology and innovation interact within research papers?
    What are the overlapping areas between sustainability and corporate governance in business reports?
    Can CTM identify interdependent themes in interdisciplinary studies?

Example Use Case:

    Exploring how topics such as "sustainability" and "innovation" are interrelated in corporate sustainability reports.

6. Hierarchical Dirichlet Processes (HDP)

Type: Unsupervised Topic Modeling

Description:
HDP is a nonparametric Bayesian approach that allows the number of topics to be determined automatically from the data. It extends Dirichlet Processes to model multiple groups of data, sharing topics across them.

When to Use HDP:

    When the optimal number of topics is unknown and needs to be inferred from the data.
    When working with multiple related corpora that may share some common topics.
    When flexibility in the number of topics is required to accommodate evolving thematic structures.

Types of Questions HDP Can Investigate:

    What is the natural number of topics present in a diverse set of scientific journals?
    How do topics distribute across different subsets of a large corpus without predefining the number?
    Can HDP uncover emerging themes in a rapidly growing field of study?

Example Use Case:

    Analyzing a large collection of news articles from different years to identify how topics have emerged and evolved over time without specifying the number of topics in advance.

7. Computer-Assisted Clustering (CAC)

Type: Unsupervised Clustering

Description:
CAC encompasses a variety of clustering algorithms that group documents based on thematic similarity. It involves selecting and applying different clustering techniques to identify meaningful categorizations within the data.

When to Use CAC:

    When seeking to explore different ways to cluster documents and determine the most insightful categorization.
    When requiring flexibility to compare outcomes from multiple clustering algorithms.
    When visualizing the thematic structure of a corpus through different clustering perspectives.

Types of Questions CAC Can Investigate:

    What are the distinct clusters of research topics within a set of academic papers?
    How do different clustering algorithms affect the grouping of policy documents?
    Can CAC reveal unexpected thematic groupings in a collection of literature reviews?

Example Use Case:

    Using CAC to identify and compare clusters of innovation strategies across various business case studies, determining which algorithm best captures the underlying themes.

Dictionary-Based Techniques
1. Linguistic Inquiry and Word Count (LIWC)

Type: Dictionary-Based Text Analysis

Description:
LIWC is a text analysis tool that quantifies various linguistic and psychological attributes by counting the frequency of words in predefined categories. It assesses emotional, cognitive, structural, and process components of texts based on its proprietary dictionary.

When to Use LIWC:

    When analyzing the emotional tone or psychological states expressed in a text.
    When needing to quantify specific linguistic features such as pronouns, articles, or certain word categories.
    When studying the relationship between language use and psychological or social phenomena.

Types of Questions LIWC Can Investigate:

    What is the emotional sentiment expressed in a series of corporate communications?
    How do linguistic patterns correlate with leadership styles in organizational reports?
    Can LIWC identify shifts in cognitive complexity over time in policy documents?

Example Use Case:

    Evaluating the emotional content of employee feedback to assess workplace morale and identify areas needing improvement.

Note:
While LIWC is a proprietary tool, similar functionality can be achieved using open-source libraries by defining custom dictionaries tailored to specific analysis needs.
2. Coh-Metrix

Type: Dictionary-Based Text Analysis

Description:
Coh-Metrix is a comprehensive tool that assesses the linguistic and discourse properties of texts to measure cohesion, readability, and other textual characteristics. It provides a range of metrics that reflect the structural and semantic aspects of the discourse.

When to Use Coh-Metrix:

    When needing an in-depth analysis of text cohesion and readability.
    When studying the structural properties of texts, such as syntactic complexity or discourse coherence.
    When examining the cognitive demands placed on readers by a text.

Types of Questions Coh-Metrix Can Investigate:

    How cohesive are scientific articles in a particular field?
    What are the readability levels of different sections within policy documents?
    Can Coh-Metrix metrics predict the comprehensibility of training materials?

Example Use Case:

    Analyzing educational materials to ensure they meet the appropriate readability standards for the target student population.

Note:
Coh-Metrix is also a proprietary tool. However, some of its functionalities can be approximated using open-source libraries such as nltk and textstat by focusing on specific metrics like readability scores and syntactic complexity.
Additional Analytical Techniques
1. N-grams

Type: Statistical Text Analysis

Description:
N-grams are contiguous sequences of 'n' items (typically words) from a given text. They capture local word order information and can be used to identify common phrases, compound words, or contextual patterns within a corpus.

When to Use N-grams:

    When analyzing the frequency and patterns of specific word sequences.
    When identifying commonly used phrases or collocations in a text.
    When enhancing topic models or clustering algorithms with contextual information.

Types of Questions N-grams Can Investigate:

    What are the most frequent bi-grams or tri-grams in a set of customer reviews?
    How do specific phrases correlate with customer satisfaction levels?
    Can N-grams help in distinguishing between different thematic areas within a corpus?

Example Use Case:

    Extracting and analyzing frequently occurring phrases in social media posts to understand public opinion on a topic.

2. Network Analysis

Type: Graph-Based Text Analysis

Description:
Network Analysis involves constructing and analyzing networks (graphs) of entities (such as words, topics, or documents) and their relationships. It visualizes and quantifies the interconnections and dependencies within the data.

When to Use Network Analysis:

    When exploring the relationships and interactions between different entities in a corpus.
    When needing to visualize the structure and density of connections within topics or terms.
    When identifying central or influential nodes within a network, such as key topics or terms.

Types of Questions Network Analysis Can Investigate:

    How are different research topics interconnected within a field of study?
    What are the central terms that bridge multiple topics in scientific literature?
    Can network metrics predict the influence or importance of certain topics?

Example Use Case:

    Mapping the relationships between key concepts in organizational behavior research to identify central themes and their interdependencies.

3. Regression Analysis

Type: Statistical Text Analysis

Description:
Regression Analysis models the relationship between a dependent variable and one or more independent variables. In text analysis, it can be used to explore how textual features predict certain outcomes or trends.

When to Use Regression Analysis:

    When assessing the impact of specific textual attributes on an outcome of interest.
    When modeling trends over time based on textual data.
    When quantifying the relationship between linguistic features and external variables.

Types of Questions Regression Analysis Can Investigate:

    How do changes in topic prevalence correlate with research funding over time?
    Can the frequency of certain keywords predict the citation counts of academic papers?
    What is the relationship between readability scores and student performance in educational texts?

Example Use Case:

    Analyzing how the prevalence of innovation-related terms in company reports predicts stock performance.

4. Heatmaps

Type: Data Visualization

Description:
Heatmaps are graphical representations of data where individual values are represented as colors. In text analysis, they are used to visualize the distribution and intensity of topics, terms, or other metrics across different dimensions.

When to Use Heatmaps:

    When needing to visualize the temporal or spatial distribution of topics or terms.
    When identifying hotspots or gaps within a corpus based on specific metrics.
    When presenting complex data in an easily interpretable visual format.

Types of Questions Heatmaps Can Investigate:

    Where are the research hotspots within a particular field over the past decade?
    How does topic prevalence vary across different journals or disciplines?
    Can heatmaps reveal patterns of topic emergence and decline over time?

Example Use Case:

    Creating a heatmap to display the frequency of emerging technologies across various industry sectors over the last five years.

5. Automated Frame Analysis (AFA)

Type: Discourse Analysis

Description:
AFA involves identifying and categorizing the framing of content within texts. It analyzes how information is presented and structured to uncover underlying narratives and perspectives.

When to Use AFA:

    When studying the framing strategies used in media or policy documents.
    When analyzing how different stakeholders present the same issue from varying perspectives.
    When needing to quantify and compare narrative structures across a corpus.

Types of Questions AFA Can Investigate:

    How do different news outlets frame the topic of climate change?
    What are the predominant frames used in corporate sustainability reports?
    Can AFA identify shifts in narrative strategies over time in political speeches?

Example Use Case:

    Examining how various governmental agencies frame economic policies in official statements to understand differing policy narratives.

6. Automated Narrative Analysis (ANA)

Type: Discourse Analysis

Description:
ANA involves dissecting the narrative elements within texts, such as actors, actions, and their positions. It helps in understanding the storytelling and discourse structures present in the data.

When to Use ANA:

    When analyzing the structure and components of narratives in storytelling contexts.
    When studying the roles and actions of entities within a set of documents.
    When needing to extract and quantify narrative elements for further analysis.

Types of Questions ANA Can Investigate:

    Who are the main actors and what actions do they perform in organizational case studies?
    How do narratives in customer testimonials differ between satisfied and dissatisfied customers?
    Can ANA reveal common story arcs in fiction literature?

Example Use Case:

    Analyzing corporate mission statements to identify the roles and actions companies prioritize in their narratives.

Summary

This documentation provides a structured overview of each algorithm and technique, detailing their types, descriptions, appropriate use cases, and the kinds of questions they are best suited to investigate. By leveraging these guidelines, an LLM agent can effectively select and apply the most suitable algorithm for various text analysis tasks within the context of systematic literature reviews or other document corpus analyses.
Best Practices for Selecting Algorithms

    Understand Your Objective:
    Clearly define what you aim to achieve with your analysis. Whether it's discovering hidden topics, analyzing emotional sentiment, or understanding narrative structures, your objective will guide your algorithm choice.

    Consider Data Characteristics:
    Assess the size, complexity, and nature of your text corpus. Some algorithms perform better with larger datasets (e.g., LDA, NMF), while others are suitable for smaller or more specialized datasets (e.g., PLSA).

    Evaluate Interpretability Needs:
    If interpretability is crucial, algorithms like NMF or LDA might be preferable due to their ability to produce coherent and distinct topics.

    Assess Technical Requirements:
    Consider the computational resources and technical expertise required. Tools like LIWC and Coh-Metrix offer user-friendly interfaces, whereas models like CTM and HDP may require more advanced programming and computational power.

    Iterative Testing and Validation:
    Experiment with multiple algorithms to determine which best captures the thematic structures relevant to your research questions. Validate the results through domain expert review or by comparing with known benchmarks.

    Combine Techniques for Enhanced Insights:
    Often, leveraging multiple algorithms in tandem (e.g., using topic modeling followed by network analysis) can provide deeper and more comprehensive insights.

Final Notes

    Proprietary Tools:
    Tools like LIWC and Coh-Metrix offer advanced features but require licenses. Open-source alternatives can be customized to approximate some of their functionalities.

    Extensibility:
    The provided Python classes serve as foundational implementations. Depending on specific research needs, these can be extended with additional features such as model persistence, enhanced validation methods, or integration with visualization libraries.

    Interdisciplinary Collaboration:
    Combining expertise in computational methods with domain-specific knowledge enhances the effectiveness and interpretability of the analysis.

By adhering to these guidelines and leveraging the appropriate algorithms, researchers can conduct robust and insightful analyses of large and complex document corpora, thereby advancing knowledge and informing decision-making in various fields.
"""


fine_grained_algorithm_description = """
Unsupervised Techniques
1. Latent Dirichlet Allocation (LDA)

Type: Unsupervised Topic Modeling

Description:
LDA is a generative probabilistic model that assumes documents are mixtures of topics, where each topic is characterized by a distribution of words. It identifies latent topics within a corpus by analyzing the co-occurrence patterns of words across documents.

When to Use LDA:

    General Use:
        When you need to discover hidden thematic structures in a large collection of documents.
        When performing exploratory analysis to understand the main topics within a corpus.
        When summarizing large datasets by grouping similar documents based on topic distribution.
    Special/Extended Use:
        Dynamic Topic Modeling (DTM):
            Description: Extends LDA to capture the evolution of topics over time by splitting data into temporal slices and allowing topics to change.
            When to Use:
                When investigating how topics evolve over time within a specific academic field.
                (Example: How do topics evolve over time within a specific academic field? (Dynamic Topic Modeling))

Types of Questions LDA Can Investigate:

    General:
        What are the predominant themes in a set of research articles on innovation?
        Which topics are most associated with high-impact journals?
    Special/Extended Use:
        How do topics evolve over time within a specific academic field?
        (Dynamic Topic Modeling)
        What are the emerging trends in scientific literature over the past decade?
        (Dynamic Topic Modeling)

Example Use Case:

    General:
        Analyzing abstracts of scientific papers to identify key research areas and their interrelationships.
    Special/Extended Use:
        Tracking the rise and fall of specific research topics over multiple years to understand the development trajectory of a field.
        (Dynamic Topic Modeling)

2. Latent Semantic Indexing/Analysis (LSI/LSA)

Type: Unsupervised Topic Modeling and Dimensionality Reduction

Description:
LSI/LSA reduces the dimensionality of the term-document matrix using Singular Value Decomposition (SVD) to uncover the latent semantic structure in the data. It captures the underlying relationships between terms and documents by identifying patterns in the usage of words.

When to Use LSI/LSA:

    General Use:
        When you aim to improve information retrieval by understanding the semantic relationships between terms.
        When seeking to reduce noise and dimensionality in text data for better visualization or further analysis.
        When addressing synonymy and polysemy issues in text corpora.
    Special/Extended Use:
        Temporal LSI (TLSI):
            Description: Applies LSI in a temporal context to analyze how the semantic structure changes over different time periods.
            When to Use:
                When investigating the semantic evolution of topics across different time frames.
                (Example: How do semantic relationships between research topics change over time? (Temporal LSI))

Types of Questions LSI/LSA Can Investigate:

    General:
        How are different research topics semantically related within a field?
        Can we identify subfields or specialized areas within a broader discipline?
    Special/Extended Use:
        How do semantic relationships between research topics change over time?
        (Temporal LSI)
        What are the shifts in conceptual structures within a corpus across different periods?
        (Temporal LSI)

Example Use Case:

    General:
        Organizing a collection of business articles into coherent topics for easier navigation and analysis.
    Special/Extended Use:
        Analyzing a corpus of historical texts to understand how the meaning and association of key terms have evolved.
        (Temporal LSI)

3. Probabilistic Latent Semantic Analysis (PLSA)

Type: Unsupervised Topic Modeling

Description:
PLSA is a statistical technique similar to LDA that models the relationship between documents and words by associating each word with a latent topic. Unlike LDA, PLSA does not assume a Dirichlet prior for the distribution of topics in documents.

When to Use PLSA:

    General Use:
        When you require a probabilistic framework to model topics without the need for prior distribution assumptions.
        When working with smaller datasets where the flexibility of PLSA can capture nuanced topic structures.
        When conducting detailed semantic analysis where topic dependencies are of interest.
    Special/Extended Use:
        Hierarchical PLSA (HPLSA):
            Description: Extends PLSA to incorporate a hierarchy of topics, allowing for more complex topic relationships.
            When to Use:
                When exploring hierarchical relationships between topics in a corpus.
                (Example: What hierarchical structures exist among topics in corporate reports? (Hierarchical PLSA))

Types of Questions PLSA Can Investigate:

    General:
        What probabilistic topics emerge from a collection of legal documents?
        How do specific topics correlate with certain types of legal cases?
    Special/Extended Use:
        What hierarchical structures exist among topics in corporate reports?
        (Hierarchical PLSA)
        How do subtopics organize within broader thematic categories in academic literature?
        (Hierarchical PLSA)

Example Use Case:

    General:
        Identifying legal themes and their prevalence across different case types in a law review database.
    Special/Extended Use:
        Exploring nested thematic structures within a set of policy documents to understand overarching and subsidiary themes.
        (Hierarchical PLSA)

4. Non-negative Matrix Factorization (NMF)

Type: Unsupervised Topic Modeling and Dimensionality Reduction

Description:
NMF factorizes the term-document matrix into two non-negative matrices, representing topics and their associated words. This additive, part-based representation emphasizes the presence of features without allowing for negative values, making the topics more interpretable.

When to Use NMF:

    General Use:
        When interpretability of topics is a priority, as NMF tends to produce more coherent and distinct topics.
        When dealing with data where the assumption of additive combinations of topics is valid.
        When requiring a method that naturally enforces non-negativity constraints, which can be beneficial for certain types of data.
    Special/Extended Use:
        Hierarchical NMF (HNMF):
            Description: Extends NMF to uncover hierarchical topic structures, allowing for nested topics within broader categories.
            When to Use:
                When investigating hierarchical relationships between topics in a corpus.
                (Example: How are specific innovation strategies nested within broader business practices? (Hierarchical NMF))
        Temporal NMF (TNMF):
            Description: Applies NMF across different time slices to analyze the temporal evolution of topics.
            When to Use:
                When examining how topics change and develop over time within a corpus.
                (Example: How do innovation topics shift over the past decade in technology journals? (Temporal NMF))

Types of Questions NMF Can Investigate:

    General:
        What clear and distinct topics can be extracted from a set of marketing reports?
        How can we decompose customer feedback into actionable thematic areas?
    Special/Extended Use:
        How are specific innovation strategies nested within broader business practices?
        (Hierarchical NMF)
        How do innovation topics shift over the past decade in technology journals?
        (Temporal NMF)

Example Use Case:

    General:
        Analyzing customer reviews to extract key service attributes and sentiment-related topics for improving service quality.
    Special/Extended Use:
        Tracking the development and transformation of specific business strategies over multiple years to inform strategic planning.
        (Temporal NMF)

5. Correlated Topic Models (CTM)

Type: Unsupervised Topic Modeling

Description:
CTM extends LDA by allowing topics to be correlated rather than assuming independence. It captures the co-occurrence of topics within documents, providing a more nuanced understanding of topic relationships.

When to Use CTM:

    General Use:
        When you suspect that topics are not independent and may influence each other within documents.
        When analyzing complex corpora where thematic overlap is common.
        When needing a more sophisticated model to capture topic dependencies and correlations.
    Special/Extended Use:
        Dynamic Correlated Topic Models (DCTM):
            Description: Combines CTM with dynamic modeling to capture both topic correlations and their evolution over time.
            When to Use:
                When investigating how topic correlations change over time within a corpus.
                (Example: How do the relationships between sustainability and innovation topics evolve over time? (Dynamic Correlated Topic Modeling))

Types of Questions CTM Can Investigate:

    General:
        How do topics related to technology and innovation interact within research papers?
        What are the overlapping areas between sustainability and corporate governance in business reports?
    Special/Extended Use:
        How do the relationships between sustainability and innovation topics evolve over time?
        (Dynamic Correlated Topic Modeling)
        What are the shifting correlations between emerging and established topics in interdisciplinary studies?
        (Dynamic Correlated Topic Modeling)

Example Use Case:

    General:
        Exploring how topics such as "sustainability" and "innovation" are interrelated in corporate sustainability reports.
    Special/Extended Use:
        Analyzing the evolution of topic correlations in academic research to understand changing interdisciplinary influences.
        (Dynamic Correlated Topic Modeling)

6. Hierarchical Dirichlet Processes (HDP)

Type: Unsupervised Topic Modeling

Description:
HDP is a nonparametric Bayesian approach that allows the number of topics to be determined automatically from the data. It extends Dirichlet Processes to model multiple groups of data, sharing topics across them.

When to Use HDP:

    General Use:
        When the optimal number of topics is unknown and needs to be inferred from the data.
        When working with multiple related corpora that may share some common topics.
        When flexibility in the number of topics is required to accommodate evolving thematic structures.
    Special/Extended Use:
        Temporal HDP (THDP):
            Description: Incorporates temporal dynamics into HDP to model how topics evolve over different time periods.
            When to Use:
                When analyzing the evolution of topics across different time slices without predefining the number of topics.
                (Example: How do topics emerge and evolve in news articles over the years? (Temporal HDP))
        Hierarchical HDP (HHDP):
            Description: Extends HDP to include multiple levels of topic hierarchies, allowing for more granular topic categorization.
            When to Use:
                When investigating multi-level topic structures within a large and complex corpus.
                (Example: What are the nested topic hierarchies within corporate strategy documents? (Hierarchical HDP))

Types of Questions HDP Can Investigate:

    General:
        What is the natural number of topics present in a diverse set of scientific journals?
        How do topics distribute across different subsets of a large corpus without predefining the number of topics?
    Special/Extended Use:
        How do topics emerge and evolve in news articles over the years?
        (Temporal HDP)
        What are the nested topic hierarchies within corporate strategy documents?
        (Hierarchical HDP)

Example Use Case:

    General:
        Analyzing a large collection of news articles from different years to identify how topics have emerged and evolved over time without specifying the number of topics in advance.
    Special/Extended Use:
        Exploring multi-level thematic structures in extensive corporate reports to understand broad strategies and their specific components.
        (Hierarchical HDP)

7. Computer-Assisted Clustering (CAC)

Type: Unsupervised Clustering

Description:
CAC encompasses a variety of clustering algorithms that group documents based on thematic similarity. It involves selecting and applying different clustering techniques to identify meaningful categorizations within the data.

When to Use CAC:

    General Use:
        When seeking to explore different ways to cluster documents and determine the most insightful categorization.
        When requiring flexibility to compare outcomes from multiple clustering algorithms.
        When visualizing the thematic structure of a corpus through different clustering perspectives.
    Special/Extended Use:
        Hierarchical Clustering Analysis (HCA):
            Description: Uses hierarchical clustering algorithms to create a tree-like structure of clusters, allowing for multi-level cluster analysis.
            When to Use:
                When investigating nested or multi-level topic structures within a corpus.
                (Example: How are specific innovation strategies nested within broader business practices? (Hierarchical Clustering Analysis))
        Density-Based Spatial Clustering of Applications with Noise (DBSCAN):
            Description: A density-based clustering algorithm that identifies clusters based on the density of data points, effectively handling noise and outliers.
            When to Use:
                When dealing with data that contains noise or outliers and requires the identification of clusters with varying shapes.
                (Example: Can DBSCAN identify non-spherical topic clusters in customer feedback data? (DBSCAN))
        K-Means Clustering:
            Description: A centroid-based clustering algorithm that partitions data into K distinct clusters based on feature similarity.
            When to Use:
                When you have a predefined number of clusters and need to partition the corpus accordingly.
                (Example: How can we group marketing reports into five distinct thematic categories using K-Means? (K-Means Clustering))

Types of Questions CAC Can Investigate:

    General:
        What are the distinct clusters of research topics within a set of academic papers?
        How do different clustering algorithms affect the grouping of policy documents?
    Special/Extended Use:
        How are specific innovation strategies nested within broader business practices?
        (Hierarchical Clustering Analysis)
        Can DBSCAN identify non-spherical topic clusters in customer feedback data?
        (DBSCAN)
        How can we group marketing reports into five distinct thematic categories using K-Means?
        (K-Means Clustering)

Example Use Case:

    General:
        Using CAC to identify and compare clusters of innovation strategies across various business case studies, determining which algorithm best captures the underlying themes.
    Special/Extended Use:
        Applying hierarchical clustering to corporate strategy documents to uncover nested thematic structures that reflect different layers of business strategies.
        (Hierarchical Clustering Analysis)

Dictionary-Based Techniques
1. Linguistic Inquiry and Word Count (LIWC)

Type: Dictionary-Based Text Analysis

Description:
LIWC is a text analysis tool that quantifies various linguistic and psychological attributes by counting the frequency of words in predefined categories. It assesses emotional, cognitive, structural, and process components of texts based on its proprietary dictionary.

When to Use LIWC:

    General Use:
        When analyzing the emotional tone or psychological states expressed in a text.
        When needing to quantify specific linguistic features such as pronouns, articles, or certain word categories.
        When studying the relationship between language use and psychological or social phenomena.
    Special/Extended Use:
        Psycholinguistic Analysis:
            Description: Uses LIWC to delve deeper into the psychological underpinnings of language, analyzing aspects like cognitive complexity, emotional regulation, and social dynamics.
            When to Use:
                When investigating the psychological profiles of authors based on their writing styles.
                (Example: What does the use of emotional language in corporate communications reveal about organizational culture? (Psycholinguistic Analysis))
        Sentiment Dynamics:
            Description: Analyzes changes in sentiment over time within a text corpus.
            When to Use:
                When exploring how sentiment trends evolve across different periods or events.
                (Example: How does the sentiment in financial reports change before and after economic downturns? (Sentiment Dynamics))

Types of Questions LIWC Can Investigate:

    General:
        What is the emotional sentiment expressed in a series of corporate communications?
        How do linguistic patterns correlate with leadership styles in organizational reports?
    Special/Extended Use:
        What does the use of emotional language in corporate communications reveal about organizational culture?
        (Psycholinguistic Analysis)
        How does the sentiment in financial reports change before and after economic downturns?
        (Sentiment Dynamics)

Example Use Case:

    General:
        Evaluating the emotional content of employee feedback to assess workplace morale and identify areas needing improvement.
    Special/Extended Use:
        Analyzing how the use of cognitive and emotional words in leadership speeches correlates with organizational change initiatives.
        (Psycholinguistic Analysis)

Note:
While LIWC is a proprietary tool, similar functionality can be achieved using open-source libraries by defining custom dictionaries tailored to specific analysis needs.

2. Network Analysis

Type: Graph-Based Text Analysis

Description:
Network Analysis involves constructing and analyzing networks (graphs) of entities (such as words, topics, or documents) and their relationships. It visualizes and quantifies the interconnections and dependencies within the data.

When to Use Network Analysis:

    General Use:
        When exploring the relationships and interactions between different entities in a corpus.
        When needing to visualize the structure and density of connections within topics or terms.
        When identifying central or influential nodes within a network, such as key topics or terms.
    Special/Extended Use:
        Temporal Network Analysis:
            Description: Analyzes how the network structure evolves over different time periods.
            When to Use:
                When studying the temporal dynamics of topic relationships and their evolution over time.
                (Example: How do the interconnections between research topics change annually? (Temporal Network Analysis))
        Multiplex Network Analysis:
            Description: Incorporates multiple types of relationships between entities within the same network.
            When to Use:
                When examining different kinds of interactions or dependencies between topics or terms simultaneously.
                (Example: How do co-occurrence and semantic similarity networks of terms overlap in scientific literature? (Multiplex Network Analysis))

Types of Questions Network Analysis Can Investigate:

    General:
        How are different research topics interconnected within a field of study?
        What are the central terms that bridge multiple topics in scientific literature?
    Special/Extended Use:
        How do the interconnections between research topics change annually?
        (Temporal Network Analysis)
        How do co-occurrence and semantic similarity networks of terms overlap in scientific literature?
        (Multiplex Network Analysis)

Example Use Case:

    General:
        Mapping the relationships between key concepts in organizational behavior research to identify central themes and their interdependencies.
    Special/Extended Use:
        Analyzing how the network of interrelated topics in sustainability research evolves in response to global environmental events.
        (Temporal Network Analysis)

6. Automated Narrative Analysis (ANA)

Type: Discourse Analysis

Description:
ANA involves dissecting the narrative elements within texts, such as actors, actions, and their positions. It helps in understanding the storytelling and discourse structures present in the data.

When to Use ANA:

    General Use:
        When analyzing the structure and components of narratives in storytelling contexts.
        When studying the roles and actions of entities within a set of documents.
        When needing to extract and quantify narrative elements for further analysis.
    Special/Extended Use:
        Sentiment-Integrated Narrative Analysis (SINA):
            Description: Combines narrative element extraction with sentiment analysis to assess the emotional tone of narratives.
            When to Use:
                When investigating how emotions are conveyed through narratives in different contexts.
                (Example: How do emotional tones vary within the narratives of crisis management reports? (Sentiment-Integrated Narrative Analysis))
        Role-Based Narrative Analysis (RBNA):
            Description: Focuses on identifying and categorizing the roles of different actors within narratives.
            When to Use:
                When examining the distribution and characteristics of roles (e.g., protagonist, antagonist) in organizational case studies.
                (Example: What roles do different departments play in company success narratives? (Role-Based Narrative Analysis))

Types of Questions ANA Can Investigate:

    General:
        Who are the main actors and what actions do they perform in organizational case studies?
        How do narratives in customer testimonials differ between satisfied and dissatisfied customers?
    Special/Extended Use:
        How do emotional tones vary within the narratives of crisis management reports?
        (Sentiment-Integrated Narrative Analysis)
        What roles do different departments play in company success narratives?
        (Role-Based Narrative Analysis)

Example Use Case:

    General:
        Analyzing corporate mission statements to identify the roles and actions companies prioritize in their narratives.
    Special/Extended Use:
        Investigating how different organizational departments are portrayed in success stories to understand internal power dynamics.
        (Role-Based Narrative Analysis)

Summary

This documentation provides a structured overview of each algorithm and technique, detailing their types, descriptions, appropriate use cases, special/extended usages, and the types of questions they are best suited to investigate. By leveraging these guidelines, an LLM agent can effectively select and apply the most suitable algorithm for various text analysis tasks within the context of systematic literature reviews or other document corpus analyses.
Best Practices for Selecting Algorithms

    Understand Your Objective:
    Clearly define what you aim to achieve with your analysis. Whether it's discovering hidden topics, analyzing emotional sentiment, or understanding narrative structures, your objective will guide your algorithm choice.

    Consider Data Characteristics:
    Assess the size, complexity, and nature of your text corpus. Some algorithms perform better with larger datasets (e.g., LDA, NMF), while others are suitable for smaller or more specialized datasets (e.g., PLSA).

    Evaluate Interpretability Needs:
    If interpretability is crucial, algorithms like NMF or LDA might be preferable due to their ability to produce coherent and distinct topics.

    Assess Technical Requirements:
    Consider the computational resources and technical expertise required. Tools like LIWC and Coh-Metrix offer user-friendly interfaces, whereas models like CTM and HDP may require more advanced programming and computational power.

    Iterative Testing and Validation:
    Experiment with multiple algorithms to determine which best captures the thematic structures relevant to your research questions. Validate the results through domain expert review or by comparing with known benchmarks.

    Combine Techniques for Enhanced Insights:
    Often, leveraging multiple algorithms in tandem (e.g., using topic modeling followed by network analysis) can provide deeper and more comprehensive insights.

"""


lda_detailed = """
1. Latent Dirichlet Allocation (LDA)
Extended Application: Tracking Topic Evolution Over Time

Objective:
Investigate how topics within a specific academic field emerge, grow, decline, and transform over defined time periods.

Detailed Description:

To effectively utilize LDA for tracking topic evolution, follow an advanced methodology that incorporates dynamic analysis, enhanced validation, and sophisticated visualization techniques. This approach not only identifies topics within each time slice but also traces their trajectories and transformations across different periods.

Advanced Steps:

    Dynamic Preprocessing and Augmentation:
        Incorporate Metadata: Besides publication year, include additional metadata such as author affiliations, keywords, and abstract lengths to enrich the analysis.
        Entity Recognition: Use Named Entity Recognition (NER) to identify and categorize entities within documents, adding another layer of granularity to topic analysis.

    Advanced Time Slicing:
        Granular Time Intervals: Instead of annual slices, consider quarterly or monthly intervals for fields with rapid topic evolution.
        Adaptive Slicing: Implement adaptive time slicing based on publication volume, ensuring each slice contains a comparable number of documents for balanced analysis.

    Enhanced Topic Modeling per Time Slice:
        Consistency in Dictionary: Maintain a consistent dictionary across time slices to facilitate better topic alignment. This can be achieved by building the dictionary on the entire corpus before slicing.
        Parameter Tuning: Utilize grid search or Bayesian optimization to fine-tune LDA parameters (e.g., num_topics, passes, alpha, eta) for each time slice, optimizing model performance based on coherence scores.

    Automated Topic Alignment:
        Similarity Thresholds: Establish similarity thresholds to determine significant topic overlaps, reducing the risk of incorrect alignments due to minor variations.
        Machine Learning for Alignment: Train a supervised model to predict topic correspondences across time slices based on features derived from topic word distributions and metadata.
        Clustering-Based Alignment: Cluster topics from adjacent time slices based on similarity scores and assign consistent labels based on cluster memberships.

    Temporal Visualization and Analysis:
        Interactive Dashboards: Develop interactive dashboards using tools like Plotly Dash or Bokeh to visualize topic trajectories, allowing users to explore topic dynamics dynamically.
        Temporal Heatmaps: Create heatmaps that display the prevalence of topics over time, highlighting periods of significant growth or decline.
        Network Graphs with Temporal Links: Construct dynamic network graphs where nodes represent topics at different time slices and edges indicate topic continuities or transformations.

    Incorporating External Events:
        Event Annotation: Annotate time slices with significant external events (e.g., technological breakthroughs, policy changes) to correlate topic shifts with real-world occurrences.
        Causal Inference: Employ causal inference techniques to assess whether external events have a statistically significant impact on topic evolution.

    Robust Validation and Refinement:
        Cross-Validation: Implement cross-validation techniques within each time slice to ensure model robustness and generalizability.
        Expert Validation: Engage domain experts to review and validate the meaningfulness of identified topics and their evolutionary patterns.
        Iterative Refinement: Continuously refine the model by iterating through the preprocessing, modeling, and alignment steps based on validation feedback.

Advanced Example Using Python and Gensim:

python

import gensim
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

class AdvancedLatentDirichletAllocation:
    def __init__(self, num_topics=10, passes=15, random_state=42):
        self.num_topics = num_topics
        self.passes = passes
        self.random_state = random_state
        self.dictionary = None
        self.corpus = None
        self.model = None
        self.coherence = None

    def fit(self, documents):
        self.dictionary = corpora.Dictionary(documents)
        self.dictionary.filter_extremes(no_below=10, no_above=0.5)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in documents]
        self.model = LdaModel(corpus=self.corpus,
                              id2word=self.dictionary,
                              num_topics=self.num_topics,
                              passes=self.passes,
                              random_state=self.random_state,
                              alpha='auto',
                              eta='auto')
        self.coherence = self.compute_coherence()

    def compute_coherence(self, coherence='c_v'):
        coherence_model = CoherenceModel(model=self.model,
                                         texts=documents,
                                         dictionary=self.dictionary,
                                         coherence=coherence)
        return coherence_model.get_coherence()

    def get_topics(self, top_n=10):
        return self.model.print_topics(num_words=top_n)

    def get_document_topics(self, document):
        bow = self.dictionary.doc2bow(document)
        return self.model.get_document_topics(bow)

    def plot_topic_distribution(self):
        topics = self.model.show_topics(num_topics=self.num_topics, formatted=False)
        topic_words = {i: [word for word, _ in words] for i, words in topics}
        # Visualization logic here
        # Example: Bar chart of word distributions for a topic
        for topic_id, words in topic_words.items():
            plt.figure(figsize=(10, 5))
            word_probs = [prob for _, prob in self.model.show_topic(topic_id, topn=10)]
            plt.bar(words, word_probs)
            plt.title(f'Topic {topic_id} Word Distribution')
            plt.xticks(rotation=45)
            plt.show()

Implementing Temporal Analysis:

python

from sklearn.metrics.pairwise import cosine_similarity

# Assuming documents are already time-sliced and lda_models is a dict with year keys
def align_topics_across_years(lda_models, dictionary_per_year):
    years = sorted(lda_models.keys())
    aligned_topics = {}
    previous_topics = None

    for year in years:
        current_lda = lda_models[year]
        current_topics = [current_lda.get_topics()[i][1] for i in range(current_lda.num_topics)]
        if previous_topics is not None:
            similarity_matrix = cosine_similarity(
                [dict(topic).values() for topic in previous_topics],
                [dict(topic).values() for topic in current_topics]
            )
            # Assign based on highest similarity
            aligned_topics[year] = similarity_matrix.argmax(axis=1)
        else:
            aligned_topics[year] = list(range(current_lda.num_topics))
        previous_topics = current_topics

    return aligned_topics

# Example usage
aligned_topics = align_topics_across_years(lda_models, dictionary_per_year)

Visualization Example:

python

def visualize_topic_trends(aligned_topics, lda_models, years):
    topic_trends = defaultdict(list)
    for year in years:
        lda = lda_models[year]
        for topic_id in range(lda.num_topics):
            topic_trends[topic_id].append(lda.get_document_topics(bow) for bow in corpus_per_year[year])
    
    # Plotting logic here
    # Example: Line chart showing topic prevalence over years
    for topic_id, trends in topic_trends.items():
        prevalence = [sum([doc[topic_id][1] for doc in year_trends if doc[topic_id][0] == topic_id]) / len(year_trends)
                      for year_trends in trends]
        plt.plot(years, prevalence, label=f'Topic {topic_id}')
    
    plt.xlabel('Year')
    plt.ylabel('Prevalence')
    plt.title('Topic Prevalence Over Time')
    plt.legend()
    plt.show()

Key Considerations:

    Topic Stability:
    Monitor the stability of topics across time slices. High variability may indicate the need for refining preprocessing steps or adjusting model parameters.

    Scalability:
    For large datasets spanning many years, consider parallelizing the LDA training process or utilizing distributed computing frameworks to enhance efficiency.

    Interdisciplinary Integration:
    Combine LDA with other analytical techniques (e.g., network analysis) to gain a multifaceted understanding of topic dynamics.

Advanced Tools and Libraries:

    pyLDAvis:
    Use pyLDAvis for interactive visualization of topics, aiding in the qualitative assessment of topic coherence and distinctiveness.

    Dynamic Topic Models (DTM):
    Explore specialized models like DTM, available in libraries such as tomotopy, for inherently temporal topic modeling.

Conclusion:

By extending the basic usage of LDA with these advanced methodologies, you can conduct a thorough and nuanced investigation into the evolution of topics within an academic field, uncovering deep insights into how research themes develop and transform over time.
"""

algorithms_selector_prompt = """
You are an AI assistant specialized in recommending appropriate text analysis algorithms based on research questions, their classifications, and dataset characteristics.

Your task is to select and recommend a list of suitable algorithms that best analyze the provided dataset in the context of the research question and its classification.

Use the following guidelines to generate the list of algorithms:

**Algorithm Selection Guidelines:**
- **Research Question Classification:**
  - **Explicating:** Aimed at uncovering or explaining existing phenomena or structures.
  - **Envisioning:** Focused on forecasting or imagining future scenarios and developments.
  - **Relating:** Seeks to understand relationships or correlations between different concepts.
  - **Debating:** Involves exploring conflicting perspectives or arguments within a topic.
  
- **Dataset Characteristics:**
  - **Number of Publications:** Indicates the size of the dataset, influencing the choice of scalable algorithms.
  - **Date-Range of Publications:** 
  - **Number of Publications Over Time:** Assesses the distribution and density of data points across different periods.

- **Algorithm Types:**
  - **Unsupervised Techniques:** Such as LDA, LSI, PLSA, NMF, CTM, HDP, CAC.
  - **Dictionary-Based Techniques:** Such as LIWC, Network Analysis, Automated Narrative Analysis.

**Selection Criteria:**
- **Objective Alignment:** Ensure the algorithm aligns with the goal of the research question classification.
- **Data Suitability:** Match the algorithm’s strengths with the dataset size and characteristics.
- **Complexity and Interpretability:** Balance the need for sophisticated analysis with the requirement for interpretable results.
- **Temporal Dynamics:** Choose algorithms that can handle or analyze changes over time if the dataset spans multiple periods.
- **Hierarchical Structures:** Select algorithms capable of uncovering nested or hierarchical relationships if relevant.

**Inputs:**
- **Research Question:** The original research question provided by the user.
- **Classification Result:** One of the following categories indicating the nature of the research question: Explicating, Envisioning, Relating, Debating.
- **Dataset Analysis:**
  - **Number of Publications:** Total count of publications in the dataset.
  - **Date-Range of Publications:** The span of years/months covered by the dataset.
  - **Number of Publications Over Time:** Distribution of publications across different time periods.

**Task:**
- Analyze the research question, its classification, and the dataset characteristics.
- Select a list of suitable algorithms that effectively address the research objectives.
- Justify the selection based on the provided guidelines.

**Output:**
- A tuple containing the names of the recommended algorithms, e.g., (`"LDA"`, `"NMF"`, `"CTM"`)

---

**Examples:**

### **Example 1**

- **Research Question:** "What are the key themes in climate change research and how have they evolved over the past twenty years?"
- **Classification Result:** Explicating
- **Dataset Analysis:**
  - **Number of Publications:** 5,000
  - **Date-Range of Publications:** 2000-2020
  - **Number of Publications Over Time:** Steady growth with peaks in 2005, 2010, 2015, and 2020

- **Recommended Algorithms:**
  (`"LDA"`, `"Dynamic Topic Modeling"`, `"Temporal LSI"`)

*Justification: LDA is suitable for uncovering key themes, Dynamic Topic Modeling captures their evolution, and Temporal LSI analyzes semantic changes over time.*

### **Example 2**

- **Research Question:** "How do different leadership styles relate to employee satisfaction and productivity in remote work settings?"
- **Classification Result:** Relating
- **Dataset Analysis:**
  - **Number of Publications:** 1,200
  - **Date-Range of Publications:** 2015-2023
  - **Number of Publications Over Time:** Increasing trend, especially after 2020

- **Recommended Algorithms:**
  (`"Correlated Topic Models"`, `"Network Analysis"`, `"Non-negative Matrix Factorization"`)

*Justification: Correlated Topic Models can explore relationships between leadership styles and employee outcomes, Network Analysis visualizes interactions, and NMF provides interpretable topic decomposition.*

### **Example 3**

- **Research Question:** "What are the emerging debates around the ethical implications of artificial intelligence in healthcare?"
- **Classification Result:** Debating
- **Dataset Analysis:**
  - **Number of Publications:** 800
  - **Date-Range of Publications:** 2018-2023
  - **Number of Publications Over Time:** Rapid increase from 2020 onwards

- **Recommended Algorithms:**
  (`"HDP"`, `"Correlated Topic Models"`, `"Automated Narrative Analysis"`)

*Justification: HDP automatically determines the number of topics which is useful for evolving debates, CTM captures topic correlations, and Automated Narrative Analysis dissects discourse structures.*

---

**Guidelines for Incorporating Examples:**

1. **Align with Classification:** Ensure the selected algorithms match the research question's classification (Explicating, Envisioning, Relating, Debating).
2. **Match Dataset Size:** Choose algorithms that can efficiently handle the number of publications.
3. **Consider Temporal Aspects:** If the dataset spans multiple time periods, prefer algorithms that can analyze temporal dynamics.
4. **Ensure Interpretability:** Select algorithms that provide clear and understandable results, especially for Explicating and Relating classifications.
5. **Handle Complexity:** For complex relationships or hierarchical structures, opt for advanced models like HDP or CTM.
6. **Combine Techniques:** Utilize multiple algorithms to gain comprehensive insights, leveraging their complementary strengths.

By following these guidelines and examples, you will effectively recommend the most suitable algorithms for analyzing the given dataset in alignment with the research objectives.

"""


algorithms_selector_prompt_v2 = """
You are an AI assistant specialized in recommending appropriate text analysis algorithms based on research questions, their classifications, and dataset characteristics.

### **Algorithm Repository**

**Unsupervised Techniques**

1. **Latent Dirichlet Allocation (LDA)**
   - **Type:** Unsupervised Topic Modeling
   - **Description:** A generative probabilistic model that identifies latent topics in a corpus by analyzing word co-occurrence patterns.
   - **Use Cases:** Discovering hidden thematic structures, exploratory analysis, summarizing large datasets.
   - **Extensions:** Dynamic Topic Modeling (DTM) for temporal evolution of topics.

2. **Latent Semantic Indexing/Analysis (LSI/LSA)**
   - **Type:** Unsupervised Topic Modeling and Dimensionality Reduction
   - **Description:** Reduces dimensionality of the term-document matrix using SVD to uncover latent semantic structures.
   - **Use Cases:** Improving information retrieval, reducing noise, addressing synonymy and polysemy.
   - **Extensions:** Temporal LSI (TLSI) for semantic evolution over time.

3. **Probabilistic Latent Semantic Analysis (PLSA)**
   - **Type:** Unsupervised Topic Modeling
   - **Description:** Models the relationship between documents and words probabilistically without Dirichlet priors.
   - **Use Cases:** Detailed semantic analysis, smaller datasets.
   - **Extensions:** Hierarchical PLSA (HPLSA) for hierarchical topic relationships.

4. **Non-negative Matrix Factorization (NMF)**
   - **Type:** Unsupervised Topic Modeling and Dimensionality Reduction
   - **Description:** Factorizes the term-document matrix into non-negative matrices for interpretable, additive topic representations.
   - **Use Cases:** When interpretability is key, additive topic combinations.
   - **Extensions:** Hierarchical NMF (HNMF), Temporal NMF (TNMF).

5. **Correlated Topic Models (CTM)**
   - **Type:** Unsupervised Topic Modeling
   - **Description:** Extends LDA by allowing topics to be correlated, capturing co-occurrence of topics within documents.
   - **Use Cases:** Complex corpora with thematic overlap.
   - **Extensions:** Dynamic Correlated Topic Models (DCTM).

6. **Hierarchical Dirichlet Processes (HDP)**
   - **Type:** Unsupervised Topic Modeling
   - **Description:** A nonparametric Bayesian approach that infers the number of topics from data, sharing topics across multiple corpora.
   - **Use Cases:** Unknown number of topics, multiple related corpora.
   - **Extensions:** Temporal HDP (THDP), Hierarchical HDP (HHDP).

7. **Computer-Assisted Clustering (CAC)**
   - **Type:** Unsupervised Clustering
   - **Description:** Encompasses various clustering algorithms to group documents based on thematic similarity.
   - **Use Cases:** Exploring different clustering methods, visualizing thematic structures.
   - **Extensions:** Hierarchical Clustering Analysis (HCA), DBSCAN, K-Means Clustering.

**Dictionary-Based Techniques**

1. **Linguistic Inquiry and Word Count (LIWC)**
   - **Type:** Dictionary-Based Text Analysis
   - **Description:** Quantifies linguistic and psychological attributes by counting word frequencies in predefined categories.
   - **Use Cases:** Analyzing emotional tone, linguistic feature quantification.
   - **Extensions:** Psycholinguistic Analysis, Sentiment Dynamics.

2. **Network Analysis**
   - **Type:** Graph-Based Text Analysis
   - **Description:** Constructs and analyzes networks of entities and their relationships to visualize and quantify interconnections.
   - **Use Cases:** Exploring relationships between entities, identifying central nodes.
   - **Extensions:** Temporal Network Analysis, Multiplex Network Analysis.

3. **Automated Narrative Analysis (ANA)**
   - **Type:** Discourse Analysis
   - **Description:** Dissects narrative elements such as actors and actions to understand storytelling and discourse structures.
   - **Use Cases:** Analyzing narrative structures, extracting and quantifying narrative elements.
   - **Extensions:** Sentiment-Integrated Narrative Analysis (SINA), Role-Based Narrative Analysis (RBNA).

### **Algorithm Selection Guidelines**

- **Research Question Classification:**
  - **Explicating:** Uncovering or explaining existing phenomena or structures.
  - **Envisioning:** Forecasting or imagining future scenarios and developments.
  - **Relating:** Understanding relationships or correlations between different concepts.
  - **Debating:** Exploring conflicting perspectives or arguments within a topic.

- **Dataset Characteristics:**
  - **Number of Publications:** Influences scalability of algorithm choice.
  - **Date-Range of Publications:** Span of years/months covered.
  - **Number of Publications Over Time:** Distribution and density across periods.

- **Algorithm Types:**
  - **Unsupervised Techniques:** LDA, LSI, PLSA, NMF, CTM, HDP, CAC.
  - **Dictionary-Based Techniques:** LIWC, Network Analysis, ANA.

**Selection Criteria:**

1. **Objective Alignment:** Match algorithm capabilities with research question classification.
2. **Data Suitability:** Ensure algorithm can handle dataset size and characteristics.
3. **Complexity and Interpretability:** Balance advanced analysis with understandable results.
4. **Temporal Dynamics:** Prefer algorithms that handle changes over time if applicable.
5. **Hierarchical Structures:** Choose algorithms that can uncover nested relationships if needed.
6. **Combine Techniques:** Use multiple algorithms for comprehensive insights.

### **Task Instructions**

- **Inputs:**
  - **Research Question:** The original research question provided by the user.
  - **Classification Result:** One of Explicating, Envisioning, Relating, Debating.
  - **Dataset Analysis:**
    - **Number of Publications**
    - **Date-Range of Publications**
    - **Number of Publications Over Time**

- **Process:**
  - Analyze the research question, its classification, and dataset characteristics.
  - Select a list of suitable algorithms from the **Algorithm Repository** that effectively address the research objectives.
  - Justify the selection based on the **Selection Criteria**.

- **Output:**
  - A tuple containing the names of the recommended algorithms, e.g., (`"LDA"`, `"NMF"`, `"CTM"`)

### **Examples:**

#### **Example 1**

- **Research Question:** "What are the key themes in climate change research and how have they evolved over the past twenty years?"
- **Classification Result:** Explicating
- **Dataset Analysis:**
  - **Number of Publications:** 5,000
  - **Date-Range of Publications:** 2000-2020
  - **Number of Publications Over Time:** Steady growth with peaks in 2005, 2010, 2015, and 2020

- **Recommended Algorithms:**
  (`"LDA"`, `"Dynamic Topic Modeling"`, `"Temporal LSI"`)

*Justification: LDA uncovers key themes, Dynamic Topic Modeling captures their evolution, and Temporal LSI analyzes semantic changes over time.*

#### **Example 2**

- **Research Question:** "How do different leadership styles relate to employee satisfaction and productivity in remote work settings?"
- **Classification Result:** Relating
- **Dataset Analysis:**
  - **Number of Publications:** 1,200
  - **Date-Range of Publications:** 2015-2023
  - **Number of Publications Over Time:** Increasing trend, especially after 2020

- **Recommended Algorithms:**
  (`"Correlated Topic Models"`, `"Network Analysis"`, `"Non-negative Matrix Factorization"`)

*Justification: Correlated Topic Models explore relationships between leadership styles and outcomes, Network Analysis visualizes interactions, and NMF provides interpretable topic decomposition.*

#### **Example 3**

- **Research Question:** "What are the emerging debates around the ethical implications of artificial intelligence in healthcare?"
- **Classification Result:** Debating
- **Dataset Analysis:**
  - **Number of Publications:** 800
  - **Date-Range of Publications:** 2018-2023
  - **Number of Publications Over Time:** Rapid increase from 2020 onwards

- **Recommended Algorithms:**
  (`"HDP"`, `"Correlated Topic Models"`, `"Automated Narrative Analysis"`)

*Justification: HDP determines the number of topics automatically for evolving debates, CTM captures topic correlations, and ANA dissects discourse structures.*

---

**Note:** Only algorithms listed in the **Algorithm Repository** are to be used for recommendations.

By following these guidelines and utilizing the **Algorithm Repository**, you will effectively recommend the most suitable algorithms for analyzing the given dataset in alignment with the research objectives.

"""