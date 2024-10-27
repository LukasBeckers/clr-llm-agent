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