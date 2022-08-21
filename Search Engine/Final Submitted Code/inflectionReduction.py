import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

class InflectionReduction:

    def __init__(self, algo):
        self.algo = algo

    def reduce(self, text):
        """
        Stemming/Lemmatization

        Parameters
        ----------
        arg1 : list
                A list of lists where each sub-list a sequence of tokens
                representing a sentence

        Returns
        -------
        list
                A list of lists where each sub-list is a sequence of
                stemmed/lemmatized tokens representing a sentence
        """
        if self.algo == "lemmatizer":
            lemmetizer = WordNetLemmatizer()
            reducedText = [[lemmetizer.lemmatize(word) for word in sent] for sent in text]
        else:
            stemmer = PorterStemmer()
            reducedText = [[stemmer.stem(word) for word in sent] for sent in text]
        return reducedText