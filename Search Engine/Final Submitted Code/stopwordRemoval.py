import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

class StopwordRemoval():

    def fromList(self, text):
        """
        Stopword Removal from the text
        
        Parameters
        ----------
        arg1 : list
                A list of lists where each sub-list is a sequence of tokens
                representing a sentence

        Returns
        -------
        list
                A list of lists where each sub-list is a sequence of tokens
                representing a sentence with stopwords removed
        """

        allStopwords = set(stopwords.words("english"))
        allStopwords.add("viz")

        stopwordRemovedText = [[t for t in tokens if len(t) > 1 and not t in allStopwords]  for tokens in text]
        return stopwordRemovedText