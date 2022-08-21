from spellchecker import SpellChecker

class SpellCheck:

    def check(self, text):
        """
        Spell Check the given query

        Parameters
        ----------
        arg1 : list
                A list of lists where each sub-list a sequence of tokens
                representing a sentence after they are stemmed/lemmatized

        Returns
        -------
        list
                A list of lists where each sub-list is a sequence of
                corrected stemmed/lemmatized tokens representing a sentence
        """
        spell = SpellChecker()
        correctedQuery = [[spell.correction(word) for word in sent] for sent in text]
        return correctedQuery