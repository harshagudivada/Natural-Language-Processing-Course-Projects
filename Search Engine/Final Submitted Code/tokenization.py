from nltk.tokenize import TreebankWordTokenizer


class Tokenization():

    def naive(self, text):
        """
        Tokenization using a Naive Approach

        Parameters
        ----------
        arg1 : list
                A list of strings where each string is a single sentence

        Returns
        -------
        list
                A list of lists where each sub-list is a sequence of tokens
        """
        tokenizedText = []
        for sent in text:
            words = [w.replace(',', '').strip() for w in sent.split(' ')]
            tokenizedText.append(words)
        return tokenizedText


    def pennTreeBank(self, text):
        """
        Tokenization using the Penn Tree Bank Tokenizer

        Parameters
        ----------
        arg1 : list
                A list of strings where each string is a single sentence

        Returns
        -------
        list
                A list of lists where each sub-list is a sequence of tokens
        """

        tokenizedText = []
        tokenizer = TreebankWordTokenizer()
        for sentence in text:
            tokenizedText.append(tokenizer.tokenize(sentence))
        return tokenizedText