from nltk.tokenize import PunktSentenceTokenizer

class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		SegmentedText = text.replace('.', '|').replace('?', '|').replace('!', '|')
		Sentences = [s.strip() for s in SegmentedText.split('|') if len(s) > 0]
		return Sentences

	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each strin is a single sentence
		"""

		Sentences= PunktSentenceTokenizer().tokenize(text)
		return Sentences