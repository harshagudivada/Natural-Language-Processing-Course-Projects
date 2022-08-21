from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from spellCheckQuery import SpellCheck
from informationRetrieval_LSA import InformationRetrieval
from evaluation import Evaluation
from multiprocessing import cpu_count
from joblib import delayed, Parallel
import argparse
import json
import matplotlib.pyplot as plt
plt.style.use("ggplot")


n_jobs = cpu_count()
print(f"parallelizing on {n_jobs} cores")


class SearchEngine:

	def __init__(self, args):
		self.args = args
		self.tokenizer = Tokenization()
		self.sentenceSegmenter = SentenceSegmentation()
		self.inflectionReducer = InflectionReduction(algo = self.args.algo)
		self.stopwordRemover = StopwordRemoval()
		self.informationRetriever = InformationRetrieval()
		self.spellCheck = SpellCheck()
		self.evaluator = Evaluation()


	def segmentSentences(self, text):
		"""
		Call the required sentence segmenter
		"""
		if self.args.segmenter == "naive":
			return self.sentenceSegmenter.naive(text)
		elif self.args.segmenter == "punkt":
			return self.sentenceSegmenter.punkt(text)

	def tokenize(self, text):
		"""
		Call the required tokenizer
		"""
		if self.args.tokenizer == "naive":
			return self.tokenizer.naive(text)
		elif self.args.tokenizer == "ptb":
			return self.tokenizer.pennTreeBank(text)

	def reduceInflection(self, text):
		"""
		Call the required stemmer/lemmatizer
		"""
		return self.inflectionReducer.reduce(text)

	def removeStopwords(self, text):
		"""
		Call the required stopword remover
		"""
		return self.stopwordRemover.fromList(text)

	def checkSpelling(self, text):
		"""
		Call the spell-checker
		"""
		return self.spellCheck.check(text)


	def preprocessQueries(self, queries):
		"""
		Preprocess the queries - segment, tokenize, stem/lemmatize and remove stopwords
		"""
		# Segment queries
		queries = Parallel(n_jobs=n_jobs)(delayed(self.segmentSentences)(query) for query in queries)
		json.dump(queries, open(f"{self.args.out_folder}segmented_queries.txt", 'w'))
		# Tokenize queries
		queries = Parallel(n_jobs=n_jobs)(delayed(self.tokenize)(query) for query in queries)
		json.dump(queries, open(f"{self.args.out_folder}tokenized_queries.txt", 'w'))
		if self.args.spellcheck:
			# correct queries
			queries = Parallel(n_jobs=n_jobs)(delayed(self.checkSpelling)(query) for query in queries)
			json.dump(queries, open(f"{self.args.out_folder}corrected_queries.txt", 'w'))
		# Stem/Lemmatize queries
		queries = Parallel(n_jobs=n_jobs)(delayed(self.reduceInflection)(query) for query in queries)
		json.dump(queries, open(f"{self.args.out_folder}reduced_queries.txt", 'w'))
		# remove stop words 
		queries = Parallel(n_jobs=n_jobs)(delayed(self.removeStopwords)(query) for query in queries)
		json.dump(queries, open(f"{self.args.out_folder}stopword_removed_queries.txt", 'w'))
		
		return queries


	def preprocessDocs(self, docs):
		"""
		Preprocess the documents
		"""
		# Segment docs
		docs = Parallel(n_jobs=n_jobs)(delayed(self.segmentSentences)(doc) for doc in docs)
		json.dump(docs, open(f"{self.args.out_folder}segmented_docs.txt", 'w'))
		# Tokenize docs
		docs = Parallel(n_jobs=n_jobs)(delayed(self.tokenize)(doc) for doc in docs)
		json.dump(docs, open(f"{self.args.out_folder}tokenized_docs.txt", 'w'))
		# Stem/Lemmatize docs
		docs = Parallel(n_jobs=n_jobs)(delayed(self.reduceInflection)(doc) for doc in docs)
		json.dump(docs, open(f"{self.args.out_folder}reduced_docs.txt", 'w'))
		# Remove stopwords from docs
		docs = Parallel(n_jobs=n_jobs)(delayed(self.removeStopwords)(doc) for doc in docs)
		json.dump(docs, open(f"{self.args.out_folder}stopword_removed_docs.txt", 'w'))

		return docs


	def evaluateDataset(self):
		"""
		- preprocesses the queries and documents, stores in output folder
		- invokes the IR system
		- evaluates precision, recall, fscore, nDCG and MAP 
		  for all queries in the Cranfield dataset
		- produces graphs of the evaluation metrics in the output folder
		"""

		# Read queries
		queries_json = json.load(open(f"{args.dataset}cran_queries.json", 'r'))
		query_ids, queries = [item["query number"] for item in queries_json], \
							 [item["query"] for item in queries_json]
		# Process queries 
		processedQueries = self.preprocessQueries(queries)
		print("queries pre-processed!")

		# Read documents
		docs_json = json.load(open(f"{args.dataset}cran_docs.json", 'r'))
		doc_ids, docs = [item["id"] for item in docs_json], \
						[item["body"] for item in docs_json]
		# Process documents
		processedDocs = self.preprocessDocs(docs)
		print("docs pre-processed!")

		# Build document index
		self.informationRetriever.buildIndexWithSVD(processedDocs, doc_ids, n_comp=350)
		# Rank the documents for each query
		doc_IDs_ordered = self.informationRetriever.rank(processedQueries)

		# Read relevance judements
		qrels = json.load(open(f"{args.dataset}cran_qrels.json", 'r'))

		# Calculate precision, recall, f-score, MAP and nDCG for k = 1 to 10
		precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
		for k in range(1, 11):
			precision = self.evaluator.meanPrecision(doc_IDs_ordered, query_ids, qrels, k)
			recall = self.evaluator.meanRecall(doc_IDs_ordered, query_ids, qrels, k)
			fscore = self.evaluator.meanFscore(doc_IDs_ordered, query_ids, qrels, k)
			nDCG = self.evaluator.meanNDCG(doc_IDs_ordered, query_ids, qrels, k)
			MAP = self.evaluator.meanAveragePrecision(doc_IDs_ordered, query_ids, qrels, k)
			
			precisions.append(precision)
			recalls.append(recall)
			fscores.append(fscore)
			nDCGs.append(nDCG)
			MAPs.append(MAP)
			
			print(f"Precision, Recall and F-score @ {k} : {precision:.4f}, {recall:.4f}, {fscore:.4f}")
			print(f"MAP, nDCG @ {k} : {MAP:.4f}, {nDCG:.4f}")

		# Plot the metrics and save plot 
		plt.plot(range(1, 11), precisions, label="Precision")
		plt.plot(range(1, 11), recalls, label="Recall")
		plt.plot(range(1, 11), fscores, label="F-Score")
		plt.plot(range(1, 11), MAPs, label="MAP")
		plt.plot(range(1, 11), nDCGs, label="nDCG")
		plt.title("Evaluation Metrics - Cranfield Dataset")
		plt.xlabel("k")
		plt.legend(bbox_to_anchor=(1.04, 0.5), 
				   loc="center left", 
				   borderaxespad=0)
		plt.savefig(f"{args.out_folder}eval_plot_LSA.png",
					bbox_inches="tight")

		
	def handleCustomQuery(self):
		"""
		Take a custom query as input and return top five relevant documents
		"""

		#Get query
		print("Enter query below")
		query = input()
		# Process documents
		processedQuery = self.preprocessQueries([query])[0]
		print("queries pre-preprocessed!")

		# Read documents
		docs_json = json.load(open(f"{args.dataset}cran_docs.json", 'r'))
		doc_ids, docs = [item["id"] for item in docs_json], \
						[item["body"] for item in docs_json]
		# Process documents
		processedDocs = self.preprocessDocs(docs)
		print("docs pre-preprocessed!")

		# Build document index
		self.informationRetriever.buildIndexWithSVD(processedDocs, doc_ids, n_comp=350)
		# Rank the documents for the query
		doc_IDs_ordered = self.informationRetriever.rank([processedQuery])[0]

		# Print the IDs of first five documents
		print("\nTop five document IDs : ")
		for id in doc_IDs_ordered[:5]:
			print(id)


if __name__ == "__main__":
	# Create an argument parser
	parser = argparse.ArgumentParser(description='main.py')

	# Tunable parameters as external arguments
	parser.add_argument('-dataset', default = "cranfield/", 
						help = "Path to the dataset folder")
	parser.add_argument('-out_folder', default = "output/", 
						help = "Path to output folder")
	parser.add_argument('-segmenter', default = "punkt",
	                    help = "Sentence Segmenter Type [naive|punkt]")
	parser.add_argument('-tokenizer',  default = "ptb",
	                    help = "Tokenizer Type [naive|ptb]")
	parser.add_argument('-custom', action = "store_true", 
						help = "Take custom query as input")
	parser.add_argument('-algo', default = "lemmatizer",
					    help = "Reducer Type [lemmatizer|stemmer]")
	parser.add_argument("-spellcheck", action="store_true",
						help = "Use SpellCheck ")
	
	# Parse the input arguments
	args = parser.parse_args()

	# Create an instance of the Search Engine
	searchEngine = SearchEngine(args)
	
	# Either handle query from user or evaluate on the complete dataset 
	if args.custom:
		searchEngine.handleCustomQuery()
	else:
		searchEngine.evaluateDataset()