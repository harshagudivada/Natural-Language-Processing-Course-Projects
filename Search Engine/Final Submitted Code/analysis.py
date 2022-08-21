from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from spellCheckQuery import SpellCheck
from informationRetrieval_LSA import InformationRetrieval
from informationRetrieval_VSM import InformationRetrieval as InformationRetrievalBaseline
from evaluation import Evaluation
from multiprocessing import cpu_count
from joblib import delayed, Parallel
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from time import time
plt.style.use("ggplot")


n_jobs = cpu_count()
print(f"parallelizing on {n_jobs} cores")


class SearchEngine:

	def __init__(self, args):
		self.args = args
		self.tokenizer = Tokenization()
		self.sentenceSegmenter = SentenceSegmentation()
		self.inflectionReducer = InflectionReduction()
		self.stopwordRemover = StopwordRemoval()
		self.informationRetriever = InformationRetrieval()
		self.informationRetrieverBaseline = InformationRetrievalBaseline()
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
			print("running spellcheck")
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

		qrels = json.load(open(f"{args.dataset}cran_qrels.json", 'r'))

		#---------------------------------------------------------------------------------------------------------------------------#

		## Plot nDCG for Different n_comp ##
		nDCGs, best_nDCG, best_n_comp = [], 0, 50
		for n_comp in range(50, doc_ids[-1], 50):
			print(f"running LSA with n_comp={n_comp};", end=' ')
			start = time()
			self.informationRetriever.buildIndexWithSVD(processedDocs, doc_ids, n_comp=n_comp)
			doc_IDs_ordered = self.informationRetriever.rank(processedQueries)
			nDCG = self.evaluator.meanNDCG(doc_IDs_ordered, query_ids, qrels, k=10)
			print(f"runtime: {(time() - start):.4f}s")
			nDCGs.append(nDCG)
			if nDCG >= best_nDCG:
				best_nDCG, best_n_comp = nDCG, n_comp

		print(f"best nDCG @ 10: {best_nDCG}, best_n_comp: {best_n_comp}")
		plt.plot(range(50, doc_ids[-1], 50), nDCGs, linestyle="dashed", marker='o')
		plt.title("nDCG for varying n_comp - Cranfield Dataset")
		plt.xlabel("n_comp")
		plt.ylabel("nDCG @ 10")
		plt.savefig(f"{args.out_folder}nDCG @ 10.png")
		plt.show()

		#---------------------------------------------------------------------------------------------------------------------------#

		print('\n')
		## Compare VSM and LSA ##
		# Get Metrics of VSM
		self.informationRetrieverBaseline.buildIndex(processedDocs, doc_ids)
		doc_IDs_ordered = self.informationRetrieverBaseline.rank(processedQueries)
		precisions_vsm, recalls_vsm, fscores_vsm, MAPs_vsm, nDCGs_vsm = [], [], [], [], []
		for k in range(1, 11):
			precision = self.evaluator.meanPrecision(doc_IDs_ordered, query_ids, qrels, k)
			recall = self.evaluator.meanRecall(doc_IDs_ordered, query_ids, qrels, k)
			fscore = self.evaluator.meanFscore(doc_IDs_ordered, query_ids, qrels, k)
			nDCG = self.evaluator.meanNDCG(doc_IDs_ordered, query_ids, qrels, k)
			MAP = self.evaluator.meanAveragePrecision(doc_IDs_ordered, query_ids, qrels, k)
			
			precisions_vsm.append(precision)
			recalls_vsm.append(recall)
			fscores_vsm.append(fscore)
			nDCGs_vsm.append(nDCG)
			MAPs_vsm.append(MAP)
			
			print(f"[VSM] Precision, Recall and F-score @ {k} : {precision:.4f}, {recall:.4f}, {fscore:.4f}")
			print(f"[VSM] MAP, nDCG @ {k} : {MAP:.4f}, {nDCG:.4f}")

		print('\n')
		# Get Metrics of LSA with best_n_comp
		self.informationRetriever.buildIndexWithSVD(processedDocs, doc_ids, n_comp=best_n_comp)
		doc_IDs_ordered = self.informationRetriever.rank(processedQueries)
		precisions_lsa, recalls_lsa, fscores_lsa, MAPs_lsa, nDCGs_lsa = [], [], [], [], []
		for k in range(1, 11):
			precision = self.evaluator.meanPrecision(doc_IDs_ordered, query_ids, qrels, k)
			recall = self.evaluator.meanRecall(doc_IDs_ordered, query_ids, qrels, k)
			fscore = self.evaluator.meanFscore(doc_IDs_ordered, query_ids, qrels, k)
			nDCG = self.evaluator.meanNDCG(doc_IDs_ordered, query_ids, qrels, k)
			MAP = self.evaluator.meanAveragePrecision(doc_IDs_ordered, query_ids, qrels, k)
			
			precisions_lsa.append(precision)
			recalls_lsa.append(recall)
			fscores_lsa.append(fscore)
			nDCGs_lsa.append(nDCG)
			MAPs_lsa.append(MAP)
			
			print(f"[LSA] Precision, Recall and F-score @ {k} : {precision:.4f}, {recall:.4f}, {fscore:.4f}")
			print(f"[LSA] MAP, nDCG @ {k} : {MAP:.4f}, {nDCG:.4f}")

		# Plot the metrics and save plot 
		plt.plot(range(1, 11), precisions_vsm, label="Precision_VSM", linestyle="dashed", color='b')
		plt.plot(range(1, 11), precisions_lsa, label="Precision_LSA", color='b')
		plt.plot(range(1, 11), recalls_vsm, label="Recall_VSM", linestyle="dashed", color='r')
		plt.plot(range(1, 11), recalls_lsa, label="Recall_LSA", color='r')
		plt.plot(range(1, 11), fscores_vsm, label="F-Score_VSM", linestyle="dashed", color='g')
		plt.plot(range(1, 11), fscores_lsa, label="F-Score_LSA", color='g')
		plt.plot(range(1, 11), MAPs_vsm, label="MAP_VSM", linestyle="dashed", color='m')
		plt.plot(range(1, 11), MAPs_lsa, label="MAP_LSA", color='m')
		plt.plot(range(1, 11), nDCGs_vsm, label="nDCG_VSM", linestyle="dashed", color='k')
		plt.plot(range(1, 11), nDCGs_lsa, label="nDCG_LSA", color='k')
		plt.title("VSM vs LSA - Evaluation Metrics - Cranfield Dataset")
		plt.xlabel("k")
		plt.xticks(np.arange(1, 11, step=1))
		plt.yticks(np.arange(0, 0.75, step=0.05))
		plt.legend(bbox_to_anchor=(1.04, 0.5), 
				   loc="center left", 
				   borderaxespad=0)
		plt.savefig(f"{args.out_folder}vsm_lsa_comparison.png",
					bbox_inches="tight")
		plt.show()


		#---------------------------------------------------------------------------------------------------------------------------#

	
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
	parser.add_argument("-spellcheck", action="store_true",
						help = "Use SpellCheck ")
	
	args = parser.parse_args()
	SearchEngine(args).evaluateDataset()

