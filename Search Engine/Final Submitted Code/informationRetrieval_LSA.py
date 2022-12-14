import numpy as np
from numpy.linalg import norm

class InformationRetrieval:

    def __init__(self):
        self.matrix = None

    def buildIndexWithSVD(self, docs, docIDs, n_comp=750):
        """
        Builds the document index in terms of the document
        IDs and stores it in the 'index' class variable

        Parameters
        ----------
        arg1 : list
                A list of lists of lists where each sub-list is
                a document and each sub-sub-list is a sentence of the document
        arg2 : list
                A list of integers denoting IDs of the documents
        Returns
        -------
        None
        """
        N = len(docIDs)
        for i in range(N):
            docIDs[i] -= 1

        words = []
        for doc in docs:
            for sent in doc:
                for word in sent:
                    word = word.lower()
                    if '-' in word:
                        words.extend(word.split('-'))
                    else:
                        words.append(word)

        unique_words = list(set(words))
        postings = dict((v, k) for (k, v) in enumerate(unique_words))
        matrix = np.zeros((len(unique_words), N))

        for idx in docIDs:
            for sent in docs[idx]:
                for word in sent:
                    word = word.lower()
                    if '-' in word:
                        for w in word.split('-'):
                            matrix[postings[w]][idx] += 1
                    else:
                        matrix[postings[word]][idx] += 1

        # compute idf with smoothening
        idf = np.zeros(len(unique_words))
        for i, word in enumerate(unique_words):
            n = matrix[postings[word]].sum()
            idf[i] = np.log((N + 1)/(n + 1)) + 1

        idf = idf.reshape(-1, 1)
        matrix *= idf

        U, s, Vt = np.linalg.svd(matrix)
        matrix_recon = np.linalg.multi_dot([U[:, :n_comp], np.diag(s[:n_comp]), Vt[:n_comp]])

        self.docIDs = docIDs
        self.idf = idf
        self.unique_words = unique_words
        self.matrix = matrix_recon
        self.postings = postings


    def rank(self, queries):
        """
        Rank the documents according to relevance for each query

        Parameters
        ----------
        arg1 : list
                A list of lists of lists where each sub-list is a query and
                each sub-sub-list is a sentence of the query


        Returns
        -------
        list
                A list of lists of integers where the ith sub-list is a list of IDs
                of documents in their predicted order of relevance to the ith query
        """
        q_mat = np.zeros((len(self.unique_words), len(queries)))
        for i, query in enumerate(queries):
            for sent in query:
                for word in sent:
                    if '-' in word:
                        for w in word.split('-'):
                            try:
                                q_mat[self.postings[w]][i] += 1
                            except KeyError:
                                pass
                    else:
                        try:
                            q_mat[self.postings[word]][i] += 1
                        except KeyError:
                            pass

        q_mat *= self.idf
        q_norms = np.array([np.linalg.norm(q) for q in q_mat.T])
        d_norms = np.array([np.linalg.norm(d) for d in self.matrix.T])
        norms_prod = np.outer(q_norms, d_norms)
        sim = np.dot(q_mat.T, self.matrix) / (norms_prod + 1e-8)
        return np.flip(np.argsort(sim, axis=1, kind="mergesort")+1, axis=1)