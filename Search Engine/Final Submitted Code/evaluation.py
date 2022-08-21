from math import log2

class Evaluation():

    def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of precision of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
                A list of integers denoting the IDs of documents in
                their predicted order of relevance to a query
        arg2 : int
                The ID of the query in question
        arg3 : list
                The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
                The k value

        Returns
        -------
        float
                The precision value as a number between 0 and 1
        """

        num_docs = len(query_doc_IDs_ordered)
        try:
            assert k <= num_docs, "Insufficient documents retrieved for given k"

            num_true_docs = 0
            for doc_ID in query_doc_IDs_ordered[:k]:
                if int(doc_ID) in true_doc_IDs:
                    num_true_docs += 1
            return num_true_docs / k

        except AssertionError as msg:
            print(msg)
            return -1

    def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of precision of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
                A list of lists of integers where the ith sub-list is a list of IDs
                of documents in their predicted order of relevance to the ith query
        arg2 : list
                A list of IDs of the queries for which the documents are ordered
        arg3 : list
                A list of dictionaries containing document-relevance
                judgements - Refer cran_qrels.json for the structure of each
                dictionary
        arg4 : int
                The k value

        Returns
        -------
        float
                The mean precision value as a number between 0 and 1
        """

        num_queries = len(query_ids)
        precisions = []

        try:
            assert len(doc_IDs_ordered) == len(query_ids), "Number of queries and documents do not match"

            for i in range(num_queries):
                query_doc = doc_IDs_ordered[i]
                query_id = int(query_ids[i])
                true_doc_ID = []

                for dict_ in qrels:
                    if int(dict_["query_num"]) == int(query_id):
                        true_doc_ID.append(int(dict_["id"]))
                precision = self.queryPrecision(query_doc, query_id, true_doc_ID, k)
                precisions.append(precision)

            try:
                assert len(precisions) != 0, "Empty list."
                return sum(precisions)/len(precisions)
            except AssertionError as msg:
                print(msg)
                return -1

        except AssertionError as msg:
            print(msg)
            return -1

    def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of recall of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
                A list of integers denoting the IDs of documents in
                their predicted order of relevance to a query
        arg2 : int
                The ID of the query in question
        arg3 : list
                The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
                The k value

        Returns
        -------
        float
                The recall value as a number between 0 and 1
        """

        num_docs = len(query_doc_IDs_ordered)
        num_true_docs = len(true_doc_IDs)
        try:
            assert k <= num_docs, "Error: Insufficient number of retreived documents for given k"
            num_docs_retrieved = 0
            for doc_ID in query_doc_IDs_ordered[:k]:
                if int(doc_ID) in true_doc_IDs:
                    num_docs_retrieved += 1
            return num_docs_retrieved/num_true_docs

        except AssertionError as msg:
            print(msg)
            return -1

    def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of recall of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
                A list of lists of integers where the ith sub-list is a list of IDs
                of documents in their predicted order of relevance to the ith query
        arg2 : list
                A list of IDs of the queries for which the documents are ordered
        arg3 : list
                A list of dictionaries containing document-relevance
                judgements - Refer cran_qrels.json for the structure of each
                dictionary
        arg4 : int
                The k value

        Returns
        -------
        float
                The mean recall value as a number between 0 and 1
        """

        num_queries = len(query_ids)
        recalls = []
        try:
            assert len(doc_IDs_ordered) == len(query_ids), "Number of queries and documents do not match"
            for i in range(num_queries):
                query_doc = doc_IDs_ordered[i]
                query_id = query_ids[i]
                true_doc_ID = []
                for dict_ in qrels:
                    if int(dict_["query_num"]) == int(query_id):
                        true_doc_ID.append(int(dict_["id"]))

                recall = self.queryRecall(query_doc, query_id, true_doc_ID, k)
                recalls.append(recall)

            try:
                assert len(recalls) != 0, "List is empty."
                return sum(recalls)/len(recalls)
            except AssertionError as msg:
                print(msg)
                return -1

        except AssertionError as msg:
            print(msg)
            return -1

    def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of fscore of the Information Retrieval System
        at a given value of k for a single query

        Parameters
        ----------
        arg1 : list
                A list of integers denoting the IDs of documents in
                their predicted order of relevance to a query
        arg2 : int
                The ID of the query in question
        arg3 : list
                The list of IDs of documents relevant to the query (ground truth)
        arg4 : int
                The k value

        Returns
        -------
        float
                The fscore value as a number between 0 and 1
        """

        fscore = 0
        precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
        recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
        if (precision > 0 and recall > 0):
            fscore = 2 * precision * recall / (precision + recall)

        return fscore

    def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of fscore of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
                A list of lists of integers where the ith sub-list is a list of IDs
                of documents in their predicted order of relevance to the ith query
        arg2 : list
                A list of IDs of the queries for which the documents are ordered
        arg3 : list
                A list of dictionaries containing document-relevance
                judgements - Refer cran_qrels.json for the structure of each
                dictionary
        arg4 : int
                The k value

        Returns
        -------
        float
                The mean fscore value as a number between 0 and 1
        """

        num_queries = len(query_ids)
        fscores = []
        try:
            assert len(doc_IDs_ordered) == len(query_ids), "Number of queries and documents do not match"
            for i in range(num_queries):
                query_doc = doc_IDs_ordered[i]
                query_id = query_ids[i]
                true_doc_ID = []

                for dict_ in qrels:
                    if int(dict_["query_num"]) == int(query_id):
                        true_doc_ID.append(int(dict_["id"]))
                fscore = self.queryFscore(query_doc, query_id, true_doc_ID, k)
                fscores.append(fscore)

            try:
                assert len(fscores) != 0, "Error! Empty list. Returning -1."
                return sum(fscores)/len(fscores)

            except AssertionError as msg:
                print(msg)
                return -1

        except AssertionError as msg:
            print(msg)
            return -1

    def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of nDCG of the Information Retrieval System
        at given value of k for a single query

        Parameters
        ----------
        arg1 : list
                A list of integers denoting the IDs of documents in
                their predicted order of relevance to a query
        arg2 : int
                The ID of the query in question
        arg3 : list
                The qrels list of dictionaries 
        arg4 : int
                The k value

        Returns
        -------
        float
                The nDCG value as a number between 0 and 1
        """

        num_docs = len(query_doc_IDs_ordered)
        try:
            assert k <= num_docs, "Insufficient documents retrieved for given k"
            rel_vals = {}
            rel_docs = []
            DCGk = 0
            IDCGk = 0

            for dict_ in true_doc_IDs:
                if int(dict_["query_num"]) == int(query_id):
                    id_ = int(dict_["id"])
                    relevance = 5 - dict_["position"]
                    rel_vals[int(id_)] = relevance
                    rel_docs.append(int(id_))

            for i in range(1, k+1):
                doc_ID = int(query_doc_IDs_ordered[i-1])
                if doc_ID in rel_docs:
                    relevance = rel_vals[doc_ID]
                    DCGk += (2 ** relevance - 1) / log2(i+1)

            ordered_vals = sorted(rel_vals.values(), reverse=True)
            num_docs = len(ordered_vals)

            for i in range(1, min(num_docs, k)+1):
                relevance = ordered_vals[i-1]
                IDCGk += (2**relevance-1)/log2(i+1)

            try:
                assert IDCGk != 0, "IDCGk is zero."
                nDCGk = DCGk/IDCGk
                return nDCGk

            except AssertionError as msg:
                print(msg)
                return -1

        except AssertionError as msg:
            print(msg)
            return -1

    def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of nDCG of the Information Retrieval System
        at a given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
                A list of lists of integers where the ith sub-list is a list of IDs
                of documents in their predicted order of relevance to the ith query
        arg2 : list
                A list of IDs of the queries for which the documents are ordered
        arg3 : list
                A list of dictionaries containing document-relevance
                judgements - Refer cran_qrels.json for the structure of each
                dictionary
        arg4 : int
                The k value

        Returns
        -------
        float
                The mean nDCG value as a number between 0 and 1
        """

        num_queries = len(query_ids)
        nDCGs = []
        try:
            assert len(doc_IDs_ordered) == len(query_ids), "Number of queries and documents do not match"

            for i in range(num_queries):
                query_doc = doc_IDs_ordered[i]
                query_id = int(query_ids[i])
                nDCG = self.queryNDCG(query_doc, query_id, qrels, k)
                nDCGs.append(nDCG)
            try:
                assert len(nDCGs) != 0, "No values are present."
                return sum(nDCGs)/len(nDCGs)

            except AssertionError as msg:
                print(msg)
                return -1

        except AssertionError as msg:
            print(msg)
            return -1

    def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
        Computation of average precision of the Information Retrieval System
        at a given value of k for a single query (the average of precision@i
        values for i such that the ith document is truly relevant)

        Parameters
        ----------
        arg1 : list
                A list of integers denoting the IDs of documents in
                their predicted order of relevance to a query
        arg2 : int
                The ID of the query in question
        arg3 : list
                The list of documents relevant to the query (ground truth)
        arg4 : int
                The k value

        Returns
        -------
        float
                The average precision value as a number between 0 and 1
        """

        num_true_docs = len(true_doc_IDs)
        num_docs_retrieved = len(query_doc_IDs_ordered)

        try:
            assert k <= num_docs_retrieved, "Insufficient documents retrieved for given k"
            relevances = []
            precisions = []

            for doc_ID in query_doc_IDs_ordered:
                if int(doc_ID) in true_doc_IDs:
                    relevances.append(1)
                else:
                    relevances.append(0)

            for i in range(1, k+1):
                precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, i)
                precisions.append(precision)

            precision_at_k = []
            for i in range(k):
                value = precisions[i]*relevances[i]
                precision_at_k.append(value)

            try:
                assert num_true_docs != 0, "No true documents are present."
                if(sum(relevances[:k]) != 0):
                    AveP = sum(precision_at_k)/sum(relevances[:k])
                else:
                    AveP = 0
                return AveP

            except AssertionError as msg:
                print(msg)
                return -1

        except AssertionError as msg:
            print(msg)
            return -1

    def meanAveragePrecision(self, doc_IDs_ordered, query_ids, qrels, k):
        """
        Computation of MAP of the Information Retrieval System
        at given value of k, averaged over all the queries

        Parameters
        ----------
        arg1 : list
                A list of lists of integers where the ith sub-list is a list of IDs
                of documents in their predicted order of relevance to the ith query
        arg2 : list
                A list of IDs of the queries
        arg3 : list
                A list of dictionaries containing document-relevance
                judgements - Refer cran_qrels.json for the structure of each
                dictionary
        arg4 : int
                The k value

        Returns
        -------
        float
                The MAP value as a number between 0 and 1
        """

        num_queries = len(query_ids)
        AvePs = []
        try:
            assert len(doc_IDs_ordered) == len(query_ids), "Number of queries and documents do not match"

            for i in range(num_queries):
                query_doc = doc_IDs_ordered[i]
                query_id = int(query_ids[i])
                true_doc_ID = []

                for dict_ in qrels:
                    if int(dict_["query_num"]) == int(query_id):
                        true_doc_ID.append(int(dict_["id"]))
                AveP = self.queryAveragePrecision(
                    query_doc, query_id, true_doc_ID, k)
                AvePs.append(AveP)

            try:
                assert len(AvePs) != 0, "List is empty"
                return sum(AvePs)/len(AvePs)

            except AssertionError as msg:
                print(msg)
                return -1

        except AssertionError as msg:
            print(msg)
            return -1