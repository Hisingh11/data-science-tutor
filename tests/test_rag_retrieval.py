import unittest

from utils.rag_engine import RAGEngine


class RAGRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rag = RAGEngine()
        cls.rag.load_initial_knowledge()

    def test_financial_projection_does_not_receive_unrelated_data_science_notes(self):
        query = (
            "Act as a financial analyst. Prepare a 10-year year-wise projection for a "
            "monthly fixed deposit under the Indian tax regime. Assumptions: FD interest "
            "rate 6.25% per annum, monthly investment ₹1,00,000. Use an effective tax "
            "rate of 31.2%, account for TDS, report total tax liability, and show "
            "year-wise closing balances with tax deducted before reinvestment."
        )

        self.assertEqual(self.rag.relevant(query, n_results=4), [])

    def test_p_value_question_still_retrieves_its_knowledge_note(self):
        hits = self.rag.relevant("What is a p-value?", n_results=4)

        self.assertEqual(hits[0]["metadata"]["topic"], "Hypothesis testing and p-values")

    def test_learning_rate_question_still_retrieves_gradient_descent_note(self):
        hits = self.rag.relevant("learning rate too high loss diverges", n_results=4)

        self.assertEqual(hits[0]["metadata"]["topic"], "Gradient descent")

    def test_generic_value_word_does_not_retrieve_p_value_note(self):
        hits = self.rag.relevant("what is the best value for money laptop", n_results=4)

        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
