import json
import math
import os
import re
from typing import Dict, List

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "from",
    "is", "are", "was", "were", "be", "been", "being", "what", "why", "how", "when",
    "where", "who", "which", "that", "this", "these", "those", "it", "its", "as",
    "at", "by", "about", "into", "than", "then", "can", "could", "would", "should",
    "do", "does", "did", "me", "my", "we", "our", "you", "your", "please", "explain",
    "tell", "give", "difference", "between", "versus", "vs", "using", "use", "used",
}

GREETINGS = {"hi", "hello", "hey", "thanks", "thank", "ok", "okay", "yo"}

EXPANSIONS = {
    "rag": ["retrieval", "augmented", "generation"],
    "crag": ["corrective", "retrieval", "augmented"],
    "llm": ["language", "model", "transformer"],
    "llms": ["language", "model", "transformer"],
    "eda": ["exploratory", "analysis"],
    "ml": ["machine", "learning"],
    "ai": ["artificial", "intelligence"],
    "overfitting": ["overfit", "generalization", "variance"],
    "overfit": ["overfitting", "generalization"],
    "underfitting": ["underfit", "bias"],
    "underfit": ["underfitting", "bias"],
    "pca": ["principal", "component", "dimensionality"],
    "auc": ["roc", "curve"],
    "roc": ["auc", "curve"],
    "f1": ["precision", "recall"],
    "react": ["reason", "act", "agent"],
    "finetuning": ["fine", "tuning"],
    "finetune": ["fine", "tuning"],
    "cnn": ["convolutional", "neural"],
    "rnn": ["recurrent", "neural"],
    "nlp": ["language", "text"],
    "svm": ["support", "vector"],
    "knn": ["nearest", "neighbors"],
}

KNOWLEDGE = [
    {
        "text": "Data science turns raw data into decisions. Artificial intelligence (AI) is the broader field of making software behave intelligently. Machine learning (ML) is the part of AI that learns patterns from data instead of following only hand-written rules. A typical data science lifecycle is: define the problem and the metric that would count as success, collect data, clean it, explore it, engineer features, choose and train a model, evaluate on data the model did not train on, deploy, and monitor for drift. Skipping the problem definition or the holdout evaluation is the usual reason a project looks good in a notebook and fails in production.",
        "metadata": {"topic": "Data science lifecycle"},
    },
    {
        "text": "Supervised learning trains on examples that already have labels. Regression predicts a number, such as house price. Classification predicts a category, such as spam or not spam. Unsupervised learning has no labels. Clustering groups similar rows, and principal component analysis (PCA) compresses many columns into fewer components that keep most of the variation. Use supervised learning when the outcome you care about is already recorded.",
        "metadata": {"topic": "Supervised and unsupervised learning"},
    },
    {
        "text": "Overfitting means the model memorized the training set, including its noise, so training score is high and performance on new data is poor. Underfitting means the model is too simple to capture the real pattern, so both training and new data look bad. The bias-variance tradeoff is that simple models have high bias and low variance, while very flexible models have low bias and high variance. Reduce overfitting with more data, simpler models, regularization, dropout, or early stopping. Reduce underfitting with a more flexible model or better features.",
        "metadata": {"topic": "Overfitting and bias-variance"},
    },
    {
        "text": "Never evaluate a model only on the rows it trained on. Split data into training, validation, and test. Use validation to choose hyperparameters and the test set once, at the end. K-fold cross-validation rotates which fold is held out, which gives a steadier estimate when the dataset is small. Stratified folds keep the class proportions stable. Data leakage is when information from the future or from the label sneaks into the features, and it makes validation scores look falsely high.",
        "metadata": {"topic": "Validation and data leakage"},
    },
    {
        "text": "For classification, accuracy is the share of correct predictions and is misleading when classes are imbalanced. Precision is the share of positive predictions that were truly positive. Recall is the share of actual positives that the model found. The F1 score is the harmonic mean of precision and recall. A ROC curve plots true positive rate against false positive rate as the threshold moves, and AUC summarizes that curve. Use precision when false alarms are costly, and recall when missing a positive is costly.",
        "metadata": {"topic": "Classification metrics"},
    },
    {
        "text": "For regression, mean absolute error (MAE) is the average absolute gap between prediction and truth. Mean squared error (MSE) squares the gaps, so large mistakes count more. Root mean squared error (RMSE) is the square root of MSE and uses the same units as the target. R-squared says what fraction of the variation in the target the model explains. A low RMSE with a low R-squared can happen when the target itself barely varies.",
        "metadata": {"topic": "Regression metrics"},
    },
    {
        "text": "Exploratory data analysis (EDA) happens before modeling. Check shape, dtypes, missing values, duplicates, and impossible values. Plot distributions and look for skew. Compare groups with the target. A correlation heatmap shows linear relationships, but correlation is not causation and it misses curved relationships. The point of EDA is to decide what to clean and which features are worth engineering, not to prove the final model.",
        "metadata": {"topic": "Exploratory data analysis"},
    },
    {
        "text": "Feature engineering creates columns the model can use. Examples are extracting the hour from a timestamp, combining related fields, or binning a continuous value. Scaling matters for distance-based models and gradient descent: standardization subtracts the mean and divides by the standard deviation, min-max scaling squashes values into a range. Fit the scaler on the training fold only, then apply it to validation and test, or the scale itself leaks information.",
        "metadata": {"topic": "Feature engineering and scaling"},
    },
    {
        "text": "Regularization penalizes large weights so the model stays simpler. L2 regularization (ridge) shrinks weights toward zero but rarely makes them exactly zero. L1 regularization (lasso) can set some weights to exactly zero, which also selects features. Elastic net mixes both. A stronger penalty reduces variance and can increase bias. Choose the penalty strength on the validation set, not by staring at the training loss.",
        "metadata": {"topic": "L1 and L2 regularization"},
    },
    {
        "text": "Gradient descent updates weights by stepping opposite the gradient of the loss. Batch gradient descent uses the whole dataset each step. Stochastic gradient descent uses one row. Mini-batch is the usual compromise. The learning rate controls the step size: too large and the loss diverges, too small and training crawls. Adam adapts the step size per weight and is a common default for neural networks. Always shuffle training rows.",
        "metadata": {"topic": "Gradient descent"},
    },
    {
        "text": "A decision tree splits features with yes-or-no questions. A single deep tree overfits easily. A random forest builds many trees on bootstrap samples and random feature subsets, then averages them, which lowers variance. Boosting, including gradient boosting and XGBoost, builds trees one after another, each one correcting the previous errors. Boosting often wins on tabular data but needs careful tuning of learning rate and tree depth so it does not overfit.",
        "metadata": {"topic": "Trees, forests, and boosting"},
    },
    {
        "text": "A neural network stacks layers of weighted sums and nonlinear functions. Backpropagation computes the gradient of the loss through those layers. The vanishing gradient problem appears when gradients shrink in deep stacks and early layers stop learning. Transformers replaced recurrence for text. Self-attention lets each token look at the other tokens and weigh them. The attention formula scales the query-key dot products by the square root of the key dimension before the softmax.",
        "metadata": {"topic": "Neural networks and transformers"},
    },
    {
        "text": "Retrieval-augmented generation (RAG) answers a question by first searching a knowledge base, then giving the retrieved passages to the language model. The model is grounded in those passages instead of relying only on memory. RAG helps when facts change or live in private documents. It fails when the search returns the wrong passages, because the model will confidently use them. Chunk size, the retrieval score, and a check that the passage actually answers the question all matter.",
        "metadata": {"topic": "Retrieval-augmented generation"},
    },
    {
        "text": "CRAG is the corrective step that runs after a knowledge-base search. Each passage is graded as useful, only partly related, or irrelevant. Useful passages are trimmed to the sentences that support the answer. Irrelevant passages are thrown away and a web search is used instead. Partly related passages are kept together with those web results. The point of CRAG is to stop a wrong search hit from becoming a wrong answer.",
        "metadata": {"topic": "Corrective RAG"},
    },
    {
        "text": "Prompt engineering changes the instructions and examples you send a model, without changing its weights. Zero-shot asks with no examples. Few-shot includes a handful of examples. Fine-tuning continues training on your own examples and changes the weights, which costs more and is worth it when the style or the task is stable. Retrieval is usually a better first step than fine-tuning when the problem is missing facts rather than missing behavior.",
        "metadata": {"topic": "Prompting and fine-tuning"},
    },
    {
        "text": "A hallucination is a fluent answer that is not supported by the source or by the facts. Models hallucinate more when the prompt is vague, when retrieved text is irrelevant, or when they are pushed to answer anyway. Reduce it by retrieving relevant sources, asking the model to abstain when the notes do not contain the answer, keeping temperature lower for factual questions, and checking names, numbers, and library APIs before you trust them.",
        "metadata": {"topic": "Hallucinations"},
    },
    {
        "text": "An AI agent is a model that can plan, use tools, and keep memory, rather than only reply once. The ReAct pattern alternates a reasoning step with an action, such as a search or a function call, then reads the observation. Tools are APIs the agent is allowed to call. Memory can be the chat so far or a vector store of older notes. Multi-agent systems split work across specialized agents. Production agents need limits on which tools they may call and a way for a person to stop them.",
        "metadata": {"topic": "Agents and ReAct"},
    },
    {
        "text": "Embeddings turn text or rows into vectors so that similar items sit near each other. They are how semantic search and RAG find a passage even when it does not share the exact keywords. Cosine similarity compares direction, ignoring length. Embeddings do not replace a relevance check: a nearby passage can still be about the wrong fact. For a small curated knowledge base, a keyword ranker such as BM25 can be more precise than a general embedding model.",
        "metadata": {"topic": "Embeddings"},
    },
    {
        "text": "Class imbalance means one class is rare, as in fraud or disease. Accuracy then looks high if the model always predicts the common class. Prefer precision, recall, F1, or average precision. Practical fixes include collecting more rare cases, class weights, and resampling such as SMOTE. Changing the decision threshold is often more effective than changing the model. Evaluate with a split that still contains the rare class.",
        "metadata": {"topic": "Imbalanced data"},
    },
    {
        "text": "Pandas is the usual Python tool for tables. A DataFrame holds columns, and loc or boolean masks select rows. Missing values show up as NaN. NumPy provides arrays and vectorized math that is much faster than Python loops. Scikit-learn fits models with fit and predicts with predict. Put preprocessing and the model in a Pipeline so the same steps run on new data. Matplotlib and seaborn draw the charts used during exploration.",
        "metadata": {"topic": "Python for data science"},
    },
]


class RAGEngine:
    """BM25 retrieval plus a corrective check (CRAG).

    Retrieved notes are graded. Notes that do not answer the question are
    dropped. Web search fills the gap when the local notes are irrelevant
    or only partly related.
    """

    def __init__(self, persist_directory="./data/knowledge_base"):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        self.documents: List[Dict] = []
        self._loaded = False
        self._index = None

    def add_documents(self, documents: List[Dict]):
        for doc in documents:
            self.documents.append({
                "text": doc["text"],
                "metadata": doc.get("metadata", {}),
            })
        self._build_index()

    def load_initial_knowledge(self):
        if self._loaded:
            return len(self.documents)
        self.add_documents(KNOWLEDGE)
        self._loaded = True
        return len(self.documents)

    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        if not self.documents or not self._index:
            return []
        cleaned = self._clean_query(query)
        tokens = self._query_tokens(cleaned)
        if not tokens:
            return []
        scores = self._index.score(tokens)
        ranked = sorted(
            ((score, idx) for idx, score in enumerate(scores) if score > 0),
            key=lambda item: item[0],
            reverse=True,
        )[:n_results]
        if not ranked:
            return []
        top = ranked[0][0]
        results = []
        for score, idx in ranked:
            doc = self.documents[idx]
            results.append({
                "text": doc["text"],
                "metadata": doc["metadata"],
                "relevance": score / top if top else 0.0,
                "score": score,
            })
        return results

    def corrective_retrieve(self, query: str, model=None, n_results: int = 4) -> Dict:
        cleaned = self._clean_query(query)
        if not self._worth_retrieving(cleaned):
            return self._bundle("skip", "", [])

        hits = self.search(cleaned, n_results=n_results)
        graded = self._grade(cleaned, hits, model) if hits else []
        correct = [item for item in graded if item["grade"] == "correct"]
        ambiguous = [item for item in graded if item["grade"] == "ambiguous"]

        if not graded and hits:
            # The grader was unavailable. Keep a hit only when the question's
            # own words, not just expanded synonyms, appear in it.
            anchored = [hit for hit in hits if self._anchored(cleaned, hit["text"])]
            if anchored:
                correct = [{
                    "grade": "correct",
                    "strip": anchored[0]["text"],
                    "metadata": anchored[0]["metadata"],
                    "source": "knowledge",
                }]

        web_notes: List[Dict] = []
        if correct:
            action = "correct"
            chosen = correct
        elif ambiguous:
            action = "ambiguous"
            chosen = ambiguous
            web_notes = self._web_search(cleaned)
        else:
            action = "incorrect"
            chosen = []
            web_notes = self._web_search(cleaned)

        blocks = []
        sources = []
        for item in chosen:
            topic = item["metadata"].get("topic", "note")
            blocks.append(f"[Knowledge base: {topic}]\n{item['strip']}")
            sources.append({"title": topic, "kind": "knowledge"})
        for note in web_notes:
            blocks.append(f"[Web: {note['title']}]\n{note['body']}")
            sources.append({"title": note["title"], "url": note.get("href", ""), "kind": "web"})

        return self._bundle(action, "\n\n".join(blocks), sources)

    def _bundle(self, action: str, body: str, sources: List[Dict]) -> Dict:
        instructions = {
            "correct": (
                "Retrieval check: the local notes below directly support this question. "
                "Base the factual parts of the answer on them. Ignore a note if it does not apply."
            ),
            "ambiguous": (
                "Retrieval check: the local notes are only partly related, so short web snippets were added. "
                "Prefer the local notes if they conflict with a snippet. Do not invent citations."
            ),
            "incorrect": (
                "Retrieval check: the local knowledge base did not contain this answer. "
                "Web snippets are unverified leads. Use them only when they clearly answer the question. "
                "If they are off topic, ignore them and answer from standard data science knowledge, "
                "or say you are not sure."
            ),
            "skip": "",
        }
        context = ""
        if body:
            context = instructions[action] + "\n\n" + body
        elif action == "incorrect":
            context = instructions[action]
        return {"action": action, "context": context, "sources": sources}

    def _grade(self, query: str, hits: List[Dict], model) -> List[Dict]:
        if model is None or not getattr(model, "client", None):
            return []
        numbered = []
        for idx, hit in enumerate(hits, start=1):
            numbered.append(f"{idx}. {hit['text']}")
        prompt = (
            "Grade each passage for a data science tutor.\n"
            f"Question: {query}\n\n"
            "Passages:\n"
            + "\n".join(numbered)
            + "\n\nFor each passage set grade to correct, ambiguous, or incorrect.\n"
            "correct: the passage contains the facts needed to answer.\n"
            "ambiguous: same general topic, but it does not answer this question.\n"
            "incorrect: a different topic.\n"
            "strip: if correct or ambiguous, the one or two sentences that help. Otherwise empty."
        )
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "grades": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "id": {"type": "integer"},
                            "grade": {"type": "string"},
                            "strip": {"type": "string"},
                        },
                        "required": ["id", "grade", "strip"],
                    },
                }
            },
            "required": ["grades"],
        }
        parsed = None
        if hasattr(model, "complete_json"):
            wrapped = model.complete_json(prompt, schema, "passage_grades", "fast", 0.0)
            if isinstance(wrapped, dict):
                parsed = wrapped.get("grades")
        if not isinstance(parsed, list):
            raw = model.complete(prompt + '\nReturn only a JSON list like [{"id": 1, "grade": "correct", "strip": "..."}].', "fast", 0.0)
            if not raw or raw.startswith("Error:"):
                return []
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if not match:
                return []
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                return []
        graded = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("id", 0)) - 1
            except (TypeError, ValueError):
                continue
            if idx < 0 or idx >= len(hits):
                continue
            grade = str(item.get("grade", "")).strip().lower()
            if grade not in {"correct", "ambiguous", "incorrect"}:
                continue
            strip = str(item.get("strip") or "").strip()
            if grade != "incorrect" and len(strip) < 40:
                strip = hits[idx]["text"]
            graded.append({
                "grade": grade,
                "strip": strip[:700],
                "metadata": hits[idx]["metadata"],
                "source": "knowledge",
            })
        return graded

    def _web_search(self, query: str, max_results: int = 3) -> List[Dict]:
        try:
            from ddgs import DDGS
            rows = DDGS().text(query, max_results=max_results) or []
        except Exception:
            return []
        notes = []
        for row in rows:
            title = (row.get("title") or "").strip()
            body = (row.get("body") or "").strip()
            if not body:
                continue
            notes.append({
                "title": title or "Web result",
                "body": body[:360],
                "href": row.get("href", ""),
            })
        return notes

    def _build_index(self):
        self._index = BM25Index([self._doc_tokens(doc["text"]) for doc in self.documents])

    def _clean_query(self, query: str) -> str:
        text = query or ""
        text = re.split(r"\[(Image analysis|Attached file|Attached PDF)", text, maxsplit=1)[0]
        text = re.sub(r"\s+", " ", text).strip()
        return text[:500]

    def _tokens(self, text: str) -> List[str]:
        keep = set(EXPANSIONS)
        return [
            token for token in re.findall(r"[a-z0-9]+", text.lower())
            if token not in STOPWORDS and (len(token) > 2 or token in keep)
        ]

    def _doc_tokens(self, text: str) -> List[str]:
        return self._tokens(text)

    def _query_tokens(self, query: str) -> List[str]:
        original = self._tokens(query)
        extra = []
        for token in original:
            extra.extend(EXPANSIONS.get(token, []))
        # Original words count twice so a synonym cannot outrank the question.
        return original + original + extra

    def _worth_retrieving(self, query: str) -> bool:
        tokens = self._tokens(query)
        if not tokens:
            return False
        return any(token not in GREETINGS for token in tokens)

    def _anchored(self, query: str, text: str) -> bool:
        needles = set(self._tokens(query))
        haystack = set(self._doc_tokens(text))
        return len(needles & haystack) >= 1


class BM25Index:
    def __init__(self, docs: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.n = len(docs)
        self.avgdl = sum(len(doc) for doc in docs) / self.n if self.n else 0.0
        self.df: Dict[str, int] = {}
        for doc in docs:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1

    def score(self, query_tokens: List[str]) -> List[float]:
        scores = []
        for doc in self.docs:
            length = len(doc) or 1
            counts: Dict[str, int] = {}
            for term in doc:
                counts[term] = counts.get(term, 0) + 1
            total = 0.0
            for term in query_tokens:
                freq = counts.get(term)
                if not freq:
                    continue
                doc_freq = self.df.get(term, 0)
                idf = math.log(1 + (self.n - doc_freq + 0.5) / (doc_freq + 0.5))
                denom = freq + self.k1 * (1 - self.b + self.b * length / (self.avgdl or 1))
                total += idf * (freq * (self.k1 + 1)) / denom
            scores.append(total)
        return scores
