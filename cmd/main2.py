# main.py
import ujson
import dspy
from sentence_transformers import SentenceTransformer
from dspy.retrievers import Embeddings
from typing import List

# -----------------------------
# CONFIGURATION
# -----------------------------
CORPUS_FILE = "rag_corpus.jsonl"  # From ingest.py
TOP_K = 5  # Number of relevant chunks to retrieve
MAX_CHARS = 6000  # Truncate long chunks
EMBEDDING_MODEL = "intfloat/e5-base"  # Local embedding model
OLLAMA_MODEL = "mistral"  # Ollama must be running this model


# -----------------------------
# 1. Load Corpus
# -----------------------------
def load_corpus(jsonl_path: str) -> List[str]:
    with open(jsonl_path) as f:
        return [ujson.loads(line)['text'][:MAX_CHARS] for line in f]


corpus = load_corpus(CORPUS_FILE)
print(f"✅ Loaded {len(corpus)} documents from corpus")


# -----------------------------
# 2. Local Embedder for DSPy
# -----------------------------
class LocalSBERTEmbedder(dspy.Embedder):
    def __init__(self):
        super().__init__()
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def embed(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)


embedder = LocalSBERTEmbedder()

# -----------------------------
# 3. DSPy Retriever
# -----------------------------
search = Embeddings(embedder=embedder, corpus=corpus, k=TOP_K)


# -----------------------------
# 4. RAG with Classification + Response
# -----------------------------
class TriageModule(dspy.Signature):
    """A module to classify and respond to GitHub issues."""
    context = dspy.InputField(desc="Relevant code and documentation")
    issue = dspy.InputField(desc="GitHub issue content")

    classification = dspy.OutputField(desc="bug, feature-request, question, enhancement, etc.")
    response = dspy.OutputField(desc="Suggested GitHub comment reply")


class RAGTriageAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.triage = dspy.ChainOfThought(TriageModule)

    def forward(self, issue):
        retrieved = search(issue).passages
        context = "\n\n".join(retrieved)
        return self.triage(context=context, issue=issue)


# -----------------------------
# 5. Run with Ollama (Mistral)
# -----------------------------
dspy.settings.configure(
    llm=dspy.OllamaLocal(model=OLLAMA_MODEL)
)

# EXAMPLE: GitHub Issue (e.g., https://github.com/openshift-pipelines/pipelines-as-code/issues/1951)
issue_text = """
We need to handle the GitHub `pull_request_review` event, especially the `submitted` and `declined` actions. 
This should allow pipelines-as-code to react to PR reviews and take action accordingly.
"""

agent = RAGTriageAgent()
result = agent.forward(issue_text)

# -----------------------------
# 6. Output Results
# -----------------------------
print("\n🔍 Classification:")
print(result.classification)

print("\n💬 Suggested Comment:")
print(result.response)