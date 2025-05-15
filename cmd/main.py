# main.py
import dspy
import faiss
from langchain.vectorstores import FAISS
# from langchain.document_loaders import Document
from typing import List
import os
from langchain_ollama import OllamaLLM
import numpy as np


from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
model_name = "ollama_chat/mistral"
# Use Ollama local model
# dspy.settings.configure(
#     llm=dspy.OllamaLLM(model=model_name)
# )


lm = dspy.LM(model_name, api_base='http://localhost:11434', api_key='')
# dspy.settings.configure(lm=lm)
dspy.configure(lm=lm)


embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
# llm = OllamaLLM(model=model_name)
# Load vector store
db = FAISS.load_local("faiss_index_2", embeddings, allow_dangerous_deserialization=True)


def retrieve_context(issue_text: str, k=5) -> List[str]:
    print(f"Retrieving context for issue")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(issue_text)
    chunk_embeddings = [embeddings.embed_query(chunk) for chunk in chunks]
    avg_embedding = np.mean(chunk_embeddings, axis=0)

    # docs = db.similarity_search(issue_text, k=k)

    docs = db.similarity_search_by_vector(avg_embedding.tolist(), k=k)

    print(f"Retrieved {len(docs)} docs")
    return [doc.page_content for doc in docs]


# DSPy Module for Issue Classification and Response
class TriageModule(dspy.Signature):
    """Classify a GitHub issue and respond using context."""

    issue = dspy.InputField(desc="The content of the GitHub issue")
    context = dspy.InputField(desc="Documentation snippets related to the issue")

    classification = dspy.OutputField(desc="bug, feature-request, question, enhancement, etc.")
    response = dspy.OutputField(desc="A helpful, friendly GitHub comment")


class TriageAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.program = dspy.ChainOfThought(TriageModule)

    def forward(self, issue):
        context_docs = retrieve_context(issue)
        context = "\n\n".join(context_docs)
        return self.program(issue=issue, context=context)


# Example issue
sample_issue = """
Build Status Not Updated When PipelineRun is Deleted.  In Bitbucket, the build status does not change if a PipelineRun is manually deleted by the user (in the OpenShift Web Console for example). Currently, the status only updates if the pipeline is successfully completed or canceled. However, if the PipelineRun is deleted, the status remains "in progress" indefinitely.

Desired Behavior:
When a PipelineRun is deleted by the user, the build status in Bitbucket should be updated accordingly, indicating that the pipeline no longer exists and is not in progress --> status Failed.

"""

# Run the agent
agent = TriageAgent()

try:
    result = agent.forward(issue=sample_issue)

except Exception as e:
    print(f"Error: {e}")

# # Print result
print(f"--- Classification ---\n{result.classification}")
print(f"\n--- Suggested Comment ---\n{result.response}")

