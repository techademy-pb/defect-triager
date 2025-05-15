# main.py
import dspy
import faiss
from langchain.vectorstores import FAISS
# from langchain.document_loaders import Document
from typing import List
import os
from langchain_ollama import OllamaLLM

from langchain_community.embeddings import HuggingFaceEmbeddings


print(dir(dspy))

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
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)


def retrieve_context(issue_text: str, k=5) -> List[str]:
    docs = db.similarity_search(issue_text, k=k)
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
Handle pull_request_review event on Github

They are basically issue_comment and can be handle the same but with a different event type "pull_request_review_{submitted, declined}"

"""

# Run the agent
agent = TriageAgent()

result = agent.forward(issue=sample_issue)

# # Print result
print(f"--- Classification ---\n{result.classification}")
print(f"\n--- Suggested Comment ---\n{result.response}")

