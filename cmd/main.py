# main.py
import json

import dspy
import faiss
from langchain.vectorstores import FAISS
# from langchain.document_loaders import Document
from typing import List
import os
from langchain_ollama import OllamaLLM
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import List

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


app = FastAPI()

class IssueRequest(BaseModel):
    issue: str

@app.post("/triage")
async def triage_issue(request: IssueRequest):
    agent = TriageAgent()
    try:
        result = agent.forward(issue=request.issue)
        return JSONResponse({
            "classification": result.classification,
            "response": result.response
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

class RequestHandler(BaseHTTPRequestHandler):
    def _send_response(self, response_data):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response_data).encode())

    def do_POST(self):
        if self.path == "/triage":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)

            try:
                request_data = json.loads(post_data)
                issue_text = request_data.get("issue", "")

                agent = TriageAgent()
                result = agent.forward(issue=issue_text)

                response_data = {
                    "classification": result.classification,
                    "response": result.response
                }
                self._send_response(response_data)

            except Exception as e:
                response_data = {"error": str(e)}
                self.send_response(500)
                self._send_response(response_data)


def run(server_class=HTTPServer, handler_class=RequestHandler, port=8000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting HTTP server on port {port}')
    httpd.serve_forever()


if __name__ == "__main__":
    run(port=8000)