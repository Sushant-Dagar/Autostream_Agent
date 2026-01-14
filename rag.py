"""
RAG (Retrieval-Augmented Generation) pipeline for AutoStream AI Agent.
Uses FAISS for vector storage and retrieval.
"""

import json
import os
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class KnowledgeBase:
    """RAG-powered knowledge base for AutoStream product information."""

    def __init__(self, knowledge_file: str = "knowledge_base.json", embeddings=None):
        self.knowledge_file = knowledge_file
        self.embeddings = embeddings
        self.vector_store: Optional[FAISS] = None
        self.documents: List[Document] = []

    def load_and_process(self) -> None:
        """Load knowledge base and create vector store."""
        # Load JSON data
        with open(self.knowledge_file, "r") as f:
            data = json.load(f)

        # Create documents from different sections
        documents = []

        # Company info
        company = data["company"]
        documents.append(
            Document(
                page_content=f"Company: {company['name']}. {company['description']}",
                metadata={"source": "company_info", "type": "general"},
            )
        )

        # Pricing - Basic Plan
        basic = data["pricing"]["basic_plan"]
        basic_content = f"""Basic Plan Pricing:
Price: {basic['price']}
Features: {', '.join(basic['features'])}
The Basic plan is great for beginners and casual content creators who need standard video editing capabilities."""
        documents.append(
            Document(
                page_content=basic_content,
                metadata={"source": "pricing", "type": "basic_plan"},
            )
        )

        # Pricing - Pro Plan
        pro = data["pricing"]["pro_plan"]
        pro_content = f"""Pro Plan Pricing:
Price: {pro['price']}
Features: {', '.join(pro['features'])}
The Pro plan is ideal for professional content creators who need unlimited videos, 4K resolution, and AI-powered features like automatic captions."""
        documents.append(
            Document(
                page_content=pro_content,
                metadata={"source": "pricing", "type": "pro_plan"},
            )
        )

        # Pricing comparison
        comparison_content = f"""AutoStream Pricing Comparison:
- Basic Plan: {basic['price']} - Includes {basic['features'][0]}, {basic['features'][1]}
- Pro Plan: {pro['price']} - Includes unlimited videos, 4K resolution, AI captions, 24/7 support

The Pro plan costs ${79-29}=$50 more per month but offers significantly more features including unlimited videos and AI-powered captions."""
        documents.append(
            Document(
                page_content=comparison_content,
                metadata={"source": "pricing", "type": "comparison"},
            )
        )

        # Policies
        policies = data["policies"]
        for policy_name, policy_content in policies.items():
            documents.append(
                Document(
                    page_content=f"{policy_name.replace('_', ' ').title()} Policy: {policy_content}",
                    metadata={"source": "policies", "type": policy_name},
                )
            )

        # FAQs
        for faq in data["faqs"]:
            documents.append(
                Document(
                    page_content=f"Q: {faq['question']}\nA: {faq['answer']}",
                    metadata={"source": "faq", "type": "faq"},
                )
            )

        self.documents = documents

        # Create vector store
        if self.embeddings:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """Retrieve relevant documents for a query."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call load_and_process() first.")

        return self.vector_store.similarity_search(query, k=k)

    def get_context(self, query: str, k: int = 3) -> str:
        """Get formatted context string for a query."""
        docs = self.retrieve(query, k=k)
        context_parts = []
        for doc in docs:
            context_parts.append(doc.page_content)
        return "\n\n---\n\n".join(context_parts)

    def get_all_content(self) -> str:
        """Get all knowledge base content as a formatted string."""
        return "\n\n---\n\n".join([doc.page_content for doc in self.documents])


def create_knowledge_base(embeddings) -> KnowledgeBase:
    """Factory function to create and initialize a knowledge base."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    kb_path = os.path.join(script_dir, "knowledge_base.json")

    kb = KnowledgeBase(knowledge_file=kb_path, embeddings=embeddings)
    kb.load_and_process()
    return kb
