from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        if not question or not question.strip():
            return "Vui lòng nhập câu hỏi hợp lệ."

        if self.store.get_collection_size() == 0:
            return "Không tìm thấy thông tin trong cơ sở dữ liệu."

        chunks = self.store.search(question, top_k=top_k)
        if not chunks:
            return "Không tìm thấy thông tin phù hợp trong cơ sở dữ liệu."

        context_blocks = []
        for idx, chunk in enumerate(chunks, 1):
            doc_id = chunk.get("metadata", {}).get("doc_id", chunk.get("id", "unknown"))
            content = chunk.get("content", "")
            context_blocks.append(f"[{idx}] (Nguồn: {doc_id}):\n{content}")

        context_str = "\n\n".join(context_blocks)

        prompt = (
            f"Dựa vào các thông tin ngữ cảnh được cung cấp dưới đây, hãy trả lời câu hỏi một cách chính xác.\n"
            f"Chỉ sử dụng ngữ cảnh dưới đây. Nếu ngữ cảnh không chứa câu trả lời, hãy trả lời 'Không tìm thấy thông tin trong dữ liệu'.\n"
            f"Trích dẫn số thứ tự nguồn [1], [2] khi đưa ra câu trả lời.\n\n"
            f"--- NGỮ CẢNH ---\n"
            f"{context_str}\n\n"
            f"--- CÂU HỎI ---\n"
            f"{question}\n"
        )

        return self.llm_fn(prompt)
