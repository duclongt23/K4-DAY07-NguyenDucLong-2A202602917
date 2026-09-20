#!/usr/bin/env python3
"""
bench.py — Comprehensive Retrieval Benchmark & Evaluation Script for K4-L3B

Evaluates:
- 3 Chunking Strategies: FixedSize, Recursive, Heading (Heading-based)
- 5 Benchmark Queries with strict content-level signature checking
- A/B Test for Metadata Filter on Query #2
- Generates ket_qua_benchmark.txt
"""

import csv
import io
import os
import re
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.chunking import FixedSizeChunker, HeadingChunker, RecursiveChunker
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    GEMINI_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    GeminiEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

# 5 Benchmark Queries + Gold Document IDs + Gold Content Signatures
BENCHMARK_QUERIES = [
    {
        "id": 1,
        "query": "Tôi cần thực hiện những bước nào để gửi yêu cầu Trả hàng/Hoàn tiền trực tiếp trên ứng dụng Shopee?",
        "gold_doc_id": "shopee-buyer-return-request-guide",
        "gold_signature": "Mở ứng dụng Shopee, vào mục Tôi",
        "filter": None,
    },
    {
        "id": 2,
        "query": "Thời hạn tối đa để xử lý và phản hồi khiếu nại Trả hàng/Hoàn tiền là bao nhiêu ngày?",
        "gold_doc_id": "shopee-seller-return-processing-policy",
        "gold_signature": "02 ngày làm việc",
        "filter": {"audience": "seller"},
    },
    {
        "id": 3,
        "query": "Nếu Người bán Shopee Mall vi phạm quy định bán hàng giả, hàng nhái thì bị phạt bao nhiêu tiền và xử lý như thế nào?",
        "gold_doc_id": "shopee-mall-seller-terms-of-service",
        "gold_signature": "9.818.180",
        "filter": None,
    },
    {
        "id": 4,
        "query": "Trường hợp nào Người mua được miễn 100% cước phí vận chuyển hoàn trả hàng?",
        "gold_doc_id": "shopee-return-shipping-fee-policy",
        "gold_signature": "miễn phí 100%",
        "filter": None,
    },
    {
        "id": 5,
        "query": "Những trường hợp/lý do nào không được áp dụng chính sách trả hàng với lý do 'Đổi ý'?",
        "gold_doc_id": "shopee-general-return-policy",
        "gold_signature": "danh sách hạn chế trả hàng",
        "filter": None,
    },
]


def load_embedder():
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            return LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            return _mock_embed
    elif provider == "openai":
        try:
            return OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            return _mock_embed
    elif provider == "gemini":
        try:
            return GeminiEmbedder(model_name=os.getenv("GEMINI_EMBEDDING_MODEL", GEMINI_EMBEDDING_MODEL))
        except Exception:
            return _mock_embed
    return _mock_embed


def parse_markdown_file(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    parts = text.split("---")
    if len(parts) >= 3:
        fm_text = parts[1]
        content = "---".join(parts[2:]).strip()
        fm = dict(re.findall(r"^(\w+):\s*(.+)$", fm_text, re.M))
        cleaned_fm = {k: v.strip('"\'') for k, v in fm.items()}
        return cleaned_fm, content
    return {}, text.strip()


def build_store(chunker, files: list[Path], embedder) -> tuple[EmbeddingStore, int]:
    all_chunks: list[Document] = []
    for p in files:
        metadata, content = parse_markdown_file(p)
        doc_id = p.stem
        metadata["doc_id"] = doc_id
        raw_chunks = chunker.chunk(content)
        for idx, chunk_str in enumerate(raw_chunks):
            chunk_doc = Document(
                id=f"{doc_id}#{idx}",
                content=chunk_str,
                metadata=dict(metadata),
            )
            all_chunks.append(chunk_doc)

    store = EmbeddingStore(collection_name="bench_store", embedding_fn=embedder)
    store.add_documents(all_chunks)
    return store, len(all_chunks)


def evaluate_retrieval(store: EmbeddingStore, query_info: dict, metadata_override=...):
    query = query_info["query"]
    gold_doc = query_info["gold_doc_id"]
    signature = query_info["gold_signature"]
    m_filter = query_info["filter"] if metadata_override is Ellipsis else metadata_override

    if m_filter:
        results = store.search_with_filter(query, top_k=3, metadata_filter=m_filter)
    else:
        results = store.search(query, top_k=3)

    # Level 1: Naive (is gold_doc in top-3?)
    naive_found = any(r["metadata"].get("doc_id") == gold_doc for r in results)

    # Level 2: Strict (is gold_doc present AND context contains signature?)
    strict_rank = 0
    strict_found = False
    for rank, r in enumerate(results, start=1):
        if r["metadata"].get("doc_id") == gold_doc and signature.lower() in r["content"].lower():
            strict_rank = rank
            strict_found = True
            break

    # Score calculation (docs/SCORING.md)
    if strict_found and strict_rank == 1:
        score = 2
    elif strict_found and strict_rank in (2, 3):
        score = 1
    else:
        score = 0

    return {
        "results": results,
        "naive_found": naive_found,
        "strict_found": strict_found,
        "strict_rank": strict_rank,
        "score": score,
    }


def run_benchmark():
    buf = io.StringIO()

    def p(text=""):
        print(text)
        buf.write(text + "\n")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    data_dir = Path("data/chinh-sach-doi-tra")
    md_files = sorted(data_dir.glob("*.md"))
    embedder = load_embedder()
    embedder_name = getattr(embedder, "_backend_name", embedder.__class__.__name__)

    p("=" * 80)
    p(f"   K4-L3B BENCHMARK RETRIEVAL EVALUATION REPORT")
    p(f"   Embedding Backend: {embedder_name}")
    p(f"   Total Documents: {len(md_files)}")
    p("=" * 80)

    strategies = {
        "FixedSize (300/50)": FixedSizeChunker(chunk_size=300, overlap=50),
        "Recursive (300)": RecursiveChunker(chunk_size=300),
        "Heading (400)": HeadingChunker(max_chunk_size=400),
    }

    summary_scores = {}

    for strat_name, chunker in strategies.items():
        p("\n" + "-" * 80)
        p(f"📌 CHIẾN LƯỢC: {strat_name}")
        p("-" * 80)

        store, total_chunks = build_store(chunker, md_files, embedder)
        p(f"Số lượng Chunks sinh ra: {total_chunks}")

        total_score = 0
        for q in BENCHMARK_QUERIES:
            eval_res = evaluate_retrieval(store, q)
            total_score += eval_res["score"]

            p(f"\n❓ [Câu {q['id']}]: {q['query']}")
            if q['filter']:
                p(f"   🔍 Filter: {q['filter']}")
            p(f"   🎯 Gold Doc: {q['gold_doc_id']} | Signature: '{q['gold_signature']}'")
            p(f"   📊 Score: {eval_res['score']}/2 (Strict Rank: {eval_res['strict_rank'] or 'N/A'}, Naive Found: {eval_res['naive_found']})")

            for rank, r in enumerate(eval_res["results"], 1):
                doc_id = r["metadata"].get("doc_id", "unknown")
                chunk_id = r.get("id", "N/A")
                score_val = r["score"]
                preview = r["content"][:100].replace("\n", " ")
                p(f"      Top {rank} ({score_val:.3f}) [{chunk_id}] (File: {source_doc if 'source_doc' in locals() else doc_id})")
                p(f"         Preview: \"{preview}...\"")

        summary_scores[strat_name] = total_score
        p(f"\n➡️ TỔNG ĐIỂM CHIẾN LƯỢC [{strat_name}]: {total_score} / 10 điểm")

    # A/B TEST FOR METADATA FILTER ON QUERY 2
    p("\n" + "=" * 80)
    p(" 🧪 THỬ NGHIỆM A/B: METADATA FILTERING CÓ GIÚP ÍCH KHÔNG? (CÂU HỎI #2)")
    p("=" * 80)
    q2 = BENCHMARK_QUERIES[1]
    for strat_name, chunker in strategies.items():
        store, _ = build_store(chunker, md_files, embedder)
        res_with = evaluate_retrieval(store, q2, metadata_override={"audience": "seller"})
        res_without = evaluate_retrieval(store, q2, metadata_override=None)

        top1_with = res_with["results"][0]["metadata"].get("doc_id") if res_with["results"] else "N/A"
        top1_without = res_without["results"][0]["metadata"].get("doc_id") if res_without["results"] else "N/A"

        p(f"\n🔸 Chiến lược [{strat_name}]:")
        p(f"   - CÓ Filter (audience=seller) : Score={res_with['score']}/2 | Top-1 Doc: {top1_with}")
        p(f"   - KHÔNG Filter                : Score={res_without['score']}/2 | Top-1 Doc: {top1_without}")

    # FINAL SUMMARY
    p("\n" + "=" * 80)
    p(" 🏆 BẢNG TỔNG HỢP ĐIỂM CHẤT LƯỢNG TRUY XUẤT CÁC CHIẾN LƯỢC")
    p("=" * 80)
    for strat_name, score in summary_scores.items():
        p(f"  • {strat_name:25}: {score} / 10 điểm")

    # Write output to ket_qua_benchmark.txt
    output_path = Path("ket_qua_benchmark.txt")
    output_path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"\n✅ Đã xuất báo cáo chi tiết ra file: {output_path.absolute()}")


if __name__ == "__main__":
    run_benchmark()
