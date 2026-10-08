# RAG Evaluation Report

## 1. Evaluation Configuration

- Dataset Cases：18
- Retrieval Top K：3
- Rerank Threshold：70.0
- Stability Repeat Count：2
- Generated At：2026-08-24T19:02:25.365995

## 2. Base Evaluation

| Metric | Result |
|---|---:|
| Answerability Accuracy | 100.00% |
| Hit Rate | 100.00% |
| Mean Recall | 100.00% |
| MRR | 1.0000 |
| Rerank Top1 Accuracy | 100.00% |

## 3. Retrieval Variant Comparison

| Variant | Hit@3 | Recall@3 | MRR | Top1 Accuracy |
|---|---:|---:|---:|---:|
| basic_vector | 100.00% | 100.00% | 0.9444 | 88.89% |
| multi_query | 100.00% | 100.00% | 0.9444 | 88.89% |
| hybrid | 100.00% | 100.00% | 1.0000 | 100.00% |
| hybrid_rerank | 100.00% | 100.00% | 1.0000 | 100.00% |

## 4. Rerank Stability

| Metric | Result |
|---|---:|
| Worst Positive Score | 85.0 |
| Worst Negative Score | 50.0 |
| Robust Margin | 35.0 |
| Configured Threshold | 70.0 |

## 5. Citation Evaluation

| Metric | Result |
|---|---:|
| Citation Presence Accuracy | 100.00% |
| Citation Reference Accuracy | 100.00% |
| Relevant Citation Hit Rate | 88.89% |
| All Citations Relevant Rate | 88.89% |

## 6. Conclusion

本项目构建了完整的 Advanced RAG Pipeline，包括 Query Rewrite、Multi-Query Retrieval、Vector Search、Keyword Search、Hybrid Search、RRF、Rerank、Answerability Threshold、Citation 和 Citation Pruning。

通过 Evaluation Dataset、Hard Cases、Retrieval Variant Comparison、Threshold Sweep、Rerank Stability 和 Citation Evaluation，对 RAG 系统进行了量化评测和稳定性验证。