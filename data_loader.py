"""MS MARCO v2.1 dataset loader with real human-annotated relevance judgments."""

from datasets import load_dataset


def load_msmarco(n_queries: int = 500):
    """Load MS MARCO v2.1 validation set and build corpus + query_data.

    Returns:
        corpus: list of unique passage texts
        query_data: list of dicts with 'query', 'candidate_ids', 'relevant_ids'
    """
    print(f"Loading MS MARCO v2.1 validation set ({n_queries} queries)...")
    ds = load_dataset("ms_marco", "v2.1", split="validation", streaming=True)
    samples = list(ds.take(n_queries))

    corpus = []
    passage_to_idx = {}
    query_data = []

    for sample in samples:
        passages = sample["passages"]["passage_text"]
        labels = sample["passages"]["is_selected"]
        query_passage_ids = []
        relevant_ids = []

        for text, label in zip(passages, labels):
            if text not in passage_to_idx:
                passage_to_idx[text] = len(corpus)
                corpus.append(text)
            pid = passage_to_idx[text]
            query_passage_ids.append(pid)
            if label == 1:
                relevant_ids.append(pid)

        if relevant_ids:
            query_data.append({
                "query": sample["query"],
                "candidate_ids": query_passage_ids,
                "relevant_ids": relevant_ids,
            })

    print(f"Corpus size: {len(corpus)} unique passages")
    print(f"Valid queries (with relevant passages): {len(query_data)}")
    return corpus, query_data


if __name__ == "__main__":
    corpus, query_data = load_msmarco()
    print(f"\nExample query: {query_data[0]['query']}")
    print(f"Relevant passage IDs: {query_data[0]['relevant_ids']}")
    print(f"Relevant passage: {corpus[query_data[0]['relevant_ids'][0]][:200]}...")
