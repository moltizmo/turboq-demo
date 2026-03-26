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


def load_hotpotqa(n_queries=500):
    """Load HotpotQA distractor validation set for multi-hop reasoning evaluation.

    Returns:
        corpus: list of unique passage texts
        query_data: list of dicts with 'query', 'candidate_ids', 'relevant_ids'
    """
    print(f"Loading HotpotQA distractor validation set ({n_queries} queries)...")
    ds = load_dataset('hotpot_qa', 'distractor', split='validation', streaming=True)
    samples = list(ds.take(n_queries))

    corpus = []
    passage_to_idx = {}
    query_data = []

    for sample in samples:
        context = sample['context']
        supporting = sample['supporting_facts']

        candidate_ids = []
        relevant_ids = []
        supporting_titles = set(supporting['title'])

        for title, sentences in zip(context['title'], context['sentences']):
            passage_text = ' '.join(sentences)
            if passage_text not in passage_to_idx:
                passage_to_idx[passage_text] = len(corpus)
                corpus.append(passage_text)
            pid = passage_to_idx[passage_text]
            candidate_ids.append(pid)
            if title in supporting_titles:
                relevant_ids.append(pid)

        if relevant_ids:
            query_data.append({
                'query': sample['question'],
                'candidate_ids': candidate_ids,
                'relevant_ids': relevant_ids,
            })

    print(f"Corpus size: {len(corpus)} unique passages")
    print(f"Valid queries (with relevant passages): {len(query_data)}")
    return corpus, query_data


def load_nq_with_wikipedia(n_queries=500):
    """Load Natural Questions Open with Wikipedia passages as corpus.

    Uses answer-string matching to determine relevance since NQ Open
    only provides questions + answers without passage context.

    Returns:
        corpus: list of Wikipedia passage chunks
        query_data: list of dicts with 'query', 'candidate_ids', 'relevant_ids'
    """
    print(f"Loading NQ Open validation set ({n_queries} queries)...")
    nq = load_dataset('nq_open', split='validation', streaming=True)
    nq_samples = list(nq.take(n_queries * 2))

    print("Loading Wikipedia corpus (500 articles, ~5000 chunks)...")
    wiki = load_dataset('wikimedia/wikipedia', '20231101.en', split='train', streaming=True)

    corpus = []
    for article in wiki.take(500):
        text = article['text']
        words = text.split()
        for i in range(0, min(len(words), 400), 200):
            chunk = ' '.join(words[i:i+200])
            if len(chunk) > 50:
                corpus.append(chunk)
        if len(corpus) >= 5000:
            break
    corpus = corpus[:5000]
    print(f"Wikipedia corpus: {len(corpus)} chunks")

    query_data = []
    for sample in nq_samples[:n_queries]:
        answers = sample['answer']
        relevant_ids = []
        for pid, passage in enumerate(corpus):
            passage_lower = passage.lower()
            if any(ans.lower() in passage_lower for ans in answers):
                relevant_ids.append(pid)

        if relevant_ids:
            query_data.append({
                'query': sample['question'],
                'candidate_ids': list(range(len(corpus))),
                'relevant_ids': relevant_ids[:5],
            })

        if len(query_data) >= 300:
            break

    print(f"Valid queries (with evidence in corpus): {len(query_data)}")
    return corpus, query_data


if __name__ == "__main__":
    corpus, query_data = load_msmarco()
    print(f"\nExample query: {query_data[0]['query']}")
    print(f"Relevant passage IDs: {query_data[0]['relevant_ids']}")
    print(f"Relevant passage: {corpus[query_data[0]['relevant_ids'][0]][:200]}...")
