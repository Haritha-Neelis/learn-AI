from src.retrieval import retrieve

TEST_CASES = [
    {
        "question": "What checks should I perform when an SD-WAN tunnel is down?",
        "expected_source": "sdwan_troubleshooting_sample.txt"
    },
    {
        "question": "What was the sample healthcare revenue in Q2?",
        "expected_source": "business_kpi_sample.txt"
    },
    {
        "question": "How should a RAG system handle prompt injection in retrieved documents?",
        "expected_source": "rag_security_sample.txt"
    }
]

def main():
    correct = 0
    for case in TEST_CASES:
        results = retrieve(case["question"])
        sources = [r["metadata"].get("source") for r in results]
        passed = case["expected_source"] in sources
        correct += int(passed)
        print(f"\nQ: {case['question']}")
        print(f"Expected source: {case['expected_source']}")
        print(f"Retrieved sources: {sources}")
        print("PASS" if passed else "CHECK")
    print(f"\nSource-retrieval smoke test: {correct}/{len(TEST_CASES)}")
    print("This is a tiny smoke test, not a complete quality benchmark.")

if __name__ == "__main__":
    main()
