import argparse
from src.pipeline import answer_question

def main():
    parser = argparse.ArgumentParser(description="Ask the sample RAG knowledge base")
    parser.add_argument("question", help="Question to ask")
    args = parser.parse_args()
    result = answer_question(args.question)
    print("\nQUESTION:\n", result["question"])
    print("\nANSWER:\n", result["answer"])
    print("\nRETRIEVED SOURCES:")
    for source in result["sources"]:
        print(f"- {source['source']} (chunk {source['chunk_index']})")

if __name__ == "__main__":
    main()
