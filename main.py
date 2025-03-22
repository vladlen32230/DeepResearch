from src.pipelines import process_question
from src.chat import answer_question
from src.chroma_init import init_chroma

def deepresearch(question: str):
    print("Deepresearch...")
    research = process_question(question)
    print(research)

def lightresearch(question: str):
    init_chroma()
    
    use_internet = input("Do you want to use internet? (y/n): ").lower() == 'y'
    use_rag = input("Do you want to use RAG? (y/n): ").lower() == 'y'

    if use_rag:
        use_advanced_rag = input("Do you want to use advanced RAG? (y/n): ").lower() == 'y'
    else:
        use_advanced_rag = False
    
    print("Lightresearch...")

    answer = answer_question(
        question, 
        use_internet=use_internet, 
        use_rag=use_rag, 
        use_advanced_rag=use_advanced_rag
    )

    print(answer)

def main():
    question = input("Enter your question: ")
    use_deepresearch = input("Do you want to use deepresearch? (y/n): ").lower() == 'y'
    if use_deepresearch:
        deepresearch(question)
    else:
        lightresearch(question)

if __name__ == "__main__":
    main()