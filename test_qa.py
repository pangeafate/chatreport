# test_qa.py
from qa import answer_question

if __name__ == "__main__":
    question = "What were the factors negatively impacting revenue in 2023 mentioned in the reports?"
    answer = answer_question(question)
    print("Question:", question)
    print("Answer:", answer)
