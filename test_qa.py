# test_qa.py
from qa import answer_question

if __name__ == "__main__":
    question = "what were the financial highlights in Q4?"
    answer = answer_question(question)
    print("Question:", question)
    print("Answer:", answer)
