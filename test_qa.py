# test_qa.py
from qa import answer_question

if __name__ == "__main__":
    question = "what they focused on in Artificial Intelligence Software and Hardware in Q2?"
    answer = answer_question(question)
    print("Question:", question)
    print("Answer:", answer)
