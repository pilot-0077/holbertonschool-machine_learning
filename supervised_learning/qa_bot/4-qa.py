#!/usr/bin/env python3
"""Interactive question answering over a corpus of reference documents."""


def question_answer(corpus_path):
    """Answer questions using the most relevant document in the corpus."""
    semantic_search = __import__(
        '3-semantic_search'
    ).semantic_search
    qa_function = __import__('0-qa').question_answer

    exit_commands = ("exit", "quit", "goodbye", "bye")
    fallback = "Sorry, I do not understand your question."

    while True:
        try:
            question = input("Q: ")
        except EOFError:
            break

        if question.strip().lower() in exit_commands:
            print("A: Goodbye")
            break

        reference = semantic_search(corpus_path, question)

        if not reference:
            answer = fallback
        else:
            answer = qa_function(question, reference)
            if not answer or not answer.strip():
                answer = fallback

        print("A: {}".format(answer))
