#!/usr/bin/env python3
"""Run an interactive question-answering loop over a reference text."""


def answer_loop(reference):
    """Answer user questions using the supplied reference text."""
    question_answer = __import__('0-qa').question_answer

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

        answer = question_answer(question, reference)

        if not answer or not answer.strip():
            answer = fallback

        print("A: {}".format(answer))
