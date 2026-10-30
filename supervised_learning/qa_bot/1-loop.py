#!/usr/bin/env python3
"""Run a simple interactive question-and-answer loop."""

while True:
    try:
        question = input("Q: ")
    except EOFError:
        break

    if question.strip().lower() in ("exit", "quit", "goodbye", "bye"):
        print("A: Goodbye")
        break

    print("A:")
