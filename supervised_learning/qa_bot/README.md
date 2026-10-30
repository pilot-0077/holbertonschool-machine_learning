# Question Answering Bot

This project focuses on building a question answering system that can process
natural-language questions and return relevant answers from available text or
knowledge sources.

## Learning Objectives

By the end of this project, I should be able to explain:

- What question answering systems are
- The difference between extractive and generative question answering
- How text is tokenized and represented for Transformer models
- How pretrained language models can be used for question answering
- How context and user questions are passed to a model
- How answer spans or generated responses are produced
- How to build an interactive question answering workflow

## Core Concepts

A QA bot combines natural-language processing with a model capable of finding
or generating an answer from context. The pipeline generally includes input
processing, model inference, answer extraction or generation, and presentation
of the final response to the user.

## Repository

- GitHub repository: `holbertonschool-machine_learning`
- Directory: `supervised_learning/qa_bot`

## Task 0: Question Answering

The `0-qa.py` module implements extractive question answering with the
TensorFlow Hub `bert-uncased-tf2-qa` model and the Hugging Face
`BertTokenizer` associated with
`bert-large-uncased-whole-word-masking-finetuned-squad`.

The `question_answer(question, reference)` function tokenizes the question
and reference, builds BERT inputs, and uses the model's start and end logits
to select an answer span from the reference. Long references are processed
in overlapping windows to respect BERT's 512-token input limit. It returns
the extracted text, or `None` when the input is empty or the best span is
below the configured confidence threshold.

The first call downloads the TensorFlow Hub model and tokenizer assets if
they are not already cached locally.

## Task 1: Create the Loop

The `1-loop.py` script repeatedly prompts the user with `Q: ` and prints
`A: ` after each ordinary input. The commands `exit`, `quit`, `goodbye`,
and `bye` terminate the loop regardless of letter case and print
`A: Goodbye`. Leading or trailing whitespace around a quit command is
ignored. The script also exits cleanly when standard input reaches EOF.

## Task 2: Answer Questions

The `2-qa.py` module defines `answer_loop(reference)`, an interactive
question-answering loop built on the `question_answer` function from
`0-qa.py`. Each question is answered using the supplied reference text.
When no answer is returned, the loop prints
`Sorry, I do not understand your question.`. The commands `exit`,
`quit`, `goodbye`, and `bye` are accepted case-insensitively and end
the session with `A: Goodbye`.
