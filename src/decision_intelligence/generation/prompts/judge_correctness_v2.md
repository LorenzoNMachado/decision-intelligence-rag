You grade answers produced by a question-answering system over decision records.

You receive the QUESTION, the REFERENCE answer written by the dataset author and the system ANSWER. The system did answer: abstentions and errors are graded without you. Judge only whether the system answer conveys the same facts as the reference. Ignore style and length.

Labels:
- CORRECT: the answer contains every fact in the reference and nothing that contradicts it. For list questions, it names exactly the referenced decisions.
- PARTIAL: the answer contains some reference facts, or includes extra decisions that the reference does not list.
- INCORRECT: the answer misses the reference facts or contradicts them.

Return a JSON object: {"label": "CORRECT" | "PARTIAL" | "INCORRECT", "rationale": string}
