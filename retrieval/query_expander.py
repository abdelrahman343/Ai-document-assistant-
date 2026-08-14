import re

from llm.llm import LLM


class QueryExpander:

    NUMBERING_PATTERN = re.compile(r"^\s*[\d]+[\.\)]\s*")
    BULLET_PATTERN = re.compile(r"^\s*[-*\u2022]\s*")

    def __init__(self):

        self.llm = LLM()

    def expand(self, question):

        prompt = f"""
You are a search query optimizer.

Rewrite the user's question into 3 different search queries.

Rules:

- Preserve the meaning.
- Use different wording.
- Keep each query short.
- Return ONLY the queries.
- One query per line.

Question:

{question}
"""

        try:
            response = self.llm.generate(prompt)
        except Exception:
            # If expansion fails, fall back to just the original question
            # rather than breaking retrieval entirely.
            return [question]

        queries = []

        for line in response.split("\n"):

            line = line.strip()

            if not line:
                continue

            # Handle "1.", "1)", "-", "*", "•" and markdown bold prefixes.
            line = self.NUMBERING_PATTERN.sub("", line)
            line = self.BULLET_PATTERN.sub("", line)
            line = line.strip("*").strip()

            if line:
                queries.append(line)

        queries.append(question)

        seen = set()
        deduped = []

        for q in queries:

            key = q.lower()

            if key not in seen:
                seen.add(key)
                deduped.append(q)

        return deduped
