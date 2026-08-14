import re

from llm.llm import LLM


class DocumentSummarizer:

    ARABIC_PATTERN = re.compile(
        r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
    )

    LANGUAGE_NAMES = {
        "en": "English",
        "ar": "Arabic"
    }

    FALLBACK_MESSAGE = (
        "I couldn't generate this right now. Please try again."
    )

    def __init__(self):

        self.llm = LLM()

    def _detect_lang(self, text):

        return "ar" if self.ARABIC_PATTERN.search(text) else "en"

    def _prepare_text(
        self,
        chunks,
        max_chunks_per_document=3
    ):

        # Group by source document so summaries cover every uploaded
        # file instead of just the first N chunks overall.
        grouped = {}

        for chunk in chunks:

            source = chunk.get(
                "metadata", {}
            ).get("source", "unknown")

            grouped.setdefault(source, []).append(chunk)

        selected = []

        for source, doc_chunks in grouped.items():

            selected.extend(
                doc_chunks[:max_chunks_per_document]
            )

        return "\n\n".join(
            chunk["text"] for chunk in selected
        )

    # -----------------------------------------
    # Document Summary
    # -----------------------------------------

    def summarize(
        self,
        chunks,
        language=None
    ):
        """language: explicit "en"/"ar" override (used when this is
        triggered from a chat question, so the question's language
        wins). If not given, auto-detects from the document's own
        text - covers the button-triggered case where there's no
        question to read a language from."""

        text = self._prepare_text(
            chunks
        )

        lang = language or self._detect_lang(text)
        lang_name = self.LANGUAGE_NAMES.get(lang, "English")

        prompt = f"""
You are an expert document analyst.

Read the document below and produce a professional summary.

Respond in {lang_name}, regardless of markup or headers below.

Return the answer using this format:

## Overview
(2-3 sentences)

## Key Topics
- Topic 1
- Topic 2
- Topic 3

## Important Details
- ...
- ...
- ...

Document:

{text}
"""

        try:
            return self.llm.generate(
                prompt
            )
        except Exception:
            return self.FALLBACK_MESSAGE

    # -----------------------------------------
    # Suggested Questions
    # -----------------------------------------

    def generate_questions(
        self,
        chunks,
        language=None
    ):

        text = self._prepare_text(
            chunks
        )

        lang = language or self._detect_lang(text)
        lang_name = self.LANGUAGE_NAMES.get(lang, "English")

        prompt = f"""
You are helping a user explore a document.

Generate five useful questions someone would naturally ask after uploading this document.

Respond in {lang_name}.

Rules:

- Only output the questions.
- Don't answer them.
- Make them specific to the document.
- Number them.

Document:

{text}
"""

        try:
            return self.llm.generate(
                prompt
            )
        except Exception:
            return self.FALLBACK_MESSAGE