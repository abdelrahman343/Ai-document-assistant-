import logging
import re

from embeddings.embedding_model import EmbeddingModel
from llm.llm import LLM

from retrieval.faiss_retriever import FaissRetriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.query_expander import QueryExpander

from services.document_summary import DocumentSummarizer

logger = logging.getLogger(__name__)


class QAService:

    # Covers Arabic + Arabic Supplement + Arabic Extended-A +
    # Arabic Presentation Forms. Good enough to tell "this question
    # is in Arabic" without a full language-detection dependency.
    ARABIC_PATTERN = re.compile(
        r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
    )

    NO_ANSWER_MESSAGES = {
        "en": "I couldn't find the answer in the uploaded documents.",
        "ar": "لم أتمكن من العثور على الإجابة في المستندات المرفوعة."
    }

    GREETING_MESSAGES = {
        "en": (
            "Hi! Ask me anything about the document you've uploaded "
            "and I'll do my best to answer from it."
        ),
        "ar": (
            "مرحبًا! اسألني أي شيء عن المستند الذي رفعته "
            "وسأبذل قصارى جهدي للإجابة منه."
        )
    }

    def __init__(
        self,
        vector_store,
        bm25_store=None
    ):

        self.embedding_model = EmbeddingModel()

        self.vector_store = vector_store

        self.bm25_store = bm25_store

        self.retriever = HybridRetriever(
            FaissRetriever(vector_store),
            bm25_store,
            min_score=0.02
        )

        self.query_expander = QueryExpander()

        self.llm = LLM()

        self.summarizer = DocumentSummarizer()

        # question -> result, so asking the same thing twice doesn't
        # redo retrieval + an LLM call.
        self._answer_cache = {}

    def _detect_lang(self, text):

        return "ar" if self.ARABIC_PATTERN.search(text) else "en"

    # ----------------------------------------------------

    def _is_summary_request(
        self,
        question
    ):

        question = question.lower()

        keywords = [

            "summary",
            "summarize",
            "overview",
            "what is this",
            "what's this",
            "whats this",
            "what is this document about",
            "what does this document include",
            "what is this file about",
            "describe this document",
            "describe this file",
            "tell me about this document",
            "tell me about this file"

        ]

        return any(
            keyword in question
            for keyword in keywords
        )

    # ----------------------------------------------------

    def _is_greeting(
        self,
        question
    ):

        stripped = question.strip().lower().rstrip("!?. ")

        greetings = [

            "hi",
            "hello",
            "hey",
            "how are you",
            "what's up",
            "whats up",
            "good morning",
            "good afternoon",
            "good evening",
            "thanks",
            "thank you",
            "who are you",
            "what can you do",

            # Arabic
            "مرحبا",
            "مرحباً",
            "اهلا",
            "أهلا",
            "السلام عليكم",
            "كيف حالك",
            "شكرا",
            "شكراً",
            "من انت",
            "من أنت",
            "هاي",
            "هلا"

        ]

        return stripped in greetings

    # ----------------------------------------------------

    def answer(
        self,
        question
    ):

        cache_key = question.strip().lower()

        if cache_key in self._answer_cache:
            return self._answer_cache[cache_key]

        result = self._answer_uncached(question)

        self._answer_cache[cache_key] = result

        return result

    def _answer_uncached(
        self,
        question
    ):

        # ===========================================
        # Greeting / Chitchat
        # ===========================================

        if self._is_greeting(question):

            lang = self._detect_lang(question)

            return {

                "answer": self.GREETING_MESSAGES[lang],

                "sources": []

            }

        # ===========================================
        # Summary
        # ===========================================

        if self._is_summary_request(question):

            # DocumentSummarizer now samples chunks per-source-document,
            # so this covers every uploaded file, not just the first 8
            # chunks overall.
            summary = self.summarizer.summarize(
                self.vector_store.documents,
                language=self._detect_lang(question)
            )

            return {

                "answer": summary,

                "sources": []

            }

        # ===========================================
        # Query Expansion
        # ===========================================

        # Expansion helps ambiguous or underspecified questions find
        # more phrasings of the same intent. A short, already-specific
        # question doesn't need 3 extra LLM-generated rewrites - that
        # just triples retrieval cost and adds an LLM call for little
        # benefit.
        if len(question.split()) <= 4:
            expanded_queries = [question]
        else:
            expanded_queries = self.query_expander.expand(
                question
            )

        logger.debug("Expanded queries: %s", expanded_queries)

        # ===========================================
        # Multi Query Retrieval
        # ===========================================

        merged = {}

        # Batch-encode all expanded queries in one call instead of
        # one model call per query.
        query_embeddings = self.embedding_model.encode(
            expanded_queries
        )

        for query, question_embedding in zip(
            expanded_queries,
            query_embeddings
        ):

            results = self.retriever.retrieve(
                query,
                question_embedding,
                k=5
            )

            for result in results:

                doc = result["document"]

                key = (
                    doc["metadata"]["source"],
                    doc["metadata"]["page"],
                    doc["text"]
                )

                if key not in merged:

                    merged[key] = result

                else:

                    merged[key]["score"] += result["score"]

        results = sorted(

            merged.values(),

            key=lambda x: x["score"],

            reverse=True

        )[:5]

        # ===========================================
        # No Results
        # ===========================================

        if not results:

            return {

                "answer": self.NO_ANSWER_MESSAGES[self._detect_lang(question)],

                "sources": []

            }

        # ===========================================
        # Context
        # ===========================================

        context = ""

        sources = []

        for i, result in enumerate(results, start=1):

            metadata = result["document"]["metadata"]

            context += f"""
Source {i}

File:
{metadata['source']}

Page:
{metadata['page']}

Content:
{result['document']['text']}

------------------------------------
"""

            sources.append({

                "file": metadata["source"],

                "page": metadata["page"]

            })

        # ===========================================
        # Prompt
        # ===========================================

        no_answer_phrase = self.NO_ANSWER_MESSAGES[self._detect_lang(question)]

        prompt = f"""
You are an AI Document Intelligence Assistant.

Rules:

- Answer ONLY using the provided context.

- Never hallucinate.

- If the answer isn't present in the context, say exactly:

"{no_answer_phrase}"

- Combine information from multiple chunks when appropriate.

- Respond in the same language the question was asked in, even if the context above is written in a different language. Translate the relevant facts rather than answering in the context's language.

Context:

{context}

Question:

{question}

Answer:
"""

        try:
            answer = self.llm.generate(
                prompt
            )
        except Exception:
            answer = (
                "I ran into an error generating an answer. Please try again."
            )

        return {

            "answer": answer,

            "sources": sources,

            "context": context

        }