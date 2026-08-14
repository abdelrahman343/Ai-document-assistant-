import os
import logging

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class LLM:

    # Same idea as EmbeddingModel: avoid re-creating a client on every
    # LLM() instantiation (which happens once per document's QAService,
    # DocumentSummarizer, and QueryExpander).
    _client = None

    def __init__(self, model="llama-3.3-70b-versatile"):

        if LLM._client is None:
            LLM._client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        self.client = LLM._client
        self.model = model

    def generate(self, prompt, temperature=0.2):

        try:

            response = self.client.chat.completions.create(

                model=self.model,

                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                temperature=temperature,
            )

            return response.choices[0].message.content

        except Exception as exc:

            logger.error("LLM generation failed: %s", exc)
            raise