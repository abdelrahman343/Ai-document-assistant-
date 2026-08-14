from typing import List


class TextChunker:

    def __init__(
        self,
        chunk_size=500,
        overlap=100
    ):

        if overlap >= chunk_size:
            raise ValueError(
                "overlap must be smaller than chunk_size, "
                "otherwise chunking will never make progress."
            )

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(
        self,
        pages: List[dict]
    ):

        records = []
        chunk_id = 0

        for page in pages:

            text = page["text"]
            start = 0

            while start < len(text):

                end = start + self.chunk_size

                # Avoid cutting a word in half: extend to the next
                # whitespace boundary if we're not already at the end.
                if end < len(text):
                    next_space = text.find(" ", end)
                    if next_space != -1:
                        end = next_space

                chunk_text = text[start:end].strip()

                if chunk_text:

                    metadata = page["metadata"].copy()

                    metadata.update({
                        "chunk_id": chunk_id
                    })

                    records.append({
                        "text": chunk_text,
                        "metadata": metadata
                    })

                    chunk_id += 1

                start += (
                    self.chunk_size
                    -
                    self.overlap
                )

        total_chunks = len(records)

        for record in records:
            record["metadata"]["total_chunks"] = total_chunks

        return records
