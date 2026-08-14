from pathlib import Path

from processing.cleaner import TextCleaner

from loaders.base_loader import BaseLoader


class TXTLoader(BaseLoader):

    ENCODINGS_TO_TRY = ("utf-8", "utf-8-sig", "latin-1")

    def __init__(self, file_path):

        self.file_path = Path(file_path)

    def load(self):

        if not self.file_path.exists():

            raise FileNotFoundError(
                f"{self.file_path} not found."
            )

        text = None

        for encoding in self.ENCODINGS_TO_TRY:

            try:

                with open(
                    self.file_path,
                    "r",
                    encoding=encoding
                ) as f:

                    text = f.read()

                break

            except UnicodeDecodeError:
                continue

        if text is None:

            # Last resort: don't crash the whole pipeline over one file,
            # replace undecodable bytes instead.
            with open(
                self.file_path,
                "r",
                encoding="utf-8",
                errors="replace"
            ) as f:

                text = f.read()

        return [

            {

                "page_number": 1,

                "text": TextCleaner.clean(text),

                "metadata": {

                    "source": self.file_path.name,

                    "page": 1,

                    "file_type": "txt"

                }

            }

        ]
