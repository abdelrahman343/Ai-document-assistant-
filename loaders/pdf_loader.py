import fitz

from pathlib import Path

from processing.cleaner import TextCleaner
from loaders.base_loader import BaseLoader


class PDFLoader(BaseLoader):

    def __init__(self, file_path):

        self.file_path = Path(file_path)


    def load(self):

        if not self.file_path.exists():

            raise FileNotFoundError(
                f"{self.file_path} not found."
            )


        document = fitz.open(
            self.file_path
        )

        pages = []


        for page_number, page in enumerate(
            document,
            start=1
        ):

            text = TextCleaner.clean(
                page.get_text()
            )


            pages.append({

                "page_number": page_number,

                "text": text,

                "metadata": {

                    "source": self.file_path.name,

                    "page": page_number,

                    "file_type": "pdf"

                }

            })


        return pages