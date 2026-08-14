import re


class TextCleaner:

    @staticmethod
    def clean(text: str) -> str:

        # إزالة المسافات المتكررة
        text = re.sub(r"\s+", " ", text)

        # إزالة المسافات في البداية والنهاية
        text = text.strip()

        return text