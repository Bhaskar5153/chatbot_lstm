from typing import List, Dict, Optional, Any

class TextChunker:
    def __init__(self, max_chars: int = 1200, overlap: int = 150):
        self.max_chars = max_chars
        self.overlap = overlap

    def chunk(self, text: str):
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.max_chars, len(text))
            _chunk = text[start:end]
            chunks.append(_chunk.strip())
            if end == len(text):
                break
            start = end - self.overlap

        return [c for c in chunks if c]
    

    def make_qas(self, chunks: List[str]):
        qas = []
        for c in chunks:
            first = c.split(". ")
            ans = first[0].strip() if first else c[:300].strip()
            question = "summarize the following content:"
            qas.append([question, ans])
        return qas
    

tc = TextChunker()
text = "The society has a right  to expect  of  him  such  ideal  behaviour."
"It  must  not  be forgotten that the legal profession has always been held  in high esteem and its members have played an enviable role  in public life."
# print(tc.chunk(text=text))
chunks = tc.chunk(text=text)
qas_pair = tc.make_qas(chunks=chunks)
print(qas_pair)

