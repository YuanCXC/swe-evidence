"""为 Dense 与 Rerank 提供一致的文件 Token 分块。"""

from __future__ import annotations

from transformers import AutoTokenizer


BGE_TOKENIZER_NAME = "BAAI/bge-m3"
BGE_TOKENIZER_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


class TokenChunker:
    """使用冻结的 BGE tokenizer 将文件切为有限数量的重叠块。"""

    def __init__(
        self,
        *,
        tokenizer_name: str,
        tokenizer_revision: str,
        chunk_tokens: int,
        overlap_tokens: int,
        max_chunks_per_file: int,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            revision=tokenizer_revision,
        )
        self.tokenizer_name = tokenizer_name
        self.tokenizer_revision = tokenizer_revision
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.max_chunks_per_file = max_chunks_per_file

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(
            text,
            add_special_tokens=False,
            verbose=False,
        )

    def _fit_chunk(self, text: str) -> str:
        special_tokens = self.tokenizer.num_special_tokens_to_add(pair=False)
        token_ids = self._encode(text)[: self.chunk_tokens - special_tokens]
        chunk = self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        while len(self._encode(chunk)) + special_tokens > self.chunk_tokens:
            token_ids.pop()
            chunk = self.tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        return chunk

    def split(self, content: str, *, path: str) -> list[str]:
        """为每个块保留文件路径，并按文件顺序截取前若干块。"""

        prefix = f"文件路径：{path}\n\n"
        prefix_ids = self._encode(prefix)
        special_tokens = self.tokenizer.num_special_tokens_to_add(pair=False)
        content_budget = self.chunk_tokens - len(prefix_ids) - special_tokens
        if content_budget <= self.overlap_tokens:
            raise ValueError(f"文件路径过长，无法按当前 Token 预算分块：{path}")
        step = content_budget - self.overlap_tokens
        required_tokens = content_budget + step * (self.max_chunks_per_file - 1)
        character_limit = min(len(content), required_tokens * 8)
        content_ids = self._encode(content[:character_limit])
        while len(content_ids) < required_tokens and character_limit < len(content):
            character_limit = min(len(content), character_limit * 2)
            content_ids = self._encode(content[:character_limit])
        chunks = []
        for start in range(0, max(len(content_ids), 1), step):
            body = self.tokenizer.decode(
                content_ids[start : start + content_budget],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            chunks.append(self._fit_chunk(prefix + body))
            if len(chunks) >= self.max_chunks_per_file:
                break
        return chunks
