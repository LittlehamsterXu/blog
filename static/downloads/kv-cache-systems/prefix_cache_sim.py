
from hashlib import sha256


class PrefixCache:
    def __init__(self, block_size=4):
        self.block_size = block_size
        self.cache = {}
        self.next_block_id = 0

    def _block_hash(self, previous_hash, tokens):
        # 同时考虑前面的上下文和当前 Block。
        content = repr(
            (previous_hash, tuple(tokens))
        ).encode("utf-8")

        return sha256(content).hexdigest()

    def process_prompt(self, request_id, tokens):
        block_size = self.block_size
        full_blocks = len(tokens) // block_size
        partial_tokens = len(tokens) % block_size

        previous_hash = "ROOT"
        block_ids = []
        cache_hits = 0
        cache_misses = 0

        for i in range(full_blocks):
            start = i * block_size
            block = tokens[start:start + block_size]

            key = self._block_hash(previous_hash, block)

            if key in self.cache:
                # 复用已经计算好的完整 Block。
                block_id = self.cache[key]
                cache_hits += 1
            else:
                # 模拟计算并缓存这个 Block。
                block_id = self.next_block_id
                self.next_block_id += 1

                self.cache[key] = block_id
                cache_misses += 1

            block_ids.append(block_id)
            previous_hash = key

        reused_tokens = cache_hits * block_size

        # 简化估算：未命中完整 Block + 未缓存的尾部。
        computed_tokens = (
            cache_misses * block_size + partial_tokens
        )

        print(f"\n===== Request {request_id} =====")
        print(f"Prompt length: {len(tokens)}")
        print(f"Full blocks: {full_blocks}")
        print(f"Partial tail tokens: {partial_tokens}")
        print(f"Cache hits: {cache_hits}/{full_blocks}")
        print(f"Reused tokens: {reused_tokens}")
        print(f"Tokens to process: {computed_tokens}")
        print(f"Physical block IDs: {block_ids}")


def main():
    cache = PrefixCache(block_size=4)

    # 这里每个字符串代表一个抽象 Token，
    # 并非真实 tokenizer 的分词结果。
    common_prefix = [
        "请", "解释", "KV", "Cache",
        "工作", "原理", "以及", "例子",
    ]

    prompt_a = common_prefix + [
        "先", "讲", "原理"
    ]

    prompt_b = common_prefix + [
        "再", "比较", "GQA", "MLA"
    ]

    cache.process_prompt("A", prompt_a)
    cache.process_prompt("B", prompt_b)


if __name__ == "__main__":
    main()
