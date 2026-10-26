
from math import ceil


class PagedKVCache:
    """简化的 KV Cache 分页分配模拟器。"""

    def __init__(self, block_size=4, total_blocks=32):
        self.block_size = block_size
        self.free_blocks = list(range(total_blocks))

        # request_id -> physical block IDs
        self.block_table = {}

        # request_id -> current sequence length
        self.lengths = {}

    def _allocate_block(self):
        if not self.free_blocks:
            raise MemoryError("No free KV blocks!")

        # 真实系统会使用更复杂的 block allocator。
        return self.free_blocks.pop(0)

    def start_request(self, request_id, prompt_len):
        if request_id in self.block_table:
            raise ValueError(f"{request_id} already exists")

        self.block_table[request_id] = []
        self.lengths[request_id] = 0

        # 为提示词分配足够的 Block。
        num_blocks = ceil(prompt_len / self.block_size)

        for _ in range(num_blocks):
            self.block_table[request_id].append(
                self._allocate_block()
            )

        self.lengths[request_id] = prompt_len

    def append_token(self, request_id):
        """模拟生成一个 Token，并为其分配所需空间。"""
        current_len = self.lengths[request_id]

        # 如果现有 Block 已满，才申请新 Block。
        if current_len % self.block_size == 0:
            self.block_table[request_id].append(
                self._allocate_block()
            )

        self.lengths[request_id] += 1

    def release_request(self, request_id):
        """请求结束时，释放它占用的所有 Block。"""
        blocks = self.block_table.pop(request_id)

        self.free_blocks.extend(blocks)
        self.lengths.pop(request_id)

    def allocated_blocks(self, request_id):
        return len(self.block_table[request_id])

    def allocated_slots(self, request_id):
        return (
            self.allocated_blocks(request_id)
            * self.block_size
        )

    def report(self, title, max_len):
        used = sum(self.lengths.values())

        allocated_blocks = sum(
            len(blocks)
            for blocks in self.block_table.values()
        )

        allocated_slots = (
            allocated_blocks * self.block_size
        )

        # 固定最大长度预留方案的容量。
        fixed_reserved = len(self.lengths) * max_len

        print(f"\n===== {title} =====")
        print(f"Active requests: {len(self.lengths)}")
        print(f"Used token slots: {used}")

        print("\n[Fixed max-length reservation]")
        print(f"Reserved slots: {fixed_reserved}")
        print(f"Unused reserved slots: {fixed_reserved - used}")

        print("\n[Paged allocation]")
        print(f"Allocated blocks: {allocated_blocks}")
        print(f"Allocated slots: {allocated_slots}")
        print(f"Internal fragmentation: {allocated_slots - used}")
        print(f"Free blocks: {len(self.free_blocks)}")

        print("\n[Block tables]")
        for request_id, blocks in self.block_table.items():
            capacity = len(blocks) * self.block_size
            length = self.lengths[request_id]

            print(
                f"{request_id}: length={length:2d}, "
                f"blocks={blocks}, "
                f"capacity={capacity:2d}, "
                f"unused={capacity - length}"
            )


def main():
    block_size = 4
    total_blocks = 32
    max_len = 32

    prompt_lengths = {
        "A": 8,
        "B": 19,
        "C": 5,
        "D": 30,
    }

    generation_targets = {
        "A": 3,
        "B": 5,
        "C": 2,
        "D": 0,
    }

    cache = PagedKVCache(
        block_size=block_size,
        total_blocks=total_blocks,
    )

    # Prefill：分配提示词对应的缓存。
    for request_id, prompt_len in prompt_lengths.items():
        cache.start_request(request_id, prompt_len)

    cache.report("After Prefill", max_len)

    # Decode：模拟多个请求交替生成 Token。
    generated = {name: 0 for name in prompt_lengths}

    max_steps = max(generation_targets.values())

    for _ in range(max_steps):
        for request_id in prompt_lengths:
            if generated[request_id] < generation_targets[request_id]:
                cache.append_token(request_id)
                generated[request_id] += 1

    cache.report("After Decode", max_len)


if __name__ == "__main__":
    main()
