
# import os
# from nanovllm import LLM, SamplingParams


# def main():
#     path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

#     llm = LLM(
#         path,
#         enforce_eager=True,
#         tensor_parallel_size=1,
#         max_model_len=1024,
#         max_num_batched_tokens=1024,
#         max_num_seqs=4,
#         gpu_memory_utilization=0.7,
#     )

#     params_a = SamplingParams(temperature=0.6, max_tokens=64)
#     params_c = SamplingParams(temperature=0.6, max_tokens=16)

#     # 阶段 1：只有请求 A
#     llm.add_request("Write a short story about a robot.", params_a)

#     print("\n===== A starts =====")
#     for _ in range(3):
#         llm.step()

#     # 阶段 2：A 已经生成了一段时间，现在 C 才到达
#     llm.add_request(
#         "Explain KV cache in one sentence.",
#         params_c,
#     )

#     print("\n===== C arrives =====")
#     while not llm.is_finished():
#         outputs, _ = llm.step()
#         for seq_id, token_ids in outputs:
#             print(f"Finished seq={seq_id}, tokens={len(token_ids)}")


# if __name__ == "__main__":
#     main()


import os
from nanovllm import LLM, SamplingParams


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")

    llm = LLM(
        path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        max_num_seqs=4,
        gpu_memory_utilization=0.7,
    )

    params = SamplingParams(temperature=0.6, max_tokens=64)

    llm.add_request("Write a long story about a robot.", params)

    # 先让 A 开始执行
    llm.step()
    llm.step()

    # 每次只提交一个新请求，并执行一轮
    for prompt in [
        "Explain KV cache.",
        "Explain continuous batching.",
        "Explain GPU memory management.",
    ]:
        llm.add_request(prompt, params)
        llm.step()

    # 最后让所有请求完成
    while not llm.is_finished():
        llm.step()

    llm.exit()


if __name__ == "__main__":
    main()