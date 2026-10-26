
from copy import deepcopy


REQUESTS = [
    {"id": "A", "arrival": 0, "tokens": 8},
    {"id": "B", "arrival": 1, "tokens": 5},
    {"id": "C", "arrival": 2, "tokens": 3},
    {"id": "D", "arrival": 4, "tokens": 6},
    {"id": "E", "arrival": 5, "tokens": 2},
]

MAX_BATCH_SIZE = 3


def simulate(request_specs, max_batch_size, policy):
    """
    简化的 Decode 调度模拟器。

    每轮：
      1. 接收当前时刻到达的请求。
      2. 根据策略决定哪些请求进入活动批次。
      3. 每个活动请求生成一个 Token。
      4. 更新剩余 Token 数与完成时间。
    """
    states = {}

    for req in request_specs:
        states[req["id"]] = {
            **req,
            "remaining": req["tokens"],
            "start": None,
            "finish": None,
            "generated": 0,
        }

    arrivals = {}
    for req in request_specs:
        arrivals.setdefault(req["arrival"], []).append(req["id"])

    pending = []
    active = []
    timeline = []
    time = 0

    if policy not in ("static", "continuous"):
        raise ValueError(f"Unknown policy: {policy}")

    while any(s["finish"] is None for s in states.values()):
        # 收集当前时刻到达的请求。
        pending.extend(arrivals.get(time, []))

        if policy == "static":
            # 只有当前批次完全结束后，才能建立新批次。
            if not active and pending:
                active = pending[:max_batch_size]
                pending = pending[max_batch_size:]

                for name in active:
                    states[name]["start"] = time

        else:
            # Continuous Batching：每轮开始时填充空闲槽位。
            while len(active) < max_batch_size and pending:
                name = pending.pop(0)
                active.append(name)

                if states[name]["start"] is None:
                    states[name]["start"] = time

        # 记录这一轮的槽位状态。
        # ✓ 表示请求已经结束，但 Static Batching 仍保留该槽位。
        slots = []

        for name in active:
            if states[name]["remaining"] == 0:
                slots.append(name + "✓")
            else:
                slots.append(name)

        slots.extend(["·"] * (max_batch_size - len(slots)))
        timeline.append((time, slots))

        # 每个尚未结束的请求生成一个 Token。
        for name in active:
            state = states[name]

            if state["remaining"] > 0:
                state["remaining"] -= 1
                state["generated"] += 1

                if state["remaining"] == 0:
                    state["finish"] = time + 1

        # 两种策略的核心区别。
        if policy == "static":
            # 保留已结束的槽位，直到整个批次完成。
            if all(
                states[name]["remaining"] == 0
                for name in active
            ):
                active = []

        else:
            # 下一轮可以让新请求填补刚刚释放的槽位。
            active = [
                name for name in active
                if states[name]["remaining"] > 0
            ]

        time += 1

    # 统计结果。
    first_arrival = min(s["arrival"] for s in states.values())
    last_finish = max(s["finish"] for s in states.values())
    makespan = last_finish - first_arrival

    total_tokens = sum(s["tokens"] for s in states.values())

    avg_wait = sum(
        s["start"] - s["arrival"] for s in states.values()
    ) / len(states)

    avg_turnaround = sum(
        s["finish"] - s["arrival"] for s in states.values()
    ) / len(states)

    # 这是模拟的有用 Token 槽位利用率，不是真实 GPU 利用率。
    slot_utilization = (
        total_tokens / (max_batch_size * makespan)
    )

    metrics = {
        "makespan": makespan,
        "total_tokens": total_tokens,
        "throughput": total_tokens / makespan,
        "avg_wait": avg_wait,
        "avg_turnaround": avg_turnaround,
        "slot_utilization": slot_utilization,
    }

    return states, timeline, metrics


def print_result(title, states, timeline, metrics, batch_size):
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)

    print("\n[Decode Timeline]")
    print("Time | " + " | ".join(
        f"Slot {i + 1}" for i in range(batch_size)
    ))
    print("-" * 40)

    for time, slots in timeline:
        print(f"{time:4d} | " + " | ".join(
            f"{slot:>6}" for slot in slots
        ))

    print("\n[Per-request Metrics]")
    print("ID | Arrival | Start | Finish | Wait | Turnaround")

    for name, state in states.items():
        wait = state["start"] - state["arrival"]
        turnaround = state["finish"] - state["arrival"]

        print(
            f"{name:>2} | "
            f"{state['arrival']:7d} | "
            f"{state['start']:5d} | "
            f"{state['finish']:6d} | "
            f"{wait:4d} | "
            f"{turnaround:10d}"
        )

    print("\n[Aggregate Metrics]")
    print(f"Makespan:           {metrics['makespan']}")
    print(f"Total generated:    {metrics['total_tokens']} tokens")
    print(f"Throughput:         {metrics['throughput']:.3f} tokens/unit")
    print(f"Average wait:       {metrics['avg_wait']:.3f} time units")
    print(f"Average turnaround: {metrics['avg_turnaround']:.3f} time units")
    print(f"Slot utilization:   {metrics['slot_utilization']:.2%}")


def main():
    static_states, static_timeline, static_metrics = simulate(
        deepcopy(REQUESTS), MAX_BATCH_SIZE, "static"
    )

    continuous_states, continuous_timeline, continuous_metrics = simulate(
        deepcopy(REQUESTS), MAX_BATCH_SIZE, "continuous"
    )

    print_result(
        "STATIC BATCHING",
        static_states,
        static_timeline,
        static_metrics,
        MAX_BATCH_SIZE,
    )

    print_result(
        "CONTINUOUS BATCHING",
        continuous_states,
        continuous_timeline,
        continuous_metrics,
        MAX_BATCH_SIZE,
    )


if __name__ == "__main__":
    main()
