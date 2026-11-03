"""Run the learning script while recording scheduler decisions; no source edits."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, default=Path("test_continuous_batching.py"))
    parser.add_argument("--output-dir", type=Path, default=Path("recorded-schedule"))
    args = parser.parse_args()
    from nanovllm.engine.scheduler import Scheduler

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    events = []
    original = Scheduler.schedule

    def schedule(scheduler):
        # Replace local debugging prints with one consistent line per decision.
        with contextlib.redirect_stdout(io.StringIO()):
            seqs, is_prefill = original(scheduler)
        event = dict(
            step=len(events) + 1,
            mode="PREFILL" if is_prefill else "DECODE",
            waiting=len(scheduler.waiting),
            running=len(scheduler.running),
            scheduled=[seq.seq_id for seq in seqs],
        )
        events.append(event)
        print(
            f"[SCHEDULER] mode={event['mode']} waiting={event['waiting']} "
            f"running={event['running']} scheduled={event['scheduled']}",
            flush=True,
        )
        return seqs, is_prefill

    class Tee:
        def __init__(self, console, log):
            self.console, self.log = console, log

        def write(self, text):
            self.console.write(text)
            self.log.write(text)
            return len(text)

        def flush(self):
            self.console.flush()
            self.log.flush()

    Scheduler.schedule = schedule
    completed = False
    try:
        with (output_dir / "continuous-batching.log").open("w", encoding="utf-8") as log:
            with contextlib.redirect_stdout(Tee(sys.stdout, log)):
                runpy.run_path(str(args.script.resolve()), run_name="__main__")
        completed = True
    finally:
        Scheduler.schedule = original
        (output_dir / "schedule.json").write_text(
            json.dumps(dict(completed=completed, events=events), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
