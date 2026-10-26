import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import platform
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import argparse

parser = argparse.ArgumentParser(description='Run the three original standard-library simulators and save their outputs.')
parser.add_argument('--output-dir', type=Path, default=Path('my-results'))
args = parser.parse_args()
SRC = Path(__file__).resolve().parent
OUT = args.output_dir
OUT.mkdir(parents=True, exist_ok=True)


def load(name):
    spec = importlib.util.spec_from_file_location(name, SRC / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


results = {'environment': {'python': platform.python_version(), 'platform': platform.system(), 'device': 'CPU', 'dependencies': ['Python standard library']}}
logs = []
sources = []
for name in ['paged_kv_sim', 'prefix_cache_sim', 'continuous_batching_sim']:
    source = SRC / (name + '.py')
    raw = source.read_bytes()
    text = raw.decode('utf-8')
    (OUT / source.name).write_bytes(raw)
    meta = {'file': source.name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    sources.append(meta)
    module = load(name)
    snapshots = []
    if name == 'paged_kv_sim':
        original_report = module.PagedKVCache.report

        def record_report(cache, title, max_len):
            used = sum(cache.lengths.values())
            count = sum(len(blocks) for blocks in cache.block_table.values())
            snapshots.append({
                'title': title,
                'block_size': cache.block_size,
                'total_blocks': len(cache.free_blocks) + count,
                'max_len_for_fixed_reservation': max_len,
                'used_token_slots': used,
                'allocated_blocks': count,
                'allocated_slots': count * cache.block_size,
                'internal_fragmentation': count * cache.block_size - used,
                'fixed_reserved_slots': len(cache.lengths) * max_len,
                'fixed_unused_slots': len(cache.lengths) * max_len - used,
                'free_blocks': list(cache.free_blocks),
                'lengths': dict(cache.lengths),
                'block_tables': copy.deepcopy(cache.block_table),
            })
            return original_report(cache, title, max_len)

        module.PagedKVCache.report = record_report
        results['paged_allocation'] = {'snapshots': snapshots, 'cache_payload': 'allocation and integer IDs only', 'release_in_main': False, 'reference_counting': False}
    elif name == 'prefix_cache_sim':
        original_process = module.PrefixCache.process_prompt

        def record_prompt(cache, request_id, tokens):
            old = set(cache.cache)
            original_process(cache, request_id, tokens)
            previous = 'ROOT'
            blocks = []
            for offset in range(0, (len(tokens) // cache.block_size) * cache.block_size, cache.block_size):
                block = tokens[offset:offset + cache.block_size]
                key = cache._block_hash(previous, block)
                blocks.append({'tokens': block, 'parent_hash': previous, 'hash': key, 'physical_id': cache.cache[key], 'hit': key in old})
                previous = key
            hits = sum(b['hit'] for b in blocks)
            tail = len(tokens) % cache.block_size
            snapshots.append({'request_id': request_id, 'tokens': list(tokens), 'prompt_length': len(tokens), 'block_size': cache.block_size,
                              'full_blocks': len(blocks), 'partial_tail_tokens': tail, 'cache_hits': hits, 'cache_misses': len(blocks) - hits,
                              'reused_tokens': hits * cache.block_size, 'tokens_to_process': (len(blocks) - hits) * cache.block_size + tail,
                              'physical_ids': [b['physical_id'] for b in blocks], 'blocks': blocks,
                              'partial_tail': tokens[len(blocks) * cache.block_size:], 'cached_block_count': len(cache.cache)})

        module.PrefixCache.process_prompt = record_prompt
        results['prefix_caching'] = {'requests': snapshots, 'hash': 'SHA-256 of repr((previous_hash, tuple(block_tokens))) encoded as UTF-8', 'cache_payload': 'hash-to-block-ID mapping only', 'reference_counting': False, 'eviction': False}
    else:
        original_simulate = module.simulate

        def record_simulate(specs, capacity, policy):
            states, timeline, metrics = original_simulate(specs, capacity, policy)
            snapshots.append({'policy': policy, 'request_specs': copy.deepcopy(specs), 'max_batch_size': capacity,
                              'states': states, 'timeline': [{'time': t, 'slots': slots} for t, slots in timeline], 'metrics': metrics})
            return states, timeline, metrics

        module.simulate = record_simulate
        results['batch_scheduling'] = {'runs': snapshots, 'time_unit': 'one abstract decode iteration', 'tokens_per_active_request_per_iteration': 1,
                                      'prefill_modeled': False, 'kv_capacity_modeled': False, 'gpu_execution_modeled': False}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        module.main()
    logs.append(f'### {source.name}\n' + output.getvalue())

(OUT / 'simulation_results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
(OUT / 'simulation_stdout.txt').write_text('\n\n'.join(logs), encoding='utf-8')
(OUT / 'simulation_sources.json').write_text(json.dumps(sources, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({
    'paged': [{k: s[k] for k in ['title', 'used_token_slots', 'allocated_blocks', 'allocated_slots', 'internal_fragmentation', 'fixed_reserved_slots', 'fixed_unused_slots']} for s in results['paged_allocation']['snapshots']],
    'prefix': [{k: s[k] for k in ['request_id', 'prompt_length', 'cache_hits', 'cache_misses', 'reused_tokens', 'tokens_to_process', 'physical_ids']} for s in results['prefix_caching']['requests']],
    'scheduling': [{k: r[k] for k in ['policy', 'metrics']} for r in results['batch_scheduling']['runs']],
    'sources': sources,
}, ensure_ascii=False, indent=2))
