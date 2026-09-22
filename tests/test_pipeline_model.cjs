// 双缓冲时间线模型的独立校验。运行：node tests/test_pipeline_model.cjs
// 这里只验调度代数和 ping-pong 不变式，不涉及 DOM、也不涉及真实耗时。
const assert = require('node:assert/strict');

(async () => {
  const { RATIOS, schedule, stateAt, stats } = await import('../static/js/pipeline-model.mjs');

  const combos = [];
  for (const [label, [load, compute]] of Object.entries(RATIOS)) {
    for (const tiles of [4, 6, 8]) combos.push({ label, load, compute, tiles });
  }

  for (const { label, load, compute, tiles } of combos) {
    const serial = schedule('serial', tiles, load, compute);
    const pipe = schedule('pipeline', tiles, load, compute);
    const where = `${label} / ${tiles} tiles`;

    assert.equal(serial.span, tiles * (load + compute), where);
    assert.equal(pipe.span, load + (tiles - 1) * Math.max(load, compute) + compute, where);
    assert.ok(pipe.span < serial.span, `${where}: 流水应当比串行短`);

    // 串行调度里没有一格是重叠的。
    assert.equal(stats(serial).overlap, 0, where);

    let overlap = 0;
    for (let slot = 0; slot < pipe.span; slot++) {
      const state = stateAt(pipe, slot);
      assert.ok(state.loads.length + state.computes.length > 0, `${where}: 第 ${slot} 格不该空转`);
      if (state.overlap) overlap += 1;
      // ping-pong 不变式：同一格不会对同一块 buffer 边读边写，所以两块就够用。
      for (const { buffer, computing, loading } of state.buffers) {
        assert.ok(!(computing && loading), `${where}: Buffer ${buffer} 在同一格被读写`);
      }
    }
    assert.equal(stats(pipe).overlap, overlap, where);

    // 每个 tile 都是先搬完再算；同时每个 tile 的 Load 都确实与上一格 Compute 重叠。
    const loads = pipe.events.filter(event => event.kind === 'load');
    const computes = pipe.events.filter(event => event.kind === 'compute');
    loads.forEach((event, index) => {
      assert.ok(event.end <= computes[index].start, `${where}: tile ${index} 的 Compute 早于自己的 Load 完成`);
      if (index > 0) {
        assert.ok(event.start < computes[index - 1].end, `${where}: tile ${index} 的 Load 没有和上一格重叠`);
      }
    });
  }

  // 几个手算过的具体数字：6 个 tile，Load : Compute = 1 : 2 时串行 18 格、流水 13 格。
  assert.equal(schedule('serial', 6, 1, 2).span, 18);
  assert.equal(schedule('pipeline', 6, 1, 2).span, 13);
  assert.equal(schedule('pipeline', 6, 1, 1).span, 7);
  assert.equal(schedule('pipeline', 6, 2, 1).span, 13);

  console.log('Pipeline timeline: 串行与流水的格数、重叠计数和 ping-pong buffer 不变式全部通过。');
})();
