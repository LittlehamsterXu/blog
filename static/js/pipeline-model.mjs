// 双缓冲流水线的时间线模型。
//
// 只回答一个问题：把「搬一个 K 分块」和「算一个 K 分块」各自看成整数个时间片之后，
// 串行调度和流水调度分别需要多少个时间片。
//
// 模型对应文章里那版 kernel 的等待方式：每次迭代末尾 __pipeline_wait_prior(0)，
// 也就是同时只允许一批拷贝在飞，重叠窗口只有一格。真实耗时不按整数刻度走，
// 所以这里的倍数只用来说明调度结构，不能当成 GPU 上的实测加速比。

export const RATIOS = { '1:1': [1, 1], '1:2': [1, 2], '2:1': [2, 1] };
export const TILE_CHOICES = [4, 6, 8];
export const DEFAULT_TILES = 6;
export const DEFAULT_RATIO = '1:2';

export function parseRatio(label) {
  const [load, compute] = RATIOS[label] || RATIOS[DEFAULT_RATIO];
  return { load, compute };
}

// 串行：搬完 tile k 才开算 tile k，两者永不重叠。
export function serialSpan(tiles, load, compute) {
  return tiles * (load + compute);
}

// 流水：tile 0 的 Load 必须先完成；之后第 k 个 tile 的 Load 与第 k-1 个 tile 的 Compute 同时进行。
export function pipelineSpan(tiles, load, compute) {
  return load + (tiles - 1) * Math.max(load, compute) + compute;
}

export function spanOf(mode, tiles, load, compute) {
  return mode === 'serial'
    ? serialSpan(tiles, load, compute)
    : pipelineSpan(tiles, load, compute);
}

// 每个事件的 buffer 就是 ping-pong 的两块 shared memory：读 k % 2、写 (k + 1) % 2。
// 同一时刻读写必然落在不同 buffer 上，所以两块就够了——这一点由测试守住。
export function schedule(mode, tiles, load, compute) {
  const events = [];
  const step = Math.max(load, compute);

  for (let tile = 0; tile < tiles; tile++) {
    const buffer = tile % 2;
    if (mode === 'serial') {
      const base = tile * (load + compute);
      events.push({ kind: 'load', tile, start: base, end: base + load, buffer });
      events.push({ kind: 'compute', tile, start: base + load, end: base + load + compute, buffer });
    } else {
      events.push({ kind: 'load', tile, start: tile * step, end: tile * step + load, buffer });
      events.push({ kind: 'compute', tile, start: load + tile * step, end: load + tile * step + compute, buffer });
    }
  }

  return {
    mode,
    tiles,
    load,
    compute,
    span: spanOf(mode, tiles, load, compute),
    serial: serialSpan(tiles, load, compute),
    events,
  };
}

// 某个时间片里谁在动、两块 buffer 分别处于什么状态。
export function stateAt(model, slot) {
  const active = model.events.filter(event => event.start <= slot && slot < event.end);
  const loads = active.filter(event => event.kind === 'load');
  const computes = active.filter(event => event.kind === 'compute');

  return {
    slot,
    loads,
    computes,
    overlap: loads.length > 0 && computes.length > 0,
    buffers: [0, 1].map(buffer => ({
      buffer,
      computing: computes.some(event => event.buffer === buffer),
      loading: loads.some(event => event.buffer === buffer),
    })),
  };
}

export function stats(model) {
  let overlap = 0;
  for (let slot = 0; slot < model.span; slot++) {
    if (stateAt(model, slot).overlap) overlap += 1;
  }
  return {
    span: model.span,
    serial: model.serial,
    overlap,
    speedup: model.serial / model.span,
  };
}
