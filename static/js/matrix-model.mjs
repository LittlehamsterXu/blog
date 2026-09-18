// 演示矩阵的规模，以及教学缓存模型的参数。
//
// 这里只保留"结构"。A、B 的元素取值由矩阵面板在服务端渲染，
// matrix-access.js 再从 DOM 里读回来，所以取值只有一个来源，
// 不存在"模板和 JS 各写一份公式、改一边忘另一边"的问题。
export const N = 4;                 // 方阵边长
export const T = 2;                 // 分块模式的块边长
export const L = 2;                 // 教学缓存的块大小：相邻 L 个 float 算一块
export const STEPS = N * N * N;     // 每种实现的总步数（N³ 次乘加）
export const TILE_STEPS = T * T * T; // 分块模式下"一组块"（一个 ii, kk, jj 组合）的步数

export function makeSteps(mode) {
  const steps = [];
  const add = (i, k, j, ii = 0, kk = 0, jj = 0) => steps.push({ i, k, j, ii, kk, jj });
  if (mode === "ijk") {
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) for (let k = 0; k < N; k++) add(i, k, j);
  } else if (mode === "ikj") {
    for (let i = 0; i < N; i++) for (let k = 0; k < N; k++) for (let j = 0; j < N; j++) add(i, k, j);
  } else {
    for (let ii = 0; ii < N; ii += T) for (let kk = 0; kk < N; kk += T) for (let jj = 0; jj < N; jj += T)
      for (let i = ii; i < ii + T; i++) for (let k = kk; k < kk + T; k++) for (let j = jj; j < jj + T; j++) add(i, k, j, ii, kk, jj);
  }
  return steps;
}

export function cacheTrace(mode, end, capacity) {
  const lines = [], steps = makeSteps(mode);
  let hits = 0, misses = 0, lastEvents = [];
  for (let n = 0; n <= end; n++) {
    const s = steps[n], accesses = [];
    // ikj 和分块版的局部变量 a 在一次内层 j 循环里复用：只有这段的第一列才真的读 A。
    if (mode === "ijk" || s.j === (mode === "tiled" ? s.jj : 0)) accesses.push(["A", s.i * N + s.k, false]);
    accesses.push(["B", s.k * N + s.j, false]);
    // C 的读改写合并成一次块访问；ijk 要等 sum 累加完才写回。
    if (mode !== "ijk" || s.k === N - 1) accesses.push(["C", s.i * N + s.j, true]);
    const events = [];
    for (const [matrix, offset, write] of accesses) {
      const base = Math.floor(offset / L) * L, key = matrix + ":" + base;
      const index = lines.findIndex(line => line.key === key);
      let evicted = null;
      const hit = index >= 0;
      let line;
      if (hit) { hits++; line = lines.splice(index, 1)[0]; }
      else {
        misses++;
        if (lines.length === capacity) evicted = lines.shift();
        line = { key, matrix, base, dirty: false };
      }
      line.dirty ||= write;
      lines.push(line);
      events.push({ matrix, offset, base, key, hit, evicted });
    }
    lastEvents = events;
  }
  return { lines, hits, misses, events: lastEvents };
}
