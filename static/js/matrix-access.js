import { N, T, STEPS, TILE_STEPS, makeSteps, cacheTrace } from "./matrix-model.mjs";
(() => {
  const LAST = STEPS - 1;
  document.querySelectorAll(".matrix-demo").forEach(root => {
    let mode = root.dataset.initialMode || "ijk", steps = makeSteps(mode), step = 0, timer = null;
    const find = selector => root.querySelector(selector);
    const play = find("[data-play]"), seek = find("[data-seek]"), calculation = find(".matrix-calculation");
    const offsets = s => ({ A: s.i * N + s.k, B: s.k * N + s.j, C: s.i * N + s.j });
    // A、B 的取值由模板在服务端渲染好，这里读回 DOM，不再各写一份公式。
    const readMatrix = name => Array.from(
      find('[data-matrix="' + name + '"]').querySelectorAll("[data-cell]"),
      cell => Number(cell.dataset.value)
    );
    const values = { A: readMatrix("A"), B: readMatrix("B"), C: Array(N * N).fill(0) };
    function stop() {
      clearInterval(timer); timer = null; play.textContent = "播放";
      calculation.setAttribute("aria-live", "polite");
    }
    // 对比表只跟缓存容量有关，跟当前步无关，所以不放在每一步都会跑的 render 里。
    function renderCompare() {
      const capacity = Number(find("[data-capacity]").value);
      root.querySelectorAll("[data-compare]").forEach(row => {
        const result = cacheTrace(row.dataset.compare, LAST, capacity);
        row.querySelector("[data-hits]").textContent = result.hits;
        row.querySelector("[data-misses]").textContent = result.misses;
        row.querySelector("[data-total]").textContent = result.hits + result.misses;
        row.classList.toggle("cache-selected", row.dataset.compare === mode);
      });
    }
    function render() {
      const s = steps[step], current = offsets(s), previous = step ? offsets(steps[step - 1]) : null;
      const C = Array(N * N).fill(0), partial = Array(N * N).fill(0);
      let before = 0, after = 0;
      for (let n = 0; n <= step; n++) {
        const p = offsets(steps[n]);
        before = partial[p.C]; after = before + values.A[p.A] * values.B[p.B];
        partial[p.C] = after;
        if (mode !== "ijk" || steps[n].k === N - 1) C[p.C] = after;
      }
      values.C = C;
      const write = mode !== "ijk" || s.k === N - 1;
      const capacity = Number(find("[data-capacity]").value);
      const cache = cacheTrace(mode, step, capacity);
      const slots = find("[data-cache-slots]");
      slots.replaceChildren();
      for (let n = 0; n < capacity; n++) {
        const line = cache.lines[n], slot = document.createElement("span");
        slot.className = "cache-slot";
        if (line) {
          slot.dataset.matrix = line.matrix;
          slot.textContent = line.matrix + "[" + line.base + ", " + (line.base + 1) + "]" + (line.dirty ? " *" : "");
          slot.title = "连续的两个元素；" + (line.dirty ? "已修改，淘汰时需写回" : "只读缓存块");
          if (cache.events.some(event => event.key === line.key)) slot.classList.add("cache-touched");
        } else slot.textContent = "空";
        slots.append(slot);
      }
      find("[data-cache-count]").textContent = "到当前步：命中 " + cache.hits + " 次 · 未命中 " + cache.misses + " 次";
      const events = find("[data-cache-events]");
      events.replaceChildren();
      for (const event of cache.events) {
        const li = document.createElement("li");
        li.className = event.hit ? "cache-hit" : "cache-miss";
        li.textContent = event.matrix + "[" + event.offset + "]：" + (event.hit ? "命中，移到最近使用端" : "未命中，载入 " + event.matrix + "[" + event.base + ", " + (event.base + 1) + "]") + (event.evicted ? "；淘汰 " + event.evicted.matrix + "[" + event.evicted.base + ", " + (event.evicted.base + 1) + "]" + (event.evicted.dirty ? "（脏块需写回）" : "") : "");
        events.append(li);
      }
      const aAccessed = cache.events.some(event => event.matrix === "A");
      find("[data-register]").textContent = mode === "ijk" ? "局部 sum = " + after + (write ? "，本步写回 C。" : "，继续留在局部变量中。") : "局部 a = " + values.A[current.A] + (aAccessed ? "，本步从 A 读取。" : "，沿用前一步的值，不再访问 A 缓存。");
      const inRegion = (name, p) => {
        const r = Math.floor(p / N), c = p % N;
        if (mode === "tiled") {
          const origin = name === "A" ? [s.ii, s.kk] : name === "B" ? [s.kk, s.jj] : [s.ii, s.jj];
          return r >= origin[0] && r < origin[0] + T && c >= origin[1] && c < origin[1] + T;
        }
        return name === "A" ? r === s.i : name === "B" ? (mode === "ijk" ? c === s.j : r === s.k) : (mode === "ijk" ? p === current.C : r === s.i);
      };
      for (const name of ["A", "B", "C"]) {
        const panel = find('[data-matrix="' + name + '"]'), memory = find('[data-memory="' + name + '"]');
        for (const parent of [panel, memory]) parent.querySelectorAll("[data-cell]").forEach(cell => {
          const p = Number(cell.dataset.cell);
          if (parent === panel) cell.textContent = values[name][p];
          cell.classList.toggle("is-region", inRegion(name, p));
          cell.classList.toggle("is-cached", parent === memory && cache.lines.some(line => line.matrix === name && p >= line.base && p <= line.base + 1));
          cell.classList.toggle("is-current", p === current[name] && (name !== "C" || write));
          cell.classList.toggle("is-target", name === "C" && p === current.C && !write);
          cell.classList.toggle("is-previous", previous !== null && p === previous[name] && p !== current[name]);
          cell.title = name + "(" + Math.floor(p / N) + ", " + p % N + ") = " + values[name][p] + "；偏移 " + p;
        });
        const p = current[name];
        panel.querySelector("[data-address]").textContent = name + "(" + Math.floor(p / N) + ", " + p % N + ") · 偏移 " + p + (name === "C" ? (write ? " · 写回" : " · 等待写回") : name === "A" && !aAccessed ? " · 复用局部 a" : " · 读取");
        let jump = "首次访问";
        if (previous) {
          const delta = p - previous[name];
          jump = delta === 0 ? "同一位置 · 可复用" : delta === 1 ? "+1 · 相邻元素" : (delta > 0 ? "+" : "") + delta + " · 跨位置";
        }
        memory.querySelector("[data-jump]").textContent = name === "C" && !write ? "sum 暂存累加结果，C 的这个元素尚未写回" : name === "A" && !aAccessed ? "沿用局部变量 a，本步没有新的 A 内存访问" : jump;
      }
      find("[data-formula]").textContent = (mode === "ijk" ? "sum" : "C(" + s.i + ", " + s.j + ")") + "：" + before + " + " + values.A[current.A] + " × " + values.B[current.B] + " = " + after + (mode === "ijk" && write ? " → 写回 C" : "");
      find("[data-explanation]").textContent = mode === "ijk" ? "固定一个输出位置：A 沿行前进，B 沿列跳转。4 次乘加完成后，才把 sum 写回 C。" : mode === "ikj" ? "固定 A(i, k)，连续更新 C 的一行；B 和 C 的内层访问都相邻，A 的值可重复使用。" : "固定三个 2 × 2 块，块内按 i → k → j 计算。一个 A 元素用于两列，一个 B 元素用于两行。";
      const tile = find("[data-tile]");
      tile.hidden = mode !== "tiled";
      tile.textContent = "块起点 (ii, kk, jj) = (" + s.ii + ", " + s.kk + ", " + s.jj + ")。每组块做 " + TILE_STEPS + " 次乘加；沿 kk 累加两组，才能完成一块 C。";
      root.querySelectorAll("[data-mode]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.mode === mode)));
      find("[data-count]").textContent = (step + 1) + " / " + STEPS;
      seek.value = step;
      find("[data-prev]").disabled = step === 0;
      find("[data-next]").disabled = step === LAST;
      find("[data-block]").hidden = mode !== "tiled";
      find("[data-block]").disabled = step >= STEPS - TILE_STEPS;
    }
    function start() {
      if (step === LAST) step = 0;
      calculation.setAttribute("aria-live", "off");
      play.textContent = "暂停"; render();
      timer = setInterval(() => { step++; render(); if (step === LAST) stop(); }, Number(find("[data-speed]").value));
    }
    play.addEventListener("click", () => timer === null ? start() : stop());
    root.querySelectorAll("[data-mode]").forEach(button => button.addEventListener("click", () => {
      stop(); mode = button.dataset.mode; steps = makeSteps(mode); step = 0; renderCompare(); render();
    }));
    for (const [selector, advance] of [["[data-prev]", () => step - 1], ["[data-next]", () => step + 1], ["[data-reset]", () => 0], ["[data-block]", () => (Math.floor(step / TILE_STEPS) + 1) * TILE_STEPS]]) {
      find(selector).addEventListener("click", () => { stop(); step = Math.max(0, Math.min(LAST, advance())); render(); });
    }
    seek.addEventListener("input", () => { stop(); step = Number(seek.value); render(); });
    find("[data-capacity]").addEventListener("change", () => { stop(); renderCompare(); render(); });
    find("[data-speed]").addEventListener("change", () => { if (timer !== null) { stop(); start(); } });
    document.addEventListener("visibilitychange", () => { if (document.hidden) stop(); });
    new IntersectionObserver(entries => { if (!entries[0].isIntersecting) stop(); }).observe(root);
    find(".matrix-controls").hidden = false;
    renderCompare();
    render();
  });
})();
