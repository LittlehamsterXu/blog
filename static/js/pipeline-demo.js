import { parseRatio, schedule, stateAt, stats, DEFAULT_RATIO, DEFAULT_TILES } from "./pipeline-model.mjs";

(() => {
  document.querySelectorAll("figure.pipeline-demo").forEach(root => {
    const find = selector => root.querySelector(selector);
    const chart = find("[data-chart]");
    let mode = root.dataset.initialMode === "serial" ? "serial" : "pipeline";
    let tiles = DEFAULT_TILES;
    let ratio = parseRatio(DEFAULT_RATIO);
    let slot = 0;
    let timer = null;
    let model = schedule(mode, tiles, ratio.load, ratio.compute);

    const name = event => (event.kind === "load" ? "L" : "C") + event.tile;
    const describe = event =>
      (event.kind === "load" ? "拷贝 tile " : "计算 tile ") + event.tile +
      " → Buffer " + event.buffer + "（时间片 " + (event.start + 1) + "–" + event.end + "）";

    function rebuild() {
      model = schedule(mode, tiles, ratio.load, ratio.compute);
      slot = 0;
      const seek = find("[data-seek]");
      seek.max = String(model.span - 1);
      seek.value = "0";
    }

    function stop() {
      clearInterval(timer);
      timer = null;
      find("[data-play]").textContent = "播放";
      find("[data-readout]").setAttribute("aria-live", "polite");
    }

    function render() {
      const state = stateAt(model, slot);
      const blocks = [];
      for (const [row, kind] of [[1, "load"], [2, "compute"]]) {
        for (const event of model.events.filter(item => item.kind === kind)) {
          const node = document.createElement("span");
          node.className = "pipeline-block is-" + kind +
            (event.start <= slot && slot < event.end ? " is-active" : "");
          node.style.gridColumn = (event.start + 1) + " / span " + (event.end - event.start);
          node.style.gridRow = String(row);
          node.textContent = name(event);
          node.title = describe(event);
          blocks.push(node);
        }
      }
      const cursor = document.createElement("span");
      cursor.className = "pipeline-cursor";
      cursor.style.gridColumn = String(slot + 1);
      cursor.style.gridRow = "1 / span 2";
      blocks.push(cursor);
      chart.style.setProperty("--slots", String(model.span));
      chart.replaceChildren(...blocks);

      const bufferNodes = state.buffers.map(({ buffer, computing, loading }) => {
        const node = document.createElement("div");
        node.className = "pipeline-buffer" + (computing ? " is-computing" : "") + (loading ? " is-loading" : "");
        const title = document.createElement("strong");
        title.textContent = "Buffer " + buffer;
        const status = document.createElement("span");
        status.textContent = computing && loading ? "读写同时"
          : computing ? "计算中 · 正在被读"
          : loading ? "接收中 · 正在被写"
          : "空闲";
        node.append(title, status);
        return node;
      });
      find("[data-buffers]").replaceChildren(...bufferNodes);
      renderReadout(state);
      find("[data-prev]").disabled = slot === 0;
      find("[data-next]").disabled = slot === model.span - 1;
      find("[data-seek]").value = String(slot);
      find("[data-progress]").textContent = (slot + 1) + " / " + model.span;
    }

    function renderReadout(state) {
      const summary = stats(model);
      const active = [...state.loads, ...state.computes].filter(Boolean);
      const what = state.overlap ? "搬和算同时在跑"
        : state.loads.length ? "只有拷贝在跑，计算单元在等"
        : state.computes.length ? "只有计算在跑，拷贝已经就位"
        : "这一格没有工作";
      find("[data-readout]").textContent =
        "时间片 " + (slot + 1) + " / " + model.span + "：" + what +
        (active.length ? "（" + active.map(name).join(" + ") + "）" : "") +
        "。串行要 " + summary.serial + " 格，流水要 " + summary.span + " 格，重叠 " + summary.overlap +
        " 格，模型加速比 " + summary.speedup.toFixed(2) + "×。";
    }

    function start() {
      if (slot === model.span - 1) slot = 0;
      find("[data-readout]").setAttribute("aria-live", "off");
      find("[data-play]").textContent = "暂停";
      render();
      timer = setInterval(() => {
        slot += 1;
        render();
        if (slot === model.span - 1) stop();
      }, Number(find("[data-speed]").value));
    }

    find("[data-play]").addEventListener("click", () => (timer === null ? start() : stop()));
    find("[data-speed]").addEventListener("change", () => {
      if (timer !== null) {
        stop();
        start();
      }
    });
    root.querySelectorAll("[data-mode]").forEach(button => {
      button.addEventListener("click", () => {
        stop();
        mode = button.dataset.mode;
        rebuild();
        render();
      });
    });
    for (const [selector, next] of [
      ["[data-prev]", () => slot - 1],
      ["[data-next]", () => slot + 1],
      ["[data-reset]", () => 0],
    ]) {
      find(selector).addEventListener("click", () => {
        stop();
        slot = Math.max(0, Math.min(model.span - 1, next()));
        render();
      });
    }
    find("[data-seek]").addEventListener("input", () => {
      stop();
      slot = Number(find("[data-seek]").value);
      render();
    });
    find("[data-tiles]").addEventListener("change", () => {
      stop();
      tiles = Number(find("[data-tiles]").value);
      rebuild();
      render();
    });
    find("[data-ratio]").addEventListener("change", () => {
      stop();
      ratio = parseRatio(find("[data-ratio]").value);
      rebuild();
      render();
    });
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) stop();
    });
    new IntersectionObserver(entries => {
      if (!entries[0].isIntersecting) stop();
    }).observe(root);

    find("[data-tiles]").value = String(DEFAULT_TILES);
    find("[data-ratio]").value = DEFAULT_RATIO;
    root.querySelectorAll("[data-mode]").forEach(button =>
      button.setAttribute("aria-pressed", String(button.dataset.mode === mode)));
    find(".pipeline-controls").hidden = false;
    rebuild();
    render();
  });
})();
