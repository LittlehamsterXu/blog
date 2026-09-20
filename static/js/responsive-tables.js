// Only scrollable tables need a hint and a keyboard focus stop.
for (const wrapper of document.querySelectorAll('.post-content .article-table')) {
  const scroll = wrapper.querySelector('.table-scroll');
  const table = scroll.querySelector('table');
  const hint = wrapper.querySelector('.table-scroll-hint');
  function update() {
    const overflows = scroll.scrollWidth > scroll.clientWidth + 1;
    hint.hidden = !overflows;
    if (overflows) scroll.setAttribute('tabindex', '0');
    else scroll.removeAttribute('tabindex');
  }
  const observer = new ResizeObserver(update);
  observer.observe(scroll);
  observer.observe(table);
  update();
}
