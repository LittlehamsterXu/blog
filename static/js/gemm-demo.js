// These diagrams count logical scalar input reads, never physical memory transactions.
const A = [[2, 1], [3, 2]];
const B = [[1, 2, 3, 4], [2, 1, 2, 1]];
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const colors = {a:'#b85d22', b:'#2774a6', c:'#287560'};
const text = (x,y,value,cls='',anchor='middle') => `<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${value}</text>`;
const box = (x,y,w,h,cls='g-base',rx=6) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" class="${cls}"/>`;
const cell = (x,y,w,h,label,value,kind='',active=true) => `<g${active?'':' class="g-inactive"'}>${box(x,y,w,h,kind?`g-${kind}`:'g-base')}${text(x+w/2,y+14,label,`g-small ${kind?`g-${kind}-text`:''}`)}${text(x+w/2,y+h-10,value,`g-number ${kind?`g-${kind}-text`:''}`)}</g>`;
function arrow(points,kind,moving=false) {
  const d = points.map(([x,y],i)=>`${i?'L':'M'}${x} ${y}`).join(' ');
  const [x,y] = points.at(-1), [px,py] = points.at(-2);
  const angle = Math.atan2(y-py,x-px)*180/Math.PI;
  return `<path d="${d}" class="g-arrow" stroke="${colors[kind]}"/><path d="M-7 -4 L0 0 L-7 4" fill="none" stroke="${colors[kind]}" stroke-width="2" transform="translate(${x},${y}) rotate(${angle})"/>${moving&&!reduceMotion.matches?`<circle r="4" fill="${colors[kind]}"><animateMotion dur=".85s" path="${d}" fill="freeze"/></circle>`:''}`;
}
const svg = (w,h,label,body,cls='') => `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}" role="img" aria-label="${label}" class="${cls}">${body}</svg>`;
const stat = (value,label) => `<div class="gemm-stat"><b>${value}</b><span>${label}</span></div>`;
const panel = (title,desc,drawing) => `<section class="gemm-panel"><h4>${title}</h4><p>${desc}</p>${drawing}</section>`;

function dotScene(step,target,moving) {
  const r=Math.floor(target/2), c=target%2;
  const k=step===1?0:1;
  let body = text(80,24,'A · 2×2','g-heading')+text(265,24,'B · 2×2','g-heading')+text(450,24,'C · 2×2','g-heading');
  for(let i=0;i<2;i++)for(let j=0;j<2;j++) {
    const ac=step>0&&i===r&&j===k, bc=step>0&&i===k&&j===c;
    body+=cell(25+j*57,42+i*57,50,50,`A${i}${j}`,A[i][j],ac?'a':'');
    body+=cell(210+j*57,42+i*57,50,50,`B${i}${j}`,B[i][j],bc?'b':'');
    const sum=i===r&&j===c? (step===0?0: A[r][0]*B[0][c]+(step>=2?A[r][1]*B[1][c]:0)):'—';
    body+=cell(395+j*57,42+i*57,50,50,`C${i}${j}`,sum,i===r&&j===c?'c':'');
  }
  body+=text(174,103,'×','g-number')+text(360,103,'=','g-number');
  if(step>0&&step<3) {
    body+=arrow([[50+k*57,97+r*57],[50+k*57,185],[167,225]],'a',moving);
    body+=arrow([[235+c*57,97+k*57],[235+c*57,185],[267,225]],'b',moving);
    body+=text(167,246,A[r][k],'g-number')+text(217,246,'×','g-number')+text(267,246,B[k][c],'g-number');
    body+=text(322,246,'→','g-number')+text(410,242,`sum = ${step===1?A[r][0]*B[0][c]:A[r][0]*B[0][c]+A[r][1]*B[1][c]}`,'g-heading');
    body+=text(265,284,step===3?`k 已算完 → 将 sum 写回 C${r}${c}`:`k = ${k}：${step===1?'0':A[r][0]*B[0][c]} + ${A[r][k]} × ${B[k][c]}`,'g-heading');
  } else if(step===3) {
    body+=text(265,241,`sum = ${A[r][0]*B[0][c]+A[r][1]*B[1][c]} → 写回 C${r}${c}`,'g-heading');
    body+=arrow([[400,223],[420+c*57,163],[420+c*57,97+r*57]],'c',moving);
  } else body+=text(265,235,`选中 C${r}${c}：先令 sum = 0，再沿 k 累加。`,'g-heading');
  return {html:`<div class="gemm-dot-wrap">${svg(540,305,`矩阵乘法，观察 C${r}${c}`,body,'gemm-dot-svg')}</div>`,stats:stat(step===0?'—':step===3?'完成':k,'当前 k')+stat(step===0?0:step===1?1:2,'已完成的乘加')+stat(step===0?0:A[r][0]*B[0][c]+(step>=2?A[r][1]*B[1][c]:0),`C${r}${c} 的部分和`),explain:[`固定输出 C${r}${c}。A 选第 ${r} 行，B 选第 ${c} 列；k 每次移动一格。`,`k=0：先乘 ${A[r][0]}×${B[0][c]}，得到 ${A[r][0]*B[0][c]}。这只是第一项，sum 要保留。`,`k=1：再乘 ${A[r][1]}×${B[1][c]}，把 ${A[r][1]*B[1][c]} 加入已有的 ${A[r][0]*B[0][c]}。输出位置始终没变。`,`两项已累加完毕，将结果 ${A[r][0]*B[0][c]+A[r][1]*B[1][c]} 写回 C${r}${c}。换一个输出，仍然做同样的点积。`][step]};
}

function tileScene(step,moving) {
  const aa=[[1,2,3,4],[2,1,2,1],[1,1,2,2],[2,2,1,1]];
  const bb=[[1,2,1,2],[2,1,2,1],[1,1,2,2],[2,2,1,1]];
  const phase=step>=4?1:0, loaded=step>0, n=step>=5?4:step>=2?2:0;
  let nodes=text(95,21,'A · 4×4','g-heading')+text(289,21,'B · 4×4','g-heading')+text(480,21,'C · 左上 2×2','g-heading');
  for(let r=0;r<4;r++)for(let c=0;c<4;c++) {
    const ac=loaded&&r<2&&c>=phase*2&&c<phase*2+2;
    const bc=loaded&&r>=phase*2&&r<phase*2+2&&c<2;
    nodes+=box(20+c*38,38+r*38,34,34,ac?'g-a':'g-base')+text(37+c*38,61+r*38,aa[r][c],ac?'g-a-text':'g-muted');
    nodes+=box(213+c*38,38+r*38,34,34,bc?'g-b':'g-base')+text(230+c*38,61+r*38,bb[r][c],bc?'g-b-text':'g-muted');
  }
  for(let r=0;r<2;r++)for(let c=0;c<2;c++) {
    let sum=0;for(let k=0;k<n;k++)sum+=aa[r][k]*bb[k][c];
    nodes+=cell(411+c*62,57+r*62,55,55,`C${r}${c}`,sum,'c');
  }
  nodes+=text(95,213,`A 选择列 ${phase*2}、${phase*2+1}`,'g-small')+text(289,213,`B 选择行 ${phase*2}、${phase*2+1}`,'g-small')+text(473,213,'输出位置始终不动','g-small');
  nodes+=arrow([[36,237],[160,237]],'a',step===4&&moving)+text(98,258,'下一轮 → 右移','g-small');
  nodes+=arrow([[384,46],[384,178]],'b',step===4&&moving);
  if(loaded) {
    nodes+=text(276,291,step===3?'本轮读完，先同步，再覆盖 Shared':step===6?'所有 K 块算完：写回最终 C':`tile = ${phase}：当前窗口覆盖 k = ${phase*2}、${phase*2+1}`,'g-heading');
    nodes+=text(276,318,n===0?'尚未计算：C 的部分和仍为 0':n===2?'C00 = 1×1 + 2×2 = 5（保留，后面继续加）':'C00 = 5 + 3×1 + 4×2 = 16（两轮相加）','g-muted');
  } else nodes+=text(276,291,'每轮只把彩色窗口搬到 Shared，输出部分和留在寄存器。','g-small');
  return {html:`<div class="gemm-dot-wrap">${svg(555,344,'A 窗口右移，B 窗口下移，C 累加不移动',nodes,'gemm-dot-svg')}</div>`,stats:stat(step===0?'—':phase,'tile 轮次')+stat(`${n} / 4`,'每个输出已累加的 k 数')+stat(n===0?0:n===2?5:16,'C00 的部分和'),explain:['先固定 C 左上角的 2×2 输出。完整 K=4，但每轮只处理 BK=2 个 k。','第 0 轮：A 取前两行的第 0、1 列，B 取第 0、1 行的前两列。将两块输入放进 Shared，搬完后同步。','在当前窗口里完成 k=0、1 的乘加：C 的部分和变成 [5,4;4,5]。这还不是最终结果，不要写回后清零。','所有线程用完第 0 轮输入，第二次同步完成。Shared 可以换下一块，但各线程的 C 累加器继续保留。','第 1 轮：A 窗口向右，B 窗口向下。加载 k=2、3 的输入；绿色 C 窗口没有移动。','将 k=2、3 的乘积加进原来的部分和，得到 [16,15;8,9]。移动的是输入窗口，积累的是同一块输出。','K 方向已经处理完，现在把寄存器中的最终结果写回 C。外层 tile 管窗口移动，内层 k 管窗口内的乘加。'][step]};
}

function sharedScene(step,moving) {
  // Step 1 stages all four inputs; step 2 is the barrier; steps 3..6 update four outputs.
  const count=Math.max(0,step-2), current=count-1;
  const r=Math.floor(current/2), c=current%2;
  const vals=[2,3,1,2], names=['A0k','A1k','Bk0','Bk1'];
  function draw(shared) {
    let arrows='', nodes=text(160,21,'Global · 固定 k=0','g-heading');
    for(let i=0;i<4;i++) {
      const used=current>=0&&(i===r||i===c+2);
      nodes+=cell(20+i*74,36,62,45,names[i],vals[i],i<2?'a':'b',!shared||step===1||used);
      if(shared) {
        nodes+=cell(20+i*74,144,62,45,names[i],step>=1?vals[i]:'·',step>=1?(i<2?'a':'b'):'');
        if(step===1) arrows+=arrow([[51+i*74,84],[51+i*74,140]],i<2?'a':'b',moving);
      }
    }
    nodes+=text(160,shared?130:144,shared?'Shared · 这个 Block 的公共输入':'无显式 Shared 暂存','g-muted');
    nodes+=text(160,229,shared?'四个线程读取 Shared，各算一格':'四个线程直接读取 Global','g-small');
    for(let i=0;i<4;i++) {
      const rr=Math.floor(i/2),cc=i%2,x=84+cc*80,y=247+rr*62;
      nodes+=cell(x,y,68,53,`T${i} → C${rr}${cc}`,i<count?vals[rr]*vals[cc+2]:0,i<count?'c':'');
      if(i===current) {
        const fromY=shared?193:84;
        arrows+=arrow([[51+rr*74,fromY],[12,fromY+18],[12,y+26],[x-3,y+26]],'a',moving);
        arrows+=arrow([[51+(cc+2)*74,fromY],[308,fromY+18],[308,y+26],[x+71,y+26]],'b',moving);
      }
    }
    if(shared&&step===2) nodes+=text(160,211,'✓ __syncthreads()：输入到齐','g-heading');
    return svg(320,378,shared?'Shared 复用数据路径':'Naive 重复读取路径',arrows+nodes);
  }
  const lines=['两边都要完成 C00、C01、C10、C11 的同一轮增量。先看四份输入的位置。','右边四个线程协作，将两个 A、两个 B 搬进 Shared。左边没有这张公共工作台。','右边等待 Block 同步，确保所有输入都已到位。接下来不再为这一轮填充 Shared 而读 Global。','更新 C00：2×1=2。左边从 Global 取两值；右边从 Shared 取相同两值。','更新 C01：2×2=4。橙色 A0k 又用一次；右边复用工作台上的 A0k，Global 读取量仍是 4。','更新 C10：3×1=3。蓝色 Bk0 再次参与计算；右边仍然不增加 Global 读取量。','更新 C11：3×2=6。四次乘加都完成。左边逻辑读 8 个 Global 输入值，右边只搬 4 个，但额外使用了 Shared 和同步。'];
  return {html:`<div class="gemm-compare">${panel('Naive · 各自取料',`Global 逻辑输入读取：${count*2} 个值`,draw(false))}${panel('Shared · 先搬一次',`Global 逻辑输入读取：${step>=1?4:0} 个值`,draw(true))}</div>`,stats:stat(count*2,'左：Global 输入值')+stat(step>=1?4:0,'右：Global 输入值')+stat(`${count} / 4`,'两边各自完成的 FMA'),explain:lines[step]};
}

function groupsFor(mode) {
  if(mode==='1x1') return Array.from({length:8},(_,i)=>({r:Math.floor(i/4),c:i%4,rows:1,cols:1}));
  if(mode==='1x4') return [{r:0,c:0,rows:1,cols:4},{r:1,c:0,rows:1,cols:4}];
  return [{r:0,c:0,rows:2,cols:4}];
}
function registerScene(step,mode,moving) {
  const groups=groupsFor(mode), rounds=groups.length*2;
  const max=rounds*2+1, finished=step===max;
  const job=step===0?0:Math.min(rounds-1,Math.floor((step-1)/2));
  const k=Math.floor(job/groups.length), gi=job%groups.length, g=groups[gi];
  const computing=step>0&&step%2===0&&!finished;
  const loadedJobs=Math.min(rounds,Math.ceil(step/2));
  const completedJobs=Math.min(rounds,Math.floor(step/2));
  const sums=Array(8).fill(0);
  for(let j=0;j<completedJobs;j++) {
    const gg=groups[j%groups.length],kk=Math.floor(j/groups.length);
    for(let r=gg.r;r<gg.r+gg.rows;r++)for(let c=gg.c;c<gg.c+gg.cols;c++) sums[r*4+c]+=A[r][kk]*B[kk][c];
  }
  const active=step>0&&!finished;
  let nodes=text(300,23,`k = ${k} · ${mode.replace('x','×')} 分工 · 输入从 Shared 读入线程寄存器`,'g-heading');
  let arrows='';
  for(let r=0;r<2;r++) nodes+=cell(16,151+r*77,66,62,`a${r}=A[${r},${k}]`,A[r][k],'a',active&&r>=g.r&&r<g.r+g.rows);
  for(let c=0;c<4;c++) nodes+=cell(146+c*103,49,88,54,`b${c}=B[${k},${c}]`,B[k][c],'b',active&&c>=g.c&&c<g.c+g.cols);
  nodes+=text(47,137,'A · 小托盘','g-small')+text(350,124,'B · 小托盘','g-small');
  for(let r=0;r<2;r++)for(let c=0;c<4;c++) {
    const hit=active&&r>=g.r&&r<g.r+g.rows&&c>=g.c&&c<g.c+g.cols;
    nodes+=cell(146+c*103,151+r*77,88,62,`C${r}${c}`,sums[r*4+c],sums[r*4+c]?'c':'');
    if(computing&&hit) {
      arrows+=arrow([[85,182+r*77],[105,182+r*77],[105,222+r*77],[190+c*103,222+r*77],[190+c*103,216+r*77]],'a',moving);
      arrows+=arrow([[190+c*103,106],[243+c*103,113],[243+c*103,182+r*77],[237+c*103,182+r*77]],'b',moving);
    }
  }
  groups.forEach((gg,i)=> { nodes+=box(141+gg.c*103,146+gg.r*77,gg.cols*103-5,gg.rows*77-5,`g-thread ${active&&i===gi?'g-thread-current':''}`,8); });
  nodes+=text(300,330,finished?'全部 k 已结束：八个输出相同，可写回 C':active?`当前线程 T${gi}：负责 ${g.rows} 行 × ${g.cols} 列 · ${computing?'乘加并保留部分和':'加载本线程的 A/B 寄存器值'}`:'虚线框 = 一个线程负责的输出；初始部分和全部为 0','g-heading');
  nodes+=text(300,356,computing?`这一组 ${g.rows+g.cols} 个输入 → ${g.rows*g.cols} 次 FMA`:'每个线程的累加器跨 k 保留，不会在下一轮清零','g-muted');
  const reads=loadedJobs*(g.rows+g.cols), fmas=completedJobs*g.rows*g.cols;
  const explain=finished?`三种分工最终都会得到 [4,5,8,9; 7,8,13,14]。当前模式两轮 k 共逻辑读取 ${reads} 个 Shared 输入值，完成 16 次 FMA；线程数量和资源代价也随分工变化。`:step===0?'选择分工，再点播放。比较时始终计算这同一块输出；切换模式会清零重来。橙色沿行复用，蓝色沿列复用。':computing?`k=${k}，T${gi} 用刚才读入的 ${g.rows} 个 A、${g.cols} 个 B 更新 ${g.rows*g.cols} 个累加器。${g.cols>1?'同一个 A 沿橙色箭头服务四列。':''}${g.rows>1?'同一个 B 沿蓝色箭头服务两行。':''}`:`k=${k}，T${gi} 从 Shared 读入 ${g.rows+g.cols} 个值，暂存在线程寄存器中。紫色虚线圈出的输出由它负责；加载阶段还没做新的乘加。`;
  return {html:`<div class="gemm-register-wrap">${svg(600,377,'线程分工与寄存器外积，橙色 A 横向复用，蓝色 B 纵向复用',arrows+nodes,'gemm-register-svg')}</div>`,stats:stat(reads,'累计 Shared 标量输入读取')+stat(`${fmas} / 16`,'累计 FMA（相同工作量）')+stat(groups.length,'负责该输出区域的线程数'),explain};
}

function wideScene(step,moving) {
  function draw(wide) {
    let nodes=text(160,20,'Global · 同一段 BK=16','g-heading'),arrows='';
    nodes+=cell(21,37,86,51,'同一份 A','512','a')+cell(126,37,76,51,'B 左','512','b')+cell(218,37,76,51,'B 右','512','b');
    nodes+=text(160,115,'下方是各 Block 私有的 Shared','g-small');
    const xs=wide?[19]:[9,171];
    for(let b=0;b<(wide?1:2);b++) {
      const x=xs[b],w=wide?282:140;
      nodes+=box(x,132,w,227,'g-area',10)+text(x+w/2,154,wide?'一个 Block':'Block '+b,'g-heading');
      const aLoaded=wide?step>=1:step>=b+1;
      const bLoaded=step>=3;
      const ax=wide?35:x+9,aw=wide?88:52;
      nodes+=cell(ax,168,aw,49,'A',aLoaded?'512':'·',aLoaded?'a':'');
      if(wide) {
        nodes+=cell(142,168,64,49,'B 左',bLoaded?'512':'·',bLoaded?'b':'')+cell(221,168,64,49,'B 右',bLoaded?'512':'·',bLoaded?'b':'');
      } else nodes+=cell(x+76,168,55,49,b===0?'B 左':'B 右',bLoaded?'512':'·',bLoaded?'b':'');
      if(step===1&&b===0||!wide&&step===2&&b===1) arrows+=arrow([[64,91],[ax+aw/2,164]],'a',moving);
      if(step===3) {
        if(wide) {arrows+=arrow([[164,91],[174,164]],'b',moving)+arrow([[256,91],[253,164]],'b',moving);}
        else arrows+=arrow([[b===0?164:256,91],[x+103,164]],'b',moving);
      }
      const cx=wide?35:x+9,cw=wide?250:122;
      nodes+=box(cx,260,cw,72,step>=4?'g-c':'g-base');
      if(wide) nodes+=`<path d="M160 260 V332" stroke="${colors.c}" stroke-dasharray="4 3"/>`;
      nodes+=text(cx+cw/2,288,wide?'C · 32 行 × 64 列':`C ${b===0?'左':'右'} · 32×32`,'g-small')+text(cx+cw/2,312,wide?'左半 ← 同一份 A → 右半':'各自使用一份 A','g-small');
      if(step>=4) {
        if(wide) arrows+=arrow([[79,220],[79,239],[95,256]],'a',moving)+arrow([[79,220],[79,239],[222,239],[222,256]],'a',moving);
        else arrows+=arrow([[ax+aw/2,220],[ax+aw/2,256]],'a',moving);
      }
    }
    return svg(320,377,wide?'一个宽 Block 共用 A':'两个 Block 各自加载 A',arrows+nodes);
  }
  const old=step===0?0:step===1?512:step===2?1024:2048;
  const newer=step===0?0:step<3?512:1536;
  return {html:`<div class="gemm-compare">${panel('两块 32×32 输出',`逻辑输入：${old} floats`,draw(false))}${panel('一块 32×64 输出',`逻辑输入：${newer} floats`,draw(true))}</div>`,stats:stat(old,'左：Global → Shared 元素数')+stat(newer,'右：Global → Shared 元素数')+stat(step>=3?'25%':'—','最终省下的逻辑输入'),explain:['两边负责完全相同的 C 区域。左边用两个 Block，右边用一个更宽的 Block。每个 Block 都有自己的一份 Shared。','先搬 A：两种方案都搬 512 个值。右边这份 A 已能服务整个宽输出块。','左边第二个 Block 也要 A，于是同一份 global 输入再加载 512 个值到另一份 Shared。右边无需再搬。','搬 B：左右半区需要不同的 B，两种方案都必须搬 1024 个值。累计左 2048、右 1536。','开始使用输入：右侧橙色路径分叉，同一个 Shared A 服务输出的左右半区；左侧每个 Block 使用各自的 A。','本轮输入逻辑加载减少 25%，乘加数量保持不变。这是更宽输出块带来的复用收益；实际耗时还受缓存、并行度与资源影响。'][step]};
}

for (const root of document.querySelectorAll('[data-gemm]')) {
  const kind=root.dataset.gemm;
  let step=0, mode='2x4', target=0, timer=null;
  const play=root.querySelector('[data-play]'),seek=root.querySelector('[data-seek]'),speed=root.querySelector('[data-speed]');
  const limit=()=>kind==='dot'?3:kind==='shared'||kind==='tile'?6:kind==='wide'?5:groupsFor(mode).length*4+1;
  function pause(){if(timer!==null)clearInterval(timer);timer=null;play.textContent='播放';}
  function render(moving=false){
    const scene=kind==='dot'?dotScene(step,target,moving):kind==='shared'?sharedScene(step,moving):kind==='tile'?tileScene(step,moving):kind==='register'?registerScene(step,mode,moving):wideScene(step,moving);
    root.querySelector('[data-stage]').innerHTML=scene.html;
    root.querySelector('[data-stats]').innerHTML=scene.stats;
    root.querySelector('[data-explain]').textContent=scene.explain;
    root.querySelector('[data-progress]').textContent=`${step} / ${limit()}`;
    seek.max=limit();seek.value=step;
    root.querySelector('[data-prev]').disabled=step===0;
    root.querySelector('[data-next]').disabled=step===limit();
    root.querySelectorAll('[data-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.mode===mode)));
  }
  function start(){if(step===limit())step=0;render();play.textContent='暂停';timer=setInterval(()=>{step++;render(true);if(step>=limit())pause();},Number(speed.value));}
  play.addEventListener('click',()=>timer===null?start():pause());
  for(const [sel,delta] of [['[data-prev]',-1],['[data-next]',1]])root.querySelector(sel).addEventListener('click',()=>{pause();step=Math.max(0,Math.min(limit(),step+delta));render(true);});
  root.querySelector('[data-reset]').addEventListener('click',()=>{pause();step=0;render();});
  root.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>{pause();mode=b.dataset.mode;step=0;render();}));
  root.querySelector('[data-target]')?.addEventListener('change',e=>{pause();target=Number(e.target.value);step=0;render();});
  seek.addEventListener('input',()=>{pause();step=Number(seek.value);render();});
  speed.addEventListener('change',()=>{if(timer!==null){pause();start();}});
  new IntersectionObserver(entries=>{if(!entries[0].isIntersecting)pause();}).observe(root);
  document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
  root.querySelector('.gemm-controls').hidden=false;
  render();
}
