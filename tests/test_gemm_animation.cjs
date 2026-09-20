// Independent arithmetic checks for the teaching diagrams. Run: node tests/test_gemm_animation.cjs
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const context = vm.createContext({window:{matchMedia:()=>({matches:true})},document:{querySelectorAll:()=>[]}});
vm.runInContext(fs.readFileSync('static/js/gemm-demo.js','utf8'),context);
const scene=(name,...args)=>vm.runInContext(`${name}(...${JSON.stringify(args)})`,context);
const counts=html=>Array.from(html.matchAll(/<b>(.*?)<\/b>/g),m=>m[1]);
const outputs=html=>Array.from(html.matchAll(/>[^<]*C\d\d<\/text><text[^>]*>(\d+)<\/text>/g),m=>Number(m[1]));
// Compare against ordinary matrix multiplication, not the animation's grouping logic.
const a=[[2,1],[3,2]], b=[[1,2,3,4],[2,1,2,1]];
const expected=a.flatMap(row=>b[0].map((_,c)=>row.reduce((v,x,k)=>v+x*b[k][c],0)));
for(const [mode,last,reads,threads] of [['1x1',33,32,8],['1x4',9,20,2],['2x4',5,12,1]]) {
  const final=scene('registerScene',last,mode,false);
  assert.deepEqual(outputs(final.html),expected);
  assert.deepEqual(counts(final.stats),[String(reads),'16 / 16',String(threads)]);
  let previous=Array(8).fill(0);
  for(let step=0;step<=last;step++) {
    const current=outputs(scene('registerScene',step,mode,false).html);
    current.forEach((value,i)=>assert.ok(value>=previous[i]&&value<=expected[i]));
    previous=current;
  }
}
assert.deepEqual(counts(scene('sharedScene',6,false).stats),['8','4','4 / 4']);
assert.deepEqual(outputs(scene('sharedScene',6,false).html),[2,4,3,6,2,4,3,6]);
assert.deepEqual(counts(scene('wideScene',5,false).stats),['2048','1536','25%']);
assert.deepEqual(outputs(scene('tileScene',2,false).html),[5,4,4,5]);
assert.deepEqual(outputs(scene('tileScene',6,false).html),[16,15,8,9]);
for(let target=0;target<4;target++)assert.equal(counts(scene('dotScene',3,target,false).stats)[2],String([4,5,7,8][target]));
console.log('GEMM diagrams: all three mappings produce identical results; read counts and K-tile sums verified.');
