// Execute the demo page's script in a stub DOM and count what it builds.
//
//     node experiments/check_demo.js
//
// This exists because I shipped a broken page. `const truthEl` was declared after the loop that
// used it -- hoisted but in the temporal dead zone -- so the script threw a ReferenceError, died,
// and left exactly one static tile visible where six were expected. Nothing in the Python suite
// could see that: the markup was fine and the failure was at runtime.
// Minimal DOM good enough to execute the page's script and count what it builds.
const fs = require('fs');
const html = fs.readFileSync('demo/index.html', 'utf8');
const script = html.slice(html.indexOf('const DATA ='), html.lastIndexOf('</script>'));

let rafCalls = 0;
class El {
  constructor(tag='div'){ this.tag=tag; this.children=[]; this._html=''; this.className='';
    this.style={}; this.classList={toggle(){}, add(){}, remove(){}}; this.textContent=''; }
  set innerHTML(v){ this._html=v; }
  get innerHTML(){ return this._html; }
  appendChild(c){ this.children.push(c); return c; }
  insertBefore(c, ref){ const i=this.children.indexOf(ref);
    if(i<0) throw new Error('insertBefore: reference node not found');
    this.children.splice(i,0,c); return c; }
  querySelector(sel){ return this.children.find(c => c.className.includes(sel.replace('.',''))) || null; }
  getContext(){ return {fillStyle:'', fillRect(){}}; }
  addEventListener(){}
}
const boards = new El(); const truthTile = new El(); truthTile.className='board truth';
boards.appendChild(truthTile);
const byId = { boards, truth:new El('canvas'), play:new El('button'), speed:{value:'18'},
               restart:new El('button'), clock:new El(), table:new El() };
byId.table.querySelector = () => { const tb=new El(); tb.appendChild=(c)=>{tb.children.push(c);return c;}; return tb; };
global.document = {
  getElementById: id => byId[id] || (byId[id] = new El('canvas')),
  createElement: t => new El(t),
  querySelector: s => byId.table,
  documentElement: { style:{}, setAttribute(){}, getAttribute(){return null;} },
};
global.getComputedStyle = () => ({ getPropertyValue: () => '#000' });
global.requestAnimationFrame = () => { rafCalls++; };
global.matchMedia = () => ({ addEventListener(){} });
global.MutationObserver = class { observe(){} };

try {
  new Function(script)();
  const entrantTiles = boards.children.filter(c => !c.className.includes('truth')).length;
  console.log('SCRIPT RAN with no error');
  console.log('  entrant boards built :', entrantTiles);
  console.log('  truth tiles          :', boards.children.filter(c=>c.className.includes('truth')).length);
  console.log('  total tiles in grid  :', boards.children.length);
  console.log('  animation started    :', rafCalls > 0);
  console.log(entrantTiles === 5 && boards.children.length === 6
    ? '  => 6 mazes on the page. correct.'
    : '  => WRONG tile count');
} catch (e) {
  console.log('SCRIPT THREW:', e.constructor.name + ':', e.message);
  process.exit(1);
}
