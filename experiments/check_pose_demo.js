// Execute the generated page's own script against a stub DOM and assert what a person would
// look for. No Python test can see a runtime browser failure -- the maze demo shipped once with
// a temporal-dead-zone ReferenceError that killed the script and left one tile on the page,
// and every Python test passed.
const fs = require('fs');
const html = fs.readFileSync(__dirname + '/../demo/pose.html', 'utf8');
const src = html.slice(html.indexOf('<script>') + 8, html.lastIndexOf('</script>'));

let rafCalls = 0, painted = 0;
const canvasStub = (id) => ({
  id, width: 0, height: 0,
  getContext: () => ({
    createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }),
    putImageData: () => { painted++; },
    strokeRect: () => {}, strokeStyle: '', lineWidth: 0,
  }),
});
const els = {};
const document = {
  getElementById: (id) => (els[id] ||= id === 'world' || id === 'eye' || id === 'built'
    ? canvasStub(id)
    : { id, textContent: '', innerHTML: '', className: '', set onclick(f) { this._f = f; } }),
  documentElement: {},
};
const getComputedStyle = () => ({ getPropertyValue: () => '#4aa3ff' });
const requestAnimationFrame = () => { rafCalls++; };

let err = null;
try { new Function('document', 'getComputedStyle', 'requestAnimationFrame', src)(
  document, getComputedStyle, requestAnimationFrame); } catch (e) { err = e; }

const ok = (c, m) => { if (!c) { console.error('FAIL: ' + m); process.exit(1); } };
ok(!err, 'script threw: ' + (err && err.stack));
ok(painted >= 3, `expected all three panels painted, got ${painted}`);
ok(rafCalls > 0, 'animation never started');
ok(els.who.innerHTML.length > 0, 'the shape is never named');
ok(/NEVER SEEN|familiar/.test(els.tag.textContent), 'the held-out label is missing');
ok(els.world.width === els.built.width, 'world and reconstruction are drawn at different sizes');
ok(els.eye.width < els.world.width, 'the eye is not smaller than the world -- not a fovea');

console.log('SCRIPT RAN with no error');
console.log(`  panels painted    : ${painted}`);
console.log(`  world / eye       : ${els.world.width}px / ${els.eye.width}px`);
console.log(`  now showing       : ${els.who.innerHTML.replace(/<[^>]+>/g, '')} [${els.tag.textContent}]`);
console.log(`  animation started : ${rafCalls > 0}`);
console.log('=> the page runs and shows a shape, a keyhole, and a reconstruction.');
