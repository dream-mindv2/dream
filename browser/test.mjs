// The browser VM driven as a page's worker drives it, with a std.graphics
// program: the counter example's frontend. What this checks is the VM's half
// -- the image loads, slices run, the bridge carries the protocol both ways
// (renders as patches, events by the id a node was given),
// a backend reply's integers survive, a closed window ends the program, and
// bad bytecode is refused -- not the page's, which mind/std/graphics tests.
//
//     node dream/browser/test.mjs build-browser/dream-browser.mjs counter/assets/frontend.dream
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';

const [runtimePath, imagePath] = process.argv.slice(2);
const { default: create } = await import(pathToFileURL(resolve(runtimePath)));
const messages = [], errors = [];
const runtime = await create({ onMessage: text => messages.push(JSON.parse(text)), onError: text => errors.push(text),
  print: () => {}, printErr: () => {} });
for (const name of ['load', 'tick', 'delay', 'event', 'exit_code', 'dispose'])
  assert.equal(typeof runtime[`_dream_browser_${name}`], 'function', `missing browser runtime export: ${name}`);
const bytes = await readFile(imagePath);
function pass(bytes, fn) {
  const ptr = runtime._malloc(bytes.length);
  assert.ok(ptr);
  try { runtime.HEAPU8.set(bytes, ptr); return fn(ptr, bytes.length); }
  finally { runtime._free(ptr); }
}
const send = event => assert.equal(pass(Buffer.from(JSON.stringify(event)), runtime._dream_browser_event), 1);
assert.equal(pass(bytes, runtime._dream_browser_load), 1, errors.join('\n'));
function pumpUntil(predicate, what) {
  for (let i = 0; i < 20000; i++) {
    const status = runtime._dream_browser_tick();
    assert.notEqual(status, -1, errors.join('\n'));
    if (predicate(status)) return;
    assert.notEqual(status, 0, `frontend ended before ${what}`);
  }
  assert.fail(`frontend exceeded slice budget waiting for ${what}`);
}
const renders = () => messages.filter(m => m.type === 'render');
// The tree the page would show: the first render's, and each patch applied to
// the one before (std.graphics `diff`). Fields left out are empty.
function apply(old, entry) {
  if (!('o' in entry)) return entry;
  const { o, c, ...fields } = entry;
  const node = 't' in entry ? fields : (({ c, ...rest }) => rest)(old);
  const kept = old.c ?? [];
  node.c = c === undefined ? kept : c.flatMap(e => Array.isArray(e) ? kept.slice(e[0], e[0] + e[1])
    : 'o' in e ? [apply(kept[e.o], e)] : [e]);
  return node;
}
const shown = () => renders().reduce((tree, m) => apply(tree, m.patch), null);
const textOf = node => node.t === '#text' ? node.x ?? '' : (node.c ?? []).map(textOf).join('');
const find = (node, f) => f(node) ? node : (node.c ?? []).map(c => find(c, f)).find(Boolean);
const output = () => textOf(find(shown(), n => n.t === 'output'));
const button = label => find(shown(), n => n.t === 'button' &&
  (n.a?.['aria-label'] === label || textOf(n) === label));

pumpUntil(() => renders().length === 1, 'the first render');
assert.deepEqual(messages[0], { type: 'open', options: { title: 'Dream counter', width: 960, height: 640 } });
assert.ok(messages.some(m => m.type === 'styles' && m.css.includes('.counter')), 'the program stylesheet is sent');
assert.equal(output(), '0');
const add = button('Increase count');
assert.deepEqual(add.e, ['click']);
send({ type: 'dom', id: add.i, event: 'click', value: null });
pumpUntil(() => renders().length === 2, 'the second render');
assert.equal(output(), '1');
send({ type: 'dom', id: [0, 'nowhere', 7], event: 'click', value: null });
send({ type: 'dom', id: button('Call Dream backend').i, event: 'click', value: null });
pumpUntil(() => messages.some(m => m.type === 'request'), 'the backend request');
const request = messages.find(m => m.type === 'request');
assert.deepEqual([request.service, request.args], ['backend', { command: 'echo', payload: '1' }]);
// Past 2^53, as the native backend sends it: JSON text JavaScript never parses.
send({ type: 'reply', id: request.id, payload: '{"ok":true,"value":9007199254740993}' });
pumpUntil(() => textOf(shown()).includes('9007199254740993'), 'the reply');
send({ type: 'close' });
pumpUntil(status => status === 0, 'the end');
assert.ok(messages.some(m => m.type === 'close'), 'the driver closed');
assert.equal(runtime._dream_browser_exit_code(), 0);
assert.equal(errors.length, 0, errors.join('\n'));
runtime._dream_browser_dispose();
assert.equal(pass(Buffer.from('bad image'), runtime._dream_browser_load), 0);
assert.ok(errors.length);
console.log('WebAssembly: a std.graphics program rendered, handled events, called its backend exactly, closed, and bad bytecode was refused');
