import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';

const [runtimePath, imagePath] = process.argv.slice(2);
const { default: create } = await import(pathToFileURL(resolve(runtimePath)));
const messages = [], errors = [];
const runtime = await create({ onMessage: text => messages.push(JSON.parse(text)), onError: text => errors.push(text) });
const bytes = await readFile(imagePath);
function pass(bytes, fn) {
  const ptr = runtime._malloc(bytes.length);
  assert.ok(ptr);
  try { runtime.HEAPU8.set(bytes, ptr); return fn(ptr, bytes.length); }
  finally { runtime._free(ptr); }
}
assert.equal(pass(bytes, runtime._dream_browser_load), 1, errors.join('\n'));
function pumpUntil(predicate) {
  for (let i = 0; i < 10000; i++) {
    const status = runtime._dream_browser_tick();
    assert.notEqual(status, -1, errors.join('\n'));
    if (predicate()) return;
    assert.notEqual(status, 0, 'frontend ended early');
  }
  assert.fail('frontend exceeded slice budget');
}
const renders = () => messages.filter(m => m.type === 'render');
pumpUntil(() => renders().length === 1);
assert.equal(messages[0].type, 'open');
function textOf(node) { return node.tag === '#text' ? node.text : node.children.map(textOf).join(''); }
assert.match(textOf(renders()[0].node), /0/);
assert.equal(pass(Buffer.from(JSON.stringify({ name: 'increment', key: 'add', value: null })), runtime._dream_browser_event), 1);
pumpUntil(() => renders().length === 2);
assert.match(textOf(renders()[1].node), /1/);
assert.equal(pass(Buffer.from(JSON.stringify({ name: 'save', key: 'save', value: null })), runtime._dream_browser_event), 1);
pumpUntil(() => messages.some(m => m.type === 'invoke'));
assert.deepEqual(messages.find(m => m.type === 'invoke'), { type: 'invoke', reply: 'saved', command: 'echo', payload: '1' });
assert.equal(pass(Buffer.from(JSON.stringify({ name: 'close', key: '', value: null })), runtime._dream_browser_event), 1);
pumpUntil(() => messages.some(m => m.type === 'close'));
assert.equal(errors.length, 0, errors.join('\n'));
runtime._dream_browser_dispose();
assert.equal(pass(Buffer.from('bad image'), runtime._dream_browser_load), 0);
assert.ok(errors.length);
console.log('WebAssembly: Dream UI rendered, handled input, invoked backend, closed, and rejected malformed bytecode');
