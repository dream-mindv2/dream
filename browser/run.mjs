// Run a Dream image on the browser VM, under Node, as a page's worker would:
// one slice at a time, sleeping until the next timer when nothing is runnable.
//
//     node dream/browser/run.mjs build-browser/dream-browser.mjs program.dream
//
// Standard output and error are the program's, line by line, and the exit
// status is its own -- 0 when `main!` returns, `os.exit!`'s code, 1 when it
// fails. A message sent over the graphics bridge is printed to stderr, since
// nothing here renders it. This is what says whether an ordinary program runs
// in a browser: `dream/browser/compat.sh` runs the examples through it.
import { readFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';

const [runtimePath, imagePath] = process.argv.slice(2);
if (!runtimePath || !imagePath) {
  console.error('usage: node run.mjs dream-browser.mjs program.dream');
  process.exit(2);
}
const { default: create } = await import(pathToFileURL(resolve(runtimePath)));
let failure;
const runtime = await create({
  onMessage: text => process.stderr.write(`bridge: ${text}\n`),
  onError: text => { failure ??= text; },
  print: text => process.stdout.write(text + '\n'),
  printErr: text => process.stderr.write(text + '\n'),
});
const bytes = await readFile(imagePath);
const ptr = runtime._malloc(bytes.length);
runtime.HEAPU8.set(bytes, ptr);
const loaded = runtime._dream_browser_load(ptr, bytes.length);
runtime._free(ptr);
if (!loaded) {
  console.error(`dream (browser): cannot load ${imagePath}: ${failure ?? 'invalid image'}`);
  process.exit(1);
}
const sleep = ms => new Promise(r => setTimeout(r, ms));
for (;;) {
  const status = runtime._dream_browser_tick();
  if (status < 0) {
    console.error(`dream (browser): ${failure ?? 'the program failed'}`);
    process.exit(1);
  }
  if (status === 0) {
    const code = runtime._dream_browser_exit_code();
    runtime._dream_browser_dispose();
    process.exit(code);
  }
  if (status === 2) {
    const delay = runtime._dream_browser_delay();
    // Waiting on nothing but the bridge: no page will ever answer.
    if (delay < 0) {
      console.error('dream (browser): the program is waiting for an event that cannot come');
      process.exit(1);
    }
    await sleep(delay);
  }
}
