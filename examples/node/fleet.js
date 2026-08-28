// A live fleet, and the one boolean a router must not get wrong.
//
//   node examples/node/fleet.js
//
// This is the night hearth was built for, replayed: a model warms up, serves
// for an hour, and then the host takes the card away. Watch what the router is
// told at each step — and in particular, who gets blamed at the end.

const { HearthFleet } = require('../../crates/hearth-node');

const GIB = 1024 ** 3;
const T0 = 1_756_000_000_000; // a fixed instant, so the output is stable
const SEC = 1_000;

const fleet = new HearthFleet(48 * GIB, 8, [
  { model: 'muse-local:latest', weightsBytes: 20 * GIB, kvBytes: GIB },
  // Declared but too big for what is left. Recorded, not dropped: "it used to
  // hold three models" is the first thing anyone says when a box gets slower.
  { model: 'gemma4:26b', weightsBytes: 40 * GIB, kvBytes: GIB },
]);

function show(label, model, now) {
  const r = fleet.route(model, now);
  const flags = [
    r.ready ? 'READY' : null,
    r.tryElsewhere ? 'try-elsewhere' : null,
    r.operatorFault ? 'OPERATOR-FAULT' : null,
  ].filter(Boolean).join(' ');
  console.log(`${label.padEnd(24)} ${model.padEnd(20)} ${flags || '—'}`);
  console.log(`${''.padEnd(24)} ${JSON.stringify(r)}`);
  console.log();
}

console.log('=== the night, replayed ===\n');

// 1. Nothing observed yet. An honest "I do not know" beats a confident wrong
//    answer in either direction.
show('never probed', 'muse-local:latest', T0);

// 2. Weights are materializing. NOT ready — and critically, not a fault.
fleet.observe('muse-local:latest', 'load_started', {}, T0);
show('loading', 'muse-local:latest', T0 + 20 * SEC);

// 3. Loaded, answering, accounted for.
fleet.setEndpoint('muse-local:latest', '127.0.0.1:8090');
fleet.observe('muse-local:latest', 'probe_ok', { vramBytes: 21 * GIB }, T0 + 40 * SEC);
show('resident', 'muse-local:latest', T0 + 40 * SEC);

// 4. An hour later the probe fails — and the card is GONE. This is the whole
//    reason hearth exists. `gpuPresent: false` is the difference between "this
//    operator over-committed their card" and "their provider reclaimed it".
fleet.observe('muse-local:latest', 'probe_failed', {
  gpuPresent: false,
  detail: 'no CUDA device',
}, T0 + 3600 * SEC);
show('GPU detached', 'muse-local:latest', T0 + 3600 * SEC);

// The same failure with the card still present is a DIFFERENT diagnosis, and
// this one IS the operator's to answer for.
const other = new HearthFleet(48 * GIB, 8, [
  { model: 'muse-local:latest', weightsBytes: 20 * GIB, kvBytes: GIB },
]);
other.observe('muse-local:latest', 'load_started', {}, T0);
other.observe('muse-local:latest', 'probe_ok', { vramBytes: 21 * GIB }, T0 + SEC);
other.observe('muse-local:latest', 'probe_failed', {
  gpuPresent: true,              // the card is there; the runtime dropped the model
  detail: 'model not loaded',
}, T0 + 100 * SEC);
const evicted = other.route('muse-local:latest', T0 + 100 * SEC);
console.log('same probe failure, card still PRESENT:');
console.log(`${''.padEnd(24)} ${JSON.stringify(evicted)}`);
console.log(`${''.padEnd(24)} operatorFault: ${evicted.operatorFault}  <- over-committing IS theirs`);
console.log();

// 5. The model that never fit. Permanent until the hardware or the
//    declaration changes, so a router should stop asking.
show('never fit', 'gemma4:26b', T0 + 3600 * SEC);

// 6. Not declared here at all.
show('not declared', 'llama3:70b', T0);

console.log('=== what should be brought up next ===');
// Never by evicting. If it does not fit, the honest answer is that it does not.
console.log(`nextToLoad: ${fleet.nextToLoad()}`);
console.log();
console.log(fleet.report(T0 + 3600 * SEC));
