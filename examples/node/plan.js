// Will this card hold this roster? Answered before anything loads.
//
//   node examples/node/plan.js
//
// Requires the native addon. From a checkout:
//   cd crates/hearth-node && npm install && npm run build
// Or just: npm install @interchained/hearth

const { plan } = require('../../crates/hearth-node');

const GIB = 1024 ** 3;

// The roster that started hearth: five models an operator wanted resident on
// one rented RTX A6000 (48 GiB). Nothing in any runtime refuses this — it
// loads, evicts, loads, evicts, and presents to everyone as "the models got
// slow." Here it is arithmetic, up front.
const roster = [
  { model: 'muse-local:latest',  weightsBytes: 20 * GIB, kvBytes: GIB },
  { model: 'deepseek-r1:32b',    weightsBytes: 20 * GIB, kvBytes: GIB },
  { model: 'gemma4:26b',         weightsBytes: 16 * GIB, kvBytes: GIB },
  { model: 'qwen3.6:27b',        weightsBytes: 17 * GIB, kvBytes: GIB },
  { model: 'gemma4-extract:31b', weightsBytes: 19 * GIB, kvBytes: GIB },
];

const p = plan(48 * GIB, 8, roster);

console.log(p.explain);
console.log();

// The count you declared, echoed back. If you sent five and this says five,
// the plan is about your roster and not a silently emptied one — the 0.1.0
// binding ate rosters whole and reported `fits: true` over nothing.
console.log(`declared: ${p.declared}  admitted: ${p.admitted.length}  rejected: ${p.rejected.length}`);
console.log();

for (const m of p.admitted) {
  console.log(`  ADMITTED  ${m}`);
}
for (const r of p.rejected) {
  console.log(
    `  REFUSED   ${r.model} — needs ${(r.neededBytes / GIB).toFixed(1)} GiB, ` +
    `${(r.freeBytes / GIB).toFixed(1)} GiB free, short by ${(r.shortBytes / GIB).toFixed(1)} GiB`
  );
}

console.log();
console.log(`headroom: ${(p.headroomBytes / GIB).toFixed(1)} GiB of ${(p.usableBytes / GIB).toFixed(1)} GiB usable`);

// Declaration order is priority order: first fit, never best fit. Reordering
// to squeeze one more model in would silently demote whatever the operator
// listed first, and on a serving box first means most important.
if (!p.fits) {
  console.log('\nThis roster does not fit. That is the answer — nothing is evicted to make room.');
}
