# Task 0020: ACCELERATOR FLOAT32 Reductions After Exact-Zero Sign Refinement

## Status

Draft

Draft reason: Model 0028 is the sole Ready frontier and must complete its coordinated semantic,
Javadoc, documentation, proof, and independent Class C gates first. After that integration, a
planner must reverify this task's base and metadata and promote it explicitly; Metal 0017 remains a
Blocked record under the old contract.

## Change class

Class C — this would add profile-qualified reduction capability, native schema, scalar publication,
cache identity, prepared lifecycle, and public Engine execution across the native boundary. A full
fresh corrected pre-edit Apple M3 probe and independent review are mandatory.

## Goal

After Model 0028 is Complete, add canonical `FLOAT32` full, single-axis, ordered multi-axis
`SUM`/`MEAN`, and binding-resolved `SUM_TO_SHAPE` under `NumericalProfile.ACCELERATOR` only.
Preserve every declared term and admit only the Model-owned binary-tree/per-step-rounding DAZ/FTZ
set plus final exact-zero arithmetic-result sign freedom. Keep strict reductions false and every
existing profile-qualified operation unchanged.

## Exact capability and Shape domain

- Admit one canonical dense-contiguous non-view `FLOAT32` input of static positive rank `1..16`,
  checked geometry, one canonical output, and equal `requiresGrad`.
- `SUM` accepts only no attrs, `AxisReductionAttrs`, `MultiAxisReductionAttrs`, or
  `SumToShapeAttrs`; `MEAN` accepts only the first three. Masked and all other forms remain false.
- Full form reduces all axes to rank zero. Single-axis uses one normalized axis and exact
  `keepDimensions`. Multi-axis preserves ordered distinct normalized axes; empty axes mean a
  bit-preserving identity, not full reduction.
- `SUM_TO_SHAPE` right-aligns the bound target: leading axes reduce/disappear, target-one axes
  reduce/remain, and equal axes remain. Equal Shape copies exact bits; scalar target reduces all
  axes; every other pair rejects before native creation.
- Output rank is `0..16` with positive extents. Rank zero is Metal-local result/publication only,
  never a feed or ABS/binary operand; CPU↔Metal transfer remains positive rank `1..16`.
- Accelerator reductions may compose with admitted canonical ABS/binary nodes when every
  occurrence qualifies. Outputs are canonical. Strict remains exactly NEG/ABS/affine/`CONTIGUOUS`;
  no strict reduction reaches native creation.

## Controlling numerical contract

For each output coordinate, derive all and only the declared terms from kind, ordered axes,
`keepDimensions`, and bound Shapes. For an arithmetic cell, use a Model-permitted full binary tree
with per-step `FLOAT32` rounding and row-scoped DAZ/FTZ. A SUM/SUM-to-Shape root addition that is
exact zero may publish either zero sign. `MEAN` performs the mandatory final `FLOAT32` divide by the
declared positive count; only an exact-zero quotient may publish either sign.

Empty-axis point domains, equal-Shape `SUM_TO_SHAPE`, and one-term SUM/SUM-to-Shape cells perform no
addition and preserve the selected input bits. Do not insert a zero identity. Empty-domain rules are
unchanged. Never drop, duplicate, or invent a term, and do not grant per-step exact-zero sign choice,
tolerance, reduced precision, reciprocal substitution, algebraic rewrite, cross-node contraction,
or special-value repair.

## Mandatory full fresh corrected Apple M3 probe

Before any production edit, create a new disposable probe in `/tmp`, compile it with `xcrun clang`
against Foundation/Metal/MPSGraph, run it on the real M3, and remove source and binary. Do not reuse
Metal 0017's result as the gate.

The harness must retain every graph label/shape string for its full native use, with no dangling
autoreleased `NSString`, and wrap each compile/run cell in its own autorelease pool. First pass a
minimal rank-16 middle-axis (`axis=8`, `keepDimensions=false`) smoke cell. Then run the complete
matrix at optimization levels `0` and `1`, reduced-precision fast math disabled, three independently
created executables per cell, eight runs per executable, and two fresh contexts: at least 300
executables and 2,400 runs.

Report SUM and MEAN separately for full, first/middle/last single axis, ordered non-monotonic
multi-axis, and empty-axis forms, both `keepDimensions`, rank-zero and positive-rank outputs, Shapes
`[4]`, `[2,3,4]`, and bounded rank 16. Report SUM_TO_SHAPE scalar, leading-axis, target-one,
combined, and equal-Shape mappings. Include `[-0,-0]`, mixed zeros, one-term cells, historical
cancellation `[1.0e20f,1.0f,1.0f,-1.0e20f]`, tree-sensitive permutations, signed minimum/ordinary/
maximum subnormals, minimum normals/neighbors, FTZ-producing cancellation, ordinary/maximum finite
values, infinities, and signed quiet/signaling NaNs with varied payloads.

An independent raw-bit oracle must enumerate the Model 0028 allowed set: every permutation and full
binary tree for bounded cells, per-step rounding, independent DAZ operands, optional FTZ for each
finite subnormal arithmetic result, root-only exact-zero sign freedom, and MEAN's positive-count
final divide and exact-zero quotient freedom. Compare set membership, not one preferred tree. Verify
that identities and one-term SUM cells preserve bits, equal-Shape SUM_TO_SHAPE copies bits, and no
added identity or per-step zero-sign choice enters the oracle.

Use guarded direct targets; permute feeds/targets; verify Shape/count, inputs, canaries, selector
isolation, repetition, contexts, and exact probe totals. Any out-of-set value, term loss, identity
change, Shape error, ownership/lifetime fault, or control failure blocks 0020 without production,
schema, or capability edits.

## Native schema, identity, lifecycle, and cache

- Keep ABI v4, statuses, create signature, and exactly thirteen exports. Evolve fixed 160-byte node
  schema `6 -> 7`; retain wires `1..12`; append only `SUM=13` and `MEAN=14`.
- Schema 7 carries a typed full/single/multi discriminator, ordered normalized axes including empty,
  exact `keepDimensions`, and SUM-to-Shape target rank/dimensions. SUM_TO_SHAPE remains SUM.
- Java/native validation agrees on output Shape, reduced axes, target mapping, positive count,
  canonical states, and bytes. Rank zero is legal only for a locally produced one-element target;
  malformed/profile-incompatible forms and wires `15+` reject before resource creation.
- Advance workload-signature, exact-policy, candidate, compatibility, route-policy, and codec
  identities `7 -> 8`; include profile/schema, form/axes/flags/target Shape, term/count geometry,
  descriptors/states/topology, targets, ABI, and splats. Old/cross-profile/corrupt/foreign data
  rejects or misses safely.
- Use MPSGraph only. Analysis declares exact buffers/workspace; finalization creates one
  shape-specialized executable transactionally; cold binding validates buffers; hot execution
  makes one synchronous submission. Preserve leases, rollback, reuse, concurrency, close, cleanup,
  direct targets, and session/context isolation.
- Extend only Metal-local canonical materialization to download rank-zero FLOAT32 as four detached
  host bytes. Do not widen transfer or add shared Prepare/Runtime/Trace profile state.

## Non-goals

No strict or masked reduction; other reduction/scan/MATMUL/convolution/scalar family; custom kernel;
host fallback/rewrite; rank-zero transfer; dynamic, zero-extent, or view Shape; other type;
backward/training; packaging/discovery; or performance claim. Metal 0017 stays Blocked under the old
contract, 0018 stays Draft after this task, 0016 and historical 0006–0007/0009–0013 stay Blocked,
and Model 0026 stays Draft.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) — Model owns
  results; profile and route decisions remain cold.
- [Foundational modules — Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
  — after Model 0028, the sole all-term/tree/rounding/DAZ/FTZ/root-zero/count/identity rules.
- [Backend execution — Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)
  and [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend) —
  truthful capability, typed candidates, lowering, native state, and materialization.
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle),
  [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle), and
  [Numerical-profile lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#numerical-profile-lifecycle)
  — fixed resources/routes, direct hot execution, and profile-free Runtime/Trace.

If this needs another shared semantic change, per-step zero-sign freedom, a new export, host
fallback, reduced precision, wider transfer, or Runtime profile state, stop and report it.

## Dependencies and integration

- Depends on: Model 0028; Metal 0019; Model 0027; Config 0006; Engine 0018; not Metal 0016/0017
- Conflicts with: Metal 0018, any 0016 restart, and every Metal capability/preparation/native
  schema/candidate/codec/materialization/public Engine scope
- Parallel group: None
- Common base revision: N/A until Model 0028 is integrated
- Integration order: Serial; after Model 0028 and Metal 0019, before Metal 0018
- Integration validation: accelerator all-term Metal reduction checkpoint under Model 0028
- Shared-document integration owner: task implementer, finalized by independent Class C review

## Files and symbols

- `MetalCapabilityProvider`, `MetalNegPartitionPreparer`, `MetalNegPreparationPlan` — exact
  profile/form/Shape/state matrix, lowering, counts, and profile.
- `MetalMpsGraphProgram`, `MetalNativeApi`, `MetalDeviceContext`, native foundation — schema 7,
  wires 13/14, rank-zero target, validation/selectors, and ABI-v4 stability.
- Metal candidate/tuning/codec/decision types — version-eight deterministic identity and stale
  rejection; backend runtime/integration — local four-byte scalar materialization.
- Focused capability/schema/native/prepared/candidate tests, Metal conformance, and
  `EngineExplicitCompositionMetalIntegrationTest` — narrow reduction behavior and negatives.
- Affected Javadocs, Metal/native guides, architecture/API/user/capability/preparer status,
  glossary, and synchronized planning.

## Acceptance criteria

1. The full fresh corrected probe passes at least 300 executables/2,400 runs and every operation,
   form, axis, keep-dimensions, Shape, tree-sensitive, zero-sign, identity, and control gate before
   edits; its artifacts are removed.
2. Strict reductions stay false. Accelerator adds exactly the declared forms; malformed or wrong
   profile/type/layout/rank/Shape forms reject before native creation; prior capability is unchanged.
3. Every arithmetic result belongs to the Model 0028 oracle; identity/one-term forms preserve bits,
   MEAN divides by positive count, and no tolerance, term loss, added identity, per-step sign choice,
   host rewrite, value fallback, or reduced precision exists.
4. Schema 7/Java/native agree; ABI v4 keeps thirteen exports; version-eight identity rejects old,
   cross-profile, malformed, corrupt, stale, unknown, and changed-Shape data.
5. Lifecycle proof covers every form/chain, direct targets, scalar/positive-rank publication, reuse,
   concurrency, contexts/sessions, rollback/close, and local four-byte materialization. Transfer is
   unchanged.
6. CPU-free Engine executes and materializes oracle-checked every-form results. Strict Metal-only
   reduction fails ownership selection; no not-throw-only claim or backward/training claim exists.
7. Native/export, Metal/Javadoc, conformance, Engine, architecture, full-build, Markdown/status,
   probe-removal, and diff checks pass; independent Class C review has no unresolved finding.

## Validation

Worker validation:

```bash
xcrun clang -fobjc-arc -framework Foundation -framework Metal -framework MetalPerformanceShadersGraph /tmp/synaptik-metal-0020-reduction-probe.m -o /tmp/synaptik-metal-0020-reduction-probe
/tmp/synaptik-metal-0020-reduction-probe
rm -f /tmp/synaptik-metal-0020-reduction-probe.m /tmp/synaptik-metal-0020-reduction-probe
./native/metal-macos-arm64/build.sh
nm -gU native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" ./gradlew :backends:metal:test :backends:metal:javadoc
./gradlew :testing:backend-conformance:test --tests '*Metal*'
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" ./gradlew :testing:integration-tests:test --tests '*EngineExplicitCompositionMetalIntegrationTest*'
./gradlew :testing:architecture-tests:test
./gradlew build
python3 /tmp/validate_synaptik_markdown.py
git diff --check
```

Record probe totals/removal, exports, schema/identity, test counts/skips, scalar bits, strict/profile
negatives, Markdown/status/frontier evidence, and exact changed scope. The serial checkpoint owns one
full build; review reruns executable checks only for changed code or concretely stale evidence.

## Documentation and review impact

Update affected Javadocs, Metal/native guides, scoped capability, API/user and preparer status,
targeted glossary, and synchronized planning. Independent Class C review inspects the fresh probe
harness ownership, totals, root-only oracle, all terms/counts/Shapes, strict false matrix,
identity/one-term copies, schema/cache identity, local-only rank zero, lifecycle/concurrency/cleanup,
real Engine proof, Runtime/Trace absence, changed scope, and artifact removal.

## Result

Empty until execution.
