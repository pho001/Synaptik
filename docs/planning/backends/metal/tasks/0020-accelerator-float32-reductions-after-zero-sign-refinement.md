# Task 0020: [ACCELERATOR](../../../../glossary.md#term-accelerator) [FLOAT32](../../../../glossary.md#term-float32) Reductions After Exact-Zero Sign Refinement

## Status

Complete

Completed at implementation `9ddb75f6` plus documentation remediation `5b77c742`. The fresh
corrected Apple M3 [oracle](../../../../glossary.md#term-test-oracle), worker [validation](../../../../glossary.md#term-validation), documentation remediation, and independent Class C
final review are complete; the review returned `APPROVE` with zero findings. [ABI](../../../../glossary.md#term-application-binary-interface-abi) v4 retains exactly
thirteen exports, the fixed 160-byte [node](../../../../glossary.md#node) [schema](../../../../glossary.md#term-schema) is version seven, workload/candidate/
compatibility/route/[codec](../../../../glossary.md#term-codec) [identities](../../../../glossary.md#term-identity-meanings) are version eight, strict reduction capability remains false,
and [rank](../../../../glossary.md#term-rank) zero remains local-only. Historical [Metal](../../../../glossary.md#term-metal) 0017 stays Blocked.

## Change class

Class C — this would add profile-qualified reduction capability, native [schema](../../../../glossary.md#term-schema), [scalar](../../../../glossary.md#term-scalar) [publication](../../../../glossary.md#term-publication),
[cache](../../../../glossary.md#term-cache) [identity](../../../../glossary.md#term-identity-meanings), prepared [lifecycle](../../../../glossary.md#lifecycle), and public [Engine](../../../../glossary.md#term-engine) execution across the native boundary. A full
fresh corrected pre-edit Apple M3 probe and independent review are mandatory.

## Goal

Add [canonical](../../../../glossary.md#term-canonical-form) [`FLOAT32`](../../../../glossary.md#term-float32) full, single-[axis](../../../../glossary.md#term-axis), ordered multi-axis
`SUM`/[`MEAN`](../../../../glossary.md#term-mean), and [binding](../../../../glossary.md#term-binding)-resolved `SUM_TO_SHAPE` under `NumericalProfile.ACCELERATOR` only.
Preserve every declared term and admit only the [Model](../../../../glossary.md#term-model)-owned [binary](../../../../glossary.md#term-binary-descriptor-and-metadata)-tree/per-step-[rounding](../../../../glossary.md#term-rounding) [DAZ](../../../../glossary.md#term-daz)/[FTZ](../../../../glossary.md#term-ftz)
set plus final exact-zero arithmetic-result sign freedom. Keep strict reductions false and every
existing profile-qualified [operation](../../../../glossary.md#operation) unchanged.

## Exact capability and [Shape](../../../../glossary.md#shape) domain

- Admit one [canonical](../../../../glossary.md#term-canonical-form) dense-[contiguous](../../../../glossary.md#term-contiguous) non-[view](../../../../glossary.md#term-view-meanings) [`FLOAT32`](../../../../glossary.md#term-float32) input of static positive [rank](../../../../glossary.md#term-rank) `1..16`,
  checked geometry, one canonical output, and equal `requiresGrad`.
- `SUM` accepts only no attrs, `AxisReductionAttrs`, `MultiAxisReductionAttrs`, or
  `SumToShapeAttrs`; [`MEAN`](../../../../glossary.md#term-mean) accepts only the first three. Masked and all other forms remain false.
- Full form reduces all [axes](../../../../glossary.md#term-axis) to rank zero. Single-axis uses one [normalized axis](../../../../glossary.md#normalized-axis) and exact
  `keepDimensions`. Multi-axis preserves ordered distinct normalized axes; empty axes mean a
  bit-preserving identity, not full reduction.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `SUM_TO_SHAPE` right-aligns the bound target: leading axes reduce/disappear, target-one axes
  reduce/remain, and equal axes remain. Equal Shape copies exact bits; scalar target reduces all
  axes; every other pair rejects before native creation.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Output rank is `0..16` with positive extents. Rank zero is Metal-local result/publication only,
  never a feed or ABS/binary operand; CPU↔Metal transfer remains positive rank <span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [publication glossary](../../../../glossary.md#term-publication); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [CPU glossary](../../../../glossary.md#term-cpu))</span>`1..16`.
- Accelerator reductions may compose with admitted canonical ABS/binary nodes when every
  occurrence qualifies. Outputs are canonical. Strict remains exactly NEG/ABS/affine/<span data-glossary-navigation="true"> ([Accelerator glossary](../../../../glossary.md#term-accelerator); [nodes glossary](../../../../glossary.md#node); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>`CONTIGUOUS`;
  no strict reduction reaches native creation.

## Controlling numerical contract

For each output [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), derive all and only the declared terms from kind, ordered [axes](../../../../glossary.md#term-axis),
`keepDimensions`, and bound [Shapes](../../../../glossary.md#shape). For an arithmetic cell, use a [Model](../../../../glossary.md#term-model)-permitted full [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) tree
with per-step [`FLOAT32`](../../../../glossary.md#term-float32) [rounding](../../../../glossary.md#term-rounding) and row-scoped [DAZ](../../../../glossary.md#term-daz)/[FTZ](../../../../glossary.md#term-ftz). A SUM/[SUM-to-Shape](../../../../glossary.md#sum-to-shape) root addition that is
exact zero may publish either zero sign. [`MEAN`](../../../../glossary.md#term-mean) performs the mandatory final `FLOAT32` divide by the
declared positive count; only an exact-zero quotient may publish either sign.

Empty-axis point domains, equal-Shape `SUM_TO_SHAPE`, and one-term SUM/SUM-to-Shape cells perform no
addition and preserve the selected input bits. Do not insert a zero [identity](../../../../glossary.md#term-identity-meanings). Empty-domain rules are
unchanged. Never drop, duplicate, or invent a term, and do not grant per-step exact-zero sign choice,
[tolerance](../../../../glossary.md#term-tolerance), reduced [precision](../../../../glossary.md#term-precision), reciprocal substitution, algebraic rewrite, cross-[node](../../../../glossary.md#node) [contraction](../../../../glossary.md#contraction-dimension),
or special-value repair.

## Mandatory full fresh corrected Apple M3 probe

Before any production edit, create a new disposable probe in `/tmp`, [compile](../../../../glossary.md#compile) it with `xcrun clang`
against Foundation/[Metal](../../../../glossary.md#term-metal)/[MPSGraph](../../../../glossary.md#term-mpsgraph), [run](../../../../glossary.md#term-run) it on the real M3, and remove source and [binary](../../../../glossary.md#term-binary-descriptor-and-metadata). Do not reuse
Metal 0017's result as the gate.

The harness must retain every [graph](../../../../glossary.md#graph) label/[shape](../../../../glossary.md#shape) string for its full native use, with no dangling
autoreleased `NSString`, and wrap each compile/run cell in its own [autorelease pool](../../../../glossary.md#native-packaging-and-platform-vocabulary). First pass a
minimal [rank](../../../../glossary.md#term-rank)-16 middle-[axis](../../../../glossary.md#term-axis) (`axis=8`, `keepDimensions=false`) [smoke](../../../../glossary.md#term-smoke-test) cell. Then run the complete
[matrix](../../../../glossary.md#term-matrix-meanings) at optimization levels `0` and `1`, reduced-[precision](../../../../glossary.md#term-precision) fast math disabled, three independently
created executables per cell, eight runs per executable, and two fresh contexts: at least 300
executables and 2,400 runs.

Report SUM and [MEAN](../../../../glossary.md#term-mean) separately for full, first/middle/last single axis, ordered non-monotonic
multi-axis, and empty-axis forms, both `keepDimensions`, [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) and positive-rank outputs, Shapes
`[4]`, `[2,3,4]`, and bounded rank 16. Report SUM_TO_SHAPE [scalar](../../../../glossary.md#term-scalar), leading-axis, [target](../../../../glossary.md#term-target)-one,
combined, and equal-Shape mappings. Include `[-0,-0]`, mixed zeros, one-term cells, historical
[cancellation](../../../../glossary.md#term-cancellation) `[1.0e20f,1.0f,1.0f,-1.0e20f]`, tree-sensitive permutations, signed minimum/ordinary/
maximum [subnormals](../../../../glossary.md#term-subnormal), minimum normals/neighbors, [FTZ](../../../../glossary.md#term-ftz)-producing cancellation, ordinary/maximum [finite](../../../../glossary.md#term-finite)
values, [infinities](../../../../glossary.md#nan-infinity-and-signed-zero), and signed quiet/signaling NaNs with varied payloads.

An independent raw-bit [oracle](../../../../glossary.md#term-test-oracle) must enumerate the [Model](../../../../glossary.md#term-model) 0028 allowed set: every [permutation](../../../../glossary.md#permutation) and full
binary tree for bounded cells, per-step [rounding](../../../../glossary.md#term-rounding), independent [DAZ](../../../../glossary.md#term-daz) operands, optional FTZ for each
finite subnormal arithmetic result, root-only exact-zero sign freedom, and MEAN's positive-count
final divide and exact-zero quotient freedom. Compare set membership, not one preferred tree. Verify
that [identities](../../../../glossary.md#term-identity-meanings) and one-term SUM cells preserve bits, equal-Shape SUM_TO_SHAPE copies bits, and no
added [identity](../../../../glossary.md#term-identity-meanings) or per-step zero-sign choice enters the oracle.

Use guarded direct targets; permute feeds/targets; verify Shape/count, inputs, canaries, selector
isolation, repetition, contexts, and exact probe totals. Any out-of-set value, term [loss](../../../../glossary.md#loss), identity
change, Shape error, [ownership](../../../../glossary.md#term-ownership-meanings)/[lifetime](../../../../glossary.md#term-lifetime) fault, or control failure blocks 0020 without production,
[schema](../../../../glossary.md#term-schema), or capability edits.

## Native [schema](../../../../glossary.md#term-schema), [identity](../../../../glossary.md#term-identity-meanings), [lifecycle](../../../../glossary.md#lifecycle), and [cache](../../../../glossary.md#term-cache)

- Keep ABI v4, statuses, create signature, and exactly thirteen exports. Evolve fixed 160-byte node
  schema <span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [signature glossary](../../../../glossary.md#term-signature); [node glossary](../../../../glossary.md#node))</span>`6 -> 7`; retain wires `1..12`; append only `SUM=13` and `MEAN=14`.
- Schema 7 carries a typed full/single/multi discriminator, ordered normalized axes including empty,
  exact <span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis))</span>`keepDimensions`, and [SUM-to-Shape](../../../../glossary.md#sum-to-shape) [target](../../../../glossary.md#term-target) [rank](../../../../glossary.md#term-rank)/[dimensions](../../../../glossary.md#dimension). SUM_TO_SHAPE remains SUM.
- Java/native validation agrees on output Shape, reduced axes, target mapping, positive count,
  canonical states, and bytes. Rank zero is legal only for a locally produced one-element target;
  malformed/profile-incompatible forms and wires <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Shape glossary](../../../../glossary.md#shape); [canonical glossary](../../../../glossary.md#term-canonical-form); [element glossary](../../../../glossary.md#term-element))</span>`15+` reject before resource creation.
- Advance workload-signature, exact-policy, candidate, compatibility, route-policy, and codec
  identities <span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec))</span>`7 -> 8`; include profile/schema, form/axes/flags/target Shape, term/count geometry,
  descriptors/states/topology, targets, ABI, and splats. Old/cross-profile/corrupt/foreign data
  rejects or misses safely.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [topology glossary](../../../../glossary.md#term-topology))</span>
- Use MPSGraph only. Analysis declares exact buffers/workspace; finalization creates one
  shape-specialized executable transactionally; cold binding validates buffers; hot execution
  makes one synchronous submission. Preserve leases, rollback, reuse, concurrency, close, cleanup,
  direct targets, and session/context isolation.<span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [buffers glossary](../../../../glossary.md#term-buffer); [binding glossary](../../../../glossary.md#term-binding); [synchronous glossary](../../../../glossary.md#term-synchronous); [leases glossary](../../../../glossary.md#term-lease); [rollback glossary](../../../../glossary.md#term-rollback); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- Extend only Metal-local canonical materialization to download rank-zero FLOAT32 as four detached
  host bytes. Do not widen transfer or add shared Prepare/Runtime/Trace profile state.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [materialization glossary](../../../../glossary.md#materialization); [rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [FLOAT32 glossary](../../../../glossary.md#term-float32); [detached glossary](../../../../glossary.md#detached-model-results); [host glossary](../../../../glossary.md#term-host); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Trace glossary](../../../../glossary.md#trace))</span>

## Non-goals

No strict or [masked reduction](../../../../glossary.md#masked-reduction); other reduction/[scan](../../../../glossary.md#term-cumulative-scan)/[MATMUL](../../../../glossary.md#matrix-multiplication--matmul)/[convolution](../../../../glossary.md#term-channels-first-convolution-layers)/[scalar](../../../../glossary.md#term-scalar) family; custom [kernel](../../../../glossary.md#kernel);
[host](../../../../glossary.md#term-host) fallback/rewrite; [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) transfer; dynamic, zero-[extent](../../../../glossary.md#term-extent), or [view](../../../../glossary.md#term-view-meanings) [Shape](../../../../glossary.md#shape); other type;
backward/training; packaging/discovery; or performance claim. [Metal](../../../../glossary.md#term-metal) 0017 stays Blocked under the old
contract. This task did not itself authorize 0018; post-completion [planning](../../../../glossary.md#term-planning) may promote that serial
successor. Metal 0016 and historical 0006–0007/0009–0013 stay Blocked, and [Model](../../../../glossary.md#term-model) 0026 stays Draft.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) — Model owns
  results; profile and route decisions remain cold.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Foundational modules — Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module))</span>
  — the sole current all-term/tree/rounding/DAZ/FTZ/root-zero/count/identity rules.<span data-glossary-navigation="true"> ([rounding glossary](../../../../glossary.md#term-rounding); [DAZ glossary](../../../../glossary.md#term-daz); [FTZ glossary](../../../../glossary.md#term-ftz); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- [Backend execution — Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [Concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
  and [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal))</span> —
  truthful capability, typed candidates, lowering, native state, and materialization.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [materialization glossary](../../../../glossary.md#materialization))</span>
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>,
  [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span>, and
  [Numerical-profile lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#numerical-profile-lifecycle)
  — fixed resources/routes, direct hot execution, and profile-free Runtime/Trace.<span data-glossary-navigation="true"> ([Trace glossary](../../../../glossary.md#trace))</span>

If this needs another shared semantic change, per-step zero-sign freedom, a new export, [host](../../../../glossary.md#term-host)
fallback, reduced [precision](../../../../glossary.md#term-precision), wider transfer, or [Runtime](../../../../glossary.md#term-runtime) profile state, stop and report it.

## Dependencies and integration

- Depends on: [Model](../../../../glossary.md#term-model) 0028; [Metal](../../../../glossary.md#term-metal) 0019; Model 0027; Config 0006; [Engine](../../../../glossary.md#term-engine) 0018; not Metal 0016/0017
- Conflicts with: Metal 0018, any 0016 restart, and every Metal capability/preparation/native
  schema/candidate/codec/materialization/public Engine scope<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [codec glossary](../../../../glossary.md#term-codec); [materialization glossary](../../../../glossary.md#materialization); [scope glossary](../../../../glossary.md#term-scope))</span>
- Parallel group: None
- Common base revision: `fc003ab83d4d8c03c557aea867f5f4f3f388fa5d` plus this [planning](../../../../glossary.md#term-planning)-only authorization
- Integration order: Serial; after Model 0028 and Metal 0019, before Metal 0018
- Integration [validation](../../../../glossary.md#term-validation): [accelerator](../../../../glossary.md#term-accelerator) all-term Metal reduction [checkpoint](../../../../glossary.md#term-checkpoint) under Model 0028
- Shared-document integration [owner](../../../../glossary.md#term-owner): task implementer, finalized by independent Class C review

## Files and symbols

- `MetalCapabilityProvider`, `MetalNegPartitionPreparer`, `MetalNegPreparationPlan` — exact
  profile/form/Shape/state matrix, lowering, counts, and profile.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [lowering glossary](../../../../glossary.md#lowering))</span>
- `MetalMpsGraphProgram`, `MetalNativeApi`, `MetalDeviceContext`, native foundation — schema 7,
  wires 13/14, rank-zero target, validation/selectors, and ABI-v4 stability.<span data-glossary-navigation="true"> ([rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [target glossary](../../../../glossary.md#term-target); [validation glossary](../../../../glossary.md#term-validation); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
- Metal candidate/tuning/codec/decision types — version-eight deterministic identity and stale
  rejection; backend runtime/integration — local four-byte scalar materialization.<span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec); [identity glossary](../../../../glossary.md#term-identity-meanings); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime); [scalar glossary](../../../../glossary.md#term-scalar); [materialization glossary](../../../../glossary.md#materialization))</span>
- Focused capability/[schema](../../../../glossary.md#term-schema)/native/prepared/candidate tests, [Metal](../../../../glossary.md#term-metal) [conformance](../../../../glossary.md#term-conformance), and
  `EngineExplicitCompositionMetalIntegrationTest` — narrow reduction behavior and negatives.
- Affected Javadocs, Metal/native guides, architecture/API/user/capability/preparer status,
  glossary, and synchronized planning.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>

## Acceptance criteria

1. The full fresh corrected probe passes at least 300 executables/2,400 runs and every operation,
   form, axis, keep-dimensions, Shape, tree-sensitive, zero-sign, identity, and control gate before
   edits; its artifacts are removed.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [axis glossary](../../../../glossary.md#term-axis); [dimensions glossary](../../../../glossary.md#dimension); [Shape glossary](../../../../glossary.md#shape); [identity glossary](../../../../glossary.md#term-identity-meanings); [artifacts glossary](../../../../glossary.md#term-artifact))</span>
2. Strict reductions stay false. Accelerator adds exactly the declared forms; malformed or wrong
   profile/type/layout/rank/Shape forms reject before native creation; prior capability is unchanged.<span data-glossary-navigation="true"> ([Accelerator glossary](../../../../glossary.md#term-accelerator); [layout glossary](../../../../glossary.md#layout); [rank glossary](../../../../glossary.md#term-rank))</span>
3. Every arithmetic result belongs to the Model 0028 oracle; identity/one-term forms preserve bits,
   MEAN divides by positive count, and no tolerance, term loss, added identity, per-step sign choice,
   host rewrite, value fallback, or reduced precision exists.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [oracle glossary](../../../../glossary.md#term-test-oracle); [MEAN glossary](../../../../glossary.md#term-mean); [tolerance glossary](../../../../glossary.md#term-tolerance); [loss glossary](../../../../glossary.md#loss); [host glossary](../../../../glossary.md#term-host); [precision glossary](../../../../glossary.md#term-precision))</span>
4. Schema 7/Java/native agree; ABI v4 keeps thirteen exports; version-eight identity rejects old,
   cross-profile, malformed, corrupt, stale, unknown, and changed-Shape data.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
5. Lifecycle proof covers every form/chain, direct targets, scalar/positive-rank publication, reuse,
   concurrency, contexts/sessions, rollback/close, and local four-byte materialization. Transfer is
   unchanged.<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle); [targets glossary](../../../../glossary.md#term-target); [scalar glossary](../../../../glossary.md#term-scalar); [publication glossary](../../../../glossary.md#term-publication); [concurrency glossary](../../../../glossary.md#term-concurrency); [rollback glossary](../../../../glossary.md#term-rollback); [materialization glossary](../../../../glossary.md#materialization))</span>
6. CPU-free Engine executes and materializes oracle-checked every-form results. Strict Metal-only
   reduction fails ownership selection; no not-throw-only claim or backward/training claim exists.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine); [Metal glossary](../../../../glossary.md#term-metal); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
7. Native/export, Metal/Javadoc, conformance, Engine, architecture, full-build, Markdown/status,
   probe-removal, and diff checks pass; independent Class C review has no unresolved finding.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## [Validation](../../../../glossary.md#term-validation)

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

Record probe totals/removal, exports, [schema](../../../../glossary.md#term-schema)/[identity](../../../../glossary.md#term-identity-meanings), test counts/skips, [scalar](../../../../glossary.md#term-scalar) bits, strict/profile
negatives, Markdown/status/frontier evidence, and exact changed [scope](../../../../glossary.md#term-scope). The serial [checkpoint](../../../../glossary.md#term-checkpoint) owns one
full build; review reruns executable checks only for changed code or concretely stale evidence.

## Documentation and review impact

Update affected Javadocs, [Metal](../../../../glossary.md#term-metal)/native guides, scoped capability, [API](../../../../glossary.md#term-application-programming-interface-api)/user and preparer status,
targeted glossary, and synchronized [planning](../../../../glossary.md#term-planning). Independent Class C review inspects the fresh probe
harness [ownership](../../../../glossary.md#term-ownership-meanings), totals, root-only [oracle](../../../../glossary.md#term-test-oracle), all terms/counts/[Shapes](../../../../glossary.md#shape), strict false [matrix](../../../../glossary.md#term-matrix-meanings),
[identity](../../../../glossary.md#term-identity-meanings)/one-term copies, [schema](../../../../glossary.md#term-schema)/[cache](../../../../glossary.md#term-cache) identity, local-only [rank](../../../../glossary.md#term-rank) zero, [lifecycle](../../../../glossary.md#lifecycle)/[concurrency](../../../../glossary.md#term-concurrency)/cleanup,
real [Engine](../../../../glossary.md#term-engine) proof, [Runtime](../../../../glossary.md#term-runtime)/[Trace](../../../../glossary.md#trace) absence, changed [scope](../../../../glossary.md#term-scope), and [artifact](../../../../glossary.md#term-artifact) removal.

## Result

Implemented from clean base `89ca7961` at `9ddb75f6`, with documentation remediation at
`5b77c742`; the task is Complete.

- The corrected fresh pre-edit M3 gate passed 50 matrix cells at each of optimization levels zero
  and one: 22 SUM, 21 MEAN, and 7 SUM-to-Shape cells across two fresh contexts, 300 executables,
  and 2,400 runs, plus the independent rank-16/axis-8 smoke executable and run. The oracle checked
  every declared term, permutation, full binary tree, per-step FLOAT32 rounding with DAZ/FTZ,
  root-only final exact-zero sign freedom, mandatory positive-count MEAN quotient, and exact copy/
  identity bits. The retained-name/Shape, autorelease-pool, buffer-guard, order, repetition,
  canary, and target-isolation controls passed; all <span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [MEAN glossary](../../../../glossary.md#term-mean); [SUM-to-Shape glossary](../../../../glossary.md#sum-to-shape); [rank glossary](../../../../glossary.md#term-rank); [axis glossary](../../../../glossary.md#term-axis); [smoke glossary](../../../../glossary.md#term-smoke-test); [run glossary](../../../../glossary.md#term-run); [oracle glossary](../../../../glossary.md#term-test-oracle); [permutation glossary](../../../../glossary.md#permutation); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [FLOAT32 glossary](../../../../glossary.md#term-float32); [rounding glossary](../../../../glossary.md#term-rounding); [DAZ glossary](../../../../glossary.md#term-daz); [FTZ glossary](../../../../glossary.md#term-ftz); [identity glossary](../../../../glossary.md#term-identity-meanings); [Shape glossary](../../../../glossary.md#shape); [buffer glossary](../../../../glossary.md#term-buffer); [guard glossary](../../../../glossary.md#term-guard); [canary glossary](../../../../glossary.md#term-canary); [target glossary](../../../../glossary.md#term-target))</span>`/tmp` probe source and binary artifacts were
  removed before implementation.<span data-glossary-navigation="true"> ([artifacts glossary](../../../../glossary.md#term-artifact))</span>
- Metal now advertises only accelerator canonical FLOAT32 full, normalized single-axis, ordered
  multi-axis (including empty identity) SUM/MEAN and binding-resolved SUM-to-Shape. Strict
  reductions, masked reductions, unsupported kinds/types/layouts/Shapes, affine chaining, and
  rank-zero feeds or transfers remain rejected.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [accelerator glossary](../../../../glossary.md#term-accelerator); [canonical glossary](../../../../glossary.md#term-canonical-form); [binding glossary](../../../../glossary.md#term-binding); [layouts glossary](../../../../glossary.md#layout); [rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- ABI v4 and all thirteen exports are unchanged. Node schema seven retains fixed 160-byte records,
  adds SUM/MEAN wires 13/14 and a typed reduction form, while version-eight workload, candidate,
  compatibility, route, and codec identity prevents stale reuse. Scalar reduction publication is
  a local four-byte canonical materialization only.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [Node glossary](../../../../glossary.md#node); [schema glossary](../../../../glossary.md#term-schema); [codec glossary](../../../../glossary.md#term-codec); [Scalar glossary](../../../../glossary.md#term-scalar); [publication glossary](../../../../glossary.md#term-publication); [materialization glossary](../../../../glossary.md#materialization))</span>
- Native, Java preflight, prepared-resource reuse, input preservation, direct-target permutations,
  concurrent sessions, close rejection, strict/profile negatives, one-term raw-bit copies, and a
  CPU-free public Engine graph were exercised. Runtime and Trace sources were not changed.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine); [graph glossary](../../../../glossary.md#graph); [Runtime glossary](../../../../glossary.md#term-runtime); [Trace glossary](../../../../glossary.md#trace))</span>
- Validation passed the native build and exact 13-export inspection; 93 Metal backend tests with
  zero skips plus Javadocs; 4 Metal conformance tests with zero skips; all 9 explicit-composition
  Metal integration tests with zero skips against the real dylib; 9 architecture tests with zero
  skips; and the full 87-actionable-task repository build. The Markdown validator passed 18
  changed files; <span data-glossary-navigation="true"> ([Validation glossary](../../../../glossary.md#term-validation); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [dylib glossary](../../../../glossary.md#term-dylib))</span>`tensor-api.md` has the same 172 duplicate-heading diagnostics as the base and no
  new Markdown structural diagnostic. Status/frontier, probe-removal, changed-path, and diff checks
  passed.
- The independent Class C final review examined the committed implementation and remediated
  documentation against the fresh oracle, all-term/count and exact-zero rules,
  strict-false matrix, schema/cache identity, local scalar publication, lifecycle/cleanup, real
  Engine proof, Runtime/Trace absence, and probe removal. It returned <span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>`APPROVE` with zero findings.
  No executable source changed after the recorded worker validation.
- Documentation remediation synchronized affected Javadocs and explanatory/current-status
  documents without changing Runtime or Trace. Historical 0017 remains Blocked under its old
  contract; Model 0026 remains Draft.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>

Status: Complete
