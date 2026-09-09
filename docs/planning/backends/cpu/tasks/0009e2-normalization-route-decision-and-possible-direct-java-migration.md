# Task 0009E2: Normalization Route Decision and Possible Direct-Java Migration

## Status

Ready

## Goal

Decide whether the complete current CPU trailing Layer/RMS-normalization route remains bounded
generated execution or becomes finite direct typed Java. Select only the route with demonstrably
lower total implementation-and-verification cost; this is not a benchmark or performance task.

## Scope

Decide exactly one first-class, fully static `LAYER_NORM` or `RMS_NORM` node admitted today:
Layer no-affine `[input]`, affine Layer `[input, scale, bias]`, RMS `[input]`, or scaled RMS
`[input, scale]`, all with positive trailing `normalizedShape`. Preserve the current
`BFLOAT16`/`FLOAT32`/`FLOAT64` promotion/result/epsilon boundary, gradient eligibility,
positive-rank static Shape, matching affine/scale Shape, resolved non-negative layouts, and
distinct injective output.

Preserve canonical leading-slice and trailing-domain traversal; dense-linear/general-odometer
geometry; non-injective input reads/injective output writes; ordered first-occurrence boundary
deduplication; typed heap-array/native-order `MemorySegment` and ordered mixed carriers; checked
spans/access/alignment/non-overlap; scalar or disjoint caller-parallel complete-slice ranges;
immutable invocation; and one final typed store per output. Empty leading or normalized extent
remains a no-read/no-write/no-workspace/no-invocation/no-submission output.

Preserve the exact existing numerical algorithms and special-value behavior. Layer remains its
three-pass finite/nonconstant mean, corrected compensated centered-square, then standardize and
optional affine body; it retains finite-constant positive-zero standardization, NaN/infinity slice
propagation, epsilon inside the root, only negative roundoff-residue correction, exact-mean state,
and result-format operation/narrowing boundaries. RMS remains its two-pass scaled-square then
divide/optional-scale body, including epsilon/root construction and its NaN/infinity,
overflow/underflow, and signed-zero behavior. Layer retains exactly one disjoint max-size
`AGGREGATE_EXACT_STATE` slice per simultaneously used range; RMS retains zero workspace.

Cold-reject before output mutation or worker submission every currently excluded semantic,
lowering, binding, and invocation fact: not exactly one admitted first-class occurrence;
decomposed graph; wrong kind/attrs/arity/value identity/type/result/epsilon/gradient/Shape/rank/
trailing normalized Shape/layout; unsupported carrier, inaccessible/undersized/misaligned span,
negative/overflowing geometry, non-injective output, output/input overlap, invalid range/workers/
resources, or unsupported empty/nonempty resource arrangement. Confirm the current boundary from
code before implementation; do not add a finite-input rejection policy or any silent fallback.

## Out of scope

`BATCH_NORM_INFERENCE` and `BATCH_NORM_TRAINING` (later CPU 0009F), loss, softmax, attention,
ordinary reductions, partial/combine protocols, vectors, native routes, fusion, decomposed-kind
recognition, Model/Compiler semantics or gradients, public/shared Prepare/Runtime, resources,
architecture/modules/build, benchmarks, and performance claims. Do not create E3, F, or G tasks.
Legacy is read-only behavioral context only.

## Architecture references

Read [architecture](../../../../../ARCHITECTURE.md), [current architecture plan](../../../../architecture/current-architecture-plan.md), [Planning Guide](../../../planning-guide.md), [documentation rules](../../../../developer-guide/documentation-rules.md), [CPU master plan](../master-plan.md), parent [CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md), completed [E1](0009e1-partial-integral-reduction-direct-java-migration.md) through [E1C](0009e1c-softmax-style-reduction-route-decision.md), and completed [0007F](0007f-portable-layer-and-rms-normalization-coverage.md).

## Architecture constraints

`ARCHITECTURE.md` is authoritative. CPU Prepare owns lowering, finite cold route selection,
resource declaration, and post-slot finalization; Runtime invokes immutable prepared work only.
Keep borrowed carriers, isolated `RunState`, prepared-region ownership, and cold validation
before writes/submission. No public/shared API, generic normalization executor, resource kind,
dependency, architecture, module, or build change: stop and report any need for one.

## Package impact

Existing CPU-private packages only: `internal.ir` and `internal.lowering` retain normalization
identity and geometry; `internal.codegen.emit` retains both emitters if generation remains;
`internal.prepare`, `internal.executable`, `internal.cache`, and `route.portable` retain
cold selection, validation, artifact/binding, finalization, and invocation; `internal.reference`
retains the independent clean-Java oracle. Direct Java may add narrowly named Layer/RMS private
executable owners, never a generic service or public type.

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.ir`
- `io.github.pho001.synaptik.backend.cpu.internal.lowering`
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit`
- `io.github.pho001.synaptik.backend.cpu.internal.prepare`
- `io.github.pho001.synaptik.backend.cpu.internal.executable`
- `io.github.pho001.synaptik.backend.cpu.internal.cache`
- `io.github.pho001.synaptik.backend.cpu.internal.route.portable`
- `io.github.pho001.synaptik.backend.cpu.internal.reference`

Packages added or changed: None.

Type placement:

- Retained generation adds no production type.
- If direct Java wins the strict cost rule, add only
  `internal.executable.CpuLayerNormDirectExecutable` and
  `internal.executable.CpuRmsNormDirectExecutable`; each owns one finite typed family and no
  generic normalization dispatch.

## Affected files

Expected planning paths are this task plus the CPU master plan, parent CPU 0009, and roadmap.
Review-only seams: capability; trailing-normalization IR/lowering; both emitters/classfile
generator; exact-state access; preparation/finalization/executable; portable route; artifact/cache;
reference oracle; generated coverage inventory/evidence; and focused tests.

## Maximum scope

This task may create or modify at most 32 paths: 16 production/Javadoc, 12 test/resource, and
four planning paths; at most two narrowly named CPU-private direct executable types. A 33rd path,
third type, shared/public/resource/build/architecture/conformance/integration change,
materialization, partial/combine state, or generic element dispatch is a stop-and-replan
condition.

## Acceptance criteria

- The final decision covers all four Layer/RMS forms and every current admitted/rejected fact by
  total implementation-plus-verification cost, without a benchmark gate or performance claim.
- The selected route preserves exact algorithms, epsilon, axes/trailing shape, affine/scale
  behavior, types, dense/general geometry, carrier matrix, complete-slice ownership, workspace,
  immutable invocation, cold no-write failure, and final store shape.
- Direct Java, if selected, is finite typed cold-selected carrier/geometry code with immutable
  range-owning calls and no per-element type/carrier/layout/kind dispatch, allocation, boxing,
  reflection, map/string lookup, or avoidable virtual semantic dispatch. It fully retires every
  selected generated emitter/classfile, selection, artifact/cache, invocation, inventory, and
  evidence route.
- Retention records typed carrier/layout, Layer exact-state/RMS zero-workspace, numerical,
  validator/binding, artifact/cache, invocation, oracle, and retirement costs that make replacement
  larger. It retains an optimal clean-Java oracle with matching pass/dataflow/store shape.
- Retention includes generated Class-File and hot-loop comparison with that oracle, demonstrating
  no hidden Synaptik helper call or avoidable hot-path allocation, boxing, reflection, generic, or
  virtual semantic dispatch. Literal-bytecode, JIT, and performance assertions are not required.
- Focused tests cover forms/types, dense/general layouts, all carrier pairings, axes/ranks,
  empty/nonempty geometry, numerical/special/narrowing/one-store behavior, workspace, ranges,
  canaries/no-write failure, lowering/prepare/finalization/invocation/artifact/cache/reference,
  and selected-route retirement or retention inventory.
- A separate clean documentation-focused context finalizes Javadocs, explanatory/planning status,
  glossary impact, and documentation validation in the same change.

## Route-selection decision rule

First establish that either candidate preserves the complete Scope boundary; an unsupported
candidate is not comparable. Record an auditable non-performance comparison: (1) retained
generation's emitter/Class-File inspection, typed carrier/layout binding, Layer workspace/RMS
zero-workspace, validator, artifact/cache, invocation, oracle, and focused-evidence maintenance;
(2) direct Java's finite typed implementation and cold selection/binding; and (3) complete
generated-route retirement plus semantic, rejection, invocation, artifact/cache, inventory,
oracle, and hot-loop verification. Select direct Java only when (2) plus (3) is demonstrably
strictly lower than (1). Equal, uncertain, incomplete, or non-auditable evidence retains
generation. Record facts, buckets, comparison, selected route, and retirement consequence in
validation evidence. This rule supplies no metric, benchmark, or performance claim.

## Tests / validation

After executable stabilization run the focused current normalization-owner tests: capability and
package inventory; trailing-normalization IR/lowering; Layer/RMS generated or replacement
direct-kernel tests; normalization semantic-closure/evidence/coverage inventory; reference; and
affected preparer/finalizer/prepared-executable/artifact-cache tests. Record exact commands and XML
counts; add retirement tests before deletion. Do not run a benchmark or repository-wide suite.

The documentation context reuses stable Java evidence unless it changes executable Java; it checks
links, anchors, fences, terminology/glossary, exact scope, dependencies/status/frontier, then runs:

```bash
git diff --check
git status --short -uall
```

Architecture, conformance, integration, and repository validation defer to 0009G/CI because no
shared or end-to-end boundary may change.

## Dependencies

- E1 through E1C are Complete retained-generated-route decisions.
- Complete 0007F supplies the semantic identity, numerical algorithms, static geometry, resource
  boundary, emitters, binding, artifact/cache, and independent oracle.

## Follow-up tasks

- E3 loss remains Draft and follows E2.
- Batch normalization remains exclusively under Draft 0009F; no E2 child is authorized.

## Architecture impact

Expected impact: None. Stop and report any public, dependency, resource, module, or architecture
conflict.

## Implementation prompt

```text
You are the clean implementation agent for Synaptik CPU 0009E2. Do not commit, push, stage,
reset, revert, delete, or modify unrelated work; do not use GSD. Read AGENTS.md,
ARCHITECTURE.md, the current architecture plan, Planning Guide, CPU master plan, CPU 0009,
E1/E1A/E1B1/E1B2/E1B3/E1C/0007F, this task, current normalization source/tests, documentation
rules, and General/Planning profiles. Implement exactly this route decision. Choose direct finite
typed Java only when its complete implementation and verification cost is strictly lower;
otherwise retain generation with concrete evidence and Class-File/hot-loop comparison to the
clean-Java oracle. Preserve the confirmed boundary; stop on conflict. Hand the stabilized diff and
exact test evidence to a separate clean documentation context. No benchmarks/repository-wide
tests/commit/push/stage.
```

## Rationale

Layer and RMS share static trailing-slice lowering and cold multi-carrier binding but have distinct
pass counts, special-value behavior, and workspace. Their existing dedicated emitters and oracle
make one bounded route decision coherent without absorbing batch normalization or adjacent
reduction families.

## Local decisions

Empty until implemented.

## Known limitations

No speed, vector, scaling, bytecode-identity, JIT, or performance conclusion follows. Dynamic
Shapes/layouts, non-first-class/decomposed forms, partial/combine execution, native execution, and
batch normalization remain excluded.

## Validation evidence

Planning-only: inspected the governing architecture/planning/documentation records, completed
E1--E1C and 0007F, and current Model/Compiler and CPU normalization source/tests. Confirmed that
E2 owns only static first-class Layer/RMS normalization and that batch normalization is later
0009F. No Gradle or benchmark ran.

## Implementation notes

Empty until implemented.

## Completion summary

Empty until implemented.
