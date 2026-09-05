# CPU Task 0008J: BFLOAT16 Scalar Pointwise Closure

## Status

Complete

## Goal

Close the existing generated scalar and caller-parallel scalar pointwise route for the complete
current Model-defined BFLOAT16 pointwise subset. Preserve exact typed attributes, operand order,
broadcasting, current carrier and access-plan support, bounded pointwise-DAG fusion, publication,
and failure behavior while making every BFLOAT16-producing logical operation decode represented
raw bits to FLOAT32, evaluate with the established CPU FLOAT32-domain semantic algorithm, and
round-to-nearest-ties-to-even exactly once into BFLOAT16 at that operation boundary.

Directly generated Class-Files must match an optimal clean-Java implementation of the same
specialized case in semantic algorithm, hot-loop and dataflow shape, and avoidable-overhead
profile. This task adds no Vector API BFLOAT16 route and no generic fallback.

## Scope

### Source-backed operation inventory

The live Model vocabulary contains exactly nineteen `UnaryElementwiseKind` constants, so the old
draft count is accurate. This task admits exactly the following forty-four existing
`CpuPointwiseOpcode` forms when every numeric input, BFLOAT16-producing output, scalar immediate,
and clamp bound has the exact type required below:

- binary arithmetic (7): `ADD`, `SUB`, `MUL`, `DIV`, `MIN`, `MAX`, `POW`;
- scalar arithmetic/range (8): `SCALAR_ADD`, `SCALAR_SUB`, `SCALAR_MUL`, `SCALAR_DIV`,
  `SCALAR_MIN`, `SCALAR_MAX`, `SCALAR_POW`, `SCALAR_CLAMP`;
- unary (19): `ABS`, `NEG`, `RECIPROCAL`, `LOG`, `LOG1P`, `EXP`, `EXPM1`, `ERF`, `SQRT`,
  `RSQRT`, `FLOOR`, `CEIL`, `SIGN`, `RELU`, `SIGMOID`, `TANH`, `GELU_EXACT`,
  `GELU_TANH_APPROXIMATION`, `SILU`;
- comparisons (6): `GREATER_THAN`, `GREATER_OR_EQUAL`, `LESS_THAN`, `LESS_OR_EQUAL`, `EQUAL`,
  `NOT_EQUAL` over two BFLOAT16 operands with canonical BOOL output;
- classifications (3): `IS_FINITE`, `IS_NAN`, `IS_INF` over BFLOAT16 with canonical BOOL output;
  and
- selection (1): `WHERE` with canonical BOOL condition, two BFLOAT16 branches, and BFLOAT16
  output in exact condition/true-branch/false-branch order.

The inventory excludes all three BOOL logical opcodes because they are not BFLOAT16 operations,
and excludes `CAST` because conversion semantics belong to CPU 0008K. It also excludes any future
Model kind or CPU opcode until separately planned. Binary and scalar `POW` are admitted because
the current Model contracts permit BFLOAT16 floating operands/results; base is the left/Tensor
input and exponent is the right/scalar operand.

### Exact BFLOAT16 operation boundary

- A materialized or virtual BFLOAT16 value is represented by its raw 16-bit pattern in a Java
  `short` carrier or an integer local containing those bits. A numerical consumer expands it by
  the `BFloat16Bits.toFloat` rule: unsigned raw bits become the high sixteen bits of binary32.
- Each BFLOAT16-producing arithmetic, scalar, clamp, unary, or power instruction evaluates the
  same FLOAT32-domain algorithm currently selected for the corresponding FLOAT32 opcode,
  including that algorithm's required binary64 JDK calls and its existing final FLOAT32
  narrowing. It then applies exactly one `BFloat16Bits.fromFloat`-equivalent conversion:
  round-to-nearest with ties to even for finite values, preserve signed zero and infinities, and
  canonicalize every produced NaN to `0x7FC0`.
- `MIN` and `MAX` preserve the Model directional signed-zero and NaN-class rules; `CLAMP` remains
  the one logical operation `MIN(MAX(input, lower), upper)` and performs one BFLOAT16 encode only
  after that complete operation, not after its internal extrema steps.
- Comparisons first decode both represented BFLOAT16 values and apply the existing represented-
  numeric FLOAT32 comparison contract: ordered comparisons are false for NaN, `NOT_EQUAL` is the
  complement of numeric equality, and opposite signed zeros compare equal. Classifications decode
  once and classify the resulting binary32 value. Both families store canonical BOOL zero/one.
- `WHERE` is selection, not arithmetic conversion: it copies the exact raw bits of the selected
  BFLOAT16 branch. It therefore preserves selected signed zero, infinity, and every selected raw
  NaN pattern rather than canonicalizing an unchanged selected value.
- Exact BFLOAT16 scalar metadata enters through `ScalarValue.bfloat16Bits` and
  `ScalarValueAttrs`; lowering preserves the low sixteen raw bits without FLOAT32 scalar
  substitution. Scalar `SUB` is input minus scalar, scalar `DIV` is input divided by scalar, and
  scalar `POW` is input base raised to scalar exponent. `ClampRangeAttrs` preserves exact ordered
  BFLOAT16 lower/upper raw bits.
- Extend scalar-power realization only where its BFLOAT16 result obeys the same boundary. In
  particular, an identity realization may not raw-forward a NaN around the required operation
  encode; square and reciprocal operate in FLOAT32 then encode once. Any realization not proved
  equivalent must use the direct FLOAT32-domain power path.

### Carriers, layouts, ranges, and fusion

- Support `short[]`, native-order writable/readable `MemorySegment`, and legal mixed array/segment
  boundary assignments through the existing typed generated entry shape. BOOL outputs retain
  `byte[]`/segment carriers. Do not add `Object` bridges or carrier conversion buffers.
- Retain all currently supported pointwise access regimes: `DENSE_LINEAR`; read-only
  `SCALAR_ALL_ZERO`; `LAST_AXIS_BIAS`; `BLOCK_OUTER`; and `GENERAL_ODOMETER`. Preserve right-
  aligned broadcasting, arbitrary legal non-negative offsets/strides, referenced-span checks,
  injective outputs, and existing binding/overlap rejection. A broadcast or otherwise
  non-injective output remains unsupported.
- Reuse the same generated scalar invocation for single-thread scalar and caller-parallel scalar
  execution. Every half-open range owns disjoint logical output coordinates; empty and partial
  ranges, worker validation/joining, failure-before-publication, and zero-workspace behavior stay
  unchanged.
- Bounded pointwise fusion is legal only when every virtual BFLOAT16 value remains raw represented
  bits. A BFLOAT16-producing instruction must encode into its raw local before any consumer, and a
  numerical consumer must decode that local. This preserves one rounding boundary per logical DAG
  node without materializing a buffer and prevents final-store-only narrowing.
- If any generator, fixed guarded loop, multi-store form, or fused topology cannot prove that raw-
  local invariant, reject that contraction and retain CPU 0008B's deterministic materialized split.
  If even the bounded split topology cannot be represented, capability/preparation fails closed
  before declaration or artifact lookup. Never silently keep a FLOAT32 virtual intermediate,
  disable a required encode/decode, or route to a generic/reference fallback.

### Schema, artifacts, evidence, and documentation

- Advance `CpuGeneratorSchema.CURRENT_VERSION` exactly once from 58 to 59 after emitted bytes
  change. Schema 58 and older envelopes remain incompatible safe misses; add no migration,
  compatibility alias, or dual-schema reader.
- Existing IR structural identity already includes value data types, opcodes, typed immediate
  bits, power realization, clamp bounds, stores, and access regimes; specialization identity
  includes exact carriers and execution strategy. Extend validation/encoding for BFLOAT16 without
  adding Shape, extent, address, slot, range, or concrete segment identity to the cache key.
- Prove unchanged FLOAT64/FLOAT32/INT32/INT64/BOOL projections produce byte-identical generated
  Class-Files. BFLOAT16 creates new pointwise projections only; it must not rewrite older family
  algorithms or specialize by fixed Shape.
- Finalize affected Javadocs, package summaries, CPU backend guide, glossary impact, task evidence,
  master-plan status, and roadmap status in a distinct clean documentation-focused context after
  executable work stabilizes.

## Out of scope

- Java Vector API or BFLOAT16 SIMD, vector tails selected from a BFLOAT16 vector body, native
  execution, OpenBLAS, or a new peer route.
- `CAST`, mixed numeric promotion, implicit FLOAT32 scalar metadata, cross-type WHERE branches,
  BOOL logical closure, or a new Model/public API/Compiler/Prepare/Runtime contract.
- New materialization policy, representation candidate, profitability rule, fusion grammar,
  autotuning, fixed-Shape/fixed-trip specialization, packing, workspace, or generic fallback.
- Fusion through numerical-order, stateful, random, scan, reduction, normalization, convolution,
  attention, loss, movement, indexing, scatter, fold, or ordering units.
- Changing Model numerical meaning, approximation formula, gradient behavior, publication order,
  alias policy, module dependencies, build structure, architecture rules, conformance, or
  integration ownership.
- CPU 0008K–0008P or CPU 0009 implementation or detailed planning.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [`current architecture plan`](../../../../architecture/current-architecture-plan.md)
- [`planning guide`](../../../planning-guide.md)
- [`documentation rules`](../../../../developer-guide/documentation-rules.md)
- [`general documentation profile`](../../../../developer-guide/documentation/general-style.md)
- [`planning documentation profile`](../../../../developer-guide/documentation/planning-style.md)
- [`CPU master plan`](../master-plan.md)
- [CPU 0005E pointwise coverage](0005e-portable-pointwise-types-carriers-and-semantic-family-expansion.md)
- [CPU 0005F scalar POW](0005f-floating-division-and-exact-scalar-power-realization.md)
- [CPU 0005G pointwise closure](0005g-extrema-clamp-tensor-power-and-logical-coverage.md)
- [CPU 0005H unary closure](0005h-portable-unary-transcendental-and-activation-closure.md)
- [CPU 0005J vector/mask closure](0005j-bounded-pointwise-coverage-and-parity-hardening.md)
- [CPU 0007A0 generated parity](0007a0-generated-hot-path-parity-correction.md)
- [CPU 0007A1A scalar-body self-containment](0007a1a-generated-scalar-body-self-containment.md)
- [CPU 0007A1L general-loop parity](0007a1l-pointwise-general-loop-residual-parity.md)
- [CPU 0008B bounded DAG/fusion](0008b-general-partition-dag-computation-unit-decomposition-and-bounded-fusion.md)
- [CPU 0008E1 shared DAG adoption](0008e1-shared-partition-dag-adoption-and-reconstruction-removal.md)
- [CPU 0008I loss closure](0008i-portable-loss-family-execution.md)

## Architecture constraints

- Model remains the semantic authority. CPU capability and cold analysis admit only exact current
  occurrences; CPU lowering/fusion/strategy/resource selection precede shared assignment;
  finalization realizes the selected artifact; Runtime invokes immutable prepared primitives.
- Preserve one atomic CPU partition executable, exact resource declarations, validation before
  write, deterministic topological unit order, and caller-parallel completion/join before the next
  unit or publication.
- Generated code remains typed, direct, allocation-free in normal invocation, and free of graph,
  operation, layout, route, cache, resource, and worker interpretation.
- The optimal clean-Java specialized implementation is the design and review oracle. Any generated
  deviation in semantic algorithm, per-node BFLOAT16 round-trip, hot-loop/dataflow shape, or
  avoidable overhead needs an explicit technical reason and supporting evidence; otherwise stop.
- If a proposed BFLOAT16 operation lacks a complete current Model contract, exclude it and report
  the prerequisite. Do not infer semantics from a draft planning row.

## Package impact

Existing packages used and changed:

- `io.github.pho001.synaptik.backend.cpu` — exact occurrence-local capability;
- `...internal.ir` — BFLOAT16 validation for the existing opcode/value/immediate vocabulary;
- `...internal.lowering` — exact raw scalar/clamp lowering and fusion legality;
- `...internal.codegen.emit` — raw BFLOAT16 locals, conversion, scalar formulas, carriers, and
  generated Class-File assembly;
- `...internal.prepare` and `...internal.executable` — unchanged scalar/parallel-scalar selection,
  binding, and range execution, edited only if BFLOAT16 validation is currently closed;
- `...internal.cache` — schema 59 and current-only compatibility; and
- `...internal.reference` — test/evidence oracle only, never a production generated dependency.

No package or public type is added. At most one focused test-only BFLOAT16 performance owner may be
added; production BFLOAT16 behavior belongs in the existing pointwise owners.

## Affected files

Expected production/Javadoc paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuKernelIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPointwiseOpcode.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionDagDecomposer.java` only if the raw-local proof needs an explicit fusion gate
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScalarEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuCarrierEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutable.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecialization.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuScalarReferenceKernel.java`
- directly affected package-info files in those existing packages.

Expected focused tests are the matching capability, opcode/IR, pointwise lowering/decomposition,
pointwise/fused/DAG generated, preparer/prepared-executable, specialization/artifact-store, and
reference-oracle owners, plus at most one new
`CpuBFloat16PointwisePerformanceTest.java`. Documentation paths are the CPU backend guide, glossary
only if terminology changes, this task, CPU master plan, and roadmap.

## Maximum scope

At most 38 repository paths may change: 16 production/Javadoc paths, 16 test/evidence paths, and
6 documentation/planning paths. No new production type or package is authorized. Evidence
Class-Files, decompilation, benchmark sources/classes/logs, manifests, and environment records live
under one fresh `/private/tmp/synaptik-cpu-0008j-*` directory and do not count. If correct support
requires a Model/shared-module change, a second production abstraction, more than 38 paths, or an
out-of-scope route/policy, stop and replan.

## Acceptance criteria

1. Capability, IR, and lowering accept exactly the forty-four listed BFLOAT16 forms and reject
   mixed types, wrong result types, non-BFLOAT16 scalar/clamp attributes, malformed arity/order,
   BOOL logic, CAST, non-injective outputs, unsupported layouts, and unproved spans.
2. Raw-bit semantic tests cover every admitted opcode, scalar-power realization, arrays, segments,
   representative mixed carriers/access regimes, scalar and parallel-scalar ranges, and fused and
   split topologies. Cases include both zeros, infinities, extrema, subnormals, halfway ties with
   even/odd retained least-significant bits, overflow/underflow, and quiet/signaling/signed/payload
   NaNs according to the exact production contract.
3. Every BFLOAT16-producing DAG node encodes once before a consumer; generated/decompiled fused
   evidence exposes the raw-local encode/decode boundary and differs from a deliberately
   final-store-only control on a rounding-sensitive chain. `WHERE` preserves selected raw bits;
   comparisons/classifications produce canonical BOOL.
4. Existing array/segment/mixed carrier binding, all five access regimes where legal, injective
   output, overlap rejection, arbitrary half-open ranges, caller-parallel ownership/join,
   zero-resource behavior, and atomic publication remain intact.
5. Generated classes are final and field-free with one typed static entry and no hidden Synaptik
   helper/fallback/reference call, allocation, boxing, reflection, method handle,
   `invokedynamic`, collection/string dispatch, monitor, or operation/graph/runtime dispatch.
   Permitted JDK numerical calls are enumerated and match the FLOAT32-domain oracle.
6. Schema advances exactly 58 to 59. Schema-58/older envelopes are safe misses, malformed entries
   fail safely, BFLOAT16 typed immediates and access/carrier shapes cannot alias, and every unchanged
   pre-0008J pointwise/family projection retains byte-identical generated Class-File bytes.
7. A fresh five-fork generated-versus-direct matrix covers five representative, genuinely distinct
   emitted/dataflow forms:
   - contiguous `short[]` unary `GELU_EXACT` (binary64 internal activation branch);
   - dense all-segment binary `POW` (two-input JDK-power path);
   - mixed-carrier BFLOAT16 `SCALAR_CLAMP` with general-affine injective output (two exact raw
     immediates and general addressing);
   - broadcast/mixed BFLOAT16 comparison or classification to canonical BOOL (different result
     type and predicate branch); and
   - a bounded fused `ADD -> SIGMOID -> WHERE` or equivalently rounding-sensitive chain spanning
     raw BFLOAT16 and virtual BOOL values (per-node round-trip and fusion branch).
   This matrix is representative rather than a carrier Cartesian product because array/segment
   access, access regime, predicate output, JDK numerical branch, typed immediates, and fused raw-
   local dataflow are the code-shaping dimensions; swapping every boundary independently does not
   create a new semantic algorithm. Add a row only if implementation inspection proves another
   generated code shape. Every row in every fork and the median of fork medians must be
   generated/direct `<= 1.15x` under the repository's fixed-heap, warmup, randomized-order,
   adaptive-at-least-25-ms, no-retry/no-discard protocol.
8. The direct oracle uses ordinary clean Java with the same typed carriers, cold geometry/ranges,
   operation order, FLOAT32/binary64 work, raw-local per-node encodes, stores, and loop/address
   form. Source plus complete `javap -c -v -p` comparison proves neither side hides dispatch,
   allocation, helper indirection, fixed fixture extents, or avoidable per-element work.
9. A distinct clean documentation pass finalizes affected Javadocs/package summaries, CPU guide,
   glossary decision, evidence, and planning state. CPU 0008K–0008P retain their exact existing
   scope and order.

## Tests / validation

Tier 1: run focused capability, opcode/IR, lowering/decomposition, reference, generated pointwise,
fused/DAG, binding/preparation/executable, and schema/cache tests while developing. The matrix must
prove all forty-four forms semantically, exact raw attributes, five access regimes where legal,
array/segment/mixed carriers without exhaustive redundant permutations, range ownership, alias
failure before writes, and fused per-node rounding.

Tier 2: after executable work stabilizes, run once:

```text
./gradlew :backends:cpu:test --rerun-tasks
```

Generate the representative artifacts and retain their Class-Files, checksums, complete
decompilation, descriptor/member-reference/forbidden-opcode reports, oracle source/classes, raw
five-fork logs, summary, commands, JVM/OS/CPU metadata, and verified manifest under the one fresh
evidence root. No failed fork may be discarded or averaged away.

Tier 3: the separate documentation-focused context reuses stable Java/performance evidence unless
it changes executable behavior, then runs:

```text
./gradlew :backends:cpu:javadoc
git diff --check
git diff --cached --check
git status --short
```

It also checks Markdown links/anchors/fences/headings/final newlines, schema/status synchronization,
exact path bounds, unchanged CPU 0008K–0008P rows/order, no 0008K task file, no staged change, and
preservation of unrelated dirty work. Repository-wide, architecture, conformance, and integration
tests remain deferred to CPU 0009/CI because this task changes no shared boundary; discovering such
a change is a stop condition.

## Dependencies

- Complete CPU 0008I and current schema-58 generated infrastructure.
- Complete CPU 0005E–0005J pointwise operation, scalar-power, closure, unary, and vector-boundary
  contracts.
- Complete CPU 0007A0/0007A1A/0007A1L generated-loop, self-containment, and direct-oracle rules.
- Complete CPU 0008B and 0008E1 bounded fusion/shared-DAG infrastructure.
- Current Model typed-scalar, promotion, binary/scalar arithmetic, comparison/classification,
  WHERE, unary/activation, floating-extrema/clamp, and `BFloat16Bits` contracts.

## Follow-up tasks

- CPU 0008K remains the next Draft task after this task completes and owns only Model-defined
  cross-type CAST execution. Do not create its detailed specification here.
- CPU 0008L–0008P retain their current user-owned scope and order unchanged.
- CPU 0009 remains the generated-coverage checkpoint and also owns the missing corrected CPU 0008I
  loss performance evidence.

## Architecture impact

Expected impact: None.

This task extends a CPU-private generated representation already authorized by the architecture.
It changes no semantic authority, module edge, public/shared contract, resource lifecycle, or
Runtime responsibility. If implementation needs such a change, stop and report it.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are the clean implementation agent for Synaptik CPU task 0008J. Work on the existing dirty
worktree without committing, pushing, staging, resetting, reverting, or modifying unrelated work.
Do not use GSD. Preserve exactly the test-only CpuLossPerformanceTest correction and the completed
CPU 0008I documentation.

Read AGENTS.md, ARCHITECTURE.md, the current architecture plan, planning guide, CPU master plan,
task 0008J, its referenced completed CPU tasks, the relevant Model contracts, and every affected
pointwise production/test owner in full. Implement exactly the forty-four-form BFLOAT16 scalar and
caller-parallel scalar closure. Keep BFLOAT16 locals as raw bits; decode for numerical work and
encode once at every BFLOAT16-producing logical node before any fused consumer. Preserve WHERE raw
selection and canonical BOOL predicates. Reject fusion to the legal split topology whenever that
invariant is not proved. Do not add SIMD, CAST, mixed promotion, public API, materialization,
autotuning, fixed-Shape specialization, native routing, or fallback.

Run the focused/full CPU, semantic raw-bit, binding/range/publication, schema/cache, complete
Class-File/decompilation/forbidden-reference, and five-fork representative direct-Java performance
gates. Retain reproducible evidence under one fresh /private/tmp directory. Then hand the stable
diff and evidence to a distinct clean documentation-focused context following documentation-
rules.md. Do not mark Complete until every acceptance criterion and that final pass succeed. Stop
on architecture, Model-contract, oracle-equivalence, fusion-rounding, or path-budget uncertainty.
```

## Local decisions

- The live Model and CPU vocabularies confirm nineteen unary kinds and forty-four BFLOAT16-relevant
  opcode forms; the three BOOL logical forms and same-type `CAST` are outside this closure.
- Per-node rounding is resolved by typed operation results, not by fusion policy. Raw BFLOAT16
  virtual locals make legal fusion possible without materialization; final-store-only narrowing is
  semantically invalid. Split retention is the fail-closed alternative.
- `WHERE` copies a selected represented value and therefore preserves raw branch bits. Produced
  numerical NaNs pass through the required encode and become canonical `0x7FC0`.
- The five-row performance matrix covers each distinct algorithm/dataflow/access branch without
  demanding a redundant full carrier Cartesian product.

## Known limitations

- No BFLOAT16 SIMD or mixed-type pointwise execution is claimed.
- Performance conclusions apply only to the retained representative code shapes, environment,
  JVM, and protocol; CPU 0009 remains responsible for whole-backend evidence classification.

## Validation evidence

- The implementation closes exactly 44 BFLOAT16 pointwise forms: 7 binary, 8 scalar/range, 19
  unary, 6 comparison, 3 classification, and `WHERE`. Generated scalar and caller-parallel scalar
  execution passed across arrays, native-order `MemorySegment`, mixed carriers, and all five legal
  current access regimes. No BFLOAT16 SIMD, CAST, mixed promotion, native route, new
  materialization/autotuning policy, fixed-Shape specialization, or generic fallback was added.
- The authoritative CPU rerun completed successfully with 132 suites, 692 tests, 14 expected
  opt-in skips, and zero failures or errors. Focused/evidence results contain 14 suites, 178 tests,
  3 expected opt-in skips, and zero failures or errors. Independent review corrected BFLOAT16
  scalar-POW analysis and made the historical CPU 0008E external-evidence test opt-in through
  `-Dsynaptik.cpu.0008e.retainedEvidence=true` without weakening any assertion when enabled.
- Accepted retained evidence is `/private/tmp/synaptik-cpu-0008j-K4FprD`. Its 85-artifact manifest
  verifies completely. Five exact generated descriptors cover GELU over arrays, POW over segments,
  mixed general CLAMP, broadcast comparison to BOOL, and fused WHERE. Every generated and direct
  scan passed: each generated class is final and field-free with exactly one typed static entry;
  there are no forbidden references, allocation, boxing, reflection, dynamic dispatch, or generic
  `Object` carriers. Complete direct timed source, class, and `javap` output are retained.
- Every row in every accepted fork passed the `<= 1.15x` generated/direct gate. Median-of-five
  ratios are `0.9334689173615348` for `GELU_ARRAY`, `1.093502884572582` for `POW_SEGMENT`,
  `0.9634384935191979` for `CLAMP_MIXED_GENERAL`, `0.9369720889236289` for
  `COMPARISON_BROADCAST`, and `0.9312897148034954` for `FUSED_WHERE`. The maximum individual
  accepted ratio is POW at `1.1023740371679942`.
- The first attempt at `/private/tmp/synaptik-cpu-0008j-cfHknH` remains retained and rejected:
  POW measured `2.170433141804421` because its direct oracle used mismatched address/code shape.
  The accepted oracle instead uses matching cold geometry-derived addressing. Neither protocol
  retried nor discarded samples, and the failed attempt was not relabelled as accepted evidence.
- Generated code was compared with an optimal clean Java implementation for equivalent semantic
  algorithm, hot-loop/dataflow shape, and avoidable overhead. This is not byte-for-byte Class-File
  identity, guaranteed identical JIT assembly, a whole-backend performance-parity result, or a
  hardware-universal claim. CPU 0009 still owns CPU 0008I's missing corrected full loss evidence
  and the later complete evidence inventory.
- The independent documentation context reviewed every changed production Javadoc and found the
  final behavior, invariants, parameters, results, failures, and formatting accurate. The earlier
  passing `:backends:cpu:javadoc` result remains applicable because this pass changed no Java
  comment. The CPU guide now documents current BFLOAT16 support, exact exclusions, schema-59
  isolation, and fused per-node rounding. Existing glossary entries were corrected; no new
  reusable BFLOAT16 operation-boundary term was introduced.
- Documentation validation passed local links and anchors, heading/status/frontier/schema/path
  assertions, balanced fences, terminal newlines, whitespace, exact task scope, protected CPU
  0008I and `CpuLossPerformanceTest` preservation, unchanged CPU 0008K–0008P order/scope, absence
  of a detailed 0008K task, empty staging, `git diff --check`, and `git diff --cached --check`.

## Implementation notes

- BFLOAT16 values remain raw 16-bit virtual locals. Numerical consumers decode at use, and every
  arithmetic-producing logical DAG node performs exactly one inline ties-to-even/canonical-NaN
  encode before any fused consumer. `WHERE` copies selected raw bits; comparisons and
  classifications emit canonical BOOL. Any unproved contraction retains the deterministic legal
  split or fails closed.
- Schema 59 applies only to BFLOAT16 pointwise projections. Older projections and generated bytes
  remain unchanged and older envelopes are safe misses.
- Architecture/current-plan/ADR/architecture-test conclusion: no ownership, module boundary,
  dependency direction, public/shared contract, or architecture rule changed, so no architecture
  document, ADR, or architecture-test edit is required.
- API/lifecycle/test conclusion: Model, Compiler, Runtime, shared Prepare, Tensor, Compile, and
  Training API contracts remain accurate because this is a CPU-private realization of existing
  Model semantics. No conformance/integration, Gradle/build, module-dependency, or unrelated
  documentation change is required. Cross-type CAST remains gated by future Model conversion
  semantics in Draft CPU 0008K.

## Completion summary

Completed the exact 44-form BFLOAT16 scalar pointwise closure at schema 59, including raw-local
per-node rounding, raw-preserving WHERE, canonical predicates, all current access/carrier forms,
safe fusion splitting, semantic/Class-File evidence, and representative five-fork generated/direct
performance evidence. Finalized affected Javadocs, the CPU guide, established glossary entries,
this task, CPU master plan, and roadmap without changing executable Java or tests in the
documentation pass. CPU 0008I's waived exception remains unchanged; CPU 0008K is the next Draft
frontier with its future Model CAST-semantics prerequisite, and CPU 0008K–0008P retain their exact
scope and order.

Unresolved issues: None for CPU 0008J.

Required follow-up: None for CPU 0008J. CPU 0009 retains the separately recorded missing CPU 0008I
loss evidence and whole-backend evidence inventory responsibility.

Status: Complete
