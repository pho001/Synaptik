# Task 0005H: Portable Unary, Transcendental, and Activation Closure

## Status

Complete

## Goal

Close the current Model unary pointwise inventory for exact/default portable CPU execution without
adding a second evaluator, operation-specific class hierarchy, public fast-math surface, or graph
rewrite:

```text
nineteen UnaryElementwiseKind values
  -> one closed CPU unary opcode row per semantic kind
  -> existing scalar or parallel-scalar generated execution for FLOAT32/FLOAT64
  -> selected honest FLOAT64 Vector API rows when Java 26 supplies the required lane operations
  -> one compatible generated artifact and one partition executable
```

CPU 0005E–0005G already implement `NEG` for FLOAT32/FLOAT64 and exact-semantic `GELU` for
FLOAT64. This task preserves those rows, corrects the existing GELU negative-infinity result to
the Model's continuous extension, adds FLOAT32 GELU, and adds the other seventeen unary kinds for
FLOAT32/FLOAT64. The already-complete `IS_FINITE`, `IS_NAN`, and `IS_INF` classification rows are
audited and regression-tested but not reimplemented.

The portable route remains bytecode-first. `CpuScalarReferenceKernel` is the cold conformance and
fallback oracle over already-lowered CPU IR, not another backend or a Runtime interpreter.

## Scope

### Exact operation and type matrix

Every admitted occurrence has exactly one input and one output, exact
`NoOperationAttrs.INSTANCE`, equal input/output Shape and data type, fully static Shapes, and
resolved layouts. The only admitted data types are FLOAT32 and FLOAT64.

The live and task-0005H matrix is:

| Model kind | Before 0005H | Added or changed by 0005H | Final scalar / parallel-scalar | Final FLOAT64 vector / parallel-vector |
|---|---|---|---|---|
| `ABS` | unsupported | add FLOAT32/FLOAT64 | yes | yes, `ABS` |
| `NEG` | FLOAT32/FLOAT64 | preserve | yes | yes, existing `NEG` |
| `RECIPROCAL` | unsupported | add FLOAT32/FLOAT64 | yes | yes, exact `1.0 / x` |
| `LOG` | unsupported | add FLOAT32/FLOAT64 | yes | yes, `LOG` |
| `LOG1P` | unsupported | add FLOAT32/FLOAT64 | yes | yes, `LOG1P` |
| `EXP` | unsupported | add FLOAT32/FLOAT64 | yes | yes, `EXP` |
| `EXPM1` | unsupported | add FLOAT32/FLOAT64 | yes | yes, `EXPM1` |
| `ERF` | unsupported | add FLOAT32/FLOAT64 using the existing shared approximation | yes | yes, existing vector polynomial/`EXP` helper |
| `SQRT` | unsupported | add FLOAT32/FLOAT64 | yes | yes, `SQRT` |
| `RSQRT` | unsupported | add FLOAT32/FLOAT64 as `1.0 / sqrt(x)` | yes | yes, `SQRT` then exact division |
| `FLOOR` | unsupported | add FLOAT32/FLOAT64 | yes | no Java 26 Vector operator; scalar compute |
| `CEIL` | unsupported | add FLOAT32/FLOAT64 | yes | no Java 26 Vector operator; scalar compute |
| `SIGN` | unsupported | add FLOAT32/FLOAT64 | yes | no direct Java 26 Vector operator selected; scalar compute |
| `RELU` | unsupported | add FLOAT32/FLOAT64 as `max(x, +0)` | yes | no new extrema proof in this task; scalar compute |
| `SIGMOID` | unsupported | add stable FLOAT32/FLOAT64 realization | yes | scalar compute; no per-lane fallback loop |
| `TANH` | unsupported | add FLOAT32/FLOAT64 | yes | yes, `TANH` |
| `GELU` | FLOAT64 | add FLOAT32 and fix `-infinity -> -0` in both routes | yes | yes, existing vector helper with corrected edge handling |
| `GELU_TANH_APPROXIMATION` | unsupported | add fixed Model formula for FLOAT32/FLOAT64 | yes | scalar compute; no general fast-math meaning |
| `SILU` | unsupported | add stable FLOAT32/FLOAT64 realization | yes | scalar compute; no per-lane fallback loop |

The Java 26 incubating Vector API exposes FLOAT64-compatible `ABS`, `NEG`, `EXP`, `LOG`,
`SQRT`, `TANH`, `EXPM1`, and `LOG1P` lane operators. Its math-library operators follow the
accuracy and monotonicity specification of the equivalent Java method, but the JVM may realize
them with scalar lane calls rather than a hardware transcendental instruction. The task may use
those operators and the existing vector helper; it must make no intrinsic, throughput, or
speedup claim. It must not generate an explicit lane-extraction/call/reinsertion loop. A partition
containing any vector-ineligible opcode selects scalar or parallel-scalar compute rather than
failing.

FLOAT32 remains scalar or parallel-scalar under the existing FLOAT64-only vector architecture.
BFLOAT16 remains fail-closed even though Model unary construction accepts floating inputs and CPU
storage has a two-byte carrier. FLOAT16 remains absent pending Model 0026. Integral and BOOL unary
rows, mismatched types, malformed arity/attributes/results, dynamic Shapes, and unresolved layouts
remain unsupported.

### Numerical realization matrix

Generated scalar bytecode and the reference kernel must share the following already-lowered
realizations. FLOAT64 calls operate directly. Unless a native FLOAT32 primitive method exists,
FLOAT32 exactly widens the represented binary32 input to binary64, evaluates the listed operation,
and narrows once to binary32. This is backend-private evaluation of a same-typed FLOAT32 semantic
request, not a Model cast or promotion.

| Group | Scalar realization | Finite conformance |
|---|---|---|
| `ABS`, `NEG` | primitive/`Math.abs` | exact represented result except no NaN payload promise |
| `RECIPROCAL` | exact typed `+1.0 / x` | ordinary correctly rounded primitive division |
| `LOG`, `LOG1P`, `EXP`, `EXPM1` | matching `StrictMath` method | Java 26 method contract: within 1 ulp and semi-monotonic |
| `SQRT` | `StrictMath.sqrt` | correctly rounded closest result |
| `RSQRT` | exact typed `+1.0 / StrictMath.sqrt(x)` | two specified operations; FLOAT32 widens before both operations and narrows only their combined result; differential oracle uses the same order |
| `FLOOR`, `CEIL` | matching `StrictMath` method | exact integral-valued floating result |
| `SIGN` | matching `Math.signum` overload | exact `-1`, signed zero, `+1`, infinity sign, or NaN classification |
| `RELU` | `Math.max(x, +0)` | exact Model activation target with NaN propagation and positive zero at either zero sign |
| `SIGMOID` | sign-stable two-branch formula | target tolerance below; no avoidable overflow |
| `TANH` | `StrictMath.tanh` | Java 26 method contract: within 2.5 ulps |
| `ERF` | existing shared piecewise rational approximation | activation tolerance below |
| `GELU` | existing `0.5*x*(1+erf(x/sqrt(2)))` finite order plus explicit special cases | activation tolerance below |
| `GELU_TANH_APPROXIMATION` | exact fixed Model formula and coefficient `0.044715` | activation tolerance below, measured against that formula rather than exact GELU |
| `SILU` | sign-stable `x*sigmoid(x)` form plus explicit negative-infinity handling | activation tolerance below |

`SIGMOID` evaluates `1 / (1 + exp(-x))` for nonnegative input and
`exp(x) / (1 + exp(x))` for negative input. `SILU` uses the corresponding stable branch without
computing an overflowing exponential; negative infinity is handled explicitly as negative zero.
`GELU` and `GELU_TANH_APPROXIMATION` also handle negative infinity explicitly as negative zero so
their continuous extensions do not become `infinity * zero -> NaN`. No helper is named or exposed
as `fastExp`, `fastTanh`, or a general fast-math policy.

### Exceptional-value contract

Tests and Javadocs must lock these observable classifications and zero signs. NaN results are
checked by classification; no payload, quieting, or sign promise is added.

| Kind/group | `-infinity` | `-0` | `+0` | `+infinity` | domain/NaN rule |
|---|---|---|---|---|---|
| `ABS` | `+infinity` | `+0` | `+0` | `+infinity` | NaN -> NaN |
| `NEG` | `+infinity` | `+0` | `-0` | `-infinity` | NaN -> NaN |
| `RECIPROCAL` | `-0` | `-infinity` | `+infinity` | `+0` | NaN -> NaN |
| `LOG` | NaN | `-infinity` | `-infinity` | `+infinity` | negative finite -> NaN; NaN -> NaN |
| `LOG1P` | NaN | `-0` | `+0` | `+infinity` | `-1 -> -infinity`; below `-1` -> NaN |
| `EXP` | `+0` | `1` | `1` | `+infinity` | NaN -> NaN |
| `EXPM1` | `-1` | `-0` | `+0` | `+infinity` | NaN -> NaN |
| `ERF` | `-1` | `-0` | `+0` | `+1` | NaN -> NaN |
| `SQRT` | NaN | `-0` | `+0` | `+infinity` | negative finite -> NaN; NaN -> NaN |
| `RSQRT` | NaN | `-infinity` | `+infinity` | `+0` | negative finite -> NaN; NaN -> NaN |
| `FLOOR`, `CEIL` | `-infinity` | preserve `-0` | preserve `+0` | `+infinity` | NaN -> NaN |
| `SIGN` | `-1` | preserve `-0` | preserve `+0` | `+1` | NaN -> NaN |
| `RELU` | `+0` | `+0` | `+0` | `+infinity` | NaN -> NaN |
| `SIGMOID` | `+0` | `0.5` | `0.5` | `+1` | NaN -> NaN |
| `TANH` | `-1` | `-0` | `+0` | `+1` | NaN -> NaN |
| both GELU kinds | `-0` | `-0` | `+0` | `+infinity` | NaN -> NaN |
| `SILU` | `-0` | `-0` | `+0` | `+infinity` | NaN -> NaN |

Finite overflow, underflow, and subnormal transitions follow the listed operations. The task adds
no flush-to-zero, finite-only assumption, reassociation, contraction, reciprocal estimate, or
alternate exceptional-value mode.

### Conformance tolerances

Use an independent high-quality mathematical oracle where practical and the exact fixed Model
formula for `GELU_TANH_APPROXIMATION`. The task-level CPU tests must apply these explicit bounds:

| Result family | FLOAT64 bound | FLOAT32 bound |
|---|---:|---:|
| exact/primitive rows and fixed special cases | exact bits when the table fixes a non-NaN result | exact bits when the table fixes a non-NaN result |
| `LOG`, `LOG1P`, `EXP`, `EXPM1` scalar | Java method contract, at most 1 ulp from exact | exact widening, method result, and one narrowing; at most 1 binary32 ulp from correctly rounded oracle unless overflow/underflow fixes the result |
| `SQRT` scalar | correctly rounded | exact widening plus one final binary32 narrowing |
| `TANH` scalar | at most 2.5 ulps | at most 2.5 binary32 ulps after final narrowing |
| `ERF`, `SIGMOID`, both GELU kinds, `SILU` | `max(2e-7, 2e-7 * abs(expected))` | `max(2e-5, 2e-5 * abs(expected))` |
| FLOAT64 vector `EXP`/`LOG`/`LOG1P`/`EXPM1` versus scalar oracle | at most 2 ulps | not applicable |
| FLOAT64 vector `TANH` versus scalar oracle | at most 5 ulps | not applicable |
| vector `ERF`/`GELU` versus scalar oracle | `max(2e-7, 2e-7 * abs(expected))` | not applicable |

`RSQRT`, stable activation helpers, and GELU helpers are additionally differential-tested against
the scalar reference in the exact selected operation order. The tolerances are correctness
bounds, not permission for another algorithm family or relaxed mode. Random agreement alone is
insufficient: boundary-directed and special-value cases are mandatory.

### Capability, lowering, IR, and generated code

- Extend `CpuCapabilityProvider` to advertise exactly the final FLOAT32/FLOAT64 matrix above.
- Extend `CpuPointwiseOpcode` from thirty-one to exactly forty-eight values by adding one opcode
  for each of the seventeen previously unsupported `UnaryElementwiseKind` values. Retain `NEG`
  and `GELU_EXACT`; do not add duplicates for already-implemented semantics.
- Keep all unary opcodes in `Family.UNARY`, with one input, input-typed result, no immediate,
  power realization, or clamp fact. IR validation admits FLOAT32/FLOAT64 for every unary opcode.
- Extend the existing kind-to-opcode switch, scalar emitter, vector helper, and scalar reference
  switch. Add no operation-specific lowerer, emitter, kernel, route, executable, or planner class.
- Generated scalar code may invoke narrow static methods on `CpuScalarReferenceKernel` for the
  shared ERF/activation algorithms. Those methods remain pure primitive helpers used by generated
  bytecode; Runtime never passes them Model operations or IR.
- Preserve one-through-eight connected straight-line pointwise lowering, stored instruction order,
  virtual intermediates, and one final store. Every unary occurrence is one IR instruction.
- Correct existing GELU special-value handling in scalar/reference and vector helpers as part of
  closure; do not retain two competing GELU algorithms.

### Routes, access, materialization, and lifecycle

- Scalar and parallel-scalar cover every final unary row over contiguous, strided, broadcast,
  scalar/all-zero, last-axis, block/outer, and general-odometer access as already normalized by
  the five `CpuAccessPlan` regimes. Unary output Shape equals input Shape, but input layouts may
  still be strided or zero-stride and outputs must remain injective.
- Vector and parallel-vector are eligible only for the FLOAT64 rows in the matrix, and only when
  every existing direct contiguous/scalar-broadcast access, carrier, topology, and strategy gate
  passes. General odometers remain scalar. Scalar tails remain unchanged.
- Preserve arbitrary `start`/`end` ranges, heap arrays, native-order `MemorySegment` carriers,
  mixed carrier patterns, cold binding, and concurrent-run isolation.
- Preserve direct execution plus at most three eligible one-input materialization candidates, at
  most one selected contiguous FLOAT64 input copy, four complete candidates, one realized
  artifact, zero fixed-shape variants, and zero unrolled variants.
- Reference code remains conformance/fallback support. It does not become a competing route,
  another prepared executable, or hot Runtime interpretation.

### Schema and cache compatibility

- Canonical IR identity includes each exact unary opcode. Route strategy, vector species,
  access/carrier/materialization facts, numerical mode, and lowering fingerprint remain existing
  specialization facts.
- Bump `CpuGeneratorSchema.CURRENT_VERSION` from 7 to 8 because the closed opcode vocabulary,
  scalar helper calls, vector eligibility, and corrected GELU bytecode change.
- Schema 8 accepts only current compatible envelopes. Every schema 7 or older artifact is an
  incompatible miss; add no migration reader or converter.
- Preserve deterministic bytes, structural verification, process-local compatible interning,
  assignment-before-artifact ordering, strong prepared ownership, optional bounded persistence,
  and the CPU 0005D `KEEP_DISABLED` default.
- Do not confuse generated-class compatibility with a future workload-tuning cache. No tuning,
  benchmark, cache-policy, or Runtime cache behavior changes.

### CAST and classification audit

- Keep same-type CAST exactly as implemented for FLOAT64, FLOAT32, INT32, INT64, and BOOL.
- Keep every cross-type CAST fail-closed. Live Model `CastKind`/`CastAttrs` permit expression
  construction but still do not define numerical conversion, rounding, saturation, NaN, infinity,
  or BOOL conversion semantics. Java casts and Vector conversions are therefore not authority to
  advertise a backend route.
- Preserve `IS_FINITE`, `IS_NAN`, and `IS_INF` for FLOAT32/FLOAT64 to canonical BOOL. They are
  already complete classification-family opcodes and are not renamed, duplicated, or counted
  among the seventeen additions.

## Out of scope

- public `fastExp`, `fastTanh`, `swish`, configurable GELU coefficients, selectable activation
  approximation, general fast math, relaxed numerical permission, or a new Config policy
- BFLOAT16/FLOAT16 unary execution, mixed precision, cross-type CAST, CPU-side promotion,
  conversion helpers, integral/BOOL unary arithmetic, or another data type
- explicit scalar-lane extraction loops inside vector kernels, a claim of hardware transcendental
  intrinsics, FLOAT32 Vector API support, new species, gather, or masked tails
- fixed-shape or unrolled specialization, directed acyclic graph (DAG) fusion, epilogues,
  operation decomposition/recomposition, compiler constant discovery, graph rewrite, or
  publication remapping
- reductions, scans, softmax, normalization, layout/indexing/random families, native libraries,
  OpenBLAS/vendor routes, autotuning, benchmarks, Engine composition, or later CPU tasks
- per-operation production classes, another IR, another compiler/interpreter, a public CPU facade,
  another backend identity, dependency/Gradle/build changes, or architecture changes
- Model, Compiler, Config, Planning, shared Prepare, Runtime, Backend Contract, Trace, Engine,
  OpenBLAS provider, other backend, backend-conformance, or integration source changes
- a detailed task specification for CPU 0005H1, 0006, or any later task

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Backend-owned lowering ADR](../../../../design/decisions/0002-backend-owned-lowering.md)
- [Staged backend preparation ADR](../../../../design/decisions/0010-staged-backend-preparation.md)
- [Per-run Runtime ownership ADR](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)
- [Backend-guide style](../../../../developer-guide/documentation/backend-guide-style.md)
- [Planning style](../../../../developer-guide/documentation/planning-style.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [CPU master plan](../master-plan.md)
- [CPU backend guide](../../../../backend-guide/cpu-backend.md)
- [Task 0005F](0005f-floating-division-and-exact-scalar-power-realization.md)
- [Task 0005G](0005g-extrema-clamp-tensor-power-and-logical-coverage.md)
- [Model task 0018P](../../../modules/model/tasks/0018p-elementwise-semantic-cleanup.md)
- [Model task 0018T1](../../../modules/model/tasks/0018t1-unary-numeric-gaps-and-floating-diagnostics.md)
- [Model task 0019A](../../../modules/model/tasks/0019a-modern-activation-semantics-and-tensor-expressions.md)

## Architecture constraints

- Model owns the unary semantic identities and fixed activation formulas. CPU may select a
  conforming internal algorithm but may not add a Model kind, Tensor method, parameter, or route
  hint.
- Planning selects CPU ownership. CPU analysis owns capability truth, lowering, exact/default
  numerical eligibility, compute strategy, specialization, and declarations; CPU finalization
  owns generated artifact realization after shared slot assignment.
- Generated code and Runtime never inspect `Operation`, `CompiledNode`, Model attributes, or
  strings. Common lowering translates each occurrence once into typed CPU-private IR.
- Compiler owns graph rewrites and gradients. CPU retains every occurrence and applies the same
  forward policy to forward and compiler-generated operations.
- Shared Prepare continues to see one opaque CPU plan plus exact declarations. Runtime performs
  cold checked binding and direct invocation only.
- Bytecode-first portable execution remains the semantic baseline. Reference code is a
  conformance/fail-closed helper, not another backend route.
- No module dependency, package responsibility, lifecycle, public API, architecture rule, or
  Gradle configuration changes. Stop and report if implementation requires one.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu` — truthful occurrence-local capability reporting.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — the closed family opcode vocabulary and
  typed canonical instruction validation.
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — common Model-kind-to-IR lowering.
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — scalar bytecode emission,
  FLOAT64 vector helper emission, and generated class validation.
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — pure scalar conformance helpers over
  already-lowered primitive semantics.
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — current generator schema compatibility.
- Existing prepare, route, executable, memory, and cache packages — review-only unless current
  validated opcode/vector facts already flow through one of their existing contracts.

Packages added, moved, or removed:

- None.

Type placement:

- The seventeen new opcodes remain values of `CpuPointwiseOpcode`.
- Shared ERF and activation algorithms remain static primitive methods on the existing
  `CpuScalarReferenceKernel`, callable by generated bytecode and reference evaluation.
- No new production type is planned. If a correct implementation needs one, stop and revise this
  specification before implementation continues.

## Affected files

Expected CPU production/package paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScalarEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuVectorEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuKernelIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPointwiseOpcode.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuScalarReferenceKernel.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/package-info.java`

Expected CPU test paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuFusedGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPointwiseGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuKernelIrTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPointwiseOpcodeTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPointwisePartitionLoweringTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuReferenceDifferentialTest.java`

Expected implementation documentation and planning paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- this task specification
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

Review-only paths include current Model unary/classification kinds, Tensor methods and tests,
DataType/ScalarValue contracts; Tensor/Compile/Backend/Public APIs; Config, Planning, Prepare, and
Runtime contracts; generated-artifact/specialization/route/executable sources and tests not listed
above; Gradle; architecture/ADR/tests; backend conformance; and integration tests.

## Maximum scope

This task may create or modify at most 30 paths:

| Category | Maximum | Path accounting |
|---|---:|---|
| CPU production/package | 16 | The exact sixteen listed production/package paths |
| CPU tests | 9 | The exact nine listed test paths |
| Explanatory documentation | 2 | CPU backend guide and glossary |
| Planning/status | 3 | This task, CPU master plan, and roadmap |
| **Total** | **30** | **16 + 9 + 2 + 3** |

The path count exceeds the usual guardrail for one documented technical reason: capability,
closed opcode/IR validation, lowering, scalar and vector generated emission, reference behavior,
schema compatibility, and their existing focused tests must evolve atomically before any unary
row is advertised. The implementation adds no production class and remains one cohesive semantic
family closure. Splitting a Draft 0005H1 would create an artificial partial unary matrix and
duplicate the same family-switch/schema work, so no follow-up split is planned.

The path map is exact. Unused authorized paths may remain unchanged, but no substitute or extra
path is implied. Stop and revise planning if another path or a new production type is required.

## Acceptance criteria

- Capability reports exactly all nineteen `UnaryElementwiseKind` values for same-typed,
  shape-preserving, fully static resolved FLOAT32/FLOAT64 occurrences and rejects every adjacent
  unsupported or malformed row before declarations or artifact work.
- `NEG` remains supported for both precisions; `GELU` expands from FLOAT64 to both precisions and
  returns negative zero at negative infinity; the other seventeen kinds are added without
  duplicate opcodes.
- `IS_FINITE`, `IS_NAN`, and `IS_INF` remain the existing three classification opcodes for
  FLOAT32/FLOAT64 to canonical BOOL and are not reimplemented as unary numeric operations.
- Scalar/reference algorithms, special values, signed zeros, domains, overflow/underflow, and
  tolerances match the matrices in this task. NaNs are classified without a payload promise.
- Stable sigmoid and SiLU avoid avoidable exponential overflow; both GELU variants and SiLU
  implement their continuous negative-infinity extension explicitly.
- The closed opcode inventory becomes exactly forty-eight values and schema 8 rejects every
  older envelope without migration.
- Scalar and parallel-scalar execute every final unary row for both types across all five access
  regimes, zero-element ranges, arbitrary start/end bounds, and heap/segment/mixed carriers.
- FLOAT64 vector and parallel-vector eligibility matches the exact table. No vector-ineligible
  opcode fails support; it selects scalar compute. No explicit per-lane scalar fallback loop or
  hardware-intrinsic claim is added.
- Every unary occurrence remains one Model node and one IR instruction. One-through-eight fusion,
  virtual intermediates, one final store, access/injectivity/alias rules, one-copy/four-candidate/
  one-artifact budgets, and zero fixed-shape/unroll variants remain unchanged.
- Cross-type CAST and BFLOAT16 execution remain fail-closed. Same-type CAST and all 31 pre-0005H
  opcodes retain their completed behavior.
- No public fast-math API, relaxed policy, graph rewrite, operation-specific class, native route,
  tuning work, dependency/build/shared-module/architecture change, or later detailed task appears.
- A separate clean documentation-focused pass finalizes affected Javadocs/package summaries, CPU
  guide, glossary, task evidence, master plan, and roadmap; records reasoned no-change conclusions;
  and reuses stabilized Java evidence unless it changes executable behavior.
- CPU 0005A–0005G remain `Complete`; CPU 0005H remains `Ready` until implementation, validation,
  documentation review, and final synchronization pass; CPU 0006–0017 remain `Draft` without
  detailed specifications.

## Tests / validation

Run focused implementation tests while executable Java changes:

```bash
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.CpuCapabilityProviderTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcodeTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIrTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPointwisePartitionLoweringTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuPointwiseGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuFusedGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.reference.CpuReferenceDifferentialTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStoreTest
```

The focused matrix must include every unary kind and both precisions; exact special values and
signed-zero bits; domain boundaries and adjacent `nextUp`/`nextDown` values; subnormal, normal,
overflow, and underflow cases; scalar and parallel-scalar; every eligible FLOAT64 vector and
parallel-vector row; vector-ineligible fallback selection; all five access regimes; representative
broadcast/strided input with injective output; zero elements; arbitrary subranges; heap, segment,
and mixed carriers; fused chains; schema incompatibility; and fail-before-artifact rejections.

After executable Java stabilizes, run exactly one final affected-module suite:

```bash
./gradlew :backends:cpu:test
```

The ordinary suite keeps the opt-in CPU 0005D persistence timing method skipped. Do not rerun the
timing evidence because this task changes neither the persistence verdict nor performance policy.

The separate documentation-focused pass, after final Javadocs and Markdown, runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

It also records checks for local Markdown links and heading anchors, balanced fences, final
newlines, trailing whitespace, exact authorized paths and the 30-path ceiling, nineteen-kind/
three-classification/forty-eight-opcode inventories, type/route/special-value/tolerance matrices,
schema 8, unchanged fusion/access/materialization budgets, fail-closed CAST/BFLOAT16 rows, and
synchronized status. It confirms no Java/test path outside `backends/cpu`, Gradle,
`ARCHITECTURE.md`, focused architecture/ADR/test paths, backend-conformance, or integration path
changed.

Repository-wide validation is deferred to CPU 0009, the portable generated-coverage checkpoint,
or continuous integration. Backend-conformance integration remains deferred to that checkpoint
because no composed Engine/backend harness exists at this frontier; the current task must provide
the equivalent CPU-internal generated/reference/independent-oracle conformance matrix now.

## Dependencies

- Complete CPU 0005A–0005G partition lowering, access/carrier, strategy, materialization,
  pointwise-family, exact power, extrema/clamp, logical, and schema-7 foundations.
- Complete Model 0018P final unary vocabulary, 0018T1 numeric gaps/classifications, and 0019A
  activation formulas and special-value extensions.
- Current Model `UnaryElementwiseKind`, `FloatingClassificationKind`, Tensor unary/classification
  construction, exact DataType, Shape, and resolved-layout contracts.
- Current Java 26 `StrictMath`/`Math`, Class-File API, and incubating Vector API contracts and the
  CPU module's existing `jdk.incubator.vector` configuration.
- Current staged Prepare, Runtime cold-binding/execution, and backend-contract ownership.

## Follow-up tasks

- CPU 0006 remains Draft without a detailed specification and follows this completed unary
  closure for layout, indexing, ordering, and random family coverage.
- CPU 0007–0008C remain Draft for reductions/normalization, heavy portable families, and later DAG
  decomposition/recognition/profitability. CPU 0009 remains the portable closure checkpoint.
- Config 0006 and CPU 0017 remain Draft for future explicit relaxed numerical permission. Nothing
  in this task preauthorizes those candidates.
- Cross-type CAST requires a future completed Model conversion contract before any CPU task may
  advertise it. No CAST follow-up specification is created here.
- No CPU 0005H1 task is needed: the exact 30-path atomic family evolution is documented and adds
  no production class.

## Architecture impact

Expected impact: None.

The task implements existing backend-owned lowering, numerical realization, strategy selection,
artifact compatibility, and executable construction. It changes no owner, dependency, lifecycle,
backend identity, public surface, or shared IR. If implementation requires any such change, stop
and report the exact conflict.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, the CPU master plan, completed
CPU tasks 0005A–0005G, and
docs/planning/backends/cpu/tasks/0005h-portable-unary-transcendental-and-activation-closure.md in
full. Read every affected and directly relevant Model/Compiler/Config/Prepare/Runtime contract
named by the task before editing.

Implement CPU 0005H exactly inside its exact 30-path map. Close all nineteen Model unary kinds for
same-typed fully static resolved FLOAT32/FLOAT64 occurrences through the existing capability,
opcode/IR, lowering, scalar/vector emitter, reference, prepare, and schema pipeline. Preserve the
already-complete classification rows and all earlier pointwise behavior. Use only the specified
algorithms, special-value rules, tolerances, and honest Java 26 FLOAT64 vector eligibility. Keep
cross-type CAST and BFLOAT16 fail-closed. Do not add fast-math APIs or policy, per-operation
classes, explicit vector lane scalarization, graph rewrites, fixed-shape/unrolled/DAG fusion,
native/tuning work, public/shared/build/architecture changes, later task specs, commits, or pushes.
Stop on a semantic, architecture, dependency, affected-file, or maximum-scope conflict.

After executable Java stabilizes and the focused plus exactly one final CPU suite pass, hand the
actual diff and exact Java evidence to a separate clean documentation-focused agent/thread. That
pass must follow docs/developer-guide/documentation-rules.md; independently finalize affected
Javadocs/package summaries, CPU guide, glossary, planning evidence, links, scope/status checks,
and reasoned no-change conclusions; and not repeat successful Java tests unless executable
behavior changes or a concrete stale-evidence risk is recorded.

Update this task's decisions, limitations, evidence, implementation notes, completion summary,
and synchronized final status. Do not mark Complete until every criterion and the documentation
pass succeed. Leave CPU 0006 and every later CPU task Draft without a detailed specification.
```

## Local decisions

- Close the whole remaining unary family in 0005H. The work is one atomic evolution of existing
  family switches and schema, uses no new production class, and is safer than a partial 0005H1
  split that would duplicate compatibility work.
- Scalar FLOAT64 uses Java 26 `StrictMath` plus the existing ERF approximation and stable
  activation helpers. FLOAT32 uses exact represented widening and one final narrowing where the
  JDK lacks a float method.
- GELU keeps one semantic exact-target opcode even though its portable finite realization is the
  existing bounded approximation. `GELU_TANH_APPROXIMATION` is a separate fixed semantic formula,
  not permission for general approximate math.
- Only direct Java 26 FLOAT64 lane operators, exact division/sqrt composition, and the existing
  ERF/GELU vector helper are vector-eligible. Missing operators and unproved activation
  compositions select scalar compute; no explicit per-lane fallback loop is allowed.
- Existing GELU negative-infinity NaN behavior is incompatible with the Model continuous
  extension and is corrected in the same closure task rather than preserved as historical output.
- Cross-type CAST remains unsupported because live Model contracts still do not define numerical
  conversion. Java primitive/vector conversion availability cannot supply Model semantics.
- Schema 8 is required by the opcode and generated-bytecode changes. No migration reader is
  justified.

## Known limitations

- Execution remains limited to fully static, resolved-layout occurrences inside the current
  one-through-eight connected straight-line pointwise topology.
- BFLOAT16 remains storage-only for this family, FLOAT16 is absent, and cross-type CAST remains
  unsupported.
- FLOAT32 and vector-ineligible FLOAT64 chains use scalar compute with optional parallel
  orchestration. Vector math-library operators carry no promise of a hardware intrinsic or
  performance improvement.
- ERF and activation-family finite results use the explicit tolerances in this task; no NaN
  payload or cross-platform bitwise promise is made.
- Optional generated-class persistence remains disabled by default. Native/vendor routes, tuning,
  Engine composition, shared backend conformance, and broader family coverage remain deferred.

## Validation evidence

Planning evidence before promotion to `Ready`:

- Read the architecture contract and focused backend/module/dependency/prepare/runtime documents,
  ADRs 0002/0008/0010/0011, documentation rules and General/Planning profiles, planning guide,
  roadmap, CPU master plan, completed CPU task records through 0005G, relevant Model tasks and
  live unary/classification/Tensor/DataType/ScalarValue contracts, CPU production/tests/Javadocs,
  public API/glossary text, and Java 26 Gradle configuration.
- Audited the live Model inventory: nineteen unary kinds and three floating classifications.
  Audited the live CPU inventory: thirty-one opcodes; `NEG` for FLOAT32/FLOAT64; `GELU` for
  FLOAT64; all three classifications for FLOAT32/FLOAT64; same-type CAST only.
- Audited OpenJDK 26.0.1's installed `VectorOperators` surface and source documentation. It exposes
  the named floating lane operators and states equivalent Java accuracy/monotonicity, while making
  no hardware-intrinsic promise. It exposes no selected FLOOR, CEIL, or SIGN lane operator.
- Re-audited Model CAST contracts and found no conversion/rounding/saturation/BOOL policy; the
  fail-closed decision remains required.
- The exact expected implementation map is 30 paths with no new production class. A Draft 0005H1
  would be a mechanical split rather than a coherent capability, so it was not created.
- `ruby /tmp/validate_0005h_markdown.rb` passed for this task, the CPU master plan, and the
  roadmap, checking repository-local links and anchors, balanced fences, final newlines, and
  trailing whitespace.
- `git diff --check` passed for tracked edits. `git diff --no-index --check /dev/null <task-file>`
  produced no whitespace diagnostic; its status was the expected `1` because the task is new.
- Final status/scope inspection found exactly the three permitted planning paths, CPU 0005H
  `Ready` in the task/master/roadmap, CPU 0006–0017 still Draft without detailed specifications,
  no 0005H1 specification, and no Java, test, Gradle, architecture, ADR, conformance, integration,
  API, guide, or glossary change in this planning diff.

Implementation evidence:

- Clean implementation context `/root/cpu_0005h_impl` changed only the authorized CPU Java/test
  and planning paths. `./gradlew :backends:cpu:compileJava` and
  `./gradlew :backends:cpu:compileTestJava` passed.
- After the post-completion FLOAT32 RSQRT operation-order correction, the required focused
  nine-class command passed 43 tests with zero failures, errors, or skips. Exactly one new final
  `./gradlew :backends:cpu:test` passed 25 suites and 108 tests with zero failures or errors
  and one expected opt-in `CpuGeneratedKernelPersistenceEvidenceTest` skip. The persistence timing
  evidence remained disabled. This evidence supersedes the earlier 42-test/107-test results;
  `git diff --check` passed after implementation restabilization.
- Clean documentation context `/root/cpu_0005h_docs` inspected the final source/tests and the 25
  XML suite artifacts under `backends/cpu/build/test-results/test`. The artifacts independently
  confirm 108 tests, zero failures/errors, and the one expected skip; the nine named focused
  classes contain 43 recorded test cases. The reopened documentation pass inspected the corrected
  emitter and raw-bit regression, reused the new stabilized test evidence, and did not repeat Java
  test suites; the required Javadoc task retained its normal Gradle compilation dependency.

Documentation-focused evidence:

- Context `/root/cpu_0005h_docs` applied the General, API/Javadoc, Backend Guide, and Planning
  profiles. It independently reviewed the architecture contract, documentation and planning
  rules, CPU plan/task history and final diff, all affected production/test contracts, the CPU
  guide, glossary, public Model unary/classification/CAST boundaries, and generated test artifacts.
- Reviewed the affected provider, schema, and opcode Javadocs and retained their accurate
  implementation drafts. Finalized the generator/emitter, IR, lowering, reference, and package
  contracts. Package-private vector helpers now document their inputs/results, the
  orphaned reciprocal comment was removed from the polynomial helper, and the existing implicit
  public generator/lowering constructors were declared explicitly only to document their unchanged
  stateless construction contracts. Those documentation-only edits changed no executable semantics.
- Reopened the pass after the executable correction and documented that FLOAT32 RSQRT widens its
  represented input, performs both `StrictMath.sqrt` and the reciprocal in FLOAT64, and narrows
  only the final result. The raw-bit regression at input `0x000026f6` distinguishes the obsolete
  intermediate-root result `0x616801ad` from the required final result `0x616801ae`.
- Finalized `docs/backend-guide/cpu-backend.md` and `docs/glossary.md` for all nineteen unary kinds,
  three separate classifications, the exact FLOAT32/FLOAT64 scalar and FLOAT64 vector matrices,
  special values and tolerances, schema 8, exactly 48 opcodes, scalar fallback, and fail-closed
  cross-type CAST/BFLOAT16. Finalized this task, the CPU master plan, and the roadmap at `Complete`.
- `./gradlew :backends:cpu:javadoc` passed after final Javadocs. Its only two warnings report the
  expected incubating `jdk.incubator.vector` module; there are no missing-Javadoc warnings.
- `ruby /tmp/validate_0005h_markdown.rb` passed for the CPU guide, glossary, this task, master plan,
  and roadmap, checking repository-local links and heading anchors, balanced fences, final
  newlines, and trailing whitespace. `git diff --check` passed on the final combined change.
- Final semantic/scope checks confirmed nineteen Model unary kinds, nineteen unary opcodes, three
  classification kinds/opcodes, exactly 48 total opcodes, schema 8, the documented scalar/vector
  eligibility and exceptional-value/tolerance matrices, unchanged one-through-eight fusion/five-
  access-regime/one-copy/four-candidate/one-artifact/zero-fixed-shape/zero-unroll budgets, and
  fail-closed cross-type CAST/BFLOAT16.
- Final path/status checks found exactly 26 changed paths inside the exact 30-path map: 15 CPU
  production/package paths, 6 CPU test paths, 2 explanatory documents, and 3 planning/status
  paths. No Java/test path outside `backends/cpu`, Gradle/build, `ARCHITECTURE.md`, architecture or
  ADR/test, backend-conformance, integration, public/shared API, or later-task specification changed.
  CPU 0005A–0005H are `Complete`; CPU 0006–0017 remain `Draft` without detailed specifications;
  no 0005H1 or later detailed CPU task exists.

## Implementation notes

- Closed all nineteen Model unary kinds through the existing provider, opcode/IR, lowering,
  generated scalar/vector, reference, preparation, and schema pipeline without adding a type,
  package, route, public API, shared-module change, or operation decomposition.
- Added seventeen distinct unary opcodes while retaining `NEG`, `GELU_EXACT`, and the three
  classification opcodes. `GELU` now covers FLOAT32 and maps negative infinity to negative zero in
  scalar/reference and vector realization; schema 8 invalidates every older envelope.
- Stable sigmoid, fixed tanh-approximation GELU, and SiLU reuse pure primitive reference helpers
  from generated scalar bytecode. FLOAT32 widening/narrowing and FLOAT64 vector eligibility follow
  the fixed task matrix; FLOAT32 RSQRT performs its square root and reciprocal before the single
  final narrowing, and vector-ineligible admitted chains select scalar compute.
- Preserved the complete pre-0005H opcode behavior, one-through-eight topology, virtual
  intermediates, access/carrier/materialization/artifact budgets, same-type CAST, and fail-closed
  cross-type CAST/BFLOAT16.

## Completion summary

- Completed changes: closed the full FLOAT32/FLOAT64 unary family in the existing CPU portable
  pipeline, advanced generated compatibility to schema 8, corrected GELU negative infinity and
  FLOAT32 RSQRT operation order, and synchronized Javadocs, package summaries, guide, glossary,
  task, master plan, and roadmap.
- Files changed or created by the documentation pass: `CpuClassFileKernelGenerator.java`,
  `CpuScalarEmitter.java`, `CpuVectorEmitter.java`,
  `CpuKernelIr.java`, `CpuPartitionLowering.java`, `CpuScalarReferenceKernel.java`, the six affected
  CPU `package-info.java` files, `docs/backend-guide/cpu-backend.md`, `docs/glossary.md`, this task,
  `docs/planning/backends/cpu/master-plan.md`, and `docs/planning/roadmap.md`.
- Correction-specific documentation changes: the `CpuScalarEmitter.java` class Javadoc, emit
  `package-info.java`, CPU backend guide, this task, CPU master plan, and roadmap. The glossary and
  all other affected Javadocs/package summaries were reviewed and remain accurate without another
  correction-specific edit.
- Combined changed paths: the fifteen production/package paths
  `CpuCapabilityProvider.java`, `CpuGeneratorSchema.java`, `CpuClassFileKernelGenerator.java`,
  `CpuScalarEmitter.java`, `CpuVectorEmitter.java`, `CpuKernelIr.java`, `CpuPointwiseOpcode.java`,
  `CpuPartitionLowering.java`, `CpuScalarReferenceKernel.java`, and root/emit/IR/lowering/reference/
  prepare `package-info.java`; six tests `CpuCapabilityProviderTest.java`,
  `CpuGeneratedKernelArtifactStoreTest.java`, `CpuPointwiseGeneratedKernelTest.java`,
  `CpuPointwiseOpcodeTest.java`, `CpuPointwisePartitionLoweringTest.java`, and
  `CpuReferenceDifferentialTest.java`; the CPU guide and glossary; and this task, CPU master plan,
  and roadmap. Total: 26 authorized paths.
- Tests and validation: reused implementation context `/root/cpu_0005h_impl` compile/focused
  43-test/final 108-test evidence after inspecting the final XML artifacts; documentation context
  `/root/cpu_0005h_docs` passed CPU Javadoc, five-file Markdown/link/anchor/formatting validation,
  exact semantic/scope/status checks, and `git diff --check`.
- Documentation-agent review: `/root/cpu_0005h_docs`; reopened after the executable correction,
  complete, and no Java test suite was repeated by the documentation pass.
- Documentation impact: CPU guide, glossary, task, master plan, and roadmap finalized.
- Javadoc review: affected production and package contracts finalized; public Model, Compiler,
  Config, Planning, Prepare, Runtime, Backend/Public API Javadocs remain accurate because this task
  changes only CPU-internal realization and truthful CPU capability coverage.
- Glossary impact: current CPU portable route, specialization, artifact, IR, execution strategy,
  and preparation-boundary descriptions reflect the unary closure and schema 8. No additional
  correction-specific glossary text is needed because the glossary defines Model semantic identity
  and CPU vector eligibility, while the backend guide owns the private FLOAT32 rounding sequence.
- No-change conclusions: architecture/ADRs/tests and dependency rules remain unchanged because
  ownership and module boundaries did not move; Gradle/build remains unchanged because Java 26 and
  Vector configuration already existed; backend-conformance and integration remain deferred to
  CPU 0009 because no composed Engine/backend harness exists; all unused authorized package,
  prepare, IR, fused-kernel, and test paths remain accurate because existing typed facts and
  regression suites already cover their unchanged contracts. Model/Compiler/Config/Planning/
  shared Prepare/Runtime/public APIs need no change because Model already owns the semantic kinds
  and CPU 0005H only realizes them behind existing backend boundaries.
- Unresolved issues: None.
- Follow-up required: None. CPU 0006 remains the next Draft row and has no detailed specification.

Status: Complete
