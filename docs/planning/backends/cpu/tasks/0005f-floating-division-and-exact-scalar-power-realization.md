# Task 0005F: Floating Division and Exact Scalar-Power Realization

## Status

Complete

## Goal

Add exact/default portable CPU execution for same-typed FLOAT32/FLOAT64 tensor division, scalar
division, and scalar `POW` without rewriting the compiled or Model graph. Binary and scalar
division lower as ordinary family operations with exact result type and existing right-aligned
broadcast or Shape-preserving access. CPU lowering keeps each semantic
`ScalarElementwiseKind.POW` occurrence and its exact same-typed `ScalarValueAttrs` exponent. One
CPU-private, route-independent analysis classifies that exponent once before code generation and
selects one of five realization plans:

```text
supported DIV | SCALAR_DIV | semantic SCALAR_POW + exact typed exponent
  -> primitive division | direct power | positive one | identity | square | reciprocal
  -> existing scalar/vector and single/parallel portable machinery when eligible
  -> one compatible generated artifact and one partition executable
```

The exact selected matrix is FLOAT32 and FLOAT64. Division uses the ordinary primitive
left-divided-by-right or input-divided-by-scalar operation. Every admitted exponent has a direct
portable power fallback; exact typed positive or negative zero, positive one, positive two, and
negative one may use the narrower realizations below only because each has a complete
default-conformance proof. Other integral exponents do not use multiply chains or exponentiation
by squaring: their intermediate rounding, overflow, and underflow can differ from one semantic
power evaluation, and the current contracts supply no universal proof that permits that change.

This is backend realization selection, not graph optimization. The compiled node, operation kind,
attributes, graph phase, value identity, publication obligations, and compiler-generated gradient
expressions remain unchanged.

## Scope

### Supported division and power matrix

- Extend the existing pointwise matrix with `BinaryArithmeticKind.DIV` only when:
  - the operation has exactly two inputs and one output;
  - attributes are exactly `NoOperationAttrs.INSTANCE`;
  - both inputs and the output have the same exact data type, FLOAT32 or FLOAT64;
  - the output Shape is the ordinary right-aligned broadcast of the two input Shapes; and
  - every Shape is fully static and every layout is resolved, as required by CPU 0005E.
- Extend the matrix with `ScalarElementwiseKind.DIV` only when:
  - the operation has exactly one input and one output;
  - attributes are exactly `ScalarValueAttrs`;
  - input, output, and scalar denominator have the same exact data type, FLOAT32 or FLOAT64;
  - output Shape equals input Shape; and
  - every Shape is fully static and every layout is resolved.
- Retain the scalar denominator's exact typed primitive bits in canonical CPU IR. Do not widen a
  FLOAT32 semantic attribute, create a graph input, inspect Tensor storage, or infer a value from
  Tensor factory history.
- Extend the existing pointwise matrix with `ScalarElementwiseKind.POW` only when:
  - the operation has exactly one input and one output;
  - attributes are exactly `ScalarValueAttrs`;
  - input, output, and scalar exponent have the same exact data type;
  - that data type is FLOAT32 or FLOAT64;
  - output Shape equals input Shape; and
  - every Shape is fully static and every layout is resolved, as required by CPU 0005E.
- Retain the exponent's exact typed primitive bits in canonical CPU IR. Do not convert a FLOAT32
  exponent into a FLOAT64 semantic attribute, create another graph input, inspect Tensor storage,
  or infer a constant from Tensor factory history.
- Classify the represented exponent exactly once during common CPU lowering:

  | Exact typed exponent | Selected realization | Required computation |
  |---|---|---|
  | positive or negative zero | `POSITIVE_ONE` | store exact positive typed `1.0` without reading the base |
  | positive one | `IDENTITY` | forward the represented base value |
  | positive two | `SQUARE` | one correctly rounded primitive multiply in the result type |
  | negative one | `RECIPROCAL` | one correctly rounded primitive division of positive typed one by the base |
  | every other finite value, infinity, or NaN | `DIRECT` | the direct exact/default power realization |

- Exact classification uses typed represented values, not decimal text, widening-dependent
  equality, integer casts, string dispatch, or a tolerance. Distinct NaN payloads remain exact
  metadata and all select `DIRECT`; the Model makes no NaN-payload result promise.
- FLOAT64 `DIRECT` delegates to `StrictMath.pow(base, exponent)`. FLOAT32 `DIRECT` exactly widens
  the two represented binary32 operands to binary64, invokes `StrictMath.pow`, and narrows the
  result once to FLOAT32. This internal computation does not create cross-type scalar semantics:
  the Model input, attribute, IR value, and output all remain exact FLOAT32.
- BFLOAT16 remains fail-closed even though Model permits floating division and scalar power and CPU
  storage can represent two-byte values. CPU 0005E established no BFLOAT16 arithmetic route.
  FLOAT16 remains absent pending Model 0026. Integral and BOOL division or scalar `POW` are outside
  the Model expression contract, and mismatched floating scalar types remain invalid.
- Capability reporting returns `false` for every excluded row. Complete-partition lowering rejects
  an unsupported or malformed occurrence before resource declarations, specialization, artifact
  lookup, or executable construction. No excluded row falls back through a different type or
  route.

### Exact/default numerical contract and proofs

- Binary DIV evaluates the ordered primitive operation `left / right` in the exact FLOAT32 or
  FLOAT64 result type. Scalar DIV evaluates `input / denominator` in that same exact type. Both
  preserve ordinary Java/IEEE-754 default behavior for NaN, infinities, signed zero,
  divide-by-zero, overflow, underflow, and subnormal inputs and results. Division by floating zero
  does not add an integer-style exception. The task promises no NaN payload or bitwise result
  beyond results fixed by the primitive operation contract.
- Division permits no reassociation, reciprocal replacement, estimate, finite-only assumption,
  flush-to-zero, contraction, denominator normalization, or relaxed exception rule. In particular,
  public/scalar DIV is not implemented as multiplication by a reciprocal.
- Exact/default means the ordinary scalar-power contract with no relaxed permission,
  reassociation, approximation, finite-only assumption, reciprocal estimate, flush-to-zero,
  contraction, or altered exceptional-value rule. The direct oracle follows the current Java 26
  `StrictMath.pow` special cases, finite accuracy, and semi-monotonicity contract; FLOAT32 applies
  the one specified final narrowing.
- `POSITIVE_ONE` is legal for both exponent signs of zero because the direct contract returns
  positive typed one for every base, including NaN, positive or negative zero, and either
  infinity. This is a CPU realization plan attached to semantic `POW`, not Compiler 0007's future
  graph-level `POW(0)` replacement. It creates no logical splat, graph value, node bypass, or
  publication remapping.
- `IDENTITY` is legal only for exact positive one. The direct contract returns the base, so the
  plan preserves finite values, subnormals, infinities, and both zero signs. NaN classification is
  preserved; no NaN-payload promise is added. Compiler 0003A may already remove some guarded
  internal occurrences, but graph-output, gradient-eligible, optimization-disabled, or otherwise
  retained occurrences remain valid CPU inputs.
- `SQUARE` is legal only for exact positive two. One primitive multiply is the correctly rounded
  operation for the exact mathematical square in FLOAT32 or FLOAT64. It produces positive zero
  from either zero sign, positive infinity from either infinity, NaN from NaN, and the expected
  nonnegative finite result. Overflow, subnormal production, and underflow occur through that one
  typed multiply; there is no intermediate reassociation or extra rounding.
- `RECIPROCAL` is legal only for exact negative one. One primitive typed `+1.0 / base` implements
  the exact mathematical reciprocal with one rounding. It preserves the required signs for zero
  and infinity, propagates NaN classification, and exposes overflow, subnormal production, and
  underflow through the ordinary typed division. This realization remains semantically
  `SCALAR_POW`; it may share low-level division emission but must not become binary or scalar DIV.
- The conformance proof for each special plan must be documented in the analysis Javadoc and
  locked by generated/reference and direct-oracle edge tests. Tests may require exact bits where
  the contract fixes them, including positive-one results and signed-zero/infinity results. They
  must compare NaN by classification and finite direct-power results under the specified direct
  accuracy contract rather than inventing a payload or universal bitwise promise.
- `POW(0.5)` remains `DIRECT` and is never replaced by `SQRT`. Negative bases, signed zero,
  infinities, NaNs, domain behavior, and rounding make the intuitive identity insufficient.
- Positive or negative integral exponents other than `0`, `1`, `2`, and `-1` remain `DIRECT`.
  Repeated multiplication, multiply chains, reciprocal-plus-chain forms, and exponentiation by
  squaring introduce one or more rounded intermediate results and may change finite rounding,
  overflow, underflow, and subnormal transitions. Randomized agreement is not a proof. Reject
  these plans until a later contract establishes a universal error and exceptional-value proof.
- The same matrix applies to forward and compiler-generated gradient occurrences. CPU adds no
  derivative-specific realization or numerical permission.

### Common analysis, IR, and code generation

- Add one `CpuScalarPowerAnalysis` in `internal.lowering`. It is stateless and classifies one
  already-validated exact typed exponent into the closed realization enum. It does not inspect a
  graph, choose a route, access Config, cache decisions, or evaluate Tensor storage.
- Add family-oriented `DIV`, `SCALAR_DIV`, and `SCALAR_POW` values to
  `CpuPointwiseOpcode`, growing the closed vocabulary from nineteen to exactly twenty-two opcodes,
  rather than adding per-operation or per-exponent planner, instruction, emitter, executable, or
  route classes. Binary and scalar DIV use the existing arithmetic-family validation with no
  power-realization fact.
- Extend `CpuKernelIr.Instruction` with one optional/nested scalar-power realization fact. Exactly
  `SCALAR_POW` requires both the exact scalar immediate and one realization; other scalar opcodes
  require an immediate and forbid a power realization; parameterless opcodes forbid both. IR
  validation must recheck the opcode/type/immediate/realization relationship fail-closed.
- Keep the semantic scalar immediate even for `POSITIVE_ONE`, `IDENTITY`, `SQUARE`, and
  `RECIPROCAL`. The plan supplements the semantic operation; it does not erase or replace it.
- Extend the existing scalar emitter and scalar reference kernel through their family-grouped
  dispatch for both DIV forms and every selected power realization. Code generation consumes the
  selected realization directly and performs no exponent comparison or policy lookup in generated
  code.
- Scalar and parallel-scalar realize binary DIV, scalar DIV, and scalar POW for both FLOAT32 and
  FLOAT64. The existing FLOAT64 Vector emitter may realize DIV and the four special power plans
  only when the opcode and every existing access-run, carrier, topology, and strategy eligibility
  requirement passes. `IDENTITY`, `SQUARE`, and `RECIPROCAL` use direct vector identity,
  multiply, or divide operations; `POSITIVE_ONE` produces a typed one vector without loading the
  base. `DIRECT` power is vector-ineligible and deterministically selects scalar or
  parallel-scalar compute. FLOAT32 remains scalar or parallel-scalar under CPU 0005E's existing
  FLOAT64-only vector boundary.
- Code generation must not call `Math.pow` or `StrictMath.pow` per vector lane, scalarize a
  vector body secretly, use a reciprocal approximation, or grow one code path per exponent.
- The scalar reference remains a CPU-private conformance oracle over already-selected IR facts;
  it must not interpret Model operations in Runtime or become a second lowering compiler.

### Fusion, access, carriers, materialization, and execution

- Preserve CPU 0005E's one-to-eight connected straight-line pointwise partition. `DIV`,
  `SCALAR_DIV`, and semantic `SCALAR_POW` each count as exactly one ordinary pointwise instruction;
  DIV participates normally in legal connected fusion. The selected power realization never
  changes its instruction count. The task does not expand the fusion budget, create multiple
  units, or consume an additional graph node.
- Preserve stored node and instruction order. A special realization changes only the implementation
  of that one semantic instruction; it does not reassociate surrounding operations or fuse a
  multiply chain across node boundaries.
- Preserve all five `CpuAccessPlan` regimes, right-aligned broadcasting, arbitrary start/end
  ranges, offsets, positive and zero strides, injective writes, overlap checks, and heap/
  `MemorySegment`/mixed carriers. Binary DIV uses ordinary broadcast access; scalar DIV and scalar
  POW preserve Shape and add no scalar boundary.
- Preserve CPU 0005D's direct plus at most three eligible one-input materialization candidates,
  at most one selected FLOAT64 input copy, four-complete-candidate ceiling, one-artifact budget,
  zero fixed-shape variants, and zero unrolled variants. `POSITIVE_ONE` may avoid reading the base
  in generated computation, but it does not alter the already-derived graph boundary,
  declaration, alias, or materialization accounting in this task.
- Preserve scalar, vector, parallel-scalar, and parallel-vector strategy names and worker/chunk
  behavior. A partition containing `DIRECT` power or any FLOAT32 operation in this task uses
  scalar compute with the same optional parallel orchestration; supported work must not fail
  merely because vector compute is unavailable. FLOAT64 DIV or special power uses vector or
  parallel-vector only through the current eligibility route.
- Preserve the existing exact FLOAT32/FLOAT64 heap carriers (`float[]`, `double[]`), native-order
  segment access, representative mixed patterns, cold binding, one partition executable,
  concurrent-run isolation, and direct Runtime invocation.

### Specialization, fingerprints, manifests, and artifact compatibility

- Canonical IR structural identity must include `DIV` or `SCALAR_DIV` where present, the exact
  scalar-DIV denominator type and raw bits, and, for `SCALAR_POW`, exact exponent type/raw bits and
  selected power realization. Changing any of those facts changes the lowering fingerprint even
  when two instructions would currently emit similar bytecode.
- The selected `PowerRealization` must also be an explicit validated code-shaping fact in the
  generated specialization/compatibility bytes and optional cold lowering manifest. Do not rely
  only on an opaque digest to prove compatibility.
- Bump the generator schema because the opcode vocabulary, IR instruction shape, scalar/vector
  emission, and compatibility facts change. Current-schema stored hits remain eligible; every old
  schema entry is an incompatible miss. Add no migration reader.
- Preserve process-local compatible interning, complete structural verification, optional bounded
  current-schema persistence, assignment-before-artifact order, and strong prepared-executable
  ownership. CPU 0005D's `KEEP_DISABLED` default remains unchanged; this task adds no benchmark or
  persistence-policy claim.
- Exact/default remains the only `CpuKernelSpecialization.NumericalMode`. Do not add a broad
  numerical-policy framework, registry, service locator, Config projection, generic parameter
  map, or another compiler. Config 0006 and CPU 0017 remain the only planned relaxed-permission
  path.

### Tests and documentation

- Add table-driven capability/lowering tests for both DIV forms, every admitted type and power
  plan, and representative adjacent rejection: BFLOAT16, INT32/INT64, BOOL, mismatched scalar
  type, wrong attributes, unresolved layouts, dynamic Shapes, and unsupported topology.
- Add generated/reference differential division tests for ordinary and broadcast binary inputs and
  exact scalar denominators across NaN, infinities, both zero signs, divide-by-zero, overflow,
  underflow, subnormal transitions, finite extrema, and representative ordinary values.
- Add generated/reference/direct-oracle differential tests for FLOAT32 and FLOAT64 over exponent
  `+0`, `-0`, `+1`, `+2`, `-1`, representative `+3`/`-2`, fractional values including `0.5`,
  subnormal/large finite exponents, positive/negative infinity, and multiple NaN bit patterns.
- For each exponent plan, cover bases `+0`, `-0`, smallest/largest subnormals, minimum normals,
  representative positive and negative finite values around one, finite extrema, both infinities,
  and multiple NaNs. Include values that force overflow, underflow, and subnormal results.
- Cover scalar/parallel-scalar for both types and scalar/vector/parallel-scalar/parallel-vector for
  eligible FLOAT64 DIV and special power plans. Prove `DIRECT` power and FLOAT32 select scalar
  compute without losing parallel orchestration.
- Reuse all access/carrier/materialization/fusion/artifact/executable suites. Add representative
  scalar, zero-element, all five access regimes, arbitrary ranges, all-heap, all-segment, and mixed
  carrier cases without constructing an exponential type/exponent/carrier Cartesian product.
- Lock one-through-eight fusion accounting, one artifact, no stale schema or realization-plan
  cache hit, exact manifest/fingerprint differences, and fail-before-artifact behavior.
- A separate clean documentation-focused agent must finalize affected CPU Javadocs/package
  summaries, the CPU backend guide, glossary, this task, CPU master plan, and roadmap. It must
  review Tensor, Compile, Config, Runtime, and Prepare API documentation and record reasoned
  no-change conclusions where their current contracts remain accurate.

## Out of scope

- Tensor/Tensor `BinaryArithmeticKind.POW`, Tensor exponents, compiler constant discovery, or
  inference of uniform Tensor values from host storage or factory history.
- Compiler graph rewrites, constant folding, a graph-level `POW(0)` logical splat, changes to the
  completed guarded `POW(+1)` bypass, phase/publication remapping, or any new Compiler pass.
- Multiply chains or exponentiation by squaring for exponents beyond positive two; reciprocal
  chains for exponents below negative one; `POW(0.5) -> SQRT`; or another unproved algebraic
  identity.
- BFLOAT16/FLOAT16 power, integral power, mixed-type scalar values, cross-type promotion, a new
  public Tensor method, a new Model operation kind, or changes to Model scalar semantics.
- Binary/scalar MIN and MAX, scalar CLAMP, remaining unary/transcendental/activation and
  boolean-logical families, cross-type CAST, and other pointwise coverage assigned to Draft CPU
  0005G.
- Relaxed/fast math, Config 0006, CPU 0017, finite-only assumptions, approximate reciprocal,
  vector `DIRECT` power, another Vector species, gather, masked tails, fixed shapes, or unrolling.
- Native/vendor/OpenBLAS routes, tuning or tuning-cache work, persistence-policy enablement,
  benchmarks, Engine composition, backend-conformance integration, or end-to-end execution claims.
- Changes to Model, Compiler, Config, Planning, shared Prepare, Runtime, Backend Contract, Trace,
  Engine, OpenBLAS provider, another backend, Gradle, dependencies, `ARCHITECTURE.md`, focused
  architecture documents, ADRs, architecture tests, `testing/backend-conformance`, or
  `testing/integration-tests`.
- Creating a detailed CPU 0005G or later task specification, committing, or pushing.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially core invariants, Compiler,
  Prepare, Runtime, concrete backend ownership, CPU routes, and dependency rules.
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md).
- [Runtime, Prepare, and Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md).
- [Performance evidence and tuning](../../../../architecture/performance-evidence-and-tuning.md).
- [Planning Guide](../../../planning-guide.md).
- [CPU backend master plan](../master-plan.md).
- [Completed CPU 0005A](0005a-atomic-partition-kernel-architecture-reset.md).
- [Completed CPU 0005B](0005b-universal-access-plans-and-right-aligned-broadcasting.md).
- [Completed CPU 0005C](0005c-vector-and-parallel-portable-strategies.md).
- [Completed CPU 0005D](0005d-materialization-specialization-and-persistence-evidence-gate.md).
- [Completed CPU 0005E](0005e-portable-pointwise-types-carriers-and-semantic-family-expansion.md).
- [Compiler 0003A exact arithmetic rewriting](../../../modules/compiler/tasks/0003a-exact-arithmetic-rewriting.md).
- [Compiler 0003B compile-time constants](../../../modules/compiler/tasks/0003b-compile-time-constants-and-constant-folding.md).
- [Config master plan](../../../modules/config/master-plan.md).
- [Tensor API scalar arithmetic](../../../../api/tensor-api.md#scalar-arithmetic-and-clamp-expressions).
- [Compile API forward optimization](../../../../api/compile-api.md#current-package-private-canonicalization-and-whole-graph-optimization).
- [Glossary: Scalar-power realization](../../../../glossary.md#scalar-power-realization).

## Architecture constraints

- Model `Operation` remains the semantic owner. CPU maps supported binary DIV, scalar DIV, and
  scalar `POW` occurrences into private typed IR, adding a realization plan only for scalar power,
  without changing the graph or publishing backend semantics through Model.
- Planning selects only `BackendId("cpu")`. CPU analysis owns exact/default division eligibility,
  exponent classification, pointwise lowering, bounded fusion, access/materialization planning,
  strategy and route selection, specialization, and exact declarations.
- Shared Prepare receives an opaque selected CPU plan and exact resource requirements. It must not
  inspect an exponent, power plan, opcode, route, or numerical policy.
- Finalization occurs after slot assignment and cannot change the exponent, realization, IR,
  strategy, carrier pattern, materialization, or declared resources.
- Runtime receives one immutable prepared executable and cold-bound invocation. No operation,
  compiled node, exponent classification, IR, policy, route, or artifact lookup reaches its hot
  path.
- The implementation changes no module boundary, dependency direction, lifecycle stage, backend
  identity, or supported public CPU type. If implementation needs a new Model numerical decision,
  shared contract, another module, build edge, or architecture change, stop and report it.

The architecture audit conclusion is `None`: backend-private realization selection for a retained
semantic operation is already owned by concrete backend prepare.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu` — truthful occurrence-local capability reporting.
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — common route-independent pointwise
  division lowering and scalar-power exponent classification.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — division and power opcodes, exact scalar
  immediates, selected power realization, structural validation, and fingerprint input.
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — scalar and eligible FLOAT64
  vector realization from already-selected facts.
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — scalar conformance oracle.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — strategy selection and cold manifest.
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — schema, specialization, compatibility,
  verification, and optional persistence.
- Existing memory, portable-route, and executable packages are reviewed and changed only where the
  exact plan must flow through an existing contract.

Packages added or changed:

- No package is added and no responsibility moves.

Type placement:

- `io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScalarPowerAnalysis` — one new
  stateless classifier colocated with common route-independent lowering because it consumes exact
  semantic attributes before route/code generation.
- `CpuKernelIr.PowerRealization` — nested closed enum because it is one validated code-shaping fact
  of a scalar-power IR instruction, not a registry, policy framework, or public configuration.
- `CpuPointwiseOpcode.DIV`, `SCALAR_DIV`, and `SCALAR_POW` — the existing family opcode vocabulary
  owns the additional semantic instructions; no operation-specific class hierarchy is introduced.

## Affected files

Expected CPU production/package paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecialization.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScalarEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuVectorEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuKernelIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPointwiseOpcode.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuScalarPowerAnalysis.java` (new)
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuScalarReferenceKernel.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/CpuPortableRoutePlan.java`

Expected CPU test paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelPersistenceEvidenceTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuFusedGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPointwiseGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutableTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuKernelIrTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPointwiseOpcodeTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPointwisePartitionLoweringTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuScalarPowerAnalysisTest.java` (new)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuReferenceDifferentialTest.java`

Expected documentation and planning paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- `docs/planning/backends/cpu/tasks/0005f-floating-division-and-exact-scalar-power-realization.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

No other path is authorized by default.

## Maximum scope

This task may create or modify at most 40 paths:

| Category | Maximum | Path accounting |
|---|---:|---|
| CPU production/package | 21 | The exact 21 listed production/package paths |
| CPU tests | 14 | The exact 14 listed test paths |
| Explanatory documentation | 2 | CPU backend guide and glossary |
| Planning/status | 3 | This task, CPU master plan, and roadmap |
| **Total** | **40** | **21 + 14 + 2 + 3** |

The path map is exact; unused authorized paths may remain unchanged, but no substitute or
additional path is implied. It does not authorize another abstraction, package, operation family,
executor, route, or unrelated cleanup. If implementation needs another path, new production type,
module, Gradle/build change, or different package structure, stop and revise planning or report
the prerequisite.

## Acceptance criteria

- Capability reporting admits exactly same-typed FLOAT32/FLOAT64 binary DIV with ordinary
  right-aligned broadcast, scalar DIV with an exact matching `ScalarValueAttrs`, and scalar `POW`
  with an exact matching `ScalarValueAttrs`; Shapes are fully static and layouts resolved. Every
  adjacent excluded row remains unadvertised and fails before declarations/artifact work.
- Every admitted exact exponent deterministically selects exactly one of `DIRECT`, `POSITIVE_ONE`,
  `IDENTITY`, `SQUARE`, or `RECIPROCAL` once during common lowering. The exact matrix and proofs in
  Scope are implemented without tolerance classification or hot lookup.
- Semantic `POW`, exact scalar bits, graph phase/value/publication identity, and compiled graph
  topology remain intact. CPU adds no graph rewrite, logical constant, Tensor exponent discovery,
  or new public API.
- Generated and scalar-reference division results match primitive FLOAT32/FLOAT64 behavior for
  NaN, infinities, signed zero, divide-by-zero, overflow, underflow, subnormal transitions, and
  ordinary finite values. Power results satisfy the exact edge, sign, classification, overflow,
  underflow, subnormal, and finite-accuracy rules. Exact-result assertions and non-bitwise
  comparisons follow the explicit contracts rather than convenience.
- Every exponent outside `±0`, `+1`, `+2`, and `-1` uses `DIRECT`, including `0.5`, `3`, `-2`,
  infinities, and NaNs. Production contains no multi-multiply chain or exponentiation-by-squaring
  realization.
- `DIV`, `SCALAR_DIV`, and `SCALAR_POW` integrate into the family-oriented pipeline, whose closed
  vocabulary becomes exactly twenty-two opcodes, without a per-exponent/per-operation class
  explosion, registry, service locator, generic policy framework, or second compiler. Reciprocal
  `SCALAR_POW(-1)` remains semantically distinct from both DIV opcodes even if emission shares a
  primitive division helper.
- Scalar and parallel-scalar realize all three opcodes for FLOAT32/FLOAT64. Eligible FLOAT64 DIV
  and special power plans may use vector or parallel-vector only when every existing opcode,
  access, carrier, topology, and strategy check passes. FLOAT64 `DIRECT` power and all FLOAT32
  work use scalar or parallel-scalar compute without losing supported execution.
- DIV participates normally in the existing one-to-eight connected pointwise fusion limit and
  preserves the existing instruction order, virtual-intermediate, and one-final-store rules.
- One-to-eight fusion, virtual intermediates, one final store, all five access regimes,
  heap/segment/mixed carriers, broadcasting/access plans, arbitrary ranges, one-copy
  materialization, four candidates, one artifact, zero fixed-shape/unroll variants, and fail-closed
  alias/topology rules remain enforced.
- Opcode, exact scalar bits/type, realization, numerical mode, strategy, access form, carrier
  pattern, materialization, and schema produce compatible structural identity. A stale schema or a
  mismatched realization can never reuse an artifact; instance facts remain excluded.
- Optional persistence remains bounded and disabled by default under `KEEP_DISABLED`; ordinary
  validation does not run the timing evidence benchmark.
- No Model, Compiler, Config, Planning, shared Prepare, Runtime, Backend Contract, Trace, Engine,
  OpenBLAS provider, other backend, build, architecture, conformance, or integration path changes.
- A separate clean documentation-focused pass finalizes affected Javadocs, CPU guide, glossary,
  task evidence, master plan, and roadmap; records reasoned no-change conclusions for Tensor,
  Compile, Config, Runtime, Prepare, architecture, tests outside CPU, and Gradle; and reuses final
  Java evidence unless executable code changes afterward.
- CPU 0005A–0005E remain `Complete`; CPU 0005F remains `Ready` until implementation, validation,
  documentation review, and status synchronization all pass. CPU 0005G and CPU 0006–0017 remain
  `Draft` without detailed task specifications.

## Tests / validation

Run focused implementation tests:

```bash
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.CpuCapabilityProviderTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcodeTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIrTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScalarPowerAnalysisTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPointwisePartitionLoweringTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuPointwiseGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.reference.CpuReferenceDifferentialTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecializationTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStoreTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest
```

After executable Java stabilizes, run exactly one final affected-module suite:

```bash
./gradlew :backends:cpu:test
```

The ordinary suite must continue to skip the opt-in CPU 0005D persistence timing method. Do not
rerun that evidence benchmark: persistence remains disabled and this task makes no enablement or
performance claim.

The separate documentation-focused pass, after final Javadocs and Markdown, runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

It also validates with recorded commands and results:

- local Markdown targets and heading anchors in the CPU guide, glossary, this task, CPU master
  plan, and roadmap;
- balanced fences, final newlines, trailing whitespace, exact authorized-path membership, and the
  40-path ceiling;
- exact FLOAT32/FLOAT64 exponent/realization matrix, direct fallback, IEEE edge tables,
  finite-rounding boundaries, and rejection of unproved chains;
- exact binary/scalar DIV and scalar-POW capability/lowering/opcode matrices, primitive division
  edge behavior, and reciprocal semantic separation;
- opcode/IR/analysis/emitter/reference/package inventories and absence of per-exponent classes,
  registries, service locators, broad numerical frameworks, or another compiler;
- schema/fingerprint/specialization/manifest realization compatibility and stale-cache rejection;
- unchanged one-to-eight fusion, access/carrier/strategy/materialization/artifact budgets;
- CPU 0005A–0005E `Complete`, CPU 0005F `Ready` until completion, CPU 0005G and CPU 0006–0017
  `Draft`, and no later detailed CPU specification; and
- unchanged Java/test paths outside `backends/cpu`, Gradle, `ARCHITECTURE.md`, focused architecture
  docs, ADRs, architecture tests, backend conformance, and integration tests.

Repository-wide Java validation is deferred to CPU 0009, the portable generated-coverage closure
checkpoint, or continuous integration. This task changes one concrete backend privately and no
dependency, shared contract, public end-to-end API, or architecture boundary. The CPU-internal
generated/reference/direct-oracle matrix is the proportionate task-level proof.

The documentation pass reuses the successful implementation tests and does not repeat them unless
it changes executable Java or records a concrete stale-evidence risk.

## Dependencies

- Complete CPU 0005A partition-kernel reset and common route-independent lowering/IR architecture.
- Complete CPU 0005B static access, broadcasting, layouts, spans, and heap/segment carrier model.
- Complete CPU 0005C scalar/vector and single/parallel strategy implementation.
- Complete CPU 0005D materialization/specialization budgets and optional-persistence evidence gate.
- Complete CPU 0005E five-type pointwise family/opcode pipeline and derived boundaries.
- Current Model floating-only binary/scalar DIV and scalar `POW`, exact typed `ScalarValue`,
  right-aligned broadcasting, Shape, and layout contracts.
- Current Compiler 0003A exact guarded positive-one bypass and 0003B explicit-constant boundary.
- Current Config exact/default boundary and Draft Config 0006 relaxed-permission separation.
- Current staged Prepare, Runtime cold-binding/execution, and backend-contract ownership contracts.
- Java 26 `StrictMath.pow`, primitive floating division/arithmetic, Class-File API, and Vector API
  contracts already used by the CPU module.

## Follow-up tasks

- CPU 0005G remains Draft, without a detailed specification, for all remaining selected portable
  pointwise-family coverage. CPU 0006–0008 remain Draft for later portable operation families;
  CPU 0009 remains the portable coverage checkpoint and depends on 0005G closure.
- Compiler 0007 remains Draft for any future graph-level identities. This CPU task neither creates
  nor requires its `POW(0)` constant/publication proof.
- Config 0006 remains Draft as the future backend-neutral relaxed numerical permission. CPU 0017
  remains Draft as its consumer; neither blocks exact/default CPU 0005F.
- A later bounded task may reconsider additional integral-exponent realizations only if it first
  supplies a universal finite-rounding and exceptional-value proof. This task does not preapprove
  multiply chains or exponentiation by squaring.
- Remaining pointwise rows excluded by CPU 0005E/0005F belong to CPU 0005G and require separately
  bounded planning plus any missing Model contracts before CPU 0009 can close selected portable
  coverage.

## Architecture impact

Expected impact: None.

The architecture already assigns backend-private lowering, numerical eligibility, route/strategy
selection, generated specialization, exact declarations, and executable realization to concrete
CPU prepare. The task changes no owner, dependency, lifecycle, backend identity, or public API. If
implementation requires such a change, stop and report the conflict.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/planning/backends/cpu/master-plan.md, and
docs/planning/backends/cpu/tasks/0005f-floating-division-and-exact-scalar-power-realization.md.
Read the directly referenced completed CPU tasks and current Model/Compiler/Config/Prepare/Runtime
contracts needed to verify the exact division, exponent, and conformance matrices.

Implement CPU 0005F exactly as specified within its authorized paths and 40-path ceiling. Add
same-typed FLOAT32/FLOAT64 binary and scalar DIV through the existing family pipeline. Retain
semantic POW in the compiled graph; classify exact typed scalar exponents once in common CPU
analysis; preserve the direct FLOAT32/FLOAT64 power fallback; and implement only the four proved
special plans. Keep reciprocal POW semantically distinct from DIV. Do not add integer/BOOL
division, multi-multiply/exponentiation-by-squaring realizations, relaxed math, a broad policy
framework, public API, shared/build/architecture changes, later task specs, commits, or pushes.
Stop on a semantic proof, architecture, dependency, or scope conflict.

After executable Java stabilizes and the focused plus one final CPU suite pass, hand the exact diff
and recorded test evidence to a separate clean documentation-focused agent/thread. That pass must
follow docs/developer-guide/documentation-rules.md; independently finalize affected Javadocs, CPU
guide, glossary, planning evidence, links, scope/status checks, and reasoned no-change conclusions;
and not repeat successful Java tests unless executable behavior changes or a concrete risk
requires it.

Update this task's local decisions, known limitations, validation evidence, implementation notes,
completion summary, and synchronized final status. Do not mark Complete until every acceptance
criterion and the documentation pass succeed.
```

## Local decisions

- Binary and scalar DIV use exact primitive FLOAT32/FLOAT64 division in operand order. They are
  ordinary fused pointwise opcodes, not power realizations or reciprocal-multiply rewrites.
- FLOAT64 DIV may use the existing vector route when all current eligibility checks pass;
  FLOAT32 DIV remains scalar or parallel-scalar. This adds no broader vector promise.
- `DIRECT` is mandatory for every admitted exponent outside the four proved special cases. This
  makes scalar `POW` truthful coverage rather than a capability that accepts only optimization-
  friendly exponents.
- Positive and negative zero share `POSITIVE_ONE` because both have the same specified result,
  while their exact raw exponent bits remain in IR and structural identity.
- Positive two and negative one are admitted because each requires exactly one correctly rounded
  primitive operation and has a complete edge proof. Positive three, negative two, and other
  integral exponents remain direct because multi-step rounding is observably different in general.
- FLOAT32 direct power uses exact operand widening, `StrictMath.pow`, and one final narrowing. This
  is one CPU-private algorithm for an exact FLOAT32 semantic request, not cross-type Model
  promotion.
- Only FLOAT64 special plans may use the current vector emitter. Direct vector power and FLOAT32
  vector power remain scalar compute because availability of a vector operator is not an
  exact/default conformance proof.
- `SCALAR_POW(-1)` may reuse the emitter's primitive division mechanism, but its opcode,
  immediate, realization, structural identity, manifest, and semantic tests remain scalar power.
- One new stateless analysis type plus a nested IR realization enum is the complete abstraction
  increment. No registry, policy service, per-exponent class, or alternative IR is justified.

## Known limitations

- Executable division and scalar power are limited to fully static, resolved-layout, exact
  same-typed FLOAT32 or FLOAT64 occurrences inside CPU 0005E's bounded straight-line pointwise
  topology.
- BFLOAT16 remains storage-only for this family; FLOAT16 is absent; integral, BOOL, mixed-type,
  Tensor/Tensor power, and other Tensor-exponent power remain unsupported. Integral and BOOL
  division remain unsupported.
- Only `±0`, `+1`, `+2`, and `-1` receive special realizations. Every other exponent uses direct
  power even when a faster multiply chain is mathematically attractive.
- Direct power and all FLOAT32 work do not use Vector API compute. Existing parallel-scalar
  orchestration remains available. FLOAT64 DIV and special power plans vectorize only through the
  existing compatible route.
- The task makes no NaN-payload promise and no bitwise-equality promise beyond results explicitly
  fixed by the ordinary contract.
- Optional persistence remains disabled by default. Native/vendor routes, tuning, backend
  conformance, and Engine integration remain deferred.

## Validation evidence

The implementation context completed executable validation before this independent documentation
pass. The clean documentation-focused context `/root` changed only Javadocs/package summaries and
the five authorized Markdown records, so it reused the implementation evidence and did not rerun
Java tests:

- `./gradlew :backends:cpu:compileTestJava` passed.
- The focused generated-kernel suite passed after correction of one FLOAT32 constant-width
  bytecode issue found during implementation.
- The final focused IR/lowering/generated-kernel run passed 16 tests.
- Exactly one final `./gradlew :backends:cpu:test` passed. The preserved JUnit XML contains 25
  suites and 102 tests, with one skipped opt-in CPU 0005D persistence timing test, zero failures,
  and zero errors.
- `git diff --check` passed at implementation handoff.

The documentation pass independently reviewed the final production and test diff, generated
Javadocs, CPU 0005A–0005E contracts, Tensor and Compile API documentation, current Config package,
shared Prepare analysis contracts, Runtime prepared-execution contracts, the CPU guide, glossary,
and planning records. Its final evidence is:

- `./gradlew :backends:cpu:javadoc` passed with `BUILD SUCCESSFUL`; 11 actionable tasks reported
  two executed and nine up-to-date. The output contained the two expected incubating-Vector
  warnings and two unchanged implicit-constructor warnings, with no Javadoc error.
- A repository-local Markdown audit of the CPU guide, glossary, this task, CPU master plan, and
  roadmap passed local target, heading-anchor, balanced-fence, final-newline, and trailing-
  whitespace checks.
- The exact-scope audit found 18 authorized CPU production/package paths, eight authorized CPU
  test paths, two explanatory-documentation paths, and three planning/status paths: 31 total,
  with no unauthorized path and below the 40-path ceiling.
- Static source/test audits confirmed exactly 22 opcodes; same-typed FLOAT32/FLOAT64 binary and
  scalar DIV; all five exact-bit power classifications; FLOAT64-only vector eligibility under the
  existing gates; scalar or parallel-scalar fallback for direct power and all FLOAT32 work;
  schema 6; explicit realization compatibility; and no exponent comparison in generated code.
- Fusion inspection confirmed that the generic connected one-through-eight lowering admits binary
  DIV, scalar DIV, and scalar POW as ordinary single instructions, preserves stored order and
  virtual intermediates, and emits one final store. The existing one-through-eight test plus the
  new DIV/power lowering and generated-opcode tests cover the participating contracts without a
  second operation-specific fusion path.
- Status checks confirmed CPU 0005A–0005F `Complete`; CPU 0005G and CPU 0006–0017 `Draft`; and no
  detailed CPU 0005G or later task specification.
- Final `git diff --check` passed.

No architecture, dependency, module-boundary, public API, or shared lifecycle contract changed.
Tensor and Compile API pages remain accurate because the task realizes existing semantics only;
Config remains unchanged because exact/default needs no new permission; shared Prepare still sees
one opaque CPU plan plus exact declarations; Runtime still invokes one already-prepared executable;
and no public Tensor result, gradient rule, compiler transformation, backend-conformance,
integration, or Engine behavior is claimed. Gradle, `ARCHITECTURE.md`, focused architecture pages,
ADRs, architecture tests, other modules/backends, and tests outside CPU required no change.

## Implementation notes

CPU capability and whole-partition lowering now admit exact same-typed FLOAT32/FLOAT64 binary DIV,
scalar DIV, and scalar POW. Binary DIV retains two ordered tensor inputs and ordinary right-aligned
broadcasting; scalar DIV retains one exact denominator immediate; scalar POW retains one exact
exponent immediate plus its selected realization. The route-independent IR therefore distinguishes
Tensor/Tensor division, Tensor/scalar division, and reciprocal scalar power even when generated
emission shares primitive division.

`CpuScalarPowerAnalysis` classifies raw FLOAT32/FLOAT64 bits once during lowering. Both zero signs
select `POSITIVE_ONE`, positive one selects `IDENTITY`, positive two selects `SQUARE`, negative one
selects `RECIPROCAL`, and every other finite value, infinity, or NaN selects `DIRECT`. FLOAT32
direct power widens exact represented operands, calls `StrictMath.pow`, and narrows once; FLOAT64
calls it directly. No multiply chain, reciprocal chain, square-root replacement, tolerance, or
runtime policy lookup was added.

Scalar and reference emission implement primitive Java/IEEE-754 division and every selected power
plan. Eligible FLOAT64 DIV and special power remain available to the existing vector and parallel-
vector strategies; direct power and all FLOAT32 work use scalar compute with existing optional
parallel orchestration. The generic one-through-eight fusion, five access regimes, derived typed
carriers, one-copy materialization ceiling, four candidates, one artifact, zero fixed-shape or
unrolled variants, cold binding, concurrent-run isolation, and one final store remain unchanged.

Canonical IR identity includes opcode, exact scalar type/bits, and realization. Specialization and
the cold manifest expose the ordered scalar-power realization list, and generator schema 6 makes
older artifacts incompatible. Optional persistence remains bounded and disabled by default under
the CPU 0005D `KEEP_DISABLED` verdict.

## Completion summary

- Completed changes: implemented and documented exact/default FLOAT32/FLOAT64 binary DIV, scalar
  DIV, and scalar POW with direct fallback plus the four proved exact-bit special realizations.
- Files changed or created: 18 CPU production/package paths, eight CPU test paths, the CPU guide,
  glossary, this task, CPU master plan, and roadmap; 31 total within the exact authorized map and
  below the 40-path ceiling.
- Tests and validation: reused the successful compile-test, focused generated-kernel, final
  16-test focused, and sole final 25-suite/102-test CPU evidence with one opt-in timing skip and no
  failures or errors; final CPU Javadoc, Markdown, exact-scope, semantic, status, and whitespace
  checks passed.
- Documentation-agent review: clean documentation-focused context `/root` independently finalized
  affected Javadocs/package summaries, the CPU guide, glossary, task evidence, CPU master plan, and
  roadmap without changing executable Java or tests.
- Documentation impact: the current guide now distinguishes Tensor/Tensor DIV, Tensor/scalar DIV,
  and reciprocal scalar POW; records exact Java/IEEE behavior, realization plans, fusion,
  strategy, compatibility, schema, and lifecycle boundaries; and makes no public or end-to-end
  claim.
- Javadoc review: affected provider, schema, specialization, IR, lowering, emitter, reference,
  prepare, route-plan, and package contracts describe the exact semantic, numerical,
  compatibility, and cold/hot boundaries.
- Glossary impact: CPU implementation status, kernel IR, and scalar-power realization now reflect
  the twenty-two-opcode schema-6 implementation and reciprocal-power separation.
- Architecture/build/dependency impact: None.
- Unresolved issues: None.
- Follow-up required: None. CPU 0005G and CPU 0006–0017 remain Draft without detailed task
  specifications.

Status: Complete
