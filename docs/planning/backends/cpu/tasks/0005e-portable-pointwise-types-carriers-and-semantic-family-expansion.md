# Task 0005E: Portable [Pointwise](../../../../glossary.md#term-pointwise) Types, Carriers, and Semantic-Family Expansion

## Reader context

This checkpoint generalized the proving chain into a bounded pointwise family. A Model
occurrence described an operation and its typed inputs; CPU lowering translated it into a closed
opcode in one unit’s IR. Generated code calculated each logical output position, retaining
eligible intermediate results as typed locals rather than separate stored Tensors. The external
input and final output boundaries determined the invocation arguments.

A carrier was the container for represented bits—such as `float[]` or `byte[]`—not a new
numerical type or backend. BOOL bytes had to be canonical truth values, whereas numeric values
followed their own type rules. The five-type inventory, vector restrictions, and cross-type-cast
exclusions below are historical boundaries, not statements about every current Model method.

Current reading alongside this historical record:

- [CPU typed intermediate representation](../../../../explanation/packages/cpu.md#iogithubpho001synaptikbackendcpuinternalir)
  — distinguishes code-shaping structure from concrete geometry and live values.
- [CPU generated-code ownership](../../../../explanation/packages/cpu.md#iogithubpho001synaptikbackendcpuinternalcodegenemit)
  — separates generation-time emitters, emitted typed work, and JVM compilation.

## Status

Complete

## Goal

Deliver the first bounded expansion from the completed [FLOAT64](../../../../glossary.md#term-float64) `ADD -> exact GELU -> MUL`
proving [topology](../../../../glossary.md#term-topology) to a reusable portable [pointwise](../../../../glossary.md#term-pointwise) family. One [CPU](../../../../glossary.md#term-cpu)-owned [partition](../../../../glossary.md#partition) may contain a
single supported pointwise [occurrence](../../../../glossary.md#term-occurrence) or one bounded straight-line [fused](../../../../glossary.md#specialization-and-fusion) chain. Common CPU
analysis lowers the chain once into the existing route-independent [kernel](../../../../glossary.md#kernel) [intermediate
representation](../../../../glossary.md#term-intermediate-representation-ir) ([IR](../../../../glossary.md#intermediate-representation-dag-and-ssa)), normalizes every external value through the completed access plans, selects
the completed portable route and [scalar](../../../../glossary.md#term-scalar)/[vector](../../../../glossary.md#term-vector) plus single/parallel strategy, applies the
completed optional one-input [materialization](../../../../glossary.md#materialization) decision, and realizes one generated [artifact](../../../../glossary.md#term-artifact) and
one partition executable.

This is deliberately a first coverage increment, not "all pointwise everything." It establishes
five exact logical types and their direct [heap](../../../../glossary.md#term-heap)/[`MemorySegment`](../../../../glossary.md#term-memorysegment) carriers, one family-oriented CPU
[opcode](../../../../glossary.md#term-opcode) vocabulary, and a useful core semantic [matrix](../../../../glossary.md#term-matrix-meanings):

- same-type [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) and exact-scalar `ADD`, `SUB`, and `MUL` for FLOAT64, [FLOAT32](../../../../glossary.md#term-float32), [INT32](../../../../glossary.md#data-type--datatype), and [INT64](../../../../glossary.md#data-type--datatype);
- [unary](../../../../glossary.md#term-unary) `NEG` for FLOAT64 and FLOAT32, while preserving the existing FLOAT64 exact [`GELU`](../../../../glossary.md#gaussian-error-linear-unit--gelu);
- `IS_FINITE`, `IS_NAN`, and `IS_INF` for FLOAT64 and FLOAT32;
- all six current comparisons for same-type FLOAT64, FLOAT32, INT32, and INT64;
- `WHERE` for a [BOOL](../../../../glossary.md#data-type--datatype) condition and same-type FLOAT64 or FLOAT32 branches; and
- same-type `CAST` for FLOAT64, FLOAT32, INT32, INT64, and BOOL.

The execution boundary is:

```text
supported static pointwise partition
  -> bounded straight-line unit and family-oriented CPU opcodes
  -> existing normalized access plans and optional one-input materialization
  -> existing portable route, strategy selection, specialization, and artifact lifecycle
  -> one partition-level cold-bound executable
```

Every other [operation](../../../../glossary.md#operation), type combination, topology, conversion, or numerical mode remains
[fail-closed](../../../../glossary.md#term-fail-closed) and unadvertised.

## [Scope](../../../../glossary.md#term-scope)

### Exact semantic and data-type [matrix](../../../../glossary.md#term-matrix-meanings)

- Accept parameterless `BinaryArithmeticKind.ADD`, `SUB`, and `MUL` only when both inputs and the
  result have the same exact data type, selected from FLOAT64, FLOAT32, INT32, and INT64.<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype))</span>
- Accept `ScalarElementwiseKind.ADD`, `SUB`, and `MUL` only with `ScalarValueAttrs` whose exact
  `ScalarValue.dataType()` equals the input and output type, selected from FLOAT64, FLOAT32,
  INT32, and INT64. Retain exact scalar bits in the CPU IR and specialization identity; do not
  turn the scalar into another graph input or read Tensor storage.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [specialization glossary](../../../../glossary.md#term-specialization); [identity glossary](../../../../glossary.md#term-identity-meanings); [graph glossary](../../../../glossary.md#graph); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Accept `UnaryElementwiseKind.NEG` for same-type FLOAT64 and FLOAT32 input/output. Preserve the
  completed FLOAT64 exact [`GELU`](../../../../glossary.md#gaussian-error-linear-unit--gelu) opcode and its existing oracle, special-value, and operation-order
  contract.<span data-glossary-navigation="true"> ([opcode glossary](../../../../glossary.md#term-opcode); [oracle glossary](../../../../glossary.md#term-test-oracle); [operation glossary](../../../../glossary.md#operation))</span>
- Accept all `FloatingClassificationKind` values—`IS_FINITE`, `IS_NAN`, and `IS_INF`—for FLOAT64
  and FLOAT32 input and exact BOOL output of the same Shape.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape))</span>
- Accept all `BinaryComparisonKind` values—`GREATER_THAN`, `GREATER_OR_EQUAL`, `LESS_THAN`,
  `LESS_OR_EQUAL`, `EQUAL`, and `NOT_EQUAL`—for same-type FLOAT64, FLOAT32, INT32, and INT64
  inputs and exact BOOL broadcast result.
- Accept `WhereSelectionKind.WHERE` only for one BOOL condition, two same-type FLOAT64 or FLOAT32
  branches, and an output of that exact branch type. Derive the branch broadcast first and then
  broadcast the condition against it, matching the current Model construction contract.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- Accept `CastKind.CAST` only when `CastAttrs.targetDataType()` equals both the input and output
  type and that type is FLOAT64, FLOAT32, INT32, INT64, or BOOL. This realizes the represented
  value identity of a same-type explicit cast. Cross-type conversion remains unsupported because
  the current Model contract intentionally does not define its rounding, overflow, saturation,
  NaN, or BOOL conversion result.<span data-glossary-navigation="true"> ([rounding glossary](../../../../glossary.md#term-rounding); [overflow glossary](../../../../glossary.md#term-overflow); [saturation glossary](../../../../glossary.md#term-saturation); [NaN glossary](../../../../glossary.md#term-nan))</span>
- Require exact family-owned attribute classes: `NoOperationAttrs.INSTANCE` for parameterless
  kinds, `ScalarValueAttrs` for the selected [scalar](../../../../glossary.md#term-scalar) arithmetic kinds, and `CastAttrs` for CAST.
  Reject every custom or mismatched kind/attribute pair before lowering or artifact access.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Require every descriptor Shape to be fully static and every layout resolved. Reuse the completed
  right-aligned <span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [layout glossary](../../../../glossary.md#layout))</span>`ShapeBroadcast` and `LayoutDescriptor` normalization without adding a CPU shape,
  broadcasting, or layout contract.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [broadcasting glossary](../../../../glossary.md#broadcasting))</span>
- Continue to [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation) for [BFLOAT16](../../../../glossary.md#term-bfloat16). Its `short[]` and two-byte segment representation remains
  observable storage plumbing, not executable arithmetic permission. FLOAT16 remains absent and
  blocked on future Model task 0026.<span data-glossary-navigation="true"> ([FLOAT16 glossary](../../../../glossary.md#term-float16))</span>

### Exact/default numerical behavior

- FLOAT64 and FLOAT32 ADD, SUB, MUL, and NEG use Java's strict floating evaluation in the exact
  result type and preserved graph order. Do not reassociate, contract, widen an intermediate,
  substitute an approximation, or promise a NaN payload or bitwise result beyond the Model
  contract. Preserve IEEE classifications, infinities, underflow/overflow behavior, and signed
  zero implied by the ordered primitive operation.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [evaluation glossary](../../../../glossary.md#term-evaluation); [graph glossary](../../../../glossary.md#graph); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [Model glossary](../../../../glossary.md#term-model); [IEEE glossary](../../../../glossary.md#term-ieee); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [underflow glossary](../../../../glossary.md#term-underflow); [overflow glossary](../../../../glossary.md#term-overflow); [signed zero glossary](../../../../glossary.md#term-signed-zero); [operation glossary](../../../../glossary.md#operation))</span>
- INT32 and INT64 ADD, SUB, and MUL use the current fixed-width two's-complement modular Model
  meaning, exactly matching Java <span data-glossary-navigation="true"> ([INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [two's-complement glossary](../../../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks))</span>`int` and `long` [wraparound](../../../../glossary.md#term-wraparound).
- Floating comparisons use represented numeric values: ordered relations are false for NaN;
  negative and positive zero compare equal; <span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>`EQUAL` is numeric equality; and `NOT_EQUAL` is its
  logical complement. Integral comparisons use signed order. Comparison and classification
  results store canonical BOOL byte `0` or `1`.
- `IS_FINITE`, `IS_NAN`, and `IS_INF` use the current classifications for both signed zeros,
  subnormals, infinities, and every NaN. No NaN payload is inspected beyond classification.<span data-glossary-navigation="true"> ([subnormals glossary](../../../../glossary.md#term-subnormal))</span>
- WHERE treats [canonical](../../../../glossary.md#term-canonical-form) [BOOL](../../../../glossary.md#data-type--datatype) byte `1` as true and `0` as false and copies the selected represented
  branch value. Cold binding rejects a non-canonical BOOL condition representation rather than
  inventing truthiness. Selection does not authorize branch-value conversion.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- Same-type CAST preserves the represented value, including signed zero and classification. It
  makes no stronger NaN-payload promise than the source/target type contracts and is not removed
  from the compiled graph by CPU lowering.<span data-glossary-navigation="true"> ([payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [target glossary](../../../../glossary.md#term-target); [compiled graph glossary](../../../../glossary.md#compiled-graph--compiledgraphmodel); [CPU glossary](../../../../glossary.md#term-cpu); [lowering glossary](../../../../glossary.md#lowering))</span>
- Existing exact GELU remains the only transcendental/activation computation in this task. The
  task adds no relaxed or fast math and no new accuracy policy.<span data-glossary-navigation="true"> ([GELU glossary](../../../../glossary.md#gaussian-error-linear-unit--gelu); [activation glossary](../../../../glossary.md#term-activation))</span>

### Family-oriented [lowering](../../../../glossary.md#lowering) and emission

- Add one [CPU](../../../../glossary.md#term-cpu)-private `CpuPointwiseOpcode` enum in `internal.ir`. It contains the exact selected
  arithmetic, unary, classification, comparison, selection, cast, and existing exact-GELU opcodes.
  Each opcode declares its family, exact arity, result category, whether it carries an exact scalar
  immediate, and whether the current Vector emitter may realize it. Do not add one planner,
  lowerer, IR instruction class, emitter class, executable class, or registry entry per Model
  operation.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [GELU glossary](../../../../glossary.md#gaussian-error-linear-unit--gelu); [opcode glossary](../../../../glossary.md#term-opcode); [arity glossary](../../../../glossary.md#term-arity); [scalar glossary](../../../../glossary.md#term-scalar); [Vector glossary](../../../../glossary.md#term-vector); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [registry glossary](../../../../glossary.md#term-registry))</span>
- Generalize `CpuKernelIr.Instruction` to retain one `CpuPointwiseOpcode`, ordered topology-local
  input ordinals, one output ordinal, and an optional nested <span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology))</span>`CpuKernelIr.ScalarImmediate` holding
  exact [`DataType`](../../../../glossary.md#data-type--datatype) plus primitive bits copied from [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) during lowering. Validation is
  family-oriented and rejects an immediate on parameterless operations or its absence/type
  mismatch on selected scalar arithmetic.<span data-glossary-navigation="true"> ([Validation glossary](../../../../glossary.md#term-validation))</span>
- Keep [Model](../../../../glossary.md#term-model) [`Operation`](../../../../glossary.md#operation), [`OperationKind`](../../../../glossary.md#operationkind), [`OperationAttrs`](../../../../glossary.md#operationattrs), `CompiledNode`, graph identities, and
  strings out of the generated and Runtime hot paths. Lowering maps each admitted Model occurrence
  to the CPU opcode once during analysis.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [Runtime glossary](../../../../glossary.md#term-runtime); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Extend the existing `CpuScalarEmitter` and `CpuVectorEmitter` through family-grouped emission
  methods. Shared load/store and loop emission remains in `CpuCarrierEmitter` and
  `CpuLoopEmitter`. The generator consumes the already-typed opcode sequence; it does not contain
  a second Model-operation planner or string dispatch.<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Extend `CpuScalarReferenceKernel` with the same family grouping for differential conformance.
  It remains test/fail-closed reference code and never interprets Model operations or CPU IR in a
  Runtime invocation.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>

### [Partition](../../../../glossary.md#partition), [fusion](../../../../glossary.md#term-fusion), access, and [materialization](../../../../glossary.md#materialization) boundaries

- Accept one non-empty CPU partition of at most eight supported pointwise occurrences. It may be
  one occurrence or one straight-line chain in stored partition order: each non-final occurrence
  has one output, that output is consumed exactly once by a later occurrence in the same unit,
  has no publication or cross-partition obligation, and is not consumed outside the unit. Side
  inputs are materialized boundaries. The final result is the unit's sole materialized output.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [pointwise glossary](../../../../glossary.md#term-pointwise); [occurrence glossary](../../../../glossary.md#term-occurrence); [publication glossary](../../../../glossary.md#term-publication))</span>
- Require the internal dataflow to be connected and acyclic by the existing partition order. No
  independent subchains, multi-output node, internal fan-out, multiple materialized outputs, or
  unit sequencing is admitted in this increment.<span data-glossary-navigation="true"> ([dataflow glossary](../../../../glossary.md#term-dataflow); [node glossary](../../../../glossary.md#node); [fan-out glossary](../../../../glossary.md#term-fan-out))</span>
- Allow a later supported pointwise occurrence to expand an earlier virtual result by ordinary
  right-aligned broadcasting. Derive every node result with the current family-specific Model
  Shape rule, require the declared descriptor to match, and normalize all external boundaries
  against the final iteration Shape. Virtual values remain typed IR values without buffer
  declarations or Runtime slots.<span data-glossary-navigation="true"> ([broadcasting glossary](../../../../glossary.md#broadcasting); [Model glossary](../../../../glossary.md#term-model); [Shape glossary](../../../../glossary.md#shape); [descriptor glossary](../../../../glossary.md#term-descriptor); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [buffer glossary](../../../../glossary.md#term-buffer); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Fusion is legal only when the complete chain satisfies the exact type/Shape/attribute matrix,
  preserves stored node order and numerical order, has no internal publication/fan-out/
  cross-partition use, and passes the completed output injectivity and boundary alias checks.
  Profitability is bounded here: every legal chain of at most eight instructions is selected as
  one unit; a longer or otherwise incompatible partition fails closed rather than silently
  generating multiple per-operation kernels.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [injectivity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [alias glossary](../../../../glossary.md#aliasing-and-arena); [operation glossary](../../../../glossary.md#operation); [kernels glossary](../../../../glossary.md#kernel))</span>
- Reuse `CpuAccessPlan` unchanged as the sole external-value access system, including its five
  regimes, exact spans, injective writes, arbitrary half-open ranges, heap/segment/mixed carriers,
  scalar fallback, and no per-element cursor/division/modulo rules.<span data-glossary-navigation="true"> ([injective glossary](../../../../glossary.md#term-injective); [half-open glossary](../../../../glossary.md#term-half-open-interval); [heap glossary](../../../../glossary.md#term-heap); [scalar glossary](../../../../glossary.md#term-scalar); [element glossary](../../../../glossary.md#term-element))</span>
- Reuse the completed direct-versus-one-input materialization analysis. Enumerate direct plus at
  most the first three eligible read boundaries in deterministic boundary order, under the
  existing four-complete-candidate ceiling. Direct remains correct and wins ties. A selected copy
  remains eligible only for a FLOAT64 read boundary and uses the completed FLOAT64 contiguous
  workspace/copy path. FLOAT32, integral, and BOOL boundaries remain direct in this increment. No
  second copy, output copy, graph value, or Runtime transfer recipe is introduced.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [contiguous glossary](../../../../glossary.md#term-contiguous); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BOOL glossary](../../../../glossary.md#data-type--datatype); [graph value glossary](../../../../glossary.md#graph-value))</span>

### Carriers, route, [vector](../../../../glossary.md#term-vector), parallel, [artifact](../../../../glossary.md#term-artifact), and executable reuse

- Generalize the structural [carrier](../../../../glossary.md#term-storage-carrier) vocabulary to these exact [heap](../../../../glossary.md#term-heap) forms plus exact-segment form:

  | [Data type](../../../../glossary.md#data-type--datatype) | Heap carrier | [`MemorySegment`](../../../../glossary.md#term-memorysegment) [element](../../../../glossary.md#term-element) representation |
  |---|---|---|
  | [FLOAT64](../../../../glossary.md#term-float64) | `double[]` | native-order [IEEE](../../../../glossary.md#term-ieee) [binary64](../../../../glossary.md#floating-point-precision-and-rounding), 8-byte element width |
  | [FLOAT32](../../../../glossary.md#term-float32) | `float[]` | native-order IEEE [binary32](../../../../glossary.md#floating-point-precision-and-rounding), 4-byte element width |
  | [INT32](../../../../glossary.md#data-type--datatype) | `int[]` | native-order signed 32-bit [two's-complement](../../../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks), 4-byte width |
  | [INT64](../../../../glossary.md#data-type--datatype) | `long[]` | native-order signed 64-bit two's-complement, 8-byte width |
  | [BOOL](../../../../glossary.md#data-type--datatype) | `byte[]` | [canonical](../../../../glossary.md#term-canonical-form) byte `0` or `1`, 1-byte width |

- Each generated class still has one direct static entry for its actual ordered boundary carrier
  pattern. Boundary count is now derived from the selected chain rather than fixed at four. The
  exact ordered data-type/carrier pattern participates in entry descriptor, specialization,
  artifact metadata, verification, and compatibility. Concrete carriers, extents, offsets,
  strides, assignments, and addresses remain cold facts.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [specialization glossary](../../../../glossary.md#term-specialization); [metadata glossary](../../../../glossary.md#term-metadata); [strides glossary](../../../../glossary.md#term-stride))</span>
- `CpuPartitionAnalysisInputs.DEFAULT` remains persistence-free, manifest-disabled, scalar,
  single-thread, and direct-only, but its old four-segment carrier list cannot remain a general
  topology contract. Replace the fixed list with a default carrier policy of exact-segment for
  each derived boundary; explicit analysis input may supply an immutable ordered carrier pattern
  whose count must equal the lowered boundary count.<span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence); [manifest glossary](../../../../glossary.md#term-manifest); [scalar glossary](../../../../glossary.md#term-scalar); [topology glossary](../../../../glossary.md#term-topology); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- Reuse the completed portable route, Class-File generation, optional verified persistence,
  process-local compatible interning, post-assignment finalization, strong artifact ownership,
  direct cold binding, and one partition-level executable. No second route or cache is added.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [binding glossary](../../../../glossary.md#term-binding); [partition glossary](../../../../glossary.md#partition); [cache glossary](../../../../glossary.md#term-cache))</span>
- Scalar and parallel-scalar realize every opcode/type combination in this task. Vector and
  parallel-vector remain eligible only for chains whose values are all FLOAT64 and whose opcodes
  are drawn from ADD, SUB, MUL, NEG, and existing exact GELU, subject to the completed access-run
  rules. BOOL-producing, WHERE, CAST, FLOAT32, and integral chains select scalar compute; this is a
  cold supported fallback, not a failure or a vector-coverage claim.<span data-glossary-navigation="true"> ([opcode glossary](../../../../glossary.md#term-opcode); [GELU glossary](../../../../glossary.md#gaussian-error-linear-unit--gelu); [run glossary](../../../../glossary.md#term-run))</span>
- Preserve deterministic disjoint parallel chunks, scalar tails for vector code, worker borrowing,
  joined failure/interruption behavior, and zero-work handling. Do not add gather, masked tails,
  another species search, or worker lifecycle ownership.<span data-glossary-navigation="true"> ([borrowing glossary](../../../../glossary.md#term-borrowing); [gather glossary](../../../../glossary.md#gather); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Bump the generator/schema compatibility version because opcode structure, entry descriptors,
  carrier vocabulary, and boundary cardinality change. Old entries are safe incompatible misses;
  there is still no migration reader.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [cardinality glossary](../../../../glossary.md#term-cardinality))</span>
- Preserve the completed `KEEP_DISABLED` default persistence verdict. This task does not rerun the
  CPU 0005D timing evidence because it materially changes semantic and carrier fixtures but does
  not propose enabling persistence; a future deliberate persistence-policy evaluation owns a new
  representative evidence matrix.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-review-matrix))</span><span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>

### Tests and documentation

- Add table-driven capability and lowering tests for every exact supported kind/type/attribute
  row and representative rejected adjacent row, including BFLOAT16, mixed-type operands,
  cross-type CAST, unsupported unary/transcendental/extrema/division/power, bad BOOL, unresolved or
  dynamic layouts, and over-budget or non-linear partitions.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [unary glossary](../../../../glossary.md#term-unary); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [BOOL glossary](../../../../glossary.md#data-type--datatype); [layouts glossary](../../../../glossary.md#layout); [linear glossary](../../../../glossary.md#term-linear-meanings))</span>
- Add generated-versus-reference differential coverage across scalar/zero/non-zero shapes, all
  five access regimes, offsets/positive and zero strides, arbitrary ranges, heap/segment/mixed
  carriers, single operations, fused chains, every supported data type, and every supported opcode.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [shapes glossary](../../../../glossary.md#shape); [strides glossary](../../../../glossary.md#term-stride); [heap glossary](../../../../glossary.md#term-heap); [fused glossary](../../../../glossary.md#specialization-and-fusion); [data type glossary](../../../../glossary.md#data-type--datatype); [opcode glossary](../../../../glossary.md#term-opcode))</span>
- Include boundary vectors for floating signed zero, subnormals, finite extrema, infinities, and
  NaNs; integral min/max values and wraparound; every comparison result; classification truth
  tables; WHERE choices; canonical BOOL output; scalar immediate exact bits; and same-type CAST.<span data-glossary-navigation="true"> ([vectors glossary](../../../../glossary.md#term-vector); [signed zero glossary](../../../../glossary.md#term-signed-zero); [subnormals glossary](../../../../glossary.md#term-subnormal); [finite glossary](../../../../glossary.md#term-finite); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [wraparound glossary](../../../../glossary.md#term-wraparound); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Preserve exact-GELU oracle tests and the existing vector/parallel/materialization/artifact/
  executable suites. Add representative all-heap, all-segment, and mixed patterns per data type
  and boundary order; do not enumerate an exponential carrier-pattern Cartesian product for every
  eight-node chain.<span data-glossary-navigation="true"> ([GELU glossary](../../../../glossary.md#gaussian-error-linear-unit--gelu); [oracle glossary](../../../../glossary.md#term-test-oracle); [materialization glossary](../../../../glossary.md#materialization); [artifact glossary](../../../../glossary.md#term-artifact); [carrier glossary](../../../../glossary.md#term-storage-carrier); [Cartesian product glossary](../../../../glossary.md#term-cartesian-product); [node glossary](../../../../glossary.md#node))</span>
- Finalize affected CPU Javadocs, package summaries, the CPU backend guide, glossary, and planning
  records in the required separate clean documentation-focused context. Review the Tensor and
  Compile API guides and record a reasoned no-change conclusion: this task adds backend coverage,
  not a public API or Model semantic.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend); [planning glossary](../../../../glossary.md#term-planning); [Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Model glossary](../../../../glossary.md#term-model))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- Binary or scalar DIV, MIN, MAX, POW, scalar CLAMP, and every mixed-type promoted arithmetic or
  comparison. Exact scalar POW strength reduction remains solely CPU 0005F.<span data-glossary-navigation="true"> ([Binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Unary ABS, RECIPROCAL, LOG, LOG1P, EXP, EXPM1, ERF, SQRT, RSQRT, FLOOR, CEIL, SIGN, RELU,
  SIGMOID, TANH, GELU_TANH_APPROXIMATION, and SILU. Existing exact FLOAT64 GELU remains supported;
  no new transcendental implementation is added.<span data-glossary-navigation="true"> ([Unary glossary](../../../../glossary.md#term-unary); [LOG glossary](../../../../glossary.md#term-log); [EXP glossary](../../../../glossary.md#term-expx); [ERF glossary](../../../../glossary.md#term-erf); [SIGMOID glossary](../../../../glossary.md#term-sigmoid); [TANH glossary](../../../../glossary.md#term-tanh); [FLOAT64 glossary](../../../../glossary.md#term-float64); [GELU glossary](../../../../glossary.md#gaussian-error-linear-unit--gelu))</span>
- BFLOAT16 execution, FLOAT16, mixed precision, cross-type CAST, BOOL arithmetic/logical
  operations, integral WHERE, or a backend-defined conversion/rounding/saturation policy.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT16 glossary](../../../../glossary.md#term-float16); [mixed precision glossary](../../../../glossary.md#term-mixed-precision); [BOOL glossary](../../../../glossary.md#data-type--datatype); [backend glossary](../../../../glossary.md#backend); [rounding glossary](../../../../glossary.md#term-rounding); [saturation glossary](../../../../glossary.md#term-saturation))</span>
- Reductions, scans, statistics, softmax/normalization, layout/view/indexing/scatter/ordering,
  random/dropout, matrix multiplication, convolution, pooling, attention, loss, or any operation
  assigned to CPU 0006–0008.<span data-glossary-navigation="true"> ([softmax glossary](../../../../glossary.md#softmax--log-softmax); [normalization glossary](../../../../glossary.md#term-normalization); [layout glossary](../../../../glossary.md#layout); [view glossary](../../../../glossary.md#term-view-meanings); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [matrix multiplication glossary](../../../../glossary.md#matrix-multiplication--matmul); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [pooling glossary](../../../../glossary.md#term-pooling); [loss glossary](../../../../glossary.md#loss); [operation glossary](../../../../glossary.md#operation))</span>
- Multiple execution units, disconnected subchains, internal fan-out, multiple materialized
  outputs, more than eight instructions, more than one materialization, or cross-partition fusion.<span data-glossary-navigation="true"> ([fan-out glossary](../../../../glossary.md#term-fan-out); [materialization glossary](../../../../glossary.md#materialization); [partition glossary](../../../../glossary.md#partition); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- Relaxed/fast math, numerical-mode configuration, native/vendor routes, OpenBLAS integration,
  route tuning, tuning-cache work, benchmarks, persistence-policy enablement, fixed shapes,
  unrolling, vector gather, masked tails, or additional Vector species.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [cache glossary](../../../../glossary.md#term-cache); [persistence glossary](../../../../glossary.md#term-persistence); [shapes glossary](../../../../glossary.md#shape); [unrolling glossary](../../../../glossary.md#term-unrolling); [vector glossary](../../../../glossary.md#term-vector); [gather glossary](../../../../glossary.md#gather); [Vector species glossary](../../../../glossary.md#term-vector-species))</span>
- Model, Compiler, Planning, shared Prepare, Runtime, Config, Backend Contract, Trace, Engine,
  OpenBLAS provider, another backend, public API, Gradle, dependency, architecture contract,
  focused architecture explanation, ADR, architecture test, backend-conformance module, or
  integration-test changes.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Trace glossary](../../../../glossary.md#trace); [Engine glossary](../../../../glossary.md#term-engine); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [architecture contract glossary](../../../../glossary.md#architecture-contract); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [architecture test glossary](../../../../glossary.md#term-architecture-test); [conformance glossary](../../../../glossary.md#term-conformance); [module glossary](../../../../glossary.md#term-module))</span>
- Creating a detailed [CPU](../../../../glossary.md#term-cpu) 0005F or later task specification, committing, or pushing.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially core invariants, Model,
  Planning, Prepare, Runtime, concrete backend ownership, CPU backend routes, and dependency rules.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [concrete backend glossary](../../../../glossary.md#concrete-backend); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md).
- [Planning Guide](../../../planning-guide.md).
- [CPU backend master plan](../master-plan.md).
- [Completed CPU 0005A](0005a-atomic-partition-kernel-architecture-reset.md).
- [Completed CPU 0005B](0005b-universal-access-plans-and-right-aligned-broadcasting.md).
- [Completed CPU 0005C](0005c-vector-and-parallel-portable-strategies.md).
- [Completed CPU 0005D](0005d-materialization-specialization-and-persistence-evidence-gate.md).
- [CPU backend guide](../../../../backend-guide/cpu-backend.md#atomic-partition-kernel-reset).
- [Glossary: CPU kernel intermediate representation](../../../../glossary.md#cpu-kernel-intermediate-representation).
- [Glossary: CPU access plan](../../../../glossary.md#cpu-access-plan).

## Architecture constraints

- [`Operation`](../../../../glossary.md#operation) and the Model operation families remain the sole semantic source. CPU maps supported
  occurrences into private opcodes but does not change or replace Model meaning. The deliberate
  same-type CAST limit avoids inventing the currently undefined cross-type conversion policy.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Planning](../../../../glossary.md#term-planning) continues to select only `BackendId("cpu")` ownership. CPU analysis owns pointwise
  lowering, bounded fusion, access normalization, materialization comparison, strategy and route
  selection, specialization, and exact declarations.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [normalization glossary](../../../../glossary.md#term-normalization); [materialization glossary](../../../../glossary.md#materialization); [specialization glossary](../../../../glossary.md#term-specialization))</span>
- Shared Prepare sees only the opaque selected CPU plan and exact buffer/workspace requirements.
  It learns no opcode, carrier, fusion, access, numerical, candidate, or artifact policy.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [opaque glossary](../../../../glossary.md#term-opaque); [buffer glossary](../../../../glossary.md#term-buffer); [opcode glossary](../../../../glossary.md#term-opcode); [carrier glossary](../../../../glossary.md#term-storage-carrier); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- CPU finalization occurs after shared slot assignment and cannot change the selected unit,
  opcodes, route, strategy, carrier pattern, materialization, or resource set.
- Runtime receives one immutable prepared executable and cold-bound invocation. Its hot path sees
  no <span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [immutable glossary](../../../../glossary.md#term-immutable); [prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable); [bound invocation glossary](../../../../glossary.md#bound-invocation--boundinvocation); [hot path glossary](../../../../glossary.md#term-hot-path))</span>`Operation`, `CompiledNode`, graph, CPU IR, route selection, artifact/cache access, storage
  discovery, or semantic dispatch.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [cache glossary](../../../../glossary.md#term-cache); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- The task changes no public supported [CPU](../../../../glossary.md#term-cpu) [API](../../../../glossary.md#term-application-programming-interface-api) beyond truthful answers from the existing
  `CpuCapabilityProvider`, no [module](../../../../glossary.md#term-module) [ownership](../../../../glossary.md#term-ownership-meanings), no [dependency direction](../../../../glossary.md#java-platform-and-build-tools), and no [lifecycle](../../../../glossary.md#lifecycle) stage.
- If implementation discovers that any selected row requires a Model semantic decision, shared
  contract, another module, dependency/build change, or architecture change, stop and report the
  exact prerequisite instead of broadening CPU-local policy.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>

The architecture audit conclusion for planning is `None`: the authoritative contract already
assigns concrete [lowering](../../../../glossary.md#lowering), [fusion](../../../../glossary.md#term-fusion), route choice, generated [kernels](../../../../glossary.md#kernel), storage, finalization, and
execution to the CPU [backend](../../../../glossary.md#backend). No `ARCHITECTURE.md` change is required for this bounded expansion.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu` — [occurrence](../../../../glossary.md#term-occurrence)-local truthful capability reporting.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — canonical typed values, family-oriented
  opcodes/instructions, access plans, loop, and stores.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — complete-partition validation,
  straight-line unit formation, fusion, boundary discovery, and optional materialization.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [validation glossary](../../../../glossary.md#term-validation); [fusion glossary](../../../../glossary.md#term-fusion); [materialization glossary](../../../../glossary.md#materialization))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — direct family-grouped scalar and
  bounded vector emission over existing carrier/loop primitives.<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.memory` — exact typed [heap](../../../../glossary.md#term-heap)/segment [arguments](../../../../glossary.md#term-argument).
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — typed inputs, candidate selection,
  declarations, opaque plan retention, and post-assignment finalization.<span data-glossary-navigation="true"> ([opaque glossary](../../../../glossary.md#term-opaque))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.route.portable` — unchanged sole route [leaf](../../../../glossary.md#term-leaf).
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — specialization/schema/artifact
  compatibility and existing budgets.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [schema glossary](../../../../glossary.md#term-schema); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.executable` — one cold-bound partition recipe and
  existing worker orchestration.
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — differential [scalar](../../../../glossary.md#term-scalar) [conformance](../../../../glossary.md#term-conformance).

Packages added or changed:

- No package is added and no responsibility moves between packages.

Type placement:

- `io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode` — one new enum because the
  CPU IR needs a stable Model-free typed semantic vocabulary shared across lowering, identity,
  scalar/vector emission, and reference conformance.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [Model glossary](../../../../glossary.md#term-model); [lowering glossary](../../../../glossary.md#lowering); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `CpuKernelIr.Instruction` — generalized in place for the opcode, typed immediate, arity, and
  result validation; its nested <span data-glossary-navigation="true"> ([opcode glossary](../../../../glossary.md#term-opcode); [arity glossary](../../../../glossary.md#term-arity))</span>`ScalarImmediate` owns the CPU-private exact typed bits, and no
  per-operation instruction subtype is added.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- `CpuScalarEmitter` and `CpuVectorEmitter` — extended in place with family-grouped methods; no
  per-operation emitter class, registry, manager, or generic string dispatch is added.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>

## Affected files

Expected [CPU](../../../../glossary.md#term-cpu) production paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPointwiseOpcode.java` (new)
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuKernelIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuMaterializationPlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/memory/CpuBufferArgument.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/memory/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionAnalysisInputs.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecialization.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStore.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuCarrierEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuGeneratedKernel.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuLoopEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScalarEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuVectorEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutable.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuScalarReferenceKernel.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/CpuPortableRoutePlan.java`

Expected CPU test paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPointwiseOpcodeTest.java` (new)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuKernelIrTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPointwisePartitionLoweringTest.java` (new)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuMaterializationPlanTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPointwiseGeneratedKernelTest.java` (new)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelPersistenceEvidenceTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutableTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/memory/CpuBufferBindingTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuReferenceDifferentialTest.java`

Expected explanatory and [planning](../../../../glossary.md#term-planning) paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- `docs/planning/backends/cpu/tasks/0005e-portable-pointwise-types-carriers-and-semantic-family-expansion.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

No other path is authorized by default.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most 51 repository paths:

| Category | Maximum | Path accounting |
|---|---:|---|
| [CPU](../../../../glossary.md#term-cpu) production | 31 | Thirty existing production/package paths plus one new [opcode](../../../../glossary.md#term-opcode) type |
| CPU tests | 15 | Twelve existing tests, including the evolved-[IR](../../../../glossary.md#term-intermediate-representation-ir) [persistence](../../../../glossary.md#term-persistence) [fixture](../../../../glossary.md#term-fixture), plus three focused new tests |
| Explanatory documentation | 2 | CPU [backend](../../../../glossary.md#backend) guide and glossary |
| [Planning](../../../../glossary.md#term-planning)/status | 3 | This task, CPU master plan, and roadmap |
| **Total** | **51** | **31 + 15 + 2 + 3** |

The ceiling is larger than an ordinary task because this cohesive change replaces the fixed
four-boundary/three-[node](../../../../glossary.md#node) proving assumptions across the already-completed vertical pipeline. It
does not authorize unrelated cleanup. If implementation needs another production abstraction,
test path, package, [module](../../../../glossary.md#term-module), shared contract, or more than 51 paths, stop and revise planning or
propose a bounded follow-up.

## Acceptance criteria

- Capability reporting exactly matches the semantic/type/attribute/Shape/layout matrix in Scope
  and remains occurrence-local; complete-partition lowering applies the stricter topology,
  fusion, virtuality, alias, publication, boundary, and budget checks.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [Scope glossary](../../../../glossary.md#term-scope); [occurrence glossary](../../../../glossary.md#term-occurrence); [partition glossary](../../../../glossary.md#partition); [lowering glossary](../../../../glossary.md#lowering); [topology glossary](../../../../glossary.md#term-topology); [fusion glossary](../../../../glossary.md#term-fusion); [alias glossary](../../../../glossary.md#aliasing-and-arena); [publication glossary](../../../../glossary.md#term-publication))</span>
- One through eight supported occurrences lower through one family-oriented opcode sequence, one
  canonical IR, one unit, one artifact, one partition executable, and one bound invocation.
  Internal single-use results remain graph/logical-memory values without declarations or slots.<span data-glossary-navigation="true"> ([opcode glossary](../../../../glossary.md#term-opcode); [canonical glossary](../../../../glossary.md#term-canonical-form); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [artifact glossary](../../../../glossary.md#term-artifact); [bound invocation glossary](../../../../glossary.md#bound-invocation--boundinvocation); [graph glossary](../../../../glossary.md#graph))</span>
- Disconnected, fan-out, multi-output, multiple-final-output, over-eight, unsupported, or partially
  supported partitions fail before declarations or artifact access.<span data-glossary-navigation="true"> ([fan-out glossary](../../../../glossary.md#term-fan-out))</span>
- The exact five-type carrier table works for heap, exact segment, and representative mixed
  patterns. Boundary count is derived; default exact-segment policy and explicit pattern count/
  order are validated without a fixed four-entry assumption.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [heap glossary](../../../../glossary.md#term-heap))</span>
- Scalar and parallel-scalar generated/reference results agree for every admitted opcode/type.
  FLOAT64 eligible numeric-only chains also agree under vector and parallel-vector execution;
  other admitted chains deterministically fall back to scalar compute.<span data-glossary-navigation="true"> ([Scalar glossary](../../../../glossary.md#term-scalar); [FLOAT64 glossary](../../../../glossary.md#term-float64); [vector glossary](../../../../glossary.md#term-vector))</span>
- FLOAT32 operations stay in FLOAT32; INT32/INT64 arithmetic is modular; comparisons,
  classifications, WHERE, canonical BOOL results/conditions, same-type CAST, exact scalar bits,
  and existing exact GELU satisfy the precise rules in Scope.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [BOOL glossary](../../../../glossary.md#data-type--datatype); [GELU glossary](../../../../glossary.md#gaussian-error-linear-unit--gelu))</span>
- Structural identity changes for opcode sequence, types, exact scalar bits, topology, access
  form, ordered carrier pattern, vector compute/species, materialized source, numerical mode, or
  generator schema. It excludes compatible extents, concrete strides/offsets, graph/value/slot
  identities, resources, workers, roots, runs, and cost evidence.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [schema glossary](../../../../glossary.md#term-schema); [strides glossary](../../../../glossary.md#term-stride))</span>
- Existing access-plan regimes, span/write-injectivity/alias checks, arbitrary ranges, zero-work
  behavior, one-copy materialization, four-candidate/one-artifact/zero-shape/zero-unroll budgets,
  worker behavior, optional persistence verification, and prepared-executable ownership remain
  enforced.<span data-glossary-navigation="true"> ([injectivity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [materialization glossary](../../../../glossary.md#materialization); [persistence glossary](../../../../glossary.md#term-persistence); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Cross-type CAST, mixed promotion, BFLOAT16, unselected arithmetic/unary semantics, POW,
  reductions/layout/indexing/random work, relaxed math, and native/vendor routes are rejected and
  unadvertised.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [unary glossary](../../../../glossary.md#term-unary))</span>
- Production contains no per-operation planner/emitter/executable class, Model-operation or graph
  interpretation in Runtime, string dispatch, registry, service locator, reflection, hot carrier/
  route/opcode switch, cursor allocation, or per-element division/modulo.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [Model glossary](../../../../glossary.md#term-model); [Runtime glossary](../../../../glossary.md#term-runtime); [dispatch glossary](../../../../glossary.md#term-dispatch); [registry glossary](../../../../glossary.md#term-registry); [service locator glossary](../../../../glossary.md#term-service-locator); [element glossary](../../../../glossary.md#term-element))</span>
- No Java, test, build, architecture, [dependency](../../../../glossary.md#term-dependency), or shared-[module path](../../../../glossary.md#term-module-path) changes outside
  `backends/cpu`; no backend-conformance or integration claim is made before the planned portable
  coverage checkpoint.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- A separate clean documentation-focused agent finalizes affected Javadocs, package summaries,
  CPU guide, glossary, task evidence, master plan, and roadmap in the same overall change. It
  records reasoned no-change conclusions for public API guides, architecture, other modules,
  Gradle, architecture tests, backend-conformance, and integration tests.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [modules glossary](../../../../glossary.md#term-module); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- [CPU](../../../../glossary.md#term-cpu) 0005E remains [`Ready`](../../../../glossary.md#term-ready) until implementation, final CPU tests, documentation review, and all
  validation gates pass. CPU 0005A–0005D remain <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>`Complete`; CPU 0005F and every later CPU task
  remain `Draft` without a detailed specification.

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation development may use focused classes. Before the single final [CPU](../../../../glossary.md#term-cpu) [module](../../../../glossary.md#term-module) [run](../../../../glossary.md#term-run), run
the focused semantic/differential [matrix](../../../../glossary.md#term-matrix-meanings):

```bash
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.CpuCapabilityProviderTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcodeTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPointwisePartitionLoweringTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuPointwiseGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.reference.CpuReferenceDifferentialTest
```

After executable Java stabilizes, run exactly one final affected-module suite:

```bash
./gradlew :backends:cpu:test
```

The ordinary suite must continue to skip the opt-in CPU 0005D [persistence](../../../../glossary.md#term-persistence) timing method. Do not
rerun that timing evidence because persistence remains disabled and this task makes no enablement
claim.

Documentation-focused pass, after final Javadocs and Markdown:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

Also validate:

- local Markdown targets and heading anchors in this task, CPU master plan, roadmap, CPU guide,
  and glossary;<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>
- fences, final newlines, trailing whitespace, exact 51-path ceiling, and no unauthorized paths;
- exact supported/rejected semantic and data-type inventory, all five carrier mappings, family
  opcode/type inventory, public/internal type inventory, and absence of per-operation classes;<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [opcode glossary](../../../../glossary.md#term-opcode); [operation glossary](../../../../glossary.md#operation))</span>
- CPU 0005A–0005D `Complete`, CPU 0005E [`Ready`](../../../../glossary.md#term-ready) until implementation completion, CPU 0005F and
  tasks 0006–0017 `Draft`, and absence of later detailed task specifications; and
- unchanged `ARCHITECTURE.md`, focused architecture/ADR files, other modules, Gradle files,
  architecture tests, <span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Gradle glossary](../../../../glossary.md#term-gradle))</span>`testing/backend-conformance`, and `testing/integration-tests`.

Repository-wide Java validation is deferred to CPU 0009, the portable generated-coverage closure
[checkpoint](../../../../glossary.md#term-checkpoint), or [continuous integration](../../../../glossary.md#term-continuous-integration-ci). The current [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) project is only a
placeholder and adding a CPU [dependency](../../../../glossary.md#term-dependency) would violate this task's cross-module/build exclusions;
the CPU-internal generated/reference differential matrix is the proportionate task-level proof.

The documentation pass reuses the implementation pass's successful Java evidence and must not
repeat it unless executable Java changes afterward or it records a concrete stale-evidence risk.

## Dependencies

- Complete [CPU](../../../../glossary.md#term-cpu) 0005A [partition](../../../../glossary.md#partition)-[kernel](../../../../glossary.md#kernel) reset and CPU-private unit/[IR](../../../../glossary.md#term-intermediate-representation-ir)/route/executable architecture.
- Complete CPU 0005B static right-aligned access plans, [layouts](../../../../glossary.md#layout), carriers, spans, and [broadcasting](../../../../glossary.md#broadcasting).
- Complete CPU 0005C [scalar](../../../../glossary.md#term-scalar)/[vector](../../../../glossary.md#term-vector) and single/parallel strategy implementation.
- Complete CPU 0005D one-copy materialization, specialization budgets, and optional-persistence
  evidence gate.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [specialization glossary](../../../../glossary.md#term-specialization); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- Current [Model](../../../../glossary.md#term-model) [operation](../../../../glossary.md#operation), [`DataType`](../../../../glossary.md#data-type--datatype), [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue), `ShapeBroadcast`, and `LayoutDescriptor`
  contracts; Compiler and Prepare validation already preserve the exact semantic occurrences.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [validation glossary](../../../../glossary.md#term-validation))</span>
- Current staged [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime) cold-[binding](../../../../glossary.md#term-binding)/execution, and [backend](../../../../glossary.md#backend)-contract [ownership](../../../../glossary.md#term-ownership-meanings) contracts.

## Follow-up tasks

- CPU 0005F remains Draft and separately owns exact/default scalar-POW strength reduction. It must
  consume semantic <span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [scalar glossary](../../../../glossary.md#term-scalar))</span>`POW` and cannot broaden the 0005E numerical mode.
- CPU 0006–0008 remain Draft for layout/indexing/random, reduction/scan/normalization, and larger
  computation families.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [scan glossary](../../../../glossary.md#term-cumulative-scan); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- CPU 0009 remains the portable generated-coverage closure checkpoint. Remaining pointwise rows
  excluded here—including extrema, division, cross-type promotion/conversion, BFLOAT16, and the
  other unary functions—must receive a later bounded implementation task and Model semantic
  prerequisites where needed before that checkpoint can claim complete selected coverage. This
  task does not silently assign them to 0005F.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [pointwise glossary](../../../../glossary.md#term-pointwise); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [unary glossary](../../../../glossary.md#term-unary); [Model glossary](../../../../glossary.md#term-model))</span>
- CPU 0010 and later native/tuning/numerical-policy work remains Draft and cannot replace the
  portable fallback or grant relaxed permission.

## Architecture impact

Expected impact: None.

The current architecture explicitly assigns concrete [lowering](../../../../glossary.md#lowering), [backend](../../../../glossary.md#backend)-private [IR](../../../../glossary.md#term-intermediate-representation-ir), [fusion](../../../../glossary.md#term-fusion),
route/strategy selection, generated [artifacts](../../../../glossary.md#term-artifact), physical [CPU](../../../../glossary.md#term-cpu) representations, finalization, and
execution to the CPU backend. The task adds no [module](../../../../glossary.md#term-module) edge, [lifecycle](../../../../glossary.md#lifecycle) stage, [backend identity](../../../../glossary.md#backend-identity--backendid),
public [API](../../../../glossary.md#term-application-programming-interface-api), or architecture rule. If implementation requires one, stop and report the conflict.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/planning/backends/cpu/master-plan.md, and
docs/planning/backends/cpu/tasks/0005e-portable-pointwise-types-carriers-and-semantic-family-expansion.md.
Read the directly referenced completed CPU tasks and current Model/Compiler/Prepare/Runtime/backend
contracts needed to verify each semantic row.

Implement CPU 0005E exactly as specified. Preserve the completed common partition lowering,
CPU-private IR/access/materialization/route/strategy/artifact/executable architecture and the exact
51-path ceiling. Do not invent cross-type CAST semantics, implement excluded pointwise or later
families, change another module/build/architecture contract, create per-operation classes, commit,
or push. Stop and report any semantic, architecture, dependency, or scope conflict.

After executable Java stabilizes and the focused plus one final CPU suite pass, hand the exact diff
and recorded test evidence to a separate clean documentation-focused agent/thread. That pass must
follow docs/developer-guide/documentation-rules.md; independently finalize affected Javadocs, CPU
guide, glossary, planning evidence, links, scope and status checks; record reasoned no-change
conclusions for public APIs and excluded layers; and not repeat successful Java tests unless
executable behavior changes or a concrete risk requires it.

Update this task's decisions, limitations, validation evidence, implementation notes, completion
summary, and synchronized final status. Do not mark Complete until every acceptance criterion and
the documentation pass succeed.
```

## Local decisions

- The first increment chooses basic ordered primitive arithmetic plus predicates and selection.
  Floating extrema, division, broad unary/transcendental functions, BFLOAT16, promotion, and
  cross-type conversion are deferred because they add distinct exceptional-value, accuracy,
  conversion, or representation decisions.<span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [unary glossary](../../../../glossary.md#term-unary); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>
- Cross-type CAST is not a CPU-local decision: current Model documentation explicitly leaves
  numerical conversion undefined. Same-type CAST is included because represented-value identity
  is sufficient and useful for fused compiler-generated chains.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Model glossary](../../../../glossary.md#term-model); [identity glossary](../../../../glossary.md#term-identity-meanings); [fused glossary](../../../../glossary.md#specialization-and-fusion); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- One family-oriented opcode enum and the existing scalar/vector emitter owners replace the fixed
  three-semantic proving vocabulary. Per-operation implementation classes are prohibited.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([opcode glossary](../../../../glossary.md#term-opcode); [scalar glossary](../../../../glossary.md#term-scalar); [vector glossary](../../../../glossary.md#term-vector); [operation glossary](../../../../glossary.md#operation))</span>
- The bounded topology is one connected straight-line chain with at most eight instructions and
  one materialized result. This removes the fixed three-node/four-boundary proof assumption without
  prematurely implementing arbitrary DAG unit scheduling.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [node glossary](../../../../glossary.md#node); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag))</span>
- Scalar/parallel-scalar is the complete fallback for the selected matrix. Vector coverage remains
  intentionally limited to compatible FLOAT64 numeric-only chains and is never inferred from
  carrier availability alone.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [FLOAT64 glossary](../../../../glossary.md#term-float64); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- The existing four-candidate/one-artifact/zero-shape/zero-unroll budget remains authoritative.
  Materialization considers at most three eligible boundaries in stable order rather than growing
  candidates with chain width.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [shape glossary](../../../../glossary.md#shape); [Materialization glossary](../../../../glossary.md#materialization))</span>
- The actual evolved-[IR](../../../../glossary.md#term-intermediate-representation-ir) [fixture](../../../../glossary.md#term-fixture) is
  `CpuGeneratedKernelPersistenceEvidenceTest`; it replaces the planned but unchanged
  `CpuClassFileKernelGeneratorTest` allocation in the fifteen-test path map. This preserves the
  exact 51-path ceiling and records the mechanical schema-5 constructor update truthfully.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>

## Known limitations

- Supported partitions are fully static, resolved-layout, connected straight-line pointwise
  chains of at most eight occurrences with one final materialized output.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [pointwise glossary](../../../../glossary.md#term-pointwise))</span>
- This task does not close the Model pointwise inventory. The excluded exact rows remain
  fail-closed and require later bounded planning before CPU 0009 closure.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [planning glossary](../../../../glossary.md#term-planning))</span>
- [BFLOAT16](../../../../glossary.md#term-bfloat16) storage can still bind as `short[]` or [`MemorySegment`](../../../../glossary.md#term-memorysegment) at the representation layer, but
  no 0005E operation advertises or executes BFLOAT16. FLOAT16 does not exist yet.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [FLOAT16 glossary](../../../../glossary.md#term-float16))</span>
- General cross-type CAST execution remains blocked on a Model-owned numerical conversion
  contract. Same-type CAST does not imply redundant-cast compiler optimization.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler))</span>
- Newly supported FLOAT32, integral, predicate, WHERE, and CAST chains use scalar compute, with
  existing optional parallel orchestration. This task makes no vector or performance claim for
  them.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [scalar glossary](../../../../glossary.md#term-scalar); [vector glossary](../../../../glossary.md#term-vector))</span>
- Backend conformance and end-to-end integration remain deferred because their current project
  surfaces do not yet consume the CPU implementation without prohibited build/dependency work.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Optional class-byte [persistence](../../../../glossary.md#term-persistence) remains disabled by default under the [CPU](../../../../glossary.md#term-cpu) 0005D
  `KEEP_DISABLED` verdict; this task records no new timing evidence.

## [Validation](../../../../glossary.md#term-validation) evidence

The implementation context completed executable validation before this independent documentation
pass. No executable Java or test changed afterward, so the documentation context reused the exact
evidence and did not rerun `:backends:cpu:test`:

- The required focused semantic/differential command from this task passed.
- A broader focused [CPU](../../../../glossary.md#term-cpu) [regression](../../../../glossary.md#term-regression) [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) passed after remediation.
- The sole final `./gradlew :backends:cpu:test` passed. Preserved JUnit XML was independently
  recounted as 24 suites, 73 tests, zero failures, zero errors, and one skipped test. The skip is
  the opt-in CPU 0005D persistence timing measurement; CPU 0005E retains the <span data-glossary-navigation="true"> ([JUnit glossary](../../../../glossary.md#term-junit); [persistence glossary](../../../../glossary.md#term-persistence))</span>`KEEP_DISABLED`
  default and did not rerun that evidence benchmark.

Clean documentation context `/root/cpu_0005e_docs` independently reviewed the final diff, source,
tests, completed CPU 0005A–0005D contracts, public [Model](../../../../glossary.md#term-model)/[Planning](../../../../glossary.md#term-planning)/[Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime) boundaries,
[Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile)/Runtime [API](../../../../glossary.md#term-application-programming-interface-api) documentation, CPU guide, glossary, planning records, and the General,
API/[Javadoc](../../../../glossary.md#term-javadoc), [Backend](../../../../glossary.md#backend) Guide, Planning, Developer Guide, and Example profiles. It changed only
Javadocs/package comments and the five authorized Markdown records. Final documentation commands
and audit results were:

- `./gradlew :backends:cpu:javadoc` passed after final Javadoc edits with only the two expected
  incubating-Vector-module warnings and two warnings for intentionally implicit constructors.<span data-glossary-navigation="true"> ([Vector glossary](../../../../glossary.md#term-vector); [module glossary](../../../../glossary.md#term-module))</span>
- Local Markdown target/anchor, balanced-fence, final-newline, and trailing-whitespace checks
  passed for the CPU guide, glossary, this task, CPU master plan, and roadmap: 696 local links and
  293 explicit anchors were validated.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- The exact scope audit found 24 changed CPU production/package paths, nine changed CPU test paths,
  two explanatory-documentation paths, and three planning/status paths: 38 total, all authorized
  and below the exact 51-path ceiling. The test allocation truthfully includes the persistence
  evidence fixture and leaves six permitted test paths unchanged.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [fixture glossary](../../../../glossary.md#term-fixture))</span>
- The semantic audit confirmed one nineteen-opcode vocabulary; exact carrier types FLOAT64,
  FLOAT32, INT32, INT64, and BOOL; schema 5; same-type CAST only; dynamic derived boundary count;
  one-to-eight connected straight-line chains; virtual intermediates and one final store; scalar
  generated coverage for every admitted row; FLOAT64 numeric-only vector eligibility; all-heap,
  all-segment, and mixed binding; and cold canonical-BOOL validation.<span data-glossary-navigation="true"> ([opcode glossary](../../../../glossary.md#term-opcode); [carrier glossary](../../../../glossary.md#term-storage-carrier); [FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [BOOL glossary](../../../../glossary.md#data-type--datatype); [schema glossary](../../../../glossary.md#term-schema); [scalar glossary](../../../../glossary.md#term-scalar); [heap glossary](../../../../glossary.md#term-heap); [binding glossary](../../../../glossary.md#term-binding); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Package/type and hot-boundary checks found no per-operation implementation classes, later route
  leaf, registry, service locator, Model operation or graph interpretation in Runtime, hot opcode/
  carrier/route dispatch, or new public supported CPU type.<span data-glossary-navigation="true"> ([leaf glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [registry glossary](../../../../glossary.md#term-registry); [service locator glossary](../../../../glossary.md#term-service-locator); [graph glossary](../../../../glossary.md#graph); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Status checks confirmed CPU 0005A–0005E `Complete`; CPU 0005F and tasks 0006–0017 remain `Draft`,
  and no later detailed CPU task specification exists.
- `git diff --check` passed after final documentation and status edits.

## Implementation notes

[CPU](../../../../glossary.md#term-cpu) capability and [lowering](../../../../glossary.md#lowering) now map the exact admitted [occurrence](../../../../glossary.md#term-occurrence) [matrix](../../../../glossary.md#term-matrix-meanings) into one nineteen-[opcode](../../../../glossary.md#term-opcode)
CPU-private vocabulary. Lowering accepts one through eight connected straight-line occurrences,
derives external reads deterministically, retains internal single-use values as virtual [IR](../../../../glossary.md#term-intermediate-representation-ir) values,
and emits exactly one final materialized store. Boundary [cardinality](../../../../glossary.md#term-cardinality) is derived rather than fixed;
default analysis selects one exact segment [carrier](../../../../glossary.md#term-storage-carrier) per boundary, while explicit patterns support
the exact typed primitive-array forms and mixed [bindings](../../../../glossary.md#term-binding).

Generated and reference [scalar](../../../../glossary.md#term-scalar) execution cover [FLOAT64](../../../../glossary.md#term-float64), [FLOAT32](../../../../glossary.md#term-float32), [INT32](../../../../glossary.md#data-type--datatype), [INT64](../../../../glossary.md#data-type--datatype), and [BOOL](../../../../glossary.md#data-type--datatype) results
for every admitted opcode row. [Vector](../../../../glossary.md#term-vector) and parallel-vector remain restricted to FLOAT64 numeric-
only opcode sequences that satisfy the existing access-[run](../../../../glossary.md#term-run) rules; all other admitted rows use
scalar or parallel-scalar compute. Cold binding validates [canonical](../../../../glossary.md#term-canonical-form) BOOL bytes. The existing
access regimes, optional one-FLOAT64-input [materialization](../../../../glossary.md#materialization), strategy/orchestration, [artifact](../../../../glossary.md#term-artifact),
[persistence](../../../../glossary.md#term-persistence), direct invocation, and [ownership](../../../../glossary.md#term-ownership-meanings) boundaries remain intact; the generator [schema](../../../../glossary.md#term-schema) is 5.

No public [Tensor](../../../../glossary.md#tensor), [Compile](../../../../glossary.md#compile), [Runtime](../../../../glossary.md#term-runtime), [Prepare](../../../../glossary.md#prepare), [Planning](../../../../glossary.md#term-planning), [Backend](../../../../glossary.md#backend) Contract, Config, [Trace](../../../../glossary.md#trace), [Engine](../../../../glossary.md#term-engine),
[OpenBLAS](../../../../glossary.md#term-openblas) provider, [architecture contract](../../../../glossary.md#architecture-contract), focused architecture explanation, [ADR](../../../../glossary.md#term-architecture-decision-record-adr), [module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency),
[Gradle](../../../../glossary.md#term-gradle) configuration, [architecture test](../../../../glossary.md#term-architecture-test), backend-[conformance](../../../../glossary.md#term-conformance) test, or [integration test](../../../../glossary.md#term-integration-test) changed.
Tensor and Compile [API](../../../../glossary.md#term-application-programming-interface-api) guides remain accurate because [Model](../../../../glossary.md#term-model) semantics and [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) behavior did
not change; the Runtime API remains accurate because CPU 0005E implements the existing cold-binding
and prepared-executable contracts privately. Backend-conformance and integration remain deferred
because their current projects do not consume CPU without out-of-[scope](../../../../glossary.md#term-scope) build/dependency changes.

## Completion summary

- Completed changes: implemented and documented the bounded five-type portable pointwise matrix,
  family-oriented nineteen-opcode IR, dynamic boundary cardinality, typed heap/segment carriers,
  scalar execution for all admitted rows, and FLOAT64 numeric-only vector eligibility.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [opcode glossary](../../../../glossary.md#term-opcode); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [cardinality glossary](../../../../glossary.md#term-cardinality); [heap glossary](../../../../glossary.md#term-heap); [scalar glossary](../../../../glossary.md#term-scalar); [FLOAT64 glossary](../../../../glossary.md#term-float64); [vector glossary](../../../../glossary.md#term-vector))</span>
- Files changed or created: 24 CPU production/package paths, nine CPU tests, the CPU guide,
  glossary, this task, CPU master plan, and roadmap; 38 total under the 51-path ceiling.
- Tests and validation: reused the successful required focused matrix, broader focused regression,
  and sole final 24-suite/73-test CPU pass with zero failures/errors and one evidence skip; final
  CPU Javadoc, Markdown, scope, semantic, status, and whitespace gates passed.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [regression glossary](../../../../glossary.md#term-regression); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: `/root/cpu_0005e_docs` independently finalized affected Javadocs,
  package summaries, CPU guide, glossary, and synchronized planning/status evidence.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: current pointwise coverage, topology, carrier, vector fallback, BOOL,
  materialization, artifact, and excluded-family boundaries are explicit without new authority.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [carrier glossary](../../../../glossary.md#term-storage-carrier); [BOOL glossary](../../../../glossary.md#data-type--datatype); [materialization glossary](../../../../glossary.md#materialization); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Javadoc review: affected CPU contracts now describe the derived-boundary, typed opcode/carrier,
  scalar-immediate, BOOL validation, specialization, and cold/hot lifecycle contracts.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Glossary impact: CPU portable route, generated kernel, specialization, artifact, plan,
  executable, IR, access, and materialization entries now reflect completed CPU 0005E.<span data-glossary-navigation="true"> ([CPU portable route glossary](../../../../glossary.md#cpu-portable-route); [kernel glossary](../../../../glossary.md#kernel))</span>
- Architecture/build/[dependency](../../../../glossary.md#term-dependency) impact: None.
- Unresolved issues: None.
- Follow-up required: None. [CPU](../../../../glossary.md#term-cpu) 0005F and later work remain Draft.

Status: Complete
