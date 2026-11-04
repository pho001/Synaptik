# Task 0007A: Portable Ordinary [Extrema](../../../../glossary.md#saturation-pole-and-extremum) and [Boolean](../../../../glossary.md#data-type--datatype) Reductions

## Reader context

Ordinary aggregate geometry associates each output cell with all input positions selected for
its reduction. CPU analysis derived that geometry once, and scalar or parallel-scalar execution
assigned complete output cells to ranges. No worker computed a partial reduction that another
worker later combined; extrema and Boolean rows therefore needed no accumulator workspace here.

An empty selected domain still had a result identity when an output cell existed, while an empty
output domain had no cell to write. Those are different cases. This checkpoint’s generated entry
called a CPU-owned static body rather than containing the reduction loop; later parity work
replaced that bridge. The current aggregate reference distinguishes axis removal/retention and
public expression descriptions from actual value reduction.

Current reading alongside this historical record:

- [Numeric aggregate expressions](../../../../reference/tensor/reductions.md#numeric-aggregate-expressions)
  — distinguishes output-shape derivation, selected domains, and current numerical semantics.
- [CPU generated-code ownership](../../../../explanation/packages/cpu.md#iogithubpho001synaptikbackendcpuinternalcodegenemit)
  — separates generation-time emitters, emitted typed work, and JVM compilation.

## Status

Complete

## Goal

Add the next bounded [CPU](../../../../glossary.md#term-cpu) frontier after completed CPU 0007: execute exactly one fully static,
resolved-[layout](../../../../glossary.md#layout) current [Model](../../../../glossary.md#term-model) ordinary `MIN`, `MAX`, `ALL`, or `ANY` [occurrence](../../../../glossary.md#term-occurrence) through the
portable generated-[kernel](../../../../glossary.md#kernel) route.

This task establishes one shared full/single-[axis](../../../../glossary.md#term-axis)/multi-axis aggregate geometry without also
choosing a floating summation or product algorithm. Each output cell owns one complete reduction
domain, visits that domain in [canonical](../../../../glossary.md#term-canonical-form) logical [row-major](../../../../glossary.md#term-row-major) order, and may execute independently in
parallel. No worker splits or combines one reduction domain, and the task declares no workspace,
partial [buffer](../../../../glossary.md#term-buffer), or hidden [accumulator](../../../../glossary.md#accumulator-type) resource.

## Why [CPU](../../../../glossary.md#term-cpu) 0007A was split

The former row combined three independently implementable boundaries:

1. ordinary selector and [boolean](../../../../glossary.md#data-type--datatype) folds (`MIN`, `MAX`, `ALL`, and `ANY`);
2. ordinary numerical accumulation (`SUM`, [`MEAN`](../../../../glossary.md#term-mean), and `PROD`); and
3. [binding](../../../../glossary.md#term-binding)-aware [target](../../../../glossary.md#term-target)-[Shape](../../../../glossary.md#shape) `SUM`.

The first boundary has one output-domain geometry, represented-value [extrema](../../../../glossary.md#saturation-pole-and-extremum)/boolean state, fixed
empty [identities](../../../../glossary.md#term-identity-meanings), and zero scratch. The second must separately implement [Model](../../../../glossary.md#term-model)'s exact-real
floating `SUM`/`MEAN` target, product special values, [finite](../../../../glossary.md#term-finite)-result [rounding](../../../../glossary.md#term-rounding), and any required
declared [accumulator](../../../../glossary.md#accumulator-type) resource; reusing CPU 0007's per-step typed [scan](../../../../glossary.md#term-cumulative-scan) policy would not satisfy the
ordinary `SUM` contract. The third adds right-aligned bound-Shape compatibility and a different
[axis](../../../../glossary.md#term-axis)-selection geometry. Combining them would pre-decide two numerical/resource designs and one
binding-aware geometry in a task whose executable foundation does not require them.

CPU 0007A therefore owns only the first cohesive boundary. Draft CPU 0007A1 follows with ordinary
`SUM`/`MEAN`/`PROD`; Draft CPU 0007A2 follows with binding-aware `SUM_TO_SHAPE`. This refinement
does not change the [ownership](../../../../glossary.md#term-ownership-meanings) or order of CPU 0007B–0007F.

## [Scope](../../../../glossary.md#term-scope)

### Exact [occurrence](../../../../glossary.md#term-occurrence) and attribute [matrix](../../../../glossary.md#term-matrix-meanings)

- Admit exactly one [CPU](../../../../glossary.md#term-cpu)-owned [node](../../../../glossary.md#node) whose kind is `AggregateReductionKind.MIN`, `MAX`, `ALL`, or
  `ANY`, with one ordered input and one output.
- Admit exactly these current ordinary attribute forms:
  - exact `NoOperationAttrs.INSTANCE`, meaning full reduction over every input axis with the
    canonical scalar output;<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [scalar glossary](../../../../glossary.md#term-scalar))</span>
  - exact `AxisReductionAttrs`, carrying one already normalized input [axis](../../../../glossary.md#term-axis) and its exact
    `keepDimensions` flag; or
  - exact `MultiAxisReductionAttrs`, carrying distinct already normalized axes and the exact
    `keepDimensions` flag. An empty axis list selects a one-element point domain at every input
    position; it is not a full reduction.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element))</span>
- Reject `SUM`, [`MEAN`](../../../../glossary.md#term-mean), `PROD`, `ARG_MIN`, `ARG_MAX`, masked, statistical, target-Shape, and every
  other kind or attributes class. Do not reinterpret another signature as ordinary reduction.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [Shape glossary](../../../../glossary.md#shape); [signature glossary](../../../../glossary.md#term-signature))</span>
- Revalidate the current compiler-produced occurrence: exactly one input/output, exact attributes,
  fully static descriptors, exact Model result Shape and type, and resolved layouts. CPU prepare
  must not assume that Tensor construction was the only producer of the captured node.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [descriptors glossary](../../../../glossary.md#term-descriptor); [Model glossary](../../../../glossary.md#term-model); [layouts glossary](../../../../glossary.md#layout); [prepare glossary](../../../../glossary.md#prepare); [Tensor glossary](../../../../glossary.md#tensor); [producer glossary](../../../../glossary.md#term-producer))</span>

### Exact data-type and result [matrix](../../../../glossary.md#term-matrix-meanings)

| Kind | Accepted input types | Output type |
|---|---|---|
| `MIN` | [FLOAT64](../../../../glossary.md#term-float64), [FLOAT32](../../../../glossary.md#term-float32), [BFLOAT16](../../../../glossary.md#term-bfloat16), [INT32](../../../../glossary.md#data-type--datatype), [INT64](../../../../glossary.md#data-type--datatype) | exact input type |
| `MAX` | FLOAT64, FLOAT32, BFLOAT16, INT32, INT64 | exact input type |
| `ALL` | [BOOL](../../../../glossary.md#data-type--datatype) only | BOOL |
| `ANY` | BOOL only | BOOL |

There is no promotion, widened result, cast, integer truth conversion, or floating truth
conversion. BFLOAT16 comparisons use the exact represented BFLOAT16 value widened only for
comparison; the selected output remains an exact BFLOAT16 represented value. BOOL input follows
the current [canonical](../../../../glossary.md#term-canonical-form) one-byte representation and every output is canonical false or true.

### [Shape](../../../../glossary.md#shape), domains, and empty behavior

- Full reduction selects every input axis and produces the canonical scalar Shape, including for
  rank-zero input. A rank-zero scalar contributes its one value. An input Shape with any
  zero-extent axis has an empty full domain and still has one scalar output identity.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [extent glossary](../../../../glossary.md#term-extent); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Single-[axis](../../../../glossary.md#term-axis) reduction selects exactly the [normalized axis](../../../../glossary.md#normalized-axis). `keepDimensions=false` removes it;
  `true` retains it with extent one. A zero selected extent creates an empty domain independently
  for every output cell.
- Multi-axis reduction selects axis membership, not attribute-list order. Removal drops every
  selected axis; retention replaces each selected extent with one. Empty axes preserve the input
  Shape and create a point domain containing exactly the corresponding input value, irrespective
  of <span data-glossary-navigation="true"> ([Multi-axis reduction glossary](../../../../glossary.md#multi-axis-reduction))</span>`keepDimensions`.
- An output cell exists for each row-major coordinate over unselected input axes. If any
  unselected extent is zero, the output has zero logical elements: no identity is materialized,
  no generated call occurs, and no worker is submitted.<span data-glossary-navigation="true"> ([row-major glossary](../../../../glossary.md#term-row-major))</span>
- Selected zero extents make each otherwise existing domain empty. Materialize these exact
  identities:

| Kind/type | Empty-domain result |
|---|---|
| floating `MIN` | [positive infinity](../../../../glossary.md#nan-infinity-and-signed-zero) in the result format |
| floating `MAX` | [negative infinity](../../../../glossary.md#nan-infinity-and-signed-zero) in the result format |
| [INT32](../../../../glossary.md#data-type--datatype) `MIN` / [INT64](../../../../glossary.md#data-type--datatype) `MIN` | `Integer.MAX_VALUE` / `Long.MAX_VALUE` |
| INT32 `MAX` / INT64 `MAX` | `Integer.MIN_VALUE` / `Long.MIN_VALUE` |
| [BOOL](../../../../glossary.md#data-type--datatype) `ALL` | [canonical](../../../../glossary.md#term-canonical-form) true |
| BOOL `ANY` | canonical false |

Zero-[element](../../../../glossary.md#term-element) input/output Shapes are valid when these rules permit them. All element counts,
domain counts, [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) products, referenced spans, and byte sizes are checked during cold
[lowering](../../../../glossary.md#lowering)/[binding](../../../../glossary.md#term-binding); [overflow](../../../../glossary.md#term-overflow) fails before execution.

### Traversal, [extrema](../../../../glossary.md#saturation-pole-and-extremum), and determinism

- For each output cell, traverse selected coordinates in canonical input logical row-major order:
  the greatest input-axis index changes fastest. Physical offsets and strides never determine
  logical order. Attribute order for multi-axis reduction does not alter traversal.<span data-glossary-navigation="true"> ([coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [canonical glossary](../../../../glossary.md#term-canonical-form); [row-major glossary](../../../../glossary.md#term-row-major); [axis index glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [strides glossary](../../../../glossary.md#term-stride); [multi-axis reduction glossary](../../../../glossary.md#multi-axis-reduction))</span>
- INT32 and INT64 extrema use exact signed order. BOOL uses conjunction/disjunction over each
  logical value and may short-circuit only within the same canonical domain; short-circuiting must
  not change any observable result, failure, write, or worker behavior.<span data-glossary-navigation="true"> ([INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- Floating non-[NaN](../../../../glossary.md#term-nan) extrema use numerical order with [infinities](../../../../glossary.md#nan-infinity-and-signed-zero) ordered normally. `MIN` selects
  negative zero when either zero sign occurs; <span data-glossary-navigation="true"> ([negative zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>`MAX` selects [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) when either occurs.
- A NaN anywhere in a floating domain produces NaN. The CPU-private deterministic policy selects
  the represented bits of the first NaN in canonical logical traversal order; it does not promise
  that payload choice across backends or future schema versions. Non-NaN selected values retain
  their exact represented bits.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [represented bits glossary](../../../../glossary.md#term-represented-bits); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [backends glossary](../../../../glossary.md#backend); [schema glossary](../../../../glossary.md#term-schema))</span>
- Scalar and parallel-scalar execution use the same per-output-cell traversal and must be bitwise
  identical. Floating arithmetic is not performed, BFLOAT16 is never rounded or accumulated, and
  integers never overflow because extrema only compare represented values.<span data-glossary-navigation="true"> ([Scalar glossary](../../../../glossary.md#term-scalar); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [overflow glossary](../../../../glossary.md#term-overflow))</span>

### Generated execution, parallelism, and resources

- Add focused [CPU](../../../../glossary.md#term-cpu)-private `CpuAggregateIr`, `CpuAggregateLowering`, and `CpuAggregateEmitter`
  owners in the existing IR, lowering, and Class-File emission packages. A different equally
  focused <span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lowering glossary](../../../../glossary.md#lowering))</span>`ExtremaBoolean` name is acceptable only if every integration seam uses it consistently.
- Lower one occurrence to one computation unit and one generated artifact. The declared boundary
  and buffer order is exactly <span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [artifact glossary](../../../../glossary.md#term-artifact); [buffer glossary](../../../../glossary.md#term-buffer))</span>`[input, output]`. There is no materialization, workspace, partial
  result, combine buffer, per-domain heap object, persistent accumulator, or shared mutable state.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [heap glossary](../../../../glossary.md#term-heap); [accumulator glossary](../../../../glossary.md#accumulator-type); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- Generated primitive `start`/`end` bounds denote a contiguous range of independent flattened
  output-cell ordinals. Scalar execution covers all output cells; parallel-scalar execution
  partitions only this output domain into deterministic disjoint ranges. Never split a reduction
  domain across workers and never perform partial/combine work.<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [Scalar glossary](../../../../glossary.md#term-scalar))</span>
- The generated entry is a direct Class-File bridge to a CPU-owned static aggregate body, following
  current scan/fold structure. The task does not require the reduction loop to be embedded in the
  generated bytecode and must not document the bridge as if it were embedded emission.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- Cold per-range geometry may allocate one invocation-private primitive `long[]` coordinate pack,
  as the current scan/fold routes do. That pack is prepared invocation state, not numerical
  workspace. The body allocates no object per output cell or selected element.<span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [element glossary](../../../../glossary.md#term-element))</span>
- Vector and parallel-vector bodies are out of scope. Empty outputs perform no generated call;
  one output cell is scalar. Parallelism requires multiple independent output cells.<span data-glossary-navigation="true"> ([Vector glossary](../../../../glossary.md#term-vector); [scope glossary](../../../../glossary.md#term-scope))</span>

### [Layouts](../../../../glossary.md#layout), carriers, and pre-mutation [validation](../../../../glossary.md#term-validation)

- Support arbitrary currently supported fully resolved input/output layouts with non-negative
  storage offsets and strides: dense, offset, positive-strided, interleaved, transposed, and
  zero-strided reads; and arbitrary proven-injective writable output layouts.<span data-glossary-navigation="true"> ([strides glossary](../../../../glossary.md#term-stride); [injective glossary](../../../../glossary.md#term-injective))</span>
- Use exact current carriers: `double[]`, `float[]`, `short[]`, `int[]`, `long[]`, or `byte[]` for
  the matching type, accessible native-order [`MemorySegment`](../../../../glossary.md#term-memorysegment), and compatible heap/segment mixed
  boundaries. Retain current ownership, alignment, accessibility, writability, and size rules.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [alignment glossary](../../../../glossary.md#term-alignment))</span>
- Require a distinct output value/binding and an injective output layout. Use the completed bounded
  injectivity decision, including exact enumeration for the currently permitted small-layout
  bound and the monotone-span sufficient proof above it; do not replace it with a stricter dense
  requirement or an unbounded address set.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [injectivity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity))</span>
- Validate complete physical input/output referenced spans, not range prefixes. Reject any actual
  complete-span overlap before coordinate-pack mutation, output initialization/write, generated
  invocation, or worker submission. Disjoint slices of one underlying allocation remain legal
  when their complete referenced byte spans do not intersect.<span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Every carrier, layout, Shape, type, address, span, overlap, and resource failure occurs before
  the first observable output mutation. No failure may expose a partially reduced output.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [Shape glossary](../../../../glossary.md#shape))</span>

### [IR](../../../../glossary.md#term-intermediate-representation-ir), [specialization](../../../../glossary.md#term-specialization), compatibility, and capability

- `CpuAggregateIr` is the canonical CPU-private semantic structure. It includes kind, represented
  data type, ordinary form (<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [CPU glossary](../../../../glossary.md#term-cpu); [data type glossary](../../../../glossary.md#data-type--datatype))</span>`FULL`, `SINGLE_AXIS`, or `MULTI_AXIS`), canonical selected-axis
  membership, retention policy, structural input/output access plans, deterministic floating-NaN
  selection policy, output-domain range meaning, and zero-workspace policy.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [NaN glossary](../../../../glossary.md#term-nan))</span>
- Canonicalize multi-axis membership in increasing normalized-axis order for geometry and
  compatible artifact identity; retain/revalidate that the original attributes contain distinct
  normalized axes. Attribute list order is not a traversal or result fact.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Specialization/fingerprint/persistence identity additionally includes ranks, boundary roles and
  count, carrier pattern, compute mode, generated entry descriptor, and every other fact that
  changes generated bytes or compatibility. Concrete compatible extents, offsets, stride
  magnitudes, element/domain counts, slots, carrier instances, addresses, run/worker identity, and
  selected range count remain cold when they do not change emitted bytes.<span data-glossary-navigation="true"> ([fingerprint glossary](../../../../glossary.md#term-fingerprint); [persistence glossary](../../../../glossary.md#term-persistence); [carrier glossary](../../../../glossary.md#term-storage-carrier); [descriptor glossary](../../../../glossary.md#term-descriptor); [stride glossary](../../../../glossary.md#term-stride); [element glossary](../../../../glossary.md#term-element); [run glossary](../../../../glossary.md#term-run))</span>
- Advance generated compatibility from schema 20 to schema 21 with no migration reader. A schema-
  20 artifact is incompatible; corrupt/incompatible artifacts remain rejected before definition.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>
- Extend the independent `CpuScalarReferenceKernel` without calling the production aggregate body,
  generated bridge, lowering helper, or production coordinate packer. The oracle follows logical
  Shape coordinates and the exact identities/NaN/zero rules above.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [oracle glossary](../../../../glossary.md#term-test-oracle); [Shape glossary](../../../../glossary.md#shape))</span>
- `CpuCapabilityProvider.supports` may return true only when the complete one-node partition can
  pass lowering, declarations, assignment/finalization, artifact generation, binding, complete
  overlap validation, and execution for the exact matrix. Occurrence-local truth must not imply
  that a mixed or multi-node partition is supported before complete partition lowering exists.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [partition glossary](../../../../glossary.md#partition); [binding glossary](../../../../glossary.md#term-binding); [validation glossary](../../../../glossary.md#term-validation); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [Occurrence glossary](../../../../glossary.md#term-occurrence))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- ordinary `SUM`, [`MEAN`](../../../../glossary.md#term-mean), or `PROD`, including any exact/reproducible floating accumulator,
  reciprocal division, product parity implementation, partial reduction, combine tree, or scratch<span data-glossary-navigation="true"> ([accumulator glossary](../../../../glossary.md#accumulator-type))</span>
- [binding](../../../../glossary.md#term-binding)-aware `SumToShapeAttrs`/`SUM_TO_SHAPE`, right-aligned target-Shape axis derivation, or
  unresolved/dynamic bound Shapes<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [Shape glossary](../../../../glossary.md#shape); [axis glossary](../../../../glossary.md#term-axis))</span>
- `ARG_MIN`, `ARG_MAX`, masked reductions, `LOG_SUM_EXP`, variance, standard deviation, L1/L2
  norms, softmax/log-softmax, layer/RMS/batch normalization, attention, or fusion<span data-glossary-navigation="true"> ([variance glossary](../../../../glossary.md#term-variance); [standard deviation glossary](../../../../glossary.md#term-standard-deviation); [L1 glossary](../../../../glossary.md#term-l1); [L2 glossary](../../../../glossary.md#term-l2); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [log-softmax glossary](../../../../glossary.md#softmax--log-softmax); [batch normalization glossary](../../../../glossary.md#batch-normalization--batch-normalization-inference-and-training); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- more than one node, general partition-DAG decomposition, reduction epilogues, materialized
  splits, in-place execution, output overlap, negative offsets/strides, unresolved layouts, or
  runtime-bound axes/extents<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [partition glossary](../../../../glossary.md#partition); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [decomposition glossary](../../../../glossary.md#term-decomposition); [strides glossary](../../../../glossary.md#term-stride); [layouts glossary](../../../../glossary.md#layout); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- vector reduction, within-domain parallelism, partial/combine work, atomics, hidden resources,
  native/provider routes, tuning, benchmarks, relaxed numerics, or performance claims<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector))</span>
- Model, Compiler, shared Prepare/Runtime, Config, Trace, Engine, NN, training, gradient, public API,
  architecture, ADR, dependency, Gradle/Java-version, or build-structure changes<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Prepare glossary](../../../../glossary.md#prepare); [Trace glossary](../../../../glossary.md#trace); [Engine glossary](../../../../glossary.md#term-engine); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [gradient glossary](../../../../glossary.md#term-gradient); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- new architecture-test, backend-conformance, or integration infrastructure; portable closure and
  repository-wide conformance remain owned by CPU 0009<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [CPU glossary](../../../../glossary.md#term-cpu))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Runtime/prepare/backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend))</span>
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Roadmap](../../../roadmap.md)
- [CPU master plan](../master-plan.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- [CPU 0005A partition-kernel reset](0005a-atomic-partition-kernel-architecture-reset.md)<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [kernel glossary](../../../../glossary.md#kernel))</span>
- [CPU 0005C vector and parallel strategies](0005c-vector-and-parallel-portable-strategies.md)<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector))</span>
- [CPU 0005D evidence gate](0005d-materialization-specialization-and-persistence-evidence-gate.md)
- [CPU 0005E typed portable family expansion](0005e-portable-pointwise-types-carriers-and-semantic-family-expansion.md)
- [CPU 0006 static affine views](0006-portable-static-affine-views-and-boundary-materialization.md)<span data-glossary-navigation="true"> ([views glossary](../../../../glossary.md#term-view-meanings))</span>
- [CPU 0006B2 overlap fold](0006b2-portable-overlap-fold.md)
- [CPU 0006C stable ordering](0006c-portable-stable-ordering-and-selection.md)
- [CPU 0007 cumulative scans](0007-portable-cumulative-scan-coverage.md)
- [Model 0018U1 integral reductions](../../../modules/model/tasks/0018u1-integral-reductions-and-arg-min-normalization.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model 0018V multi-axis reductions](../../../modules/model/tasks/0018v-multi-axis-and-statistical-reductions.md)<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [CPU backend guide](../../../../backend-guide/cpu-backend.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- Model owns operation identity, attribute/Shape/type eligibility, extrema/boolean meaning, empty
  identities, and signed-zero/NaN requirements. CPU owns only revalidation, private IR/geometry,
  deterministic realization, resource declarations, generated route, binding, and execution.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [operation glossary](../../../../glossary.md#operation); [identity glossary](../../../../glossary.md#term-identity-meanings); [Shape glossary](../../../../glossary.md#shape); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [boolean glossary](../../../../glossary.md#data-type--datatype); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN glossary](../../../../glossary.md#term-nan); [CPU glossary](../../../../glossary.md#term-cpu); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [binding glossary](../../../../glossary.md#term-binding))</span>
- CPU analysis declares exactly two buffers and zero workspace before CPU-blind shared assignment.
  Finalization validates assignments before realizing exactly one compatible generated artifact.<span data-glossary-navigation="true"> ([buffers glossary](../../../../glossary.md#term-buffer); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Prepared recipes are [immutable](../../../../glossary.md#term-immutable) and reusable. Concurrent runs use distinct [`RunState`](../../../../glossary.md#run-state--runstate) values and
  invocation-private coordinate packs; no prepared accumulator or mutable reduction state is
  shared across runs.<span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [accumulator glossary](../../../../glossary.md#accumulator-type); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- Runtime receives one direct prepared invocation and no operation, graph, axis list, route choice,
  layout discovery, resource discovery, or semantic dispatcher.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [graph glossary](../../../../glossary.md#graph); [axis glossary](../../../../glossary.md#term-axis); [layout glossary](../../../../glossary.md#layout))</span>
- Work remains inside `backends/cpu`, adds no dependency/package/public supported type, and
  preserves the existing generated portable route and optional persistence lifecycle.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [persistence glossary](../../../../glossary.md#term-persistence); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Capability truth may not exceed complete-partition lowering and executable evidence. Stop if
  exact behavior requires Model/compiler/shared-contract changes, numerical accumulation, hidden
  scratch, within-domain combination, or another semantic family.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [lowering glossary](../../../../glossary.md#lowering); [compiler glossary](../../../../glossary.md#term-compiler))</span>

## Package impact

No package is added, removed, moved, or exported.

- `io.github.pho001.synaptik.backend.cpu` remains the sole supported [fail-closed](../../../../glossary.md#term-fail-closed) capability [owner](../../../../glossary.md#term-owner).
- `internal.ir.CpuAggregateIr` owns [immutable](../../../../glossary.md#term-immutable) ordinary-form/kind/type/access/determinism [identity](../../../../glossary.md#term-identity-meanings).
- `internal.lowering.CpuAggregateLowering` owns one-node revalidation, selected-axis membership,
  checked layouts, output/domain counts, declarations, and compact cold geometry.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [axis glossary](../../../../glossary.md#term-axis); [layouts glossary](../../../../glossary.md#layout))</span>
- `internal.codegen.emit.CpuAggregateEmitter` owns the direct generated bridge and CPU-owned static
  scalar body with represented-value comparison/load/store mechanics.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Existing prepare, portable-route, cache, executable, memory, and reference packages gain only
  the focused integrations required by this family.<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare); [cache glossary](../../../../glossary.md#term-cache))</span>
- Tests mirror the corresponding production packages. Add no generic reduction registry,
  accumulator abstraction, manager, service, cursor, or utility package.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [accumulator glossary](../../../../glossary.md#accumulator-type))</span>

## Affected files

Expected production/package paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPortableKernelIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuAggregateIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuAggregateLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuAggregateEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecialization.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/CpuPortableRoutePlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutable.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuScalarReferenceKernel.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/package-info.java`

Expected [CPU](../../../../glossary.md#term-cpu) tests:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuAggregateIrTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuAggregateLoweringTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuAggregateGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGeneratorTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutableTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuReferenceDifferentialTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/memory/CpuBufferBindingTest.java`

Expected documentation/[planning](../../../../glossary.md#term-planning) paths during implementation:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- this task
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

No [Model](../../../../glossary.md#term-model)/[compiler](../../../../glossary.md#term-compiler)/shared [Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime)/[API](../../../../glossary.md#term-application-programming-interface-api)/architecture/[Gradle](../../../../glossary.md#term-gradle)/[conformance](../../../../glossary.md#term-conformance)/integration path is
expected to change.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most 42 paths: 24 production/package paths, 13 [CPU](../../../../glossary.md#term-cpu) test paths,
and 5 documentation/[planning](../../../../glossary.md#term-planning) paths. Stop before a 43rd path, a new package, or any path outside the
listed CPU [module](../../../../glossary.md#term-module) and documentation/planning set. An existing listed integration or inventory test
may remain unchanged, but an unlisted replacement must not increase the ceiling.

If implementation requires numerical accumulation, [target](../../../../glossary.md#term-target)-[Shape](../../../../glossary.md#shape) geometry, a shared contract,
another workspace, another semantic family, or more paths, stop and report the exact conflict. Do
not conceal incomplete coverage behind capability truth or a follow-up.

## Acceptance criteria

- Capability and [lowering](../../../../glossary.md#lowering) admit exactly one fully static resolved-[layout](../../../../glossary.md#layout) ordinary `MIN`, `MAX`,
  `ALL`, or `ANY` [occurrence](../../../../glossary.md#term-occurrence) with exact full/single-/multi-[axis](../../../../glossary.md#term-axis) attributes and type [matrix](../../../../glossary.md#term-matrix-meanings).
- Full, removed/retained single-axis, ordered-but-membership-equivalent multi-axis, and empty-axis
  point forms derive the exact current Model output Shape and domain geometry.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Shape glossary](../../../../glossary.md#shape))</span>
- FLOAT64/FLOAT32/BFLOAT16 extrema, INT32/INT64 extrema, and BOOL folds produce exact output types,
  empty identities, infinity/NaN/signed-zero/signed-order behavior, and canonical BOOL results.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [BOOL glossary](../../../../glossary.md#data-type--datatype); [infinity glossary](../../../../glossary.md#term-infinity); [NaN glossary](../../../../glossary.md#term-nan); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Every output cell follows canonical logical row-major selected-domain traversal; first-logical-
  NaN selection and scalar/parallel-scalar output bits are deterministic.<span data-glossary-navigation="true"> ([row-major glossary](../../../../glossary.md#term-row-major); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Parallelism is only across complete independent output cells. There is zero declared workspace,
  no partial/combine work, no hidden accumulator, and no vector route.<span data-glossary-navigation="true"> ([accumulator glossary](../../../../glossary.md#accumulator-type); [vector glossary](../../../../glossary.md#term-vector))</span>
- Dense, offset, positive/zero-strided, transposed/interleaved reads; injective non-dense writes;
  heap/native/mixed carriers; scalars; and selected/unselected zero extents match an independent
  reference oracle that does not invoke production aggregate helpers.<span data-glossary-navigation="true"> ([injective glossary](../../../../glossary.md#term-injective); [heap glossary](../../../../glossary.md#term-heap); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- Complete physical overlap and every binding/layout/resource failure occur before coordinate
  mutation, output writes, generated calls, or worker submission.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- The Class-File route is documented and tested as a generated direct bridge to a CPU-owned static
  body. Hot loops allocate no per-cell/per-element object and inspect no Model/graph/runtime policy.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [element glossary](../../../../glossary.md#term-element); [graph glossary](../../../../glossary.md#graph); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Schema 21 rejects schema-20 artifacts. Canonical IR, fingerprint, specialization, persistence,
  and cold-geometry inclusion/exclusion rules match this specification.<span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [artifacts glossary](../../../../glossary.md#term-artifact); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [fingerprint glossary](../../../../glossary.md#term-fingerprint); [specialization glossary](../../../../glossary.md#term-specialization); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- `SUM`, [`MEAN`](../../../../glossary.md#term-mean), `PROD`, target-Shape SUM, later reduction/normalization families, multi-node
  partitions, unsupported types/attrs, and dynamic/unresolved forms remain fail-closed.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [normalization glossary](../../../../glossary.md#term-normalization); [node glossary](../../../../glossary.md#node); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- Existing pointwise, movement, indexing, scatter, fold, ordering, random, scan, cache, persistence,
  worker, and resource behavior remains unchanged.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [scan glossary](../../../../glossary.md#term-cumulative-scan); [cache glossary](../../../../glossary.md#term-cache))</span>
- A separate clean documentation-focused context finalizes affected Javadocs/package summaries,
  CPU guide, glossary impact, task/master/roadmap evidence, and explicit API/architecture/
  conformance no-change conclusions before this task becomes Complete.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

During implementation, [run](../../../../glossary.md#term-run) focused tests covering capability negatives/positives, aggregate [IR](../../../../glossary.md#term-intermediate-representation-ir),
all three attribute forms, [Shape](../../../../glossary.md#shape)/domain geometry, every type/kind, [identities](../../../../glossary.md#term-identity-meanings), empty selected and
unselected [axes](../../../../glossary.md#term-axis), empty multi-axis selection, [scalar](../../../../glossary.md#term-scalar) input, [NaN payload](../../../../glossary.md#nan-infinity-and-signed-zero) selection, [signed zero](../../../../glossary.md#term-signed-zero),
[infinities](../../../../glossary.md#nan-infinity-and-signed-zero), signed integral [extrema](../../../../glossary.md#saturation-pole-and-extremum), [canonical](../../../../glossary.md#term-canonical-form) [BOOL](../../../../glossary.md#data-type--datatype), arbitrary [layouts](../../../../glossary.md#layout)/carriers, bounded
[injectivity](../../../../glossary.md#storage-mapping-contiguity-and-injectivity), preparation/finalization, complete pre-mutation overlap, scalar/parallel parity,
[schema](../../../../glossary.md#term-schema)/[cache](../../../../glossary.md#term-cache) [identity](../../../../glossary.md#term-identity-meanings), and an independent reference comparison.

After executable Java stabilizes, run one final [module](../../../../glossary.md#term-module) suite:

```bash
./gradlew :backends:cpu:test
```

The separate clean documentation pass reuses that successful evidence unless it changes
executable Java and runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

It also validates affected Markdown links/anchors, balanced fences, one terminal newline,
trailing whitespace, rendered Javadocs, exact changed-path membership/ceiling, package/type
placement, schema 21, type/attribute/empty-domain [matrix](../../../../glossary.md#term-matrix-meanings), task/master/roadmap status/[dependency](../../../../glossary.md#term-dependency)
coherence, and absence of detailed [CPU](../../../../glossary.md#term-cpu) 0007A1/0007A2 task files.

Repository-wide tests, architecture tests, [backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance), and integration tests are deferred
to CPU 0009 or [continuous integration](../../../../glossary.md#term-continuous-integration-ci). Run them here only if implementation unexpectedly changes
a repository-wide, dependency, architecture, or reusable cross-backend contract; that is outside
[scope](../../../../glossary.md#term-scope) and normally requires stopping first.

## Dependencies

- CPU 0005A–0007: Complete, including typed carriers, resolved-layout access plans, bounded
  injectivity, direct Class-File bridge patterns, scalar/parallel-scalar workers, exact resource
  declarations, complete overlap validation, independent reference patterns, and schema 20.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [layout glossary](../../../../glossary.md#layout); [injectivity glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [scalar glossary](../../../../glossary.md#term-scalar); [validation glossary](../../../../glossary.md#term-validation); [schema glossary](../../../../glossary.md#term-schema))</span>
- Model 0018U1 and 0018V: Complete current ordinary full/single-/multi-axis signatures, five-type
  numeric extrema, BOOL ALL/ANY, Shape/zero-domain identities, signed-zero/NaN semantics, and
  empty-axis multi-axis meaning.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [axis glossary](../../../../glossary.md#term-axis); [signatures glossary](../../../../glossary.md#term-signature); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [BOOL glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN glossary](../../../../glossary.md#term-nan))</span>
- Current Compiler capture/inference: Complete and unchanged; produces revalidated exact
  one-input/one-output descriptors and normalized axes before CPU prepare.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [descriptors glossary](../../../../glossary.md#term-descriptor); [prepare glossary](../../../../glossary.md#prepare))</span>
- Current shared Prepare/Runtime contracts: Complete and unchanged; carry CPU declarations and
  invoke only the finalized direct prepared executable.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [carry glossary](../../../../glossary.md#term-carry); [prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable))</span>

## Follow-up tasks

- Draft [CPU](../../../../glossary.md#term-cpu) 0007A1 owns ordinary full/single-/multi-[axis](../../../../glossary.md#term-axis) `SUM`, [`MEAN`](../../../../glossary.md#term-mean), and `PROD`, including the
  exact floating numerical target, integral modular behavior, product special values,
  deterministic strategy, and explicit scratch/accumulator design.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [accumulator glossary](../../../../glossary.md#accumulator-type))</span>
- Draft CPU 0007A2 depends on CPU 0007A1 and owns [binding](../../../../glossary.md#term-binding)-aware `SUM` with `SumToShapeAttrs`, exact
  right-aligned bound-Shape compatibility, leading/aligned reduction geometry, and truthful
  resources. Neither follow-up has a detailed specification yet.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape))</span>
- CPU 0007B–0007F retain arg-extrema, masked, advanced statistical/norm, stable softmax, and
  normalization ownership. CPU 0009 retains portable closure and conformance.<span data-glossary-navigation="true"> ([extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [norm glossary](../../../../glossary.md#term-norm); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [normalization glossary](../../../../glossary.md#term-normalization); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Architecture impact

Expected impact: None.

The task uses existing [backend ownership](../../../../glossary.md#backend-ownership) of [lowering](../../../../glossary.md#lowering), route selection, generated [artifacts](../../../../glossary.md#term-artifact),
resources, [binding](../../../../glossary.md#term-binding), and execution. It changes no [module](../../../../glossary.md#term-module) responsibility or [dependency direction](../../../../glossary.md#java-platform-and-build-tools).
If implementation requires architecture, another module, or a shared contract change, stop and
report the exact conflict instead of editing around it.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Implement Synaptik CPU task 0007A exactly from its Ready specification. Do not use GSD. Read
AGENTS.md, ARCHITECTURE.md, the current architecture plan and directly relevant architecture
documents, documentation rules/profiles, planning guide, roadmap, CPU master plan, task 0007A,
completed CPU tasks 0005A/0005B/0005C/0005D/0005E/0006/0006B2/0006C/0006D/0007, Model tasks
0018U1/0018V and current reduction source/tests, Compiler reduction inference, and every affected
CPU scan/fold/capability/lowering/IR/generator/prepare/executable/reference source and test in full
before editing.

Deliver exactly one fully static resolved-layout ordinary MIN/MAX/ALL/ANY occurrence through the
portable generated route for exact NoOperationAttrs, AxisReductionAttrs, and
MultiAxisReductionAttrs forms. Implement the specified five-type extrema and BOOL matrix, exact
Shape/empty-domain identities, logical row-major first-NaN/signed-zero behavior, arbitrary
supported layouts/carriers, injective outputs, complete pre-mutation overlap rejection,
output-cell-only scalar/parallel-scalar work, zero workspace/partial/combine state, an independent
reference oracle, truthful complete-partition capability, and schema 21. The generated entry is a
bridge to a CPU-owned static body; do not overclaim embedded bytecode. Preserve all exclusions and
the 42-path ceiling. Stop on any architecture, shared-contract, Model-semantic, numerical,
resource, or scope conflict.

Run focused tests and one final ./gradlew :backends:cpu:test after executable Java stabilizes. Do
not commit or push. Then hand the uncommitted diff and exact Java evidence to a distinct clean
documentation-focused context following docs/developer-guide/documentation-rules.md. That pass
must independently finalize affected Javadocs/package summaries, CPU guide, glossary impact,
task/master/roadmap, no-change conclusions, Javadoc, rendered pages, Markdown, exact-scope, and
whitespace validation without repeating the successful Java suite unless executable behavior
changes. Do not mark Complete until both passes and every acceptance criterion succeed. Return a
completion summary with completed changes, exact files, tests/validation, documentation and
Javadoc/glossary impact, unresolved issues/follow-up, own CODEX_THREAD_ID when available, and
Status: Complete or Status: Incomplete with Follow-up required: <specific follow-up>.
```

## Local decisions

- Split the old ordinary-reduction row at numerical/resource and binding-geometry boundaries.
  Extrema/boolean folds are first because they establish complete output/domain geometry with no
  floating accumulation algorithm or scratch dependency.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [Extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [boolean glossary](../../../../glossary.md#data-type--datatype); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Canonical logical row-major traversal makes behavior independent of physical layout and gives a
  deterministic CPU-private first-NaN payload policy without adding public semantics.<span data-glossary-navigation="true"> ([Canonical glossary](../../../../glossary.md#term-canonical-form); [row-major glossary](../../../../glossary.md#term-row-major); [layout glossary](../../../../glossary.md#layout); [CPU glossary](../../../../glossary.md#term-cpu); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Parallelize only across independent output cells. Keeping each complete domain within one worker
  eliminates partial/combine state, reassociation, and scratch from this task.<span data-glossary-navigation="true"> ([reassociation glossary](../../../../glossary.md#term-reassociation))</span>
- Treat empty selected domains by [identity](../../../../glossary.md#term-identity-meanings) and empty unselected/output domains as no work.
- Reject physical overlap rather than introduce in-place alias/traversal semantics absent from the
  current Model and CPU prepared boundary.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [Model glossary](../../../../glossary.md#term-model))</span>

## Known limitations

- Coverage is one fully static, resolved-[layout](../../../../glossary.md#layout) [occurrence](../../../../glossary.md#term-occurrence) with non-negative physical [strides](../../../../glossary.md#term-stride).
- Floating NaN payload choice is deterministic only for CPU schema 21 and is not a cross-backend
  promise. No floating arithmetic or accuracy/performance claim is made.<span data-glossary-navigation="true"> ([NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [schema glossary](../../../../glossary.md#term-schema); [backend glossary](../../../../glossary.md#backend))</span>
- One full reduction or any occurrence with only one output cell remains scalar, even for a large
  domain. This task has no within-domain parallelism.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- Numerical aggregates and [target](../../../../glossary.md#term-target)-[Shape](../../../../glossary.md#shape) SUM remain [fail-closed](../../../../glossary.md#term-fail-closed) pending [CPU](../../../../glossary.md#term-cpu) 0007A1 and 0007A2.

## [Validation](../../../../glossary.md#term-validation) evidence

- [Planning](../../../../glossary.md#term-planning) context `019ffe96-884f-7bb0-be25-56ccfeeaa7e3` split the former CPU 0007A row and
  produced this Ready implementation specification after reviewing the architecture/planning
  contracts, completed CPU foundations, current Model/Compiler reduction contracts, and CPU
  scan/fold/resource/determinism patterns.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Ready glossary](../../../../glossary.md#term-ready); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- Local planning links, balanced fences, terminal newlines, trailing whitespace, every required
  planning-guide section, exact three-path CPU planning inventory, Ready/Draft dependency/frontier
  consistency, one new detailed task, absence of 0007A1/0007A2 specifications, and<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
  `git diff --check` passed. The planning thread applied patches only to this task, the CPU master
  plan, and CPU-specific roadmap text; concurrent NN planning/source/test work remained outside
  its patch targets and was neither staged nor rewritten.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [targets glossary](../../../../glossary.md#term-target))</span>
- Implementation context `019ffea4-7930-70c2-9773-ef9c76fefc17` completed the executable slice
  in 32 CPU paths: 23 production/package paths and 9 test paths, all in this task's allowlist.
  Its focused aggregate run passed 10 tests; its broader preparation/cache/integration run passed
  50 tests. Its final <span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice); [run glossary](../../../../glossary.md#term-run); [cache glossary](../../../../glossary.md#term-cache))</span>`./gradlew :backends:cpu:test` run on OpenJDK 26.0.1+8-34 with Gradle 9.6.1
  passed 53 suites and 303 tests with zero failures and zero errors. No executable Java or test
  changed after that run.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle))</span>
- Documentation context `019ffeb8-d37c-7c31-9c89-26a0264258d8` independently recounted the
  preserved final JUnit XML as 53 suites, 303 tests, 0 failures, 0 errors, and 1 skip. The sole
  skip is<span data-glossary-navigation="true"> ([JUnit glossary](../../../../glossary.md#term-junit))</span>
  `CpuGeneratedKernelPersistenceEvidenceTest.recordsFixedForkedPersistenceEvidence()`, whose
  existing assumption reports that explicit persistence evidence is disabled. This context did
  not rerun a Java test suite.<span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence))</span>
- The documentation context reviewed the final implementation diff, all 23 affected production/
  package paths and 9 test paths, the current Model aggregate contracts/tests, generated pages,
  and the required architecture, lifecycle, performance, planning, API, backend-guide, Javadoc,
  developer-guide, example, and glossary contracts. It finalized detailed aggregate IR/lowering/
  emitter/executable Javadocs, affected package summaries, the CPU backend guide, this existing
  glossary term, and CPU planning/status records without changing executable statements or tests.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [backend glossary](../../../../glossary.md#backend); [Javadoc glossary](../../../../glossary.md#term-javadoc); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lowering glossary](../../../../glossary.md#lowering))</span>
- `./gradlew :backends:cpu:javadoc` passed after the final Java documentation edit. Earlier
  successful generations exposed comment-only rendered-page defects; the context corrected them
  and reran as permitted. The final build emitted only the two expected incubating Vector API
  warnings. Rendered inspection covered <span data-glossary-navigation="true"> ([Vector API glossary](../../../../glossary.md#term-vector-api))</span>`CpuAggregateIr` and both nested enums,
  `CpuAggregateLowering`, `Geometry`, [`Layout`](../../../../glossary.md#layout), `CpuAggregateEmitter`,
  `CpuCapabilityProvider`, `CpuPreparedExecutable`, `CpuGeneratorSchema`,
  `CpuPortableRoutePlan`, `CpuScalarReferenceKernel`, and affected package summaries.
- Final targeted validation confirmed local Markdown targets and generated heading anchors,
  balanced backtick/tilde fences, the worked INT32 calculation, glossary terminology, terminal
  newlines, trailing whitespace, package/type placement, exact type/attribute/empty-domain
  coverage, schema 21 plus explicit schema-20 rejection, direct-bridge wording, status/dependency
  coherence, absence of detailed 0007A1/0007A2 files, concurrent NN-scope preservation, the exact
  37-path CPU-attributable allowlist, the 42-path ceiling, and <span data-glossary-navigation="true"> ([INT32 glossary](../../../../glossary.md#data-type--datatype); [schema glossary](../../../../glossary.md#term-schema); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check`.
- Tensor API: no change. The public full/single-/multi-axis reduction meanings, Shapes, types,
  identities, and attributes were already documented; this task adds only a concrete CPU
  realization and no public Tensor contract.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [multi-axis reduction glossary](../../../../glossary.md#multi-axis-reduction); [Shapes glossary](../../../../glossary.md#shape))</span>
- Compile API: no change. Compiler capture/inference and reduction semantics are unchanged, and
  CPU lowering consumes the existing projected occurrence without adding a compiler promise.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- [Runtime](../../../../glossary.md#term-runtime) [API](../../../../glossary.md#term-application-programming-interface-api): no change. The existing prepared-executable, cold-[binding](../../../../glossary.md#term-binding), [`RunState`](../../../../glossary.md#run-state--runstate), ownership,
  and schedule contracts already cover this backend-private recipe; no shared Runtime API or hot-
  path policy changed.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Training API: no change. The task adds no gradient, optimizer, parameter, training-session, or
  backend-specific training contract.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [optimizer glossary](../../../../glossary.md#term-optimizer); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- Model/compiler/shared capabilities or master plans outside CPU: no change. Model semantics and
  Compiler production remain inputs to CPU revalidation; capability truth changes only in the
  concrete CPU provider and complete one-node CPU lowering.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node))</span>
- Architecture, architecture decision records, and architecture tests: no change. Backend-owned
  lowering, artifact realization, binding, and execution remain within the existing contract,
  with no dependency or module-boundary change.<span data-glossary-navigation="true"> ([Backend-owned lowering glossary](../../../../glossary.md#backend-owned-lowering); [artifact glossary](../../../../glossary.md#term-artifact); [module glossary](../../../../glossary.md#term-module))</span>
- Backend conformance and integration tests: no change. This bounded CPU unit/integration slice
  does not introduce shared conformance infrastructure; portable closure remains CPU 0009 work.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance))</span>
- Gradle, dependencies, Java version, build structure, Config, Trace, Engine, NN, training, and
  every other module: no change. The implementation uses existing Java 26 CPU facilities and
  adds no dependency or cross-module surface.<span data-glossary-navigation="true"> ([Trace glossary](../../../../glossary.md#trace); [Engine glossary](../../../../glossary.md#term-engine))</span>

## Implementation notes

- Added [CPU](../../../../glossary.md#term-cpu)-private `CpuAggregateIr`, `CpuAggregateLowering`, and `CpuAggregateEmitter` owners and
  integrated them through capability, portable-route specialization, schema/cache, prepare,
  finalization, direct prepared execution, and independent scalar reference evaluation.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [schema glossary](../../../../glossary.md#term-schema); [cache glossary](../../../../glossary.md#term-cache); [prepare glossary](../../../../glossary.md#prepare); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [scalar glossary](../../../../glossary.md#term-scalar); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- Full, normalized single-axis, and canonical-membership multi-axis forms use one checked output/
  domain geometry. Empty axis membership is a one-value point domain; selected zero extents write
  identities, while unselected zero extents produce no output work.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>
- The exact [matrix](../../../../glossary.md#term-matrix-meanings) is five represented numeric types for `MIN`/`MAX` and [canonical](../../../../glossary.md#term-canonical-form) [BOOL](../../../../glossary.md#data-type--datatype) for
  `ALL`/`ANY`. Logical input row-major traversal selects the first represented NaN, enforces
  signed-zero extrema, compares BFLOAT16 after widening while copying selected raw bits, and uses
  signed integral order.<span data-glossary-navigation="true"> ([row-major glossary](../../../../glossary.md#term-row-major); [NaN glossary](../../../../glossary.md#term-nan); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [widening glossary](../../../../glossary.md#term-widening); [raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity))</span>
- Scalar and parallel-scalar execution partition only independent complete output cells. The plan
  declares exactly input/output buffers and zero workspace/materialization/partial/combine state.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [buffers glossary](../../../../glossary.md#term-buffer); [materialization glossary](../../../../glossary.md#materialization))</span>
- Binding validates carriers, layouts, injective output, canonical BOOL input, and complete
  physical overlap before mutation or submission. Schema 21 has no migration reader and treats
  schema 20 as incompatible.<span data-glossary-navigation="true"> ([Binding glossary](../../../../glossary.md#term-binding); [layouts glossary](../../../../glossary.md#layout); [injective glossary](../../../../glossary.md#term-injective))</span>
- The generated class contains a direct bridge to the CPU-owned static body; documentation does
  not describe the reduction loop as embedded bytecode.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode))</span>

## Completion summary

- Completed changes: Implemented exact one-[node](../../../../glossary.md#node) portable ordinary `MIN`, `MAX`, `ALL`, and `ANY`
  execution across the specified forms, types, layouts, carriers, empty domains, deterministic
  selection rules, output-cell ranges, resources, reference oracle, capability, and schema 21.<span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout); [oracle glossary](../../../../glossary.md#term-test-oracle); [schema glossary](../../../../glossary.md#term-schema))</span>
- Files changed or created: 37 CPU-attributable paths: 23 production/package paths, 9 CPU test
  paths, and 5 documentation/planning paths. All are task-allowlisted and below the 42-path limit.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: Reused implementation-context focused 10-test, broader 50-test, and final
  53-suite/303-test CPU evidence; independently recounted 0 failures, 0 errors, and the 1 expected
  opt-in persistence skip. Final CPU Javadoc, rendered-page, Markdown, scope, schema, status,
  package, concurrency-preservation, and whitespace validation passed.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [persistence glossary](../../../../glossary.md#term-persistence); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- Documentation-agent review: Clean context `019ffeb8-d37c-7c31-9c89-26a0264258d8` completed the
  mandatory independent documentation finalization without executable or test changes.
- Documentation impact: Updated affected Javadocs/package summaries, the CPU backend guide, this
  task, CPU master plan, and CPU-specific roadmap evidence. Public API and architecture documents
  remain unchanged for the reasons recorded above.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Javadoc review: Finalized aggregate IR/enums, lowering/Geometry/Layout, emitter, capability,
  executable, schema, route, reference, and package contracts; final generation and rendered-page
  inspection passed.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lowering glossary](../../../../glossary.md#lowering))</span>
- Glossary impact: Updated the existing CPU portable-route entry for current schema 21 and the
  implemented ordinary aggregate boundary; no unnecessary new term was introduced.
- Unresolved issues: None.
- Follow-up required: None for CPU 0007A. CPU 0007A1 is the next Draft planning frontier; CPU
  0007A2 remains Draft and dependent on 0007A1. Neither has a detailed specification.

Status: Complete
