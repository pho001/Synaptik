# Task 0003B: [Compile](../../../../glossary.md#compile)-time [Constants](../../../../glossary.md#term-constant) and [Constant Folding](../../../../glossary.md#term-constant-folding)

> **Editorial reading key — not part of the historical record.**
>
> A logical splat is one exact typed value repeated at every logical coordinate, supplied through
> explicit compiler ingress. It is not a snapshot inferred from mutable Tensor storage. A
> provenance-free leaf still means only no producing expression; absent explicit constant ingress
> it remains bindable. The constant sidecar fixes source meaning while Prepare and the backend
> separately arrange physical materialization. Scalar operation attributes and graph-value
> constant facts have different roles.
>
> Current reading: [structural capture and exact leaf identities](../../../../reference/compiler/capture.md#current-package-private-structural-capture); [exact graph transformations and final logical layouts](../../../../reference/compiler/optimization.md#current-package-private-canonicalization-and-whole-graph-optimization).

## Status

Complete

## Goal

Add one [compiler](../../../../glossary.md#term-compiler)-owned [immutable](../../../../glossary.md#term-immutable) representation for explicit [compile](../../../../glossary.md#compile)-time [constant](../../../../glossary.md#term-constant) facts and one
deliberately small constant-folding [scan](../../../../glossary.md#term-cumulative-scan) over a successful [canonical](../../../../glossary.md#term-canonical-form) `ValidatedGraph`.

The current representation is a **[logical splat](../../../../glossary.md#logical-splat)**: one exact typed [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) applies to every
logical [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) of one [graph value](../../../../glossary.md#graph-value). It is compile-time semantic data, not public [mutable](../../../../glossary.md#term-mutable)
[`Tensor`](../../../../glossary.md#tensor) state, [host storage](../../../../glossary.md#host-storage), a physical [buffer](../../../../glossary.md#term-buffer), or a [backend](../../../../glossary.md#backend) value. The compiler receives splats
only through an explicit package-private ingress. It never discovers them by reading
`Tensor.hostStorage()`, and a [provenance](../../../../glossary.md#provenance)-free factory [leaf](../../../../glossary.md#term-leaf) is an ordinary bindable [graph](../../../../glossary.md#graph) input
unless the ingress explicitly names that exact Tensor reference.

The selected fold [matrix](../../../../glossary.md#term-matrix-meanings) has exactly these four rows:

| Family | Selected kinds | Selected domains |
|---|---|---|
| [Boolean](../../../../glossary.md#data-type--datatype) logical | `NOT`, `AND`, `OR` | exact [`BOOL`](../../../../glossary.md#data-type--datatype) splats |
| [Binary](../../../../glossary.md#term-binary-descriptor-and-metadata) arithmetic | `ADD`, `SUB`, `MUL`, `MIN`, `MAX` | exact [`INT32`](../../../../glossary.md#data-type--datatype)/[`INT64`](../../../../glossary.md#data-type--datatype) splats, including current mixed-width signed promotion |
| Binary comparison | `GREATER_THAN`, `GREATER_OR_EQUAL`, `LESS_THAN`, `LESS_OR_EQUAL`, `EQUAL`, `NOT_EQUAL` | exact `INT32`/`INT64` splats, including current mixed-width signed promotion |
| [Scalar](../../../../glossary.md#term-scalar) [elementwise](../../../../glossary.md#term-elementwise) | `ADD`, `SUB`, `MUL`, `MIN`, `MAX` | one exact `INT32`/`INT64` splat plus exact same-typed immutable `ScalarValueAttrs` |

Every fold replaces one internal, one-output, non-[gradient](../../../../glossary.md#term-gradient) `FORWARD` [occurrence](../../../../glossary.md#term-occurrence) with one synthetic
constant source. A synthetic constant source is structurally a graph input, as required by the
unchanged [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel) source rules, but its sidecar fact excludes it from the derived
bindable-input list. No caller or later [run](../../../../glossary.md#term-run) may [rebind](../../../../glossary.md#capture-graph-ports-and-reconstruction) it. Graph-output [producers](../../../../glossary.md#term-producer), multi-output
producers, and non-forward work remain occurrences.

The bounded pipeline becomes:

```text
successful captured-graph validation with explicit constant sidecar
  -> mandatory canonicalization plus sidecar input-position remapping
  -> Compiler 0002 validation
  -> optional Compiler 0003A exact arithmetic rewriting
  -> Compiler 0002 validation when changed
  -> optional compile-time constant folding
  -> Compiler 0002 validation when changed
  -> optional sidecar-aware DCE -> CSE -> sidecar-aware DCE
  -> Compiler 0002 validation after each changed candidate
  -> successful canonical ValidatedGraph with live constant sources only
```

[Dead-code elimination](../../../../glossary.md#term-dead-code-elimination-dce) ([DCE](../../../../glossary.md#compiler-transformations-and-dominance)) and [common-subexpression elimination](../../../../glossary.md#term-common-subexpression-elimination-cse) ([CSE](../../../../glossary.md#compiler-transformations-and-dominance)) keep the one-shot order
established by Compiler 0003. The existing graph-only DCE entry keeps its exact contract and
retains every graph input. A new sidecar-aware overload additionally prunes a constant source when
it is neither a graph output nor an input of a retained [node](../../../../glossary.md#node), while retaining every [bindable input](../../../../glossary.md#bindable-input)
even when unused. This prevents folded or ingress constants from becoming permanently live
[materialization](../../../../glossary.md#materialization) obligations.

Compiler 0003A scalar-attribute [identities](../../../../glossary.md#term-identity-meanings) remain a separate earlier proof. Its exact
`ScalarValueAttrs` values are [operation](../../../../glossary.md#operation) [parameters](../../../../glossary.md#term-parameter), not graph-value constants. Compiler 0003A
reads an [identity](../../../../glossary.md#term-identity-meanings) attribute alone to bypass a selected operation over an arbitrary graph input.
This task never places an attribute in the constant sidecar; it evaluates only the selected
integral scalar kinds when the graph-valued input is also an explicit constant splat.

## [Mental model](../../../../glossary.md#term-mental-model) and examples

Read the sidecar and [graph](../../../../glossary.md#graph) together:

```text
CompiledGraphModel.inputs()         every structural source
  - IDs absent from splat facts     bindable external inputs
  - IDs present in splat facts      fixed compile-time constant sources

CompileTimeConstantGraph
  = unchanged structural graph
  + immutable ValueId -> exact typed splat facts
  + derived bindable inputs
```

For a conceptual integral example, suppose explicit ingress marks two [provenance](../../../../glossary.md#provenance)-free [leaves](../../../../glossary.md#term-leaf) as
[`INT32`](../../../../glossary.md#data-type--datatype) splats `2` and `3`, and the captured graph contains their [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) `ADD`:

```text
input v0 = splat(INT32, 2)   fixed, not bindable
input v1 = splat(INT32, 3)   fixed, not bindable
node  n0 = ADD(v0, v1) -> v2
node  n1 = MUL(v2, runtimeInput) -> output
```

The fold computes exact modular `INT32` value `5`, removes `n0`, and creates a synthetic fixed
source for the result:

```text
input v0 = splat(INT32, 2)   later pruned when unused
input v1 = splat(INT32, 3)   later pruned when unused
input v2 = runtimeInput      bindable
input v3 = splat(INT32, 5)   synthetic, not bindable
node  n0 = MUL(v3, v2) -> output
```

The first sidecar-aware [DCE](../../../../glossary.md#term-dead-code-elimination-dce) prunes the now-unused `2` and `3` sources and retains the [runtime](../../../../glossary.md#term-runtime) input
and the live `5` source. IDs are rebuilt densely and deterministically. No [host](../../../../glossary.md#term-host) memory is read and
no [buffer](../../../../glossary.md#term-buffer) holding repeated fives is allocated.

For a dynamic [Shape](../../../../glossary.md#shape) such as `[batch, 128]`, `splat(BOOL, true)` still means true at every logical
[coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) after [`batch`](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) is bound. The [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) remains one [scalar](../../../../glossary.md#term-scalar); [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) does not enumerate
[elements](../../../../glossary.md#term-element) or bind `batch`.

For a mixed-width example, `INT32(-1) < INT64(0)` promotes the signed `INT32` value to [`INT64`](../../../../glossary.md#data-type--datatype),
compares it in signed order, and produces the [`BOOL`](../../../../glossary.md#data-type--datatype) splat `true`. `INT32(MAX_VALUE) + INT32(1)`
produces exact modular `INT32(MIN_VALUE)`. These are [model](../../../../glossary.md#term-model) semantics, not Java [widening](../../../../glossary.md#term-widening) accidents.

## [Scope](../../../../glossary.md#term-scope)

- Add one package-private [immutable](../../../../glossary.md#term-immutable) `CompileTimeConstantGraph` aggregate in
  `io.github.pho001.synaptik.compiler`.
- Nest exactly three package-private immutable request/value records in that type:
  `Splat`, [`Binding`](../../../../glossary.md#term-binding), and `Ingress`.
- Keep the existing public [model](../../../../glossary.md#term-model) [graph](../../../../glossary.md#graph) types and record components unchanged.
- Keep the existing `GraphCapture.capture(List<Tensor>) -> CompiledGraphModel` entry and behavior
  unchanged for callers with no constant ingress.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant))</span>
- Add one package-private GraphCapture overload accepting explicit `Ingress` and returning
  `CompileTimeConstantGraph`.
- Match ingress bindings by exact Tensor object identity. Do not use labels, storage identity,
  structural descriptor equality, or a global registry.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [descriptor glossary](../../../../glossary.md#term-descriptor); [registry glossary](../../../../glossary.md#term-registry))</span>
- Require every ingress binding to name one reachable provenance-free leaf. Reject produced
  Tensors and unrelated leaves rather than silently ignoring them.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [leaf glossary](../../../../glossary.md#term-leaf))</span>
- Treat an unbound provenance-free Tensor as an ordinary bindable graph input even when it was
  created by `TensorFactory.scalar`, `zeros`, `ones`, `full`, an import, allocation, or random
  factory path.
- [Carry](../../../../glossary.md#term-carry) the sidecar through `ValidatedGraph`, canonicalization, Compiler 0003A rewriting, DCE,
  CSE, and Compiler 0002 revalidation without adding a public compile facade.<span data-glossary-navigation="true"> ([canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [Compiler glossary](../../../../glossary.md#term-compiler); [DCE glossary](../../../../glossary.md#term-dead-code-elimination-dce); [CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [compile glossary](../../../../glossary.md#compile); [facade glossary](../../../../glossary.md#term-facade))</span>
- Add one package-private stateless `ForwardConstantFolding` helper with this [shape](../../../../glossary.md#shape):

  ```java
  static CompileTimeConstantGraph fold(CompileTimeConstantGraph constantGraph)
  ```

- Visit nodes once in stored topological order, propagating only the selected splat facts so one
  earlier selected fold can make a later selected occurrence foldable in the same scan.<span data-glossary-navigation="true"> ([nodes glossary](../../../../glossary.md#node); [topological order glossary](../../../../glossary.md#term-topological-order); [scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- Fold exactly the four-row [matrix](../../../../glossary.md#term-matrix-meanings) in [Selected fold matrix](#selected-fold-matrix).
- Return the exact sidecar [argument](../../../../glossary.md#term-argument) when no [occurrence](../../../../glossary.md#term-occurrence) folds.
- When folding changes the graph, rebuild one immutable canonical candidate with original graph
  inputs first in boundary order, synthetic folded constants next in original node/output order,
  then retained node outputs in topological/output-slot order.<span data-glossary-navigation="true"> ([topological glossary](../../../../glossary.md#intermediate-representation-dag-and-ssa))</span><span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Keep graph-output producers as occurrences and preserve graph-output order and identity
  separation.<span data-glossary-navigation="true"> ([producers glossary](../../../../glossary.md#term-producer))</span>
- Keep every multi-output occurrence indivisible and unfolded.
- Extend `ForwardDeadCodeElimination` with one sidecar-aware overload. Do not change its existing
  graph-only signature or behavior.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>
- Prune only unused constant-source inputs/facts after graph-only forward DCE. Preserve all
  bindable inputs, every graph output, every constant consumed by a retained node, every
  non-forward node, and their dependency closures.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
- Integrate constant folding after Compiler 0003A and before Compiler 0003's unchanged one-shot
  DCE/CSE/DCE order.<span data-glossary-navigation="true"> ([constant folding glossary](../../../../glossary.md#term-constant-folding))</span>
- [Run](../../../../glossary.md#term-run) folding and sidecar-aware optional passes only when
  `GraphOptimizationConfig.optionalOptimizationsEnabled()` is true. Disabled optimization still
  performs mandatory canonicalization, sidecar remapping, and Compiler 0002 validation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Revalidate every changed immutable sidecar graph through Compiler 0002 before another
  transformation consumes it. Do not revalidate an exact unchanged sidecar result.
- Add complete Javadocs to every affected package-private production contract.
- Add focused tests for representation, ingress, fold semantics, pruning, pipeline order, IDs,
  failure order, immutability, and exclusions.
- Update Compile API, targeted Tensor API status text, glossary, this task, compiler master plan,
  and roadmap after implementation.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Complete the required separate documentation-focused pass in the same overall change.

### [Compiler](../../../../glossary.md#term-compiler)-owned sidecar [shape](../../../../glossary.md#shape)

Implement this package-private outer shape:

```java
record CompileTimeConstantGraph(
        CompiledGraphModel graph,
        Map<ValueId, Splat> constants) {

    record Splat(ScalarValue value) {}

    record Binding(Tensor tensor, Splat splat) {}

    record Ingress(List<Binding> bindings) {
        static Ingress empty() { ... }
    }

    static CompileTimeConstantGraph withoutConstants(CompiledGraphModel graph) { ... }

    List<ValueId> bindableInputs() { ... }

    CompileTimeConstantGraph replaceGraphPreservingInputRoles(
            CompiledGraphModel replacement) { ... }
}
```

The outer record and all nested records are package-private. Do not add public or protected
members, a builder, [registry](../../../../glossary.md#term-registry), service, listener, serializer, generic [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) hierarchy, or
physical [materialization](../../../../glossary.md#materialization) method.

`Splat` owns exactly one non-null [immutable](../../../../glossary.md#term-immutable) [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) reference. Record equality and hashing
therefore use `ScalarValue`'s exact [data type](../../../../glossary.md#data-type--datatype) and bit equality: floating signed zeros and distinct
[NaN](../../../../glossary.md#term-nan) payloads remain different, raw [BFLOAT16](../../../../glossary.md#term-bfloat16) patterns remain exact, and integral/[BOOL](../../../../glossary.md#data-type--datatype) values remain
[canonical](../../../../glossary.md#term-canonical-form) in their existing representation. Retaining the immutable reference is the complete
[snapshot](../../../../glossary.md#term-snapshot); no [host storage](../../../../glossary.md#host-storage), array, segment, [Tensor](../../../../glossary.md#tensor), [descriptor](../../../../glossary.md#term-descriptor), or Shape is retained in a splat.

[`Binding`](../../../../glossary.md#term-binding) owns one exact non-null Tensor reference and one non-null splat. It rejects a Tensor with
present [provenance](../../../../glossary.md#provenance) because ingress is limited to leaves. It rejects a Tensor whose descriptor has
`requiresGrad == true`; deleting or fixing a [gradient](../../../../glossary.md#term-gradient)-eligible [leaf](../../../../glossary.md#term-leaf) before Compiler 0004 would
pre-decide [parameter](../../../../glossary.md#term-parameter) and gradient semantics.

`Ingress` snapshots a non-null ordered binding list with `List.copyOf`, validates null [elements](../../../../glossary.md#term-element) in
encounter order, and rejects the first later binding that repeats an exact Tensor reference. Its
list order controls deterministic diagnostics only; [graph](../../../../glossary.md#graph) input order still follows GraphCapture
encounter order. `empty()` returns an ingress with an empty immutable list.

The outer record snapshots [`constants`](../../../../glossary.md#term-constant) with `Map.copyOf` after validating:

1. `graph`, then `constants`, are non-null;
2. no key or value is null;
3. every constant key is in `graph.inputs()`;
4. the splat data type equals that graph input's descriptor data type; and
5. the descriptor has `requiresGrad == false`.

Validate known graph inputs in graph-input order so descriptor/type/gradient failures are
deterministic. Reject an extra fact key that is not a graph input in ascending numeric [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot)
order. Do not require a static Shape, resolved [layout](../../../../glossary.md#layout), canonical density, positive element count,
or physical storage.

`bindableInputs()` returns a new immutable list in exact `graph.inputs()` order containing only
IDs absent from `constants`. It is the sole current classification of caller-bindable inputs. A
constant source must never also appear in this result.

`replaceGraphPreservingInputRoles` supports graph-only transformations that preserve the ordered
source boundary. It requires the replacement to have the same input count and equal descriptor at
every input position, remaps each old constant fact to the replacement input at the same position,
and returns the exact receiver if `replacement == graph`. It rejects a count or descriptor
contradiction instead of guessing a correspondence. [Constant folding](../../../../glossary.md#term-constant-folding) and constant-aware pruning,
which add or remove constant sources, construct a new sidecar directly.

### Explicit ingress and [capture](../../../../glossary.md#term-capture-meanings)

Add exactly this overload while preserving the existing entry:

```java
static CompileTimeConstantGraph capture(
        List<Tensor> outputs,
        CompileTimeConstantGraph.Ingress ingress)
```

The existing one-[argument](../../../../glossary.md#term-argument) method delegates through `Ingress.empty()` and returns only `.graph()`;
its observable [validation](../../../../glossary.md#term-validation), capture order, IDs, [graph](../../../../glossary.md#graph) structure, and exact return type remain
unchanged.

The overload applies the existing output-container, empty, [element](../../../../glossary.md#term-element), and duplicate checks first,
then rejects null `ingress` with `NullPointerException("ingress")`. The ingress records own their
own deterministic construction validation. Capture traverses exactly as today. When a reachable
[provenance](../../../../glossary.md#provenance)-free [leaf](../../../../glossary.md#term-leaf) is first allocated, capture looks up that exact [Tensor](../../../../glossary.md#tensor) reference in the
ingress and, when present, associates the resulting graph input ID with the supplied splat.

After traversal and requested-output resolution, reject the first ingress [binding](../../../../glossary.md#term-binding) whose exact
Tensor was not encountered as a reachable leaf. Use its ingress-list index in a deterministic
message. This prevents a typo or unrelated binding from being accepted as if it affected
[compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names). A requested provenance-free Tensor may be both a [constant](../../../../glossary.md#term-constant) source and a graph output;
it remains in both graph boundary lists and is excluded from `bindableInputs()`.

Capture reads only Tensor [identity](../../../../glossary.md#term-identity-meanings), [descriptor](../../../../glossary.md#term-descriptor), and provenance as it already does. It must not call
`hostStorage()`, inspect `HostTensorStorage`, read a [`MemorySegment`](../../../../glossary.md#term-memorysegment), recognize a factory method,
or infer a splat from a label, [layout](../../../../glossary.md#layout), storage contents, zero-sized [shape](../../../../glossary.md#shape), or other [metadata](../../../../glossary.md#term-metadata).

### Selected fold [matrix](../../../../glossary.md#term-matrix-meanings)

For a [node](../../../../glossary.md#node) encountered in [topological order](../../../../glossary.md#term-topological-order), create and propagate a result splat only when all of
these common conditions hold:

1. the node phase is exactly `GraphPhase.FORWARD`;
2. no node output is in `graph.outputs()`;
3. the [operation](../../../../glossary.md#operation) kind and exact attributes match one selected row;
4. the [signature](../../../../glossary.md#term-signature) supplies the selected fixed input count and exactly one output;
5. every input position has a known splat fact;
6. every input fact's type agrees with its already validated input [descriptor](../../../../glossary.md#term-descriptor);
7. the sole output descriptor has `requiresGrad == false`; and
8. exact [evaluation](../../../../glossary.md#term-evaluation) produces a splat whose type equals the stored output descriptor type.

Selected operation rows are:

| Operation | Attributes | Inputs | Result |
|---|---|---|---|
| `BooleanLogicalKind.NOT` | exact `NoOperationAttrs.INSTANCE` | one [BOOL](../../../../glossary.md#data-type--datatype) splat | [canonical](../../../../glossary.md#term-canonical-form) BOOL negation |
| `BooleanLogicalKind.AND`, `OR` | exact `NoOperationAttrs.INSTANCE` | two BOOL splats | canonical BOOL conjunction/disjunction |
| `BinaryArithmeticKind.ADD`, `SUB`, `MUL` | exact `NoOperationAttrs.INSTANCE` | two integral splats | fixed-width modular result in the validated promoted type |
| `BinaryArithmeticKind.MIN`, `MAX` | exact `NoOperationAttrs.INSTANCE` | two integral splats | signed-order selected value in the validated promoted type |
| all six `BinaryComparisonKind` values | exact `NoOperationAttrs.INSTANCE` | two integral splats | canonical BOOL result after signed promotion and comparison |
| `ScalarElementwiseKind.ADD`, `SUB`, `MUL` | exact `ScalarValueAttrs` | one integral splat plus an exact same-typed [scalar](../../../../glossary.md#term-scalar) attribute | fixed-width modular result in the unchanged input/output type |
| `ScalarElementwiseKind.MIN`, `MAX` | exact `ScalarValueAttrs` | one integral splat plus an exact same-typed scalar attribute | signed-order selected value in the unchanged input/output type |

For two [`INT32`](../../../../glossary.md#data-type--datatype) operands, calculate in signed 32-bit primitives so ADD/SUB/MUL wrap modulo
`2^32`. If either operand is [`INT64`](../../../../glossary.md#data-type--datatype), sign-extend any `INT32` operand and calculate or compare in
signed 64-bit primitives; ADD/SUB/MUL wrap modulo `2^64`. MIN/MAX and all relational operators use
ordinary signed order. Equality compares the promoted exact signed values. Construct results only
through the existing named `ScalarValue.int32`, `int64`, and `bool` factories.

Scalar evaluation preserves operand order: the [graph](../../../../glossary.md#graph) input is the left operand and
`ScalarValueAttrs.value()` is the right operand. Require the attribute value type to equal the
input and output descriptor type exactly. Apply the same INT32/INT64 modular and signed-order
rules without promotion. Integral scalar DIV and POW are not current valid domains and remain
excluded.

Do not [dispatch](../../../../glossary.md#term-dispatch) through strings, reflection, `Number`, `double`, a generic evaluator interface, or
an operation [registry](../../../../glossary.md#term-registry). Use closed typed switches after exact kind/attributes checks. Invalid
operation/descriptor combinations cannot reach this pass because [Compiler](../../../../glossary.md#term-compiler) 0002 already accepted
the graph; the fold helper retains an unselected [occurrence](../../../../glossary.md#term-occurrence) rather than becoming another semantic
validator.

### Deliberately excluded [evaluation](../../../../glossary.md#term-evaluation) [matrix](../../../../glossary.md#term-matrix-meanings)

| Candidate | This task's decision |
|---|---|
| Floating ADD/SUB/MUL/DIV/POW/MIN/MAX | Not folded. [Rounding](../../../../glossary.md#term-rounding), intermediate [precision](../../../../glossary.md#term-precision), [NaN payload](../../../../glossary.md#nan-infinity-and-signed-zero)/sign, [signed zero](../../../../glossary.md#term-signed-zero), [infinities](../../../../glossary.md#nan-infinity-and-signed-zero), [underflow](../../../../glossary.md#term-underflow), [overflow](../../../../glossary.md#term-overflow), and [backend](../../../../glossary.md#backend) numerical permissions are not reduced to Java evaluation here. |
| [BFLOAT16](../../../../glossary.md#term-bfloat16) arithmetic | Not folded. Raw BFLOAT16 splats remain representable exactly, but no current [compiler](../../../../glossary.md#term-compiler) numerical contract selects an arithmetic rounding implementation. |
| Floating comparisons | Not folded in this bounded matrix, even where a result could be defined from value classes. |
| Casts | Not folded. Conversion, overflow, [NaN](../../../../glossary.md#term-nan), [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero), and BFLOAT16 rounding need a separate exact conversion matrix. |
| Other [scalar](../../../../glossary.md#term-scalar) [elementwise](../../../../glossary.md#term-elementwise) operations | Floating scalar operations, CLAMP, and every kind outside integral ADD/SUB/MUL/MIN/MAX are not folded. `ScalarValueAttrs` remains [operation](../../../../glossary.md#operation) [metadata](../../../../glossary.md#term-metadata) and is never a [graph](../../../../glossary.md#graph)-value fact. |
| [Unary](../../../../glossary.md#term-unary) numerical functions or [activation](../../../../glossary.md#term-activation) functions | Not folded. The [model](../../../../glossary.md#term-model) names mathematical [targets](../../../../glossary.md#term-target) without a compiler-selected [correctly rounded](../../../../glossary.md#nan-infinity-and-signed-zero) algorithm. |
| Reductions, scans, [softmax](../../../../glossary.md#softmax--log-softmax), [normalization](../../../../glossary.md#term-normalization), [loss](../../../../glossary.md#loss), [matrix multiplication](../../../../glossary.md#matrix-multiplication--matmul), [convolution](../../../../glossary.md#term-channels-first-convolution-layers), [pooling](../../../../glossary.md#term-pooling) | Not folded. They require [element](../../../../glossary.md#term-element) traversal, domain policies, or numerical algorithms and would violate the splat-only bounded-work contract. |
| [Shape](../../../../glossary.md#shape)/[layout](../../../../glossary.md#layout)/index/[view](../../../../glossary.md#term-view-meanings) operations | Not folded. This task does not reinterpret [aliases](../../../../glossary.md#aliasing-and-arena), indices, bounds, [materialization](../../../../glossary.md#materialization), or layout. |
| Random/state/[dropout](../../../../glossary.md#dropout--inverted-dropout) operations | Never [constant](../../../../glossary.md#term-constant); explicit state and randomness remain semantic sources. |
| `WHERE` and constant-control bypass | Deferred. Bypassing a branch can change future [gradient](../../../../glossary.md#term-gradient) [occurrence](../../../../glossary.md#term-occurrence) and operand-use semantics. |
| Multi-output operations | Never partially folded; the complete [producer](../../../../glossary.md#term-producer) remains. |
| Graph-output producer | Remains distinct even when every input is constant, preserving requested occurrence and later [publication](../../../../glossary.md#term-publication) [identity](../../../../glossary.md#term-identity-meanings). |

Floating, BFLOAT16, and [BOOL](../../../../glossary.md#data-type--datatype)/integral splats all have exact representation equality. The exclusion
is from evaluation, not from the [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) type. A [FLOAT32](../../../../glossary.md#term-float32) negative-zero splat remains different
from [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero); a [FLOAT64](../../../../glossary.md#term-float64) or BFLOAT16 NaN retains its raw payload. No selected fold consumes
those facts, canonicalizes them, or promises a result payload.

### Rebuild, source roles, and deterministic IDs

The fold [scan](../../../../glossary.md#term-cumulative-scan) first records selected fold decisions and propagated facts in original [topological
order](../../../../glossary.md#term-topological-order). It performs no [graph](../../../../glossary.md#graph) mutation. If no [node](../../../../glossary.md#node) folds, return the exact sidecar reference.

For a changed result, rebuild in two deterministic stages:

1. allocate every original graph input in exact boundary order and preserve its bindable or
   constant role;
2. allocate one synthetic constant input for each folded node in original topological and output-
   slot order; then allocate retained node outputs while visiting retained nodes in original
   topological order.<span data-glossary-navigation="true"> ([topological glossary](../../../../glossary.md#intermediate-representation-dag-and-ssa))</span>

All input IDs are therefore dense from zero before any retained node output ID. Retained node IDs
are dense in retained topological order. Remap every retained ordered input, graph output, phase,
[operation](../../../../glossary.md#operation), and [descriptor](../../../../glossary.md#term-descriptor). Retain exact operation, descriptor, phase, and splat references where
their contracts permit. A folded node and its original output are not retained; later uses map to
the corresponding synthetic [constant](../../../../glossary.md#term-constant) input.

Do not merge equal splats or reuse an earlier constant source merely because exact [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) and
descriptor values match. Constant-source interning would change graph-value [identity](../../../../glossary.md#term-identity-meanings) and source
[occurrence](../../../../glossary.md#term-occurrence) policy and is not needed for this bounded task. [CSE](../../../../glossary.md#term-common-subexpression-elimination-cse) remains based on graph input IDs,
not constant value equality.

### Sidecar-aware dead [constant](../../../../glossary.md#term-constant) pruning

Keep this existing entry unchanged:

```java
static CompiledGraphModel eliminate(CompiledGraphModel graph)
```

Add exactly this overload:

```java
static CompileTimeConstantGraph eliminate(CompileTimeConstantGraph constantGraph)
```

The overload first applies the existing [graph](../../../../glossary.md#graph)-only forward [DCE](../../../../glossary.md#term-dead-code-elimination-dce) and remaps source roles by input
position. It then considers a constant source live exactly when its ID is a graph output or occurs
in any retained [node](../../../../glossary.md#node) input position. Every [bindable input](../../../../glossary.md#bindable-input) remains live regardless of use. Rebuild
only when graph DCE or constant pruning changed the candidate, using bindable and live-constant
inputs in their existing relative graph-input order, followed by retained node outputs in stored
[topological](../../../../glossary.md#term-topological-order)/output order. Rebuild dense [canonical](../../../../glossary.md#term-canonical-form) IDs and remap facts, nodes, phases, and outputs.

This pruning is structural only. It does not inspect [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) values, merge constants, remove a
bindable input, remove a graph-output constant, or decide physical [materialization](../../../../glossary.md#materialization). An unchanged
sidecar result returns the exact [argument](../../../../glossary.md#term-argument).

### Pipeline and failure ordering

`ForwardGraphOptimization.optimize` retains its two existing null checks in order. Mandatory work
is:

1. canonicalize `validatedGraph.graph()`;
2. remap the validated sidecar by input position;
3. validate the [canonical](../../../../glossary.md#term-canonical-form) sidecar [graph](../../../../glossary.md#graph) through [Compiler](../../../../glossary.md#term-compiler) 0002; and
4. return that result immediately when optional optimization is disabled.

When enabled, continue exactly:

5. [run](../../../../glossary.md#term-run) Compiler 0003A rewriting once and remap its unchanged input roles;
6. validate only if changed;
7. run `ForwardConstantFolding.fold` once;
8. validate only if changed;
9. run sidecar-aware forward [DCE](../../../../glossary.md#term-dead-code-elimination-dce) once and validate only if changed;
10. run existing exact forward [CSE](../../../../glossary.md#term-common-subexpression-elimination-cse) once, remap input roles, and validate only if changed;
11. run sidecar-aware forward DCE once and validate only if changed; and
12. return the final exact `ValidatedGraph`.

No pass repeats, no fixed point is sought, and no second folding [scan](../../../../glossary.md#term-cumulative-scan) runs after CSE. Compiler
0002 continues to infer [operation](../../../../glossary.md#operation) [descriptors](../../../../glossary.md#term-descriptor) and deferred constraints from the graph only; the
sidecar constructor separately validates [constant](../../../../glossary.md#term-constant)-source agreement. Add exactly this package-
private Compiler 0002 overload while preserving its existing entry:

```java
static ValidatedGraph inferAndValidate(CompileTimeConstantGraph constantGraph)
```

The existing `inferAndValidate(CompiledGraphModel)` delegates through
`CompileTimeConstantGraph.withoutConstants(graph)`. Both preserve the existing null and semantic
failure behavior for their declared [argument](../../../../glossary.md#term-argument).

Change the package-private successful result to this internal [shape](../../../../glossary.md#shape):

```java
record ValidatedGraph(
        CompileTimeConstantGraph constantGraph,
        List<DeferredGraphConstraint> constraints) {

    ValidatedGraph(CompiledGraphModel graph, List<DeferredGraphConstraint> constraints) { ... }

    CompiledGraphModel graph() { ... }

    Map<ValueId, CompileTimeConstantGraph.Splat> constants() { ... }

    List<ValueId> bindableInputs() { ... }
}
```

The two-argument compatibility constructor preserves existing same-package construction by
wrapping the graph without facts. `graph()` returns the exact graph inside `constantGraph`;
`constants()` and `bindableInputs()` delegate to that [immutable](../../../../glossary.md#term-immutable) sidecar. The record snapshots
constraints exactly as today. Do not add another stored graph, fact map, or bindable-input list.

## Out of [scope](../../../../glossary.md#term-scope)

- reading, snapshotting, hashing, comparing, or trusting public [Tensor](../../../../glossary.md#tensor) [host](../../../../glossary.md#term-host)-storage contents
- inferring [constants](../../../../glossary.md#term-constant) from `TensorFactory` method history, provenance absence, labels, layouts,
  zero-sized Shapes, storage allocation, or Tensor identity alone<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [layouts glossary](../../../../glossary.md#layout); [Shapes glossary](../../../../glossary.md#shape); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- allowing a constant source to be rebound as a public/[run](../../../../glossary.md#term-run) input
- public [compile](../../../../glossary.md#compile) configuration, public constant [binding](../../../../glossary.md#term-binding), `GraphCompiler`, `CompiledGraph`, engine
  facade, registry, service, or general compile entry point<span data-glossary-navigation="true"> ([engine glossary](../../../../glossary.md#term-engine); [facade glossary](../../../../glossary.md#term-facade); [registry glossary](../../../../glossary.md#term-registry))</span>
- dense, sparse, array, byte-buffer, memory-segment, per-element, tuple, string, boxed, or generic
  constant payloads<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [element glossary](../../../../glossary.md#term-element))</span>
- materializing a splat, allocating host/device/physical buffers, choosing a storage layout,
  determining residency, or retaining mutable memory<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device); [residency glossary](../../../../glossary.md#residency); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- folding any [operation](../../../../glossary.md#operation) outside the exact four-row [matrix](../../../../glossary.md#term-matrix-meanings)
- floating or BFLOAT16 arithmetic/comparison, cast evaluation, numerical unary evaluation,
  reduction, scan, softmax, normalization, loss, linear algebra, convolution, pooling, index,
  layout/view, random, state, or dropout evaluation<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [unary glossary](../../../../glossary.md#term-unary); [scan glossary](../../../../glossary.md#term-cumulative-scan); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [normalization glossary](../../../../glossary.md#term-normalization); [loss glossary](../../../../glossary.md#loss); [linear glossary](../../../../glossary.md#term-linear-meanings); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [pooling glossary](../../../../glossary.md#term-pooling); [view glossary](../../../../glossary.md#term-view-meanings); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span>
- general partial [evaluation](../../../../glossary.md#term-evaluation), arbitrary interpretation of `Operation`, a generic evaluator,
  registry, pass framework, string dispatch, reflection, or plugin mechanism<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [plugin glossary](../../../../glossary.md#term-plugin))</span>
- constant interning, deduplication by value, serialization, persistence, hashing protocol, or
  artifact file format<span data-glossary-navigation="true"> ([serialization glossary](../../../../glossary.md#term-serialization); [persistence glossary](../../../../glossary.md#term-persistence); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- large or unbounded compile work, per-element loops, data-dependent loops, recursive evaluation,
  fixed-point iteration, or payload growth beyond one scalar per fact<span data-glossary-navigation="true"> ([fixed-point glossary](../../../../glossary.md#fixed-point-arithmetic); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- relaxed/fast-math, reassociation, commutation, reciprocal substitution, finite-only assumptions,
  approximate evaluation, tolerance-based equality, or backend numerical selection<span data-glossary-navigation="true"> ([reassociation glossary](../../../../glossary.md#term-reassociation); [commutation glossary](../../../../glossary.md#term-commutation); [finite glossary](../../../../glossary.md#term-finite); [tolerance glossary](../../../../glossary.md#term-tolerance); [backend glossary](../../../../glossary.md#backend))</span>
- autograd, backward construction, gradient rules, saved-value policy, gradient-eligible constant
  leaves, post-autograd folding, or training behavior<span data-glossary-navigation="true"> ([leaves glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- changing graph-output producers, publication bindings, public output identity, or multi-output
  producer structure<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [publication glossary](../../../../glossary.md#term-publication); [producer glossary](../../../../glossary.md#term-producer))</span>
- changing [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel), `GraphValue`, `CompiledNode`, `Operation`, [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue), Tensor,
  storage, model semantics, config, planning, prepare, runtime, backend, or engine contracts<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- final public `CompileArtifacts`, `ConstantPlan`, prepare materialization, or task 0005
  orchestration implementation<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization))</span>
- dependencies, Gradle/build configuration, Java version, architecture contract, focused
  architecture documents, ADRs, or architecture tests<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- later [compiler](../../../../glossary.md#term-compiler) task specifications

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially compiler ownership of
  constant folding, immutable compile-time state, the compile lifecycle, and the prohibition on
  physical buffers and runtime/backend state<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [constant folding glossary](../../../../glossary.md#term-constant-folding); [immutable glossary](../../../../glossary.md#term-immutable); [compile glossary](../../../../glossary.md#compile); [lifecycle glossary](../../../../glossary.md#lifecycle); [buffers glossary](../../../../glossary.md#term-buffer); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Compiler master plan](../master-plan.md)
- [Compiler 0001 capture](0001-tensor-expression-graph-capture.md)<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- [Compiler 0002 validation](0002-captured-graph-inference-and-validation.md)<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- [Compiler 0003 canonicalization and optimization](0003-canonicalization-and-forward-optimization.md)<span data-glossary-navigation="true"> ([canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form))</span>
- [Compiler 0003A exact arithmetic rewriting](0003a-exact-arithmetic-rewriting.md)
- [Model compiled graph task](../../model/tasks/0009-compiled-graph-model.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [compiled graph glossary](../../../../glossary.md#compiled-graph--compiledgraphmodel))</span>
- [Model constant creation task](../../model/tasks/0012d-constant-tensor-creation.md)<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant))</span>
- [Model provenance task](../../model/tasks/0018l-shared-multi-output-tensor-provenance.md)<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- [Model typed scalar task](../../model/tasks/0018n-typed-scalar-value-contract.md)<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- [Model capability closure audit](../../model/model-capability-contract-closure-audit.md)
- [Planning logical memory task](../../planning/tasks/0005-logical-materialization-and-memory-requirements.md)
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>, [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>, and
  [glossary](../../../../glossary.md)

## Architecture constraints

- Compiler owns constant recognition, propagation, folding, graph rewriting, and the immutable
  logical sidecar.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [graph glossary](../../../../glossary.md#graph); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- Model graph and scalar contracts remain unchanged. A constant source uses the existing valid
  graph-input structural form rather than a new model operation kind or <span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [scalar glossary](../../../../glossary.md#term-scalar); [operation glossary](../../../../glossary.md#operation))</span>`GraphValue` component.
- [`Tensor`](../../../../glossary.md#tensor) remains [mutable](../../../../glossary.md#term-mutable) public [API](../../../../glossary.md#term-application-programming-interface-api) state and never becomes authoritative [constant](../../../../glossary.md#term-constant) data.
- Compiler must not allocate physical buffers, read host storage, construct runtime values,
  execute a backend, select a numerical kernel, or create prepared state.<span data-glossary-navigation="true"> ([buffers glossary](../../../../glossary.md#term-buffer); [host storage glossary](../../../../glossary.md#host-storage); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [kernel glossary](../../../../glossary.md#kernel))</span>
- Every constant source is classified out of `bindableInputs()`; there is no hidden runtime
  rebinding or assumption that a later supplied input equals a compile-time value.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile))</span>
- Compiler 0005 must preserve and transport the exact immutable constant sidecar, or a named
  constant-plan component with exactly equivalent semantics, in the future immutable compile
  artifacts. It must give planning the unchanged graph for logical requirements, derive public
  bindable inputs from this task's classification, and make the logical splat facts available to
  prepare without turning them into physical buffers. Task 0005 must stop if its artifact design
  would discard facts, expose constant sources as bindable inputs, or require a model/architecture
  change.<span data-glossary-navigation="true"> ([compile artifacts glossary](../../../../glossary.md#compile-artifacts); [planning glossary](../../../../glossary.md#term-planning); [logical splat glossary](../../../../glossary.md#logical-splat); [prepare glossary](../../../../glossary.md#prepare); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Planning may treat a constant source as a logical graph value requiring materialization. It does
  not inspect the payload, allocate memory, or select implementation routes.<span data-glossary-navigation="true"> ([graph value glossary](../../../../glossary.md#graph-value); [materialization glossary](../../../../glossary.md#materialization); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>
- Future prepare/backend work owns physical splat materialization, storage allocation, lowering,
  and execution. Runtime binds only the nonconstant inputs selected by the prepared contract.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering))</span>
- The compiler package remains internal. No facade, registry, service locator, or generic
  evaluator is justified.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade); [registry glossary](../../../../glossary.md#term-registry); [service locator glossary](../../../../glossary.md#term-service-locator))</span>
- If implementation cannot carry source roles through every changed candidate, prune unused
  constants deterministically, or preserve the sidecar for Compiler 0005 without changing the
  architecture contract, stop and report the exact conflict.<span data-glossary-navigation="true"> ([carry glossary](../../../../glossary.md#term-carry); [architecture contract glossary](../../../../glossary.md#architecture-contract))</span>

## Package impact

Existing package used:

- `io.github.pho001.synaptik.compiler` — owns internal capture, validation, canonicalization,
  exact rewriting, constant facts/folding, and the bounded forward pipeline.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [validation glossary](../../../../glossary.md#term-validation); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [constant glossary](../../../../glossary.md#term-constant))</span>

Packages added or changed:

- No package is added. The compiler root gains two package-private types and targeted changes to
  five existing package-private types.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler))</span>

Type placement:

- `io.github.pho001.synaptik.compiler.CompileTimeConstantGraph` — owns the compiler-only graph
  sidecar, exact splat value, explicit ingress binding, bindable-input derivation, and safe
  input-position remapping.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [binding glossary](../../../../glossary.md#term-binding))</span>
- `io.github.pho001.synaptik.compiler.ForwardConstantFolding` — owns the selected one-scan typed
  evaluation and synthetic-source rebuild.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- `io.github.pho001.synaptik.compiler.GraphCapture` — owns the only Tensor-identity-to-graph-ID
  moment and therefore maps explicit ingress facts without retaining Tensor state.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `io.github.pho001.synaptik.compiler.ValidatedGraph` and `CapturedGraphInference` — retain and
  revalidate the compiler-only sidecar without changing model inference.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [inference glossary](../../../../glossary.md#term-inference))</span>
- `io.github.pho001.synaptik.compiler.ForwardDeadCodeElimination` — owns liveness and adds the
  sidecar-aware unused-constant-source pruning overload while keeping graph-only behavior intact.<span data-glossary-navigation="true"> ([liveness glossary](../../../../glossary.md#term-liveness))</span>
- `io.github.pho001.synaptik.compiler.ForwardGraphOptimization` — owns the exact bounded order and
  changed-candidate validation.

Test placement mirrors the production package so tests can exercise package-private contracts.

## Affected files

Expected production files:

- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileTimeConstantGraph.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ForwardConstantFolding.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/GraphCapture.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CapturedGraphInference.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ValidatedGraph.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ForwardDeadCodeElimination.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ForwardGraphOptimization.java`

Expected test files:

- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/CompileTimeConstantGraphTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/ForwardConstantFoldingTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/CapturedGraphInferenceTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GraphCaptureTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/ForwardDeadCodeEliminationTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/ForwardGraphOptimizationTest.java`

Expected documentation and [planning](../../../../glossary.md#term-planning) files:

- `docs/api/compile-api.md`
- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/compiler/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most the 19 paths listed above.

The larger path count is justified because one cohesive [compiler](../../../../glossary.md#term-compiler) [invariant](../../../../glossary.md#term-invariant) must survive [capture](../../../../glossary.md#term-capture-meanings),
[validation](../../../../glossary.md#term-validation), every ID-rebuilding transformation, [constant](../../../../glossary.md#term-constant)-aware [liveness](../../../../glossary.md#term-liveness), tests, Javadocs, and
current-status documentation atomically. Splitting the sidecar from folding would either create an
unused representation or permit a [graph](../../../../glossary.md#graph) transform to discard source roles. The authorized maximum
expanded from 18 to 19 only because adding the package-private
`CapturedGraphInference.inferAndValidate(CompileTimeConstantGraph)` overload made the existing
untyped null invocation ambiguous; `CapturedGraphInferenceTest` now casts that null explicitly to
[`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel) and preserves the original graph-only failure assertion.

Do not modify [model](../../../../glossary.md#term-model)/config/[planning](../../../../glossary.md#term-planning)/[prepare](../../../../glossary.md#prepare)/[runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend)/[engine](../../../../glossary.md#term-engine) Java, [Gradle](../../../../glossary.md#term-gradle), architecture,
[conformance](../../../../glossary.md#term-conformance), integration, or additional documentation paths. If another production type, test
class, public type, [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) form, path, or [module](../../../../glossary.md#term-module) is required, stop and propose a follow-up rather
than exceeding this maximum.

## Acceptance criteria

- `CompileTimeConstantGraph` and its nested `Splat`, [`Binding`](../../../../glossary.md#term-binding), and `Ingress` records have exactly
  the planned package-private shape, immutability, validation, and Javadocs.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [validation glossary](../../../../glossary.md#term-validation))</span>
- Splat equality preserves exact type/bits for every current data type, including raw BFLOAT16,
  floating signed zero and NaN payloads, exact integrals, and canonical BOOL.<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [signed zero glossary](../../../../glossary.md#term-signed-zero); [NaN glossary](../../../../glossary.md#term-nan); [canonical glossary](../../../../glossary.md#term-canonical-form); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- Payloads retain no Tensor, Shape, descriptor, storage, array, segment, physical buffer, backend,
  or runtime value.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [descriptor glossary](../../../../glossary.md#term-descriptor); [buffer glossary](../../../../glossary.md#term-buffer); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Explicit ingress accepts only ordered unique exact provenance-free non-gradient Tensor leaves;
  unrelated and produced Tensors fail deterministically.<span data-glossary-navigation="true"> ([leaves glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- GraphCapture's existing method is behaviorally unchanged. The overload maps only explicit facts
  and never reads host storage or infers factory constants.<span data-glossary-navigation="true"> ([host storage glossary](../../../../glossary.md#host-storage))</span>
- A provenance-free leaf absent from ingress remains bindable, including factory scalar/zero/one/
  full/import/random leaves.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- `bindableInputs()` is exactly the [graph](../../../../glossary.md#graph) input list minus [constant](../../../../glossary.md#term-constant) fact keys, in graph input order.
- Canonicalization, 0003A, CSE, and validation preserve source roles by exact input position and
  reject a source-boundary contradiction.<span data-glossary-navigation="true"> ([Canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse))</span>
- The fold helper implements every selected BOOL, binary-integral, comparison, and same-typed
  scalar-integral row, all binary same/mixed integral width combinations, signed comparison, and
  modular overflow exactly.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [overflow glossary](../../../../glossary.md#term-overflow))</span>
- The fold helper implements no floating/BFLOAT16 evaluation, cast, scalar operation outside the
  selected same-typed integral row, unary numerical function, reduction, layout/index operation,
  random/state operation, or generic evaluator behavior.<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation); [operation glossary](../../../../glossary.md#operation); [unary glossary](../../../../glossary.md#term-unary); [layout glossary](../../../../glossary.md#layout))</span>
- One scan propagates earlier selected folded splats to later selected occurrences without a fixed
  point or second scan.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- Changed folding creates deterministic dense input-first IDs, places synthetic constants after
  original inputs in fold occurrence order, and preserves retained node/output order and exact
  descriptors/operations/phases.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [node glossary](../../../../glossary.md#node))</span>
- Graph-output producers, BACKWARD nodes, gradient-eligible values, and all multi-output producers
  remain occurrences.<span data-glossary-navigation="true"> ([producers glossary](../../../../glossary.md#term-producer))</span>
- Existing graph-only [DCE](../../../../glossary.md#term-dead-code-elimination-dce) behavior and tests remain unchanged.
- Sidecar-aware DCE prunes every unused constant source/fact after forward liveness, retains every
  bindable input, graph-output constant, consumed constant, non-forward root, and dependency, and
  rebuilds deterministic dense IDs.<span data-glossary-navigation="true"> ([liveness glossary](../../../../glossary.md#term-liveness); [bindable input glossary](../../../../glossary.md#bindable-input); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- The exact optional order is 0003A -> constant folding -> sidecar-aware DCE -> CSE ->
  sidecar-aware DCE, once each, with Compiler 0002 revalidation after every changed candidate.<span data-glossary-navigation="true"> ([constant folding glossary](../../../../glossary.md#term-constant-folding); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
- Disabled optimization performs no 0003A, folding, DCE, CSE, or constant pruning but still
  canonicalizes, remaps sidecar inputs, and validates.
- The maximum fact count is structural: at most one splat per graph input, and each folded
  one-output node replaces one existing value with one synthetic source. Evaluation is linear in
  graph size and constant in payload size; no element count or byte-size expansion occurs.<span data-glossary-navigation="true"> ([linear glossary](../../../../glossary.md#term-linear-meanings); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [element glossary](../../../../glossary.md#term-element))</span>
- Compiler 0005 is explicitly recorded as the owner that must transport the exact immutable facts
  into compile artifacts and exclude constant sources from runtime binding; prepare/backend remain
  the later physical materialization owners.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [immutable glossary](../../../../glossary.md#term-immutable); [compile artifacts glossary](../../../../glossary.md#compile-artifacts); [prepare glossary](../../../../glossary.md#prepare); [materialization glossary](../../../../glossary.md#materialization))</span>
- No public Java declaration, new operation kind, model component, physical allocation,
  dependency/build change, architecture change, or later task spec is added.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- All affected production Javadocs describe ownership, inputs, results, null/failure behavior,
  immutability, equality, source-role classification, ordering, and limitations.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Compile API, targeted Tensor API wording, glossary, task, master plan, and roadmap describe only
  the implemented current boundary and preserve planned lifecycle distinctions.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- A separate documentation-focused agent pass has finalized Javadocs, explanatory documentation,
  glossary impact, links, examples, planning evidence, and no-change conclusions in the same
  overall change.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Focused representation/[capture](../../../../glossary.md#term-capture-meanings) tests must cover:

- all six exact splat data types, signed-zero inequality, distinct NaN payloads, raw BFLOAT16,
  integral endpoints, BOOL, immutability, nulls, and value equality;<span data-glossary-navigation="true"> ([signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN glossary](../../../../glossary.md#term-nan))</span>
- ingress ordered snapshots, duplicate exact Tensor references, produced/gradient Tensor
  rejection, reachable/unreachable facts, and deterministic failures;<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- explicit scalar/zero/one/full/import leaf facts versus identical unbound leaves remaining
  bindable;<span data-glossary-navigation="true"> ([leaf glossary](../../../../glossary.md#term-leaf))</span>
- [mutable](../../../../glossary.md#term-mutable)/replaceable/dead [host storage](../../../../glossary.md#host-storage) having no effect on an already supplied splat or capture;
- static/dynamic/empty [Shapes](../../../../glossary.md#shape) and resolved/unresolved [layouts](../../../../glossary.md#layout);
- exact [graph](../../../../glossary.md#graph)-input-to-fact mapping and bindable-input order.

Focused fold tests must parameterize:

- [BOOL](../../../../glossary.md#data-type--datatype) NOT, AND, and OR truth tables;
- INT32 and INT64 ADD/SUB/MUL, including positive/negative values, zero, endpoints, and modular
  overflow;<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow))</span>
- [INT32](../../../../glossary.md#data-type--datatype) and [INT64](../../../../glossary.md#data-type--datatype) signed MIN/MAX;
- all four INT32/INT64 operand-width combinations for every integral family;
- all six signed comparisons, including equal, unequal, negative, and endpoint cases;
- same-typed INT32 and INT64 [scalar](../../../../glossary.md#term-scalar) ADD/SUB/MUL/MIN/MAX using exact `ScalarValueAttrs`, including
  operand-order-sensitive subtraction, endpoints, and modular overflow;
- same-scan propagation, deterministic synthetic source order, exact unchanged identity, and
  immutable result containers;<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [identity glossary](../../../../glossary.md#term-identity-meanings); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- graph-output, BACKWARD, multi-output, and unselected-[operation](../../../../glossary.md#operation) retention;
- floating/[BFLOAT16](../../../../glossary.md#term-bfloat16) facts with [signed zero](../../../../glossary.md#term-signed-zero), [infinities](../../../../glossary.md#nan-infinity-and-signed-zero), and NaNs remaining unevaluated;
- floating, CLAMP, and otherwise unselected `ScalarValueAttrs` operations remaining unevaluated,
  plus proof that selected integral attributes are never sidecar facts;
- dynamic Shapes and both layout states without [element](../../../../glossary.md#term-element) enumeration.

Focused [DCE](../../../../glossary.md#term-dead-code-elimination-dce)/pipeline tests must cover:

- unused ingress and synthetic [constant](../../../../glossary.md#term-constant) pruning;
- retained unused [bindable input](../../../../glossary.md#bindable-input);
- retained constant graph output and consumed constant;
- pruning after DCE removes a folded consumer and after [CSE](../../../../glossary.md#term-common-subexpression-elimination-cse) removes a duplicate consumer;
- non-forward roots and complete multi-output retention;
- dense deterministic remapping of graph, facts, bindable inputs, [nodes](../../../../glossary.md#node), phases, and outputs;
- exact enabled order, changed-only validation, disabled behavior, and one-shot pass counts.

During implementation, [run](../../../../glossary.md#term-run) focused tests as needed:

```bash
./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.CompileTimeConstantGraphTest
./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.ForwardConstantFoldingTest
./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.GraphCaptureTest
./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.ForwardDeadCodeEliminationTest
./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.ForwardGraphOptimizationTest
```

After executable Java stabilizes, the implementation context runs exactly one final [compiler](../../../../glossary.md#term-compiler)
[module](../../../../glossary.md#term-module) suite:

```bash
./gradlew :modules:compiler:test
```

This is task-tier validation. Repository-wide and architecture validation remain deferred to the
compiler transformation-and-[autograd](../../../../glossary.md#autograd) capability [checkpoint](../../../../glossary.md#term-checkpoint) after task 0004 or [continuous
integration](../../../../glossary.md#term-continuous-integration-ci) because this task changes one internal compiler module, no [dependency](../../../../glossary.md#term-dependency), no public [API](../../../../glossary.md#term-application-programming-interface-api),
no architecture rule, and no second module's executable behavior.

The implementation context records exact commands and XML counts and hands them to the separate
documentation context. The documentation context does not repeat successful Java tests unless it
changes executable behavior or records a concrete stale-evidence risk.

Documentation pass:

```bash
./gradlew :modules:compiler:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
{ git diff --name-only; git ls-files --others --exclude-standard; } | sort -u
git status --short
```

The documentation pass must also verify:

- exact package-private API shape and absence of public declarations;
- complete meaningful source and generated-[Javadoc](../../../../glossary.md#term-javadoc) coverage;
- exact 19-path maximum scope and the recorded overload-ambiguity reason for the added existing
  test path;<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- local Markdown links/anchors, balanced fences, final newlines, and trailing whitespace;
- no Tensor host-storage access, inferred factory constants, public binding, model operation kind,
  generic evaluator, dense payload, reflection/string dispatch, physical allocation, fixed point,
  autograd, or downstream implementation;<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [binding glossary](../../../../glossary.md#term-binding); [model glossary](../../../../glossary.md#term-model); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- selected kind/domain switches and complete negative [matrix](../../../../glossary.md#term-matrix-meanings);
- exact pipeline order and changed-only Compiler 0002 validation;
- graph-only DCE remains unchanged while sidecar-aware pruning preserves bindable inputs;
- task 0001–0003A remain `Complete`, 0003B becomes `Complete` only after all evidence, and 0004+
  remain Draft without detailed specs.

## Dependencies

- [Compiler](../../../../glossary.md#term-compiler) 0001 deterministic [structural capture](../../../../glossary.md#term-structural-capture) — Complete.
- Compiler 0002 captured-[graph](../../../../glossary.md#graph) [inference](../../../../glossary.md#term-inference)/[validation](../../../../glossary.md#term-validation) — Complete.
- Compiler 0003 [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form) and one-shot [DCE](../../../../glossary.md#term-dead-code-elimination-dce)/[CSE](../../../../glossary.md#term-common-subexpression-elimination-cse)/DCE — Complete.
- Compiler 0003A exact arithmetic rewriting — Complete.
- [Model](../../../../glossary.md#term-model) exact [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue), immutable descriptor/Shape/layout, Tensor/provenance, operation
  signatures, graph structural-source rules, phases, graph boundaries, and multi-output contracts
  — Complete.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [descriptor glossary](../../../../glossary.md#term-descriptor); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [Tensor glossary](../../../../glossary.md#tensor); [provenance glossary](../../../../glossary.md#provenance); [operation glossary](../../../../glossary.md#operation); [signatures glossary](../../../../glossary.md#term-signature))</span>
- Config 0002 optional-optimization permission — Complete.

No [runtime](../../../../glossary.md#term-runtime), [prepare](../../../../glossary.md#prepare), [backend](../../../../glossary.md#backend), physical storage, public compiler [facade](../../../../glossary.md#term-facade), or Compiler 0005 [artifact](../../../../glossary.md#term-artifact)
is a [dependency](../../../../glossary.md#term-dependency) for implementing this internal transformation. Compiler 0005 is a required
downstream consumer obligation: it must not discard the sidecar before prepare.

## Follow-up tasks

Future Draft [compiler](../../../../glossary.md#term-compiler) rows remain, in order:

- 0004 — construct backward graph state after the exact forward 0003A and 0003B boundaries,
  define any post-autograd optimization separately, and run the transformation/autograd
  capability checkpoint.<span data-glossary-navigation="true"> ([backward graph glossary](../../../../glossary.md#backward-graph); [run glossary](../../../../glossary.md#term-run); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- 0005 — transport the final immutable constant sidecar, or a named semantically identical
  constant-plan component, in <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>`CompileArtifacts`; derive public bindable inputs without constant
  sources; pass the unchanged graph to planning for logical requirements; and make constants
  available to prepare for physical materialization without compiler allocation. This is an
  explicit acceptance obligation and stop condition for 0005, not optional cleanup.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [materialization glossary](../../../../glossary.md#materialization))</span>

Possible future tasks may add exact cast folding, dense [constants](../../../../glossary.md#term-constant) with a separately justified
size/[ownership](../../../../glossary.md#term-ownership-meanings) budget, constant-control [identities](../../../../glossary.md#term-identity-meanings) after [autograd](../../../../glossary.md#autograd) semantics, or relaxed numerical
[evaluation](../../../../glossary.md#term-evaluation) only under new explicit contracts. Do not create those detailed specs now.

## Architecture impact

Expected impact: None.

The architecture explicitly assigns [constant folding](../../../../glossary.md#term-constant-folding) and [immutable](../../../../glossary.md#term-immutable) [compile](../../../../glossary.md#compile)-time [graph](../../../../glossary.md#graph) work to the
[compiler](../../../../glossary.md#term-compiler) and forbids physical [buffers](../../../../glossary.md#term-buffer), [backend](../../../../glossary.md#backend) execution, and [runtime](../../../../glossary.md#term-runtime) state there. This task uses
an internal immutable logical sidecar around the unchanged [model](../../../../glossary.md#term-model) graph and leaves physical
[materialization](../../../../glossary.md#materialization) to [prepare](../../../../glossary.md#prepare)/backend. The architecture's illustrative `CompileArtifacts` [shape](../../../../glossary.md#shape) does
not prohibit an immutable logical [constant](../../../../glossary.md#term-constant) component; task 0005 must choose the exact [artifact](../../../../glossary.md#term-artifact)
transport from its concrete consumer without inventing that public [facade](../../../../glossary.md#term-facade) here.

If implementation or task 0005 reveals that constant facts cannot be transported as immutable
compile-time artifacts without changing [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel), [module](../../../../glossary.md#term-module) [ownership](../../../../glossary.md#term-ownership-meanings), [dependency](../../../../glossary.md#term-dependency) rules,
or the [architecture contract](../../../../glossary.md#architecture-contract), stop and report that exact decision rather than hiding facts in
diagnostics, [publication](../../../../glossary.md#term-publication), runtime [bindings](../../../../glossary.md#term-binding), or [mutable](../../../../glossary.md#term-mutable) state.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, the compiler master plan, and
docs/planning/modules/compiler/tasks/0003b-compile-time-constants-and-constant-folding.md. Read the
directly referenced compiler/model/config/planning source, tests, API documentation, and completed
contracts needed to verify representation, source roles, exact folds, pruning, and pipeline order.

Implement Compiler 0003B exactly within its 19-path maximum. Add only the package-private logical
splat sidecar, explicit leaf ingress, selected BOOL/binary-integral/comparison/scalar-integral
folds, deterministic synthetic
constant sources, and sidecar-aware unused-constant pruning. Preserve graph-only DCE and the
existing public/model contracts. Place folding after Compiler 0003A and before the unchanged
one-shot DCE/CSE/DCE order, and revalidate every changed candidate through Compiler 0002.

Do not read Tensor storage, infer factory leaves as constants, add dense payloads, evaluate
floating/BFLOAT16/casts/unselected scalar attributes or arbitrary operations, expose constants as bindable
inputs, add a public facade, allocate physical storage, implement autograd/downstream lifecycle
work, or change dependencies/build/architecture. Stop on a representation, artifact-transport,
scope, numerical-proof, or architecture conflict.

After executable implementation and final compiler-test evidence, hand the actual diff and exact
evidence to a separate documentation-focused agent/thread with clean context. That pass must
follow docs/developer-guide/documentation-rules.md, finalize Javadocs, Compile/Tensor API,
glossary, planning status/evidence, terminology, links, examples, and no-change conclusions in
the same overall change, and must not repeat successful Java tests unless executable behavior
changes or a concrete stale-evidence risk is recorded.

Update this task with local decisions, exact validation evidence, implementation notes,
completion summary, and final status. Do not mark it Complete before every acceptance criterion
and the documentation pass finish.
```

## Local decisions

- Use one compiler-owned logical splat rather than dense bytes. It represents every current data
  type exactly with constant payload size and supports static, dynamic, scalar, and empty Shapes
  without element enumeration.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [logical splat glossary](../../../../glossary.md#logical-splat); [data type glossary](../../../../glossary.md#data-type--datatype); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [scalar glossary](../../../../glossary.md#term-scalar); [Shapes glossary](../../../../glossary.md#shape); [element glossary](../../../../glossary.md#term-element))</span>
- Represent [constant](../../../../glossary.md#term-constant) sources as existing structural [graph](../../../../glossary.md#graph) inputs plus a sidecar fact. This obeys
  [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel) closure without a new [model](../../../../glossary.md#term-model) [operation](../../../../glossary.md#operation) kind. Derived `bindableInputs()` keeps
  fixed sources out of caller/run binding.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [binding glossary](../../../../glossary.md#term-binding))</span>
- Match ingress by exact Tensor object identity and require reachable provenance-free non-gradient
  leaves. No factory history or mutable storage becomes semantic evidence.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-reference-identity); [leaves glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [provenance glossary](../../../../glossary.md#provenance); [gradient glossary](../../../../glossary.md#term-gradient); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- Select only four exact family rows: BOOL logic, signed-integral binary arithmetic, signed-
  integral comparison, and same-typed signed-integral scalar ADD/SUB/MUL/MIN/MAX over a known
  splat. Exclude floating/BFLOAT16 evaluation, casts, other scalar operations, and every
  element-traversing operation.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- Preserve graph-output producers and every multi-output occurrence. The first avoids changing
  requested occurrence/publication identity; the second avoids partial producer semantics.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [publication glossary](../../../../glossary.md#term-publication); [producer glossary](../../../../glossary.md#term-producer))</span>
- Place Compiler 0003B after 0003A. Scalar attributes remain operation metadata and are never
  inserted into the graph-value sidecar; 0003A uses selected identity attributes for bypass,
  whereas 0003B combines a selected integral attribute with a known graph-value splat to compute a
  new splat.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>
- Keep graph-only DCE unchanged. Its sidecar overload prunes only dead constant inputs, preserving
  every bindable input and live/non-forward dependency.<span data-glossary-navigation="true"> ([DCE glossary](../../../../glossary.md#term-dead-code-elimination-dce); [bindable input glossary](../../../../glossary.md#bindable-input); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Use graph-size structural budgeting instead of an arbitrary numeric limit: one scalar per fact,
  at most one fact per graph input, one synthetic source per removed one-output node, one scan, and
  no fixed point or per-element work.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- Require Compiler 0005 to transport the exact immutable facts into compile artifacts while
  leaving physical materialization to prepare/backend.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [compile artifacts glossary](../../../../glossary.md#compile-artifacts); [materialization glossary](../../../../glossary.md#materialization); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend))</span>
- Expand the authorized path maximum from 18 to 19 solely for
  `CapturedGraphInferenceTest`: the new internal overload makes its existing untyped null call
  ambiguous, so the graph-only null assertion now casts to `CompiledGraphModel`. This is a source
  compatibility repair in the affected module test, not additional behavior or scope.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [scope glossary](../../../../glossary.md#term-scope))</span>

## Known limitations

- Only logical splats are representable. Dense values, nonuniform values, sparse values, tuples,
  and storage snapshots are unsupported.
- Floating and [BFLOAT16](../../../../glossary.md#term-bfloat16) splats retain exact bits but participate in no fold in this task.
- Scalar operation attributes remain outside constant facts. Only integral ADD/SUB/MUL/MIN/MAX
  with a known same-typed splat input fold; floating, CLAMP, and other scalar operations remain.<span data-glossary-navigation="true"> ([Scalar glossary](../../../../glossary.md#term-scalar); [operation glossary](../../../../glossary.md#operation))</span>
- [Graph](../../../../glossary.md#graph)-output [producers](../../../../glossary.md#term-producer) and multi-output producers remain executable occurrences.
- Equal [constants](../../../../glossary.md#term-constant) are not interned, and [CSE](../../../../glossary.md#term-common-subexpression-elimination-cse) does not treat distinct constant-source IDs as equal.
- Constant-control bypass, cast folding, views, indexing, reductions, and numerical functions are
  deferred.<span data-glossary-navigation="true"> ([views glossary](../../../../glossary.md#term-view-meanings))</span>
- The sidecar is package-private compiler state. Compiler 0005 must select its public immutable
  artifact transport before prepare can materialize constants.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [immutable glossary](../../../../glossary.md#term-immutable); [artifact glossary](../../../../glossary.md#term-artifact); [prepare glossary](../../../../glossary.md#prepare))</span>
- No public compile entry exists, so current ingress is an internal foundation exercised through
  compiler tests and later compiler orchestration.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation-context Java evidence was reused as required by the documentation workflow:
  `./gradlew :modules:compiler:test` completed with `BUILD SUCCESSFUL`; compiler JUnit XML reported
  101 tests, 0 failures, 0 errors, and 0 skipped. That context also reported a passing<span data-glossary-navigation="true"> ([JUnit glossary](../../../../glossary.md#term-junit))</span>
  `git diff --check`. Documentation context `/root/docs_compiler_0003b` changed no executable Java
  after that run, so it did not repeat the successful Java suite.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- Documentation context `/root/docs_compiler_0003b` applied the General, API/Javadoc, Planning,
  and Example profiles. It independently reviewed <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Planning glossary](../../../../glossary.md#term-planning))</span>`AGENTS.md`, `ARCHITECTURE.md`, the current
  architecture index, documentation rules and profiles, planning guide, roadmap, compiler master
  plan, tasks 0001 through 0003A, this task, the final affected production source and tests,
  generated compiler Javadocs, Compile API, targeted Tensor API contracts, glossary, and the model
  graph/scalar/source-role boundaries referenced by those documents.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [Tensor glossary](../../../../glossary.md#tensor); [model glossary](../../../../glossary.md#term-model); [graph glossary](../../../../glossary.md#graph))</span>
- `./gradlew :modules:compiler:javadoc` completed with `BUILD SUCCESSFUL`; generated output
  includes `CompileTimeConstantGraph`, its `Splat`, [`Binding`](../../../../glossary.md#term-binding), and `Ingress` records,
  `ForwardConstantFolding`, the explicit `GraphCapture` overload, the sidecar validation overload,
  `ValidatedGraph`, sidecar-aware [DCE](../../../../glossary.md#term-dead-code-elimination-dce), and the final [optimizer](../../../../glossary.md#term-optimizer) contract.
- `python3 /tmp/validate_synaptik_markdown.py` passed for 236 Markdown files, 4,305 local links,
  266 local anchors, 3,006 fence markers, final newlines, and trailing whitespace.
- `git diff --check` passed after the documentation edits. The combined sorted tracked/untracked
  path audit reported exactly the authorized 19 paths. The 19th path is only
  `CapturedGraphInferenceTest.java`: the new internal overload made the existing untyped null call
  ambiguous, and the test now casts null to [`CompiledGraphModel`](../../../../glossary.md#compiled-graph--compiledgraphmodel) without changing the asserted
  graph-only behavior.
- Source and generated-Javadoc inspection confirmed that the new outer/nested records, helpers,
  overloads, and result remain package-private as planned; no public compiler facade or model
  declaration was added. Closed typed switches cover BOOL <span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>`NOT`/`AND`/`OR`, integral
  `ADD`/`SUB`/`MUL`/`MIN`/`MAX`, all six integral comparisons, and same-typed integral [scalar](../../../../glossary.md#term-scalar)
  `ADD`/`SUB`/`MUL`/`MIN`/`MAX`; `DIV`, `POW`, `CLAMP`, floating/BFLOAT16, casts, and every other
  family remain unevaluated.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>
- Source inspection confirmed exact optional order `0003A -> constant folding -> sidecar-aware
  DCE -> CSE -> sidecar-aware DCE`, once each, with identity-based changed-candidate revalidation.
  It also confirmed input-first deterministic rebuilds, synthetic sources in fold occurrence
  order, graph-only DCE preservation, bindable-input retention, and live-constant pruning.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [occurrence glossary](../../../../glossary.md#term-occurrence); [constant glossary](../../../../glossary.md#term-constant))</span>
- Negative-boundary inspection found no Tensor host-storage read, factory inference, dense
  payload, generic evaluator, reflection/string dispatch, physical allocation, fixed-point loop,
  autograd, public binding, downstream artifact/materialization implementation, or new operation
  kind. Existing iterative capture and DCE worklists are traversal mechanics, not constant-folding
  fixed points.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [inference glossary](../../../../glossary.md#term-inference); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [dispatch glossary](../../../../glossary.md#term-dispatch); [fixed-point glossary](../../../../glossary.md#fixed-point-arithmetic); [autograd glossary](../../../../glossary.md#autograd); [artifact glossary](../../../../glossary.md#term-artifact); [materialization glossary](../../../../glossary.md#materialization); [operation glossary](../../../../glossary.md#operation); [capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- [Scope](../../../../glossary.md#term-scope) inspection found no change to `ARCHITECTURE.md`, focused architecture documents, ADRs,
  Gradle/build files, dependencies, architecture tests, backend conformance, integration tests, or
  another module's Java. Architecture remains unchanged because the existing contract already
  assigns constant folding and immutable graph transformations to compiler and physical
  materialization to later prepare/backend work. Tensor/model public contracts remain unchanged
  because the sidecar is internal and retains no Tensor. Public Compile/Training Java contracts,
  capability contracts, build configuration, and unrelated modules require no changes for the
  same reason.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [module glossary](../../../../glossary.md#term-module); [constant folding glossary](../../../../glossary.md#term-constant-folding); [immutable glossary](../../../../glossary.md#term-immutable); [prepare glossary](../../../../glossary.md#prepare))</span>
- Status inspection confirmed [compiler](../../../../glossary.md#term-compiler) tasks 0001, 0002, 0003, 0003A, and 0003B are `Complete`;
  0004 and 0005 remain `Draft` master-plan rows with no detailed task specifications, and no
  compiler task is [`Ready`](../../../../glossary.md#term-ready).

## Implementation notes

- Added compiler-owned immutable logical-splat facts, explicit identity-based leaf ingress, and a
  derived non-bindable fixed-source classification without retaining Tensor or storage state.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [immutable glossary](../../../../glossary.md#term-immutable); [identity glossary](../../../../glossary.md#term-identity-meanings); [leaf glossary](../../../../glossary.md#term-leaf); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Added one closed one-scan fold implementation for the selected BOOL and signed-integral matrix.
  Folded values become deterministic synthetic structural inputs; no payload expands with Shape.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [BOOL glossary](../../../../glossary.md#data-type--datatype); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [Shape glossary](../../../../glossary.md#shape))</span>
- Carried source roles through validation, canonicalization, exact rewriting, CSE, and DCE.
  Sidecar-aware DCE removes dead fixed sources while graph-only DCE continues to retain every graph
  input.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [DCE glossary](../../../../glossary.md#term-dead-code-elimination-dce); [graph glossary](../../../../glossary.md#graph))</span>
- Integrated the exact enabled order after Compiler 0003A and preserved changed-only Compiler 0002
  revalidation and disabled-optimization canonicalization/validation.
- Finalized package-private Javadocs, Compile API, targeted Tensor API wording, glossary terms,
  compiler master-plan state, roadmap state, and this task's decisions/evidence. No executable Java
  was changed during the documentation-focused pass.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Completion summary

- Completed changes: implemented explicit immutable logical splats, identity-only capture ingress,
  fixed/bindable source classification, the exact selected constant-fold matrix, deterministic
  synthetic sources, sidecar-aware constant pruning, pipeline integration, and revalidation.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [identity glossary](../../../../glossary.md#term-identity-meanings); [capture glossary](../../../../glossary.md#term-capture-meanings); [constant glossary](../../../../glossary.md#term-constant); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Files changed or created: exactly the authorized 19 paths—seven compiler production paths, six
  compiler test paths, Compile API, Tensor API, glossary, this task, compiler master plan, and
  roadmap.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Tests and validation: reused the implementation context's successful compiler suite (101 tests,
  0 failures/errors/skips); documentation context passed compiler Javadoc, repository Markdown
  links/anchors/fences/newline/whitespace validation, exact scope/status/boundary checks, and final<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
  `git diff --check`.
- Documentation-agent review: completed by clean context `/root/docs_compiler_0003b` using the
  General, API/Javadoc, Planning, and Example profiles.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Compile API, targeted Tensor API status text, glossary, task, master plan,
  and roadmap now describe the implemented internal boundary and preserve planned public and
  physical-lifecycle distinctions.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Javadoc review: all affected package-private production contracts and generated pages were
  reviewed; the implementation-authored Javadocs were accurate and required no Java edit in this
  pass.
- Glossary impact: added reusable definitions for bindable input, compile-time constant source,
  and logical splat; updated forward-optimization status and boundaries.<span data-glossary-navigation="true"> ([bindable input glossary](../../../../glossary.md#bindable-input); [compile-time constant source glossary](../../../../glossary.md#compile-time-constant-source); [logical splat glossary](../../../../glossary.md#logical-splat))</span>
- Architecture impact: None. No architecture rule, dependency, module boundary, or build contract
  changed.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module))</span>
- Unresolved issues: None.
- Follow-up required: None for task 0003B. Compiler 0005 remains responsible for transporting the
  immutable facts in compile artifacts and excluding fixed sources from runtime binding; future
  prepare/backend work remains responsible for physical splat materialization. These are planned
  downstream obligations, not incomplete 0003B work.<span data-glossary-navigation="true"> ([compile artifacts glossary](../../../../glossary.md#compile-artifacts); [runtime glossary](../../../../glossary.md#term-runtime); [binding glossary](../../../../glossary.md#term-binding); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [materialization glossary](../../../../glossary.md#materialization))</span>

Status: Complete
