# Task 0003B: Compile-time Constants and Constant Folding

## Status

Complete

## Goal

Add one compiler-owned immutable representation for explicit compile-time constant facts and one
deliberately small constant-folding scan over a successful canonical `ValidatedGraph`.

The current representation is a **logical splat**: one exact typed `ScalarValue` applies to every
logical coordinate of one graph value. It is compile-time semantic data, not public mutable
`Tensor` state, host storage, a physical buffer, or a backend value. The compiler receives splats
only through an explicit package-private ingress. It never discovers them by reading
`Tensor.hostStorage()`, and a provenance-free factory leaf is an ordinary bindable graph input
unless the ingress explicitly names that exact Tensor reference.

The selected fold matrix has exactly these four rows:

| Family | Selected kinds | Selected domains |
|---|---|---|
| Boolean logical | `NOT`, `AND`, `OR` | exact `BOOL` splats |
| Binary arithmetic | `ADD`, `SUB`, `MUL`, `MIN`, `MAX` | exact `INT32`/`INT64` splats, including current mixed-width signed promotion |
| Binary comparison | `GREATER_THAN`, `GREATER_OR_EQUAL`, `LESS_THAN`, `LESS_OR_EQUAL`, `EQUAL`, `NOT_EQUAL` | exact `INT32`/`INT64` splats, including current mixed-width signed promotion |
| Scalar elementwise | `ADD`, `SUB`, `MUL`, `MIN`, `MAX` | one exact `INT32`/`INT64` splat plus exact same-typed immutable `ScalarValueAttrs` |

Every fold replaces one internal, one-output, non-gradient `FORWARD` occurrence with one synthetic
constant source. A synthetic constant source is structurally a graph input, as required by the
unchanged `CompiledGraphModel` source rules, but its sidecar fact excludes it from the derived
bindable-input list. No caller or later run may rebind it. Graph-output producers, multi-output
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

Dead-code elimination (DCE) and common-subexpression elimination (CSE) keep the one-shot order
established by Compiler 0003. The existing graph-only DCE entry keeps its exact contract and
retains every graph input. A new sidecar-aware overload additionally prunes a constant source when
it is neither a graph output nor an input of a retained node, while retaining every bindable input
even when unused. This prevents folded or ingress constants from becoming permanently live
materialization obligations.

Compiler 0003A scalar-attribute identities remain a separate earlier proof. Its exact
`ScalarValueAttrs` values are operation parameters, not graph-value constants. Compiler 0003A
reads an identity attribute alone to bypass a selected operation over an arbitrary graph input.
This task never places an attribute in the constant sidecar; it evaluates only the selected
integral scalar kinds when the graph-valued input is also an explicit constant splat.

## Mental model and examples

Read the sidecar and graph together:

```text
CompiledGraphModel.inputs()         every structural source
  - IDs absent from splat facts     bindable external inputs
  - IDs present in splat facts      fixed compile-time constant sources

CompileTimeConstantGraph
  = unchanged structural graph
  + immutable ValueId -> exact typed splat facts
  + derived bindable inputs
```

For a conceptual integral example, suppose explicit ingress marks two provenance-free leaves as
`INT32` splats `2` and `3`, and the captured graph contains their binary `ADD`:

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

The first sidecar-aware DCE prunes the now-unused `2` and `3` sources and retains the runtime input
and the live `5` source. IDs are rebuilt densely and deterministically. No host memory is read and
no buffer holding repeated fives is allocated.

For a dynamic Shape such as `[batch, 128]`, `splat(BOOL, true)` still means true at every logical
coordinate after `batch` is bound. The payload remains one scalar; compilation does not enumerate
elements or bind `batch`.

For a mixed-width example, `INT32(-1) < INT64(0)` promotes the signed `INT32` value to `INT64`,
compares it in signed order, and produces the `BOOL` splat `true`. `INT32(MAX_VALUE) + INT32(1)`
produces exact modular `INT32(MIN_VALUE)`. These are model semantics, not Java widening accidents.

## Scope

- Add one package-private immutable `CompileTimeConstantGraph` aggregate in
  `io.github.pho001.synaptik.compiler`.
- Nest exactly three package-private immutable request/value records in that type:
  `Splat`, `Binding`, and `Ingress`.
- Keep the existing public model graph types and record components unchanged.
- Keep the existing `GraphCapture.capture(List<Tensor>) -> CompiledGraphModel` entry and behavior
  unchanged for callers with no constant ingress.
- Add one package-private GraphCapture overload accepting explicit `Ingress` and returning
  `CompileTimeConstantGraph`.
- Match ingress bindings by exact Tensor object identity. Do not use labels, storage identity,
  structural descriptor equality, or a global registry.
- Require every ingress binding to name one reachable provenance-free leaf. Reject produced
  Tensors and unrelated leaves rather than silently ignoring them.
- Treat an unbound provenance-free Tensor as an ordinary bindable graph input even when it was
  created by `TensorFactory.scalar`, `zeros`, `ones`, `full`, an import, allocation, or random
  factory path.
- Carry the sidecar through `ValidatedGraph`, canonicalization, Compiler 0003A rewriting, DCE,
  CSE, and Compiler 0002 revalidation without adding a public compile facade.
- Add one package-private stateless `ForwardConstantFolding` helper with this shape:

  ```java
  static CompileTimeConstantGraph fold(CompileTimeConstantGraph constantGraph)
  ```

- Visit nodes once in stored topological order, propagating only the selected splat facts so one
  earlier selected fold can make a later selected occurrence foldable in the same scan.
- Fold exactly the four-row matrix in [Selected fold matrix](#selected-fold-matrix).
- Return the exact sidecar argument when no occurrence folds.
- When folding changes the graph, rebuild one immutable canonical candidate with original graph
  inputs first in boundary order, synthetic folded constants next in original node/output order,
  then retained node outputs in topological/output-slot order.
- Keep graph-output producers as occurrences and preserve graph-output order and identity
  separation.
- Keep every multi-output occurrence indivisible and unfolded.
- Extend `ForwardDeadCodeElimination` with one sidecar-aware overload. Do not change its existing
  graph-only signature or behavior.
- Prune only unused constant-source inputs/facts after graph-only forward DCE. Preserve all
  bindable inputs, every graph output, every constant consumed by a retained node, every
  non-forward node, and their dependency closures.
- Integrate constant folding after Compiler 0003A and before Compiler 0003's unchanged one-shot
  DCE/CSE/DCE order.
- Run folding and sidecar-aware optional passes only when
  `GraphOptimizationConfig.optionalOptimizationsEnabled()` is true. Disabled optimization still
  performs mandatory canonicalization, sidecar remapping, and Compiler 0002 validation.
- Revalidate every changed immutable sidecar graph through Compiler 0002 before another
  transformation consumes it. Do not revalidate an exact unchanged sidecar result.
- Add complete Javadocs to every affected package-private production contract.
- Add focused tests for representation, ingress, fold semantics, pruning, pipeline order, IDs,
  failure order, immutability, and exclusions.
- Update Compile API, targeted Tensor API status text, glossary, this task, compiler master plan,
  and roadmap after implementation.
- Complete the required separate documentation-focused pass in the same overall change.

### Compiler-owned sidecar shape

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
members, a builder, registry, service, listener, serializer, generic payload hierarchy, or
physical materialization method.

`Splat` owns exactly one non-null immutable `ScalarValue` reference. Record equality and hashing
therefore use `ScalarValue`'s exact data type and bit equality: floating signed zeros and distinct
NaN payloads remain different, raw BFLOAT16 patterns remain exact, and integral/BOOL values remain
canonical in their existing representation. Retaining the immutable reference is the complete
snapshot; no host storage, array, segment, Tensor, descriptor, or Shape is retained in a splat.

`Binding` owns one exact non-null Tensor reference and one non-null splat. It rejects a Tensor with
present provenance because ingress is limited to leaves. It rejects a Tensor whose descriptor has
`requiresGrad == true`; deleting or fixing a gradient-eligible leaf before Compiler 0004 would
pre-decide parameter and gradient semantics.

`Ingress` snapshots a non-null ordered binding list with `List.copyOf`, validates null elements in
encounter order, and rejects the first later binding that repeats an exact Tensor reference. Its
list order controls deterministic diagnostics only; graph input order still follows GraphCapture
encounter order. `empty()` returns an ingress with an empty immutable list.

The outer record snapshots `constants` with `Map.copyOf` after validating:

1. `graph`, then `constants`, are non-null;
2. no key or value is null;
3. every constant key is in `graph.inputs()`;
4. the splat data type equals that graph input's descriptor data type; and
5. the descriptor has `requiresGrad == false`.

Validate known graph inputs in graph-input order so descriptor/type/gradient failures are
deterministic. Reject an extra fact key that is not a graph input in ascending numeric `ValueId`
order. Do not require a static Shape, resolved layout, canonical density, positive element count,
or physical storage.

`bindableInputs()` returns a new immutable list in exact `graph.inputs()` order containing only
IDs absent from `constants`. It is the sole current classification of caller-bindable inputs. A
constant source must never also appear in this result.

`replaceGraphPreservingInputRoles` supports graph-only transformations that preserve the ordered
source boundary. It requires the replacement to have the same input count and equal descriptor at
every input position, remaps each old constant fact to the replacement input at the same position,
and returns the exact receiver if `replacement == graph`. It rejects a count or descriptor
contradiction instead of guessing a correspondence. Constant folding and constant-aware pruning,
which add or remove constant sources, construct a new sidecar directly.

### Explicit ingress and capture

Add exactly this overload while preserving the existing entry:

```java
static CompileTimeConstantGraph capture(
        List<Tensor> outputs,
        CompileTimeConstantGraph.Ingress ingress)
```

The existing one-argument method delegates through `Ingress.empty()` and returns only `.graph()`;
its observable validation, capture order, IDs, graph structure, and exact return type remain
unchanged.

The overload applies the existing output-container, empty, element, and duplicate checks first,
then rejects null `ingress` with `NullPointerException("ingress")`. The ingress records own their
own deterministic construction validation. Capture traverses exactly as today. When a reachable
provenance-free leaf is first allocated, capture looks up that exact Tensor reference in the
ingress and, when present, associates the resulting graph input ID with the supplied splat.

After traversal and requested-output resolution, reject the first ingress binding whose exact
Tensor was not encountered as a reachable leaf. Use its ingress-list index in a deterministic
message. This prevents a typo or unrelated binding from being accepted as if it affected
compilation. A requested provenance-free Tensor may be both a constant source and a graph output;
it remains in both graph boundary lists and is excluded from `bindableInputs()`.

Capture reads only Tensor identity, descriptor, and provenance as it already does. It must not call
`hostStorage()`, inspect `HostTensorStorage`, read a `MemorySegment`, recognize a factory method,
or infer a splat from a label, layout, storage contents, zero-sized shape, or other metadata.

### Selected fold matrix

For a node encountered in topological order, create and propagate a result splat only when all of
these common conditions hold:

1. the node phase is exactly `GraphPhase.FORWARD`;
2. no node output is in `graph.outputs()`;
3. the operation kind and exact attributes match one selected row;
4. the signature supplies the selected fixed input count and exactly one output;
5. every input position has a known splat fact;
6. every input fact's type agrees with its already validated input descriptor;
7. the sole output descriptor has `requiresGrad == false`; and
8. exact evaluation produces a splat whose type equals the stored output descriptor type.

Selected operation rows are:

| Operation | Attributes | Inputs | Result |
|---|---|---|---|
| `BooleanLogicalKind.NOT` | exact `NoOperationAttrs.INSTANCE` | one BOOL splat | canonical BOOL negation |
| `BooleanLogicalKind.AND`, `OR` | exact `NoOperationAttrs.INSTANCE` | two BOOL splats | canonical BOOL conjunction/disjunction |
| `BinaryArithmeticKind.ADD`, `SUB`, `MUL` | exact `NoOperationAttrs.INSTANCE` | two integral splats | fixed-width modular result in the validated promoted type |
| `BinaryArithmeticKind.MIN`, `MAX` | exact `NoOperationAttrs.INSTANCE` | two integral splats | signed-order selected value in the validated promoted type |
| all six `BinaryComparisonKind` values | exact `NoOperationAttrs.INSTANCE` | two integral splats | canonical BOOL result after signed promotion and comparison |
| `ScalarElementwiseKind.ADD`, `SUB`, `MUL` | exact `ScalarValueAttrs` | one integral splat plus an exact same-typed scalar attribute | fixed-width modular result in the unchanged input/output type |
| `ScalarElementwiseKind.MIN`, `MAX` | exact `ScalarValueAttrs` | one integral splat plus an exact same-typed scalar attribute | signed-order selected value in the unchanged input/output type |

For two `INT32` operands, calculate in signed 32-bit primitives so ADD/SUB/MUL wrap modulo
`2^32`. If either operand is `INT64`, sign-extend any `INT32` operand and calculate or compare in
signed 64-bit primitives; ADD/SUB/MUL wrap modulo `2^64`. MIN/MAX and all relational operators use
ordinary signed order. Equality compares the promoted exact signed values. Construct results only
through the existing named `ScalarValue.int32`, `int64`, and `bool` factories.

Scalar evaluation preserves operand order: the graph input is the left operand and
`ScalarValueAttrs.value()` is the right operand. Require the attribute value type to equal the
input and output descriptor type exactly. Apply the same INT32/INT64 modular and signed-order
rules without promotion. Integral scalar DIV and POW are not current valid domains and remain
excluded.

Do not dispatch through strings, reflection, `Number`, `double`, a generic evaluator interface, or
an operation registry. Use closed typed switches after exact kind/attributes checks. Invalid
operation/descriptor combinations cannot reach this pass because Compiler 0002 already accepted
the graph; the fold helper retains an unselected occurrence rather than becoming another semantic
validator.

### Deliberately excluded evaluation matrix

| Candidate | This task's decision |
|---|---|
| Floating ADD/SUB/MUL/DIV/POW/MIN/MAX | Not folded. Rounding, intermediate precision, NaN payload/sign, signed zero, infinities, underflow, overflow, and backend numerical permissions are not reduced to Java evaluation here. |
| BFLOAT16 arithmetic | Not folded. Raw BFLOAT16 splats remain representable exactly, but no current compiler numerical contract selects an arithmetic rounding implementation. |
| Floating comparisons | Not folded in this bounded matrix, even where a result could be defined from value classes. |
| Casts | Not folded. Conversion, overflow, NaN, signed-zero, and BFLOAT16 rounding need a separate exact conversion matrix. |
| Other scalar elementwise operations | Floating scalar operations, CLAMP, and every kind outside integral ADD/SUB/MUL/MIN/MAX are not folded. `ScalarValueAttrs` remains operation metadata and is never a graph-value fact. |
| Unary numerical functions or activation functions | Not folded. The model names mathematical targets without a compiler-selected correctly rounded algorithm. |
| Reductions, scans, softmax, normalization, loss, matrix multiplication, convolution, pooling | Not folded. They require element traversal, domain policies, or numerical algorithms and would violate the splat-only bounded-work contract. |
| Shape/layout/index/view operations | Not folded. This task does not reinterpret aliases, indices, bounds, materialization, or layout. |
| Random/state/dropout operations | Never constant; explicit state and randomness remain semantic sources. |
| `WHERE` and constant-control bypass | Deferred. Bypassing a branch can change future gradient occurrence and operand-use semantics. |
| Multi-output operations | Never partially folded; the complete producer remains. |
| Graph-output producer | Remains distinct even when every input is constant, preserving requested occurrence and later publication identity. |

Floating, BFLOAT16, and BOOL/integral splats all have exact representation equality. The exclusion
is from evaluation, not from the payload type. A FLOAT32 negative-zero splat remains different
from positive zero; a FLOAT64 or BFLOAT16 NaN retains its raw payload. No selected fold consumes
those facts, canonicalizes them, or promises a result payload.

### Rebuild, source roles, and deterministic IDs

The fold scan first records selected fold decisions and propagated facts in original topological
order. It performs no graph mutation. If no node folds, return the exact sidecar reference.

For a changed result, rebuild in two deterministic stages:

1. allocate every original graph input in exact boundary order and preserve its bindable or
   constant role;
2. allocate one synthetic constant input for each folded node in original topological and output-
   slot order; then allocate retained node outputs while visiting retained nodes in original
   topological order.

All input IDs are therefore dense from zero before any retained node output ID. Retained node IDs
are dense in retained topological order. Remap every retained ordered input, graph output, phase,
operation, and descriptor. Retain exact operation, descriptor, phase, and splat references where
their contracts permit. A folded node and its original output are not retained; later uses map to
the corresponding synthetic constant input.

Do not merge equal splats or reuse an earlier constant source merely because exact payload and
descriptor values match. Constant-source interning would change graph-value identity and source
occurrence policy and is not needed for this bounded task. CSE remains based on graph input IDs,
not constant value equality.

### Sidecar-aware dead constant pruning

Keep this existing entry unchanged:

```java
static CompiledGraphModel eliminate(CompiledGraphModel graph)
```

Add exactly this overload:

```java
static CompileTimeConstantGraph eliminate(CompileTimeConstantGraph constantGraph)
```

The overload first applies the existing graph-only forward DCE and remaps source roles by input
position. It then considers a constant source live exactly when its ID is a graph output or occurs
in any retained node input position. Every bindable input remains live regardless of use. Rebuild
only when graph DCE or constant pruning changed the candidate, using bindable and live-constant
inputs in their existing relative graph-input order, followed by retained node outputs in stored
topological/output order. Rebuild dense canonical IDs and remap facts, nodes, phases, and outputs.

This pruning is structural only. It does not inspect payload values, merge constants, remove a
bindable input, remove a graph-output constant, or decide physical materialization. An unchanged
sidecar result returns the exact argument.

### Pipeline and failure ordering

`ForwardGraphOptimization.optimize` retains its two existing null checks in order. Mandatory work
is:

1. canonicalize `validatedGraph.graph()`;
2. remap the validated sidecar by input position;
3. validate the canonical sidecar graph through Compiler 0002; and
4. return that result immediately when optional optimization is disabled.

When enabled, continue exactly:

5. run Compiler 0003A rewriting once and remap its unchanged input roles;
6. validate only if changed;
7. run `ForwardConstantFolding.fold` once;
8. validate only if changed;
9. run sidecar-aware forward DCE once and validate only if changed;
10. run existing exact forward CSE once, remap input roles, and validate only if changed;
11. run sidecar-aware forward DCE once and validate only if changed; and
12. return the final exact `ValidatedGraph`.

No pass repeats, no fixed point is sought, and no second folding scan runs after CSE. Compiler
0002 continues to infer operation descriptors and deferred constraints from the graph only; the
sidecar constructor separately validates constant-source agreement. Add exactly this package-
private Compiler 0002 overload while preserving its existing entry:

```java
static ValidatedGraph inferAndValidate(CompileTimeConstantGraph constantGraph)
```

The existing `inferAndValidate(CompiledGraphModel)` delegates through
`CompileTimeConstantGraph.withoutConstants(graph)`. Both preserve the existing null and semantic
failure behavior for their declared argument.

Change the package-private successful result to this internal shape:

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
`constants()` and `bindableInputs()` delegate to that immutable sidecar. The record snapshots
constraints exactly as today. Do not add another stored graph, fact map, or bindable-input list.

## Out of scope

- reading, snapshotting, hashing, comparing, or trusting public Tensor host-storage contents
- inferring constants from `TensorFactory` method history, provenance absence, labels, layouts,
  zero-sized Shapes, storage allocation, or Tensor identity alone
- allowing a constant source to be rebound as a public/run input
- public compile configuration, public constant binding, `GraphCompiler`, `CompiledGraph`, engine
  facade, registry, service, or general compile entry point
- dense, sparse, array, byte-buffer, memory-segment, per-element, tuple, string, boxed, or generic
  constant payloads
- materializing a splat, allocating host/device/physical buffers, choosing a storage layout,
  determining residency, or retaining mutable memory
- folding any operation outside the exact four-row matrix
- floating or BFLOAT16 arithmetic/comparison, cast evaluation, numerical unary evaluation,
  reduction, scan, softmax, normalization, loss, linear algebra, convolution, pooling, index,
  layout/view, random, state, or dropout evaluation
- general partial evaluation, arbitrary interpretation of `Operation`, a generic evaluator,
  registry, pass framework, string dispatch, reflection, or plugin mechanism
- constant interning, deduplication by value, serialization, persistence, hashing protocol, or
  artifact file format
- large or unbounded compile work, per-element loops, data-dependent loops, recursive evaluation,
  fixed-point iteration, or payload growth beyond one scalar per fact
- relaxed/fast-math, reassociation, commutation, reciprocal substitution, finite-only assumptions,
  approximate evaluation, tolerance-based equality, or backend numerical selection
- autograd, backward construction, gradient rules, saved-value policy, gradient-eligible constant
  leaves, post-autograd folding, or training behavior
- changing graph-output producers, publication bindings, public output identity, or multi-output
  producer structure
- changing `CompiledGraphModel`, `GraphValue`, `CompiledNode`, `Operation`, `ScalarValue`, Tensor,
  storage, model semantics, config, planning, prepare, runtime, backend, or engine contracts
- final public `CompileArtifacts`, `ConstantPlan`, prepare materialization, or task 0005
  orchestration implementation
- dependencies, Gradle/build configuration, Java version, architecture contract, focused
  architecture documents, ADRs, or architecture tests
- later compiler task specifications

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially compiler ownership of
  constant folding, immutable compile-time state, the compile lifecycle, and the prohibition on
  physical buffers and runtime/backend state
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Planning guide](../../../planning-guide.md)
- [Compiler master plan](../master-plan.md)
- [Compiler 0001 capture](0001-tensor-expression-graph-capture.md)
- [Compiler 0002 validation](0002-captured-graph-inference-and-validation.md)
- [Compiler 0003 canonicalization and optimization](0003-canonicalization-and-forward-optimization.md)
- [Compiler 0003A exact arithmetic rewriting](0003a-exact-arithmetic-rewriting.md)
- [Model compiled graph task](../../model/tasks/0009-compiled-graph-model.md)
- [Model constant creation task](../../model/tasks/0012d-constant-tensor-creation.md)
- [Model provenance task](../../model/tasks/0018l-shared-multi-output-tensor-provenance.md)
- [Model typed scalar task](../../model/tasks/0018n-typed-scalar-value-contract.md)
- [Model capability closure audit](../../model/model-capability-contract-closure-audit.md)
- [Planning logical memory task](../../planning/tasks/0005-logical-materialization-and-memory-requirements.md)
- [Compile API](../../../../api/compile-api.md), [Tensor API](../../../../api/tensor-api.md), and
  [glossary](../../../../glossary.md)

## Architecture constraints

- Compiler owns constant recognition, propagation, folding, graph rewriting, and the immutable
  logical sidecar.
- Model graph and scalar contracts remain unchanged. A constant source uses the existing valid
  graph-input structural form rather than a new model operation kind or `GraphValue` component.
- `Tensor` remains mutable public API state and never becomes authoritative constant data.
- Compiler must not allocate physical buffers, read host storage, construct runtime values,
  execute a backend, select a numerical kernel, or create prepared state.
- Every constant source is classified out of `bindableInputs()`; there is no hidden runtime
  rebinding or assumption that a later supplied input equals a compile-time value.
- Compiler 0005 must preserve and transport the exact immutable constant sidecar, or a named
  constant-plan component with exactly equivalent semantics, in the future immutable compile
  artifacts. It must give planning the unchanged graph for logical requirements, derive public
  bindable inputs from this task's classification, and make the logical splat facts available to
  prepare without turning them into physical buffers. Task 0005 must stop if its artifact design
  would discard facts, expose constant sources as bindable inputs, or require a model/architecture
  change.
- Planning may treat a constant source as a logical graph value requiring materialization. It does
  not inspect the payload, allocate memory, or select implementation routes.
- Future prepare/backend work owns physical splat materialization, storage allocation, lowering,
  and execution. Runtime binds only the nonconstant inputs selected by the prepared contract.
- The compiler package remains internal. No facade, registry, service locator, or generic
  evaluator is justified.
- If implementation cannot carry source roles through every changed candidate, prune unused
  constants deterministically, or preserve the sidecar for Compiler 0005 without changing the
  architecture contract, stop and report the exact conflict.

## Package impact

Existing package used:

- `io.github.pho001.synaptik.compiler` — owns internal capture, validation, canonicalization,
  exact rewriting, constant facts/folding, and the bounded forward pipeline.

Packages added or changed:

- No package is added. The compiler root gains two package-private types and targeted changes to
  five existing package-private types.

Type placement:

- `io.github.pho001.synaptik.compiler.CompileTimeConstantGraph` — owns the compiler-only graph
  sidecar, exact splat value, explicit ingress binding, bindable-input derivation, and safe
  input-position remapping.
- `io.github.pho001.synaptik.compiler.ForwardConstantFolding` — owns the selected one-scan typed
  evaluation and synthetic-source rebuild.
- `io.github.pho001.synaptik.compiler.GraphCapture` — owns the only Tensor-identity-to-graph-ID
  moment and therefore maps explicit ingress facts without retaining Tensor state.
- `io.github.pho001.synaptik.compiler.ValidatedGraph` and `CapturedGraphInference` — retain and
  revalidate the compiler-only sidecar without changing model inference.
- `io.github.pho001.synaptik.compiler.ForwardDeadCodeElimination` — owns liveness and adds the
  sidecar-aware unused-constant-source pruning overload while keeping graph-only behavior intact.
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

Expected documentation and planning files:

- `docs/api/compile-api.md`
- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/compiler/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum scope

This task may create or modify at most the 19 paths listed above.

The larger path count is justified because one cohesive compiler invariant must survive capture,
validation, every ID-rebuilding transformation, constant-aware liveness, tests, Javadocs, and
current-status documentation atomically. Splitting the sidecar from folding would either create an
unused representation or permit a graph transform to discard source roles. The authorized maximum
expanded from 18 to 19 only because adding the package-private
`CapturedGraphInference.inferAndValidate(CompileTimeConstantGraph)` overload made the existing
untyped null invocation ambiguous; `CapturedGraphInferenceTest` now casts that null explicitly to
`CompiledGraphModel` and preserves the original graph-only failure assertion.

Do not modify model/config/planning/prepare/runtime/backend/engine Java, Gradle, architecture,
conformance, integration, or additional documentation paths. If another production type, test
class, public type, payload form, path, or module is required, stop and propose a follow-up rather
than exceeding this maximum.

## Acceptance criteria

- `CompileTimeConstantGraph` and its nested `Splat`, `Binding`, and `Ingress` records have exactly
  the planned package-private shape, immutability, validation, and Javadocs.
- Splat equality preserves exact type/bits for every current data type, including raw BFLOAT16,
  floating signed zero and NaN payloads, exact integrals, and canonical BOOL.
- Payloads retain no Tensor, Shape, descriptor, storage, array, segment, physical buffer, backend,
  or runtime value.
- Explicit ingress accepts only ordered unique exact provenance-free non-gradient Tensor leaves;
  unrelated and produced Tensors fail deterministically.
- GraphCapture's existing method is behaviorally unchanged. The overload maps only explicit facts
  and never reads host storage or infers factory constants.
- A provenance-free leaf absent from ingress remains bindable, including factory scalar/zero/one/
  full/import/random leaves.
- `bindableInputs()` is exactly the graph input list minus constant fact keys, in graph input order.
- Canonicalization, 0003A, CSE, and validation preserve source roles by exact input position and
  reject a source-boundary contradiction.
- The fold helper implements every selected BOOL, binary-integral, comparison, and same-typed
  scalar-integral row, all binary same/mixed integral width combinations, signed comparison, and
  modular overflow exactly.
- The fold helper implements no floating/BFLOAT16 evaluation, cast, scalar operation outside the
  selected same-typed integral row, unary numerical function, reduction, layout/index operation,
  random/state operation, or generic evaluator behavior.
- One scan propagates earlier selected folded splats to later selected occurrences without a fixed
  point or second scan.
- Changed folding creates deterministic dense input-first IDs, places synthetic constants after
  original inputs in fold occurrence order, and preserves retained node/output order and exact
  descriptors/operations/phases.
- Graph-output producers, BACKWARD nodes, gradient-eligible values, and all multi-output producers
  remain occurrences.
- Existing graph-only DCE behavior and tests remain unchanged.
- Sidecar-aware DCE prunes every unused constant source/fact after forward liveness, retains every
  bindable input, graph-output constant, consumed constant, non-forward root, and dependency, and
  rebuilds deterministic dense IDs.
- The exact optional order is 0003A -> constant folding -> sidecar-aware DCE -> CSE ->
  sidecar-aware DCE, once each, with Compiler 0002 revalidation after every changed candidate.
- Disabled optimization performs no 0003A, folding, DCE, CSE, or constant pruning but still
  canonicalizes, remaps sidecar inputs, and validates.
- The maximum fact count is structural: at most one splat per graph input, and each folded
  one-output node replaces one existing value with one synthetic source. Evaluation is linear in
  graph size and constant in payload size; no element count or byte-size expansion occurs.
- Compiler 0005 is explicitly recorded as the owner that must transport the exact immutable facts
  into compile artifacts and exclude constant sources from runtime binding; prepare/backend remain
  the later physical materialization owners.
- No public Java declaration, new operation kind, model component, physical allocation,
  dependency/build change, architecture change, or later task spec is added.
- All affected production Javadocs describe ownership, inputs, results, null/failure behavior,
  immutability, equality, source-role classification, ordering, and limitations.
- Compile API, targeted Tensor API wording, glossary, task, master plan, and roadmap describe only
  the implemented current boundary and preserve planned lifecycle distinctions.
- A separate documentation-focused agent pass has finalized Javadocs, explanatory documentation,
  glossary impact, links, examples, planning evidence, and no-change conclusions in the same
  overall change.

## Tests / validation

Focused representation/capture tests must cover:

- all six exact splat data types, signed-zero inequality, distinct NaN payloads, raw BFLOAT16,
  integral endpoints, BOOL, immutability, nulls, and value equality;
- ingress ordered snapshots, duplicate exact Tensor references, produced/gradient Tensor
  rejection, reachable/unreachable facts, and deterministic failures;
- explicit scalar/zero/one/full/import leaf facts versus identical unbound leaves remaining
  bindable;
- mutable/replaceable/dead host storage having no effect on an already supplied splat or capture;
- static/dynamic/empty Shapes and resolved/unresolved layouts;
- exact graph-input-to-fact mapping and bindable-input order.

Focused fold tests must parameterize:

- BOOL NOT, AND, and OR truth tables;
- INT32 and INT64 ADD/SUB/MUL, including positive/negative values, zero, endpoints, and modular
  overflow;
- INT32 and INT64 signed MIN/MAX;
- all four INT32/INT64 operand-width combinations for every integral family;
- all six signed comparisons, including equal, unequal, negative, and endpoint cases;
- same-typed INT32 and INT64 scalar ADD/SUB/MUL/MIN/MAX using exact `ScalarValueAttrs`, including
  operand-order-sensitive subtraction, endpoints, and modular overflow;
- same-scan propagation, deterministic synthetic source order, exact unchanged identity, and
  immutable result containers;
- graph-output, BACKWARD, multi-output, and unselected-operation retention;
- floating/BFLOAT16 facts with signed zero, infinities, and NaNs remaining unevaluated;
- floating, CLAMP, and otherwise unselected `ScalarValueAttrs` operations remaining unevaluated,
  plus proof that selected integral attributes are never sidecar facts;
- dynamic Shapes and both layout states without element enumeration.

Focused DCE/pipeline tests must cover:

- unused ingress and synthetic constant pruning;
- retained unused bindable input;
- retained constant graph output and consumed constant;
- pruning after DCE removes a folded consumer and after CSE removes a duplicate consumer;
- non-forward roots and complete multi-output retention;
- dense deterministic remapping of graph, facts, bindable inputs, nodes, phases, and outputs;
- exact enabled order, changed-only validation, disabled behavior, and one-shot pass counts.

During implementation, run focused tests as needed:

```bash
./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.CompileTimeConstantGraphTest
./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.ForwardConstantFoldingTest
./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.GraphCaptureTest
./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.ForwardDeadCodeEliminationTest
./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.ForwardGraphOptimizationTest
```

After executable Java stabilizes, the implementation context runs exactly one final compiler
module suite:

```bash
./gradlew :modules:compiler:test
```

This is task-tier validation. Repository-wide and architecture validation remain deferred to the
compiler transformation-and-autograd capability checkpoint after task 0004 or continuous
integration because this task changes one internal compiler module, no dependency, no public API,
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
- complete meaningful source and generated-Javadoc coverage;
- exact 19-path maximum scope and the recorded overload-ambiguity reason for the added existing
  test path;
- local Markdown links/anchors, balanced fences, final newlines, and trailing whitespace;
- no Tensor host-storage access, inferred factory constants, public binding, model operation kind,
  generic evaluator, dense payload, reflection/string dispatch, physical allocation, fixed point,
  autograd, or downstream implementation;
- selected kind/domain switches and complete negative matrix;
- exact pipeline order and changed-only Compiler 0002 validation;
- graph-only DCE remains unchanged while sidecar-aware pruning preserves bindable inputs;
- task 0001–0003A remain `Complete`, 0003B becomes `Complete` only after all evidence, and 0004+
  remain Draft without detailed specs.

## Dependencies

- Compiler 0001 deterministic structural capture — Complete.
- Compiler 0002 captured-graph inference/validation — Complete.
- Compiler 0003 canonicalization and one-shot DCE/CSE/DCE — Complete.
- Compiler 0003A exact arithmetic rewriting — Complete.
- Model exact `ScalarValue`, immutable descriptor/Shape/layout, Tensor/provenance, operation
  signatures, graph structural-source rules, phases, graph boundaries, and multi-output contracts
  — Complete.
- Config 0002 optional-optimization permission — Complete.

No runtime, prepare, backend, physical storage, public compiler facade, or Compiler 0005 artifact
is a dependency for implementing this internal transformation. Compiler 0005 is a required
downstream consumer obligation: it must not discard the sidecar before prepare.

## Follow-up tasks

Future Draft compiler rows remain, in order:

- 0004 — construct backward graph state after the exact forward 0003A and 0003B boundaries,
  define any post-autograd optimization separately, and run the transformation/autograd
  capability checkpoint.
- 0005 — transport the final immutable constant sidecar, or a named semantically identical
  constant-plan component, in `CompileArtifacts`; derive public bindable inputs without constant
  sources; pass the unchanged graph to planning for logical requirements; and make constants
  available to prepare for physical materialization without compiler allocation. This is an
  explicit acceptance obligation and stop condition for 0005, not optional cleanup.

Possible future tasks may add exact cast folding, dense constants with a separately justified
size/ownership budget, constant-control identities after autograd semantics, or relaxed numerical
evaluation only under new explicit contracts. Do not create those detailed specs now.

## Architecture impact

Expected impact: None.

The architecture explicitly assigns constant folding and immutable compile-time graph work to the
compiler and forbids physical buffers, backend execution, and runtime state there. This task uses
an internal immutable logical sidecar around the unchanged model graph and leaves physical
materialization to prepare/backend. The architecture's illustrative `CompileArtifacts` shape does
not prohibit an immutable logical constant component; task 0005 must choose the exact artifact
transport from its concrete consumer without inventing that public facade here.

If implementation or task 0005 reveals that constant facts cannot be transported as immutable
compile-time artifacts without changing `CompiledGraphModel`, module ownership, dependency rules,
or the architecture contract, stop and report that exact decision rather than hiding facts in
diagnostics, publication, runtime bindings, or mutable state.

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
  without element enumeration.
- Represent constant sources as existing structural graph inputs plus a sidecar fact. This obeys
  `CompiledGraphModel` closure without a new model operation kind. Derived `bindableInputs()` keeps
  fixed sources out of caller/run binding.
- Match ingress by exact Tensor object identity and require reachable provenance-free non-gradient
  leaves. No factory history or mutable storage becomes semantic evidence.
- Select only four exact family rows: BOOL logic, signed-integral binary arithmetic, signed-
  integral comparison, and same-typed signed-integral scalar ADD/SUB/MUL/MIN/MAX over a known
  splat. Exclude floating/BFLOAT16 evaluation, casts, other scalar operations, and every
  element-traversing operation.
- Preserve graph-output producers and every multi-output occurrence. The first avoids changing
  requested occurrence/publication identity; the second avoids partial producer semantics.
- Place Compiler 0003B after 0003A. Scalar attributes remain operation metadata and are never
  inserted into the graph-value sidecar; 0003A uses selected identity attributes for bypass,
  whereas 0003B combines a selected integral attribute with a known graph-value splat to compute a
  new splat.
- Keep graph-only DCE unchanged. Its sidecar overload prunes only dead constant inputs, preserving
  every bindable input and live/non-forward dependency.
- Use graph-size structural budgeting instead of an arbitrary numeric limit: one scalar per fact,
  at most one fact per graph input, one synthetic source per removed one-output node, one scan, and
  no fixed point or per-element work.
- Require Compiler 0005 to transport the exact immutable facts into compile artifacts while
  leaving physical materialization to prepare/backend.
- Expand the authorized path maximum from 18 to 19 solely for
  `CapturedGraphInferenceTest`: the new internal overload makes its existing untyped null call
  ambiguous, so the graph-only null assertion now casts to `CompiledGraphModel`. This is a source
  compatibility repair in the affected module test, not additional behavior or scope.

## Known limitations

- Only logical splats are representable. Dense values, nonuniform values, sparse values, tuples,
  and storage snapshots are unsupported.
- Floating and BFLOAT16 splats retain exact bits but participate in no fold in this task.
- Scalar operation attributes remain outside constant facts. Only integral ADD/SUB/MUL/MIN/MAX
  with a known same-typed splat input fold; floating, CLAMP, and other scalar operations remain.
- Graph-output producers and multi-output producers remain executable occurrences.
- Equal constants are not interned, and CSE does not treat distinct constant-source IDs as equal.
- Constant-control bypass, cast folding, views, indexing, reductions, and numerical functions are
  deferred.
- The sidecar is package-private compiler state. Compiler 0005 must select its public immutable
  artifact transport before prepare can materialize constants.
- No public compile entry exists, so current ingress is an internal foundation exercised through
  compiler tests and later compiler orchestration.

## Validation evidence

- Implementation-context Java evidence was reused as required by the documentation workflow:
  `./gradlew :modules:compiler:test` completed with `BUILD SUCCESSFUL`; compiler JUnit XML reported
  101 tests, 0 failures, 0 errors, and 0 skipped. That context also reported a passing
  `git diff --check`. Documentation context `/root/docs_compiler_0003b` changed no executable Java
  after that run, so it did not repeat the successful Java suite.
- Documentation context `/root/docs_compiler_0003b` applied the General, API/Javadoc, Planning,
  and Example profiles. It independently reviewed `AGENTS.md`, `ARCHITECTURE.md`, the current
  architecture index, documentation rules and profiles, planning guide, roadmap, compiler master
  plan, tasks 0001 through 0003A, this task, the final affected production source and tests,
  generated compiler Javadocs, Compile API, targeted Tensor API contracts, glossary, and the model
  graph/scalar/source-role boundaries referenced by those documents.
- `./gradlew :modules:compiler:javadoc` completed with `BUILD SUCCESSFUL`; generated output
  includes `CompileTimeConstantGraph`, its `Splat`, `Binding`, and `Ingress` records,
  `ForwardConstantFolding`, the explicit `GraphCapture` overload, the sidecar validation overload,
  `ValidatedGraph`, sidecar-aware DCE, and the final optimizer contract.
- `python3 /tmp/validate_synaptik_markdown.py` passed for 236 Markdown files, 4,305 local links,
  266 local anchors, 3,006 fence markers, final newlines, and trailing whitespace.
- `git diff --check` passed after the documentation edits. The combined sorted tracked/untracked
  path audit reported exactly the authorized 19 paths. The 19th path is only
  `CapturedGraphInferenceTest.java`: the new internal overload made the existing untyped null call
  ambiguous, and the test now casts null to `CompiledGraphModel` without changing the asserted
  graph-only behavior.
- Source and generated-Javadoc inspection confirmed that the new outer/nested records, helpers,
  overloads, and result remain package-private as planned; no public compiler facade or model
  declaration was added. Closed typed switches cover BOOL `NOT`/`AND`/`OR`, integral
  `ADD`/`SUB`/`MUL`/`MIN`/`MAX`, all six integral comparisons, and same-typed integral scalar
  `ADD`/`SUB`/`MUL`/`MIN`/`MAX`; `DIV`, `POW`, `CLAMP`, floating/BFLOAT16, casts, and every other
  family remain unevaluated.
- Source inspection confirmed exact optional order `0003A -> constant folding -> sidecar-aware
  DCE -> CSE -> sidecar-aware DCE`, once each, with identity-based changed-candidate revalidation.
  It also confirmed input-first deterministic rebuilds, synthetic sources in fold occurrence
  order, graph-only DCE preservation, bindable-input retention, and live-constant pruning.
- Negative-boundary inspection found no Tensor host-storage read, factory inference, dense
  payload, generic evaluator, reflection/string dispatch, physical allocation, fixed-point loop,
  autograd, public binding, downstream artifact/materialization implementation, or new operation
  kind. Existing iterative capture and DCE worklists are traversal mechanics, not constant-folding
  fixed points.
- Scope inspection found no change to `ARCHITECTURE.md`, focused architecture documents, ADRs,
  Gradle/build files, dependencies, architecture tests, backend conformance, integration tests, or
  another module's Java. Architecture remains unchanged because the existing contract already
  assigns constant folding and immutable graph transformations to compiler and physical
  materialization to later prepare/backend work. Tensor/model public contracts remain unchanged
  because the sidecar is internal and retains no Tensor. Public Compile/Training Java contracts,
  capability contracts, build configuration, and unrelated modules require no changes for the
  same reason.
- Status inspection confirmed compiler tasks 0001, 0002, 0003, 0003A, and 0003B are `Complete`;
  0004 and 0005 remain `Draft` master-plan rows with no detailed task specifications, and no
  compiler task is `Ready`.

## Implementation notes

- Added compiler-owned immutable logical-splat facts, explicit identity-based leaf ingress, and a
  derived non-bindable fixed-source classification without retaining Tensor or storage state.
- Added one closed one-scan fold implementation for the selected BOOL and signed-integral matrix.
  Folded values become deterministic synthetic structural inputs; no payload expands with Shape.
- Carried source roles through validation, canonicalization, exact rewriting, CSE, and DCE.
  Sidecar-aware DCE removes dead fixed sources while graph-only DCE continues to retain every graph
  input.
- Integrated the exact enabled order after Compiler 0003A and preserved changed-only Compiler 0002
  revalidation and disabled-optimization canonicalization/validation.
- Finalized package-private Javadocs, Compile API, targeted Tensor API wording, glossary terms,
  compiler master-plan state, roadmap state, and this task's decisions/evidence. No executable Java
  was changed during the documentation-focused pass.

## Completion summary

- Completed changes: implemented explicit immutable logical splats, identity-only capture ingress,
  fixed/bindable source classification, the exact selected constant-fold matrix, deterministic
  synthetic sources, sidecar-aware constant pruning, pipeline integration, and revalidation.
- Files changed or created: exactly the authorized 19 paths—seven compiler production paths, six
  compiler test paths, Compile API, Tensor API, glossary, this task, compiler master plan, and
  roadmap.
- Tests and validation: reused the implementation context's successful compiler suite (101 tests,
  0 failures/errors/skips); documentation context passed compiler Javadoc, repository Markdown
  links/anchors/fences/newline/whitespace validation, exact scope/status/boundary checks, and final
  `git diff --check`.
- Documentation-agent review: completed by clean context `/root/docs_compiler_0003b` using the
  General, API/Javadoc, Planning, and Example profiles.
- Documentation impact: Compile API, targeted Tensor API status text, glossary, task, master plan,
  and roadmap now describe the implemented internal boundary and preserve planned public and
  physical-lifecycle distinctions.
- Javadoc review: all affected package-private production contracts and generated pages were
  reviewed; the implementation-authored Javadocs were accurate and required no Java edit in this
  pass.
- Glossary impact: added reusable definitions for bindable input, compile-time constant source,
  and logical splat; updated forward-optimization status and boundaries.
- Architecture impact: None. No architecture rule, dependency, module boundary, or build contract
  changed.
- Unresolved issues: None.
- Follow-up required: None for task 0003B. Compiler 0005 remains responsible for transporting the
  immutable facts in compile artifacts and excluding fixed sources from runtime binding; future
  prepare/backend work remains responsible for physical splat materialization. These are planned
  downstream obligations, not incomplete 0003B work.

Status: Complete
