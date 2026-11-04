# [General Scan](../../glossary.md#general-differentiable-scan) source construction

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

## The body builder is not the loop executor

Application or NN composition code constructs a `ScanBody` from placeholder Tensors and expression
roots, then calls `Scan.apply` with actual initial state, time sequences and invariant inputs.
The placeholders are leaves because no expression produces them; they intentionally contain no
stored data. Body and apply calls snapshot role lists but retain exact Tensor references,
not copies of their values. Compiler later captures this description; a qualified backend executes
the repeated body. Compare [fixed scans](fixed-scans.md) when the transition is already a built-in
RNN/GRU/LSTM formula, and [compiled attachments](compiled-graph.md#graph-values-and-compiled-nodes)
for the separately scoped representation after capture.

**[Reader model](../../glossary.md#term-reader-model).** General Scan describes a repeated calculation with a fixed number of steps.
The **body** is the expression template for one step. **[Carry](../../glossary.md#term-carry)** is state passed to the next
step; a **sequence** supplies the current time [slice](../../glossary.md#slice); an **[invariant](../../glossary.md#term-invariant)** supplies the same input
at every step. An **emission** is a requested per-step output, stacked along a time [axis](../../glossary.md#term-axis).
A **formal** is a declared placeholder; an **actual** is the [Tensor](../../glossary.md#tensor) supplied for that position.
Declaring these roles does not execute a loop.

A **[residual](../../glossary.md#scan-residual-port-and-history-demand)** names an internal value that can be retained for later use, such as computing
[gradients](../../glossary.md#term-gradient). **Demand** describes which histories a downstream consumer actually needs, rather
than requiring all internal values to be stored. ROW(axis) identifies a proved independent
[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)-row [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix); SHARED has no such per-row coordinate. Equal axis sizes alone do not prove
either role. The detailed source, [compiler](../../glossary.md#term-compiler), and [backend](../../glossary.md#backend) restrictions below remain separate.

Read the cycle as the meaning of a step, not a [callback](../../glossary.md#term-callback) [API](../../glossary.md#term-application-programming-interface-api) or a [runtime](../../glossary.md#term-runtime) schedule.

```mermaid
flowchart LR
    accTitle: Scan distinguishes changing state from per-step and unchanged inputs
    accDescr: One body takes previous carry, the current sequence slice, and invariants. It produces next carry for the next step and emissions for the stacked result.
    C["Previous carry"] --> B["One step body"]
    S["Current sequence slice"] --> B
    I["Invariant inputs"] --> B
    B --> N["Next carry"]
    N --> C
    B --> E["Emission at this time position"]
```

<a id="general-scan-source-construction"></a>
### General-[scan](../../glossary.md#term-cumulative-scan) source construction

`ScanBody`, `Scan` and `ScanResult` in `model.tensor` describe a fixed-count recurrence over
ordinary [Tensor](../../glossary.md#tensor) expressions. `ScanKind.SCAN` and [scalar](../../glossary.md#term-scalar)-only `ScanAttrs` in `model.operation.scan`
describe its meaning; `CompiledScanBody` in `model.graph` is the separately scoped compiled
attachment. These [Model](../../glossary.md#term-model) types provide source/intermediate-representation ([IR](../../glossary.md#term-intermediate-representation-ir)) contracts only;
they do not perform [capture](../../glossary.md#term-capture-meanings), [automatic differentiation](../../glossary.md#autograd) ([AD](../../glossary.md#autograd)), demand analysis or execution.
[Compiler](../../glossary.md#term-compiler) now owns scoped capture, first-order AD and demand emission; [Prepare](../../glossary.md#prepare) validates scoped
[bindings](../../glossary.md#term-binding) and demand. CPU0030 implements qualified [FLOAT32](../../glossary.md#term-float32) [pointwise](../../glossary.md#term-pointwise) execution, extended by CPU0031
with qualified broadcast, SUM/sumToShape and bounded [shape](../../glossary.md#shape)/[layout](../../glossary.md#layout) closure, including independently
qualified actual reverse bodies. CPU0032 adds static [homogeneous](../../glossary.md#term-homogeneous) FLOAT32 [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix) [MATMUL](../../glossary.md#matrix-multiplication--matmul) and
independently qualified actual Compiler reverse bodies. CPU0033 implements homogeneous [FLOAT64](../../glossary.md#term-float64)
across the complete bounded declared algebra and actual first-order reverse closure. These are
historical layers: current [general Scan](../../glossary.md#general-differentiable-scan) composes every ordinary [CPU](../../glossary.md#term-cpu)-supported [PURE](../../glossary.md#term-pure) unit with
the same ordinary [dtype](../../glossary.md#data-type--datatype)/layout/attribute guards and independently qualifies the complete forward
and selected ordinary-[VJP](../../glossary.md#term-vector-jacobian-product-vjp) reverse bodies. [ND](../../glossary.md#term-nd) cuts and [finite](../../glossary.md#term-finite)-path [derivative](../../glossary.md#term-derivative) guards remain exact.
All-valid WholeTensor differs from proved length-aware ROW([axis](../../glossary.md#term-axis)) and SHARED roles; SHARED
tensors may have arbitrary Shape under ordinary guards. Original length-AD carries/sequences
and emissions stay ROW(0); intermediate ROW axes follow the proved [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix). Row-mixing
requires materialized same-length support, not Tensor [identity](../../glossary.md#term-identity-meanings). Structural absence differs from a present
all-zero [seed](../../glossary.md#term-seed). CPU low types use the same ordinary/shared-AD [owners](../../glossary.md#term-owner). Cold/wire and [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [admission](../../glossary.md#term-admission) stay
separately bounded; no [Metal](../../glossary.md#term-metal) general Scan grant follows.
Fixed `RecurrentScan` APIs and routes are unchanged.

```text
declared Tensor formals + ordinary expression roots -> reusable ScanBody
actual carry / sequences / invariants + Scan.apply -> one producer and all canonical ports
independently supplied CompiledGraphModel + role/demand lists -> CompiledScanBody
```

Read the first two lines as source construction and the last as a separate structural IR
container. Model does not perform the capture between them. The
[scan semantic owner](../../architecture/contracts/recurrent-scan.md#general-differentiable-scan)<span data-glossary-navigation="true"> ([owner glossary](../../glossary.md#term-owner))</span>
defines the complete protocol; the [focused explanation](../../architecture/general-differentiable-scan.md)
distinguishes this foundation from downstream work.

<a id="formals-actual-positions-and-source-closure"></a>
#### Formals, actual positions and source closure

Construct `new ScanBody(carry, step, invariants, nextCarry, emissions)` from [immutable](../../glossary.md#term-immutable)-[identity](../../glossary.md#term-identity-meanings)
[Tensor](../../glossary.md#tensor) references. Every [argument](../../glossary.md#term-argument) list is defensively snapshotted. A formal is an explicitly
declared [leaf](../../glossary.md#term-leaf) standing for one input position, not a [graph](../../glossary.md#graph) [`ValueId`](../../glossary.md#logical-valueid-versus-physical-memory-slot). All formals are
identity-distinct across groups and storage/[provenance](../../glossary.md#provenance)-free. Fully static formals have [canonical](../../glossary.md#term-canonical-form)
[row-major](../../glossary.md#term-row-major) geometry. All-valid source also admits one [occurrence](../../glossary.md#term-occurrence) of the same named [batch dimension](../../glossary.md#batch-dimension)
per value, with unresolved numeric [layouts](../../glossary.md#layout). At least one step formal is required. [Carry](../../glossary.md#term-carry) and
next-carry lists may both be empty; at least one next carry or emission must remain a semantic
root. Unused declarations are allowed. Next-carry roots match carry [arity](../../glossary.md#term-arity), [shape](../../glossary.md#shape) and [dtype](../../glossary.md#data-type--datatype).
Emissions may repeat any valid body value, including a next carry.

The all-valid `Scan.apply(initialCarry, sequences, invariants, body, reverse)` binds each role by position.
Actual inputs have the exact static or named-[symbol](../../glossary.md#term-symbol) shape of their formals and are disjoint from the declared formal identities. Actual identities
may repeat within or across roles: input order is all carries, then sequences, then invariants,
without deduplication. Carry/[invariant](../../glossary.md#term-invariant) shape, dtype and `requiresGrad` match their formals.
Each sequence has [rank](../../glossary.md#term-rank) at least one and the same static [axis](../../glossary.md#term-axis)-zero [extent](../../glossary.md#term-extent) `T >= 0`; removing that
axis gives its step formal's shape, dtype and eligibility. Static step geometry is canonical, not a
[borrowed](../../glossary.md#ownership-borrowing-and-lease) sequence-[stride](../../glossary.md#term-stride) [view](../../glossary.md#term-view-meanings). Construction does not read actual input storage.

Only declared formals may be body leaves. Pass a [weight](../../glossary.md#term-weight) or Tensor [constant](../../glossary.md#term-constant) explicitly as an
invariant rather than capturing an undeclared outer leaf. [Scalar](../../glossary.md#term-scalar) [operation](../../glossary.md#operation) attributes remain
valid ordinary algebra. [Validation](../../glossary.md#term-validation) covers the whole contained [producer](../../glossary.md#term-producer), including unused
canonical siblings, using existing known pure [Model](../../glossary.md#term-model) semantics and their ordinary guards.
Valid [FLOAT16](../../glossary.md#term-float16) [TANH](../../glossary.md#term-tanh) and [FLOAT32](../../glossary.md#term-float32) RELU bodies can construct independently of the initial [CPU](../../glossary.md#term-cpu) domain;
this is not an [AD](../../glossary.md#autograd) or provider support answer. Unknown/custom semantics, effects, random-number
operations, mutation, [host](../../glossary.md#term-host)-backed body values and nested scans reject. A template and its producers
may be reused across occurrences. Collective outer escape/cross-template sharing checks and
scoped [capture](../../glossary.md#term-capture-meanings) remain [Compiler](../../glossary.md#term-compiler) work, with no Model [ownership](../../glossary.md#term-ownership-meanings) [registry](../../glossary.md#term-registry).

<a id="descriptor-only-ordinary-checks"></a>
#### [Descriptor](../../glossary.md#term-descriptor)-only ordinary checks

`ScanBody.describeOrdinary(operation, inputs, outputCount)` checks an already [canonical](../../glossary.md#term-canonical-form) ordinary
[operation](../../glossary.md#operation) using ordered `TensorDescriptor` inputs. It returns an [immutable](../../glossary.md#term-immutable) list of every
canonical output descriptor, including unused [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[normalization](../../glossary.md#term-normalization), attention and Top-K siblings.
It uses the same family-owned guards and descriptor construction as real [Tensor](../../glossary.md#tensor) expressions,
without creating a Tensor, [producer](../../glossary.md#term-producer), storage or identifier. Invalid [signatures](../../glossary.md#term-signature), [shapes](../../glossary.md#shape), types
and noncanonical operations reject before any Tensor [identity](../../glossary.md#term-identity-meanings) allocation.

The scoped overload `ScanBody.describeOrdinary(operation, inputs, outputCount, batch)` additionally
accepts `Optional<DynamicDimension>` naming the exact sole declared batch [symbol](../../glossary.md#term-symbol), or empty for a
static template. It reuses the same ordinary family guards and factored source/deferred symbolic
proof; the three-[argument](../../glossary.md#term-argument) ordinary [API](../../glossary.md#term-application-programming-interface-api) retains its deferred-[inference](../../glossary.md#term-inference) semantics. The scoped
check validates every operand, canonical output and shape-bearing attribute against that
declared vocabulary, including unused siblings and zero-count shapes. It cannot authorize a
symbol merely because the local output mentions it.

All-valid reshape uses canonical [element](../../glossary.md#term-element)-count proof across the declared template, so
`[2, B] -> [B, 2]` is valid without inventing a ROW [axis](../../glossary.md#term-axis). A length-aware ROW reshape additionally
requires its independent [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix)-mapping proof. This descriptor check grants neither ROW
[provenance](../../glossary.md#provenance) nor differentiation, [backend](../../glossary.md#backend) [admission](../../glossary.md#term-admission) or [runtime](../../glossary.md#term-runtime) execution.

This is [metadata](../../glossary.md#term-metadata) checking, not [capture](../../glossary.md#term-capture-meanings), differentiation or execution. Unknown/custom operations,
randomness, [dropout](../../glossary.md#dropout--inverted-dropout), fixed recurrent scans and nested general scans remain excluded. A successful
ordinary check establishes neither a length-aware row role nor backend support.

<a id="computed-logical-layout-completion"></a>
#### Computed logical-[layout](../../glossary.md#layout) completion

`ScanBody.closeOrdinaryLayouts(operation, inputs, outputs)` completes eligible unresolved
fully static computed result layouts after the caller independently validates [canonical](../../glossary.md#term-canonical-form)
semantics. Supply current topologically resolved operands and every canonical output [descriptor](../../glossary.md#term-descriptor).
Completion preserves data types, [Shapes](../../glossary.md#shape), [gradient](../../glossary.md#term-gradient) flags, output order and already resolved
layouts. It reuses exact ordinary [views](../../glossary.md#term-view-meanings), the existing positive-static crop geometry and the
existing noncontiguous-RESHAPE [materialization](../../glossary.md#materialization) decision. Negative-step, empty, named and
otherwise unproved extraction layouts remain unresolved; unknown layouts are not automatically
made [contiguous](../../glossary.md#term-contiguous).

`ScanBody.hasCanonicalContiguousResults(operation)` exposes the separate geometry-independent
generic materialization fact used by flat [Compiler](../../glossary.md#term-compiler) closure and [symbolic batch binding](../../glossary.md#symbolic-batch-binding). Its
existing generic family set is unchanged; specialized static [convolution](../../glossary.md#term-channels-first-convolution-layers) completion does not
widen that classifier or symbolic eligibility. A positive fact does not establish [PURE](../../glossary.md#term-pure) [Scan](../../glossary.md#term-cumulative-scan)
membership, valid descriptors, [backend](../../glossary.md#backend) support or execution.

These are shared cold [metadata](../../glossary.md#term-metadata) operations, not another constructor, type/Shape [inference](../../glossary.md#term-inference)
[engine](../../glossary.md#term-engine) or allocator. Ordinary [Tensor](../../glossary.md#tensor) construction and both `describeOrdinary` forms retain
their existing behavior. [CPU](../../glossary.md#term-cpu) Scan preparation overlays only computed projection descriptors
and matching memory requirements, leaving actual/formal guards and the captured body/activity
[graph](../../glossary.md#graph) associations unchanged.

<a id="per-output-ordinary-coordinate-facts"></a>
#### Per-output ordinary [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) facts

`Scan.declaredSemantics(descriptor, rowAxis, batch)` checks an explicit SHARED or ROW([axis](../../glossary.md#term-axis))
declaration without a [Tensor](../../glossary.md#tensor) or identifier. `Scan.ordinarySemantics(operation, inputs,
inputSemantics, outputs, batch)` derives an [immutable](../../glossary.md#term-immutable) `List<Scan.ValueSemantics>` in [canonical](../../glossary.md#term-canonical-form)
output order after the caller independently checks the ordinary [descriptors](../../glossary.md#term-descriptor).
`ValueSemantics` contains an optional identified row axis, terminal active-aggregation taint,
and immutable increasing direct value/control input positions. It is [metadata](../../glossary.md#term-metadata), not a persisted
[certificate](../../glossary.md#formal-proof-certificate-and-trusted-assumptions), numerical value or [derivative](../../glossary.md#term-derivative) eligibility.

All known entirely SHARED [PURE](../../glossary.md#term-pure) families use the same ordinary semantics; no separate row-kind
whitelist is needed. ROW intermediates may have any completely proved axis/[rank](../../glossary.md#term-rank). Canonical
[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[normalization](../../glossary.md#term-normalization) outputs have separate routes; attention [weights](../../glossary.md#term-weight) exclude V numerically even
when V supplies a broadcast batch domain. Exact coordinate mappings, not equal extents or B=0/1,
justify reshape/window/indexing rules. A changed reshape may select the lowest of several fully
proved boundary witnesses; exact-[shape](../../glossary.md#shape) reshape keeps its original axis.

This source/forward-coordinate [qualification](../../glossary.md#term-qualification) does not widen the independent typed [AD](../../glossary.md#autograd), sequence
header, [Prepare](../../glossary.md#prepare)/[CPU](../../glossary.md#term-cpu), cold or wire domains. In particular retained sequence declarations allow
only axes 0/1 and [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix) ROW(1), even though intermediate facts can represent more. AD rejects
unrepresentable retained histories before derivative [identities](../../glossary.md#term-identity-meanings). See the
[coordinate contract](../../architecture/contracts/recurrent-scan.md#bounded-row-role-analysis).


<a id="static-and-symbolic-construction-through-the-same-api"></a>
#### Static and symbolic construction through the same [API](../../glossary.md#term-application-programming-interface-api)

Model0042 adds construction-only symbolic source, independently reviewed and MAIN accepted 2026-10-25.
The named [dimension](../../glossary.md#dimension) is compared by [canonical](../../glossary.md#term-canonical-form) name, not object [identity](../../glossary.md#term-reference-identity) or a hard-coded `B` spelling.
All declarations, including unused ones, determine that one [symbol](../../glossary.md#term-symbol). Independent symbols,
anonymous/derived dimensions, multiple [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) [axes](../../glossary.md#term-axis) in one value and output-only symbols reject.
Reshape must preserve the fixed [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) products on each side of the named axis; for example,
`[B,6] → [B,2,3]` is proved, but `[B,6] → [B,3]` and `[B,6] → [6,B]` reject.
Expand cannot equate B with a concrete non-[singleton](../../glossary.md#term-singleton) batch size; [MATMUL](../../glossary.md#matrix-multiplication--matmul) [contraction](../../glossary.md#contraction-dimension) dimensions
must match exactly. Source proof also covers [shape](../../glossary.md#shape)-bearing attributes and deferred family
equalities/broadcasts: attention cannot specialize B through a fixed non-singleton batch or [mask](../../glossary.md#term-boolean-mask);
[normalization](../../glossary.md#term-normalization)'s declared shape must exactly match the input suffix; [convolution](../../glossary.md#term-channels-first-convolution-layers)/BatchNorm channels
and [loss](../../glossary.md#loss) [targets](../../glossary.md#term-target) must satisfy their exact named-coordinate relations. For example, a hidden
`Other` in `layerNorm([Other,6], epsilon)` is not admitted merely because the result stays `[B,6]`.
Ordinary family guards still own [operation](../../glossary.md#operation) semantics, [dtypes](../../glossary.md#data-type--datatype) and numerical-domain obligations;
source-shape proof does not inspect [runtime](../../glossary.md#term-runtime) index values or prove every reduction count.

Goal: construct the same fixed-ten-step recurrence for static batch 32 and symbolic batch `Rows`.
The complete program uses only storage-free [FLOAT32](../../glossary.md#term-float32) [metadata](../../glossary.md#term-metadata), no [Engine](../../glossary.md#term-engine), values or [backend](../../glossary.md#backend):
[carry](../../glossary.md#term-carry) is `[batch,64]`, each step is `[batch,16]`, sequence is `[10,batch,16]`, and the shared
projection [weight](../../glossary.md#term-weight) is `[16,64]`.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.*;
import io.github.pho001.synaptik.model.tensor.*;
import java.util.List;
import java.util.Optional;

public final class StaticAndSymbolicScanSourceExample {
    private static Tensor leaf(Shape shape) {
        return TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, shape,
                shape.isFullyStatic() ? Optional.of(LayoutDescriptor.contiguous(shape))
                        : Optional.empty(), false));
    }

    private static ScanResult describe(Dimension batch) {
        Shape carryShape = Shape.ofDimensions(batch, new StaticDimension(64));
        Shape stepShape = Shape.ofDimensions(batch, new StaticDimension(16));
        Tensor h = leaf(carryShape), step = leaf(stepShape), weight = leaf(Shape.of(16, 64));
        // Declare placeholders first: only these leaves may be used inside this body.
        Tensor next = step.matmul(weight).add(h).tanh();
        ScanBody body = new ScanBody(List.of(h), List.of(step), List.of(weight),
                List.of(next), List.of(h, weight));
        Shape sequenceShape = Shape.ofDimensions(new StaticDimension(10), batch, new StaticDimension(16));
        // Actual values occupy the same roles without being the placeholder objects.
        return Scan.apply(List.of(leaf(carryShape)), List.of(leaf(sequenceShape)),
                List.of(leaf(Shape.of(16, 64))), body, false);
    }

    public static void main(String[] args) {
        for (Dimension batch : List.of(new StaticDimension(32), new DynamicDimension("Rows"))) {
            ScanResult result = describe(batch);
            System.out.println(result.finalCarry().getFirst().descriptor().shape());
            System.out.println(result.outputs().getFirst().descriptor().layout().isPresent());
            System.out.println(result.outputs().get(1).descriptor().shape());
        }
    }
}
```

The static form prints `Shape[32, 64]`, `true`, `Shape[10, 16, 64]`; the symbolic form prints
`Shape[Rows, 64]`, `false`, `Shape[10, 16, 64]`. MATMUL preserves the named row coordinate,
while the shared weight history stays `[10,16,64]` with a canonical static [layout](../../glossary.md#layout). No numeric
batch is bound, and neither form is silently converted into the other. Replacing the symbolic
actual carry with `[32,64]` fails exact source matching rather than [binding](../../glossary.md#term-binding) `Rows=32`.

This example constructs source only. [Compiler](../../glossary.md#term-compiler) separately implements symbolic [inference](../../glossary.md#term-inference),
FLOAT32/[FLOAT64](../../glossary.md#term-float64) first-order [AD](../../glossary.md#autograd) and fresh all-port metadata binding. [Prepare](../../glossary.md#prepare)/[CPU](../../glossary.md#term-cpu)/Engine implement
the independently qualified ordinary CPU [PURE](../../glossary.md#term-pure) composition and selected ordinary-[VJP](../../glossary.md#term-vector-jacobian-product-vjp) reverse closure,
with fixed time/[features](../../glossary.md#term-feature) and the same ordinary dtype/layout guards; see the
[public execution example](../../user-guide/running-models.md#symbolic-scan-with-changing-batch-size).
Final [qualification](../../glossary.md#term-qualification) is tracked in [Prepare0014](../../planning/modules/prepare/tasks/0014-symbolic-scan-cpu-vertical.md).
Symbolic cold [capture](../../glossary.md#term-capture-meanings)/reconstruction still rejects, even if B occurs only in an unused formal
and all public ports are static. Symbolic [persistence](../../glossary.md#term-persistence) and [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/training remain unsupported.
Model0044 also implements static-or-one-B length-aware source construction below; Compiler0015
adds independent inference, actual first-order AD and fresh all-port binding for that combination.
Prepare0015 separately implements bounded combined FLOAT32/FLOAT64 length-aware execution below.
Engine geometry [memoization](../../glossary.md#term-memoization) remains future work.

<a id="canonical-results-residual-metadata-and-eligibility"></a>
#### [Canonical](../../glossary.md#term-canonical-form) results, [residual](../../glossary.md#scan-residual-port-and-history-demand) [metadata](../../glossary.md#term-metadata) and eligibility

One [occurrence](../../glossary.md#term-occurrence) owns every distinct canonical wrapper before returning. `ScanResult` exposes
[immutable](../../glossary.md#term-immutable) lists in [producer](../../glossary.md#term-producer) order: `finalCarry()`, `outputs()` (stacked emissions), then
`residuals()`. Repeated body-role [bindings](../../glossary.md#term-binding) do not merge outer wrappers. Final-[carry](../../glossary.md#term-carry) [shapes](../../glossary.md#shape) match
their carry roles; each emission/residual prepends `T` to the body value's shape. Outer [layouts](../../glossary.md#layout)
are canonical for static shapes and absent for symbolic shapes; wrappers are unlabeled and storage-free. Standalone `new ScanResult(...)` only
snapshots non-null groups/[elements](../../glossary.md#term-element): it permits empty groups and repeated references and does not
prove common [provenance](../../glossary.md#provenance). Record equality includes ordered group contents.

`body.retainedValues()` is the deterministic [identity](../../glossary.md#term-identity-meanings)-unique residual inventory: visit next-carry
roots then emissions, use producer postorder and ascending canonical sibling slots, include every
carry formal (also unused ones), and exclude step/[invariant](../../glossary.md#term-invariant) leaves. A previous-carry residual
denotes state before its step. Emissions and residuals describe original-time indices, even for
reverse traversal. These ports allocate no [history storage](../../glossary.md#scan-residual-port-and-history-demand) and choose no [derivative](../../glossary.md#term-derivative) recipe.
[Compiler](../../glossary.md#term-compiler)/[Planning](../../glossary.md#term-planning)/[Prepare](../../glossary.md#prepare) propagate demand, and the qualified [backend](../../glossary.md#backend) declares and allocates
the required histories through the existing [run](../../glossary.md#term-run) [lifecycle](../../glossary.md#lifecycle). Merely using a value internally is
not a [history demand](../../glossary.md#scan-residual-port-and-history-demand); [Model](../../glossary.md#term-model) construction itself allocates none.

Let `G` be the OR of `requiresGrad` over all actual carry, sequence and invariant positions,
including unused inputs. Every [differentiable](../../glossary.md#derivative-and-chain-rule) outer port in all three groups receives `G`;
nondifferentiable [dtypes](../../glossary.md#data-type--datatype) receive false. Formal and ordinary body-value flags are not rewritten.
This conservative flag is neither a nonzero-[dependency](../../glossary.md#term-dependency) promise nor an available derivative rule.
For `T=0`, construction still reserves distinct ports with the same flags: emission/residual
shapes have a zero time [axis](../../glossary.md#term-axis), and final carry semantically preserves initial values/[raw bits](../../glossary.md#storage-mapping-contiguity-and-injectivity),
not the input wrapper identity. No body executes during construction for any `T`.

Whole-body [validation](../../glossary.md#term-validation) creates no [Tensor](../../glossary.md#tensor) identities. Binding, header, [descriptor](../../glossary.md#term-descriptor) and checked
shape/count failures precede outer output-ID allocation, including direct attachment seams.
[Mutable](../../glossary.md#term-mutable) [host](../../glossary.md#term-host)-storage associations on formals and all retained values are rechecked when binding
each occurrence. This is a point-in-time check, not a freeze of Tensor storage or a concurrent
multi-Tensor [transaction](../../glossary.md#term-transaction). Later ID exhaustion consumes any earlier output IDs but exposes no
partial occurrence. Internal descriptor-check callbacks run synchronously, are not retained and
only validate metadata; they are not user bodies, [runtime](../../glossary.md#term-runtime) callbacks or an execution abstraction.

<a id="length-aware-source-only-construction"></a>
<a id="length-aware-source-construction-and-compiler-support"></a>

<a id="length-aware-source-construction-and-qualified-execution"></a>
#### Length-aware source construction and qualified execution

The additional implemented overload is:

```java
ScanResult Scan.apply(List<Tensor> initialCarry, List<Tensor> sequences,
        List<Tensor> invariants, Tensor validLengths, int rowCarryCount,
        List<OptionalInt> sequenceRowAxes, ScanBody body, boolean reverse);
```

The overload constructs source [metadata](../../glossary.md#term-metadata) only. Compiler0013 separately implements independent
[inference](../../glossary.md#term-inference), bounded first-order reverse construction and complete role/demand query emission.
Original-source [automatic differentiation](../../glossary.md#autograd) ([AD](../../glossary.md#autograd)) requires all carries and **every original sequence**
to be ROW(0), even unused/non-[gradient](../../glossary.md#term-gradient) positions. Generated reverse histories/[seeds](../../glossary.md#commit-release-distribution-log-and-seed-meanings) may instead
mix ROW(0), [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix) ROW(1) and SHARED roles derived from checked [bindings](../../glossary.md#term-binding)/[dataflow](../../glossary.md#term-dataflow).
Prepare0011 separately projects these static bindings, explicit roles and complete-[graph](../../glossary.md#graph) demand
through the existing resource [handoff](../../glossary.md#term-handoff), without reading lengths or re-proving body algebra.
CPU0042 separately qualifies fixed T/B/[features](../../glossary.md#term-feature) [FLOAT32](../../glossary.md#term-float32)/[FLOAT64](../../glossary.md#term-float64) forward and actual first-order
reverse through the existing [Engine](../../glossary.md#term-engine)/[Prepare](../../glossary.md#prepare)/generated [CPU](../../glossary.md#term-cpu) path. It validates every produced
length in `[0,T]` before any [scan](../../glossary.md#term-cumulative-scan)-local write, then skips inactive loads and arithmetic. See the
[CPU execution scope and example](../../backend-guide/cpu-backend.md#length-aware-fixed-batch-execution).
Static floating length mode has bounded [cold reconstruction](../../glossary.md#term-cold-reconstruction), including per-role floating
intermediates and explicit casts, and selective graph [persistence](../../glossary.md#term-persistence) ([FP8](../../glossary.md#term-fp8) v7, otherwise v5/v6 as
needed). [The cold contract](../../architecture/contracts/compiler-autograd.md#bounded-general-scan-cold-reconstruction) remains narrower
than [Model](../../glossary.md#term-model) source expressibility. Prepare0015 implements the bounded symbolic-length CPU/Engine path
below; symbolic cold/wire remains unavailable. [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)'s qualified static [runtime](../../glossary.md#term-runtime)-length [RNN](../../glossary.md#term-recurrent-neural-network-rnn) workflow
supports owned training, save/load and fresh training-session reuse.
The old overload keeps its broader known-[PURE operation](../../glossary.md#admission-preflight-and-fail-closed-validation)/type inventory and all-valid behavior.
The user-approved [Model0044 source foundation](../../planning/modules/model/tasks/0044-symbolic-length-scan-source-foundation.md)
extends this same overload and direct attachment to fully static [shapes](../../glossary.md#shape) **or** one named [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) B,
with fixed T/features. Source construction is Complete under the authority accepted on 2026-10-26;
independent source/docs review and MAIN's root test/[Javadoc](../../glossary.md#term-javadoc) [checkpoint](../../glossary.md#term-checkpoint) passed, with MAIN acceptance on 2026-10-26.
Compiler0015 implements independent symbolic-length FLOAT32/FLOAT64 inference, actual first-order
sibling AD and complete fresh scoped all-port binding; the [Compiler](../../glossary.md#term-compiler)-only task is Complete.
[Compile](../../glossary.md#compile)/bind reads [descriptors](../../glossary.md#term-descriptor) only, never input or length values. Prepare0015 separately implements
static-or-one-B projection and [homogeneous](../../glossary.md#term-homogeneous) FLOAT32/FLOAT64 generated [scalar](../../glossary.md#term-scalar), single-range forward
and actual Compiler reverse through the ordinary Engine [lifecycle](../../glossary.md#lifecycle). Its necessary typed outer/control/
ROW(1) closure uses explicit [occurrence](../../glossary.md#term-occurrence) predicates, not an AD-origin tag. Owning [validation](../../glossary.md#term-validation) and
independent code/structural review passed; root validation and the sole bounded probe completed;
[Prepare0015 Result](../../planning/modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result) records
evidence and limits; final MAIN acceptance remains pending. See the
[runnable changing-B example](../../user-guide/running-models.md#symbolic-length-scan-with-changing-batch-size)
and [bounded CPU scope](../../backend-guide/cpu-backend.md#symbolic-length-scan-cpu-execution).
The named [Dimension](../../glossary.md#dimension) must occur only at each identified row [axis](../../glossary.md#term-axis) and in [INT64](../../glossary.md#data-type--datatype) `[B]` lengths;
SHARED shapes/features stay static, including unused declarations. Equal sizes never prove a role.
Static formals/ports keep [canonical](../../glossary.md#term-canonical-form) [layouts](../../glossary.md#layout); symbolic formals/actuals/ports stay unresolved.
For example, h/step `[Rows,2]`, sequence `[0,Rows,2]`, fixed [bias](../../glossary.md#term-bias) `[2]` and INT64 lengths `[Rows]`
describe final `[Rows,2]` and row-history `[0,Rows,2]` ports with unresolved layouts. A retained
shared value such as `bias.neg()` instead has a static canonical history `[0,2]`; the [invariant](../../glossary.md#term-invariant)
bias [leaf](../../glossary.md#term-leaf) itself gains no [residual port](../../glossary.md#scan-residual-port-and-history-demand). These source descriptors allocate no histories or execute
a body; downstream [qualification](../../glossary.md#term-qualification) is separate. A bias containing `Rows` or a different lengths [symbol](../../glossary.md#term-symbol) rejects.

At execution, bind fully static actual inputs/seeds in compiled order and preserve all fixed extents,
types and gradient flags. Direct non-gradient INT64 `[B]` lengths need no [producer](../../glossary.md#term-producer); the bounded
ordinary control producer is only non-gradient FLOAT32/FLOAT64 `[B]` `.cast(DataType.INT64)`.
Each call performs fresh complete binding and actual-sized resource [admission](../../glossary.md#term-admission), even for repeated B.
After producers, every length must lie in `[0,T]` before the Scan writes output, [padding](../../glossary.md#padding), history
or scratch; T=0 requires all zero lengths and `[B,0]` does not hide a nonzero B. B=0 permits empty
row outputs but does not skip other required outputs or checks. Inactive [carry](../../glossary.md#term-carry)/dInitial preserves
raw initial/final-[seed](../../glossary.md#term-seed) values; inactive emissions/dX are +0. Lengths do not change an outer [mean](../../glossary.md#term-mean)'s
dense denominator. The initial Prepare0015 layer excluded all-valid symbolic FLOAT64 and ordinary
symbolic FLOAT64 [MATMUL](../../glossary.md#matrix-multiplication--matmul); that historical exclusion does not limit current ordinary CPU Scan
composition under the same ordinary guards. Its native/mixed and [vector](../../glossary.md#term-vector)/parallel exclusions
record that layer's historical execution [scope](../../glossary.md#term-scope). CPU low parity uses these same [owners](../../glossary.md#term-owner);
higher-order and symbolic cold/copy/wire admission are not widened.

The first `rowCarryCount` carries identify a leading batch [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) B (ROW(0)); the source count
is in `[0, carryCount]`. B comes from the INT64 lengths descriptor, not the first carry.
Remaining carries are terminal SHARED additive accumulators. A length-aware source may have no
ROW carry, no ROW formal, or no carry at all. These newly representable Model0046 topologies
remain independently rejected by current Compiler/Prepare/CPU/cold/wire consumers. Each sequence
has one non-null `OptionalInt`: absent means SHARED, zero means ROW(0), and one means ROW(1) in
a rank-two step shape. Axes address the step after time is removed, not the outer sequence.
Roles are declarations, never guesses from equal extents, flags or [rank](../../glossary.md#term-rank). Tied actual [identities](../../glossary.md#term-identity-meanings)
must identify the same outer batch axis, or both be SHARED: step axis r maps to actual axis r+1.
This also applies to unused positions and B=0/1.

Model0045 source retains each value's own type and gradient eligibility under ordinary family
guards and the separately bounded length-source row proof. Descriptor checking does not bypass
that proof. CPU composition and selected ordinary-[VJP](../../glossary.md#term-vector-jacobian-product-vjp) AD remain independent admissions under
ordinary guards, narrower than source expressibility; consult the linked CPU and Compiler scopes.
Row carries/emissions remain ROW(0). SHARED broadcast operands omit the
row axis or have [extent](../../glossary.md#term-extent) one there; `[1,H]` SHARED bias may accompany ROW(0) `[1,H]` at B=1.
Equal aligned sumToShape extents preserve a row role, including `[1,H] → [1,H]` and
ROW(1) `[F,1] → [F,1]`. An explicit SUM over the row axis is terminal even at extent one.
Row-reduced dependencies cannot flow through shared algebra back into row feedback or emissions.
Each shared carry uses exactly its own [tensor](../../glossary.md#tensor) `old.add(contribution)` root; contribution shape
and [dtype](../../glossary.md#data-type--datatype) must match, without implicit [broadcasting](../../glossary.md#broadcasting) or dependence on any shared carry. Its
updated value cannot feed another contribution or row feedback. See the
[bounded row rules](../../architecture/contracts/recurrent-scan.md#bounded-row-role-analysis) for exact mappings.

Actual order is carry, sequence, invariant, then non-gradient INT64 `[B]` lengths. All sequences
share static T; declared row axes share B. Lengths has no body formal, contributes nothing to
gradient eligibility, and its values/storage are not read. Header/roles are [immutable](../../glossary.md#term-immutable) snapshots.
All bindings, body storage, siblings, roles and checked geometry are validated before output IDs,
including at T=0/B=0. Ports retain their canonical order and conservative flags. [Residual](../../glossary.md#scan-residual-port-and-history-demand) shapes
remain `[T] + localShape`; a SHARED `[F,H]` history is `[T,F,H]`, never `[T,B,F,H]`.

The reverse sibling retains the exact lengths actual and opposite traversal. Exact saved
[activation](../../glossary.md#term-activation)/previous-state primals are bound, not replayed. Each incoming [cotangent](../../glossary.md#cotangent) for a SHARED
value is normalized before accumulation, even when B=1; disconnected ROW cotangents and shaped
literal seeds have explicit sequence roles. This describes construction and query metadata,
not validation of runtime lengths or physical [history storage](../../glossary.md#scan-residual-port-and-history-demand) by this constructor. Execution
requires independent [backend](../../glossary.md#backend) qualification and value [preflight](../../glossary.md#term-preflight). See the
[Compiler boundary](../compiler/verification-scan.md#current-scoped-scan-capture-and-first-order-ad).

Goal: construct and inspect a zero-time, [singleton](../../glossary.md#term-singleton)-batch recurrence. Only canonical storage-free
metadata exists: FLOAT32 h/step/bias have shape `[1,2]`, sequence has shape `[0,1,2]`, and lengths
has INT64 shape `[1]`. No length values or backend are needed.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Scan;
import io.github.pho001.synaptik.model.tensor.ScanBody;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.List;
import java.util.Optional;
import java.util.OptionalInt;

public final class LengthScanMetadataExample {
    private static Tensor leaf(DataType type, boolean grad, long... dimensions) {
        Shape shape = Shape.of(dimensions);
        return TensorFactory.create(new TensorDescriptor(type, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), grad));
    }

    public static void main(String[] args) {
        Tensor h = leaf(DataType.FLOAT32, false, 1, 2);
        Tensor step = leaf(DataType.FLOAT32, true, 1, 2);
        Tensor bias = leaf(DataType.FLOAT32, false, 1, 2);
        Tensor next = h.sumToShape(Shape.of(1, 2)).add(step).add(bias);
        ScanBody body = new ScanBody(List.of(h), List.of(step), List.of(bias),
                List.of(next), List.of(h));
        Tensor initial = leaf(DataType.FLOAT32, false, 1, 2);
        Tensor sequence = leaf(DataType.FLOAT32, true, 0, 1, 2);
        Tensor actualBias = leaf(DataType.FLOAT32, false, 1, 2);
        Tensor lengths = leaf(DataType.INT64, false, 1);
        // Lengths is an outer control input, not a value the body closes over.
        var result = Scan.apply(List.of(initial), List.of(sequence), List.of(actualBias),
                lengths, 1, List.of(OptionalInt.of(0)), body, false);
        var producer = result.finalCarry().getFirst().provenance().orElseThrow().producer();
        System.out.println(producer.inputs().size());
        System.out.println(result.outputs().getFirst().descriptor().shape().equals(Shape.of(0, 1, 2)));
        System.out.println(result.outputs().getFirst().descriptor().requiresGrad());
    }
}
```

Output is `4`, `true`, `true`, on separate lines. Binding adds one lengths actual to the three
formal positions; no lengths formal is created. The [identity](../../glossary.md#term-identity-meanings) sumToShape preserves ROW(0), the
singleton shared bias broadcasts legally, and the unused time domain still propagates the
sequence's gradient flag to the emission. These are metadata checks, not numerical execution.
Replacing next by `h.sum(0, true)` rejects as row-reduced feedback even though B=1 leaves its
shape unchanged. Passing this source to compile/[run](../../glossary.md#term-run) or cold save/copy is unsupported.

<a id="complete-metadata-construction-example"></a>
#### Complete [metadata](../../glossary.md#term-metadata)-construction example

Goal: describe `next = h + step`, emit previous `h`, and inspect temporal eligibility and port
[ownership](../../glossary.md#term-ownership-meanings). The initial state is only storage-free metadata: [scalar](../../glossary.md#term-scalar) [FLOAT32](../../glossary.md#term-float32) [carry](../../glossary.md#term-carry) with a false
flag and FLOAT32 sequence [shape](../../glossary.md#shape) `[2]` with a true flag. No values are supplied or evaluated.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Scan;
import io.github.pho001.synaptik.model.tensor.ScanBody;
import io.github.pho001.synaptik.model.tensor.ScanResult;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.List;
import java.util.Optional;

public final class ScanMetadataExample {
    private static Tensor leaf(Shape shape, boolean grad) {
        return TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), grad));
    }

    public static void main(String[] args) {
        Tensor h = leaf(Shape.scalar(), false);
        Tensor step = leaf(Shape.scalar(), true);
        Tensor next = h.add(step);
        ScanBody body = new ScanBody(List.of(h), List.of(step), List.of(),
                List.of(next), List.of(h));
        Tensor initial = leaf(Shape.scalar(), false);
        Tensor sequence = leaf(Shape.of(2), true);
        // Bind separate actual leaves; formals describe body input positions, not those objects.
        ScanResult result = Scan.apply(List.of(initial), List.of(sequence), List.of(), body, false);

        System.out.println(result.outputs().getFirst().descriptor().shape().equals(Shape.of(2)));
        System.out.println(result.outputs().getFirst().descriptor().requiresGrad());
        System.out.println(result.residuals().size());
        var producer = result.finalCarry().getFirst().provenance().orElseThrow().producer();
        System.out.println(producer.output(1) == result.outputs().getFirst());
    }
}
```

The output is `true`, `true`, `2`, `true`, on separate lines. `h` and `step` declare distinct
positions; `next` is one ordinary ADD expression. [Binding](../../glossary.md#term-binding) separate actual leaves infers `T=2`.
The retained inventory is `[h, next]`, so the [occurrence](../../glossary.md#term-occurrence) owns four ports: final carry, emission
and two residuals. Although the source emission `h` has a false flag, its outer stacked port gets
true from `G`: later emissions may depend on earlier steps. The final [identity](../../glossary.md#term-identity-meanings) check proves
[canonical](../../glossary.md#term-canonical-form) ownership, not computation. These metadata outcomes match the focused [scan](../../glossary.md#term-cumulative-scan) tests;
this example needs no [backend](../../glossary.md#backend). Replacing the sequence with shape `[0]` keeps the flags and gives
empty stacked [descriptors](../../glossary.md#term-descriptor); binding `h` itself as the actual initial carry instead rejects because
actuals must be disjoint from formals.


---

[Previous topic](normalization.md) · [Next topic](statistics-losses.md)
