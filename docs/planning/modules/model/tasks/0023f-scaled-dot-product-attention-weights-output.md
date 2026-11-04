# Task 0023F: [Scaled Dot-Product Attention](../../../../glossary.md#scaled-dot-product-attention) [Weights](../../../../glossary.md#term-weight) Output

**Editorial reading key (current context, not a status update).**

This record concerns expression construction, not attention execution; its completion evidence and
compiler exclusions describe the historical task, not present capability. The caller invokes
`WithWeights` on the query Tensor, supplying keys, values, and optional mask/attributes to the model
helper. Descriptors and input references cross this API boundary, not computed attention values. A
producer identifies one operation occurrence; slots zero and one identify its output and normalized
weights. The shallow record retains exact Tensor references; these storage-free expressions neither
execute nor own result buffers. See [output slots and shared
occurrence](../../../../reference/tensor/attention.md#output-slots-and-shared-occurrence) for
provenance and eligibility distinctions.

Weights express relative contributions of eligible keys, not raw scores: an all-masked row has
positive-zero weights, not a uniform distribution. Model semantics fixes this meaning; execution
providers realize supported routes without redefining it. Semantic responsibility is not storage
ownership. The [current metadata
example](../../../../reference/tensor/attention.md#runnable-output-and-weights-metadata-example)
distinguishes descriptor inspection from execution and describes compiler differentiation beyond
this task’s historical scope.

## Status

Complete

## Goal

Add one explicit public scaled-dot-product-attention result form that returns the ordinary
attention output and the normalized attention [weights](../../../../glossary.md#term-weight) from the same semantic [occurrence](../../../../glossary.md#term-occurrence).

The completed [API](../../../../glossary.md#term-application-programming-interface-api) must preserve all four existing chainable
`scaledDotProductAttention(...)` methods and their one-output [producer](../../../../glossary.md#term-producer) behavior. It adds four
clearly named `scaledDotProductAttentionWithWeights(...)` overloads returning one shallowly
[immutable](../../../../glossary.md#term-immutable) `ScaledDotProductAttentionResult`. The [carrier](../../../../glossary.md#term-storage-carrier) exposes output slot zero and weights slot
one from one exact shared producer, so callers and later [compiler](../../../../glossary.md#term-compiler) work never need to recompute
weights and accidentally lose the [operation](../../../../glossary.md#operation)'s selected masking and floating-special-value
semantics.

## Rationale and [mental model](../../../../glossary.md#term-mental-model)

The existing [operation](../../../../glossary.md#operation) already defines both intermediate [weights](../../../../glossary.md#term-weight) and the final output:

```text
query  [..., L, E]
key    [..., S, E]  -> scores/weights [..., L, S]
value  [..., S, Ev] -> output         [..., L, Ev]

scores  = (query @ transpose(key)) * scale
weights = maskedSoftmax(scores, final axis S)
output  = weights @ value
```

Reconstructing `weights` as a second public expression is not equivalent. The current attention
contract gives all-masked and all-negative-[infinity](../../../../glossary.md#term-infinity) rows positive-zero weights, splits unit weight
across eligible positive-infinity ties, and excludes masked score/value special values before
arithmetic. Existing general [softmax](../../../../glossary.md#softmax--log-softmax) and [mask](../../../../glossary.md#term-boolean-mask) composition cannot reproduce every selected case.

The public [model](../../../../glossary.md#term-model) therefore becomes:

```text
one attention producer
  operation: SCALED_DOT_PRODUCT_ATTENTION(attrs)
  inputs:    [query, key, value] or [query, key, value, mask]
  outputs:
    slot 0: output  [..., L, Ev]
    slot 1: weights [..., L, S]

ScaledDotProductAttentionResult
  output  -> exact Tensor wrapper for slot 0
  weights -> exact Tensor wrapper for slot 1
```

The existing one-output methods remain the conventional fluent path:

```java
Tensor next = query.scaledDotProductAttention(key, value).add(residual);
```

Callers request the auxiliary result only when needed:

```java
ScaledDotProductAttentionResult attention =
        query.scaledDotProductAttentionWithWeights(key, value);
Tensor next = attention.output().add(residual);
Tensor weights = attention.weights();
```

These examples describe expression [metadata](../../../../glossary.md#term-metadata), not [eager](../../../../glossary.md#term-eager-evaluation) computation, [compiler](../../../../glossary.md#term-compiler) [capture](../../../../glossary.md#term-capture-meanings), [gradients](../../../../glossary.md#term-gradient),
[backend](../../../../glossary.md#backend) support, or execution.

## [Scope](../../../../glossary.md#term-scope)

- Preserve the existing `SCALED_DOT_PRODUCT_ATTENTION` kind and
  `ScaledDotProductAttentionAttrs` without adding a second semantic [identity](../../../../glossary.md#term-identity-meanings) or attributes type.
- Widen the kind's one exact stable signature to accept the existing three-to-four ordered inputs
  with an output-count range of one through two.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>
- Preserve all four existing public one-output `scaledDotProductAttention(...)` signatures,
  validation, metadata, numerical meaning, freshness, producer shape, and ID effects.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>
- Add public `ScaledDotProductAttentionResult(Tensor output, Tensor weights)` under the Tensor
  package.
- Add exactly four public receiver methods:

  ```java
  ScaledDotProductAttentionResult scaledDotProductAttentionWithWeights(
          Tensor key, Tensor value)
  ScaledDotProductAttentionResult scaledDotProductAttentionWithWeights(
          Tensor key, Tensor value, ScaledDotProductAttentionAttrs attrs)
  ScaledDotProductAttentionResult scaledDotProductAttentionWithWeights(
          Tensor key, Tensor value, Tensor mask)
  ScaledDotProductAttentionResult scaledDotProductAttentionWithWeights(
          Tensor key, Tensor value, Tensor mask, ScaledDotProductAttentionAttrs attrs)
  ```

- Reuse the existing package-private attention helper for all [validation](../../../../glossary.md#term-validation) and [Shape](../../../../glossary.md#shape) derivation.
- Construct the two public outputs with one call to existing
  `TensorFactory.createDerivedOutputs(...)` and one exact [producer](../../../../glossary.md#term-producer).
- Keep the output [descriptor](../../../../glossary.md#term-descriptor) exactly equal to the existing one-output descriptor contract.
- Give weights the exact score Shape, promoted attention data type, unresolved layout, and
  query/key gradient-request OR.<span data-glossary-navigation="true"> ([weights glossary](../../../../glossary.md#term-weight); [data type glossary](../../../../glossary.md#data-type--datatype); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Retain exact ordered inputs, exact attrs reference for attrs-bearing overloads, exact producer
  reference in both wrappers, and output indices zero then one.
- Update all thirteen global public-[Tensor](../../../../glossary.md#tensor) inventory locks from 196 to 200.
- Finalize affected Javadocs, Tensor and Compile API explanations, glossary terminology,
  capability status, task evidence, master plan, and roadmap through a separate clean-context
  documentation pass in the same overall change.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- changing, removing, deprecating, or secretly making multi-output any existing
  `scaledDotProductAttention(...)` method
- a new attention kind, a weights-only kind, a second attributes type, an output-selection
  attribute, hidden outputs, a nullable result, or an optional weights component
- recomputing [weights](../../../../glossary.md#term-weight) with [MATMUL](../../../../glossary.md#matrix-multiplication--matmul), [mask](../../../../glossary.md#term-boolean-mask), WHERE, [SOFTMAX](../../../../glossary.md#softmax--log-softmax), or another attention [occurrence](../../../../glossary.md#term-occurrence)
- returning scores, log weights, dropout masks, RNG state, caches, saved values, query/key/value
  projections, or another auxiliary output<span data-glossary-navigation="true"> ([log glossary](../../../../glossary.md#term-log); [RNG state glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- attention [dropout](../../../../glossary.md#dropout--inverted-dropout), training mode, hidden/global randomness, [`GraphRngState`](../../../../glossary.md#graph-rng-state--graphrngstate), grouped-query,
  sparse, packed, cached, additive-bias, offset-causal, or quantized attention<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias))</span>
- changing query/key/value promotion, rank checks, batch broadcasting, mask broadcasting,
  explicit-scale matching, causal eligibility, Shape references, or validation messages/order<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [broadcasting glossary](../../../../glossary.md#broadcasting); [Shape glossary](../../../../glossary.md#shape); [validation glossary](../../../../glossary.md#term-validation))</span>
- changing all-masked, NaN, infinity, signed-zero, accumulation, empty-axis, stability, or
  determinism semantics selected by task 0019E<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [axis glossary](../../../../glossary.md#term-axis))</span>
- an output-only [`Tensor`](../../../../glossary.md#tensor) convenience, result-carrier chaining facade, destructuring facility,
  builder, callback, consumer, tuple hierarchy, generic multi-output API, or producer lookup API<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [facade glossary](../../../../glossary.md#term-facade); [callback glossary](../../../../glossary.md#term-callback); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [producer glossary](../../../../glossary.md#term-producer); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- gradient rules, backward kinds, autograd traversal, compiler adoption, graph capture,
  canonicalization, decomposition, fusion, saved-value lifetime, or accumulation<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [compiler glossary](../../../../glossary.md#term-compiler); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [decomposition glossary](../../../../glossary.md#term-decomposition); [fusion glossary](../../../../glossary.md#term-fusion); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- value or storage inspection, attention execution, algorithms, materialization, kernels, backend
  support, prepare, runtime, engine, training, ONNX, conformance, or integration behavior<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [kernels glossary](../../../../glossary.md#kernel); [backend glossary](../../../../glossary.md#backend); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [engine glossary](../../../../glossary.md#term-engine); [ONNX glossary](../../../../glossary.md#term-onnx); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- changes to Operation, OperationKind, OperationAttrs, OperationSignature, TensorDescriptor,
  Shape, Dimension, DataType, ScalarValue, TensorFactory, TensorProducer, TensorProvenance, or
  storage foundations<span data-glossary-navigation="true"> ([Operation glossary](../../../../glossary.md#operation); [OperationKind glossary](../../../../glossary.md#operationkind); [OperationAttrs glossary](../../../../glossary.md#operationattrs); [Dimension glossary](../../../../glossary.md#dimension); [DataType glossary](../../../../glossary.md#data-type--datatype); [ScalarValue glossary](../../../../glossary.md#typed-scalar-value--scalarvalue); [TensorProducer glossary](../../../../glossary.md#producer--tensorproducer))</span>
- dependencies, Gradle/build changes, architecture/ADR/tests, another module, unrelated refactors,
  task 0024 implementation, or a detailed task-0024 specification<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [module glossary](../../../../glossary.md#term-module))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model master plan](../master-plan.md)
- [Task 0001](0001-data-type-model.md)
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0005](0005-operation-semantic-foundation.md)
- [Task 0006](0006-operation-model.md)
- [Task 0007](0007-tensor-descriptor-model.md)
- [Task 0013](0013-tensor-provenance-skeleton.md)
- [Task 0016I](0016i-softmax-semantic-kinds-and-attributes.md)
- [Task 0016J](0016j-softmax-tensor-expressions.md)
- [Task 0018K](0018k-operation-signature-and-construction-hardening.md)
- [Task 0018L](0018l-shared-multi-output-tensor-provenance.md)
- [Task 0018N](0018n-typed-scalar-value-contract.md)
- [Task 0018Q](0018q-masked-reduction-redesign.md)
- [Task 0019](0019-matmul-semantics-and-tensor-expression.md)
- [Task 0019E](0019e-scaled-dot-product-attention.md)
- [Task 0023](0023-adjoint-expressibility-audit.md)
- [Adjoint expressibility result](../adjoint-expressibility-audit.md)<span data-glossary-navigation="true"> ([Adjoint glossary](../../../../glossary.md#term-adjoint))</span>
- [Task 0023E](0023e-cumulative-scan-normalization-and-product.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `modules/model` owns backend-independent attention meaning, output descriptors, and public
  Tensor expression construction. It does not own gradients, compiler traversal, execution,
  kernels, backend support, or saved-value lifetime.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [descriptors glossary](../../../../glossary.md#term-descriptor); [gradients glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [kernels glossary](../../../../glossary.md#kernel); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- [`Tensor`](../../../../glossary.md#tensor) remains public [mutable](../../../../glossary.md#term-mutable) [API](../../../../glossary.md#term-application-programming-interface-api) state and is not an intermediate-representation [node](../../../../glossary.md#node).
- The normalized weights are a generally useful public result and an exact output of the same
  operation occurrence. They are not a compiler-private backward kind or a reconstructed sibling.<span data-glossary-navigation="true"> ([weights glossary](../../../../glossary.md#term-weight); [operation glossary](../../../../glossary.md#operation); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- The existing kind remains the single semantic identity because one-output and two-output calls
  request the same attention mathematics. One family-owned <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>`OperationSignature` accepts the exact
  input range and both output counts; producer descriptor count selects the occurrence shape.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [shape glossary](../../../../glossary.md#shape))</span>
- The kind still declares exactly one signature for the exact attributes class. This preserves
  the <span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>`OperationKind.signatureFor(...)` invariant that duplicate attribute-class variants are
  malformed.<span data-glossary-navigation="true"> ([invariant glossary](../../../../glossary.md#term-invariant))</span>
- Existing public methods create exactly one output descriptor and one wrapper. They do not
  allocate, retain, or hide a weights output.
- New methods create exactly two descriptors and two wrappers through one producer. Both wrappers
  retain that exact producer and expose output indices zero and one.
- The output descriptor is unchanged: promoted query/key/value type, exact output Shape,
  unresolved layout, and query/key/value <span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout))</span>`requiresGrad` OR.
- The weights descriptor uses the same promoted attention type, exact score Shape, unresolved
  layout, and query/key `requiresGrad` OR. Value and BOOL mask do not affect weights and therefore
  do not contribute to its eligibility metadata.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [mask glossary](../../../../glossary.md#term-boolean-mask); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Both outputs are unlabeled and storage-free. The carrier owns no state or behavior beyond exact
  wrapper references and ordinary record value semantics.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- The existing numerical contract applies to the exposed weights exactly: excluded entries are
  positive zero; no-eligible and all-eligible-negative-infinity rows are positive zero; eligible
  NaN makes eligible weights NaN; positive-infinity ties split unit weight equally; otherwise
  finite eligible weights use the selected stable-softmax meaning and total one ideally.<span data-glossary-navigation="true"> ([positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [infinity glossary](../../../../glossary.md#term-infinity); [NaN glossary](../../../../glossary.md#term-nan); [finite glossary](../../../../glossary.md#term-finite); [softmax glossary](../../../../glossary.md#softmax--log-softmax))</span>
- Local validation completes before output descriptors, producer, wrappers, or IDs are allocated.
  Existing failures retain their messages and consume no ID.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- A successful two-output call consumes two monotonically allocated Tensor IDs in slot order. If
  identifier exhaustion occurs after slot zero, that already allocated ID remains consumed and no
  partial result carrier is returned, matching `TensorFactory.createDerivedOutputs(...)`.
- Generic Operation, producer, provenance, and factory foundations remain unchanged. The existing
  multi-output contract is sufficient and must be reused directly.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- Runtime hot paths never consume Operation or producer metadata. No architecture boundary,
  dependency direction, lifecycle ownership, or backend-selection contract changes.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [lifecycle glossary](../../../../glossary.md#lifecycle); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Stop if implementation requires another semantic kind, another public method, a generic tuple,
  another production type, a foundation change, another module, or an architecture decision.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>

## Package impact

Existing packages used:

```text
io.github.pho001.synaptik.model.operation.attention
  Attention semantic identity, typed attributes, and occurrence signatures.

io.github.pho001.synaptik.model.tensor
  Public Tensor facade/result carrier and package-private attention expression construction.
```

Type placement:

- `io.github.pho001.synaptik.model.tensor.ScaledDotProductAttentionResult` — public carrier for the
  two exact Tensor wrappers returned by one attention occurrence.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [Tensor glossary](../../../../glossary.md#tensor); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- `io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionKind` — existing
  owner of the one signature accepting both valid output counts.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [signature glossary](../../../../glossary.md#term-signature))</span>
- `io.github.pho001.synaptik.model.tensor.TensorScaledDotProductAttentionExpressions` — existing
  package-private owner of validation, Shapes, descriptors, and factory delegation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Shapes glossary](../../../../glossary.md#shape); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- `io.github.pho001.synaptik.model.tensor.Tensor` — existing public receiver [facade](../../../../glossary.md#term-facade).

No package is added or moved.

## Required contracts

### [Operation signature](../../../../glossary.md#operation-signature)

`ScaledDotProductAttentionKind` still contains exactly one enum [constant](../../../../glossary.md#term-constant) and no new public member.
Its [immutable](../../../../glossary.md#term-immutable) [singleton](../../../../glossary.md#term-singleton) [signature](../../../../glossary.md#term-signature) list becomes exactly:

```java
List.of(new OperationSignature(
        ScaledDotProductAttentionAttrs.class,
        3,
        4,
        1,
        2))
```

The output range preserves every current one-output [occurrence](../../../../glossary.md#term-occurrence) and admits only the new
same-[operation](../../../../glossary.md#operation) two-output form. Zero or more than two outputs remain invalid. Input order remains
`[query, key, value]` or `[query, key, value, mask]` for both.

### Public result [carrier](../../../../glossary.md#term-storage-carrier)

Add exactly:

```java
public record ScaledDotProductAttentionResult(Tensor output, Tensor weights)
```

Its compact constructor null-checks `output` then [`weights`](../../../../glossary.md#term-weight) with `Objects.requireNonNull` and exact
messages `output` and `weights`. It retains both exact [immutable](../../../../glossary.md#term-immutable) wrapper references and adds no
factory, helper, [alias](../../../../glossary.md#aliasing-and-arena), list [view](../../../../glossary.md#term-view-meanings), [producer](../../../../glossary.md#term-producer) [lookup](../../../../glossary.md#term-lookup), [descriptor](../../../../glossary.md#term-descriptor) [validation](../../../../glossary.md#term-validation), or chaining method.

The record's [Javadoc](../../../../glossary.md#term-javadoc) explains slot order, [Shapes](../../../../glossary.md#shape), type, eligibility difference, exact shared
producer, and its lack of execution, storage, [gradient](../../../../glossary.md#term-gradient)-rule, or [lifecycle](../../../../glossary.md#lifecycle) [ownership](../../../../glossary.md#term-ownership-meanings).

### Public [Tensor](../../../../glossary.md#tensor) methods

Preserve the existing four `scaledDotProductAttention(...)` overloads exactly. Add exactly the
four `scaledDotProductAttentionWithWeights(...)` [signatures](../../../../glossary.md#term-signature) listed in [Scope](../../../../glossary.md#term-scope). Default forms create
the same fresh `ScaledDotProductAttentionAttrs(Optional.empty(), false)` values as their existing
one-output counterparts. Attrs-bearing forms retain the exact caller attrs reference.

Each method delegates exactly once to the matching package-private helper entry. There is no
public static form, [boolean](../../../../glossary.md#data-type--datatype) shortcut, primitive-scale form, nullable [mask](../../../../glossary.md#term-boolean-mask), `Optional<Tensor>`, or
overload returning [weights](../../../../glossary.md#term-weight) alone.

### Shared helper and construction

Retain the existing field-free final helper and private constructor. Preserve its two current
package-private `apply(...)` methods and add exactly two package-private `applyWithWeights(...)`
entries, one unmasked and one masked. The helper may refactor private implementation methods but
must not add another type or expose another package-private/public method.

All four construction families use the exact existing null, type, promotion, [rank](../../../../glossary.md#term-rank), [contraction](../../../../glossary.md#contraction-dimension),
[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters), [mask](../../../../glossary.md#term-boolean-mask), and scale [validation](../../../../glossary.md#term-validation) order and messages. Score and output [Shapes](../../../../glossary.md#shape) are derived once per
call from the same selected [batch Dimension](../../../../glossary.md#batch-dimension) references.

The one-output path remains:

```java
TensorFactory.createDerived(outputDescriptor, Optional.empty(), operation, inputs)
```

The two-output path delegates exactly once:

```java
List<Tensor> outputs = TensorFactory.createDerivedOutputs(
        operation,
        inputs,
        List.of(outputDescriptor, weightsDescriptor));
return new ScaledDotProductAttentionResult(outputs.get(0), outputs.get(1));
```

No second [Operation](../../../../glossary.md#operation) or [producer](../../../../glossary.md#term-producer) is constructed. No existing output is wrapped again.

### [Descriptor](../../../../glossary.md#term-descriptor) and [identity](../../../../glossary.md#term-identity-meanings) contract

For query `[..., L, E]`, key `[..., S, E]`, and value `[..., S, Ev]` after exact three-way [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)
broadcast:

| Slot | Result | [Data type](../../../../glossary.md#data-type--datatype) | [Shape](../../../../glossary.md#shape) | `requiresGrad` |
|---|---|---|---|---|
| 0 | output | promoted query/key/value type | `[..., L, Ev]` | query OR key OR value |
| 1 | [weights](../../../../glossary.md#term-weight) | promoted query/key/value type | `[..., L, S]` | query OR key |

Both [layouts](../../../../glossary.md#layout) are unresolved. The two outer Shape objects may differ, but they reuse the exact
selected batch [Dimensions](../../../../glossary.md#dimension) and exact query-sequence reference; weights additionally retains the
exact key-sequence reference, while output retains the exact value-[embedding](../../../../glossary.md#embedding) reference.

Both [Tensors](../../../../glossary.md#tensor) retain exact inputs and the same exact [`Operation`](../../../../glossary.md#operation), [`TensorProducer`](../../../../glossary.md#producer--tensorproducer), and attrs
references. Slot indices are exactly zero and one. Repeated valid calls create distinct [producers](../../../../glossary.md#term-producer),
wrappers, and IDs without changing any input or attrs.

## Affected files

Production — exactly four paths:

- modify `modules/model/src/main/java/io/github/pho001/synaptik/model/operation/attention/ScaledDotProductAttentionKind.java`
- add `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/ScaledDotProductAttentionResult.java`
- modify `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorScaledDotProductAttentionExpressions.java`
- modify `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java`

Tests — exactly sixteen paths:

- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/OperationSignatureTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/operation/attention/ScaledDotProductAttentionSemanticsTest.java`
- add `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/ScaledDotProductAttentionResultTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorScaledDotProductAttentionExpressionTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBatchNormInferenceExpressionTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorBinaryArithmeticTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorDenseCategoricalCrossEntropyWithLogitsExpressionTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorIndexCategoricalCrossEntropyWithLogitsExpressionTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLayerNormExpressionTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorLinearExpressionTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMatmulExpressionTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorMeanSquaredErrorExpressionTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorRmsNormExpressionTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorSlicePlacementExpressionTest.java`
- modify `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorSumToShapeExpressionTest.java`

Documentation and [planning](../../../../glossary.md#term-planning) — exactly seven paths:

- `docs/api/tensor-api.md`
- `docs/api/compile-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

Review without modification unless a contradiction requires stopping: `ARCHITECTURE.md`, focused
architecture documentation, Training and [Runtime](../../../../glossary.md#term-runtime) APIs, `ScaledDotProductAttentionAttrs`,
[Operation](../../../../glossary.md#operation) foundations, TensorFactory, [producer](../../../../glossary.md#term-producer)/[provenance](../../../../glossary.md#provenance) contracts, existing result carriers,
[Shape](../../../../glossary.md#shape)/[Dimension](../../../../glossary.md#dimension)/[DataType](../../../../glossary.md#data-type--datatype)/[ScalarValue](../../../../glossary.md#typed-scalar-value--scalarvalue) contracts, [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)/[softmax](../../../../glossary.md#softmax--log-softmax)/[mask](../../../../glossary.md#term-boolean-mask)/[dropout](../../../../glossary.md#dropout--inverted-dropout) contracts,
architecture tests, [Gradle](../../../../glossary.md#term-gradle), other [modules](../../../../glossary.md#term-module), [conformance](../../../../glossary.md#term-conformance), integration, and completed task history.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify exactly the 27 authorized paths above: four production paths,
sixteen test paths, and seven documentation/[planning](../../../../glossary.md#term-planning) paths. The larger-than-normal scope is
deliberate because four new public methods activate thirteen independent exact [Tensor](../../../../glossary.md#tensor) surface
locks and the semantic [signature](../../../../glossary.md#term-signature), result [carrier](../../../../glossary.md#term-storage-carrier), shared helper, focused tests, Javadocs, and [API](../../../../glossary.md#term-application-programming-interface-api)
documentation must land atomically.

Stop if a twenty-eighth path, another production or test type, another public method, another
[operation](../../../../glossary.md#operation) kind, a foundation change, another [module](../../../../glossary.md#term-module), [dependency](../../../../glossary.md#term-dependency), [Gradle](../../../../glossary.md#term-gradle), architecture change,
[compiler](../../../../glossary.md#term-compiler) adoption, execution behavior, or detailed task 0024 specification is required.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Update `ScaledDotProductAttentionKind` to explain its one-output and two-output occurrence
  structures without implying hidden outputs in the old path.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Fully document `ScaledDotProductAttentionResult`, both components, constructor null failures,
  exact slot order, shared producer, descriptor distinctions, and non-ownership boundaries.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [descriptor glossary](../../../../glossary.md#term-descriptor); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Update the helper Javadoc and every affected package-private entry with complete parameters,
  result, Shape/type/eligibility/provenance meaning, nullability, failures, and ID side effects.<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter); [Shape glossary](../../../../glossary.md#shape); [provenance glossary](../../../../glossary.md#provenance); [nullability glossary](../../../../glossary.md#term-nullability))</span>
- Add complete Javadoc for all four public Tensor methods, including inputs, defaults, mask and
  attrs behavior, result slots, special-value meaning by reference to the operation contract,
  exact failure conditions, and identifier exhaustion.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [mask glossary](../../../../glossary.md#term-boolean-mask); [operation glossary](../../../../glossary.md#operation))</span>
- Review existing attrs, TensorFactory, TensorProducer, TensorProvenance, TopKResult, and
  BatchNormTrainingResult Javadocs. Record why they remain accurate without changes or stop on a
  contradiction.<span data-glossary-navigation="true"> ([TensorProducer glossary](../../../../glossary.md#producer--tensorproducer))</span>

## Acceptance criteria

- `ScaledDotProductAttentionKind` still has exactly one constant and exposes exactly one signature
  with input range 3–4 and output range 1–2 as specified above.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [signature glossary](../../../../glossary.md#term-signature))</span>
- `ScaledDotProductAttentionAttrs` declaration and behavior remain unchanged.
- `ScaledDotProductAttentionResult` has exactly two non-null Tensor components, exact reference
  retention, ordinary record value semantics, and no additional public API.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- All four existing `scaledDotProductAttention(...)` methods retain exact signatures,
  construction, one-output producer shape, validation, numerical contract, and one-ID behavior.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer))</span>
- Exactly four `scaledDotProductAttentionWithWeights(...)` methods exist and no other public Tensor
  method is added or removed.
- Default, attrs, mask, and mask-plus-attrs variants retain exact inputs/defaults/references and
  produce one two-output occurrence.<span data-glossary-navigation="true"> ([mask glossary](../../../../glossary.md#term-boolean-mask); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Output and [weights](../../../../glossary.md#term-weight) [descriptors](../../../../glossary.md#term-descriptor), [Shapes](../../../../glossary.md#shape), exact [Dimension](../../../../glossary.md#dimension) references, types, [layouts](../../../../glossary.md#layout),
  `requiresGrad`, labels, and storage match the required table.
- Both results retain one exact shared producer and operation with output indices zero and one;
  repeated calls are fresh.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- Existing [validation](../../../../glossary.md#term-validation) order/messages remain exact and every local failure consumes no ID.
- Successful two-output calls consume two IDs in slot order; factory exhaustion returns no partial
  carrier and retains existing non-rollback behavior.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- Focused tests cover masked/unmasked construction, defaults/attrs, dynamic/static/zero Shapes,
  promotion, eligibility differences, numerical weight meanings, exact references, freshness,
  nulls, validation failures, signature rejection, and result-carrier API shape.
- All thirteen public Tensor inventory locks move from 196 to 200 with no unrelated surface
  change.
- Focused validation and exactly one final [model](../../../../glossary.md#term-model) suite pass after executable Java stabilizes.
- A separate clean-context documentation-focused pass finalizes every affected Javadoc, Tensor
  and Compile APIs, glossary, capability/task/master/roadmap status, examples, links, and
  formatting without repeating successful Java tests unless executable behavior changes or a
  concrete risk is recorded.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Compile glossary](../../../../glossary.md#compile))</span>
- Final inventory contains exactly the authorized 27 paths; `git diff --check` passes; 0023F is
  Complete only after both passes; task 0024 remains Draft without a detailed specification.
- Architecture, dependencies, Gradle, other modules, compiler/runtime/backend behavior, and
  completed unrelated contracts remain unchanged.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation-focused tests while developing:

```bash
./gradlew :modules:model:test \
  --tests io.github.pho001.synaptik.model.operation.OperationSignatureTest \
  --tests io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionSemanticsTest \
  --tests io.github.pho001.synaptik.model.tensor.ScaledDotProductAttentionResultTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorScaledDotProductAttentionExpressionTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorTest
```

Exactly one final [model](../../../../glossary.md#term-model) suite after executable Java stabilizes:

```bash
./gradlew :modules:model:test
```

The implementation agent records exact suite/test counts and hands the actual diff plus evidence
to the documentation agent. The documentation pass does not rerun successful Java tests unless it
changes executable Java behavior or records a concrete reason.

Documentation pass:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

The documentation agent also validates Java 26 reflection/generated [Javadoc](../../../../glossary.md#term-javadoc) for the exact kind
[signatures](../../../../glossary.md#term-signature), result record, eight total attention methods, exact 200-method [Tensor](../../../../glossary.md#tensor) surface, both
[producer](../../../../glossary.md#term-producer) forms, local Markdown links/anchors/fences/final newlines, exact 27-path [scope](../../../../glossary.md#term-scope),
synchronized status, and absence of a detailed task-0024 specification. A small runnable
[metadata](../../../../glossary.md#term-metadata) example must demonstrate retrieving `output()` and `weights()` from one result without
claiming execution.

Repository-wide tests are deferred to the task-0024 model capability-selection [checkpoint](../../../../glossary.md#term-checkpoint) or [CI](../../../../glossary.md#term-continuous-integration-ci)
because this task changes only `modules/model` and no [dependency](../../../../glossary.md#term-dependency), build, architecture, or
cross-[module](../../../../glossary.md#term-module) contract.

## Dependencies

- Tasks 0018K and 0018L for family-owned occurrence signatures and exact shared multi-output
  producer/provenance construction.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [signatures glossary](../../../../glossary.md#term-signature); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- Task 0019E for the completed attention operation, attrs, validation, Shapes, numerical semantics,
  and four existing one-output receiver methods.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [validation glossary](../../../../glossary.md#term-validation); [Shapes glossary](../../../../glossary.md#shape))</span>
- Task 0023 and its final audit matrix for the proof that recomputation cannot preserve every
  current attention-weight boundary and that same-occurrence weights are generally useful.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [weight glossary](../../../../glossary.md#term-weight))</span>
- Task 0023E is the completed immediately preceding frontier.

## Follow-up tasks

- Task 0024 remains Draft and depends on completion of all six task-0023 follow-ups. Do not create
  its detailed specification in this task.
- Later compiler work may capture weights slot one when building attention adjoints and owns
  saved-value lifetime, cotangent accumulation, traversal, and optimization.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [weights glossary](../../../../glossary.md#term-weight); [lifetime glossary](../../../../glossary.md#term-lifetime); [cotangent glossary](../../../../glossary.md#cotangent))</span>
- Later planning/prepare/backend work owns fused or decomposed implementation, materialization,
  kernels, execution, accuracy validation, and performance.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [fused glossary](../../../../glossary.md#specialization-and-fusion); [materialization glossary](../../../../glossary.md#materialization); [kernels glossary](../../../../glossary.md#kernel); [validation glossary](../../../../glossary.md#term-validation))</span>

## Architecture impact

Expected impact: None.

This task exposes a second public result already defined by [backend](../../../../glossary.md#backend)-independent attention
semantics and uses the completed [model](../../../../glossary.md#term-model)-owned multi-output [producer](../../../../glossary.md#term-producer) contract. It changes no [module](../../../../glossary.md#term-module)
[ownership](../../../../glossary.md#term-ownership-meanings), [dependency](../../../../glossary.md#term-dependency), [lifecycle](../../../../glossary.md#lifecycle), [runtime](../../../../glossary.md#term-runtime) [hot path](../../../../glossary.md#term-hot-path), or backend-selection rule. Stop and report if
implementation reveals a required architecture change.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, documentation/planning rules, model capabilities/master plan,
roadmap, completed tasks 0016I–0016J/0018K–0018L/0018N/0018Q/0019/0019E/0023–0023E, task
0023F, Tensor and Compile APIs, glossary, and every affected/review-only source/test named by task
0023F in full.

Implement task 0023F exactly inside its 27 authorized paths. Preserve all four existing one-output
attention methods and add only the four explicit WithWeights forms, one exact public result
carrier, the widened one-through-two-output occurrence signature, and one shared two-output
producer. Preserve every attention validation, numerical, Shape, metadata, producer/provenance,
and architecture contract.
Add no kind, recomputation, hidden output, gradient/compiler adoption, execution behavior,
dependency, Gradle, architecture change, or later task. Stop on architecture, dependency,
completed-contract, validation-order, affected-file, or maximum-scope conflict.

Run focused tests while developing and exactly one final model suite after executable Java
stabilizes. Then hand the actual diff and exact Java evidence to a separate clean-context
documentation-focused agent in the same overall change. That agent must independently inspect
final source/tests, finalize Javadocs, Tensor/Compile APIs, glossary, capability/task/master/
roadmap status and documentation validation, and reuse successful Java evidence unless executable
behavior changes or it records a concrete reason.

Do not mark 0023F Complete until both passes and every acceptance criterion succeed. Leave task
0024 Draft without a detailed specification.
```

## Local decisions

- Preserve fluent one-output attention as the default path and use an explicit `WithWeights`
  method family only when the caller needs both results.
- Use one public operation-specific result record, consistently with existing top-K and
  batch-normalization multi-output APIs, rather than a generic tuple or output list.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- Keep one attention kind and widen its single structural signature to output range one through
  two because <span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>[`OperationKind`](../../../../glossary.md#operationkind) requires one unique signature per exact attributes class and the
  semantic operation is unchanged.
- Expose normalized post-mask/post-causal weights, not raw scores or independently recomputed
  softmax values.<span data-glossary-navigation="true"> ([mask glossary](../../../../glossary.md#term-boolean-mask); [softmax glossary](../../../../glossary.md#softmax--log-softmax))</span>
- Give [weights](../../../../glossary.md#term-weight) query/key eligibility only, while retaining query/key/value eligibility for output.
- Preserve all existing validation and numerical policy by deriving both descriptors in the one
  existing helper and delegating once to the completed shared-output factory.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>

## Known limitations

- This task constructs [metadata](../../../../glossary.md#term-metadata) and executes no attention or [softmax](../../../../glossary.md#softmax--log-softmax) values.
- The [weights](../../../../glossary.md#term-weight) [carrier](../../../../glossary.md#term-storage-carrier) adds no fluent forwarding [API](../../../../glossary.md#term-application-programming-interface-api); callers explicitly select `output()` before
  continuing an ordinary Tensor chain.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- No [log](../../../../glossary.md#term-log) weights, scores, [dropout](../../../../glossary.md#dropout--inverted-dropout) [mask](../../../../glossary.md#term-boolean-mask), [cache](../../../../glossary.md#term-cache), [RNG state](../../../../glossary.md#graph-rng-state--graphrngstate), or other auxiliary result is exposed.
- The model records gradient eligibility but defines no attention gradient rule or saved-value
  lifetime.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [gradient glossary](../../../../glossary.md#term-gradient); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- Dynamic constraints remain deferred exactly as in task 0019E; this task adds no binding or
  constraint representation.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

Implementation context: `/root/implement_0023f`.

- The exact prescribed focused command passed 40 tests across its five selected suites with no
  failures, errors, or skips.
- After executable Java stabilized, the single final `./gradlew :modules:model:test` run passed
  1,016 tests across 127 suites with no failures, errors, or skips. <span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>`git diff --check` passed
  immediately afterward. The documentation context changed Javadoc and documentation only, so it
  reused this evidence and did not rerun Java tests.

Documentation context: `/root/implement_0023f/document_0023f`, applying the General,
[API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc), [Planning](../../../../glossary.md#term-planning), and Example profiles.

- Independently reviewed the repository and architecture instructions, focused architecture
  pages, documentation/planning rules and selected profiles, roadmap, capability baseline, master
  plan, this task, final production/test diff, Tensor/Compile/Training/Runtime APIs, glossary,
  attention attrs and semantics, factory/producer/provenance foundations, existing result
  carriers, and directly relevant completed task contracts. No architecture or executable
  contradiction was found.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [Runtime glossary](../../../../glossary.md#term-runtime); [provenance glossary](../../../../glossary.md#provenance))</span>
- Finalized Javadocs in all four production paths. They now distinguish exact one-output
  occurrences from explicit two-output occurrences; document slot order, Shapes, promoted type,
  gradient eligibility, exact shared producer/operation/attributes/inputs/descriptors, validation,
  identifier exhaustion, selected special values, and lifecycle boundaries; and make no compiler,
  gradient, backend, or execution claim. No executable statement changed.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [operation glossary](../../../../glossary.md#operation); [descriptors glossary](../../../../glossary.md#term-descriptor); [lifecycle glossary](../../../../glossary.md#lifecycle); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend))</span>
- Finalized `docs/api/tensor-api.md`, `docs/api/compile-api.md`, and `docs/glossary.md`. The Tensor
  API includes an output-slot table and a runnable Java 26 metadata example retrieving<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>
  `output()` and `weights()` from one result. The Compile API describes compiler-visible metadata
  while leaving traversal, capture, constraints, saved-value lifetime, gradients, and execution
  planned. The glossary adds the reusable attention-weight term and updates current attention
  status.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- `./gradlew :modules:model:javadoc` passed with `BUILD SUCCESSFUL`; two tasks executed. Generated
  Javadoc contains the one-through-two-output kind contract, the exact result record, and all four
  `scaledDotProductAttentionWithWeights` forms. An initial generated-page scan used an inapplicable
  module-qualified output prefix after the successful reflection check and reported missing paths;
  the corrected scan used the actual generated paths and passed.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [module glossary](../../../../glossary.md#term-module))</span>
- `javac --release 26 -cp modules/model/build/classes/java/main -d
  /tmp/synaptik-attention-weights-doc-example /tmp/AttentionWeightsMetadataExample.java` and the
  matching `java` command passed. Output was exact [Shapes](../../../../glossary.md#shape) `Shape[2, 4, 10]` and
  `Shape[2, 4, 6]`, types `FLOAT32,FLOAT32`, eligibility `true,false`, shared [producer](../../../../glossary.md#term-producer) `true`,
  slots `0,1`, exact attributes reference `true`, and three ordered inputs.
- The standalone Java 26 reflection/construction check compiled and passed. It confirmed the sole
  attention kind signature has input range 3–4 and output range 1–2; the public record has exactly<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>
  `output` and [`weights`](../../../../glossary.md#term-weight) Tensor components; Tensor exposes exactly eight attention methods and 200
  total declared public methods; and ordinary versus explicit calls create exact one-output and
  shared two-output producer forms.
- `python3 /tmp/validate_synaptik_markdown.py` passed across 210 tracked and untracked Markdown
  files, 3,535 local links, 207 heading anchors, and 2,662 fence markers, including final-newline
  and trailing-whitespace checks. Final inventory validation found exactly the authorized 27
  paths: four production, sixteen tests, and seven documentation/planning paths. Status checks
  found 0023F Complete and 0024 Draft with no detailed task-0024 file. Final
  `git diff --check` passed.
- Training and Runtime APIs remain accurate without edits because this task adds no training
  workflow, gradient rule, prepared execution, schedule, storage, residency, resource, or run
  contract. <span data-glossary-navigation="true"> ([prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [residency glossary](../../../../glossary.md#residency))</span>`ScaledDotProductAttentionAttrs`, `TensorFactory`, [`TensorProducer`](../../../../glossary.md#producer--tensorproducer),
  `TensorProvenance`, `TopKResult`, and `BatchNormTrainingResult` Javadocs remain accurate because
  the new result reuses their existing semantic-attributes and shared-output foundations without
  changing those contracts. Architecture pages/tests, Gradle/dependencies, other modules,
  backend conformance, and integration tests remain unchanged because there is no module boundary,
  dependency, build, compiler, backend, runtime, or executable behavior change.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Implementation notes

Widened the existing attention kind's sole [occurrence](../../../../glossary.md#term-occurrence) [signature](../../../../glossary.md#term-signature) from exactly one output to one or
two outputs. Preserved all four existing one-output receiver paths and added four explicit
`WithWeights` paths plus the shallow two-component public result. The helper derives both
[descriptors](../../../../glossary.md#term-descriptor) after the existing complete [validation](../../../../glossary.md#term-validation) and delegates once to the existing shared-
output factory; output and normalized [weights](../../../../glossary.md#term-weight) therefore retain one exact [producer](../../../../glossary.md#term-producer) at slots zero
and one without recomputation or a hidden output.

## Completion summary

- Completed changes: preserved exact one-output attention and added explicit same-occurrence
  output-plus-normalized-weights construction with exact descriptor, provenance, validation, and
  identifier behavior.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [weights glossary](../../../../glossary.md#term-weight); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [validation glossary](../../../../glossary.md#term-validation))</span>
- Files changed or created: exactly the authorized 27 paths — four production, sixteen tests, and
  seven documentation/planning files.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: reused the passing 40-test focused run and sole 1,016-test/127-suite final
  model run; model Javadoc, runnable Java 26 example, reflection/construction, generated Javadoc,
  Markdown, exact scope, status/no-task-0024, newline/whitespace, and <span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` validation
  passed.
- Documentation-agent review: completed independently in
  `/root/implement_0023f/document_0023f` with the General, API/Javadoc, Planning, and Example
  profiles.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Documentation impact: Tensor and Compile APIs, glossary, capability baseline, task, master plan,
  and roadmap describe the implemented contract and its current-versus-planned boundaries.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile))</span>
- Javadoc review: all four affected production paths were reviewed and finalized; no executable
  behavior changed after the recorded Java tests.
- Glossary impact: added attention weight and updated scaled dot-product attention to distinguish
  one-output and explicit same-occurrence two-output construction.<span data-glossary-navigation="true"> ([attention weight glossary](../../../../glossary.md#attention-weight); [scaled dot-product attention glossary](../../../../glossary.md#scaled-dot-product-attention))</span>
- Unresolved issues: None.
- Follow-up required: None for 0023F. Task 0024 remains Draft without a detailed specification.

Status: Complete
