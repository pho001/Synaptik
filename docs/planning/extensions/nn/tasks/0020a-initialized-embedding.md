# Task 0020A: Initialized [Embedding](../../../../glossary.md#embedding)

## Status

Complete

## Goal

Add one [eager](../../../../glossary.md#term-eager-evaluation) initialized constructor to the existing final [`Embedding`](../../../../glossary.md#embedding) while preserving its
caller-supplied constructor, single [`weight`](../../../../glossary.md#term-weight) state [leaf](../../../../glossary.md#term-leaf), [unary](../../../../glossary.md#term-unary) forward contract, mode behavior,
replacement behavior, and state-dictionary [schema](../../../../glossary.md#term-schema):

```java
public Embedding(
        long vocabularySize,
        long embeddingSize,
        DataType dataType,
        ParameterInitialization weightInitialization,
        long seed)
```

`vocabularySize` and `embeddingSize` are positive explicit architecture or schema decisions. The
constructor creates the complete `[vocabularySize, embeddingSize]` table during construction; it
does not observe [token](../../../../glossary.md#term-token) IDs, reserve state, or wait for a first forward call. All eight existing
`ParameterInitialization` policies apply through the current `ParameterInitializers` [dispatch](../../../../glossary.md#term-dispatch).
The six random policies use one fresh standard `L64X128MixRandom` source created from [`seed`](../../../../glossary.md#term-seed), while
zero and one neither create nor consume a random source.

Every row in the first initialized-Embedding contract is an ordinary trainable row. There is no
[padding](../../../../glossary.md#padding) index, padding-row rewrite, frozen-row promise, or update exception. Future Text schema
owns padding-token [identity](../../../../glossary.md#term-identity-meanings), proposed Data owns [canonical](../../../../glossary.md#term-canonical-form) valid lengths for ordinary right
padding, and neither concern changes this table's [parameter](../../../../glossary.md#term-parameter) semantics.

## [Mental model](../../../../glossary.md#term-mental-model)

```text
explicit vocabularySize + embeddingSize + type + policy + seed
  -> validate the complete table schema and Java-array feasibility
  -> random policy: create one standard L64X128MixRandom source
     constant policy: create no source
  -> dispatch once to ParameterInitializers for Shape [vocabularySize, embeddingSize]
  -> bind the returned eager Tensor as the permanent Parameter named "weight"
  -> later forward calls remain currentWeight.embedding(indices)
```

Construction initializes storage, not the numerical result of a [lookup](../../../../glossary.md#term-lookup). The returned [parameter](../../../../glossary.md#term-parameter)
[Tensor](../../../../glossary.md#tensor) is an [eager](../../../../glossary.md#term-eager-evaluation) [leaf](../../../../glossary.md#term-leaf); `forward` continues to construct an ordinary declarative [Gather](../../../../glossary.md#gather)
expression without reading index values or executing it.

## [Scope](../../../../glossary.md#term-scope)

- Keep the existing final `Embedding extends UnaryTensorModule` as the only [Embedding](../../../../glossary.md#embedding) type.
- Retain `Embedding(Tensor weight)` unchanged as the advanced caller-supplied-state path.
- Add exactly the five-[argument](../../../../glossary.md#term-argument) [eager](../../../../glossary.md#term-eager-evaluation) constructor shown in the goal.
- Require positive explicit `vocabularySize` and `embeddingSize`; never infer either value from an
  index Tensor, maximum token ID, vocabulary object, batch, or first call.<span data-glossary-navigation="true"> ([token ID glossary](../../../../glossary.md#frozen-whitespace-tokenizer); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- Construct exact `Shape.of(vocabularySize, embeddingSize)` and initialize it eagerly.
- Accept exactly FLOAT64, FLOAT32, or BFLOAT16 parameter type through the existing floating-type
  contract.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>
- Apply the exact supplied `ParameterInitialization` to the complete rank-two Shape by invoking
  the current exhaustive dispatcher once.<span data-glossary-navigation="true"> ([rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- For Glorot and Kaiming policies, retain the generic current rank-two interpretation: Shape axis
  zero is <span data-glossary-navigation="true"> ([Glorot glossary](../../../../glossary.md#term-glorot-uniform); [Kaiming glossary](../../../../glossary.md#term-kaiming))</span>`fanOut` and [axis](../../../../glossary.md#term-axis) one is `fanIn`. For this complete Embedding [Shape](../../../../glossary.md#shape) that means
  `fanOut = vocabularySize` and `fanIn = embeddingSize`; add no Embedding-specific fan value,
  mode, gain, or distribution.<span data-glossary-navigation="true"> ([distribution glossary](../../../../glossary.md#term-distribution))</span>
- Use `RandomGeneratorFactory.of("L64X128MixRandom").create(seed)` exactly once for each
  random-policy construction attempt after deterministic preflight, even if a later eager effect
  fails. Retain neither factory nor generator.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight))</span>
- Route zero and one through `ParameterInitializers.initialize(shape, dataType, policy)` and do
  not look up, create, retain, or consume any random generator.
- Bind the exact returned [Tensor](../../../../glossary.md#tensor) as the sole direct permanent [`Parameter`](../../../../glossary.md#term-parameter) named [`weight`](../../../../glossary.md#term-weight).
- Preserve current `weight()`, `forward(Tensor)`, replacement, train/eval, recursive discovery,
  snapshot, strict-load, and state-dictionary behavior.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Update the directly affected Javadocs, package documentation, Training API explanation, and
  glossary in a separate clean documentation-focused pass before completion.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- `LazyEmbedding`, `Embedding.lazy`, reservation, first-forward [binding](../../../../glossary.md#term-binding), `build`, `bind`, `init`,
  `initialize`, status, retry, or public [schema](../../../../glossary.md#term-schema)-inspection APIs.
- Inferring vocabulary size from observed identifiers or accepting a vocabulary/[tokenizer](../../../../glossary.md#frozen-whitespace-tokenizer) object.
- A `paddingIndex`, special padding row, automatic row zeroing, frozen row, sparse gradient,
  maximum-norm, frequency scaling, or optimizer/update hook.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [gradient glossary](../../../../glossary.md#term-gradient); [norm glossary](../../../../glossary.md#term-norm); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- Tokenization, vocabulary construction/freeze/fingerprint, special-token assignment, batching,
  truncation, valid lengths, masks, runtime input binding, or a Text/Data module implementation.<span data-glossary-navigation="true"> ([Tokenization glossary](../../../../glossary.md#frozen-whitespace-tokenizer); [fingerprint glossary](../../../../glossary.md#term-fingerprint); [token glossary](../../../../glossary.md#term-token); [truncation glossary](../../../../glossary.md#term-truncation-toward-zero); [masks glossary](../../../../glossary.md#term-boolean-mask); [runtime glossary](../../../../glossary.md#term-runtime); [module glossary](../../../../glossary.md#term-module))</span>
- `ModuleFactory` or another construction [facade](../../../../glossary.md#term-facade); that remains Draft [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0020B.
- Bidirectional/multidirectional recurrence, runtime recurrent scan/control flow, or valid-length
  recurrence; those remain later NN tasks.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- Any new operation, numerical lookup, gradient formula, optimizer, compiler, Prepare, Runtime,
  Engine, backend, execution, conformance, integration, build, module-boundary, dependency, ADR,
  or architecture change.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [lookup glossary](../../../../glossary.md#term-lookup); [compiler glossary](../../../../glossary.md#term-compiler); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- Changing the existing caller-supplied constructor or general Module/Parameter/StateDictionary
  lifecycle.<span data-glossary-navigation="true"> ([Parameter glossary](../../../../glossary.md#term-parameter); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Copying legacy code or importing a new [dependency](../../../../glossary.md#term-dependency).

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [NN master plan](../master-plan.md)<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- [NN 0004 explicit eager parameter initializers](0004-explicit-eager-parameter-initializers.md)<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- [NN 0007 Embedding layer](0007-embedding-layer.md)<span data-glossary-navigation="true"> ([Embedding glossary](../../../../glossary.md#embedding))</span>
- [NN 0010 state dictionary and checkpoint contract](0010-state-dictionary-and-checkpoint-contract.md)<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- [NN 0019 automatic first-forward Linear initialization](0019-automatic-first-forward-linear-initialization.md)<span data-glossary-navigation="true"> ([Linear glossary](../../../../glossary.md#term-linear-meanings))</span>
- [NN 0020 automatic recurrent initialization and sequence defaults](0020-automatic-recurrent-initialization-and-sequence-defaults.md)
- [Training API](../../../../api/training-api.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- [Glossary](../../../../glossary.md)
- [Proposed Text master plan](../../text/master-plan.md)
- [Proposed Data master plan](../../data/master-plan.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation profile](../../../../developer-guide/documentation/general-style.md)
- [API/Javadoc profile](../../../../developer-guide/documentation/api-and-javadoc-style.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- [Planning profile](../../../../developer-guide/documentation/planning-style.md)
- [Example profile](../../../../developer-guide/documentation/example-format.md)

`ARCHITECTURE.md` remains authoritative if an explanatory or [planning](../../../../glossary.md#term-planning) document disagrees.

## Architecture constraints

- NN may own layer construction, parameter ownership, and typed forward composition while
  depending only on Model. This task stays inside that existing boundary.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [parameter glossary](../../../../glossary.md#term-parameter); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Model glossary](../../../../glossary.md#term-model))</span>
- Model continues to own Tensor metadata, eager leaf creation, Tensor identity, expression
  provenance, and the ordinary embedding/Gather convenience.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [metadata glossary](../../../../glossary.md#term-metadata); [eager glossary](../../../../glossary.md#term-eager-evaluation); [leaf glossary](../../../../glossary.md#term-leaf); [identity glossary](../../../../glossary.md#term-identity-meanings); [provenance glossary](../../../../glossary.md#provenance); [embedding glossary](../../../../glossary.md#embedding); [Gather glossary](../../../../glossary.md#gather))</span>
- Training remains downstream of NN. This task documents state available to training but adds no
  NN dependency on Training and no optimizer behavior.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- Text and Data remain proposed future boundaries, not dependencies. NN receives numeric schema
  values and never tokenizes or pads raw data.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>
- The constructor is eager because every state Shape fact is already explicit. Reusing the
  reservation/automatic lifecycle would add incomplete state without an input-dependent fact.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Parameter state remains one exact dense Tensor leaf. There is no hidden row mutation or second
  representation of a padding invariant.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [invariant glossary](../../../../glossary.md#term-invariant))</span>
- Existing mode/state-tree behavior and the `UnaryTensorModule` composition boundary are
  unchanged.

## Package impact

- `io.github.pho001.synaptik.nn.layers`: one constructor on existing [`Embedding`](../../../../glossary.md#embedding), plus Javadoc and
  package documentation.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- `io.github.pho001.synaptik.nn.initialization`: no executable API change; generalize explanatory
  Javadoc that currently describes fan Shapes as Linear-only so it truthfully covers the complete
  positive rank-two Shape consumed by Embedding.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Shapes glossary](../../../../glossary.md#shape); [Linear glossary](../../../../glossary.md#term-linear-meanings); [rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Tests remain in the existing `EmbeddingTest`; no new production type or test class is needed.
- No package outside `extensions/nn` changes executable source.

## Public [API](../../../../glossary.md#term-application-programming-interface-api)

After this task, the complete declared public [Embedding](../../../../glossary.md#embedding) surface remains one final class with two
constructors and two methods:

```java
public final class Embedding extends UnaryTensorModule {
    public Embedding(Tensor weight);

    public Embedding(
            long vocabularySize,
            long embeddingSize,
            DataType dataType,
            ParameterInitialization weightInitialization,
            long seed);

    public Parameter weight();

    @Override
    public Tensor forward(Tensor indices);
}
```

Do not add a getter for sizes, type, policy, [seed](../../../../glossary.md#term-seed), random source, or [padding](../../../../glossary.md#padding) configuration. The
current [weight](../../../../glossary.md#term-weight) [descriptor](../../../../glossary.md#term-descriptor) and [Parameter](../../../../glossary.md#term-parameter) [schema](../../../../glossary.md#term-schema) remain the truthful state surface.

## Construction [lifecycle](../../../../glossary.md#lifecycle) and state contract

The initialized constructor completes all work synchronously in its calling thread:

1. validate caller-controlled scalar, reference, type, Shape, element-count, and Java-array-limit
   facts in the exact order specified below;<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [element glossary](../../../../glossary.md#term-element))</span>
2. create a random source only if the selected policy requires one;
3. invoke the matching `ParameterInitializers.initialize` overload exactly once with the complete
   Shape and exact type/policy;
4. after the initializer returns, declare the exact returned [Tensor](../../../../glossary.md#tensor) under direct name [`weight`](../../../../glossary.md#term-weight);
5. return a fully initialized layer whose discovery and state export work immediately.

Successful construction has exactly one permanent state [leaf](../../../../glossary.md#term-leaf) at path `weight`, kind [Parameter](../../../../glossary.md#term-parameter),
[Shape](../../../../glossary.md#shape) `[vocabularySize, embeddingSize]`, exact requested floating type, and
`requiresGrad == true`. The Tensor is the dispatcher's fresh dense-[contiguous](../../../../glossary.md#term-contiguous), [host](../../../../glossary.md#term-host)-backed,
[provenance](../../../../glossary.md#provenance)-free, unlabeled leaf. Success consumes exactly one Tensor identifier through that
initializer. Parameter declaration creates no second Tensor, copy, label, [producer](../../../../glossary.md#term-producer), or random
draw.

There is no reservation or incomplete discovery state. Construction failure returns no layer and
publishes no Parameter wrapper. The random source is transient local construction state and is
not retained even after success.

## [Validation](../../../../glossary.md#term-validation) and effect order

The constructor must perform deterministic validation in this order before random-source [lookup](../../../../glossary.md#term-lookup),
random draw, [Model](../../../../glossary.md#term-model) [carrier](../../../../glossary.md#term-storage-carrier)/destination allocation, [Tensor](../../../../glossary.md#tensor) identifier creation, or [Parameter](../../../../glossary.md#term-parameter)
declaration:

1. reject `vocabularySize <= 0` with `IllegalArgumentException`;
2. reject `embeddingSize <= 0` with `IllegalArgumentException`;
3. require non-null `dataType` with `NullPointerException`;
4. require non-null `weightInitialization` with `NullPointerException`;
5. reject a non-floating `dataType` with `IllegalArgumentException`;
6. create exact `Shape.of(vocabularySize, embeddingSize)`, obtain its checked known element count,
   propagate `ArithmeticException` for count [overflow](../../../../glossary.md#term-overflow), and reject a count above
   `Integer.MAX_VALUE` with `IllegalArgumentException`.

Policy factory [argument](../../../../glossary.md#term-argument) validation has already occurred when the [immutable](../../../../glossary.md#term-immutable)
`ParameterInitialization` was created. After constructor [preflight](../../../../glossary.md#term-preflight):

- a random policy looks up the exact standard factory name, creates one fresh source from the
  exact [`seed`](../../../../glossary.md#term-seed), and passes it once to the four-argument dispatcher;
- zero or one invokes the three-argument dispatcher once and does not look up or create a factory
  or source; the accepted `seed` has no observable effect; and
- the dispatcher's documented allocation, draw, conversion, identifier, exception, and
  non-rollback behavior remains authoritative after effects begin.<span data-glossary-navigation="true"> ([rollback glossary](../../../../glossary.md#term-rollback))</span>

The delegated post-preflight effect order is exact:

| Policy route | Effects after dispatcher validation |
|---|---|
| six random policies | Allocate the typed source carrier, invoke the matching generator method once per [row-major](../../../../glossary.md#term-row-major) [element](../../../../glossary.md#term-element) while converting into that carrier, allocate and copy into the independent destination carrier, then obtain the one Tensor identifier. |
| zero | Allocate the directly zero-initialized destination carrier, then obtain the one Tensor identifier; there is no source carrier, fill loop, copy, factory lookup, or draw. |
| one | Allocate and fill the typed-one source carrier, allocate and copy into the independent destination carrier, then obtain the one Tensor identifier; there is no random factory, source, or draw. |

If a random-source, allocation, or identifier failure occurs after effects begin, already consumed
draws or transient allocations are not rolled back, no Parameter is declared, and no layer is
returned. A source exception occurs before destination allocation and identifier acquisition;
random-policy identifier exhaustion occurs after every draw and both carrier allocations. Zero
identifier exhaustion occurs after destination allocation, while one exhaustion occurs after
source and destination allocation. Do not add recovery, retry, [synchronization](../../../../glossary.md#term-synchronization), or [transaction](../../../../glossary.md#term-transaction)
semantics around these [eager](../../../../glossary.md#term-eager-evaluation) effects.

## Fan-policy contract

[Glorot](../../../../glossary.md#term-glorot-uniform) and [Kaiming](../../../../glossary.md#term-kaiming) policies receive the exact complete
`Shape.of(vocabularySize, embeddingSize)`. They use the existing generic positive [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) rule:

- [axis](../../../../glossary.md#term-axis) zero, `vocabularySize`, is `fanOut`;
- axis one, `embeddingSize`, is `fanIn`;
- Glorot derives its current scale/bound from both values; and
- Kaiming-[ReLU](../../../../glossary.md#term-relu) derives its current scale/bound from `fanIn`.

This is deliberate reuse of the common policy, not a claim that [Embedding](../../../../glossary.md#embedding) requires a distinct
fan abstraction. Do not [transpose](../../../../glossary.md#term-transpose) the [Shape](../../../../glossary.md#shape), substitute equal fans, use only embedding width for
Glorot, or add an Embedding-specific default. Configured normal/uniform and [constants](../../../../glossary.md#term-constant) retain their
ordinary whole-table semantics.

## [Padding](../../../../glossary.md#padding) policy

The selected first contract is explicit absence of special padding behavior:

- every row is initialized by the selected whole-table policy;
- every row has the same ordinary [gradient](../../../../glossary.md#term-gradient) eligibility and replacement semantics;
- neither row zero nor any caller-selected row is overwritten after initialization;
- `forward` remains ordinary [Gather](../../../../glossary.md#gather) and carries no padding [identity](../../../../glossary.md#term-identity-meanings); and
- state export/load retains the complete table exactly.

A future Text vocabulary may assign a padding [token ID](../../../../glossary.md#frozen-whitespace-tokenizer), and proposed Data may [carry](../../../../glossary.md#term-carry) valid lengths,
but those [schema](../../../../glossary.md#term-schema) facts do not freeze a [parameter](../../../../glossary.md#term-parameter) row. A future request for an [invariant](../../../../glossary.md#term-invariant) padding
row is a new capability trigger: it must define compatible gradient, replacement, [optimizer](../../../../glossary.md#term-optimizer)
update, load, and [checkpoint](../../../../glossary.md#term-checkpoint) semantics before [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) exposes such an [API](../../../../glossary.md#term-application-programming-interface-api). Merely zeroing a row at
construction would be misleading because current updates can change it.

## Affected files

Implementation is limited to these nine paths:

1. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/Embedding.java`
2. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/package-info.java`
3. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/initialization/ParameterInitializers.java`
4. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/initialization/package-info.java`
5. `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/EmbeddingTest.java`
6. `docs/api/training-api.md`
7. `docs/glossary.md`
8. `docs/planning/extensions/nn/master-plan.md`
9. `docs/planning/extensions/nn/tasks/0020a-initialized-embedding.md` (new)

Do not edit `ParameterInitialization`, other layer/[module](../../../../glossary.md#term-module)/state-dictionary source or tests, [Tensor](../../../../glossary.md#tensor)
or [Compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api), architecture files, ADRs/tests, global roadmap, Text/Data plans, other master/task
files, build files, Training Java source, [compiler](../../../../glossary.md#term-compiler)/[runtime](../../../../glossary.md#term-runtime)/[prepare](../../../../glossary.md#prepare)/[engine](../../../../glossary.md#term-engine), [backends](../../../../glossary.md#backend), [conformance](../../../../glossary.md#term-conformance),
or integration tests. If correct implementation requires another path, stop and report the exact
blocker before expanding [scope](../../../../glossary.md#term-scope).

## Maximum [scope](../../../../glossary.md#term-scope)

The task may create or modify at most the exact nine paths above. Production behavior is one
constructor in one existing layer. The two initialization documentation paths are included only
because their current [Linear](../../../../glossary.md#term-linear-meanings)-only fan wording becomes inaccurate when the existing generic
dispatcher is deliberately applied to [Embedding](../../../../glossary.md#embedding). The Training [API](../../../../glossary.md#term-application-programming-interface-api) and glossary are the two
current explanatory sources whose statements that Embedding has no initialized constructor must
change in the required documentation pass.

## Test requirements

### Public surface and compatibility

- Reflectively verify [`Embedding`](../../../../glossary.md#embedding) remains final, extends `UnaryTensorModule`, and declares exactly
  the existing supplied constructor plus the new five-argument initialized constructor.<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument))</span>
- Verify its only declared public methods remain `weight()` and `forward(Tensor)` and that no lazy,
  status, padding, schema, initializer, seed, source, factory, or alternate Embedding type appears.<span data-glossary-navigation="true"> ([lazy glossary](../../../../glossary.md#term-lazy-evaluation); [padding glossary](../../../../glossary.md#padding); [schema glossary](../../../../glossary.md#term-schema); [seed glossary](../../../../glossary.md#term-seed))</span>
- Re-run every current supplied-table validation, forward provenance, replacement, mode,
  parameter discovery, state snapshot, and strict-load assertion unchanged.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [validation glossary](../../../../glossary.md#term-validation); [provenance glossary](../../../../glossary.md#provenance); [parameter glossary](../../../../glossary.md#term-parameter); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>

### Initialization policy [matrix](../../../../glossary.md#term-matrix-meanings)

- For each of all eight policies and each of FLOAT64, FLOAT32, and BFLOAT16, construct a small
  positive table and assert exact Shape, type, gradient eligibility, dense host storage,
  provenance absence, label absence, state path, and one successful Tensor-identifier advance.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [Shape glossary](../../../../glossary.md#shape); [gradient glossary](../../../../glossary.md#term-gradient); [host storage glossary](../../../../glossary.md#host-storage); [provenance glossary](../../../../glossary.md#provenance); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Compare represented table values exactly with a direct call to the matching current dispatcher:
  use a separately created standard `L64X128MixRandom` with the same seed for each random policy,
  and the no-generator overload for zero/one.<span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed))</span>
- Test representative unequal `[vocabularySize, embeddingSize]` Shapes for every fan policy so a
  transpose or equal-fan shortcut fails. Assert current formulas treat vocabulary size as fanOut
  and embedding size as fanIn without introducing a new fan API.<span data-glossary-navigation="true"> ([transpose glossary](../../../../glossary.md#term-transpose); [embedding glossary](../../../../glossary.md#embedding); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Two layers with equal shape/type/policy/seed have equal represented values but distinct Tensor
  identities, Parameter wrappers, and storage. Different seeds change representative random
  values; zero/one values are seed-independent.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([Parameter glossary](../../../../glossary.md#term-parameter); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>
- Verify all rows, including row zero, contain the selected policy's ordinary values and share the
  same ordinary trainable table semantics. Do not invent numerical optimizer or gradient tests.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer))</span>

### [Validation](../../../../glossary.md#term-validation) and effects

- Cover non-positive vocabulary size before every later failure, then non-positive embedding size,
  null type, null policy, non-floating type, checked element-count overflow, and count above<span data-glossary-navigation="true"> ([embedding glossary](../../../../glossary.md#embedding); [element glossary](../../../../glossary.md#term-element); [overflow glossary](../../../../glossary.md#term-overflow))</span>
  `Integer.MAX_VALUE` in the specified order.
- For every deterministic rejection, assert no Tensor identifier is consumed and no layer/state is
  returned. Where existing test seams expose Model allocation/source effects, assert none began.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Model glossary](../../../../glossary.md#term-model))</span>
- Verify random policies use the exact `L64X128MixRandom` algorithm and exactly one source stream,
  while zero/one produce exact values for multiple seeds without any random route.<span data-glossary-navigation="true"> ([seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>
- Cover identifier exhaustion for at least one random and one constant policy and preserve the
  dispatcher's documented effect order while publishing no Parameter/layer.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [Parameter glossary](../../../../glossary.md#term-parameter))</span>
- Preserve initializer exceptions and allocation failures without wrapping them in a new layer
  exception or claiming rollback.<span data-glossary-navigation="true"> ([rollback glossary](../../../../glossary.md#term-rollback))</span>

### Forward, replacement, and state [schema](../../../../glossary.md#term-schema)

- Immediately after successful construction, assert `weight()` is stable and recursive/direct
  parameter discovery and <span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter))</span>`stateDictionary()` contain exactly [`weight`](../../../../glossary.md#term-weight).
- Assert compatible replacement and strict load install the exact candidate Tensor reference and
  affect only later calls; incompatible kind/type/Shape/gradient state remains rejected by current
  contracts.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Shape glossary](../../../../glossary.md#shape); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Assert `forward` reads current weight once and produces the same fresh ordinary axis-zero Gather
  expression with exact indices and weight provenance as the supplied constructor.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [Gather glossary](../../../../glossary.md#gather); [provenance glossary](../../../../glossary.md#provenance))</span>
- Assert train/eval remain mode-insensitive and do not change initialized state.
- Assert no [padding](../../../../glossary.md#padding) state, [buffer](../../../../glossary.md#term-buffer), row rewrite, [token](../../../../glossary.md#term-token)-ID inspection, or extra [operation](../../../../glossary.md#operation) appears.

### Exclusions

- Search source, compiled public surface, tests, Javadocs, and current explanatory docs for
  accidental `LazyEmbedding`, `paddingIndex`, padding-row mutation, ModuleFactory, tokenizer/Data,
  runtime scan, compiler/backend, or execution implementation.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [tokenizer glossary](../../../../glossary.md#frozen-whitespace-tokenizer); [runtime glossary](../../../../glossary.md#term-runtime); [scan glossary](../../../../glossary.md#term-cumulative-scan); [compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend))</span>
- Confirm no new module dependency, build path, architecture rule, operation kind, conformance
  fixture, integration fixture, or unrelated refactor is present.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [operation glossary](../../../../glossary.md#operation); [conformance glossary](../../../../glossary.md#term-conformance); [fixture glossary](../../../../glossary.md#term-fixture))</span>

## Documentation requirements

After executable implementation and focused tests, a distinct clean documentation-focused agent
must independently review and finalize the four affected production/package Javadocs, Training
[API](../../../../glossary.md#term-application-programming-interface-api), glossary, this task, and the [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) master plan. Apply the General and API/[Javadoc](../../../../glossary.md#term-javadoc) profiles to
production documentation, the General and [Planning](../../../../glossary.md#term-planning) profiles to planning files, and the Example
profile to code examples.

Documentation must explain:

- the exact new [eager](../../../../glossary.md#term-eager-evaluation) constructor and why explicit complete [Shape](../../../../glossary.md#shape) means no [lazy](../../../../glossary.md#term-lazy-evaluation) [lifecycle](../../../../glossary.md#lifecycle);
- all eight policies, exact standard random algorithm, and zero/one no-[RNG](../../../../glossary.md#graph-rng-state--graphrngstate) behavior;
- [validation](../../../../glossary.md#term-validation) and effect order, including one successful [Tensor](../../../../glossary.md#tensor) ID and non-[rollback](../../../../glossary.md#term-rollback) boundaries;
- the complete `[vocabularySize, embeddingSize]` fan interpretation;
- one permanent [`weight`](../../../../glossary.md#term-weight) state [leaf](../../../../glossary.md#term-leaf) and unchanged supplied/forward/replacement/mode/load behavior;
- explicit absence of [padding](../../../../glossary.md#padding)-row semantics and the future contract required before adding one;
- current versus proposed Text/Data boundaries without presenting proposed modules as implemented;
  and
- declarative Tensor-expression construction rather than numerical execution, gradient, compiler,
  runtime, or backend behavior.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend))</span>

Review `ParameterInitialization` Javadoc, [Module](../../../../glossary.md#term-module)/[Parameter](../../../../glossary.md#term-parameter)/StateDictionary Javadocs, Tensor API,
[Compile](../../../../glossary.md#compile) API, Training Java API, architecture documents/tests, build files, other modules, and
[conformance](../../../../glossary.md#term-conformance)/integration documentation. Record reasoned no-change conclusions unless a concrete
inaccuracy requires stopping for [scope](../../../../glossary.md#term-scope) expansion. Do not repeat a successful Java suite merely to
reproduce implementation evidence when the documentation pass changes no executable Java.

## Acceptance criteria

- The existing final [`Embedding`](../../../../glossary.md#embedding) and `Embedding(Tensor)` remain source-compatible.
- Exactly the specified [eager](../../../../glossary.md#term-eager-evaluation) constructor is added, with no companion [lifecycle](../../../../glossary.md#lifecycle) or factory [API](../../../../glossary.md#term-application-programming-interface-api).
- Both explicit sizes are positive and never inferred from [token](../../../../glossary.md#term-token) IDs or first-forward input.
- Every deterministic validation completes in the specified order before random/allocation/ID
  effects.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- All eight existing policies delegate once through `ParameterInitializers` for exact complete
  Shape <span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape))</span>`[vocabularySize, embeddingSize]`.
- Six random policies use one fresh exact standard `L64X128MixRandom` source seeded by [`seed`](../../../../glossary.md#term-seed);
  zero/one neither create nor consume an RNG.<span data-glossary-navigation="true"> ([RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- Fan-based policies use the current complete rank-two rule with vocabulary as fanOut and
  embedding width as fanIn; no new fan abstraction exists.<span data-glossary-navigation="true"> ([rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Success eagerly creates exactly one fresh gradient-enabled floating Tensor leaf and one permanent
  Parameter named <span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [Tensor glossary](../../../../glossary.md#tensor); [leaf glossary](../../../../glossary.md#term-leaf); [Parameter glossary](../../../../glossary.md#term-parameter))</span>[`weight`](../../../../glossary.md#term-weight); discovery and state export are immediately complete.
- Existing forward, replacement, mode, and state-dictionary behavior remains unchanged.
- Every row is ordinary trainable state. There is no padding index, row zeroing, frozen row, or
  token/data ownership leak.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- No [compiler](../../../../glossary.md#term-compiler)/[runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend)/build/architecture/[dependency](../../../../glossary.md#term-dependency) or unrelated behavior changes.
- Focused and full NN validation, warning-free Javadoc, documentation checks, public-surface
  checks, exact-path checks, and whitespace gates pass.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- A clean documentation-focused pass has finalized affected documentation and recorded all
  no-change reviews.

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation context, [run](../../../../glossary.md#term-run) focused tests while iterating:

```bash
./gradlew :extensions:nn:test --tests io.github.pho001.synaptik.nn.layers.EmbeddingTest
```

Then run the affected [module](../../../../glossary.md#term-module) and its Javadocs once as the authoritative Java validation:

```bash
./gradlew :extensions:nn:clean :extensions:nn:test :extensions:nn:javadoc
```

The implementation and documentation passes must also validate:

- reflected and external-use public constructor/method surface;
- exact initialization values, standard algorithm, state schema, identifier/effect order, and
  forward provenance;<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [provenance glossary](../../../../glossary.md#provenance))</span>
- forbidden source/import/[API](../../../../glossary.md#term-application-programming-interface-api) and stale-current-claim searches;
- local Markdown links and anchors, unique headings, balanced fences, rendered Javadoc pages,
  and documented examples;<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- exact nine-path scope, terminal newlines, trailing whitespace, no-index whitespace for new
  files, and <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check`; and
- master/task frontier consistency: 0020A is Complete after the mandatory documentation pass,
  0020B–0020C and 0021–0024 remain Draft, no NN task is Ready, and no later task specifications
  exist.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>

Repository-wide validation is deferred because this task changes one [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) public constructor and no
[dependency](../../../../glossary.md#term-dependency), architecture boundary, shared build configuration, or multiple-module executable
behavior. [CI](../../../../glossary.md#term-continuous-integration-ci) or the next recorded capability [checkpoint](../../../../glossary.md#term-checkpoint) owns the wider gate.

## Dependencies

- Completed [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0004 and 0004A [eager](../../../../glossary.md#term-eager-evaluation) initializer/effect contracts.
- Completed NN 0007 current [Embedding](../../../../glossary.md#embedding) state/forward contract.
- Completed NN 0010 state-dictionary replacement/load contract.
- Completed NN 0019 automatic-versus-eager [lifecycle](../../../../glossary.md#lifecycle) distinction.
- Completed NN 0020 common `ParameterInitialization`, dispatcher, and standard [PRNG](../../../../glossary.md#graph-rng-state--graphrngstate) selection.
- Current [Model](../../../../glossary.md#term-model) [`Shape`](../../../../glossary.md#shape), [`DataType`](../../../../glossary.md#data-type--datatype), eager Tensor factories/random factories, Tensor identity, and
  ordinary embedding/Gather expression contracts.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [identity glossary](../../../../glossary.md#term-identity-meanings); [Gather glossary](../../../../glossary.md#gather))</span>
- Proposed Text vocabulary/padding-token and Data valid-length boundaries only as reviewed future
  ownership; this task has no dependency on an unimplemented module.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [token glossary](../../../../glossary.md#term-token); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module))</span>

## Follow-up tasks

- NN 0020B may use the completed initialized Embedding constructor in stateless standard
  construction recipes; it remains Draft and has no task file yet.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Embedding glossary](../../../../glossary.md#embedding))</span>
- NN 0020C directionality and NN 0021–0024 scan/valid-length/integration work remain Draft and
  separate.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- A padding-row API is not scheduled. Reopen it only when a concrete consumer supplies an owning
  contract that preserves the invariant across initialization, gradient construction, Parameter
  replacement, optimizer updates, strict load, and checkpoint restore.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [invariant glossary](../../../../glossary.md#term-invariant); [gradient glossary](../../../../glossary.md#term-gradient); [Parameter glossary](../../../../glossary.md#term-parameter); [optimizer glossary](../../../../glossary.md#term-optimizer); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Proposed Text/Data tasks eventually own vocabulary size/fingerprint, special-token identity,
  batching/padding policy, and valid lengths after their required architecture/module decisions.<span data-glossary-navigation="true"> ([fingerprint glossary](../../../../glossary.md#term-fingerprint); [token glossary](../../../../glossary.md#term-token); [identity glossary](../../../../glossary.md#term-identity-meanings); [module glossary](../../../../glossary.md#term-module))</span>

## Architecture impact

None. The change adds one [eager](../../../../glossary.md#term-eager-evaluation) constructor inside the existing [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) layer boundary and delegates to
existing [Model](../../../../glossary.md#term-model)-backed initializer and [embedding](../../../../glossary.md#embedding) contracts. It adds no [module](../../../../glossary.md#term-module), [dependency](../../../../glossary.md#term-dependency),
[operation](../../../../glossary.md#operation), [runtime](../../../../glossary.md#term-runtime) [lifecycle](../../../../glossary.md#lifecycle), [backend](../../../../glossary.md#backend) responsibility, architecture rule, [ADR](../../../../glossary.md#term-architecture-decision-record-adr), or architecture-test
change.

## Implementation prompt

Work in a fresh implementation context. Read `AGENTS.md`, `ARCHITECTURE.md`, the current
architecture plan, [planning](../../../../glossary.md#term-planning) guide, roadmap, [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) master plan, this task, completed NN 0004/0007/0010/
0019/0020 tasks, final [`Embedding`](../../../../glossary.md#embedding), `ParameterInitialization`, `ParameterInitializers`, related
[Linear](../../../../glossary.md#term-linear-meanings)/recurrent constructors, [Module](../../../../glossary.md#term-module)/[Parameter](../../../../glossary.md#term-parameter)/StateDictionary contracts, [Model](../../../../glossary.md#term-model) [Shape](../../../../glossary.md#shape)/[DataType](../../../../glossary.md#data-type--datatype)/
[Tensor factory](../../../../glossary.md#tensor-factory)/random/embedding APIs, existing Embedding tests, Training [API](../../../../glossary.md#term-application-programming-interface-api), glossary, and
proposed Text/Data boundary sections. Architecture wins; stop before editing if a real conflict
appears.

Implement only the exact nine affected paths. Keep one final Embedding type and the supplied
constructor. Add exactly the [eager](../../../../glossary.md#term-eager-evaluation) five-[argument](../../../../glossary.md#term-argument) constructor, the specified [validation](../../../../glossary.md#term-validation)/effect
order, exact `L64X128MixRandom` random path, no-[RNG](../../../../glossary.md#graph-rng-state--graphrngstate) [constant](../../../../glossary.md#term-constant) path, one whole-table dispatcher call,
one permanent [`weight`](../../../../glossary.md#term-weight), generic [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) fan interpretation, and no [padding](../../../../glossary.md#padding) behavior. Preserve all
current forward/replacement/mode/state behavior. Do not implement [lazy](../../../../glossary.md#term-lazy-evaluation) state, ModuleFactory,
[tokenization](../../../../glossary.md#frozen-whitespace-tokenizer)/Data, valid lengths, directionality, [scan](../../../../glossary.md#term-cumulative-scan), execution, [gradient](../../../../glossary.md#term-gradient)/[optimizer](../../../../glossary.md#term-optimizer) behavior, or
architecture/build changes.

Use focused tests while iterating, then [run](../../../../glossary.md#term-run) the authoritative NN clean/test/[Javadoc](../../../../glossary.md#term-javadoc) command once.
Do not [commit](../../../../glossary.md#term-commit), push, stage, revert, or modify concurrent unrelated work. End with the repository-
required completion summary and exact evidence. Do not mark Complete until implementation,
documentation, and all required gates pass.

## Documentation-agent [handoff](../../../../glossary.md#term-handoff)

After implementation and authoritative Java [validation](../../../../glossary.md#term-validation), create a distinct clean documentation
context with only this task's exact contract, affected paths, final diff, Java evidence, required
documentation profiles, and no-change review list. That context must independently inspect final
source/tests/Javadocs and finalize documentation without broadening behavior. It should rerun
focused Java only if it changes executable Java or discovers a concrete reason; otherwise reuse
the implementation evidence and [run](../../../../glossary.md#term-run) [Javadoc](../../../../glossary.md#term-javadoc)/Markdown/surface/path/whitespace gates.

## Local decisions

- [Eager](../../../../glossary.md#term-eager-evaluation) construction is selected because the complete [parameter](../../../../glossary.md#term-parameter) [Shape](../../../../glossary.md#shape) is explicit.
- Vocabulary and [embedding](../../../../glossary.md#embedding) sizes are both positive explicit [schema](../../../../glossary.md#term-schema)/architecture values.
- The common eight-policy value and current dispatcher are reused without a new default.
- High-level random policies use exact standard `L64X128MixRandom`; [constants](../../../../glossary.md#term-constant) use no [RNG](../../../../glossary.md#graph-rng-state--graphrngstate).
- Fan policies consume the complete `[vocabularySize, embeddingSize]` Shape under the current
  axis-zero fanOut/axis-one fanIn rule.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>
- Every table row is ordinary trainable state. [Padding](../../../../glossary.md#padding) [identity](../../../../glossary.md#term-identity-meanings) is not parameter-update semantics.
- Existing supplied state is retained as the advanced constructor; no compatibility alias or
  separate type is needed.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- One [`weight`](../../../../glossary.md#term-weight) [leaf](../../../../glossary.md#term-leaf) remains the complete state and [checkpoint](../../../../glossary.md#term-checkpoint) schema.

## Known limitations

- No [padding](../../../../glossary.md#padding) row is protected from [gradient](../../../../glossary.md#term-gradient) or [optimizer](../../../../glossary.md#term-optimizer) updates.
- No vocabulary/[tokenizer](../../../../glossary.md#frozen-whitespace-tokenizer) [artifact](../../../../glossary.md#term-artifact), padding [identity](../../../../glossary.md#term-identity-meanings), or valid-length value is accepted.
- No default initializer, type, vocabulary size, [embedding](../../../../glossary.md#embedding) width, or [seed](../../../../glossary.md#term-seed) is supplied.
- Construction is [eager](../../../../glossary.md#term-eager-evaluation) and may allocate a large [host](../../../../glossary.md#term-host) table immediately.
- Eager initializer effects are not transactional after random draws/allocation begin.
- The layer constructs expressions only; numerical lookup, gradient execution, compilation,
  lowering, and backend execution remain separately owned.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [lowering glossary](../../../../glossary.md#lowering); [backend glossary](../../../../glossary.md#backend))</span>
- Persistent [checkpoint](../../../../glossary.md#term-checkpoint) transport and [optimizer state](../../../../glossary.md#term-optimizer-state) remain outside [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)'s in-memory dictionary.

## [Validation](../../../../glossary.md#term-validation) evidence

The isolated [planning](../../../../glossary.md#term-planning) context `/root/nn_0020a_planning` completed the mandatory architecture,
planning, documentation-profile, completed-task, final-source/[Javadoc](../../../../glossary.md#term-javadoc)/test, [Tensor](../../../../glossary.md#tensor)/Training, and
proposed Text/Data boundary review without finding an architecture conflict. It changed only the
[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) master plan and this new task file; concurrent [CPU](../../../../glossary.md#term-cpu)/[backend](../../../../glossary.md#backend) planning and global-roadmap paths
remained untouched.

Targeted planning validation passed for both NN planning paths: all local Markdown links and
anchors resolve, headings are unique, fences are balanced, terminal newlines are present, and no
trailing whitespace exists. Frontier checks found exactly one [Ready](../../../../glossary.md#term-ready) NN master row and one Ready
task status, linked 0020A from the master, retained 0020B–0020C and 0021–0024 as Draft, and found
no later task files. Final no-index whitespace, exact-path, shared-worktree diff, and
`git diff --check` gates passed. The no-index check returned only the expected new-file diff
status and no whitespace diagnostics. Worktree inspection attributed exactly
`docs/planning/extensions/nn/master-plan.md` and this new task to the NN planning pass, while
separately identifying the pre-existing concurrent CPU master/task and global-roadmap paths.

The isolated clean implementation context `/root/nn_0020a_implementation` added exactly the [eager](../../../../glossary.md#term-eager-evaluation)
five-[argument](../../../../glossary.md#term-argument) constructor, retained the caller-supplied constructor and one permanent [`weight`](../../../../glossary.md#term-weight)
[leaf](../../../../glossary.md#term-leaf), and preserved forward, mode, replacement, and strict-load behavior. Deterministic [preflight](../../../../glossary.md#term-preflight)
now completes before effects; the six random policies use one fresh standard
`L64X128MixRandom`, while zero and one use the generator-free dispatcher. All policies consume the
complete `[vocabularySize, embeddingSize]` [Shape](../../../../glossary.md#shape) and every row remains ordinary trainable state.

The focused [Embedding](../../../../glossary.md#embedding) suite passed 15 tests with no failures, errors, or skips. The sole
authoritative `./gradlew :extensions:nn:test` [run](../../../../glossary.md#term-run) passed 256 tests with no failures, errors, or
skips. Preliminary `./gradlew :extensions:nn:javadoc` passed warning-free. `javap` and reflection
confirmed exactly two public constructors, two public declared methods, one private `weight`
field, and no nested type or [lifecycle](../../../../glossary.md#lifecycle) surface. External source using both constructors and
`forward` compiled against the built NN and [Model](../../../../glossary.md#term-model) classes. [Dependency](../../../../glossary.md#term-dependency) inspection retained only
the existing NN and Model classes plus `java.base`; no build or [module](../../../../glossary.md#term-module) dependency changed.

Independent clean documentation context `/root/nn_0020a_docs` reviewed the final implementation,
tests, state contracts, Tensor construction/embedding APIs, Training and [Compile](../../../../glossary.md#compile) boundaries, and
proposed Text/Data [ownership](../../../../glossary.md#term-ownership-meanings) without finding an executable, public-[API](../../../../glossary.md#term-application-programming-interface-api), architecture, dependency,
or [scope](../../../../glossary.md#term-scope) defect. It finalized the four affected production/package Javadocs, Training API,
glossary, this task, and master evidence. The exact eager validation/effect order, one successful
Tensor identifier, whole-table fan interpretation, permanent state, unchanged forward/replacement/
mode/load behavior, and deliberate absence of [padding](../../../../glossary.md#padding)-specific rows are now synchronized.

No [architecture contract](../../../../glossary.md#architecture-contract), explanatory architecture document, [ADR](../../../../glossary.md#term-architecture-decision-record-adr), or [architecture test](../../../../glossary.md#term-architecture-test) changed
because the implementation stays inside the existing NN-to-Model dependency and adds no boundary
rule. No Tensor, Compile, or Training Java API changed: Embedding still delegates to the existing
ordinary Model [Gather](../../../../glossary.md#gather) expression; the [compiler](../../../../glossary.md#term-compiler) receives no new [operation](../../../../glossary.md#operation) or rule; and Training
still has only its placeholder Java type while the updated Training API document explains the NN
state it may later consume. `ParameterInitialization`, `Module`, [`Parameter`](../../../../glossary.md#term-parameter), and state-dictionary
contracts remain accurate because the constructor reuses the closed policy and declares the same
single complete `weight` [schema](../../../../glossary.md#term-schema). Compiler, [Runtime](../../../../glossary.md#term-runtime), [Prepare](../../../../glossary.md#prepare), [Engine](../../../../glossary.md#term-engine), backends, build/dependencies,
[conformance](../../../../glossary.md#term-conformance) tests, integration tests, other modules, and the global roadmap require no change
because the task adds no behavior in those [owners](../../../../glossary.md#term-owner). Concurrent CPU/backend and global-roadmap work
remained untouched.

Final warning-free `./gradlew :extensions:nn:javadoc` passed, and the rendered `Embedding`,
`ParameterInitializers`, layers-package, and initialization-package pages were inspected. `javap`
and a fresh external reflection/use program confirmed exactly two public constructors, two public
declared methods, one private `weight` field, no nested type or lifecycle surface, and successful
use of both constructors and `forward`. `jdeps` retained only NN, Model, and `java.base` classes.
Forbidden lifecycle/padding surface and import checks passed. Local Markdown links and anchors,
unique headings, balanced fences, exact nine-path NN scope, 0020A Complete/0020B–0020C and
0021–0024 Draft/no Ready/no future task files, terminal newlines, trailing whitespace, new-file
no-index whitespace, and `git diff --check` all passed. The documentation pass changed no
executable Java or test and therefore reused the focused 15-test and authoritative 256-test NN
evidence above rather than repeating it. Repository-wide validation remains deferred for the
unchanged scope reason recorded in this task.

## Implementation notes

- Added the exact eager constructor with positive size, null/type, checked count, and Java-array
  preflight in the required order.<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [preflight glossary](../../../../glossary.md#term-preflight))</span>
- Routed random policies through one fresh seeded standard generator and the four-argument
  dispatcher; routed zero/one through the three-argument dispatcher without generator creation.<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument))</span>
- Registered the exact initialized Tensor only after initializer success and retained one
  immediately discoverable/exportable <span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>[`weight`](../../../../glossary.md#term-weight) [parameter](../../../../glossary.md#term-parameter).
- Expanded `EmbeddingTest` across all policies and floating types, full-Shape fan semantics,
  deterministic values and seeds, state identity/storage, validation and identifier ordering,
  constant row-zero behavior, forward provenance, replacement, mode, and strict dictionary load.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [identity glossary](../../../../glossary.md#term-identity-meanings); [validation glossary](../../../../glossary.md#term-validation); [constant glossary](../../../../glossary.md#term-constant); [provenance glossary](../../../../glossary.md#provenance))</span>
- Generalized initializer fan Javadocs from Linear-only wording to the existing generic positive
  rank-two <span data-glossary-navigation="true"> ([Linear glossary](../../../../glossary.md#term-linear-meanings); [rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>`[fanOut, fanIn]` contract.

## Completion summary

The [eager](../../../../glossary.md#term-eager-evaluation) initialized [`Embedding`](../../../../glossary.md#embedding) constructor, focused coverage, and affected Javadocs and
explanatory documentation are complete. The mandatory independent documentation review found no
executable, public-[API](../../../../glossary.md#term-application-programming-interface-api), architecture, [dependency](../../../../glossary.md#term-dependency), or [scope](../../../../glossary.md#term-scope) defect and preserved the existing
supplied-state, forward, mode, replacement, and state-dictionary contracts. All required Java,
[Javadoc](../../../../glossary.md#term-javadoc), public-surface, dependency, Markdown, exact-scope, status, newline, and whitespace
[validation](../../../../glossary.md#term-validation) is recorded above. No unresolved issue or follow-up is required for task 0020A; [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)
0020B–0020C and 0021–0024 remain separate Draft work.

Status: Complete
