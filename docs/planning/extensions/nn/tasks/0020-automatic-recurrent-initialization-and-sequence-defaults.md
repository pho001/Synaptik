# Task 0020: Automatic Recurrent Initialization and Sequence Defaults

> **Editorial reading context — not a status update.** Initialization policy is immutable
> algorithm configuration; the cell owns parameter Shapes, seed use and publication.
> In a static unroll, one cell's exact weight leaves feed many fresh step expressions, allowing
> Compiler to combine gradient contributions to the same parameters. Default zero states are
> newly supplied starting values, not hidden memory retained across calls. A zero-step unroll
> may never visit an automatic cell and therefore leave its parameters unbound.
> Read today's [initialization contract](../../../../reference/nn/composition.md#current-nn-automatic-parameter-initialization-contract)
> and [recurrent sequence contract](../../../../reference/nn/recurrent-layers.md#current-nn-recurrent-composition-contract)
> to separate parameter lifetime, expression construction and explicit recurrent state.

## Status

Complete

## Goal

Make the existing final `RnnCell`, `GruCell`, and `LstmCell` usable without declaring
`inputSize` or constructing a random-generator factory at ordinary [model](../../../../glossary.md#term-model)-composition call sites.
Each cell infers only the positive static final input [extent](../../../../glossary.md#term-extent) on its first compatible forward
path, initializes and publishes its complete [parameter group](../../../../glossary.md#parameter-group--parametergroup), and then builds the ordinary
one-step [Tensor](../../../../glossary.md#tensor) expression in that same traversal. `hiddenSize`, [bias](../../../../glossary.md#term-bias) presence, [parameter](../../../../glossary.md#term-parameter)
[`DataType`](../../../../glossary.md#data-type--datatype), recurrent [weight](../../../../glossary.md#term-weight) policy, and [seed](../../../../glossary.md#term-seed) remain explicit per cell or sequence.

Before adding that second automatic consumer, replace the recent [Linear](../../../../glossary.md#term-linear-meanings)-only
`LinearWeightInitialization` enum with one reusable closed [immutable](../../../../glossary.md#term-immutable) `ParameterInitialization`
configuration value. It selects one of the existing [eager](../../../../glossary.md#term-eager-evaluation) algorithms without owning layer [schema](../../../../glossary.md#term-schema),
Tensor state, [RNG](../../../../glossary.md#graph-rng-state--graphrngstate), seed, or [callback](../../../../glossary.md#term-callback). Linear migrates in the same change and no compatibility
[alias](../../../../glossary.md#aliasing-and-arena) remains; this deliberate early break avoids freezing duplicate Linear and recurrent policy
types.

The same cohesive task makes the current static sequence containers ergonomic for the common
case. Each sequence may construct and own one standard automatic cell, synthesize non-[gradient](../../../../glossary.md#term-gradient)
zero initial states from the input [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) extent and the cell's explicit hidden-size/type schema,
and treat every time step as valid when no Java lengths are supplied. Existing constructors that
accept an exact caller-supplied cell and existing overloads with explicit states and lengths remain
the advanced contracts.

```java
var model = Model.define(topology -> {
    Embedding embedding = topology.addModule(
            "embedding",
            new Embedding(embeddingWeight));
    LstmSequence encoder = topology.addModule(
            "encoder",
            new LstmSequence(
                    128,
                    true,
                    DataType.FLOAT32,
                    ParameterInitialization.glorotUniform(),
                    42L));

    return (Tensor tokenIds) -> encoder.forward(embedding.forward(tokenIds));
});
```

For embedded input [Shape](../../../../glossary.md#shape) `[time, batch, embeddingSize]`, the first represented recurrent step
initializes `encoder.cell.inputWeight` as `[4 * 128, embeddingSize]`,
`encoder.cell.hiddenWeight` as `[4 * 128, 128]`, and optional zero bias as `[4 * 128]`. The
one-[argument](../../../../glossary.md#term-argument) sequence call creates zero hidden and cell states shaped `[batch, 128]`, derives a
Java all-[valid length](../../../../glossary.md#term-valid-length) for each row, and returns the existing `LstmSequenceForwardResult`.

One Java cell instance is intentionally reused across time. It owns one parameter set, while each
call to its forward method constructs fresh Tensor-expression occurrences:

```text
same inputWeight leaf ──> time-0 projection ──> h0
         │
         ├──────────────> time-1 projection ──> h1
         │                                     ^
         └──────────────> time-2 projection ──> h2

same hiddenWeight leaf -> each represented hidden projection
h0 -> time-1 carried-state path
h1 -> time-2 carried-state path
```

The outputs and their [producers](../../../../glossary.md#term-producer) have fresh Tensor [identities](../../../../glossary.md#term-identity-meanings), but every time-step projection
retains the same exact parameter Tensor [leaf](../../../../glossary.md#term-leaf). That static unroll is the forward expression [graph](../../../../glossary.md#graph)
needed for [backpropagation through time](../../../../glossary.md#term-backpropagation-through-time-bptt) ([BPTT](../../../../glossary.md#backpropagation-through-time-and-recurrent-state)): current [compiler](../../../../glossary.md#term-compiler) [autograd](../../../../glossary.md#autograd) traverses the resulting
expression ancestry by exact Tensor [identity](../../../../glossary.md#term-reference-identity) and combines contributions that fan out to the same
parameter with ordinary `Tensor.add`. This task preserves those existing Model/Compiler facts; it
does not add an [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) backward method, [runtime](../../../../glossary.md#term-runtime) tape, recurrent gradient rule, execution loop, or
[backend](../../../../glossary.md#backend) support claim.

## [Scope](../../../../glossary.md#term-scope)

- Keep exactly one final public type for each existing recurrent cell and sequence. Do not add
  `LazyRnnCell`, `LazyGruCell`, `LazyLstmCell`, a generic lazy module, or a public build/bind/init
  lifecycle.<span data-glossary-navigation="true"> ([lazy glossary](../../../../glossary.md#term-lazy-evaluation); [module glossary](../../../../glossary.md#term-module); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Replace the recently introduced public `LinearWeightInitialization` with one general closed
  immutable public value, `ParameterInitialization`, in `nn.initialization`. This is an intentional
  early breaking rename performed atomically across current NN source, tests, Javadocs, and
  explanatory documentation. Delete the old type and do not retain a deprecated alias, adapter,
  duplicate recurrent enum, compatibility overload, or split Linear/Recurrent policy family.
  Completed task 0019 remains unchanged as historical evidence of the API it originally delivered;
  this task owns migration of the current API.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [alias glossary](../../../../glossary.md#aliasing-and-arena); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Give `ParameterInitialization` exactly these named public factories/presets:

  ```java
  public final class ParameterInitialization {
      public static ParameterInitialization glorotNormal()
      public static ParameterInitialization glorotUniform()
      public static ParameterInitialization kaimingReluNormal()
      public static ParameterInitialization kaimingReluUniform()
      public static ParameterInitialization normal(double mean, double standardDeviation)
      public static ParameterInitialization uniform(
              double lowerBoundInclusive,
              double upperBoundExclusive)
      public static ParameterInitialization zeros()
      public static ParameterInitialization ones()
      public boolean requiresRandomGenerator()
  }
  ```

  The type has no public constructor, field, nested type, raw-kind accessor, mutable callback, or
  arbitrary implementation/subclass extension point. It owns only algorithm selection and the
  two configured binary64 values needed by <span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [callback glossary](../../../../glossary.md#term-callback); [binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>`normal` or `uniform`; it never owns a Shape, data
  type, Tensor, Module, Parameter, RNG/factory, seed, mutable state, fan value, bias policy,
  parameter name/order, or checkpoint encoding.<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [Tensor glossary](../../../../glossary.md#tensor); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Add two public [dispatch](../../../../glossary.md#term-dispatch) entries to the existing stateless `ParameterInitializers` namespace:

  ```java
  public static Tensor initialize(
          Shape shape,
          DataType dataType,
          ParameterInitialization initialization)

  public static Tensor initialize(
          Shape shape,
          DataType dataType,
          ParameterInitialization initialization,
          RandomGenerator randomGenerator)
  ```

  The three-[argument](../../../../glossary.md#term-argument) form accepts only `zeros()` and `ones()` and never requests, creates, retains,
  or consumes an RNG. The four-argument form accepts only the six random policies and delegates
  with the exact transient caller-owned generator. Both dispatch exactly once to the matching
  existing `zeros`, `ones`, `normal`, `uniform`, `glorotNormal`, `glorotUniform`,
  `kaimingReluNormal`, or `kaimingReluUniform` entry; they do not duplicate [sampling](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters)/fan formulas.
  `requiresRandomGenerator()` lets a layer select the correct overload before it creates a source.
- Preserve the existing `ParameterInitializers` [Shape](../../../../glossary.md#shape) boundary. `normal`, `uniform`, `zeros`, and
  `ones` accept the same fully static Shapes their existing methods accept. The four fan presets
  retain the current fully static positive rank-two <span data-glossary-navigation="true"> ([rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>`[fanOut, fanIn]` contract. Do not invent a
  public `Fan`, convolution schema, gain, activation, fan mode, orthogonal policy, implementation
  registry, callback, or new family of initializer classes before a concrete consumer requires it.<span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [schema glossary](../../../../glossary.md#term-schema); [activation glossary](../../../../glossary.md#term-activation); [registry glossary](../../../../glossary.md#term-registry))</span>
- A recurrent selected value applies independently to the complete input-weight Shape and then the
  complete hidden-weight Shape. One fresh generator stream is used in that exact matrix order for
  random policies; each fan preset recomputes fan values from the matrix currently being created.
  GRU/LSTM use their complete packed matrix Shapes. Optional bias remains a layer-owned complete
  typed-zero input-side vector, including LSTM's forget interval, regardless of the weight policy.<span data-glossary-navigation="true"> ([weight glossary](../../../../glossary.md#term-weight); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru); [LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm); [vector glossary](../../../../glossary.md#term-vector))</span>
- Migrate the current automatic [`Linear`](../../../../glossary.md#term-linear-meanings) constructor in place so its policy [parameter](../../../../glossary.md#term-parameter) is
  `ParameterInitialization`; preserve its explicit deterministic caller-owned
  `RandomGeneratorFactory` and seed. Its weight uses the generic dispatcher and its optional bias
  remains layer-owned zero. For `zeros()` and `ones()`, the retained factory is validated as before
  but is never invoked. Preserve all supplied/eager Linear constructors.

  ```java
  public Linear(
          long outFeatures,
          boolean bias,
          DataType dataType,
          ParameterInitialization weightInitialization,
          RandomGeneratorFactory<? extends RandomGenerator> randomGeneratorFactory,
          long seed)
  ```
- Add one ordinary automatic constructor to each existing final cell:

  ```java
  public RnnCell(
          long hiddenSize,
          boolean bias,
          DataType dataType,
          ParameterInitialization weightInitialization,
          long seed)

  public GruCell(
          long hiddenSize,
          boolean bias,
          DataType dataType,
          ParameterInitialization weightInitialization,
          long seed)

  public LstmCell(
          long hiddenSize,
          boolean bias,
          DataType dataType,
          ParameterInitialization weightInitialization,
          long seed)
  ```

  These constructors accept no `inputSize`, `RandomGenerator`, or `RandomGeneratorFactory`.
  They validate positive `hiddenSize`; non-null `dataType`; non-null
  `weightInitialization`; floating type; then checked gate-multiplier/hidden-only Shape/count facts,
  in that order. Any `long` [seed](../../../../glossary.md#term-seed) is valid. They then retain [immutable](../../../../glossary.md#term-immutable) configuration, reserve
  `inputWeight`, `hiddenWeight`, then optional [`bias`](../../../../glossary.md#term-bias), and create no generator, Tensor, Tensor ID,
  or Parameter. Each standard sequence constructor delegates this same validation exactly once to
  the one cell it constructs and installs no child on failure.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Use the exact [JDK](../../../../glossary.md#term-java-development-kit-jdk) named deterministic pseudo-[random-number generator](../../../../glossary.md#graph-rng-state--graphrngstate) ([PRNG](../../../../glossary.md#graph-rng-state--graphrngstate))
  `L64X128MixRandom` for every automatic recurrent initialization attempt that uses a random
  policy. The algorithm name and supplied seed are part of this high-level construction contract.
  Create a fresh generator from `RandomGeneratorFactory.of("L64X128MixRandom").create(seed)` only
  after preflight and only when <span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight))</span>`requiresRandomGenerator()` is true; retain neither factory nor
  generator in cell state. `zeros()` and `ones()` create and consume no RNG. No global/default
  mutable source, seed manager, implicit seed splitting, JVM-time seed, or caller-owned source is
  used.<span data-glossary-navigation="true"> ([JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm))</span>
- Preserve every existing supplied-Tensor cell constructor and the existing eager constructor
  with explicit `inputSize` and transient caller-owned `RandomGenerator`. Those remain the
  low-level paths for exact state or caller-controlled random algorithms/lifecycles.
- Infer `inputSize` only from a positive static final Dimension of the first compatible input.
  Leading Dimensions remain ordinary broadcast metadata. The automatic cell's parameter<span data-glossary-navigation="true"> ([Dimension glossary](../../../../glossary.md#dimension); [metadata glossary](../../../../glossary.md#term-metadata))</span>
  [`DataType`](../../../../glossary.md#data-type--datatype) does not silently cast an input or explicit state; existing floating promotion rules
  continue to apply. After binding, the final input Dimension must remain compatible with the
  established input-width schema under the existing cell contract.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- Reuse the complete private reservation/publication lifecycle delivered by NN 0019. Each
  automatic cell publishes all of its direct parameters together or none. Accessors, parameter
  discovery, and state export fail closed until publication. Strict state load may bind the
  complete reservation group without creating a generator or invoking an initializer.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
- Preserve the existing stable parameter names, order, Shapes, gate packing, equations, bias
  association, result carriers, supplied/eager state-dictionary compatibility, replacement
  behavior, and mode-insensitive composition.
- Let each cell retain only the private/package-private immutable schema facts required by its
  matching sequence to derive default state and prevalidate an automatic cell. Add no public
  size/type/status/schema getter and no shared public recurrent base/interface.
- Add one high-level constructor to each existing final sequence:

  ```java
  public RnnSequence(
          long hiddenSize,
          boolean bias,
          DataType dataType,
          ParameterInitialization weightInitialization,
          long seed)

  public GruSequence(
          long hiddenSize,
          boolean bias,
          DataType dataType,
          ParameterInitialization weightInitialization,
          long seed)

  public LstmSequence(
          long hiddenSize,
          boolean bias,
          DataType dataType,
          ParameterInitialization weightInitialization,
          long seed)
  ```

  Each constructs exactly one matching automatic cell and permanently owns it under the existing
  child name `cell`. Preserve each existing constructor accepting an exact caller-supplied cell.
- Add exactly these sequence forward overloads, in addition to the current explicit-state plus
  lengths overloads:

  ```java
  // RNN and GRU
  forward(Tensor input, Tensor initialHidden)
  forward(Tensor input, long[] lengths)
  forward(Tensor input)

  // LSTM
  forward(Tensor input, Tensor initialHidden, Tensor initialCell)
  forward(Tensor input, long[] lengths)
  forward(Tensor input)
  ```

  The overload without lengths means every original batch row is valid for exactly the complete
  static input time extent. It derives an internal Java <span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [extent glossary](../../../../glossary.md#term-extent))</span>`long[]` filled with `time`; this remains
  construction metadata and does not become a Tensor, Parameter, Buffer, retained field, or
  runtime input.<span data-glossary-navigation="true"> ([Buffer glossary](../../../../glossary.md#term-buffer); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- The overload with lengths but without states creates default zero state and keeps the current
  static packing behavior. The one-input overload combines all-valid lengths with default zero
  state. Explicit-state overloads without lengths combine caller-owned state with all-valid
  lengths. Existing most-explicit overloads remain canonical advanced entry points.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- A default hidden state is a fresh [eager](../../../../glossary.md#term-eager-evaluation) [provenance](../../../../glossary.md#provenance)-free `TensorFactory.zeros` [leaf](../../../../glossary.md#term-leaf) shaped
  `[batch, hiddenSize]`, using the cell's exact configured/declared floating parameter type,
  `Optional.empty()` label, and `requiresGrad == false`. LSTM creates distinct hidden and cell
  zero leaves with that same schema. Batch comes from the fully static input axis one; hidden size
  and type come from the cell's immutable validated schema. State is created per call, never
  cached, registered, retained, or placed in a state dictionary.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>
- Preserve the current fully static [time-major](../../../../glossary.md#term-time-major) input requirement and `long[]` static packing.
  Do not turn zero Tensor values into padding. Do not add runtime Tensor lengths, masks, dense
  masked recurrence, or a scan/control-flow operation.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [masks glossary](../../../../glossary.md#term-boolean-mask); [scan glossary](../../../../glossary.md#term-cumulative-scan); [operation glossary](../../../../glossary.md#operation))</span>
- Preserve the rule that a sequence invokes its cell once per non-empty represented step. An
  explicit all-zero length call invokes no cell; therefore an otherwise unbound automatic cell
  remains unbound until a later call reaches a represented step or strict load binds it. A
  default-state overload still returns its newly created zero state(s) for an all-zero request.
- Every represented step must call the same exact Java cell instance. Every cell call must create
  fresh output Tensor identities and fresh expression-producer occurrences while retaining exact
  references to the same current parameter leaves. Do not clone a cell or parameter per step.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer))</span>
- Add focused automatic lifecycle, policy/PRNG, strict-load, retry, concurrency, overload,
  zero-state, all-valid, ownership, public-surface, and static-unroll provenance tests across all
  three concrete families.<span data-glossary-navigation="true"> ([concurrency glossary](../../../../glossary.md#term-concurrency); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Finalize all affected Javadocs, package documentation, Training API, glossary, master/task
  evidence, and no-change reasoning in a separate clean documentation-focused task after the
  executable diff and authoritative NN tests stabilize.

## Out of [scope](../../../../glossary.md#term-scope)

- Initialized [`Embedding`](../../../../glossary.md#embedding), vocabulary inference, padding-row behavior, or a padding index. Those
  require the separate NN 0020A padding-row/update decision; vocabulary size must never be
  inferred from one batch's maximum token ID.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [padding glossary](../../../../glossary.md#padding); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [token ID glossary](../../../../glossary.md#frozen-whitespace-tokenizer))</span>
- A deprecated `LinearWeightInitialization` alias, a separate recurrent policy enum, public
  policy callback/service-provider interface, user-defined policy implementation, mutable
  initializer, or retention of RNG/seed/type/Shape inside <span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [callback glossary](../../../../glossary.md#term-callback); [mutable glossary](../../../../glossary.md#term-mutable); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [seed glossary](../../../../glossary.md#term-seed); [Shape glossary](../../../../glossary.md#shape))</span>`ParameterInitialization`.
- [Convolution](../../../../glossary.md#term-channels-first-convolution-layers)-specific fan derivation, a public `Fan` value, configurable gain/activation/fan
  mode, orthogonal initialization, per-gate initialization, or forget-bias offset. The current
  generic policy delegates fan presets only through the existing rank-two Shape contract; a later
  concrete convolution consumer must design any broader schema.<span data-glossary-navigation="true"> ([activation glossary](../../../../glossary.md#term-activation); [bias glossary](../../../../glossary.md#term-bias); [rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [schema glossary](../../../../glossary.md#term-schema))</span>
- `ModuleFactory`, `ModuleFactory.standard()`, static construction recipes, registries, providers,
  dependency injection, configuration aggregation, global data type, global random source, or
  seed allocation. NN 0020B owns the later stateless construction facade after initialized
  recurrent and Embedding contracts are stable.<span data-glossary-navigation="true"> ([dependency injection glossary](../../../../glossary.md#term-dependency-injection); [data type glossary](../../../../glossary.md#data-type--datatype); [facade glossary](../../../../glossary.md#term-facade))</span>
- Bidirectional, reverse, or multidirectional recurrence; independent directional parameter
  sets/seeds; directional merge policy; stacked recurrent layers; residuals; recurrent dropout;
  or type-specific bidirectional results. NN 0020C owns that separate contract.<span data-glossary-navigation="true"> ([seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span>
- `Lazy*` public types, public initialization status, manual `build`, `bind`, `initialize`,
  `reset`, model-wide descriptor tracing, whole-model preflight, or rollback of an arbitrary
  functional Model body.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [preflight glossary](../../../../glossary.md#term-preflight); [rollback glossary](../../../../glossary.md#term-rollback))</span>
- Inferring hidden size, bias, parameter data type, policy, seed, vocabulary size, embedding size,
  class count, output width, gate order, or gate count from input data.
- A generic `RecurrentCell`, `RecurrentSequence`, erased state tuple, public cell-schema value,
  generic recurrent result, common direction base, or new `UnaryTensorModule`/`Sequential`
  participation.
- Changing the existing RNN tanh, GRU reset-after, or LSTM IFGO equations, gate packing, input-
  side-only bias, all-zero bias, result-component order, or static packing/restoration semantics.<span data-glossary-navigation="true"> ([RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn); [tanh glossary](../../../../glossary.md#term-tanh); [GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru); [LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm))</span>
- Runtime Tensor lengths, Boolean mask input, dynamic loop count, runtime-dependent active Shape,
  recurrent scan/control-flow body, dense post-cell masking, or reinterpretation of <span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Tensor glossary](../../../../glossary.md#tensor); [Boolean mask glossary](../../../../glossary.md#term-boolean-mask))</span>`CUM_SUM` /
  `CUM_PROD` as a recurrent [scan](../../../../glossary.md#term-cumulative-scan).
- Dense padded output, batch-first layout, sorting by length, padding inference from numeric zero,
  or changes to the Data/Text/Vision proposals.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout))</span>
- A backward method on Module/Model/cell/sequence, retained tape, mutable Tensor gradient, new
  derivative rule, public compiler invocation, Training session, optimizer, checkpoint bytes,
  graph execution, backend route, numerical result, or performance guarantee.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [derivative glossary](../../../../glossary.md#term-derivative); [compiler glossary](../../../../glossary.md#term-compiler); [Training session glossary](../../../../glossary.md#training-session--trainingsession); [optimizer glossary](../../../../glossary.md#term-optimizer); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [graph glossary](../../../../glossary.md#graph); [backend route glossary](../../../../glossary.md#backend-route))</span>
- Changes to [`Module`](../../../../glossary.md#term-module), [`Parameter`](../../../../glossary.md#term-parameter), [`Buffer`](../../../../glossary.md#term-buffer), `StateDictionary`, [`Model`](../../../../glossary.md#term-model), [`Topology`](../../../../glossary.md#term-topology), Model Tensor/
  Shape/DataType APIs, Training Java source, Compiler, Runtime, Prepare, Engine, backend code,
  Gradle/build structure, architecture contracts/ADRs/tests, integration/conformance tests, the
  active CPU work, or the global roadmap.<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Detailed task files or implementation for [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0020A, 0020B, 0020C, or NN 0021–0024.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [ADR 0007: Neural-network module and training boundary](../../../../design/decisions/0007-neural-network-module-and-training-boundary.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [module glossary](../../../../glossary.md#term-module))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [Training API](../../../../api/training-api.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- [Glossary](../../../../glossary.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [NN master plan](../master-plan.md)<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- [Completed NN task 0012: Vanilla RNN cell](0012-vanilla-rnn-cell.md)<span data-glossary-navigation="true"> ([RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn))</span>
- [Completed NN task 0013: GRU cell](0013-gru-cell.md)<span data-glossary-navigation="true"> ([GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru))</span>
- [Completed NN task 0014: LSTM cell](0014-lstm-cell.md)<span data-glossary-navigation="true"> ([LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm))</span>
- [Completed NN task 0015: Static packed RNN sequence](0015-static-packed-rnn-sequence.md)
- [Completed NN task 0016: Static packed GRU sequence](0016-static-packed-gru-sequence.md)
- [Completed NN task 0017: Static packed LSTM sequence](0017-static-packed-lstm-sequence.md)
- [Completed NN task 0018: Typed functional Model](0018-typed-functional-model-topology.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Completed NN task 0019: Automatic Linear initialization](0019-automatic-first-forward-linear-initialization.md)<span data-glossary-navigation="true"> ([Linear glossary](../../../../glossary.md#term-linear-meanings))</span>

## Architecture constraints

- `extensions/nn` continues to depend only on `modules/model`. Production must not import
  Training, Compiler, Runtime, Prepare, Engine, CPU, Metal, CUDA, or another backend.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [CPU glossary](../../../../glossary.md#term-cpu); [Metal glossary](../../../../glossary.md#term-metal); [CUDA glossary](../../../../glossary.md#term-cuda); [backend glossary](../../../../glossary.md#backend))</span>
- NN owns module state, recurrent-cell parameters, cell-specific sequence composition, default
  initial-state convenience, and initialization policy selection. Model continues to own Tensor
  identity, descriptors, Shape algebra, promotion, eager zero/random leaves, expression
  provenance, and primitive operations. Compiler continues to own autograd and graph capture.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Model glossary](../../../../glossary.md#term-model); [Tensor glossary](../../../../glossary.md#tensor); [identity glossary](../../../../glossary.md#term-identity-meanings); [descriptors glossary](../../../../glossary.md#term-descriptor); [Shape glossary](../../../../glossary.md#shape); [eager glossary](../../../../glossary.md#term-eager-evaluation); [provenance glossary](../../../../glossary.md#provenance); [autograd glossary](../../../../glossary.md#autograd); [graph capture glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction))</span>
- A recurrent state passed between steps is caller-/expression-threaded Tensor data, not a
  Parameter, Buffer, retained field, optimizer value, or runtime tape. A default state changes
  only who constructs that explicit Tensor for the current call.<span data-glossary-navigation="true"> ([recurrent state glossary](../../../../glossary.md#backpropagation-through-time-and-recurrent-state); [Buffer glossary](../../../../glossary.md#term-buffer); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- [`Parameter`](../../../../glossary.md#term-parameter) remains always bound when publicly observable. Future state exists only as private
  Module reservations until one complete cell-local publication or strict load succeeds.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication))</span>
- [`Module`](../../../../glossary.md#term-module) retains no universal forward method. Cells and sequences stay direct final Module
  subclasses with their truthful type-specific signatures and remain outside <span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>`Sequential`.
- Current Tensor producer identity is occurrence-specific. Reusing a Java module cannot reuse a
  derived Tensor result implicitly; every public Tensor operation call must return a fresh
  identity with provenance retaining its exact inputs.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [occurrence glossary](../../../../glossary.md#term-occurrence); [Java module glossary](../../../../glossary.md#model-module-buffer-and-scan-distinctions); [operation glossary](../../../../glossary.md#operation))</span>
- Current compiler reverse mode keys contributions by exact Tensor identity and merges multiple
  contributions through ordinary <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-reference-identity))</span><span data-glossary-navigation="true"> ([reverse mode glossary](../../../../glossary.md#term-reverse-mode))</span>`Tensor.add`. NN must preserve the shared parameter leaf and
  fresh time-step producer structure that lets this existing downstream rule see temporal
  fan-out. NN must not import or invoke compiler internals to prove it.<span data-glossary-navigation="true"> ([leaf glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([fan-out glossary](../../../../glossary.md#term-fan-out))</span>
- Static Java lengths may specialize the statically unrolled expression structure. Runtime
  lengths must not be presented as supported until NN 0021's cross-module scan/input-binding
  prerequisite is complete.<span data-glossary-navigation="true"> ([unrolled glossary](../../../../glossary.md#backpropagation-through-time-and-recurrent-state); [scan glossary](../../../../glossary.md#term-cumulative-scan); [binding glossary](../../../../glossary.md#term-binding))</span>
- If implementation needs a new Model operation/method, public lifecycle/status/schema API,
  caller-owned source retention, new dependency, generic recurrent abstraction, runtime control
  flow, change outside the exact path list, or architecture-rule change, stop and report the
  blocker rather than widening this task.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [schema glossary](../../../../glossary.md#term-schema); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [dependency glossary](../../../../glossary.md#term-dependency); [widening glossary](../../../../glossary.md#term-widening))</span>

## Package impact

Existing packages changed:

- `io.github.pho001.synaptik.nn.initialization` — replace the Linear-only enum with one closed
  general policy value, add policy dispatch to the existing stateless namespace, and update
  package documentation.<span data-glossary-navigation="true"> ([Linear glossary](../../../../glossary.md#term-linear-meanings); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- `io.github.pho001.synaptik.nn.layers` — the six existing concrete cell/sequence types and
  automatic Linear policy parameter plus package documentation.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter))</span>

Existing packages consumed unchanged:

- `io.github.pho001.synaptik.nn.module` for ownership, reservation/publication, state discovery,
  strict load, and typed Model composition.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [publication glossary](../../../../glossary.md#term-publication); [Model glossary](../../../../glossary.md#term-model))</span>
- Model data-type, Shape, Tensor factory/provenance, eager random, and primitive expression
  packages.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [Tensor factory glossary](../../../../glossary.md#tensor-factory); [provenance glossary](../../../../glossary.md#provenance); [eager glossary](../../../../glossary.md#term-eager-evaluation))</span>

No package, [Gradle](../../../../glossary.md#term-gradle) [module](../../../../glossary.md#term-module), [dependency edge](../../../../glossary.md#term-dependency-edge), architecture rule, public training type, or [compiler](../../../../glossary.md#term-compiler)
type is added.

## Public [API](../../../../glossary.md#term-application-programming-interface-api)

`ParameterInitialization` is a final [immutable](../../../../glossary.md#term-immutable) closed value with the eight public static factories
and one `requiresRandomGenerator()` query shown in [Scope](../../../../glossary.md#term-scope), plus value-based `equals`, [`hashCode`](../../../../glossary.md#hash-codes-and-content-digests),
and diagnostic `toString`. It has no public/protected constructor, field, nested type, raw-kind
getter, [callback](../../../../glossary.md#term-callback), or subclass/implementation extension point. No-[argument](../../../../glossary.md#term-argument) presets may be
canonicalized, but [reference identity](../../../../glossary.md#term-reference-identity) is not part of the contract.

Two values are equal exactly when they select the same algorithm and their configured [binary64](../../../../glossary.md#floating-point-precision-and-rounding)
arguments, if any, have equal `Double.doubleToLongBits` representations. Equality is therefore
[signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero)-sensitive; non-[finite](../../../../glossary.md#term-finite) configured values cannot exist. Hashing uses the same facts.
`toString` is deterministic and descriptive but is neither parsing nor [persistence](../../../../glossary.md#term-persistence) format.

`normal(mean, standardDeviation)` validates at factory call time: finite [mean](../../../../glossary.md#term-mean), then finite
[standard deviation](../../../../glossary.md#term-standard-deviation), then numerically non-negative standard deviation; either [signed zero](../../../../glossary.md#term-signed-zero) is
accepted. `uniform(lowerBoundInclusive, upperBoundExclusive)` validates finite lower bound, then
finite upper bound, then strict `lower < upper`. Presets need no [validation](../../../../glossary.md#term-validation). These checks create no
[Tensor](../../../../glossary.md#tensor), identifier, storage, or [RNG](../../../../glossary.md#graph-rng-state--graphrngstate) effect.

`ParameterInitializers.initialize(shape, dataType, initialization)` null-checks those arguments in
that order, rejects random policies before delegating, and is the only generic zero/one route.
`ParameterInitializers.initialize(shape, dataType, initialization, randomGenerator)` null-checks in
that order, rejects zero/one policies before delegating, and is the generic random route. After
[dispatch](../../../../glossary.md#term-dispatch) selection, the matching existing initializer owns its already documented floating-type,
fully-static [Shape](../../../../glossary.md#shape), Java-array/count, [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) fan, [sampling](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters), failure, and side-effect validation
order. The dispatchers retain no input and add no fallback/default behavior.

The automatic [`Linear`](../../../../glossary.md#term-linear-meanings) constructor replaces its `LinearWeightInitialization` [parameter](../../../../glossary.md#term-parameter) with
`ParameterInitialization`; every other parameter and behavior remains. Deleting the old type is
deliberately source-breaking because the API is new, repository-owned consumers can migrate
atomically, and preserving two public names would freeze avoidable duplication before stable
external compatibility is promised.

Each cell adds only its five-argument automatic constructor. Existing Tensor-supplied and
explicit-`inputSize`/caller-`RandomGenerator` constructors, parameter accessors, and forward
methods remain source-compatible. No cell adds a public getter, initialization method, status,
factory, or overload accepting `RandomGeneratorFactory`.

Each sequence adds only its five-argument standard-cell constructor and the three forward
overloads listed in Scope. Existing caller-cell constructor, `cell()` accessor, most-explicit
forward overload, and result record remain source-compatible. Result records gain no component
or method.

The user-facing common cases become:

```java
RnnSequence rnn = new RnnSequence(
        64, true, DataType.FLOAT32,
        ParameterInitialization.glorotUniform(), 11L);
RnnSequenceForwardResult complete = rnn.forward(input);
RnnSequenceForwardResult packed = rnn.forward(input, lengths);

LstmSequence lstm = new LstmSequence(
        128, true, DataType.FLOAT32,
        ParameterInitialization.normal(0.0d, 0.02d), 12L);
LstmSequenceForwardResult continued =
        lstm.forward(input, previousHidden, previousCell, lengths);
```

The second [LSTM](../../../../glossary.md#term-long-short-term-memory-lstm) call demonstrates that automatic parameter initialization does not make [recurrent
state](../../../../glossary.md#backpropagation-through-time-and-recurrent-state) implicit. Callers that continue a stream still pass exact previous states.

## Automatic cell initialization contract

For [RNN](../../../../glossary.md#term-recurrent-neural-network-rnn), let `gateMultiplier = 1`; for [GRU](../../../../glossary.md#term-gated-recurrent-unit-gru), `3`; and for [LSTM](../../../../glossary.md#term-long-short-term-memory-lstm), `4`. One automatic cell stores
only validated [immutable](../../../../glossary.md#term-immutable) `hiddenSize`, [bias](../../../../glossary.md#term-bias) choice, floating [parameter](../../../../glossary.md#term-parameter) type, selected policy,
and [seed](../../../../glossary.md#term-seed) before [binding](../../../../glossary.md#term-binding). Its eventual state [schema](../../../../glossary.md#term-schema) is:

| Path | [Shape](../../../../glossary.md#shape) | Initialization order and policy |
|---|---|---|
| `inputWeight` | `[gateMultiplier * hiddenSize, inferredInputSize]` | first; selected general policy applied to this exact Shape |
| `hiddenWeight` | `[gateMultiplier * hiddenSize, hiddenSize]` | second; same selected policy applied independently to this exact Shape and, when random, the advanced generator |
| `bias` when configured | `[gateMultiplier * hiddenSize]` | third; deterministic typed zero, no random draw |

`normal`, `uniform`, `zeros`, and `ones` use each exact [matrix](../../../../glossary.md#term-matrix-meanings) Shape without fan [inference](../../../../glossary.md#term-inference). Each
[Glorot](../../../../glossary.md#term-glorot-uniform)/[Kaiming](../../../../glossary.md#term-kaiming) preset derives its [distribution](../../../../glossary.md#term-distribution) independently from the current complete packed
[rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) Shape. The layer owns matrix schema/order and bias policy; the general value does not.

The first compatible direct cell forward performs:

1. reject null input/state [arguments](../../../../glossary.md#term-argument) in the existing order;
2. validate every caller-controlled input/state rank, floating type, statically decidable feature,
   promotion, broadcast, gate/state Shape, packed-size, count, and Java-array-limit fact knowable
   from immutable configuration, without creating a Tensor;<span data-glossary-navigation="true"> ([feature glossary](../../../../glossary.md#term-feature))</span>
3. when still unbound, enter one cell-local initialization critical section, recheck completion,
   and revalidate the input-derived schema;
4. if and only if the policy requires randomness, create one fresh `L64X128MixRandom` generator
   from the exact retained seed;
5. create input weight, hidden weight, then optional zero bias as unpublished local Tensors,
   choosing the no-RNG or exact-generator dispatcher overload as required;<span data-glossary-navigation="true"> ([weight glossary](../../../../glossary.md#term-weight); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
6. pass the complete ordered group once to `bindReservedParameters`, which validates all values,
   prepares all wrappers, publishes all state, and release-publishes completion;<span data-glossary-navigation="true"> ([release glossary](../../../../glossary.md#term-release))</span>
7. leave the initialization critical section, read the complete current parameter bindings once,
   and construct the existing ordinary one-step Tensor formula; and
8. return that call's ordinary current result.

Later calls do not create a generator or parameter [Tensor](../../../../glossary.md#tensor). They validate against the established
schema, read current wrappers, and create fresh ordinary expression occurrences. A compatible
replacement or strict load affects later calls only; prior expressions retain earlier exact
parameter references.

The first automatic input final [Dimension](../../../../glossary.md#dimension) must be positive and static. After binding, current
cell rules continue to permit only a compatible final input Dimension; do not add a cast or infer
a second schema. Leading [rank](../../../../glossary.md#term-rank) and broadcast-compatible Dimensions may vary as allowed by the
existing one-step formulas.

## Failure, retry, load, and [concurrency](../../../../glossary.md#term-concurrency) contract

- Constructor validation consumes no random draw, Tensor ID, Tensor, storage, Parameter, or
  wrapper. It reserves the complete direct state names only after immutable configuration is
  valid.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Tensor glossary](../../../../glossary.md#tensor); [Parameter glossary](../../../../glossary.md#term-parameter); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- A caller-controlled direct-cell validation failure before initialization consumes no generator,
  draw, Tensor ID, or parameter publication.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication))</span>
- An initializer/source/allocation/ID failure may preserve completed random calls, local Tensors,
  host allocations, and opaque consumed Tensor IDs, but publishes no Parameter wrapper. A retry
  of a random policy creates a fresh standard generator from the same algorithm and seed, so its
  represented random sequence restarts deterministically; zero/one retries create no generator.
  Opaque IDs do not roll back.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [opaque glossary](../../../../glossary.md#term-opaque); [seed glossary](../../../../glossary.md#term-seed))</span>
- Complete publication happens before that first call constructs its recurrent expression. A
  later expression-construction failure does not undo already published parameters.
- Synchronization covers only competing first initialization attempts on one cell. Release/
  acquire reservation completion guarantees that racing access/discovery observes unbound failure
  or the complete group, never a partial group. Replacement, strict load, sequence construction,
  mode changes, and arbitrary functional Model bodies remain caller-coordinated.<span data-glossary-navigation="true"> ([Synchronization glossary](../../../../glossary.md#term-synchronization); [Release glossary](../../../../glossary.md#term-release); [Model glossary](../../../../glossary.md#term-model))</span>
- Strict load validates complete paths and each configured gate/type/Shape/gradient schema before
  publishing any reservation. Candidate <span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [gradient glossary](../../../../glossary.md#term-gradient); [schema glossary](../../../../glossary.md#term-schema))</span>`inputWeight` supplies the positive input width;
  `hiddenWeight` and optional bias must match configured hidden size, gate multiplier, type, and
  existing packing. Successful load retains exact candidate Tensor references and invokes no
  PRNG or initializer.<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias); [PRNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- Equivalent supplied/eager/automatic cells keep the same path/kind/type/Shape schema and can
  load dictionaries in either direction when hidden/input dimensions and bias choice match.<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [dimensions glossary](../../../../glossary.md#dimension))</span>
- In a functional Model, an earlier layer or cell can remain initialized when later user code
  fails, and a registered cell can remain unbound when its branch or every recurrent step is
  skipped. There is no model-wide transaction.<span data-glossary-navigation="true"> ([transaction glossary](../../../../glossary.md#term-transaction))</span>

## Sequence default and all-valid contract

Every sequence overload preserves fully static [time-major](../../../../glossary.md#term-time-major) input `[time, batch, inputSize]` and
the current result/packing semantics.

| Supplied [arguments](../../../../glossary.md#term-argument) | Derived values | Meaning |
|---|---|---|
| `input, explicitState(s), lengths` | none | existing most-explicit static packed call |
| `input, explicitState(s)` | `lengths[b] = time` | caller state, every step valid |
| `input, lengths` | zero state(s) | static packed call from zero state |
| `input` | zero state(s), `lengths[b] = time` | ordinary complete-sequence call |

All overloads must converge on one validated packing implementation rather than duplicate the
cell loop. Overload-specific derived array/state creation must preserve the following facts:

- all-valid lengths are a fresh internal host array, checked against Java array/index limits and
  never retained or exposed;
- zero state uses `[batch, hiddenSize]`, exact cell parameter type, no label, eager zero storage,
  false gradient eligibility, and fresh identity for each state/call;<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- the default state's type does not come from token IDs or silently mirror a wider input type;
  mixed-floating expression results continue to follow current promotion;<span data-glossary-navigation="true"> ([token glossary](../../../../glossary.md#term-token))</span>
- explicit state references are never copied, replaced, cached, or converted;
- an all-zero explicit-length call with explicit state preserves the current exact-reference
  final-state shortcut and creates no cell expression;
- an all-zero explicit-length call with default state returns its newly created zero state(s),
  invokes no cell, and leaves an otherwise automatic cell unbound; and
- each non-empty step uses the same owned cell exactly once and retains current active-set,
  stable-row-order, exit-state restoration, and compact-output semantics.

Before allocating an all-valid array, zero state, [eager](../../../../glossary.md#term-eager-evaluation) index [leaf](../../../../glossary.md#term-leaf), automatic [parameter](../../../../glossary.md#term-parameter), or
expression, each overload must complete every caller-controlled null, [rank](../../../../glossary.md#term-rank), fully static [Shape](../../../../glossary.md#shape),
time/[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)/input-[feature](../../../../glossary.md#term-feature), cell-[schema](../../../../glossary.md#term-schema), hidden-size/type, length-count/range, packed-count,
promotion, broadcast, [gather](../../../../glossary.md#gather)/select/stack, and Java-array-limit check that is knowable from its
actual and derived arguments. [Validation](../../../../glossary.md#term-validation) may inspect [immutable](../../../../glossary.md#term-immutable) package-private cell schema facts;
it must not require a public schema [API](../../../../glossary.md#term-application-programming-interface-api) or an already bound Parameter. After this [preflight](../../../../glossary.md#term-preflight):

1. allocate any required all-valid Java length [snapshot](../../../../glossary.md#term-snapshot);
2. allocate required default zero state leaf/leaves in hidden-then-cell order;
3. construct the existing static SELECT/GATHER recurrence; and
4. let the first represented `cell.forward` bind an automatic cell before that cell constructs
   its own formula.

An initialization or construction failure after these effects begin does not roll back the
derived [host](../../../../glossary.md#term-host) array, zero-state/parameter/index [Tensor](../../../../glossary.md#tensor) IDs, allocations, random draws, published
cell state, or expression prefix. No partial sequence result is returned, and no per-call value is
retained by the sequence. A call with no represented step ends after state derivation/restoration
and never reaches automatic cell [binding](../../../../glossary.md#term-binding).

## Static unroll, [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt) [provenance](../../../../glossary.md#provenance), and [parameter](../../../../glossary.md#term-parameter) [fan-out](../../../../glossary.md#term-fan-out)

This task must make the distinction between Java object [identity](../../../../glossary.md#term-reference-identity), [Model](../../../../glossary.md#term-model) [Tensor](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-identity-meanings), and
[compiler](../../../../glossary.md#term-compiler) [graph](../../../../glossary.md#graph) identity explicit:

- `sequence.cell()` returns one stable Java object and that cell owns one stable set of Parameter
  wrappers after binding.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- A current parameter wrapper's `value()` returns the same exact leaf Tensor reference for every
  step until caller-coordinated replacement/load changes it.
- Every call to `cell.forward` invokes public Tensor operations anew. Its derived results have
  fresh [`TensorId`](../../../../glossary.md#tensorid) values and distinct [`TensorProducer`](../../../../glossary.md#producer--tensorproducer) occurrences even when descriptors and
  parameter inputs are equal.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor))</span>
- A later step's compact carried state is an expression descended from the preceding step result;
  this establishes temporal dependency in the statically unrolled ancestry.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [unrolled glossary](../../../../glossary.md#backpropagation-through-time-and-recurrent-state))</span>
- Input/hidden parameter leaves occur as exact repeated producer-input references across time.
  They are not copied, relabeled, detached, or recreated per step.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [detached glossary](../../../../glossary.md#detached-model-results))</span>
- Current compiler autograd keys contribution lists by exact Tensor identity. If a downstream
  functional-gradient request selects one of these shared parameter leaves, contributions from
  every selected temporal route fan into that one target and are combined with ordinary public
  Tensor addition in the compiler's established deterministic traversal order.<span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [target glossary](../../../../glossary.md#term-target))</span>

[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) tests must inspect the Model provenance preconditions: distinct step output IDs/producers,
carried-state ancestry, and repeated exact parameter-[leaf](../../../../glossary.md#term-leaf) references/fan-out. They must not import
package-private compiler implementation, fabricate graph-local [`NodeId`](../../../../glossary.md#nodeid), claim numerical
[gradients](../../../../glossary.md#term-gradient), or add an NN backward method. Existing compiler and Training documentation supplies
the downstream exact-identity accumulation boundary; no compiler suite is rerun absent a concrete
compiler change.

## Affected files

Implementation is limited to these thirty-one paths:

1. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/initialization/LinearWeightInitialization.java` (delete)
2. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/initialization/ParameterInitialization.java` (new)
3. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/initialization/ParameterInitializers.java`
4. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/initialization/package-info.java`
5. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/Linear.java`
6. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/RnnCell.java`
7. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/GruCell.java`
8. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/LstmCell.java`
9. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/RnnSequence.java`
10. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/GruSequence.java`
11. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/LstmSequence.java`
12. `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/package-info.java`
13. `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/initialization/ParameterInitializationTest.java` (new)
14. `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/initialization/ParameterInitializersTest.java`
15. `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/LinearTest.java`
16. `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/LinearInitializationTest.java`
17. `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/module/ModelTest.java`
18. `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/module/StateDictionaryTest.java`
19. `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/RnnCellTest.java`
20. `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/RnnCellInitializationTest.java`
21. `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/GruCellTest.java`
22. `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/GruCellInitializationTest.java`
23. `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/LstmCellTest.java`
24. `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/LstmCellInitializationTest.java`
25. `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/RnnSequenceTest.java`
26. `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/GruSequenceTest.java`
27. `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/LstmSequenceTest.java`
28. `docs/api/training-api.md`
29. `docs/glossary.md`
30. `docs/planning/extensions/nn/master-plan.md`
31. `docs/planning/extensions/nn/tasks/0020-automatic-recurrent-initialization-and-sequence-defaults.md` (new)

Do not edit [`Module`](../../../../glossary.md#term-module), StateDictionary production, [Model](../../../../glossary.md#term-model)/[Topology](../../../../glossary.md#term-topology) source, existing sequence
packing tests/result carriers, [Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api), architecture files, ADRs/tests, global roadmap,
other master/task files, build files, Training Java source, [compiler](../../../../glossary.md#term-compiler)/[runtime](../../../../glossary.md#term-runtime)/[prepare](../../../../glossary.md#prepare)/[engine](../../../../glossary.md#term-engine),
[backends](../../../../glossary.md#backend), [conformance](../../../../glossary.md#term-conformance), or integration tests. If correct implementation requires any such path,
stop and report the exact blocker.

## Maximum [scope](../../../../glossary.md#term-scope)

This is an explicit cohesive exception to the usual twelve-to-eighteen-path [heuristic](../../../../glossary.md#term-heuristic). The three
cells share one public policy/[lifecycle](../../../../glossary.md#lifecycle) contract, the three containers must expose symmetric
defaults, and the current [Linear](../../../../glossary.md#term-linear-meanings) policy plus its repository-owned consumers must migrate in the
same [atomic](../../../../glossary.md#resource-transaction-and-rollback) change. Partial rollout would leave duplicate/incompatible policy names or make
matching recurrent families behave differently. The task may create, delete, or modify at most
the exact thirty-one paths above. Do not use this exception for ModuleFactory, [Embedding](../../../../glossary.md#embedding),
directionality, [scan](../../../../glossary.md#term-cumulative-scan), unrelated refactoring, or duplicated utility abstractions.

## Test requirements

### Public surface and compatibility

- Verify `ParameterInitialization` is final, immutable, closed, has exactly the eight named
  factories, <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>`requiresRandomGenerator`, value methods, and no public constructor/field/nested
  type/raw-kind/callback/implementation hook. Verify structural equality/hash behavior including
  configurable arguments and signed zero, deterministic diagnostic strings, and non-identity
  value semantics.<span data-glossary-navigation="true"> ([callback glossary](../../../../glossary.md#term-callback); [structural equality glossary](../../../../glossary.md#term-structural-equality); [hash glossary](../../../../glossary.md#term-hash); [arguments glossary](../../../../glossary.md#term-argument); [signed zero glossary](../../../../glossary.md#term-signed-zero); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Verify `LinearWeightInitialization` and `RecurrentWeightInitialization` are absent from source,
  compiled public classes, imports, constructor signatures, Javadocs, and current explanatory
  documentation. Completed task 0019 is the explicit historical exception and remains unchanged.
  Verify Linear's migrated automatic constructor accepts <span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [Linear glossary](../../../../glossary.md#term-linear-meanings))</span>`ParameterInitialization` and retains its
  other exact arguments; supplied/eager constructors remain unchanged.<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation))</span>
- Verify each cell and sequence remains final, extends [`Module`](../../../../glossary.md#term-module) directly, and declares exactly its
  prior public surface plus the planned constructor/overloads. Verify there is no `Lazy*`, public
  lifecycle/status/schema getter, generic recurrent base, <span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [schema glossary](../../../../glossary.md#term-schema))</span>`RandomGeneratorFactory` parameter, or
  unary/Sequential adapter.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [unary glossary](../../../../glossary.md#term-unary))</span>
- Re-run all existing supplied/eager construction, formula, packing, replacement, mode, result,
  and failure tests unchanged in meaning.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>

### Automatic initialization and deterministic policy

- Verify policy factories reject invalid configurable values immediately and in the documented
  order without Tensor, ID, storage, factory, or RNG effects: non-finite mean; non-finite then
  negative standard deviation; non-finite lower then upper bound; and non-increasing bounds.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [finite glossary](../../../../glossary.md#term-finite); [mean glossary](../../../../glossary.md#term-mean); [standard deviation glossary](../../../../glossary.md#term-standard-deviation))</span>
- Verify both generic `ParameterInitializers.initialize` dispatch overloads have the exact surface,
  null/wrong-overload validation order, and one-to-one storage/draw/failure behavior of all eight
  existing initializers. Cover floating types, fully static/dynamic/zero/scalar Shapes as relevant,
  fan presets' positive rank-two <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [scalar glossary](../../../../glossary.md#term-scalar); [Shapes glossary](../../../../glossary.md#shape); [rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>`[fanOut, fanIn]` limit, configurable normal/uniform values, and
  source failure/non-rollback behavior.<span data-glossary-navigation="true"> ([rollback glossary](../../../../glossary.md#term-rollback))</span>
- Verify Linear automatic initialization under all eight policies. Its four legacy presets retain
  exact values, normal/uniform/zero/one dispatch correctly, its factory/seed remain separate, and
  zero/one never call <span data-glossary-navigation="true"> ([Linear glossary](../../../../glossary.md#term-linear-meanings); [seed glossary](../../../../glossary.md#term-seed))</span>`factory.create` or consume a draw. Migrate Model/state-dictionary fixtures
  without changing their lifecycle meaning.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- For every cell family, both bias modes, representative policies from every policy category, and
  every floating parameter type,
  verify no Tensor/Parameter/generator is created at construction; access/discovery/export fails
  closed; first compatible forward derives exact input width; and complete parameters publish in
  input-weight, hidden-weight, optional-bias order before ordinary expression IDs.<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- Compare automatic [weight](../../../../glossary.md#term-weight) storage exactly with generic [dispatch](../../../../glossary.md#term-dispatch) driven by
  `RandomGeneratorFactory.of("L64X128MixRandom").create(seed)`. Verify the input matrix consumes
  the first draws, hidden matrix consumes the advanced stream, and bias consumes none. For fan
  presets, verify each complete input/hidden Shape derives its own fan values; for configurable
  and constant policies, verify exact arguments/values are applied independently to both Shapes.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [constant glossary](../../../../glossary.md#term-constant); [arguments glossary](../../../../glossary.md#term-argument))</span>
- Verify recurrent `zeros()` and `ones()` create no generator at all and optional bias still uses
  the layer-owned zero rule. Verify one shared immutable policy value can configure multiple
  layers without retaining Shape, source advancement, or layer state.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- Verify same configuration/seed produces equal represented parameter values in separate cells,
  distinct Tensor/Parameter identities, and no global/shared mutable generator state.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable))</span>
- Cover null/order, hidden-size, packed-size overflow, non-floating type, policy null, first input
  rank/final-Dimension/static/positive constraints, explicit hidden/cell mismatch, promotion/
  broadcast failure, Java-array limit, source/allocation/ID failure where controllable, retry, and
  no partial publication.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow); [rank glossary](../../../../glossary.md#term-rank); [Dimension glossary](../../../../glossary.md#dimension); [publication glossary](../../../../glossary.md#term-publication))</span>
- Verify two concurrent compatible first calls publish one complete group and both use it;
  incompatible/failing races observe only documented cell-local behavior.
- Verify strict load initializes each automatic cell and migrated automatic Linear without a
  forward call, generic-policy dispatch, factory call, or PRNG; rejects
  wrong path/kind/type/Shape/packing/gradient as applicable before publication, and round-trips
  dictionaries with equivalent eager/supplied cells. Keep this coverage in the three authorized
  initialization tests rather than widening <span data-glossary-navigation="true"> ([PRNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [gradient glossary](../../../../glossary.md#term-gradient); [eager glossary](../../../../glossary.md#term-eager-evaluation); [widening glossary](../../../../glossary.md#term-widening))</span>`StateDictionaryTest`.

### Sequence construction and defaults

- Verify each high-level sequence constructor creates exactly one matching automatic cell,
  registers it under `cell`, exposes the exact child, has no direct state, and preserves recursive
  paths/order after binding.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- Verify `forward(input)` derives all-valid lengths and one/two exact typed zero states;
  `forward(input, lengths)` derives only state; explicit-state overload without lengths derives
  only all-valid lengths; and the most-explicit overload preserves existing behavior.
- Verify zero states are fresh [eager](../../../../glossary.md#term-eager-evaluation) dense unlabeled [provenance](../../../../glossary.md#provenance)-free [leaves](../../../../glossary.md#term-leaf) with [Shape](../../../../glossary.md#shape)
  `[batch, hiddenSize]`, exact cell parameter type, false gradient eligibility, all-zero storage,
  and no module/state-dictionary registration. LSTM hidden/cell zeros are distinct.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [gradient glossary](../../../../glossary.md#term-gradient); [module glossary](../../../../glossary.md#term-module); [LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm))</span>
- Verify calls with varying compatible time/batch extents derive new arrays/states per call while
  the initialized input width remains fixed. Cover zero time, empty batch, all-zero lengths,
  length-array limits, state type promotion, and later input-width mismatch.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- Verify an explicit all-zero call invokes no cell and leaves a new automatic cell unbound;
  default-state variants return exact created zero references. A later non-empty call binds it.
- Reuse the existing packing tests as authoritative evidence that explicit lengths still omit
  padded rows, preserve stable order, restore exits, and never infer padding from zeros; do not
  edit those tests without a concrete regression.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [regression glossary](../../../../glossary.md#term-regression))</span>

### Static-unroll/[BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt) [provenance](../../../../glossary.md#provenance) preconditions

- On at least two represented steps per family, verify packed outputs have distinct [`TensorId`](../../../../glossary.md#tensorid)
  values and distinct producer identities.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer))</span>
- Traverse public provenance to verify the second step depends on the exact first-step carried
  output path and that every step's projection ancestry retains the exact current input- and
  hidden-weight Tensor leaves.<span data-glossary-navigation="true"> ([leaves glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([weight glossary](../../../../glossary.md#term-weight); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Verify those parameter leaves are one identity with multiple producer-input edges, not one
  cloned leaf per step, and that replacing parameters affects only a later unroll.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Verify all relevant derived descriptors retain gradient eligibility from the shared parameter
  leaves. Do not calculate or assert numeric gradients in NN tests.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [gradient glossary](../../../../glossary.md#term-gradient); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>

### Exclusions

- Scan source/public bytecode/imports for forbidden Lazy/factory/direction/scan/mask/runtime/
  compiler/training APIs, retained default states/lengths/generators, and dependencies outside
  Model.<span data-glossary-navigation="true"> ([Scan glossary](../../../../glossary.md#term-cumulative-scan); [bytecode glossary](../../../../glossary.md#term-bytecode); [Lazy glossary](../../../../glossary.md#term-lazy-evaluation); [mask glossary](../../../../glossary.md#term-boolean-mask); [runtime glossary](../../../../glossary.md#term-runtime); [compiler glossary](../../../../glossary.md#term-compiler); [Model glossary](../../../../glossary.md#term-model))</span>
- Verify no result carrier component, gate formula, parameter name/order, bias packing, sequence
  packing path, or static <span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [parameter glossary](../../../../glossary.md#term-parameter); [bias glossary](../../../../glossary.md#term-bias))</span>`long[]` meaning changed.

## Documentation requirements

The implementation agent may draft detailed Javadocs while coding. After executable Java and the
one authoritative [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) test [run](../../../../glossary.md#term-run) stabilize, a separate clean documentation-focused agent/thread must
read the final diff, source/tests, generated Javadocs, architecture/[planning](../../../../glossary.md#term-planning) contracts,
documentation rules, General/[API](../../../../glossary.md#term-application-programming-interface-api)-[Javadoc](../../../../glossary.md#term-javadoc)/Planning/Example profiles, completed NN 0012–0019,
relevant [Tensor](../../../../glossary.md#tensor) [producer](../../../../glossary.md#term-producer)/factory/[Shape](../../../../glossary.md#shape)/promotion contracts, [compiler](../../../../glossary.md#term-compiler) exact-[identity](../../../../glossary.md#term-identity-meanings) [autograd](../../../../glossary.md#autograd)
contract, Training API, glossary, and [Checkpoint](../../../../glossary.md#term-checkpoint)/Data/Text plans. It must finalize:

- all affected policy value/dispatcher, Linear, cell, sequence, constructor, accessor, overload,
  parameter, return, failure, lifecycle, ownership, concurrency, and side-effect Javadocs;<span data-glossary-navigation="true"> ([Linear glossary](../../../../glossary.md#term-linear-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- initialization/package Javadocs for the deliberate Linear-policy migration, eight general
  algorithms, immediate configurable-value validation, exact value semantics, dispatch overloads,
  Shape/fan boundary, zero/one no-RNG path, exact standard recurrent PRNG, explicit separate
  seed/type/hidden facts, caller-owned advanced constructors, and no global source;<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [dispatch glossary](../../../../glossary.md#term-dispatch); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [PRNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [seed glossary](../../../../glossary.md#term-seed))</span>
- layers/package Javadocs distinguishing one cell instance from fresh time-step Tensor
  occurrences, default explicit zero-state construction, static all-valid lengths, static packing,
  and future runtime scan/direction contracts;<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- Training API example and explanation of automatic cell/sequence composition, continued explicit
  state, state/load behavior, static-unroll provenance, shared parameter fan-out, compiler-owned
  BPTT construction, and current execution limits;<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [fan-out glossary](../../../../glossary.md#term-fan-out); [BPTT glossary](../../../../glossary.md#term-backpropagation-through-time-bptt))</span>
- glossary entries for general parameter-initialization policy, recurrent automatic
  initialization, default recurrent state, all-valid static sequence, and static-unroll parameter
  sharing without redefining compiler autograd; and<span data-glossary-navigation="true"> ([recurrent state glossary](../../../../glossary.md#backpropagation-through-time-and-recurrent-state))</span>
- final planning evidence/status, links/anchors/fences, exact scope, and reasoned no-change
  conclusions.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>

The documentation pass reuses stable Java test evidence unless it changes executable Java. It
must record explicit no-change conclusions for `ARCHITECTURE.md`, current architecture plan, ADRs,
architecture tests, Tensor API, [Compile](../../../../glossary.md#compile) API, [Model](../../../../glossary.md#term-model) capabilities/source, [Module](../../../../glossary.md#term-module)/[Parameter](../../../../glossary.md#term-parameter)/[Buffer](../../../../glossary.md#term-buffer)/
StateDictionary production, Training Java API, Checkpoint/Data/Text/Vision plans, [Gradle](../../../../glossary.md#term-gradle), Compiler/
[Runtime](../../../../glossary.md#term-runtime)/[Prepare](../../../../glossary.md#prepare)/[Engine](../../../../glossary.md#term-engine)/[backends](../../../../glossary.md#backend), [conformance](../../../../glossary.md#term-conformance)/integration, sequence result/packing tests, global
roadmap, active [CPU](../../../../glossary.md#term-cpu) work, and every other module. `ModelTest` and `StateDictionaryTest` are changed
only to compile against the intentional current-API rename and retain their behavioral coverage.

## Acceptance criteria

- One [immutable](../../../../glossary.md#term-immutable) closed `ParameterInitialization` value covers the eight required algorithms with
  exact validation/equality semantics and generic dispatch through existing<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
  `ParameterInitializers`; it owns no layer [schema](../../../../glossary.md#term-schema), [Tensor](../../../../glossary.md#tensor), [callback](../../../../glossary.md#term-callback), [RNG](../../../../glossary.md#graph-rng-state--graphrngstate), [seed](../../../../glossary.md#term-seed), or [mutable](../../../../glossary.md#term-mutable) state.
- The current automatic [Linear](../../../../glossary.md#term-linear-meanings) [API](../../../../glossary.md#term-application-programming-interface-api) and all repository-owned consumers migrate atomically from
  `LinearWeightInitialization`; neither that legacy type nor a recurrent-specific duplicate/alias
  remains.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- All three existing final cells support automatic first-forward input-width inference through
  one non-Lazy constructor with explicit hidden size, bias, type, general policy, and seed.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [Lazy glossary](../../../../glossary.md#term-lazy-evaluation); [bias glossary](../../../../glossary.md#term-bias))</span>
- High-level automatic recurrent construction exposes no `RandomGeneratorFactory`; it uses exact
  deterministic `L64X128MixRandom` per random attempt and no RNG for zero/one, while existing
  caller-owned `RandomGenerator` [eager](../../../../glossary.md#term-eager-evaluation) constructors remain.
- Parameter reservation, publication, retry, strict load, discovery/export gating, failure, and
  concurrency behavior matches the proven NN 0019 Module lifecycle for each complete cell group.<span data-glossary-navigation="true"> ([Parameter glossary](../../../../glossary.md#term-parameter); [publication glossary](../../../../glossary.md#term-publication); [concurrency glossary](../../../../glossary.md#term-concurrency); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Module glossary](../../../../glossary.md#term-module); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- All three existing final sequences can construct their own matching standard cell and provide
  zero-state and all-valid conveniences without changing their advanced explicit contracts.
- Default state derives batch from static input and hidden size/type from cell schema, is explicit
  non-gradient Tensor data for that call, and is never retained or registered.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Static `long[]` packing remains the honest compatibility contract; runtime lengths/masks and
  scan remain Draft.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [masks glossary](../../../../glossary.md#term-boolean-mask); [scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- One cell/parameter set is shared across time, every step builds fresh Tensor identities/
  producers, temporal state ancestry is connected, and repeated exact parameter leaves form the
  Model provenance fan-out consumed by existing compiler autograd.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings); [producers glossary](../../../../glossary.md#term-producer); [leaves glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [provenance glossary](../../../../glossary.md#provenance); [fan-out glossary](../../../../glossary.md#term-fan-out); [compiler glossary](../../../../glossary.md#term-compiler); [autograd glossary](../../../../glossary.md#autograd))</span>
- No NN backward/runtime/execution claim, generic recurrent base, ModuleFactory, initialized
  Embedding, bidirectionality, new dependency, architecture change, or unrelated refactor lands.<span data-glossary-navigation="true"> ([Embedding glossary](../../../../glossary.md#embedding); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Focused tests, one final NN suite, NN Javadoc, public-surface/external example, dependency/import,
  Markdown, exact-scope/frontier/status, newline, whitespace, and diff gates pass.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation review is completed in a separate clean context and the task becomes `Complete`
  only after its final evidence and completion summary are filled.

## Tests / [validation](../../../../glossary.md#term-validation)

The implementation task must [run](../../../../glossary.md#term-run) these tiers once after executable stabilization:

1. Focused recurrent selection:

   ```text
   ./gradlew :extensions:nn:test \
     --tests io.github.pho001.synaptik.nn.initialization.ParameterInitializationTest \
     --tests io.github.pho001.synaptik.nn.initialization.ParameterInitializersTest \
     --tests io.github.pho001.synaptik.nn.layers.LinearTest \
     --tests io.github.pho001.synaptik.nn.layers.LinearInitializationTest \
     --tests io.github.pho001.synaptik.nn.layers.RnnCellTest \
     --tests io.github.pho001.synaptik.nn.layers.RnnCellInitializationTest \
     --tests io.github.pho001.synaptik.nn.layers.GruCellTest \
     --tests io.github.pho001.synaptik.nn.layers.GruCellInitializationTest \
     --tests io.github.pho001.synaptik.nn.layers.LstmCellTest \
     --tests io.github.pho001.synaptik.nn.layers.LstmCellInitializationTest \
     --tests io.github.pho001.synaptik.nn.layers.RnnSequenceTest \
     --tests io.github.pho001.synaptik.nn.layers.GruSequenceTest \
     --tests io.github.pho001.synaptik.nn.layers.LstmSequenceTest \
     --tests io.github.pho001.synaptik.nn.layers.RnnSequencePackingTest \
     --tests io.github.pho001.synaptik.nn.layers.GruSequencePackingTest \
     --tests io.github.pho001.synaptik.nn.layers.LstmSequencePackingTest \
     --tests io.github.pho001.synaptik.nn.module.ModelTest \
     --tests io.github.pho001.synaptik.nn.module.StateDictionaryTest
   ```

2. One authoritative [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [module](../../../../glossary.md#term-module) suite after all executable changes:

   ```text
   ./gradlew :extensions:nn:test
   ```

3. After the clean documentation pass:

   ```text
   ./gradlew :extensions:nn:javadoc
   ```

4. Inspect generated Javadocs for `ParameterInitialization`, `ParameterInitializers`, Linear, all
   six cell/sequence types, and both package pages. Use <span data-glossary-navigation="true"> ([Linear glossary](../../../../glossary.md#term-linear-meanings))</span>`javap`/reflection plus a standalone
   external-package `Model.define` example to verify the exact public surface, absence of both
   forbidden legacy/duplicate policy classes, and Java overload usability.
5. Verify NN production imports only Model/JDK APIs and contains no public Lazy/build/bind/init/
   status, generic recurrent, ModuleFactory, direction, mask, runtime scan, compiler/training, or
   retained RNG/default-state mechanism.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [Lazy glossary](../../../../glossary.md#term-lazy-evaluation); [mask glossary](../../../../glossary.md#term-boolean-mask); [runtime glossary](../../../../glossary.md#term-runtime); [scan glossary](../../../../glossary.md#term-cumulative-scan); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
6. Run Markdown local-link/anchor/unique-heading/fence/newline/trailing-whitespace checks for the
   two changed planning files and final docs. Verify exactly one NN task/master row is <span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>[`Ready`](../../../../glossary.md#term-ready)
   during planning; on completion, both task/master are `Complete` and NN 0020A–0020C plus
   0021–0024 remain concise `Draft` rows without task files.
7. Verify exact thirty-one-path scope by subtracting the pre-existing shared CPU/roadmap worktree
   snapshot. Do not modify, stage, revert, or include those unrelated changes. Run<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [CPU glossary](../../../../glossary.md#term-cpu); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
   `git diff --check`; use a no-index whitespace check for this initially untracked task file.

Repository-wide tests are not required because this task changes one extension and no [dependency](../../../../glossary.md#term-dependency),
architecture, build, [compiler](../../../../glossary.md#term-compiler), execution, or [backend](../../../../glossary.md#backend) contract. Escalate to the root suite only if
the final diff unexpectedly crosses one of those boundaries.

## Dependencies

- [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0001–0019 are Complete.
- Existing supplied/eager RNN, GRU, and LSTM cell contracts, their concrete sequence/result APIs,
  static packing semantics, typed functional Model, and Module ownership/state paths are stable.<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn); [GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru); [LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm); [Model glossary](../../../../glossary.md#term-model); [Module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- NN 0019's private reservation/publication, fail-closed discovery/export, strict-load binding,
  retry, and release/acquire completion contract is implemented and reusable unchanged.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [binding glossary](../../../../glossary.md#term-binding); [release glossary](../../../../glossary.md#term-release))</span>
- Model provides eager typed zeros/random initialization, immutable Tensor identity/provenance,
  static Shape/Dimension facts, promotion/broadcast, and every primitive operation used by the
  existing cell/sequence formulas.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [Tensor glossary](../../../../glossary.md#tensor); [identity glossary](../../../../glossary.md#term-identity-meanings); [provenance glossary](../../../../glossary.md#provenance); [Shape glossary](../../../../glossary.md#shape); [Dimension glossary](../../../../glossary.md#dimension); [operation glossary](../../../../glossary.md#operation))</span>
- Compiler exact-identity reverse traversal and contribution accumulation are current downstream
  contracts. This task changes only the forward expression topology they consume.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [topology glossary](../../../../glossary.md#term-topology))</span>
- Training Java remains a marker. Its API document may explain the current compiler handoff, but
  no executable Training workflow is needed.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [handoff glossary](../../../../glossary.md#term-handoff))</span>
- The user-authorized NN interleave remains isolated from the active CPU frontier. Pre-existing
  CPU and roadmap worktree changes belong to another task and must remain untouched.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>

## Follow-up tasks

- [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0020A: initialized [Embedding](../../../../glossary.md#embedding) with explicit vocabulary size, embedding size, [data type](../../../../glossary.md#data-type--datatype),
  `ParameterInitialization`, seed, and the same documented standard PRNG for random policies; no
  vocabulary inference from token IDs. Select how an optional padding row is initialized and kept
  invariant (including future update behavior) before Ready.<span data-glossary-navigation="true"> ([PRNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [inference glossary](../../../../glossary.md#term-inference); [token glossary](../../../../glossary.md#term-token); [padding glossary](../../../../glossary.md#padding); [invariant glossary](../../../../glossary.md#term-invariant); [Ready glossary](../../../../glossary.md#term-ready))</span>
- NN 0020B: stateless standard `ModuleFactory` construction recipes after recurrent and Embedding
  initialization contracts are stable. `embedding`, [`linear`](../../../../glossary.md#term-linear-meanings), `rnn`, `gru`, and `lstm` return
  concrete types; recurrent recipes hide standard Cell+Sequence assembly. Every recipe takes
  explicit per-layer data type, `ParameterInitialization`, and [seed](../../../../glossary.md#term-seed). `ModuleFactory.standard()`
  owns no module registration/configuration/RNG/seed manager or state; <span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>`Topology.addModule` keeps
  sole ownership, and advanced direct constructors retain caller-owned generators/factories.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- NN 0020C: type-safe bidirectional/multidirectional recurrent composition with independent
  directional cells/parameters/seeds, valid-prefix reverse traversal, an explicit <span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>`CONCAT`/`SUM`
  choice or deliberately narrower first merge policy, and type-specific final states.
- NN 0021: coordinate genuine recurrent scan/control-flow and runtime input binding across owning
  modules without specializing topology to one runtime length vector.<span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [scan glossary](../../../../glossary.md#term-cumulative-scan); [binding glossary](../../../../glossary.md#term-binding); [topology glossary](../../../../glossary.md#term-topology); [vector glossary](../../../../glossary.md#term-vector))</span>
- NN 0022: consume Data-owned [runtime](../../../../glossary.md#term-runtime) valid lengths only after NN 0021.
- NN 0023: select arbitrary [mask](../../../../glossary.md#term-boolean-mask)-with-holes semantics only for a concrete consumer.
- NN 0024: [run](../../../../glossary.md#term-run) the typed [Model](../../../../glossary.md#term-model)/recurrent/Data/training/[checkpoint](../../../../glossary.md#term-checkpoint) integration checkpoint.

## Architecture impact

Expected impact: None.

The architecture already assigns neural-network [parameters](../../../../glossary.md#term-parameter), [module](../../../../glossary.md#term-module) [ownership](../../../../glossary.md#term-ownership-meanings), layers, stateful
composition, and forward conveniences to `extensions/nn`; generic [Tensor](../../../../glossary.md#tensor) construction to [Model](../../../../glossary.md#term-model);
and [automatic differentiation](../../../../glossary.md#autograd) to [Compiler](../../../../glossary.md#term-compiler). This task reuses those boundaries and adds no
[dependency](../../../../glossary.md#term-dependency). Stop if implementation requires an architecture, module-boundary, build, or
dependency change.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are implementing Synaptik NN task 0020. Do not use GSD. Do not commit or push unless the user
explicitly requests that after the complete reviewed change.

Read AGENTS.md, ARCHITECTURE.md, docs/architecture/current-architecture-plan.md,
docs/planning/planning-guide.md, docs/planning/roadmap.md,
docs/planning/extensions/nn/master-plan.md, and
docs/planning/extensions/nn/tasks/0020-automatic-recurrent-initialization-and-sequence-defaults.md
in full. Read completed NN 0012–0019, final Module reservation/state/Model contracts, all six
current recurrent cell/sequence types and result APIs/tests, initializer APIs, relevant Model
Tensor factory/identity/provenance/Shape/promotion contracts, compiler exact-identity autograd,
Training API, glossary, and documentation rules/profiles named by the task.

Implement task 0020 exactly within its thirty-one authorized paths. Atomically replace the recent
LinearWeightInitialization API with the exact closed immutable ParameterInitialization value and
generic ParameterInitializers dispatch specified by the task; leave completed task 0019 unchanged
as historical evidence and leave no alias or recurrent-specific duplicate. Preserve one final type
per cell/sequence, supplied/eager compatibility, current gate equations/packing/results, static
long[] packing, Model-only dependency, and pre-existing CPU/roadmap worktree changes. Use exactly
L64X128MixRandom plus explicit per-layer seed for random high-level recurrent policies and no RNG
for zero/one. Do not add callbacks, public fan abstractions, Lazy types, a public lifecycle/schema/
status, ModuleFactory, initialized Embedding, directionality, runtime scan/mask, a generic
recurrent base, compiler/training imports, or execution claims. Stop and report architecture,
lifecycle, public-API, JMM, overload, provenance, or exact-scope uncertainty instead of inventing
a design.

Run the focused selection and one authoritative NN suite after executable Java stabilizes. Then
hand the exact diff/evidence to a separate clean documentation-focused agent/thread. That pass
must independently finalize affected Javadocs, package docs, Training API, glossary, master/task
evidence, generated Javadoc, examples, links, no-change reasoning, exact scope, and status. Do not
mark Complete until both passes and all gates succeed.
```

## Documentation-agent [handoff](../../../../glossary.md#term-handoff)

Give the documentation-focused agent this task, final diff, focused/final Java evidence, exact
public [signatures](../../../../glossary.md#term-signature), breaking policy migration/value/[validation](../../../../glossary.md#term-validation)/[dispatch](../../../../glossary.md#term-dispatch) semantics, policy/[PRNG](../../../../glossary.md#graph-rng-state--graphrngstate)
selection, cell initialization/[publication](../../../../glossary.md#term-publication) order, retry/load/[concurrency](../../../../glossary.md#term-concurrency) effects, sequence
overload/default-state semantics, and [provenance](../../../../glossary.md#provenance) test evidence. Identify whether executable Java
changed after the authoritative suite and enumerate all thirty-one authorized paths plus the
pre-existing [CPU](../../../../glossary.md#term-cpu)/roadmap paths to ignore.

The documentation agent independently reads the architecture/documentation profiles, final
source/tests/generated Javadocs, relevant completed tasks, [Tensor](../../../../glossary.md#tensor)/[Compiler](../../../../glossary.md#term-compiler)/Training contracts,
glossary, and future [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) rows. It must distinguish [parameter](../../../../glossary.md#term-parameter)-[leaf](../../../../glossary.md#term-leaf) initialization from forward
expression construction, one Java cell from multiple fresh time-step occurrences, static unroll
from [runtime](../../../../glossary.md#term-runtime) [scan](../../../../glossary.md#term-cumulative-scan), [Model](../../../../glossary.md#term-model) provenance [fan-out](../../../../glossary.md#term-fan-out) from generated [gradients](../../../../glossary.md#term-gradient), and current compiler
[autograd](../../../../glossary.md#autograd) from an executable Training workflow. It reuses stable test evidence unless executable
behavior changes and records all final no-change conclusions.

## Local decisions

- Use one existing final cell type per family. Input width is a batch-derived binding fact; hidden
  width, gate semantics, bias, parameter type, policy, and seed remain architecture choices.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [binding glossary](../../../../glossary.md#term-binding); [bias glossary](../../../../glossary.md#term-bias); [parameter glossary](../../../../glossary.md#term-parameter); [seed glossary](../../../../glossary.md#term-seed))</span>
- Replace the new Linear-specific enum early with one closed immutable parameter-policy value.
  Keeping aliases or adding another recurrent enum would duplicate the same algorithm choice
  before the public API is stable. Layer code still owns Shapes, fans implied by those Shapes,
  parameter order, gate schema, and bias rules.<span data-glossary-navigation="true"> ([Linear glossary](../../../../glossary.md#term-linear-meanings); [immutable glossary](../../../../glossary.md#term-immutable); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Shapes glossary](../../../../glossary.md#shape); [schema glossary](../../../../glossary.md#term-schema))</span>
- Expose all four existing fan presets plus configured normal/uniform and constant zero/one because
  Linear, recurrent matrices, and immediate initialized Embedding are concrete consumers of the
  same algorithms. Selection remains explicit; Kaiming/ReLU names are not implied recurrent
  defaults. Defer convolution fan schema and orthogonal/per-gate policies.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [matrices glossary](../../../../glossary.md#term-matrix-meanings); [Embedding glossary](../../../../glossary.md#embedding); [Kaiming glossary](../../../../glossary.md#term-kaiming); [ReLU glossary](../../../../glossary.md#term-relu); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers))</span>
- Make `L64X128MixRandom` the exact ordinary high-level PRNG so users supply one seed per layer,
  not a factory. Create it only for random policies; zero/one stay genuinely RNG-free. Preserve
  direct constructors as the advanced escape hatch for caller-owned random algorithms.<span data-glossary-navigation="true"> ([PRNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- Use the cell's declared parameter type for default zero state. Input type remains independently
  governed by existing promotion and no hidden cast occurs.
- Provide both all-valid and zero-state overloads. `forward(input)` is the concise common case;
  explicit state remains available for streaming/truncated recurrence and explicit lengths remain
  available for static right padding.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding))</span>
- Keep explicit all-zero packing honest: no represented step means no cell call and therefore no
  automatic binding. Strict load or a later non-empty call supplies the first binding event.
- Share one cell and parameter set across time. Multiple Java cell instances would incorrectly
  create independent parameter owners; fresh public Tensor calls already provide the distinct
  expression occurrences required by static unroll and compiler reverse traversal.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [compiler glossary](../../../../glossary.md#term-compiler))</span>
- Keep future Embedding initialization, construction factory, bidirectionality, and runtime scan
  separate because each needs an independent policy or cross-module contract.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [scan glossary](../../../../glossary.md#term-cumulative-scan); [module glossary](../../../../glossary.md#term-module))</span>

## Known limitations

- Automatic binding still requires a positive static final input Dimension on the first
  represented cell path or a compatible complete strict dictionary. It is not runtime shape
  binding.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [Dimension glossary](../../../../glossary.md#dimension); [shape glossary](../../../../glossary.md#shape))</span>
- A static sequence requires fully static time/batch/input Shape and Java lengths. Default
  all-valid lengths do not relax this requirement.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- All-zero/zero-time static traversal can return default or explicit state without binding an
  automatic cell. Complete parameter discovery/export remains unavailable until later binding.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter))</span>
- Default state is eager host-backed data allocated per call. It is not a symbolic runtime input,
  cached constant, trainable initial state, or backend-resident value.<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [host glossary](../../../../glossary.md#term-host); [constant glossary](../../../../glossary.md#term-constant); [backend glossary](../../../../glossary.md#backend))</span>
- One seed governs both matrices in documented draw order. There is no automatic seed derivation,
  per-gate source, global source, or persisted RNG state.<span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed); [matrices glossary](../../../../glossary.md#term-matrix-meanings); [RNG state glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- Generic fan presets currently accept only complete positive [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `[fanOut, fanIn]` Shapes.
  There is no public convolution fan abstraction until a concrete convolution layer requires one.<span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers))</span>
- Initialization atomicity is cell-local. A functional model or multi-step sequence is not one
  transaction, and Tensor IDs/allocations are not rolled back.<span data-glossary-navigation="true"> ([atomicity glossary](../../../../glossary.md#term-atomicity); [model glossary](../../../../glossary.md#term-model); [transaction glossary](../../../../glossary.md#term-transaction); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Static provenance supplies a valid differentiable expression topology, but this task does not
  expose public compilation/training execution, calculate a gradient, prove backend coverage, or
  benchmark BPTT.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [differentiable glossary](../../../../glossary.md#derivative-and-chain-rule); [topology glossary](../../../../glossary.md#term-topology); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [gradient glossary](../../../../glossary.md#term-gradient); [BPTT glossary](../../../../glossary.md#term-backpropagation-through-time-bptt))</span>
- Bidirectional/multidirectional composition and [runtime](../../../../glossary.md#term-runtime) valid-length [scan](../../../../glossary.md#term-cumulative-scan) remain Draft follow-ups.

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning)-only work completed in clean planning context `/root/nn_0020_planning`; implementation,
[commit](../../../../glossary.md#term-commit), and push were not performed.

- Read the full repository instructions, architecture contract/current plan, planning guide/
  roadmap, documentation rules and General/API-Javadoc/Planning/Example profiles, NN master plan,
  completed NN 0012–0019, current recurrent cells/sequences/results/tests, Module reservation and
  strict-load lifecycle, complete <span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Module glossary](../../../../glossary.md#term-module); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>`ParameterInitializers`, `LinearWeightInitialization`, Linear
  automatic source/Javadocs and relevant initializer/Linear/Model/state-dictionary tests,
  Model/Topology, relevant Model Tensor/Shape/promotion/provenance APIs, compiler exact-identity
  autograd, Training marker/API, and glossary. No architecture, dependency, ownership, or
  compiler-boundary conflict was found.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Topology glossary](../../../../glossary.md#term-topology); [Tensor glossary](../../../../glossary.md#tensor); [Shape glossary](../../../../glossary.md#shape); [provenance glossary](../../../../glossary.md#provenance); [compiler glossary](../../../../glossary.md#term-compiler); [identity glossary](../../../../glossary.md#term-identity-meanings); [autograd glossary](../../../../glossary.md#autograd); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Replanned the initializer surface after the user selected an early atomic migration: one final
  immutable closed <span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [immutable glossary](../../../../glossary.md#term-immutable))</span>`ParameterInitialization`, eight named factories, exact value/configuration
  validation, two generic `ParameterInitializers` dispatch paths, no legacy alias or recurrent
  duplicate, and no public callback/fan abstraction. Completed task 0019 was not edited.<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [callback glossary](../../../../glossary.md#term-callback))</span>
- Cross-reference review found historical completion-era NN 0018/0019 follow-up text plus the
  current NN master/roadmap references. Established NN 0021–0024 IDs remain unchanged; 0020A,
  0020B, and 0020C use the planning guide's existing alphanumeric convention and have no detailed
  task files. Historical completed-task evidence was not rewritten.
- A targeted Ruby Markdown validator passed for the NN master and this task: all local link
  targets and heading anchors resolve, effective headings are unique, backtick/tilde fences are
  balanced, terminal newlines are present, and no trailing whitespace exists.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>
- Frontier checks passed: exactly one [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) task file has [scalar](../../../../glossary.md#term-scalar) status [`Ready`](../../../../glossary.md#term-ready), exactly one NN
  master row is `Ready`, task 0020 is linked from that row, and no 0020A–0020C or 0021–0024 task
  specification exists.
- Exact NN-owned planning scope passed: after subtracting concurrent CPU/backend-guide/roadmap
  work, the only NN paths are the modified master plan and this new task file. The global roadmap
  was reread for relevant numbering but received no hunk; all warned or newly appearing CPU-side
  paths were treated as externally owned and were never edited, staged, reverted, or included.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend))</span>
- `git diff --check` passed for the shared worktree at validation time. The no-index check for this
  new task returned the expected difference status with no whitespace diagnostic.
- No Java, Javadoc, Gradle, architecture, compiler, backend, conformance, integration, module, or
  repository-wide test was run because this is a planning-only change with no executable or
  authoritative-architecture behavior.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [run glossary](../../../../glossary.md#term-run))</span>
- Clean implementation context `/root/nn_0020_implementation` replaced the Linear-only public
  policy with the exact closed immutable `ParameterInitialization`, added both generic dispatcher
  overloads, migrated automatic Linear, and implemented the automatic reservation/binding/load
  lifecycle plus standard/default sequence surface for all three recurrent families. It changed
  no Model, Module, compiler, runtime, backend, build, architecture, or completed-task-0019 path.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Focused initializer/[Linear](../../../../glossary.md#term-linear-meanings), recurrent-cell initialization and strict-load, deterministic
  `L64X128MixRandom` stream, sequence default/preflight, all-zero skip, replacement, and provenance
  selections passed after stabilization. The first development-wide NN run reported eight
  expected exact-surface assertions against the newly added APIs; the tests were migrated and no
  production defect remained from that run.
- After the final default-state effect-order [preflight](../../../../glossary.md#term-preflight), the authoritative clean command
  `./gradlew :extensions:nn:clean :extensions:nn:test :extensions:nn:javadoc` passed. Final XML
  inspection reports 249 tests, zero skips, zero failures, and zero errors; executable Java and
  tests did not change afterward.
- Clean output contains no compiled `LinearWeightInitialization` or recurrent-specific [alias](../../../../glossary.md#aliasing-and-arena).
  `javap -public/-private`, repository reflection tests, and a standalone external-use compile
  confirmed the intended policy, dispatcher, migrated Linear, cell, and sequence surfaces and no
  retained recurrent generator/factory. Production import/dependency inspection retains only the
  existing Model dependency. <span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile))</span>`git diff --check` passed.
- Exact NN implementation scope contains only authorized affected paths. Concurrent CPU source,
  tests, backend planning, and global-roadmap changes remained present and untouched.
- Independent clean documentation context `/root/nn_0020_docs` reread the architecture,
  documentation profiles, final source/tests/diff, recurrent/initializer/state/provenance
  contracts, Training API, glossary, and planning records. It found no executable, API,
  architecture, dependency, or scope defect and changed no executable Java or test.
- The documentation pass finalized `ParameterInitialization`, dispatcher/package, Linear, all six
  cell/sequence, and layers-package Javadocs. It documented policy-versus-layer ownership,
  automatic binding/strict-load/retry/concurrency, exact random/zero-one routing, default and
  all-valid state effects, all-zero no-bind behavior, explicit-state identity, and shared-parameter
  fresh-producer static provenance without claiming numerical execution or public training.<span data-glossary-navigation="true"> ([concurrency glossary](../../../../glossary.md#term-concurrency); [parameter glossary](../../../../glossary.md#term-parameter); [producer glossary](../../../../glossary.md#term-producer))</span>
- Training API and glossary now use the common policy, describe automatic recurrent lifecycle,
  default state/all-valid overloads, and identify the existing Compiler exact-identity fan-out
  boundary. They keep runtime scan, Tensor <span data-glossary-navigation="true"> ([fan-out glossary](../../../../glossary.md#term-fan-out); [scan glossary](../../../../glossary.md#term-cumulative-scan))</span>`validLengths`, bidirectionality, `ModuleFactory`, and
  initialized Embedding as future work. Completed task 0019 remained unchanged.<span data-glossary-navigation="true"> ([Embedding glossary](../../../../glossary.md#embedding))</span>
- Final `./gradlew :extensions:nn:javadoc` passed without warnings after documentation-only edits.
  Generated pages were inspected for the policy, dispatcher, exact PRNG/strict-load lifecycle,
  length snapshot, default-state, all-zero, and shared-leaf/fresh-producer contracts. Java tests
  were not repeated because executable Java/tests were unchanged after the authoritative run.<span data-glossary-navigation="true"> ([PRNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [snapshot glossary](../../../../glossary.md#term-snapshot); [leaf glossary](../../../../glossary.md#term-leaf))</span>
- Final `javap -public/-private`, repository reflection evidence, standalone external-use
  compilation, compiled/source legacy-type absence, recurrent retained-field, production import,
  and Gradle dependency checks passed. A targeted Markdown validator passed local targets,
  anchors, unique headings, fences, terminal newlines, and trailing whitespace for the two final
  docs and two planning records.<span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>
- `ARCHITECTURE.md`, the current architecture plan, ADRs, and architecture tests correctly require
  no change because the task adds no module, dependency edge, or ownership rule. Tensor and
  Compile APIs require no change because existing Model operations/provenance and Compiler
  exact-identity accumulation already own the consumed boundary. Model, Compiler, Runtime,
  Prepare, Engine, backends, and Training Java require no change because NN only constructs
  existing expressions and exposes neither execution nor training behavior.<span data-glossary-navigation="true"> ([dependency edge glossary](../../../../glossary.md#term-dependency-edge); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Build/[dependency](../../../../glossary.md#term-dependency) rules are unchanged (`extensions/nn` still imports only Model). Backend
  conformance and integration tests require no change because no backend or end-to-end executable
  contract changed. Existing result carriers and packing semantics remain exact because the new
  overloads converge on the existing most-explicit implementations. Other modules, completed task
  0019, and the global roadmap require no task-owned edit; the concurrent CPU/backend/planning and
  global-roadmap paths were preserved exactly.
- Exact final NN scope is the authorized thirty-one paths, including the deleted legacy type, new
  policy/test/task, exactly Training API plus glossary, and no index entry. NN 0020 is Complete in
  task/master; no NN task is Ready; NN 0020A–0020C and 0021–0024 remain Draft without task files.
  Terminal-newline, trailing-whitespace, no-index, and `git diff --check` gates passed. Concurrent
  CPU/backend/planning/global-roadmap paths remained present, separately owned, and untouched.

## Implementation notes

Implementation in clean context `/root/nn_0020_implementation` added the closed
`ParameterInitialization` value and dispatchers, migrated automatic [Linear](../../../../glossary.md#term-linear-meanings), added [atomic](../../../../glossary.md#resource-transaction-and-rollback) automatic
[binding](../../../../glossary.md#term-binding) to all three recurrent cells, and added the standard-cell/default-state/all-valid sequence
surface. Focused initializer, Linear, [Module](../../../../glossary.md#term-module)/state, cell, sequence, strict-load,
deterministic-stream, and [provenance](../../../../glossary.md#provenance) tests are implemented. Independent clean documentation
context `/root/nn_0020_docs` finalized every authorized [Javadoc](../../../../glossary.md#term-javadoc), explanatory-documentation,
glossary, and [planning](../../../../glossary.md#term-planning) impact without changing executable code.

## Completion summary

- Completed changes: The common closed eight-policy value and exact dispatch routes, Linear
  migration, automatic recurrent binding/retry/strict-load lifecycle, standard sequence
  construction/defaults, and shared-parameter fresh-producer static provenance are implemented,
  tested, and documented.<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [Linear glossary](../../../../glossary.md#term-linear-meanings); [binding glossary](../../../../glossary.md#term-binding); [lifecycle glossary](../../../../glossary.md#lifecycle); [parameter glossary](../../../../glossary.md#term-parameter); [producer glossary](../../../../glossary.md#term-producer); [provenance glossary](../../../../glossary.md#provenance))</span>
- Files changed or created: Exactly the thirty-one paths listed under Affected files; no completed
  task 0019, architecture, build, compiler, Training Java, backend, global-roadmap, or other-module
  path belongs to this task.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend); [module path glossary](../../../../glossary.md#term-module-path))</span>
- Tests and validation: The authoritative clean NN run passed 249 tests with zero skips, failures,
  or errors plus warning-free Javadoc. The documentation pass changed no executable Java/test,
  reran final Javadoc, inspected rendered pages, and passed surface, external-use, legacy/import/
  dependency, Markdown, exact-scope, status/frontier, newline, no-index, whitespace, and diff
  gates.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [run glossary](../../../../glossary.md#term-run); [Javadoc glossary](../../../../glossary.md#term-javadoc); [dependency glossary](../../../../glossary.md#term-dependency); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation impact: Affected Javadocs/package pages, Training API, glossary, and planning
  records are finalized. Tensor/Compile API, architecture/current plan/ADR/tests, Model/Compiler/
  Runtime/Prepare/Engine/backends, Training Java, conformance/integration, result carriers, build
  rules, other modules, completed task 0019, and the global roadmap correctly require no change.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning); [Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Model glossary](../../../../glossary.md#term-model); [Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [conformance glossary](../../../../glossary.md#term-conformance); [modules glossary](../../../../glossary.md#term-module))</span>
- Javadoc review: Complete; generated pages match the implemented policy, ownership, lifecycle,
  failure/effect, state-default, and provenance boundaries without execution claims.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Glossary impact: Complete; current terminology and planned boundaries are synchronized.
- Unresolved issues: None.
- Follow-up required: None for task 0020. Draft [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0020A–0020C and 0021–0024 remain separate work.

Status: Complete
