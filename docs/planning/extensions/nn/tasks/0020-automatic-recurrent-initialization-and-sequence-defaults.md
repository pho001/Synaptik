# Task 0020: Automatic Recurrent Initialization and Sequence Defaults

## Status

Complete

## Goal

Make the existing final `RnnCell`, `GruCell`, and `LstmCell` usable without declaring
`inputSize` or constructing a random-generator factory at ordinary model-composition call sites.
Each cell infers only the positive static final input extent on its first compatible forward
path, initializes and publishes its complete parameter group, and then builds the ordinary
one-step Tensor expression in that same traversal. `hiddenSize`, bias presence, parameter
`DataType`, recurrent weight policy, and seed remain explicit per cell or sequence.

Before adding that second automatic consumer, replace the recent Linear-only
`LinearWeightInitialization` enum with one reusable closed immutable `ParameterInitialization`
configuration value. It selects one of the existing eager algorithms without owning layer schema,
Tensor state, RNG, seed, or callback. Linear migrates in the same change and no compatibility
alias remains; this deliberate early break avoids freezing duplicate Linear and recurrent policy
types.

The same cohesive task makes the current static sequence containers ergonomic for the common
case. Each sequence may construct and own one standard automatic cell, synthesize non-gradient
zero initial states from the input batch extent and the cell's explicit hidden-size/type schema,
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

For embedded input Shape `[time, batch, embeddingSize]`, the first represented recurrent step
initializes `encoder.cell.inputWeight` as `[4 * 128, embeddingSize]`,
`encoder.cell.hiddenWeight` as `[4 * 128, 128]`, and optional zero bias as `[4 * 128]`. The
one-argument sequence call creates zero hidden and cell states shaped `[batch, 128]`, derives a
Java all-valid length for each row, and returns the existing `LstmSequenceForwardResult`.

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

The outputs and their producers have fresh Tensor identities, but every time-step projection
retains the same exact parameter Tensor leaf. That static unroll is the forward expression graph
needed for backpropagation through time (BPTT): current compiler autograd traverses the resulting
expression ancestry by exact Tensor identity and combines contributions that fan out to the same
parameter with ordinary `Tensor.add`. This task preserves those existing Model/Compiler facts; it
does not add an NN backward method, runtime tape, recurrent gradient rule, execution loop, or
backend support claim.

## Scope

- Keep exactly one final public type for each existing recurrent cell and sequence. Do not add
  `LazyRnnCell`, `LazyGruCell`, `LazyLstmCell`, a generic lazy module, or a public build/bind/init
  lifecycle.
- Replace the recently introduced public `LinearWeightInitialization` with one general closed
  immutable public value, `ParameterInitialization`, in `nn.initialization`. This is an intentional
  early breaking rename performed atomically across current NN source, tests, Javadocs, and
  explanatory documentation. Delete the old type and do not retain a deprecated alias, adapter,
  duplicate recurrent enum, compatibility overload, or split Linear/Recurrent policy family.
  Completed task 0019 remains unchanged as historical evidence of the API it originally delivered;
  this task owns migration of the current API.
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
  two configured binary64 values needed by `normal` or `uniform`; it never owns a Shape, data
  type, Tensor, Module, Parameter, RNG/factory, seed, mutable state, fan value, bias policy,
  parameter name/order, or checkpoint encoding.
- Add two public dispatch entries to the existing stateless `ParameterInitializers` namespace:

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

  The three-argument form accepts only `zeros()` and `ones()` and never requests, creates, retains,
  or consumes an RNG. The four-argument form accepts only the six random policies and delegates
  with the exact transient caller-owned generator. Both dispatch exactly once to the matching
  existing `zeros`, `ones`, `normal`, `uniform`, `glorotNormal`, `glorotUniform`,
  `kaimingReluNormal`, or `kaimingReluUniform` entry; they do not duplicate sampling/fan formulas.
  `requiresRandomGenerator()` lets a layer select the correct overload before it creates a source.
- Preserve the existing `ParameterInitializers` Shape boundary. `normal`, `uniform`, `zeros`, and
  `ones` accept the same fully static Shapes their existing methods accept. The four fan presets
  retain the current fully static positive rank-two `[fanOut, fanIn]` contract. Do not invent a
  public `Fan`, convolution schema, gain, activation, fan mode, orthogonal policy, implementation
  registry, callback, or new family of initializer classes before a concrete consumer requires it.
- A recurrent selected value applies independently to the complete input-weight Shape and then the
  complete hidden-weight Shape. One fresh generator stream is used in that exact matrix order for
  random policies; each fan preset recomputes fan values from the matrix currently being created.
  GRU/LSTM use their complete packed matrix Shapes. Optional bias remains a layer-owned complete
  typed-zero input-side vector, including LSTM's forget interval, regardless of the weight policy.
- Migrate the current automatic `Linear` constructor in place so its policy parameter is
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
  in that order. Any `long` seed is valid. They then retain immutable configuration, reserve
  `inputWeight`, `hiddenWeight`, then optional `bias`, and create no generator, Tensor, Tensor ID,
  or Parameter. Each standard sequence constructor delegates this same validation exactly once to
  the one cell it constructs and installs no child on failure.
- Use the exact JDK named deterministic pseudo-random-number generator (PRNG)
  `L64X128MixRandom` for every automatic recurrent initialization attempt that uses a random
  policy. The algorithm name and supplied seed are part of this high-level construction contract.
  Create a fresh generator from `RandomGeneratorFactory.of("L64X128MixRandom").create(seed)` only
  after preflight and only when `requiresRandomGenerator()` is true; retain neither factory nor
  generator in cell state. `zeros()` and `ones()` create and consume no RNG. No global/default
  mutable source, seed manager, implicit seed splitting, JVM-time seed, or caller-owned source is
  used.
- Preserve every existing supplied-Tensor cell constructor and the existing eager constructor
  with explicit `inputSize` and transient caller-owned `RandomGenerator`. Those remain the
  low-level paths for exact state or caller-controlled random algorithms/lifecycles.
- Infer `inputSize` only from a positive static final Dimension of the first compatible input.
  Leading Dimensions remain ordinary broadcast metadata. The automatic cell's parameter
  `DataType` does not silently cast an input or explicit state; existing floating promotion rules
  continue to apply. After binding, the final input Dimension must remain compatible with the
  established input-width schema under the existing cell contract.
- Reuse the complete private reservation/publication lifecycle delivered by NN 0019. Each
  automatic cell publishes all of its direct parameters together or none. Accessors, parameter
  discovery, and state export fail closed until publication. Strict state load may bind the
  complete reservation group without creating a generator or invoking an initializer.
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
  static input time extent. It derives an internal Java `long[]` filled with `time`; this remains
  construction metadata and does not become a Tensor, Parameter, Buffer, retained field, or
  runtime input.
- The overload with lengths but without states creates default zero state and keeps the current
  static packing behavior. The one-input overload combines all-valid lengths with default zero
  state. Explicit-state overloads without lengths combine caller-owned state with all-valid
  lengths. Existing most-explicit overloads remain canonical advanced entry points.
- A default hidden state is a fresh eager provenance-free `TensorFactory.zeros` leaf shaped
  `[batch, hiddenSize]`, using the cell's exact configured/declared floating parameter type,
  `Optional.empty()` label, and `requiresGrad == false`. LSTM creates distinct hidden and cell
  zero leaves with that same schema. Batch comes from the fully static input axis one; hidden size
  and type come from the cell's immutable validated schema. State is created per call, never
  cached, registered, retained, or placed in a state dictionary.
- Preserve the current fully static time-major input requirement and `long[]` static packing.
  Do not turn zero Tensor values into padding. Do not add runtime Tensor lengths, masks, dense
  masked recurrence, or a scan/control-flow operation.
- Preserve the rule that a sequence invokes its cell once per non-empty represented step. An
  explicit all-zero length call invokes no cell; therefore an otherwise unbound automatic cell
  remains unbound until a later call reaches a represented step or strict load binds it. A
  default-state overload still returns its newly created zero state(s) for an all-zero request.
- Every represented step must call the same exact Java cell instance. Every cell call must create
  fresh output Tensor identities and fresh expression-producer occurrences while retaining exact
  references to the same current parameter leaves. Do not clone a cell or parameter per step.
- Add focused automatic lifecycle, policy/PRNG, strict-load, retry, concurrency, overload,
  zero-state, all-valid, ownership, public-surface, and static-unroll provenance tests across all
  three concrete families.
- Finalize all affected Javadocs, package documentation, Training API, glossary, master/task
  evidence, and no-change reasoning in a separate clean documentation-focused task after the
  executable diff and authoritative NN tests stabilize.

## Out of scope

- Initialized `Embedding`, vocabulary inference, padding-row behavior, or a padding index. Those
  require the separate NN 0020A padding-row/update decision; vocabulary size must never be
  inferred from one batch's maximum token ID.
- A deprecated `LinearWeightInitialization` alias, a separate recurrent policy enum, public
  policy callback/service-provider interface, user-defined policy implementation, mutable
  initializer, or retention of RNG/seed/type/Shape inside `ParameterInitialization`.
- Convolution-specific fan derivation, a public `Fan` value, configurable gain/activation/fan
  mode, orthogonal initialization, per-gate initialization, or forget-bias offset. The current
  generic policy delegates fan presets only through the existing rank-two Shape contract; a later
  concrete convolution consumer must design any broader schema.
- `ModuleFactory`, `ModuleFactory.standard()`, static construction recipes, registries, providers,
  dependency injection, configuration aggregation, global data type, global random source, or
  seed allocation. NN 0020B owns the later stateless construction facade after initialized
  recurrent and Embedding contracts are stable.
- Bidirectional, reverse, or multidirectional recurrence; independent directional parameter
  sets/seeds; directional merge policy; stacked recurrent layers; residuals; recurrent dropout;
  or type-specific bidirectional results. NN 0020C owns that separate contract.
- `Lazy*` public types, public initialization status, manual `build`, `bind`, `initialize`,
  `reset`, model-wide descriptor tracing, whole-model preflight, or rollback of an arbitrary
  functional Model body.
- Inferring hidden size, bias, parameter data type, policy, seed, vocabulary size, embedding size,
  class count, output width, gate order, or gate count from input data.
- A generic `RecurrentCell`, `RecurrentSequence`, erased state tuple, public cell-schema value,
  generic recurrent result, common direction base, or new `UnaryTensorModule`/`Sequential`
  participation.
- Changing the existing RNN tanh, GRU reset-after, or LSTM IFGO equations, gate packing, input-
  side-only bias, all-zero bias, result-component order, or static packing/restoration semantics.
- Runtime Tensor lengths, Boolean mask input, dynamic loop count, runtime-dependent active Shape,
  recurrent scan/control-flow body, dense post-cell masking, or reinterpretation of `CUM_SUM` /
  `CUM_PROD` as a recurrent scan.
- Dense padded output, batch-first layout, sorting by length, padding inference from numeric zero,
  or changes to the Data/Text/Vision proposals.
- A backward method on Module/Model/cell/sequence, retained tape, mutable Tensor gradient, new
  derivative rule, public compiler invocation, Training session, optimizer, checkpoint bytes,
  graph execution, backend route, numerical result, or performance guarantee.
- Changes to `Module`, `Parameter`, `Buffer`, `StateDictionary`, `Model`, `Topology`, Model Tensor/
  Shape/DataType APIs, Training Java source, Compiler, Runtime, Prepare, Engine, backend code,
  Gradle/build structure, architecture contracts/ADRs/tests, integration/conformance tests, the
  active CPU work, or the global roadmap.
- Detailed task files or implementation for NN 0020A, 0020B, 0020C, or NN 0021–0024.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [ADR 0007: Neural-network module and training boundary](../../../../design/decisions/0007-neural-network-module-and-training-boundary.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Training graph](../../../../architecture/training-graph.md)
- [Training API](../../../../api/training-api.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Compile API](../../../../api/compile-api.md)
- [Glossary](../../../../glossary.md)
- [Planning guide](../../../planning-guide.md)
- [NN master plan](../master-plan.md)
- [Completed NN task 0012: Vanilla RNN cell](0012-vanilla-rnn-cell.md)
- [Completed NN task 0013: GRU cell](0013-gru-cell.md)
- [Completed NN task 0014: LSTM cell](0014-lstm-cell.md)
- [Completed NN task 0015: Static packed RNN sequence](0015-static-packed-rnn-sequence.md)
- [Completed NN task 0016: Static packed GRU sequence](0016-static-packed-gru-sequence.md)
- [Completed NN task 0017: Static packed LSTM sequence](0017-static-packed-lstm-sequence.md)
- [Completed NN task 0018: Typed functional Model](0018-typed-functional-model-topology.md)
- [Completed NN task 0019: Automatic Linear initialization](0019-automatic-first-forward-linear-initialization.md)

## Architecture constraints

- `extensions/nn` continues to depend only on `modules/model`. Production must not import
  Training, Compiler, Runtime, Prepare, Engine, CPU, Metal, CUDA, or another backend.
- NN owns module state, recurrent-cell parameters, cell-specific sequence composition, default
  initial-state convenience, and initialization policy selection. Model continues to own Tensor
  identity, descriptors, Shape algebra, promotion, eager zero/random leaves, expression
  provenance, and primitive operations. Compiler continues to own autograd and graph capture.
- A recurrent state passed between steps is caller-/expression-threaded Tensor data, not a
  Parameter, Buffer, retained field, optimizer value, or runtime tape. A default state changes
  only who constructs that explicit Tensor for the current call.
- `Parameter` remains always bound when publicly observable. Future state exists only as private
  Module reservations until one complete cell-local publication or strict load succeeds.
- `Module` retains no universal forward method. Cells and sequences stay direct final Module
  subclasses with their truthful type-specific signatures and remain outside `Sequential`.
- Current Tensor producer identity is occurrence-specific. Reusing a Java module cannot reuse a
  derived Tensor result implicitly; every public Tensor operation call must return a fresh
  identity with provenance retaining its exact inputs.
- Current compiler reverse mode keys contributions by exact Tensor identity and merges multiple
  contributions through ordinary `Tensor.add`. NN must preserve the shared parameter leaf and
  fresh time-step producer structure that lets this existing downstream rule see temporal
  fan-out. NN must not import or invoke compiler internals to prove it.
- Static Java lengths may specialize the statically unrolled expression structure. Runtime
  lengths must not be presented as supported until NN 0021's cross-module scan/input-binding
  prerequisite is complete.
- If implementation needs a new Model operation/method, public lifecycle/status/schema API,
  caller-owned source retention, new dependency, generic recurrent abstraction, runtime control
  flow, change outside the exact path list, or architecture-rule change, stop and report the
  blocker rather than widening this task.

## Package impact

Existing packages changed:

- `io.github.pho001.synaptik.nn.initialization` — replace the Linear-only enum with one closed
  general policy value, add policy dispatch to the existing stateless namespace, and update
  package documentation.
- `io.github.pho001.synaptik.nn.layers` — the six existing concrete cell/sequence types and
  automatic Linear policy parameter plus package documentation.

Existing packages consumed unchanged:

- `io.github.pho001.synaptik.nn.module` for ownership, reservation/publication, state discovery,
  strict load, and typed Model composition.
- Model data-type, Shape, Tensor factory/provenance, eager random, and primitive expression
  packages.

No package, Gradle module, dependency edge, architecture rule, public training type, or compiler
type is added.

## Public API

`ParameterInitialization` is a final immutable closed value with the eight public static factories
and one `requiresRandomGenerator()` query shown in Scope, plus value-based `equals`, `hashCode`,
and diagnostic `toString`. It has no public/protected constructor, field, nested type, raw-kind
getter, callback, or subclass/implementation extension point. No-argument presets may be
canonicalized, but reference identity is not part of the contract.

Two values are equal exactly when they select the same algorithm and their configured binary64
arguments, if any, have equal `Double.doubleToLongBits` representations. Equality is therefore
signed-zero-sensitive; non-finite configured values cannot exist. Hashing uses the same facts.
`toString` is deterministic and descriptive but is neither parsing nor persistence format.

`normal(mean, standardDeviation)` validates at factory call time: finite mean, then finite
standard deviation, then numerically non-negative standard deviation; either signed zero is
accepted. `uniform(lowerBoundInclusive, upperBoundExclusive)` validates finite lower bound, then
finite upper bound, then strict `lower < upper`. Presets need no validation. These checks create no
Tensor, identifier, storage, or RNG effect.

`ParameterInitializers.initialize(shape, dataType, initialization)` null-checks those arguments in
that order, rejects random policies before delegating, and is the only generic zero/one route.
`ParameterInitializers.initialize(shape, dataType, initialization, randomGenerator)` null-checks in
that order, rejects zero/one policies before delegating, and is the generic random route. After
dispatch selection, the matching existing initializer owns its already documented floating-type,
fully-static Shape, Java-array/count, rank-two fan, sampling, failure, and side-effect validation
order. The dispatchers retain no input and add no fallback/default behavior.

The automatic `Linear` constructor replaces its `LinearWeightInitialization` parameter with
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

The second LSTM call demonstrates that automatic parameter initialization does not make recurrent
state implicit. Callers that continue a stream still pass exact previous states.

## Automatic cell initialization contract

For RNN, let `gateMultiplier = 1`; for GRU, `3`; and for LSTM, `4`. One automatic cell stores
only validated immutable `hiddenSize`, bias choice, floating parameter type, selected policy,
and seed before binding. Its eventual state schema is:

| Path | Shape | Initialization order and policy |
|---|---|---|
| `inputWeight` | `[gateMultiplier * hiddenSize, inferredInputSize]` | first; selected general policy applied to this exact Shape |
| `hiddenWeight` | `[gateMultiplier * hiddenSize, hiddenSize]` | second; same selected policy applied independently to this exact Shape and, when random, the advanced generator |
| `bias` when configured | `[gateMultiplier * hiddenSize]` | third; deterministic typed zero, no random draw |

`normal`, `uniform`, `zeros`, and `ones` use each exact matrix Shape without fan inference. Each
Glorot/Kaiming preset derives its distribution independently from the current complete packed
rank-two Shape. The layer owns matrix schema/order and bias policy; the general value does not.

The first compatible direct cell forward performs:

1. reject null input/state arguments in the existing order;
2. validate every caller-controlled input/state rank, floating type, statically decidable feature,
   promotion, broadcast, gate/state Shape, packed-size, count, and Java-array-limit fact knowable
   from immutable configuration, without creating a Tensor;
3. when still unbound, enter one cell-local initialization critical section, recheck completion,
   and revalidate the input-derived schema;
4. if and only if the policy requires randomness, create one fresh `L64X128MixRandom` generator
   from the exact retained seed;
5. create input weight, hidden weight, then optional zero bias as unpublished local Tensors,
   choosing the no-RNG or exact-generator dispatcher overload as required;
6. pass the complete ordered group once to `bindReservedParameters`, which validates all values,
   prepares all wrappers, publishes all state, and release-publishes completion;
7. leave the initialization critical section, read the complete current parameter bindings once,
   and construct the existing ordinary one-step Tensor formula; and
8. return that call's ordinary current result.

Later calls do not create a generator or parameter Tensor. They validate against the established
schema, read current wrappers, and create fresh ordinary expression occurrences. A compatible
replacement or strict load affects later calls only; prior expressions retain earlier exact
parameter references.

The first automatic input final Dimension must be positive and static. After binding, current
cell rules continue to permit only a compatible final input Dimension; do not add a cast or infer
a second schema. Leading rank and broadcast-compatible Dimensions may vary as allowed by the
existing one-step formulas.

## Failure, retry, load, and concurrency contract

- Constructor validation consumes no random draw, Tensor ID, Tensor, storage, Parameter, or
  wrapper. It reserves the complete direct state names only after immutable configuration is
  valid.
- A caller-controlled direct-cell validation failure before initialization consumes no generator,
  draw, Tensor ID, or parameter publication.
- An initializer/source/allocation/ID failure may preserve completed random calls, local Tensors,
  host allocations, and opaque consumed Tensor IDs, but publishes no Parameter wrapper. A retry
  of a random policy creates a fresh standard generator from the same algorithm and seed, so its
  represented random sequence restarts deterministically; zero/one retries create no generator.
  Opaque IDs do not roll back.
- Complete publication happens before that first call constructs its recurrent expression. A
  later expression-construction failure does not undo already published parameters.
- Synchronization covers only competing first initialization attempts on one cell. Release/
  acquire reservation completion guarantees that racing access/discovery observes unbound failure
  or the complete group, never a partial group. Replacement, strict load, sequence construction,
  mode changes, and arbitrary functional Model bodies remain caller-coordinated.
- Strict load validates complete paths and each configured gate/type/Shape/gradient schema before
  publishing any reservation. Candidate `inputWeight` supplies the positive input width;
  `hiddenWeight` and optional bias must match configured hidden size, gate multiplier, type, and
  existing packing. Successful load retains exact candidate Tensor references and invokes no
  PRNG or initializer.
- Equivalent supplied/eager/automatic cells keep the same path/kind/type/Shape schema and can
  load dictionaries in either direction when hidden/input dimensions and bias choice match.
- In a functional Model, an earlier layer or cell can remain initialized when later user code
  fails, and a registered cell can remain unbound when its branch or every recurrent step is
  skipped. There is no model-wide transaction.

## Sequence default and all-valid contract

Every sequence overload preserves fully static time-major input `[time, batch, inputSize]` and
the current result/packing semantics.

| Supplied arguments | Derived values | Meaning |
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
  false gradient eligibility, and fresh identity for each state/call;
- the default state's type does not come from token IDs or silently mirror a wider input type;
  mixed-floating expression results continue to follow current promotion;
- explicit state references are never copied, replaced, cached, or converted;
- an all-zero explicit-length call with explicit state preserves the current exact-reference
  final-state shortcut and creates no cell expression;
- an all-zero explicit-length call with default state returns its newly created zero state(s),
  invokes no cell, and leaves an otherwise automatic cell unbound; and
- each non-empty step uses the same owned cell exactly once and retains current active-set,
  stable-row-order, exit-state restoration, and compact-output semantics.

Before allocating an all-valid array, zero state, eager index leaf, automatic parameter, or
expression, each overload must complete every caller-controlled null, rank, fully static Shape,
time/batch/input-feature, cell-schema, hidden-size/type, length-count/range, packed-count,
promotion, broadcast, gather/select/stack, and Java-array-limit check that is knowable from its
actual and derived arguments. Validation may inspect immutable package-private cell schema facts;
it must not require a public schema API or an already bound Parameter. After this preflight:

1. allocate any required all-valid Java length snapshot;
2. allocate required default zero state leaf/leaves in hidden-then-cell order;
3. construct the existing static SELECT/GATHER recurrence; and
4. let the first represented `cell.forward` bind an automatic cell before that cell constructs
   its own formula.

An initialization or construction failure after these effects begin does not roll back the
derived host array, zero-state/parameter/index Tensor IDs, allocations, random draws, published
cell state, or expression prefix. No partial sequence result is returned, and no per-call value is
retained by the sequence. A call with no represented step ends after state derivation/restoration
and never reaches automatic cell binding.

## Static unroll, BPTT provenance, and parameter fan-out

This task must make the distinction between Java object identity, Model Tensor identity, and
compiler graph identity explicit:

- `sequence.cell()` returns one stable Java object and that cell owns one stable set of Parameter
  wrappers after binding.
- A current parameter wrapper's `value()` returns the same exact leaf Tensor reference for every
  step until caller-coordinated replacement/load changes it.
- Every call to `cell.forward` invokes public Tensor operations anew. Its derived results have
  fresh `TensorId` values and distinct `TensorProducer` occurrences even when descriptors and
  parameter inputs are equal.
- A later step's compact carried state is an expression descended from the preceding step result;
  this establishes temporal dependency in the statically unrolled ancestry.
- Input/hidden parameter leaves occur as exact repeated producer-input references across time.
  They are not copied, relabeled, detached, or recreated per step.
- Current compiler autograd keys contribution lists by exact Tensor identity. If a downstream
  functional-gradient request selects one of these shared parameter leaves, contributions from
  every selected temporal route fan into that one target and are combined with ordinary public
  Tensor addition in the compiler's established deterministic traversal order.

NN tests must inspect the Model provenance preconditions: distinct step output IDs/producers,
carried-state ancestry, and repeated exact parameter-leaf references/fan-out. They must not import
package-private compiler implementation, fabricate graph-local `NodeId`, claim numerical
gradients, or add an NN backward method. Existing compiler and Training documentation supplies
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

Do not edit `Module`, StateDictionary production, Model/Topology source, existing sequence
packing tests/result carriers, Tensor/Compile API, architecture files, ADRs/tests, global roadmap,
other master/task files, build files, Training Java source, compiler/runtime/prepare/engine,
backends, conformance, or integration tests. If correct implementation requires any such path,
stop and report the exact blocker.

## Maximum scope

This is an explicit cohesive exception to the usual twelve-to-eighteen-path heuristic. The three
cells share one public policy/lifecycle contract, the three containers must expose symmetric
defaults, and the current Linear policy plus its repository-owned consumers must migrate in the
same atomic change. Partial rollout would leave duplicate/incompatible policy names or make
matching recurrent families behave differently. The task may create, delete, or modify at most
the exact thirty-one paths above. Do not use this exception for ModuleFactory, Embedding,
directionality, scan, unrelated refactoring, or duplicated utility abstractions.

## Test requirements

### Public surface and compatibility

- Verify `ParameterInitialization` is final, immutable, closed, has exactly the eight named
  factories, `requiresRandomGenerator`, value methods, and no public constructor/field/nested
  type/raw-kind/callback/implementation hook. Verify structural equality/hash behavior including
  configurable arguments and signed zero, deterministic diagnostic strings, and non-identity
  value semantics.
- Verify `LinearWeightInitialization` and `RecurrentWeightInitialization` are absent from source,
  compiled public classes, imports, constructor signatures, Javadocs, and current explanatory
  documentation. Completed task 0019 is the explicit historical exception and remains unchanged.
  Verify Linear's migrated automatic constructor accepts `ParameterInitialization` and retains its
  other exact arguments; supplied/eager constructors remain unchanged.
- Verify each cell and sequence remains final, extends `Module` directly, and declares exactly its
  prior public surface plus the planned constructor/overloads. Verify there is no `Lazy*`, public
  lifecycle/status/schema getter, generic recurrent base, `RandomGeneratorFactory` parameter, or
  unary/Sequential adapter.
- Re-run all existing supplied/eager construction, formula, packing, replacement, mode, result,
  and failure tests unchanged in meaning.

### Automatic initialization and deterministic policy

- Verify policy factories reject invalid configurable values immediately and in the documented
  order without Tensor, ID, storage, factory, or RNG effects: non-finite mean; non-finite then
  negative standard deviation; non-finite lower then upper bound; and non-increasing bounds.
- Verify both generic `ParameterInitializers.initialize` dispatch overloads have the exact surface,
  null/wrong-overload validation order, and one-to-one storage/draw/failure behavior of all eight
  existing initializers. Cover floating types, fully static/dynamic/zero/scalar Shapes as relevant,
  fan presets' positive rank-two `[fanOut, fanIn]` limit, configurable normal/uniform values, and
  source failure/non-rollback behavior.
- Verify Linear automatic initialization under all eight policies. Its four legacy presets retain
  exact values, normal/uniform/zero/one dispatch correctly, its factory/seed remain separate, and
  zero/one never call `factory.create` or consume a draw. Migrate Model/state-dictionary fixtures
  without changing their lifecycle meaning.
- For every cell family, both bias modes, representative policies from every policy category, and
  every floating parameter type,
  verify no Tensor/Parameter/generator is created at construction; access/discovery/export fails
  closed; first compatible forward derives exact input width; and complete parameters publish in
  input-weight, hidden-weight, optional-bias order before ordinary expression IDs.
- Compare automatic weight storage exactly with generic dispatch driven by
  `RandomGeneratorFactory.of("L64X128MixRandom").create(seed)`. Verify the input matrix consumes
  the first draws, hidden matrix consumes the advanced stream, and bias consumes none. For fan
  presets, verify each complete input/hidden Shape derives its own fan values; for configurable
  and constant policies, verify exact arguments/values are applied independently to both Shapes.
- Verify recurrent `zeros()` and `ones()` create no generator at all and optional bias still uses
  the layer-owned zero rule. Verify one shared immutable policy value can configure multiple
  layers without retaining Shape, source advancement, or layer state.
- Verify same configuration/seed produces equal represented parameter values in separate cells,
  distinct Tensor/Parameter identities, and no global/shared mutable generator state.
- Cover null/order, hidden-size, packed-size overflow, non-floating type, policy null, first input
  rank/final-Dimension/static/positive constraints, explicit hidden/cell mismatch, promotion/
  broadcast failure, Java-array limit, source/allocation/ID failure where controllable, retry, and
  no partial publication.
- Verify two concurrent compatible first calls publish one complete group and both use it;
  incompatible/failing races observe only documented cell-local behavior.
- Verify strict load initializes each automatic cell and migrated automatic Linear without a
  forward call, generic-policy dispatch, factory call, or PRNG; rejects
  wrong path/kind/type/Shape/packing/gradient as applicable before publication, and round-trips
  dictionaries with equivalent eager/supplied cells. Keep this coverage in the three authorized
  initialization tests rather than widening `StateDictionaryTest`.

### Sequence construction and defaults

- Verify each high-level sequence constructor creates exactly one matching automatic cell,
  registers it under `cell`, exposes the exact child, has no direct state, and preserves recursive
  paths/order after binding.
- Verify `forward(input)` derives all-valid lengths and one/two exact typed zero states;
  `forward(input, lengths)` derives only state; explicit-state overload without lengths derives
  only all-valid lengths; and the most-explicit overload preserves existing behavior.
- Verify zero states are fresh eager dense unlabeled provenance-free leaves with Shape
  `[batch, hiddenSize]`, exact cell parameter type, false gradient eligibility, all-zero storage,
  and no module/state-dictionary registration. LSTM hidden/cell zeros are distinct.
- Verify calls with varying compatible time/batch extents derive new arrays/states per call while
  the initialized input width remains fixed. Cover zero time, empty batch, all-zero lengths,
  length-array limits, state type promotion, and later input-width mismatch.
- Verify an explicit all-zero call invokes no cell and leaves a new automatic cell unbound;
  default-state variants return exact created zero references. A later non-empty call binds it.
- Reuse the existing packing tests as authoritative evidence that explicit lengths still omit
  padded rows, preserve stable order, restore exits, and never infer padding from zeros; do not
  edit those tests without a concrete regression.

### Static-unroll/BPTT provenance preconditions

- On at least two represented steps per family, verify packed outputs have distinct `TensorId`
  values and distinct producer identities.
- Traverse public provenance to verify the second step depends on the exact first-step carried
  output path and that every step's projection ancestry retains the exact current input- and
  hidden-weight Tensor leaves.
- Verify those parameter leaves are one identity with multiple producer-input edges, not one
  cloned leaf per step, and that replacing parameters affects only a later unroll.
- Verify all relevant derived descriptors retain gradient eligibility from the shared parameter
  leaves. Do not calculate or assert numeric gradients in NN tests.

### Exclusions

- Scan source/public bytecode/imports for forbidden Lazy/factory/direction/scan/mask/runtime/
  compiler/training APIs, retained default states/lengths/generators, and dependencies outside
  Model.
- Verify no result carrier component, gate formula, parameter name/order, bias packing, sequence
  packing path, or static `long[]` meaning changed.

## Documentation requirements

The implementation agent may draft detailed Javadocs while coding. After executable Java and the
one authoritative NN test run stabilize, a separate clean documentation-focused agent/thread must
read the final diff, source/tests, generated Javadocs, architecture/planning contracts,
documentation rules, General/API-Javadoc/Planning/Example profiles, completed NN 0012–0019,
relevant Tensor producer/factory/Shape/promotion contracts, compiler exact-identity autograd
contract, Training API, glossary, and Checkpoint/Data/Text plans. It must finalize:

- all affected policy value/dispatcher, Linear, cell, sequence, constructor, accessor, overload,
  parameter, return, failure, lifecycle, ownership, concurrency, and side-effect Javadocs;
- initialization/package Javadocs for the deliberate Linear-policy migration, eight general
  algorithms, immediate configurable-value validation, exact value semantics, dispatch overloads,
  Shape/fan boundary, zero/one no-RNG path, exact standard recurrent PRNG, explicit separate
  seed/type/hidden facts, caller-owned advanced constructors, and no global source;
- layers/package Javadocs distinguishing one cell instance from fresh time-step Tensor
  occurrences, default explicit zero-state construction, static all-valid lengths, static packing,
  and future runtime scan/direction contracts;
- Training API example and explanation of automatic cell/sequence composition, continued explicit
  state, state/load behavior, static-unroll provenance, shared parameter fan-out, compiler-owned
  BPTT construction, and current execution limits;
- glossary entries for general parameter-initialization policy, recurrent automatic
  initialization, default recurrent state, all-valid static sequence, and static-unroll parameter
  sharing without redefining compiler autograd; and
- final planning evidence/status, links/anchors/fences, exact scope, and reasoned no-change
  conclusions.

The documentation pass reuses stable Java test evidence unless it changes executable Java. It
must record explicit no-change conclusions for `ARCHITECTURE.md`, current architecture plan, ADRs,
architecture tests, Tensor API, Compile API, Model capabilities/source, Module/Parameter/Buffer/
StateDictionary production, Training Java API, Checkpoint/Data/Text/Vision plans, Gradle, Compiler/
Runtime/Prepare/Engine/backends, conformance/integration, sequence result/packing tests, global
roadmap, active CPU work, and every other module. `ModelTest` and `StateDictionaryTest` are changed
only to compile against the intentional current-API rename and retain their behavioral coverage.

## Acceptance criteria

- One immutable closed `ParameterInitialization` value covers the eight required algorithms with
  exact validation/equality semantics and generic dispatch through existing
  `ParameterInitializers`; it owns no layer schema, Tensor, callback, RNG, seed, or mutable state.
- The current automatic Linear API and all repository-owned consumers migrate atomically from
  `LinearWeightInitialization`; neither that legacy type nor a recurrent-specific duplicate/alias
  remains.
- All three existing final cells support automatic first-forward input-width inference through
  one non-Lazy constructor with explicit hidden size, bias, type, general policy, and seed.
- High-level automatic recurrent construction exposes no `RandomGeneratorFactory`; it uses exact
  deterministic `L64X128MixRandom` per random attempt and no RNG for zero/one, while existing
  caller-owned `RandomGenerator` eager constructors remain.
- Parameter reservation, publication, retry, strict load, discovery/export gating, failure, and
  concurrency behavior matches the proven NN 0019 Module lifecycle for each complete cell group.
- All three existing final sequences can construct their own matching standard cell and provide
  zero-state and all-valid conveniences without changing their advanced explicit contracts.
- Default state derives batch from static input and hidden size/type from cell schema, is explicit
  non-gradient Tensor data for that call, and is never retained or registered.
- Static `long[]` packing remains the honest compatibility contract; runtime lengths/masks and
  scan remain Draft.
- One cell/parameter set is shared across time, every step builds fresh Tensor identities/
  producers, temporal state ancestry is connected, and repeated exact parameter leaves form the
  Model provenance fan-out consumed by existing compiler autograd.
- No NN backward/runtime/execution claim, generic recurrent base, ModuleFactory, initialized
  Embedding, bidirectionality, new dependency, architecture change, or unrelated refactor lands.
- Focused tests, one final NN suite, NN Javadoc, public-surface/external example, dependency/import,
  Markdown, exact-scope/frontier/status, newline, whitespace, and diff gates pass.
- Documentation review is completed in a separate clean context and the task becomes `Complete`
  only after its final evidence and completion summary are filled.

## Tests / validation

The implementation task must run these tiers once after executable stabilization:

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

2. One authoritative NN module suite after all executable changes:

   ```text
   ./gradlew :extensions:nn:test
   ```

3. After the clean documentation pass:

   ```text
   ./gradlew :extensions:nn:javadoc
   ```

4. Inspect generated Javadocs for `ParameterInitialization`, `ParameterInitializers`, Linear, all
   six cell/sequence types, and both package pages. Use `javap`/reflection plus a standalone
   external-package `Model.define` example to verify the exact public surface, absence of both
   forbidden legacy/duplicate policy classes, and Java overload usability.
5. Verify NN production imports only Model/JDK APIs and contains no public Lazy/build/bind/init/
   status, generic recurrent, ModuleFactory, direction, mask, runtime scan, compiler/training, or
   retained RNG/default-state mechanism.
6. Run Markdown local-link/anchor/unique-heading/fence/newline/trailing-whitespace checks for the
   two changed planning files and final docs. Verify exactly one NN task/master row is `Ready`
   during planning; on completion, both task/master are `Complete` and NN 0020A–0020C plus
   0021–0024 remain concise `Draft` rows without task files.
7. Verify exact thirty-one-path scope by subtracting the pre-existing shared CPU/roadmap worktree
   snapshot. Do not modify, stage, revert, or include those unrelated changes. Run
   `git diff --check`; use a no-index whitespace check for this initially untracked task file.

Repository-wide tests are not required because this task changes one extension and no dependency,
architecture, build, compiler, execution, or backend contract. Escalate to the root suite only if
the final diff unexpectedly crosses one of those boundaries.

## Dependencies

- NN 0001–0019 are Complete.
- Existing supplied/eager RNN, GRU, and LSTM cell contracts, their concrete sequence/result APIs,
  static packing semantics, typed functional Model, and Module ownership/state paths are stable.
- NN 0019's private reservation/publication, fail-closed discovery/export, strict-load binding,
  retry, and release/acquire completion contract is implemented and reusable unchanged.
- Model provides eager typed zeros/random initialization, immutable Tensor identity/provenance,
  static Shape/Dimension facts, promotion/broadcast, and every primitive operation used by the
  existing cell/sequence formulas.
- Compiler exact-identity reverse traversal and contribution accumulation are current downstream
  contracts. This task changes only the forward expression topology they consume.
- Training Java remains a marker. Its API document may explain the current compiler handoff, but
  no executable Training workflow is needed.
- The user-authorized NN interleave remains isolated from the active CPU frontier. Pre-existing
  CPU and roadmap worktree changes belong to another task and must remain untouched.

## Follow-up tasks

- NN 0020A: initialized Embedding with explicit vocabulary size, embedding size, data type,
  `ParameterInitialization`, seed, and the same documented standard PRNG for random policies; no
  vocabulary inference from token IDs. Select how an optional padding row is initialized and kept
  invariant (including future update behavior) before Ready.
- NN 0020B: stateless standard `ModuleFactory` construction recipes after recurrent and Embedding
  initialization contracts are stable. `embedding`, `linear`, `rnn`, `gru`, and `lstm` return
  concrete types; recurrent recipes hide standard Cell+Sequence assembly. Every recipe takes
  explicit per-layer data type, `ParameterInitialization`, and seed. `ModuleFactory.standard()`
  owns no module registration/configuration/RNG/seed manager or state; `Topology.addModule` keeps
  sole ownership, and advanced direct constructors retain caller-owned generators/factories.
- NN 0020C: type-safe bidirectional/multidirectional recurrent composition with independent
  directional cells/parameters/seeds, valid-prefix reverse traversal, an explicit `CONCAT`/`SUM`
  choice or deliberately narrower first merge policy, and type-specific final states.
- NN 0021: coordinate genuine recurrent scan/control-flow and runtime input binding across owning
  modules without specializing topology to one runtime length vector.
- NN 0022: consume Data-owned runtime valid lengths only after NN 0021.
- NN 0023: select arbitrary mask-with-holes semantics only for a concrete consumer.
- NN 0024: run the typed Model/recurrent/Data/training/checkpoint integration checkpoint.

## Architecture impact

Expected impact: None.

The architecture already assigns neural-network parameters, module ownership, layers, stateful
composition, and forward conveniences to `extensions/nn`; generic Tensor construction to Model;
and automatic differentiation to Compiler. This task reuses those boundaries and adds no
dependency. Stop if implementation requires an architecture, module-boundary, build, or
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

## Documentation-agent handoff

Give the documentation-focused agent this task, final diff, focused/final Java evidence, exact
public signatures, breaking policy migration/value/validation/dispatch semantics, policy/PRNG
selection, cell initialization/publication order, retry/load/concurrency effects, sequence
overload/default-state semantics, and provenance test evidence. Identify whether executable Java
changed after the authoritative suite and enumerate all thirty-one authorized paths plus the
pre-existing CPU/roadmap paths to ignore.

The documentation agent independently reads the architecture/documentation profiles, final
source/tests/generated Javadocs, relevant completed tasks, Tensor/Compiler/Training contracts,
glossary, and future NN rows. It must distinguish parameter-leaf initialization from forward
expression construction, one Java cell from multiple fresh time-step occurrences, static unroll
from runtime scan, Model provenance fan-out from generated gradients, and current compiler
autograd from an executable Training workflow. It reuses stable test evidence unless executable
behavior changes and records all final no-change conclusions.

## Local decisions

- Use one existing final cell type per family. Input width is a batch-derived binding fact; hidden
  width, gate semantics, bias, parameter type, policy, and seed remain architecture choices.
- Replace the new Linear-specific enum early with one closed immutable parameter-policy value.
  Keeping aliases or adding another recurrent enum would duplicate the same algorithm choice
  before the public API is stable. Layer code still owns Shapes, fans implied by those Shapes,
  parameter order, gate schema, and bias rules.
- Expose all four existing fan presets plus configured normal/uniform and constant zero/one because
  Linear, recurrent matrices, and immediate initialized Embedding are concrete consumers of the
  same algorithms. Selection remains explicit; Kaiming/ReLU names are not implied recurrent
  defaults. Defer convolution fan schema and orthogonal/per-gate policies.
- Make `L64X128MixRandom` the exact ordinary high-level PRNG so users supply one seed per layer,
  not a factory. Create it only for random policies; zero/one stay genuinely RNG-free. Preserve
  direct constructors as the advanced escape hatch for caller-owned random algorithms.
- Use the cell's declared parameter type for default zero state. Input type remains independently
  governed by existing promotion and no hidden cast occurs.
- Provide both all-valid and zero-state overloads. `forward(input)` is the concise common case;
  explicit state remains available for streaming/truncated recurrence and explicit lengths remain
  available for static right padding.
- Keep explicit all-zero packing honest: no represented step means no cell call and therefore no
  automatic binding. Strict load or a later non-empty call supplies the first binding event.
- Share one cell and parameter set across time. Multiple Java cell instances would incorrectly
  create independent parameter owners; fresh public Tensor calls already provide the distinct
  expression occurrences required by static unroll and compiler reverse traversal.
- Keep future Embedding initialization, construction factory, bidirectionality, and runtime scan
  separate because each needs an independent policy or cross-module contract.

## Known limitations

- Automatic binding still requires a positive static final input Dimension on the first
  represented cell path or a compatible complete strict dictionary. It is not runtime shape
  binding.
- A static sequence requires fully static time/batch/input Shape and Java lengths. Default
  all-valid lengths do not relax this requirement.
- All-zero/zero-time static traversal can return default or explicit state without binding an
  automatic cell. Complete parameter discovery/export remains unavailable until later binding.
- Default state is eager host-backed data allocated per call. It is not a symbolic runtime input,
  cached constant, trainable initial state, or backend-resident value.
- One seed governs both matrices in documented draw order. There is no automatic seed derivation,
  per-gate source, global source, or persisted RNG state.
- Generic fan presets currently accept only complete positive rank-two `[fanOut, fanIn]` Shapes.
  There is no public convolution fan abstraction until a concrete convolution layer requires one.
- Initialization atomicity is cell-local. A functional model or multi-step sequence is not one
  transaction, and Tensor IDs/allocations are not rolled back.
- Static provenance supplies a valid differentiable expression topology, but this task does not
  expose public compilation/training execution, calculate a gradient, prove backend coverage, or
  benchmark BPTT.
- Bidirectional/multidirectional composition and runtime valid-length scan remain Draft follow-ups.

## Validation evidence

Planning-only work completed in clean planning context `/root/nn_0020_planning`; implementation,
commit, and push were not performed.

- Read the full repository instructions, architecture contract/current plan, planning guide/
  roadmap, documentation rules and General/API-Javadoc/Planning/Example profiles, NN master plan,
  completed NN 0012–0019, current recurrent cells/sequences/results/tests, Module reservation and
  strict-load lifecycle, complete `ParameterInitializers`, `LinearWeightInitialization`, Linear
  automatic source/Javadocs and relevant initializer/Linear/Model/state-dictionary tests,
  Model/Topology, relevant Model Tensor/Shape/promotion/provenance APIs, compiler exact-identity
  autograd, Training marker/API, and glossary. No architecture, dependency, ownership, or
  compiler-boundary conflict was found.
- Replanned the initializer surface after the user selected an early atomic migration: one final
  immutable closed `ParameterInitialization`, eight named factories, exact value/configuration
  validation, two generic `ParameterInitializers` dispatch paths, no legacy alias or recurrent
  duplicate, and no public callback/fan abstraction. Completed task 0019 was not edited.
- Cross-reference review found historical completion-era NN 0018/0019 follow-up text plus the
  current NN master/roadmap references. Established NN 0021–0024 IDs remain unchanged; 0020A,
  0020B, and 0020C use the planning guide's existing alphanumeric convention and have no detailed
  task files. Historical completed-task evidence was not rewritten.
- A targeted Ruby Markdown validator passed for the NN master and this task: all local link
  targets and heading anchors resolve, effective headings are unique, backtick/tilde fences are
  balanced, terminal newlines are present, and no trailing whitespace exists.
- Frontier checks passed: exactly one NN task file has scalar status `Ready`, exactly one NN
  master row is `Ready`, task 0020 is linked from that row, and no 0020A–0020C or 0021–0024 task
  specification exists.
- Exact NN-owned planning scope passed: after subtracting concurrent CPU/backend-guide/roadmap
  work, the only NN paths are the modified master plan and this new task file. The global roadmap
  was reread for relevant numbering but received no hunk; all warned or newly appearing CPU-side
  paths were treated as externally owned and were never edited, staged, reverted, or included.
- `git diff --check` passed for the shared worktree at validation time. The no-index check for this
  new task returned the expected difference status with no whitespace diagnostic.
- No Java, Javadoc, Gradle, architecture, compiler, backend, conformance, integration, module, or
  repository-wide test was run because this is a planning-only change with no executable or
  authoritative-architecture behavior.
- Clean implementation context `/root/nn_0020_implementation` replaced the Linear-only public
  policy with the exact closed immutable `ParameterInitialization`, added both generic dispatcher
  overloads, migrated automatic Linear, and implemented the automatic reservation/binding/load
  lifecycle plus standard/default sequence surface for all three recurrent families. It changed
  no Model, Module, compiler, runtime, backend, build, architecture, or completed-task-0019 path.
- Focused initializer/Linear, recurrent-cell initialization and strict-load, deterministic
  `L64X128MixRandom` stream, sequence default/preflight, all-zero skip, replacement, and provenance
  selections passed after stabilization. The first development-wide NN run reported eight
  expected exact-surface assertions against the newly added APIs; the tests were migrated and no
  production defect remained from that run.
- After the final default-state effect-order preflight, the authoritative clean command
  `./gradlew :extensions:nn:clean :extensions:nn:test :extensions:nn:javadoc` passed. Final XML
  inspection reports 249 tests, zero skips, zero failures, and zero errors; executable Java and
  tests did not change afterward.
- Clean output contains no compiled `LinearWeightInitialization` or recurrent-specific alias.
  `javap -public/-private`, repository reflection tests, and a standalone external-use compile
  confirmed the intended policy, dispatcher, migrated Linear, cell, and sequence surfaces and no
  retained recurrent generator/factory. Production import/dependency inspection retains only the
  existing Model dependency. `git diff --check` passed.
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
  fresh-producer static provenance without claiming numerical execution or public training.
- Training API and glossary now use the common policy, describe automatic recurrent lifecycle,
  default state/all-valid overloads, and identify the existing Compiler exact-identity fan-out
  boundary. They keep runtime scan, Tensor `validLengths`, bidirectionality, `ModuleFactory`, and
  initialized Embedding as future work. Completed task 0019 remained unchanged.
- Final `./gradlew :extensions:nn:javadoc` passed without warnings after documentation-only edits.
  Generated pages were inspected for the policy, dispatcher, exact PRNG/strict-load lifecycle,
  length snapshot, default-state, all-zero, and shared-leaf/fresh-producer contracts. Java tests
  were not repeated because executable Java/tests were unchanged after the authoritative run.
- Final `javap -public/-private`, repository reflection evidence, standalone external-use
  compilation, compiled/source legacy-type absence, recurrent retained-field, production import,
  and Gradle dependency checks passed. A targeted Markdown validator passed local targets,
  anchors, unique headings, fences, terminal newlines, and trailing whitespace for the two final
  docs and two planning records.
- `ARCHITECTURE.md`, the current architecture plan, ADRs, and architecture tests correctly require
  no change because the task adds no module, dependency edge, or ownership rule. Tensor and
  Compile APIs require no change because existing Model operations/provenance and Compiler
  exact-identity accumulation already own the consumed boundary. Model, Compiler, Runtime,
  Prepare, Engine, backends, and Training Java require no change because NN only constructs
  existing expressions and exposes neither execution nor training behavior.
- Build/dependency rules are unchanged (`extensions/nn` still imports only Model). Backend
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
`ParameterInitialization` value and dispatchers, migrated automatic Linear, added atomic automatic
binding to all three recurrent cells, and added the standard-cell/default-state/all-valid sequence
surface. Focused initializer, Linear, Module/state, cell, sequence, strict-load,
deterministic-stream, and provenance tests are implemented. Independent clean documentation
context `/root/nn_0020_docs` finalized every authorized Javadoc, explanatory-documentation,
glossary, and planning impact without changing executable code.

## Completion summary

- Completed changes: The common closed eight-policy value and exact dispatch routes, Linear
  migration, automatic recurrent binding/retry/strict-load lifecycle, standard sequence
  construction/defaults, and shared-parameter fresh-producer static provenance are implemented,
  tested, and documented.
- Files changed or created: Exactly the thirty-one paths listed under Affected files; no completed
  task 0019, architecture, build, compiler, Training Java, backend, global-roadmap, or other-module
  path belongs to this task.
- Tests and validation: The authoritative clean NN run passed 249 tests with zero skips, failures,
  or errors plus warning-free Javadoc. The documentation pass changed no executable Java/test,
  reran final Javadoc, inspected rendered pages, and passed surface, external-use, legacy/import/
  dependency, Markdown, exact-scope, status/frontier, newline, no-index, whitespace, and diff
  gates.
- Documentation impact: Affected Javadocs/package pages, Training API, glossary, and planning
  records are finalized. Tensor/Compile API, architecture/current plan/ADR/tests, Model/Compiler/
  Runtime/Prepare/Engine/backends, Training Java, conformance/integration, result carriers, build
  rules, other modules, completed task 0019, and the global roadmap correctly require no change.
- Javadoc review: Complete; generated pages match the implemented policy, ownership, lifecycle,
  failure/effect, state-default, and provenance boundaries without execution claims.
- Glossary impact: Complete; current terminology and planned boundaries are synchronized.
- Unresolved issues: None.
- Follow-up required: None for task 0020. Draft NN 0020A–0020C and 0021–0024 remain separate work.

Status: Complete
