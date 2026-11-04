# Explicit [graph](../../glossary.md#graph) [RNG](../../glossary.md#graph-rng-state--graphrngstate) and [dropout](../../glossary.md#dropout--inverted-dropout)

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

**[Reader model](../../glossary.md#term-reader-model).** RNG means **[random-number generator](../../glossary.md#graph-rng-state--graphrngstate)**. Its state identifies where [sampling](../../glossary.md#term-sampling-machine-learning-and-hyperparameters)
continues, rather than representing a sampled value itself. A **key** names a stream or domain;
a **counter** names a logical [sample](../../glossary.md#term-sample-machine-learning-and-hyperparameters) position. Passing the returned next state expresses
successive intervals; reusing the same state branches from the same interval.
**Dropout** randomly suppresses values during training. **Inverted** dropout rescales kept values
to preserve the ideal average over repeated draws. Its keep [mask](../../glossary.md#term-boolean-mask) records the decisions needed
again during [gradient](../../glossary.md#term-gradient) calculation, so backward work need not sample a new mask.
These [model](../../glossary.md#term-model) concepts do not select a portable random-bit algorithm or promise identical samples
across [backend](../../glossary.md#backend) implementations.

<a id="explicit-graph-rng-state"></a>
### Explicit [graph RNG state](../../glossary.md#graph-rng-state--graphrngstate)

[`GraphRngState`](../../glossary.md#graph-rng-state--graphrngstate) is the implemented public boundary for one explicit [RNG state](../../glossary.md#graph-rng-state--graphrngstate) [occurrence](../../glossary.md#term-occurrence) in a
[Tensor](../../glossary.md#tensor) expression [graph](../../glossary.md#graph). It is distinct from [eager](../../glossary.md#term-eager-evaluation) `TensorRandoms`: eager initialization consumes
a caller-owned [JDK](../../glossary.md#term-java-development-kit-jdk) `RandomGenerator` immediately and returns [host](../../glossary.md#term-host)-backed [leaf](../../glossary.md#term-leaf) data, whereas graph
state records storage-free semantics for a future prepared implementation to consume.

The public initializer is `GraphRngState.initial(long key, long counter)`. Both [arguments](../../glossary.md#term-argument) are raw
unsigned 64-bit words carried by Java `long`; every bit pattern is valid. Signed decimal display
and signed comparison do not define their meaning. The key identifies a caller-selected stream or
domain. The counter identifies the next abstract logical [sample](../../glossary.md#term-sample-machine-learning-and-hyperparameters) position. A future consuming
[operation](../../glossary.md#operation) retains the key and advances the counter modulo `2^64` by that operation's documented
logical draw count.

Each call privately wraps one fresh, unlabeled, storage-free Tensor with exact [descriptor](../../glossary.md#term-descriptor)
[`INT64`](../../glossary.md#data-type--datatype), `Shape.of(2)`, unresolved [layout](../../glossary.md#layout), and `requiresGrad == false`. Its
`GraphRngKind.INITIAL_STATE` [producer](../../glossary.md#term-producer) has `GraphRngStateAttrs(key, counter)`, no inputs, one output,
and [provenance](../../glossary.md#provenance) output index zero. [Lane](../../glossary.md#term-lane) zero conceptually carries the key bits and lane one the
counter bits. The wrapper intentionally provides no public Tensor, lane, storage, mutation,
generator, split, or copy accessor, so those lanes are not an invitation to apply numerical Tensor
operations to state.

Instances use object-[identity](../../glossary.md#term-identity-meanings) equality. Two calls with equal words request replay-equivalent
abstract stream positions but create distinct state, Tensor, producer, and identifier
occurrences. The wrapper is shallowly [immutable](../../glossary.md#term-immutable) and may be shared, but it performs no
[synchronization](../../glossary.md#term-synchronization). Branching one state to multiple future consumers deliberately reuses the same
abstract interval; sequential use threads the returned next state instead.

<a id="complete-current-state-example"></a>
#### Complete current-state example

<a id="goal-and-inputs-10"></a>
##### Goal and inputs

Create two expression occurrences at the same explicit abstract position. The key bits are
`0x1234`, and the counter bits are zero.

```java
import io.github.pho001.synaptik.model.tensor.GraphRngState;

public final class GraphRngStateExample {
    public static void main(String[] args) {
        GraphRngState start = GraphRngState.initial(0x1234L, 0L);
        GraphRngState replay = GraphRngState.initial(0x1234L, 0L);

        System.out.println(start == replay);
        System.out.println(start.equals(replay));
    }
}
```

<a id="meaningful-lines-7"></a>
##### Meaningful lines

- Both initializer calls retain the same raw key/counter attributes, so they request the same
  abstract stream position.
- Each call creates a fresh expression occurrence. Both reference and inherited object equality
  therefore report that the wrappers are distinct.<span data-glossary-navigation="true"> ([occurrence glossary](../../glossary.md#term-occurrence))</span>

<a id="result-and-interpretation-10"></a>
##### Result and interpretation

The program prints:

```text
false
false
```

This proves public construction and [identity](../../glossary.md#term-identity-meanings) equality. It does not expose the private state
[Tensor](../../glossary.md#tensor), [sample](../../glossary.md#term-sample-machine-learning-and-hyperparameters) a value, choose a random algorithm, or prove a cross-[backend](../../glossary.md#backend) bitstream.

<a id="explicit-state-dropout-construction"></a>
### Explicit-state [dropout](../../glossary.md#dropout--inverted-dropout) construction

`Tensor.dropout(double probability, GraphRngState state)` is the implemented [model](../../glossary.md#term-model)-construction
boundary for training-only [inverted dropout](../../glossary.md#dropout--inverted-dropout). It accepts only a floating receiver and a [finite](../../glossary.md#term-finite) drop
probability in `[0.0, 1.0)`. The result is a `DropoutResult(output, nextState)`; it deliberately has
no public [mask](../../glossary.md#term-boolean-mask) component.

One call creates one exact [producer](../../glossary.md#term-producer) with these ordered positions:

| Slot | Role | [Descriptor](../../glossary.md#term-descriptor) | Publicly returned |
|---:|---|---|---|
| input 0 | input value | Exact input floating type and [Shape](../../glossary.md#shape) | Existing receiver |
| input 1 | [RNG state](../../glossary.md#graph-rng-state--graphrngstate) | Private `INT64 Shape[2]` state [Tensor](../../glossary.md#tensor) | Through [opaque](../../glossary.md#term-opaque) input state only |
| output 0 | dropped value | Input type/Shape/[gradient](../../glossary.md#term-gradient) eligibility; unresolved [layout](../../glossary.md#layout) | `DropoutResult.output()` |
| output 1 | auxiliary keep mask | [`BOOL`](../../glossary.md#data-type--datatype), exact input Shape, no gradients, unresolved layout | No |
| output 2 | next RNG state | `INT64 Shape[2]`, no gradients, unresolved layout | `DropoutResult.nextState()` |

The table shows producer roles, not storage. Every output is unlabeled and storage-free. The
multi-output factory creates three fresh Tensor wrappers and three Tensor IDs, one for each output
slot. The slot-one wrapper is discarded from the public result, while the producer retained by
slots zero and two still describes its BOOL mask position for later [compiler](../../glossary.md#term-compiler) [capture](../../glossary.md#term-capture-meanings).

For logical [element](../../glossary.md#term-element) position `i`, the [operation](../../glossary.md#operation) means one abstract draw `u[i]` in `[0, 1)` and:

```text
keep[i] = u[i] >= probability
output[i] = keep[i] ? input[i] / (1 - probability) : +0
```

This is inverted dropout: kept values are scaled so their ideal expectation matches the input.
Dropped values are [positive zero](../../glossary.md#nan-infinity-and-signed-zero) even for [negative zero](../../glossary.md#nan-infinity-and-signed-zero), [NaN](../../glossary.md#term-nan), or infinite input. Kept [signed zero](../../glossary.md#term-signed-zero)
and [infinity](../../glossary.md#term-infinity) retain their sign; kept NaN remains NaN without a [payload](../../glossary.md#artifact-envelope-and-atomic-publication) promise. Construction does
not evaluate this formula or select its finite-[precision](../../glossary.md#term-precision) algorithm.

The current [Metal](../../glossary.md#term-metal) [backend](../../glossary.md#backend) executes `GraphRngState.initial` and [canonical](../../glossary.md#term-canonical-form)
[FLOAT32](../../glossary.md#term-float32) dropout for static [rank](../../glossary.md#term-rank) `0..16` geometry whose extents, element
count, [referenced span](../../glossary.md#referenced-element-span), and [dispatch](../../glossary.md#term-dispatch) width fit unsigned 32 bits. Both rows select one fixed custom
program. Its private `SYNAPTIK_METAL_SPLITMIX64_COUNTER_V1` replay [identity](../../glossary.md#term-identity-meanings) derives each word from
the raw key, counter, and [row-major](../../glossary.md#term-row-major) ordinal; draw membership compares the top 53 bits with exact
integer $\lceil p2^{53}\rceil$. The FLOAT32 complement is narrowed after [binary64](../../glossary.md#floating-point-precision-and-rounding) subtraction, and
kept values use one FLOAT32 division followed by multiplication. This is not a public generator,
portable cross-backend stream, entropy source, or random-quality promise.

Metal retains all three outputs in ordinary [run](../../glossary.md#term-run)-local storage, including when only the value is a
public [target](../../glossary.md#term-target). Compiler-generated first-order backward consumes the exact saved mask through
`where(mask, gradient / typed(1 - probability), positiveZero)` and never resamples or differentiates
mask/state. Repeated calls, independent sessions, and concurrent invocations share no [mutable](../../glossary.md#term-mutable) RNG
state. The [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [evaluation](../../glossary.md#term-evaluation) path remains an identity bypass and constructs no dropout producer.

<a id="complete-dropout-construction-example"></a>
#### Complete [dropout](../../glossary.md#dropout--inverted-dropout)-construction example

<a id="goal-and-inputs-11"></a>
##### Goal and inputs

Build two replay-equivalent [dropout](../../glossary.md#dropout--inverted-dropout) occurrences and one explicitly threaded successor for a
storage-free [`FLOAT32`](../../glossary.md#term-float32) input with [Shape](../../glossary.md#shape) `[2, 3]`. The probability is `0.1`, the key is `0x1234`,
and the initial counter is zero.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.DropoutResult;
import io.github.pho001.synaptik.model.tensor.GraphRngState;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProducer;
import java.util.Optional;

public final class DropoutConstructionExample {
    public static void main(String[] args) {
        Shape shape = Shape.of(2, 3);
        Tensor activations = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.empty(), true));
        GraphRngState start = GraphRngState.initial(0x1234L, 0L);
        DropoutResult first = activations.dropout(0.1d, start);
        DropoutResult replay = activations.dropout(
                0.1d, GraphRngState.initial(0x1234L, 0L));
        DropoutResult second = first.output().dropout(0.1d, first.nextState());

        TensorProducer producer = first.output().provenance().orElseThrow().producer();
        System.out.println(producer.inputs().size());
        System.out.println(producer.outputCount());
        System.out.println(producer.outputDescriptors().get(1).dataType());
        System.out.println(first.output().descriptor().shape() == shape);
        System.out.println(first.output() == replay.output());
        System.out.println(second.output() == first.output());
    }
}
```

<a id="meaningful-lines-8"></a>
##### Meaningful lines

- `first` and `replay` use equal key/counter words, so eventual conforming execution requests the
  same abstract interval. They remain distinct Tensor and producer occurrences.<span data-glossary-navigation="true"> ([Tensor glossary](../../glossary.md#tensor); [producer glossary](../../glossary.md#term-producer))</span>
- `second` consumes `first.nextState()`, which expresses sequential non-overlap without mutating
  `start` or storing a hidden generator.
- The producer exposes two inputs and three descriptors. Descriptor slot one is the non-public
  BOOL auxiliary keep mask.<span data-glossary-navigation="true"> ([descriptors glossary](../../glossary.md#term-descriptor); [BOOL glossary](../../glossary.md#data-type--datatype); [mask glossary](../../glossary.md#term-boolean-mask))</span>

<a id="result-and-interpretation-11"></a>
##### Result and interpretation

The program prints:

```text
2
3
BOOL
true
false
false
```

This proves the current public construction surface, output-slot description, [Shape](../../glossary.md#shape) [identity](../../glossary.md#term-identity-meanings), and
fresh [occurrence](../../glossary.md#term-occurrence) identity. It does not [sample](../../glossary.md#term-sample-machine-learning-and-hyperparameters) a [mask](../../glossary.md#term-boolean-mask), calculate output values, expose the state
[Tensor](../../glossary.md#tensor), [capture](../../glossary.md#term-capture-meanings) a [graph](../../glossary.md#graph), construct a [gradient](../../glossary.md#term-gradient), or execute a [backend](../../glossary.md#backend).

<a id="state-advancement-branching-and-edge-cases"></a>
#### State advancement, branching, and edge cases

For a bound logical [element](../../glossary.md#term-element) count `N`, eventual conforming execution consumes counter positions
`counter` through `counter + N - 1` and returns the same key with `counter + N modulo 2^64`.
[Scalar](../../glossary.md#term-scalar) input uses `N == 1`. A fully static [Shape](../../glossary.md#shape) uses its mathematical [extent](../../glossary.md#term-extent) product, including
products too large for signed `long`; a dynamic or symbolic Shape uses the non-negative count after
its extents are bound. Any bound zero extent gives `N == 0`, so output and [mask](../../glossary.md#term-boolean-mask) are empty and the
next state has the same abstract key/counter value while remaining a distinct expression
[occurrence](../../glossary.md#term-occurrence).

Both signed probability-zero values are retained. Because every draw lies in `[0, 1)`, probability
zero keeps every value, but a non-empty occurrence still consumes one draw per element and advances
state. Reusing one state for two calls intentionally branches over the same interval. Threading
each returned `nextState` expresses sequential intervals. Sharing the [immutable](../../glossary.md#term-immutable) wrapper does not
synchronize execution.

Input eligibility is checked before probability, and probability before state nullity. Integral or
[BOOL](../../glossary.md#data-type--datatype) input fails; [NaN](../../glossary.md#term-nan), [infinity](../../glossary.md#term-infinity), negative probability, and probability at least one fail; then a
null state fails. These local failures allocate no [Tensor](../../glossary.md#tensor) ID. Successful construction performs no
[sampling](../../glossary.md#term-sampling-machine-learning-and-hyperparameters), reads no storage, and mutates neither input nor state.

The raw words, fixed [descriptor](../../glossary.md#term-descriptor), occurrence ordering, and modular advancement meaning are portable
[model](../../glossary.md#term-model) semantics. No
portable pseudorandom-number-generator algorithm, key schedule, counter-to-bits function,
floating conversion, or bitstream is selected, so equal state does not promise bitwise [samples](../../glossary.md#term-sample-machine-learning-and-hyperparameters)
across [backends](../../glossary.md#backend), routes, providers, or versions. Until an algorithm is selected, sampled replay is
bounded to the same conforming prepared implementation and configuration. A future [graph](../../glossary.md#graph)
serializer must preserve both raw words losslessly, but no byte encoding, parser, [schema](../../glossary.md#term-schema) version,
or stable enum [token](../../glossary.md#term-token) exists today. [Compiler](../../glossary.md#term-compiler) [capture](../../glossary.md#term-capture-meanings) and [serialization](../../glossary.md#term-serialization) policy, prepared state,
[materialization](../../glossary.md#materialization), sampling, [kernels](../../glossary.md#kernel), [runtime](../../glossary.md#term-runtime) execution, [gradients](../../glossary.md#term-gradient), and backend support remain in
their owning future layers.


---

[Previous topic](random-initialization.md) · [Next topic](import.md)
