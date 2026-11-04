# Ranges and [eager](../../glossary.md#term-eager-evaluation) random initialization

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

## There are two call boundaries during initialization

The application calls `TensorRandoms` with a generator object it created and configured.
`TensorRandoms` then becomes the caller of that object's `nextGaussian`, bounded integer,
or double-draw method, once per logical position as specified below. It uses the returned
primitive values immediately to fill a temporary carrier, then flat import copies that carrier
into the returned Tensor's independent heap storage. The generator itself is borrowed only during
the call and is not retained, reset, synchronized or closed; its advancement remains observable
to the application even if a later step fails.
Contrast [graph RNG state](graph-randomness.md), which describes future draws without consuming
a Java generator, and [import ownership](import.md) for the final copy.

**[Reader model](../../glossary.md#term-reader-model).** **Eager** initialization [samples](../../glossary.md#term-sample-machine-learning-and-hyperparameters) and stores data immediately, unlike a [graph](../../glossary.md#graph)
expression describing future [sampling](../../glossary.md#term-sampling-machine-learning-and-hyperparameters). A **[distribution](../../glossary.md#term-distribution)** specifies how draws are spread:
normal ([Gaussian](../../glossary.md#term-gaussian-distribution)) draws cluster around a [mean](../../glossary.md#term-mean) with spread set by [standard deviation](../../glossary.md#term-standard-deviation);
uniform draws use an equal-density interval before storage [rounding](../../glossary.md#term-rounding); [Bernoulli](../../glossary.md#term-bernoulli-distribution) draws produce
true with a specified probability and false otherwise.
An **inclusive** bound may be emitted; an **exclusive** bound may not be emitted by a conforming
source in its sampling format. [Narrowing](../../glossary.md#term-narrowing) to the stored format can change that interval
relationship, as the uniform example shows. A **[seed](../../glossary.md#term-seed)** initializes generator state; [reproducibility](../../glossary.md#artifact-manifest-and-source-provenance)
also depends on the generator implementation and the sequence of calls, not just that seed.

<a id="complete-integer-range-example"></a>
### Complete integer-range example

<a id="goal-and-inputs-5"></a>
#### Goal and inputs

Create the [INT32](../../glossary.md#data-type--datatype) range `[1, 4, 7]` from inclusive start `1`, exclusive end `8`, and step `3`.

```java
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Arrays;
import java.util.Optional;

public final class IntegerRangeExample {
    public static void main(String[] args) {
        // Exclusive end 8 is a bound, not an extra emitted element.
        Tensor range = TensorFactory.range(
                1, 8, 3, Optional.of("indices"));

        int[] rangeData = (int[]) range.hostStorage()
                .orElseThrow().segment().heapBase().orElseThrow();

        System.out.println(range.descriptor().dataType());
        System.out.println(range.descriptor().shape());
        System.out.println(Arrays.toString(rangeData));
    }
}
```

<a id="meaningful-lines-2"></a>
#### Meaningful lines

- The `int` range overload fixes the result [data type](../../glossary.md#data-type--datatype) as [`INT32`](../../glossary.md#data-type--datatype). Ceiling division of the distance
  by step gives three values, and the exclusive bound `8` is not emitted.
- The result shape is rank one with one position for each of those three values. Heap-base
  inspection only makes existing raw storage observable; it is not a typed Tensor access or
  export API.<span data-glossary-navigation="true"> ([shape glossary](../../glossary.md#shape); [rank glossary](../../glossary.md#term-rank); [Heap glossary](../../glossary.md#term-heap); [Tensor glossary](../../glossary.md#tensor); [API glossary](../../glossary.md#term-application-programming-interface-api))</span>

<a id="result-and-interpretation-5"></a>
#### Result and interpretation

The program prints:

```text
INT32
Shape[3]
[1, 4, 7]
```

The result demonstrates range typing, exclusive-end semantics, [canonical](../../glossary.md#term-canonical-form) dense [shape](../../glossary.md#shape), and copied
storage. It does not provide an empty or floating range, implicit conversion, a prefix [fixture](../../glossary.md#term-fixture)
[API](../../glossary.md#term-application-programming-interface-api), typed access/export, [provenance](../../glossary.md#provenance), or execution. Normal random creation is the separate
explicit-source `TensorRandoms` method described next.

<a id="failures-and-useful-variations-4"></a>
#### Failures and useful variations

- `range(0, 3, -1, label)` fails because the negative step cannot advance toward the larger end.
- Equal start and end values fail because current range construction is deliberately non-empty.
- Counts above `Integer.MAX_VALUE` fail before [carrier](../../glossary.md#term-storage-carrier), destination, or ID allocation.

`TensorRandoms` owns exactly one public normal method:
`randomNormal(shape, dataType, mean, standardDeviation, randomGenerator, label, requiresGrad)`.
It accepts only fully static Java-array-sized [shapes](../../glossary.md#shape) and every floating type, including [FP8](../../glossary.md#term-fp8).
Both [parameters](../../glossary.md#term-parameter) must be [finite](../../glossary.md#term-finite), and the [standard deviation](../../glossary.md#term-standard-deviation) must be numerically non-negative;
either [signed zero](../../glossary.md#term-signed-zero) is valid. The result has a newly synthesized [canonical](../../glossary.md#term-canonical-form) dense-[contiguous](../../glossary.md#term-contiguous)
[descriptor](../../glossary.md#term-descriptor), independent writable [heap](../../glossary.md#term-heap) storage, the explicit label and [gradient](../../glossary.md#term-gradient) intent, and a new
factory ID.

For every logical [row-major](../../glossary.md#term-row-major) [element](../../glossary.md#term-element), the method calls the exact supplied
`RandomGenerator.nextGaussian()` once, multiplies the returned [binary64](../../glossary.md#floating-point-precision-and-rounding) value by the standard
deviation, and then adds the binary64 [mean](../../glossary.md#term-mean). It does not use [fused multiply-add](../../glossary.md#term-fused-multiply-add-fma). [`FLOAT64`](../../glossary.md#term-float64) stores
the result directly, [`FLOAT32`](../../glossary.md#term-float32) narrows it once to [binary32](../../glossary.md#floating-point-precision-and-rounding), and [`BFLOAT16`](../../glossary.md#term-bfloat16) first narrows to
binary32 and then uses `BFloat16Bits.fromFloat`; [FLOAT16](../../glossary.md#term-float16) and both FP8 types round directly from
binary64. A [scalar](../../glossary.md#term-scalar) consumes one call; an empty shape consumes none. Non-finite converted values
are not rejected after [sampling](../../glossary.md#term-sampling-machine-learning-and-hyperparameters).

The generator is transient caller-owned state. `TensorRandoms` does not select an algorithm or
default source, store a [seed](../../glossary.md#term-seed), retain or replace the generator, synchronize access, or reset,
split, or close it. Equivalent output therefore requires equivalent generator implementation and
initial state, identical [arguments](../../glossary.md#term-argument), and no interfering source use. There is no sequence promise
across algorithms, providers, Java versions, seed expansion, concurrent access, or different
initial states.

<a id="complete-normal-random-example"></a>
### Complete normal-random example

<a id="goal-and-inputs-6"></a>
#### Goal and inputs

Create a `2 × 2` [FLOAT32](../../glossary.md#term-float32) [tensor](../../glossary.md#tensor) from the scripted [Gaussian](../../glossary.md#term-gaussian-distribution) sequence `[-1, 0, 1, 0.5]`, [mean](../../glossary.md#term-mean) `1`,
and [standard deviation](../../glossary.md#term-standard-deviation) `2`. A scripted source makes the transformation and exact call count
observable without introducing a production [seed](../../glossary.md#term-seed) or generator-selection [API](../../glossary.md#term-application-programming-interface-api).

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorRandoms;
import java.util.Arrays;
import java.util.Optional;
import java.util.random.RandomGenerator;

public final class NormalRandomExample {
    private static final class ScriptedGenerator implements RandomGenerator {
        private final double[] values = {-1.0, 0.0, 1.0, 0.5};
        private int calls;

        @Override
        public long nextLong() {
            throw new AssertionError("nextLong is not used");
        }

        @Override
        public double nextGaussian() {
            return values[calls++];
        }
    }

    public static void main(String[] args) {
        ScriptedGenerator source = new ScriptedGenerator();
        // Sampling advances this exact source now, unlike building a random graph expression.
        Tensor tensor = TensorRandoms.randomNormal(
                Shape.of(2, 2),
                DataType.FLOAT32,
                1.0,
                2.0,
                source,
                Optional.of("samples"),
                true);

        float[] data = (float[]) tensor.hostStorage()
                .orElseThrow().segment().heapBase().orElseThrow();

        System.out.println(Arrays.toString(data));
        System.out.println(source.calls);
        System.out.println(tensor.descriptor().layout().orElseThrow().kind());
    }
}
```

<a id="meaningful-lines-3"></a>
#### Meaningful lines

- The source implements only the methods needed by `RandomGenerator` and overrides
  `nextGaussian()` with four known values. `TensorRandoms` uses that exact object and does not call
  `nextLong()` or select another generator.
- Each ideal finite-real output is $x_i=\mu+\sigma g_i$, where $g_i$ is the scripted Gaussian
  draw, $\mu=1$ the mean and $\sigma=2$ the standard deviation. For $g_0=-1$,
  $x_0=1+2(-1)=-1$; the four results are $-1,1,3,2$ in row-major order.
  Actual evaluation multiplies in binary64, then adds in binary64, then narrows to FLOAT32;
  the real identity does not authorize fused multiply-add or a changed rounding boundary.
- Heap-base inspection makes the already implemented raw storage observable for the example; it
  is not a typed Tensor access or export API.<span data-glossary-navigation="true"> ([Heap glossary](../../glossary.md#term-heap); [Tensor glossary](../../glossary.md#tensor); [API glossary](../../glossary.md#term-application-programming-interface-api))</span>

<a id="result-and-interpretation-6"></a>
#### Result and interpretation

The program prints:

```text
[-1.0, 1.0, 3.0, 2.0]
4
DENSE_CONTIGUOUS
```

The result proves transformation order, one source call per [element](../../glossary.md#term-element), [FLOAT32](../../glossary.md#term-float32) conversion, and
[canonical](../../glossary.md#term-canonical-form) dense [layout](../../glossary.md#layout) for this scripted input. It does not promise statistical properties of a
custom generator, cross-generator [reproducibility](../../glossary.md#artifact-manifest-and-source-provenance), a Synaptik [seed](../../glossary.md#term-seed) [API](../../glossary.md#term-application-programming-interface-api), continuous-uniform,
integral, or [Bernoulli](../../glossary.md#term-bernoulli-distribution) behavior, typed export, or [runtime](../../glossary.md#term-runtime)/[backend](../../glossary.md#backend) random execution.

<a id="failures-and-useful-variations-5"></a>
#### Failures and useful variations

- A dynamic shape, non-floating data type, non-finite mean, negative or non-finite deviation, or
  logical count above <span data-glossary-navigation="true"> ([data type glossary](../../glossary.md#data-type--datatype); [mean glossary](../../glossary.md#term-mean))</span>`Integer.MAX_VALUE` fails before [sampling](../../glossary.md#term-sampling-machine-learning-and-hyperparameters) or ID allocation.
- If `nextGaussian()` throws, preceding calls remain consumed, but no destination or ID exists.
- A blank present label is detected by delegated Tensor validation after all samples, destination
  allocation, and ID allocation; the ID is consumed and no state is rolled back.<span data-glossary-navigation="true"> ([Tensor glossary](../../glossary.md#tensor); [validation glossary](../../glossary.md#term-validation))</span>
- Two equivalent caller-created generators can produce equivalent tensors when their initial
  states and all method arguments match and neither source has interfering use.

`TensorRandoms` also owns exactly one public continuous-uniform method:
`randomUniform(shape, dataType, lowerBoundInclusive, upperBoundExclusive, randomGenerator, label,
requiresGrad)`. It accepts only fully static Java-array-sized [shapes](../../glossary.md#shape) and every floating type,
including [FP8](../../glossary.md#term-fp8). Both [binary64](../../glossary.md#floating-point-precision-and-rounding) bounds must be [finite](../../glossary.md#term-finite), and the lower bound must be strictly less than the
upper bound. The result has a newly synthesized [canonical](../../glossary.md#term-canonical-form) dense-[contiguous](../../glossary.md#term-contiguous) [descriptor](../../glossary.md#term-descriptor), independent
writable [heap](../../glossary.md#term-heap) storage, the explicit label and [gradient](../../glossary.md#term-gradient) intent, and a new factory ID.

For every logical [row-major](../../glossary.md#term-row-major) [element](../../glossary.md#term-element), the method calls the exact supplied
`RandomGenerator.nextDouble(lowerBoundInclusive, upperBoundExclusive)` once. A conforming source
returns a binary64 value in the [half-open interval](../../glossary.md#term-half-open-interval) `[lowerBoundInclusive, upperBoundExclusive)`.
[`FLOAT64`](../../glossary.md#term-float64) stores that value directly, [`FLOAT32`](../../glossary.md#term-float32) narrows it once to [binary32](../../glossary.md#floating-point-precision-and-rounding), [`BFLOAT16`](../../glossary.md#term-bfloat16) first
narrows to binary32 and then uses `BFloat16Bits.fromFloat`, and [FLOAT16](../../glossary.md#term-float16)/both FP8 types round directly from the
binary64 [sample](../../glossary.md#term-sample-machine-learning-and-hyperparameters) without an intermediate binary32 [rounding](../../glossary.md#term-rounding). The [half-open](../../glossary.md#elementwise-unary-binary-and-interval-vocabulary) promise applies before
[narrowing](../../glossary.md#term-narrowing): a stored low-[precision](../../glossary.md#term-precision) or FLOAT32 value may equal the corresponding narrowed upper
bound or may round to a lower representable value. `TensorRandoms` does not clamp, resample, or
post-validate a custom non-conforming source result. A [scalar](../../glossary.md#term-scalar) consumes one bounded call; an empty shape consumes
none.

The source has the same [ownership](../../glossary.md#term-ownership-meanings) and [reproducibility](../../glossary.md#artifact-manifest-and-source-provenance) boundary as normal creation. The caller
creates, configures, [seeds](../../glossary.md#commit-release-distribution-log-and-seed-meanings), owns, and advances it, and must provide exclusive or otherwise safe
access. `TensorRandoms` does not select an algorithm or default source, store a [seed](../../glossary.md#term-seed), retain or replace
the generator, synchronize access, or reset, split, or close it. Equivalent output requires an
equivalent generator implementation and initial state, identical [arguments](../../glossary.md#term-argument), and no interfering
source use. No sequence is promised across algorithms, providers, Java versions, seed expansion,
concurrent use, or different initial states.

<a id="complete-continuous-uniform-example"></a>
### Complete continuous-uniform example

<a id="goal-and-inputs-7"></a>
#### Goal and inputs

Create a `2 × 2` [FLOAT32](../../glossary.md#term-float32) [tensor](../../glossary.md#tensor) from four known [binary64](../../glossary.md#floating-point-precision-and-rounding) values in `[-1, 1)`. The final source
value is `Math.nextDown(1.0)`, which is strictly below the binary64 upper bound but rounds to
[binary32](../../glossary.md#floating-point-precision-and-rounding) `1.0f`. A scripted source makes the exact bounded-call [arguments](../../glossary.md#term-argument), call count, and
[narrowing](../../glossary.md#term-narrowing) caveat observable without adding a production [seed](../../glossary.md#term-seed) or source-selection [API](../../glossary.md#term-application-programming-interface-api).

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorRandoms;
import java.util.Arrays;
import java.util.Optional;
import java.util.random.RandomGenerator;

public final class UniformRandomExample {
    private static final class ScriptedGenerator implements RandomGenerator {
        private final double[] values = {-1.0, 0.0, 0.5, Math.nextDown(1.0)};
        private int calls;

        @Override
        public long nextLong() {
            throw new AssertionError("nextLong is not used");
        }

        @Override
        public double nextDouble() {
            throw new AssertionError("unbounded nextDouble is not used");
        }

        @Override
        public double nextDouble(double origin, double bound) {
            if (origin != -1.0 || bound != 1.0) {
                throw new AssertionError("unexpected bounds");
            }
            return values[calls++];
        }
    }

    public static void main(String[] args) {
        ScriptedGenerator source = new ScriptedGenerator();
        // The sampling interval is binary64; stored FLOAT32 values may round to its upper bound.
        Tensor tensor = TensorRandoms.randomUniform(
                Shape.of(2, 2),
                DataType.FLOAT32,
                -1.0,
                1.0,
                source,
                Optional.of("samples"),
                true);

        float[] data = (float[]) tensor.hostStorage()
                .orElseThrow().segment().heapBase().orElseThrow();

        System.out.println(Arrays.toString(data));
        System.out.println(source.calls);
        System.out.println(tensor.descriptor().layout().orElseThrow().kind());
    }
}
```

<a id="meaningful-lines-4"></a>
#### Meaningful lines

- The source overrides only the bounded `nextDouble(origin, bound)` path used by `TensorRandoms` and
  rejects both `nextLong()` and unbounded `nextDouble()`. It also verifies the unchanged `-1.0`
  and `1.0` [arguments](../../glossary.md#term-argument) on every call.
- The first three [binary64](../../glossary.md#floating-point-precision-and-rounding) values remain `-1.0f`, `0.0f`, and `0.5f`. The final conforming value is
  below binary64 `1.0`, but one [binary32](../../glossary.md#floating-point-precision-and-rounding) [narrowing](../../glossary.md#term-narrowing) rounds it to stored `1.0f`.
- Heap-base inspection makes the already implemented raw storage observable for the example; it
  is not a typed Tensor access or export API.<span data-glossary-navigation="true"> ([Heap glossary](../../glossary.md#term-heap); [Tensor glossary](../../glossary.md#tensor); [API glossary](../../glossary.md#term-application-programming-interface-api))</span>

<a id="result-and-interpretation-7"></a>
#### Result and interpretation

The program prints:

```text
[-1.0, 0.0, 0.5, 1.0]
4
DENSE_CONTIGUOUS
```

The result proves the exact bounded-call path, one call per [row-major](../../glossary.md#term-row-major) [element](../../glossary.md#term-element), [FLOAT32](../../glossary.md#term-float32) [narrowing](../../glossary.md#term-narrowing),
the narrowed-upper-bound caveat, and [canonical](../../glossary.md#term-canonical-form) dense [layout](../../glossary.md#layout) for this scripted input. It does not
promise statistical quality of a custom generator, a closed [binary64](../../glossary.md#floating-point-precision-and-rounding) interval, cross-generator
[reproducibility](../../glossary.md#artifact-manifest-and-source-provenance), a Synaptik [seed](../../glossary.md#term-seed) [API](../../glossary.md#term-application-programming-interface-api), integral or [Bernoulli](../../glossary.md#term-bernoulli-distribution) [sampling](../../glossary.md#term-sampling-machine-learning-and-hyperparameters), typed export, or
[runtime](../../glossary.md#term-runtime)/[backend](../../glossary.md#backend) random execution.

<a id="failures-and-useful-variations-6"></a>
#### Failures and useful variations

- A dynamic shape, non-floating data type, non-finite bound, unordered bounds, or logical count
  above <span data-glossary-navigation="true"> ([data type glossary](../../glossary.md#data-type--datatype); [finite glossary](../../glossary.md#term-finite))</span>`Integer.MAX_VALUE` fails before [sampling](../../glossary.md#term-sampling-machine-learning-and-hyperparameters) or ID allocation.
- If bounded `nextDouble(lower, upper)` throws, preceding calls remain consumed, but no destination
  or ID exists.
- A blank present label is detected by delegated Tensor validation after all samples, destination
  allocation, and ID allocation; the ID is consumed and no state is rolled back.<span data-glossary-navigation="true"> ([Tensor glossary](../../glossary.md#tensor); [samples glossary](../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- Two equivalent caller-created generators can produce equivalent tensors when their initial
  states and all method arguments match and neither source has interfering use.

`TensorRandoms` owns exactly two public bounded-integral overloads named `randomInt`. Primitive
`int` bounds produce [`INT32`](../../glossary.md#data-type--datatype), while primitive `long` bounds produce [`INT64`](../../glossary.md#data-type--datatype). Both take a fully
static Java-array-sized [shape](../../glossary.md#shape), inclusive origin, exclusive bound, caller-owned `RandomGenerator`,
and optional label. The origin must be strictly less than the bound, including for an empty
result. The synthesized [descriptor](../../glossary.md#term-descriptor) is [canonical](../../glossary.md#term-canonical-form) dense-[contiguous](../../glossary.md#term-contiguous) and always has
`requiresGrad == false`; there is no data-type or [gradient](../../glossary.md#term-gradient) [parameter](../../glossary.md#term-parameter) because the bound [carrier](../../glossary.md#term-storage-carrier)
already selects a non-[differentiable](../../glossary.md#derivative-and-chain-rule) result type.

Each logical [row-major](../../glossary.md#term-row-major) [element](../../glossary.md#term-element) is the direct result of exactly one matching bounded call:
`nextInt(origin, bound)` for `INT32` or `nextLong(origin, bound)` for `INT64`. `TensorRandoms` uses no
unbounded draw, modulo reduction, floating arithmetic, [narrowing](../../glossary.md#term-narrowing), [widening](../../glossary.md#term-widening), or alternate carrier.
It fills one exact `int[]` or `long[]` and delegates once to matching flat import. A conforming
source supplies values in the [half-open interval](../../glossary.md#term-half-open-interval) `[origin, bound)` without project-owned modulo
[bias](../../glossary.md#term-bias); custom non-conforming values are copied without post-[validation](../../glossary.md#term-validation). A [scalar](../../glossary.md#term-scalar) consumes one
bounded call and a valid empty result consumes none.

The source has the same [ownership](../../glossary.md#term-ownership-meanings) and bounded [reproducibility](../../glossary.md#artifact-manifest-and-source-provenance) contract as the floating random
methods. The caller configures, [seeds](../../glossary.md#commit-release-distribution-log-and-seed-meanings), owns, advances, and provides safe access to it. `TensorRandoms`
does not substitute, retain, synchronize, reset, split, or close the source. Equivalent output
requires equivalent generator implementation and initial state, identical [arguments](../../glossary.md#term-argument), and no
interfering source use. The one-call guarantee concerns bounded method invocations rather than the
generator's internal random-bit consumption.

The bounds use the result carrier itself. `Integer.MAX_VALUE` or `Long.MAX_VALUE` can therefore be
an exclusive bound but cannot be emitted, and the mathematical exclusive bound one greater than
the carrier maximum cannot be expressed. No unbounded or full-domain convenience is supplied.
Negative and mixed-sign intervals are supported when strictly ordered.

<a id="complete-bounded-integral-example"></a>
### Complete bounded-integral example

<a id="goal-and-inputs-8"></a>
#### Goal and inputs

Create a `2 × 2` [INT32](../../glossary.md#data-type--datatype) [tensor](../../glossary.md#tensor) from the scripted values `[-3, -1, 0, 4]` over `[-3, 5)`. The
scripted source verifies that every call receives the exact bounds and that neither an unbounded
draw nor a production [seed](../../glossary.md#term-seed)/source-selection [API](../../glossary.md#term-application-programming-interface-api) is involved.

```java
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorRandoms;
import java.util.Arrays;
import java.util.Optional;
import java.util.random.RandomGenerator;

public final class IntegralRandomExample {
    private static final class ScriptedGenerator implements RandomGenerator {
        private final int[] values = {-3, -1, 0, 4};
        private int calls;

        @Override
        public long nextLong() {
            throw new AssertionError("unbounded nextLong is not used");
        }

        @Override
        public int nextInt(int origin, int bound) {
            if (origin != -3 || bound != 5) {
                throw new AssertionError("unexpected bounds");
            }
            return values[calls++];
        }
    }

    public static void main(String[] args) {
        ScriptedGenerator source = new ScriptedGenerator();
        // Primitive int bounds select both the bounded generator method and result carrier.
        Tensor tensor = TensorRandoms.randomInt(
                Shape.of(2, 2), -3, 5, source, Optional.of("indices"));

        int[] data = (int[]) tensor.hostStorage()
                .orElseThrow().segment().heapBase().orElseThrow();

        System.out.println(Arrays.toString(data));
        System.out.println(tensor.descriptor().dataType());
        System.out.println(tensor.descriptor().requiresGrad());
        System.out.println(source.calls);
    }
}
```

<a id="meaningful-lines-5"></a>
#### Meaningful lines

- Primitive `int` bounds select [`INT32`](../../glossary.md#data-type--datatype); the call has no [`DataType`](../../glossary.md#data-type--datatype) or [gradient](../../glossary.md#term-gradient) [argument](../../glossary.md#term-argument).
- The source rejects unbounded `nextLong()` and validates `-3` and `5` on every bounded
  `nextInt(origin, bound)` invocation. `TensorRandoms` stores each returned value directly.
- Heap-base inspection makes the implemented raw storage observable for the example; it is not a
  typed Tensor access or export API.<span data-glossary-navigation="true"> ([Heap glossary](../../glossary.md#term-heap); [Tensor glossary](../../glossary.md#tensor); [API glossary](../../glossary.md#term-application-programming-interface-api))</span>

<a id="result-and-interpretation-8"></a>
#### Result and interpretation

The program prints:

```text
[-3, -1, 0, 4]
INT32
false
4
```

The result proves inferred [INT32](../../glossary.md#data-type--datatype) type, false [gradient](../../glossary.md#term-gradient) intent, exact bounded-call order and count,
direct [carrier](../../glossary.md#term-storage-carrier) storage, and [half-open](../../glossary.md#term-half-open-interval) boundary values for this scripted input. The primitive
`long` overload follows the same contract with `nextLong(origin, bound)` and an [`INT64`](../../glossary.md#data-type--datatype) result. The
example does not promise statistical quality for a custom generator, full-domain [sampling](../../glossary.md#term-sampling-machine-learning-and-hyperparameters), typed
export, a random [graph](../../glossary.md#graph) [Operation](../../glossary.md#operation), or [runtime](../../glossary.md#term-runtime)/[backend](../../glossary.md#backend) generation.

<a id="failures-and-useful-variations-7"></a>
#### Failures and useful variations

- A dynamic [shape](../../glossary.md#shape) or logical count above `Integer.MAX_VALUE` fails before sampling or ID
  allocation. Checked element-count overflow remains an <span data-glossary-navigation="true"> ([overflow glossary](../../glossary.md#term-overflow))</span>`ArithmeticException`.
- Equal or reversed bounds fail even for an empty result, before source-[carrier](../../glossary.md#term-storage-carrier) allocation.
- If the bounded source method throws, earlier calls remain consumed, but no destination or ID
  exists.
- A blank label fails after all bounded calls, destination allocation, and ID allocation; the ID
  is consumed and no state is rolled back.

`TensorRandoms` owns exactly one public [Bernoulli](../../glossary.md#term-bernoulli-distribution) method:
`randomBernoulli(shape, probability, randomGenerator, label)`. It creates a [canonical](../../glossary.md#term-canonical-form) dense [BOOL](../../glossary.md#data-type--datatype)
[tensor](../../glossary.md#tensor) with [gradients](../../glossary.md#term-gradient) disabled. The shape must be fully static, its checked logical count must fit
a Java array, and the [binary64](../../glossary.md#floating-point-precision-and-rounding) probability must be [finite](../../glossary.md#term-finite) and in the closed interval `[0, 1]`.
Positive and [negative zero](../../glossary.md#nan-infinity-and-signed-zero) are accepted as zero. No data-type or gradient [argument](../../glossary.md#term-argument) is present
because Bernoulli output is logical non-[differentiable](../../glossary.md#derivative-and-chain-rule) [leaf](../../glossary.md#term-leaf) data; numeric truthiness and output
conversion are outside this method.

For each logical [row-major](../../glossary.md#term-row-major) [element](../../glossary.md#term-element), `TensorRandoms` calls the exact supplied source's unbounded
`nextDouble()` method once and stores canonical byte `1` exactly when `draw < probability`;
otherwise it stores byte `0`. The strict comparison means that a draw equal to the probability is
false. A custom non-conforming draw is compared directly without post-[validation](../../glossary.md#term-validation). Calls are not
skipped at probability zero or one: conforming draws still produce all false or all true values,
respectively, while source advancement stays independent of endpoint optimization. A [scalar](../../glossary.md#term-scalar)
consumes one call and an empty result consumes none.

The source follows the same caller-[ownership](../../glossary.md#term-ownership-meanings) and bounded-[reproducibility](../../glossary.md#artifact-manifest-and-source-provenance) contract as the other
random methods. `TensorRandoms` does not substitute, retain, [seed](../../glossary.md#term-seed), synchronize, reset, split, or close
it. Equivalent output requires equivalent generator implementation and initial state, identical
arguments, and no interfering use. One complete canonical `byte[]` is sampled before one BOOL
flat import; there is no partial Tensor, direct storage construction, or direct ID allocation in
the [sampling](../../glossary.md#term-sampling-machine-learning-and-hyperparameters) helper.

<a id="complete-bernoulli-random-example"></a>
### Complete [Bernoulli](../../glossary.md#term-bernoulli-distribution)-random example

<a id="goal-and-inputs-9"></a>
#### Goal and inputs

Create a `2 × 2` [BOOL](../../glossary.md#data-type--datatype) [tensor](../../glossary.md#tensor) with probability `0.5` from the scripted draws `0.0`, `0.25`, `0.5`,
and `Math.nextDown(1.0)`. The draw equal to the probability demonstrates strict comparison, while
the scripted source makes the unbounded call path and exact call count observable.

```java
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorRandoms;
import java.util.Arrays;
import java.util.Optional;
import java.util.random.RandomGenerator;

public final class BernoulliRandomExample {
    private static final class ScriptedGenerator implements RandomGenerator {
        private final double[] values = {0.0, 0.25, 0.5, Math.nextDown(1.0)};
        private int calls;

        @Override
        public long nextLong() {
            throw new AssertionError("nextLong is not used");
        }

        @Override
        public double nextDouble() {
            return values[calls++];
        }
    }

    public static void main(String[] args) {
        ScriptedGenerator source = new ScriptedGenerator();
        // Even endpoint probabilities consume one draw per logical position.
        Tensor tensor = TensorRandoms.randomBernoulli(
                Shape.of(2, 2), 0.5, source, Optional.of("mask"));

        byte[] data = (byte[]) tensor.hostStorage()
                .orElseThrow().segment().heapBase().orElseThrow();

        System.out.println(Arrays.toString(data));
        System.out.println(tensor.descriptor().dataType());
        System.out.println(tensor.descriptor().requiresGrad());
        System.out.println(source.calls);
    }
}
```

<a id="meaningful-lines-6"></a>
#### Meaningful lines

- The source overrides unbounded `nextDouble()` and rejects `nextLong()`. `TensorRandoms` uses the
  exact supplied object and makes one unbounded call for each output position.
- The first two draws are strictly below `0.5` and become canonical true bytes. The equal and
  greater draws become canonical false bytes.<span data-glossary-navigation="true"> ([canonical glossary](../../glossary.md#term-canonical-form))</span>
- The public method fixes [BOOL](../../glossary.md#data-type--datatype) type and false [gradient](../../glossary.md#term-gradient) intent, so the call has no [`DataType`](../../glossary.md#data-type--datatype) or
  `requiresGrad` [argument](../../glossary.md#term-argument).
- Heap-base inspection makes implemented raw storage visible for the example; it is not a typed
  Tensor access or export API.<span data-glossary-navigation="true"> ([Heap glossary](../../glossary.md#term-heap); [Tensor glossary](../../glossary.md#tensor); [API glossary](../../glossary.md#term-application-programming-interface-api))</span>

<a id="result-and-interpretation-9"></a>
#### Result and interpretation

The program prints:

```text
[1, 1, 0, 0]
BOOL
false
4
```

The result proves strict comparison, [canonical](../../glossary.md#term-canonical-form) [BOOL](../../glossary.md#data-type--datatype) storage, fixed [descriptor](../../glossary.md#term-descriptor) facts, and one
unbounded call per [row-major](../../glossary.md#term-row-major) [element](../../glossary.md#term-element) for this scripted input. It does not establish statistical
independence for an arbitrary custom generator, provide numeric [Bernoulli](../../glossary.md#term-bernoulli-distribution) output, promise
cross-generator [reproducibility](../../glossary.md#artifact-manifest-and-source-provenance), or create a random [graph](../../glossary.md#graph) [Operation](../../glossary.md#operation) or [runtime](../../glossary.md#term-runtime)/[backend](../../glossary.md#backend) generator.

<a id="failures-and-useful-variations-8"></a>
#### Failures and useful variations

- A dynamic [shape](../../glossary.md#shape), a logical count above `Integer.MAX_VALUE`, or a non-finite or out-of-range
  probability fails before sampling or ID allocation. Probability is validated for empty output.<span data-glossary-navigation="true"> ([finite glossary](../../glossary.md#term-finite); [sampling glossary](../../glossary.md#term-sampling-machine-learning-and-hyperparameters))</span>
- Probability `0.0` or `-0.0` still consumes one call per element and stores all false for a
  conforming source; probability <span data-glossary-navigation="true"> ([element glossary](../../glossary.md#term-element))</span>`1.0` consumes the same calls and stores all true.
- If `nextDouble()` throws, earlier calls remain consumed, but no destination or ID exists.
- A blank label fails after all calls, destination allocation, and ID allocation; the ID is
  consumed and no state is rolled back.


---

[Previous topic](constants.md) · [Next topic](graph-randomness.md)
