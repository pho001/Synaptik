# Ranges and eager random initialization

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

<a id="complete-integer-range-example"></a>
### Complete integer-range example

<a id="goal-and-inputs-5"></a>
#### Goal and inputs

Create the INT32 range `[1, 4, 7]` from inclusive start `1`, exclusive end `8`, and step `3`.

```java
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.Arrays;
import java.util.Optional;

public final class IntegerRangeExample {
    public static void main(String[] args) {
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

- The `int` range overload fixes the result data type as `INT32`. Ceiling division of the distance
  by step gives three values, and the exclusive bound `8` is not emitted.
- The result shape is rank one with one position for each of those three values. Heap-base
  inspection only makes existing raw storage observable; it is not a typed Tensor access or
  export API.

<a id="result-and-interpretation-5"></a>
#### Result and interpretation

The program prints:

```text
INT32
Shape[3]
[1, 4, 7]
```

The result demonstrates range typing, exclusive-end semantics, canonical dense shape, and copied
storage. It does not provide an empty or floating range, implicit conversion, a prefix fixture
API, typed access/export, provenance, or execution. Normal random creation is the separate
explicit-source `TensorRandoms` method described next.

<a id="failures-and-useful-variations-4"></a>
#### Failures and useful variations

- `range(0, 3, -1, label)` fails because the negative step cannot advance toward the larger end.
- Equal start and end values fail because current range construction is deliberately non-empty.
- Counts above `Integer.MAX_VALUE` fail before carrier, destination, or ID allocation.

`TensorRandoms` owns exactly one public normal method:
`randomNormal(shape, dataType, mean, standardDeviation, randomGenerator, label, requiresGrad)`.
It accepts only fully static Java-array-sized shapes and every floating type, including FP8.
Both parameters must be finite, and the standard deviation must be numerically non-negative;
either signed zero is valid. The result has a newly synthesized canonical dense-contiguous
descriptor, independent writable heap storage, the explicit label and gradient intent, and a new
factory ID.

For every logical row-major element, the method calls the exact supplied
`RandomGenerator.nextGaussian()` once, multiplies the returned binary64 value by the standard
deviation, and then adds the binary64 mean. It does not use fused multiply-add. `FLOAT64` stores
the result directly, `FLOAT32` narrows it once to binary32, and `BFLOAT16` first narrows to
binary32 and then uses `BFloat16Bits.fromFloat`; FLOAT16 and both FP8 types round directly from
binary64. A scalar consumes one call; an empty shape consumes none. Non-finite converted values
are not rejected after sampling.

The generator is transient caller-owned state. `TensorRandoms` does not select an algorithm or
default source, store a seed, retain or replace the generator, synchronize access, or reset,
split, or close it. Equivalent output therefore requires equivalent generator implementation and
initial state, identical arguments, and no interfering source use. There is no sequence promise
across algorithms, providers, Java versions, seed expansion, concurrent access, or different
initial states.

<a id="complete-normal-random-example"></a>
### Complete normal-random example

<a id="goal-and-inputs-6"></a>
#### Goal and inputs

Create a `2 × 2` FLOAT32 tensor from the scripted Gaussian sequence `[-1, 0, 1, 0.5]`, mean `1`,
and standard deviation `2`. A scripted source makes the transformation and exact call count
observable without introducing a production seed or generator-selection API.

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
- Each output applies `1 + gaussian × 2` in binary64 before narrowing to the requested FLOAT32
  carrier. The four transformed values are `-1`, `1`, `3`, and `2` in row-major order.
- Heap-base inspection makes the already implemented raw storage observable for the example; it
  is not a typed Tensor access or export API.

<a id="result-and-interpretation-6"></a>
#### Result and interpretation

The program prints:

```text
[-1.0, 1.0, 3.0, 2.0]
4
DENSE_CONTIGUOUS
```

The result proves transformation order, one source call per element, FLOAT32 conversion, and
canonical dense layout for this scripted input. It does not promise statistical properties of a
custom generator, cross-generator reproducibility, a Synaptik seed API, continuous-uniform,
integral, or Bernoulli behavior, typed export, or runtime/backend random execution.

<a id="failures-and-useful-variations-5"></a>
#### Failures and useful variations

- A dynamic shape, non-floating data type, non-finite mean, negative or non-finite deviation, or
  logical count above `Integer.MAX_VALUE` fails before sampling or ID allocation.
- If `nextGaussian()` throws, preceding calls remain consumed, but no destination or ID exists.
- A blank present label is detected by delegated Tensor validation after all samples, destination
  allocation, and ID allocation; the ID is consumed and no state is rolled back.
- Two equivalent caller-created generators can produce equivalent tensors when their initial
  states and all method arguments match and neither source has interfering use.

`TensorRandoms` also owns exactly one public continuous-uniform method:
`randomUniform(shape, dataType, lowerBoundInclusive, upperBoundExclusive, randomGenerator, label,
requiresGrad)`. It accepts only fully static Java-array-sized shapes and every floating type,
including FP8. Both binary64 bounds must be finite, and the lower bound must be strictly less than the
upper bound. The result has a newly synthesized canonical dense-contiguous descriptor, independent
writable heap storage, the explicit label and gradient intent, and a new factory ID.

For every logical row-major element, the method calls the exact supplied
`RandomGenerator.nextDouble(lowerBoundInclusive, upperBoundExclusive)` once. A conforming source
returns a binary64 value in the half-open interval `[lowerBoundInclusive, upperBoundExclusive)`.
`FLOAT64` stores that value directly, `FLOAT32` narrows it once to binary32, `BFLOAT16` first
narrows to binary32 and then uses `BFloat16Bits.fromFloat`, and FLOAT16/both FP8 types round directly from the
binary64 sample without an intermediate binary32 rounding. The half-open promise applies before
narrowing: a stored low-precision or FLOAT32 value may equal the corresponding narrowed upper
bound or may round to a lower representable value. `TensorRandoms` does not clamp, resample, or
post-validate a custom non-conforming source result. A scalar consumes one bounded call; an empty shape consumes
none.

The source has the same ownership and reproducibility boundary as normal creation. The caller
creates, configures, seeds, owns, and advances it, and must provide exclusive or otherwise safe
access. `TensorRandoms` does not select an algorithm or default source, store a seed, retain or replace
the generator, synchronize access, or reset, split, or close it. Equivalent output requires an
equivalent generator implementation and initial state, identical arguments, and no interfering
source use. No sequence is promised across algorithms, providers, Java versions, seed expansion,
concurrent use, or different initial states.

<a id="complete-continuous-uniform-example"></a>
### Complete continuous-uniform example

<a id="goal-and-inputs-7"></a>
#### Goal and inputs

Create a `2 × 2` FLOAT32 tensor from four known binary64 values in `[-1, 1)`. The final source
value is `Math.nextDown(1.0)`, which is strictly below the binary64 upper bound but rounds to
binary32 `1.0f`. A scripted source makes the exact bounded-call arguments, call count, and
narrowing caveat observable without adding a production seed or source-selection API.

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
  and `1.0` arguments on every call.
- The first three binary64 values remain `-1.0f`, `0.0f`, and `0.5f`. The final conforming value is
  below binary64 `1.0`, but one binary32 narrowing rounds it to stored `1.0f`.
- Heap-base inspection makes the already implemented raw storage observable for the example; it
  is not a typed Tensor access or export API.

<a id="result-and-interpretation-7"></a>
#### Result and interpretation

The program prints:

```text
[-1.0, 0.0, 0.5, 1.0]
4
DENSE_CONTIGUOUS
```

The result proves the exact bounded-call path, one call per row-major element, FLOAT32 narrowing,
the narrowed-upper-bound caveat, and canonical dense layout for this scripted input. It does not
promise statistical quality of a custom generator, a closed binary64 interval, cross-generator
reproducibility, a Synaptik seed API, integral or Bernoulli sampling, typed export, or
runtime/backend random execution.

<a id="failures-and-useful-variations-6"></a>
#### Failures and useful variations

- A dynamic shape, non-floating data type, non-finite bound, unordered bounds, or logical count
  above `Integer.MAX_VALUE` fails before sampling or ID allocation.
- If bounded `nextDouble(lower, upper)` throws, preceding calls remain consumed, but no destination
  or ID exists.
- A blank present label is detected by delegated Tensor validation after all samples, destination
  allocation, and ID allocation; the ID is consumed and no state is rolled back.
- Two equivalent caller-created generators can produce equivalent tensors when their initial
  states and all method arguments match and neither source has interfering use.

`TensorRandoms` owns exactly two public bounded-integral overloads named `randomInt`. Primitive
`int` bounds produce `INT32`, while primitive `long` bounds produce `INT64`. Both take a fully
static Java-array-sized shape, inclusive origin, exclusive bound, caller-owned `RandomGenerator`,
and optional label. The origin must be strictly less than the bound, including for an empty
result. The synthesized descriptor is canonical dense-contiguous and always has
`requiresGrad == false`; there is no data-type or gradient parameter because the bound carrier
already selects a non-differentiable result type.

Each logical row-major element is the direct result of exactly one matching bounded call:
`nextInt(origin, bound)` for `INT32` or `nextLong(origin, bound)` for `INT64`. `TensorRandoms` uses no
unbounded draw, modulo reduction, floating arithmetic, narrowing, widening, or alternate carrier.
It fills one exact `int[]` or `long[]` and delegates once to matching flat import. A conforming
source supplies values in the half-open interval `[origin, bound)` without project-owned modulo
bias; custom non-conforming values are copied without post-validation. A scalar consumes one
bounded call and a valid empty result consumes none.

The source has the same ownership and bounded reproducibility contract as the floating random
methods. The caller configures, seeds, owns, advances, and provides safe access to it. `TensorRandoms`
does not substitute, retain, synchronize, reset, split, or close the source. Equivalent output
requires equivalent generator implementation and initial state, identical arguments, and no
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

Create a `2 × 2` INT32 tensor from the scripted values `[-3, -1, 0, 4]` over `[-3, 5)`. The
scripted source verifies that every call receives the exact bounds and that neither an unbounded
draw nor a production seed/source-selection API is involved.

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

- Primitive `int` bounds select `INT32`; the call has no `DataType` or gradient argument.
- The source rejects unbounded `nextLong()` and validates `-3` and `5` on every bounded
  `nextInt(origin, bound)` invocation. `TensorRandoms` stores each returned value directly.
- Heap-base inspection makes the implemented raw storage observable for the example; it is not a
  typed Tensor access or export API.

<a id="result-and-interpretation-8"></a>
#### Result and interpretation

The program prints:

```text
[-3, -1, 0, 4]
INT32
false
4
```

The result proves inferred INT32 type, false gradient intent, exact bounded-call order and count,
direct carrier storage, and half-open boundary values for this scripted input. The primitive
`long` overload follows the same contract with `nextLong(origin, bound)` and an `INT64` result. The
example does not promise statistical quality for a custom generator, full-domain sampling, typed
export, a random graph Operation, or runtime/backend generation.

<a id="failures-and-useful-variations-7"></a>
#### Failures and useful variations

- A dynamic shape or logical count above `Integer.MAX_VALUE` fails before sampling or ID
  allocation. Checked element-count overflow remains an `ArithmeticException`.
- Equal or reversed bounds fail even for an empty result, before source-carrier allocation.
- If the bounded source method throws, earlier calls remain consumed, but no destination or ID
  exists.
- A blank label fails after all bounded calls, destination allocation, and ID allocation; the ID
  is consumed and no state is rolled back.

`TensorRandoms` owns exactly one public Bernoulli method:
`randomBernoulli(shape, probability, randomGenerator, label)`. It creates a canonical dense BOOL
tensor with gradients disabled. The shape must be fully static, its checked logical count must fit
a Java array, and the binary64 probability must be finite and in the closed interval `[0, 1]`.
Positive and negative zero are accepted as zero. No data-type or gradient argument is present
because Bernoulli output is logical non-differentiable leaf data; numeric truthiness and output
conversion are outside this method.

For each logical row-major element, `TensorRandoms` calls the exact supplied source's unbounded
`nextDouble()` method once and stores canonical byte `1` exactly when `draw < probability`;
otherwise it stores byte `0`. The strict comparison means that a draw equal to the probability is
false. A custom non-conforming draw is compared directly without post-validation. Calls are not
skipped at probability zero or one: conforming draws still produce all false or all true values,
respectively, while source advancement stays independent of endpoint optimization. A scalar
consumes one call and an empty result consumes none.

The source follows the same caller-ownership and bounded-reproducibility contract as the other
random methods. `TensorRandoms` does not substitute, retain, seed, synchronize, reset, split, or close
it. Equivalent output requires equivalent generator implementation and initial state, identical
arguments, and no interfering use. One complete canonical `byte[]` is sampled before one BOOL
flat import; there is no partial Tensor, direct storage construction, or direct ID allocation in
the sampling helper.

<a id="complete-bernoulli-random-example"></a>
### Complete Bernoulli-random example

<a id="goal-and-inputs-9"></a>
#### Goal and inputs

Create a `2 × 2` BOOL tensor with probability `0.5` from the scripted draws `0.0`, `0.25`, `0.5`,
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
  greater draws become canonical false bytes.
- The public method fixes BOOL type and false gradient intent, so the call has no `DataType` or
  `requiresGrad` argument.
- Heap-base inspection makes implemented raw storage visible for the example; it is not a typed
  Tensor access or export API.

<a id="result-and-interpretation-9"></a>
#### Result and interpretation

The program prints:

```text
[1, 1, 0, 0]
BOOL
false
4
```

The result proves strict comparison, canonical BOOL storage, fixed descriptor facts, and one
unbounded call per row-major element for this scripted input. It does not establish statistical
independence for an arbitrary custom generator, provide numeric Bernoulli output, promise
cross-generator reproducibility, or create a random graph Operation or runtime/backend generator.

<a id="failures-and-useful-variations-8"></a>
#### Failures and useful variations

- A dynamic shape, a logical count above `Integer.MAX_VALUE`, or a non-finite or out-of-range
  probability fails before sampling or ID allocation. Probability is validated for empty output.
- Probability `0.0` or `-0.0` still consumes one call per element and stores all false for a
  conforming source; probability `1.0` consumes the same calls and stores all true.
- If `nextDouble()` throws, earlier calls remain consumed, but no destination or ID exists.
- A blank label fails after all calls, destination allocation, and ID allocation; the ID is
  consumed and no state is rolled back.


---

[Previous topic](constants.md) · [Next topic](graph-randomness.md)
