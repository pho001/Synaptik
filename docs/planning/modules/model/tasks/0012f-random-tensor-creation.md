# Task 0012F: Random [Tensor](../../../../glossary.md#tensor) Creation

**Editorial reading key (later context, not a status update).**

The application passes its RandomGenerator to the factory, which calls that generator now and imports the sampled carrier into a new dense leaf. The generator is mutable source state supplied by the caller, not a retained service or graph input. A normal distribution is specified by its center (mean) and spread (standard deviation); storage narrowing is an additional representation step. Reproducibility therefore depends on generator implementation, state and call sequence, not only a seed. This is eager initialization, not a random operation executed later by a backend.

Focused current reading: [Current explicit-source normal example](../../../../reference/tensor/random-initialization.md#complete-normal-random-example). The original record below retains its own scope, decisions and evidence.


## Status

Complete

## Goal

Add one explicit public factory method for creating a dense floating [tensor](../../../../glossary.md#tensor) from a [normal
distribution](../../../../glossary.md#norms-elementary-functions-and-probability-distributions). The caller supplies the exact `RandomGenerator`, [distribution](../../../../glossary.md#term-distribution) [parameters](../../../../glossary.md#term-parameter), floating
[`DataType`](../../../../glossary.md#data-type--datatype), [shape](../../../../glossary.md#shape), optional label, and [gradient](../../../../glossary.md#term-gradient) intent.

This completes the selected TensorFactory population baseline without hidden global randomness,
generator retention, a [service locator](../../../../glossary.md#term-service-locator), default [metadata](../../../../glossary.md#term-metadata), or [runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) behavior.

## [Scope](../../../../glossary.md#term-scope)

- Add one `TensorFactory.randomNormal(...)` method using a caller-supplied
  `java.util.random.RandomGenerator`.
- Support exactly [`FLOAT64`](../../../../glossary.md#term-float64), [`FLOAT32`](../../../../glossary.md#term-float32), and [`BFLOAT16`](../../../../glossary.md#term-bfloat16) output.
- Require finite mean, finite non-negative standard deviation, and a fully static shape whose
  logical count fits a Java array.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [mean glossary](../../../../glossary.md#term-mean); [standard deviation glossary](../../../../glossary.md#term-standard-deviation); [shape glossary](../../../../glossary.md#shape))</span>
- Synthesize [canonical](../../../../glossary.md#term-canonical-form) dense-[contiguous](../../../../glossary.md#term-contiguous) [layout](../../../../glossary.md#layout) and preserve explicit label/[gradient](../../../../glossary.md#term-gradient) intent.
- Consume exactly one `RandomGenerator.nextGaussian()` call per logical [element](../../../../glossary.md#term-element) in [row-major](../../../../glossary.md#term-row-major) order.
- Calculate `mean + gaussian * standardDeviation` with ordinary binary64 multiplication followed
  by addition, without fused multiply-add substitution.<span data-glossary-navigation="true"> ([binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [fused multiply-add glossary](../../../../glossary.md#term-fused-multiply-add-fma))</span>
- Store binary64 directly for FLOAT64, narrow to binary32 for FLOAT32, and narrow to binary32 then
  use <span data-glossary-navigation="true"> ([binary32 glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>`BFloat16Bits.fromFloat(...)` for BFLOAT16.
- Create one exact typed source [carrier](../../../../glossary.md#term-storage-carrier) and delegate once to matching flat import.
- Define source [ownership](../../../../glossary.md#term-ownership-meanings), [reproducibility](../../../../glossary.md#artifact-manifest-and-source-provenance), threading, failure, allocation, and ID side effects.
- Modify `TensorFactory`, add one package-private `TensorRandoms`, update only the factory API-shape
  test, and add one focused random test.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- During implementation, update Tensor API, glossary, task evidence, master plan, and roadmap
  through the required separate clean-context documentation pass.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- factory-owned/static/global/default/thread-local randomness, `RandomGenerator.getDefault()`,
  registry, service, dependency injection, retained source, or lifecycle ownership<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [dependency injection glossary](../../../../glossary.md#term-dependency-injection); [lifecycle glossary](../../../../glossary.md#lifecycle); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- [seed](../../../../glossary.md#term-seed)-only, algorithm-name, `RandomGeneratorFactory`, split/reset/copy/close, source-creation, or
  synchronization APIs<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- cross-algorithm, cross-provider, cross-Java-version, or concurrent-use [reproducibility](../../../../glossary.md#artifact-manifest-and-source-provenance) promises
- uniform, Bernoulli, integer, boolean, categorical, truncated/multivariate normal, permutation,
  random-range, or arbitrary-distribution creation<span data-glossary-navigation="true"> ([Bernoulli glossary](../../../../glossary.md#term-bernoulli-distribution); [boolean glossary](../../../../glossary.md#data-type--datatype); [permutation glossary](../../../../glossary.md#permutation); [distribution glossary](../../../../glossary.md#term-distribution))</span>
- [`INT32`](../../../../glossary.md#data-type--datatype), [`INT64`](../../../../glossary.md#data-type--datatype), or [`BOOL`](../../../../glossary.md#data-type--datatype) random output and their [rounding](../../../../glossary.md#term-rounding)/truthiness policies
- default [mean](../../../../glossary.md#term-mean), deviation, [dtype](../../../../glossary.md#data-type--datatype), label, [gradient](../../../../glossary.md#term-gradient) flag, source, [shape](../../../../glossary.md#shape), or convenience overloads
- dynamic binding, descriptor/layout input, non-dense destinations, view/scatter population, or
  Tensor input<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [descriptor glossary](../../../../glossary.md#term-descriptor); [layout glossary](../../../../glossary.md#layout); [view glossary](../../../../glossary.md#term-view-meanings); [Tensor glossary](../../../../glossary.md#tensor))</span>
- typed access/export, backing-array exposure, mutation/version tracking, or zero-copy transfer
- provenance, random operations, graph/compiler/planning/prepare/runtime/backend/training behavior,
  execution, or ONNX mapping<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [graph glossary](../../../../glossary.md#graph); [compiler glossary](../../../../glossary.md#term-compiler); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [ONNX glossary](../../../../glossary.md#term-onnx))</span>
- new storage, native/[off-heap](../../../../glossary.md#term-off-heap-memory) allocation, [`Arena`](../../../../glossary.md#term-arena), deterministic close, [pooling](../../../../glossary.md#term-pooling), or [residency](../../../../glossary.md#residency)
- dependencies, preview/incubator [features](../../../../glossary.md#term-software-feature), [Gradle](../../../../glossary.md#term-gradle)/architecture changes, or another [module](../../../../glossary.md#term-module)
- a detailed task-0013 or later specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>, especially normal random population and the
  requirement to decide source/reproducibility without live model services<span data-glossary-navigation="true"> ([reproducibility glossary](../../../../glossary.md#artifact-manifest-and-source-provenance))</span>
- [Model master plan](../master-plan.md)
- [Task 0001](0001-data-type-model.md)
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0003](0003-layout-descriptor-model.md)
- [Task 0007](0007-tensor-descriptor-model.md)
- [Task 0011](0011-public-tensor-skeleton.md)
- [Task 0012](0012-tensor-factory.md)
- [Task 0012A](0012a-host-storage-allocation.md)
- [Task 0012B](0012b-flat-typed-tensor-import.md)
- [Task 0012D](0012d-constant-tensor-creation.md)
- [Task 0012E](0012e-range-and-prefix-population.md)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span> and [glossary](../../../../glossary.md)

## Legacy evidence and rejected coupling

Read-only inspection of legacy `TensorDataFactory.randn(...)` and [Tensor](../../../../glossary.md#tensor) conveniences confirms
[eager](../../../../glossary.md#term-eager-evaluation) normally distributed floating tensors parameterized by [shape](../../../../glossary.md#shape), [mean](../../../../glossary.md#term-mean), [standard deviation](../../../../glossary.md#term-standard-deviation),
[data type](../../../../glossary.md#data-type--datatype), and label. Legacy sampled `mean + Random.nextGaussian() * stdDev`, rejected
non-floating output and invalid deviation, and created an unseeded generator internally.

The new design rejects hidden per-call nondeterminism, default source selection, nullable labels,
positive-[dimension](../../../../glossary.md#dimension)-only shapes, implicit double-array conversion, [mutable](../../../../glossary.md#term-mutable) [metadata](../../../../glossary.md#term-metadata), and direct
Tensor construction. It retains the useful [distribution](../../../../glossary.md#term-distribution) through an explicit caller-owned source
and current [descriptor](../../../../glossary.md#term-descriptor)/import paths.

## Architecture constraints

- Production remains in `io.github.pho001.synaptik.model.tensor`. Public [API](../../../../glossary.md#term-application-programming-interface-api) belongs to
  `TensorFactory`; [sampling](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters) mechanics belong to one package-private helper beside it.
- Package direction remains toward existing datatype/shape/layout/storage foundations plus stable
  JDK <span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk))</span>`java.util.random`; no cross-[module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency) is introduced.
- `RandomGenerator` is transient caller-owned method input. No factory, helper, Tensor, descriptor,
  storage, registry, or service retains it.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [descriptor glossary](../../../../glossary.md#term-descriptor); [registry glossary](../../../../glossary.md#term-registry))</span>
- The factory performs no source lookup, default selection, algorithm choice, registration,
  seeding, splitting, reset, close, or synchronization.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- Random creation is eager leaf-data construction, not an Operation, graph node, compiler pass,
  runtime generator, backend kernel, or training policy.<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [leaf glossary](../../../../glossary.md#term-leaf); [Operation glossary](../../../../glossary.md#operation); [graph glossary](../../../../glossary.md#graph); [node glossary](../../../../glossary.md#node); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [kernel glossary](../../../../glossary.md#kernel))</span>
- Only floating types are accepted. Integral/[BOOL](../../../../glossary.md#data-type--datatype) output would require separate policy.
- Every result has explicit shape/type/gradient/label, canonical dense layout, fresh storage, and a
  fresh factory ID.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Sampling completes in one exact typed source carrier before flat import; no partial Tensor is
  observable.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Success delegates exactly once to matching flat import and does not directly construct
  Tensor/storage or allocate an ID.
- Stop if implementation needs a public random abstraction, stored/default source, another helper,
  storage-contract change, architecture change, dependency, or another module.

## Package impact

Existing packages used:

- `model.tensor` — public construction, [descriptor](../../../../glossary.md#term-descriptor) synthesis, and [eager](../../../../glossary.md#term-eager-evaluation) [leaf](../../../../glossary.md#term-leaf) population.
- `model.datatype` — floating types and [BFLOAT16](../../../../glossary.md#term-bfloat16) conversion.
- `model.shape` — fully static [shapes](../../../../glossary.md#shape).
- `model.layout` — [canonical](../../../../glossary.md#term-canonical-form) dense-[contiguous](../../../../glossary.md#term-contiguous) geometry.
- `java.util.random` — caller-owned Java 26 `RandomGenerator` contract.

Packages added or changed:

- No package is added. Only the existing `model.tensor` package changes.

Type placement:

- `io.github.pho001.synaptik.model.tensor.TensorFactory` — one public normal-random method.
- `io.github.pho001.synaptik.model.tensor.TensorRandoms` — package-private descriptor validation,
  sampling, conversion, and flat-import dispatch.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- `io.github.pho001.synaptik.model.tensor.TensorFactoryRandomTest` — same-package focused source,
  conversion, reproducibility, validation, and side-effect tests.<span data-glossary-navigation="true"> ([reproducibility glossary](../../../../glossary.md#artifact-manifest-and-source-provenance))</span>

## Required contract

### Public method

Add exactly:

```java
public static Tensor randomNormal(
        Shape shape,
        DataType dataType,
        double mean,
        double standardDeviation,
        RandomGenerator randomGenerator,
        Optional<String> label,
        boolean requiresGrad)
```

Do not add `randn`, default/unlabeled, [seed](../../../../glossary.md#term-seed), factory, algorithm-name, `Random`-specific,
[descriptor](../../../../glossary.md#term-descriptor)/[layout](../../../../glossary.md#layout), or other-[distribution](../../../../glossary.md#term-distribution) overloads.

### Public null [validation](../../../../glossary.md#term-validation)

Validate before helper delegation in exact order:

1. null [`shape`](../../../../glossary.md#shape): `NullPointerException`, message `shape`;
2. null `dataType`: `NullPointerException`, message `dataType`;
3. null `randomGenerator`: `NullPointerException`, message `randomGenerator`;
4. null `label`: `NullPointerException`, message `label`.

These failures consume no random calls, allocation, or [Tensor](../../../../glossary.md#tensor) ID.

### [Shape](../../../../glossary.md#shape), type, [distribution](../../../../glossary.md#term-distribution), and [descriptor](../../../../glossary.md#term-descriptor) [validation](../../../../glossary.md#term-validation)

The helper validates before [carrier](../../../../glossary.md#term-storage-carrier) allocation or [sampling](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters):

1. Dynamic shape: `IllegalArgumentException` with
   `random tensor creation requires a fully static shape: <shape>`.
2. Read `shape.knownElementCount()`; checked multiplication [overflow](../../../../glossary.md#term-overflow) remains
   `ArithmeticException`.
3. Count above `Integer.MAX_VALUE`: `IllegalArgumentException` with
   `random tensor element count exceeds Java array limit: required=<required>, maximum=2147483647`.
4. Non-floating type: `IllegalArgumentException` with
   `random normal creation requires floating data type: <dataType>`.
5. Non-[finite](../../../../glossary.md#term-finite) [mean](../../../../glossary.md#term-mean): `IllegalArgumentException` with
   `random normal mean must be finite: <mean>`.
6. Non-finite or numerically negative deviation: `IllegalArgumentException` with
   `random normal standard deviation must be finite and non-negative: <standardDeviation>`.
   Positive and negative zero are accepted.<span data-glossary-navigation="true"> ([negative zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
7. Create `LayoutDescriptor.contiguous(shape)`.
8. Create `TensorDescriptor(dataType, shape, Optional.of(layout), requiresGrad)`.

[Scalar](../../../../glossary.md#term-scalar) shape consumes one [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters). A fully static zero-[element](../../../../glossary.md#term-element) shape is valid, creates an empty
[tensor](../../../../glossary.md#tensor), and consumes zero samples.

### [Sampling](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters) and [carrier](../../../../glossary.md#term-storage-carrier) conversion

After [validation](../../../../glossary.md#term-validation):

1. Allocate exactly one matching source array of checked logical count.
2. For every index, call the exact supplied `randomGenerator.nextGaussian()` once.
3. Evaluate `double sample = mean + gaussian * standardDeviation` with ordinary multiplication
   then addition. Do not use `Math.fma`, streams, parallelism, batching, or
   `nextGaussian(mean, standardDeviation)`.
4. Store exactly:

| [Data type](../../../../glossary.md#data-type--datatype) | Carrier | Stored value |
|---|---|---|
| [`FLOAT64`](../../../../glossary.md#term-float64) | `double[]` | [binary64](../../../../glossary.md#floating-point-precision-and-rounding) formula result |
| [`FLOAT32`](../../../../glossary.md#term-float32) | `float[]` | Java binary64-to-[binary32](../../../../glossary.md#floating-point-precision-and-rounding) [narrowing](../../../../glossary.md#term-narrowing) |
| [`BFLOAT16`](../../../../glossary.md#term-bfloat16) | `short[]` | binary32 narrowing, then `BFloat16Bits.fromFloat(...)` |

5. Delegate once to matching `TensorFactory.fromFlatArray(...)` with synthesized descriptor and
   caller label.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor))</span>
6. Return that exact [Tensor](../../../../glossary.md#tensor).

Generated [overflow](../../../../glossary.md#term-overflow), [underflow](../../../../glossary.md#term-underflow), [signed zero](../../../../glossary.md#term-signed-zero), [infinity](../../../../glossary.md#term-infinity), or [NaN](../../../../glossary.md#term-nan) is not post-validated. A custom
generator may override normal behavior; Synaptik still applies the specified formula/conversion.
BFLOAT16 retains existing [canonical](../../../../glossary.md#term-canonical-form)-NaN conversion. Generator and carrier are not retained.

### Source [ownership](../../../../glossary.md#term-ownership-meanings) and [reproducibility](../../../../glossary.md#artifact-manifest-and-source-provenance)

- The caller creates, configures, [seeds](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings), owns, and advances the generator.
- TensorFactory never selects an algorithm/default, stores a seed, resets/splits/closes, or retains
  the source.
- Non-empty success advances it by exactly one `nextGaussian()` method call per output element in
  row-major order; empty success advances it zero times.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [row-major glossary](../../../../glossary.md#term-row-major))</span>
- Equivalent results require equivalent generator implementation and initial state, identical
  arguments, and no interfering source use.<span data-glossary-navigation="true"> ([arguments glossary](../../../../glossary.md#term-argument))</span>
- No sequence promise exists across algorithms, providers, Java versions, seed expansion, source
  states, or concurrent access.
- The method does not synchronize caller state. Callers provide exclusive or otherwise safe use.
- A newly seeded [JDK](../../../../glossary.md#term-java-development-kit-jdk) generator may support repeatable tests, but Synaptik exposes no [seed](../../../../glossary.md#term-seed) contract.

### Failure side effects

- Null, shape, count, type, distribution, layout, and descriptor failures occur before source
  carrier/sampling and consume no ID.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [distribution glossary](../../../../glossary.md#term-distribution); [layout glossary](../../../../glossary.md#layout); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Source-[carrier](../../../../glossary.md#term-storage-carrier) [`OutOfMemoryError`](../../../../glossary.md#memory-management-atomic-counters-and-complexity) occurs before [sampling](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters) and consumes neither draws nor ID.
- If `nextGaussian()` throws, completed earlier calls stay consumed; the throwing call's effect is
  generator-defined. No destination or ID exists.
- After all [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters), flat import allocates destination and then ID.
- Destination `OutOfMemoryError` occurs after all draws but before ID.
- Blank label consumes all draws, both carriers, and one ID, then fails before flat copy.
- Exhaustion occurs after all draws and both carrier allocations, before copy; no [rollback](../../../../glossary.md#term-rollback) occurs.
- Unexpected copy failure consumes the allocated ID; no state is rolled back.

### Package-private helper

Add one package-private final `TensorRandoms` with exactly this package-private entry:

```java
static Tensor randomNormal(
        Shape shape,
        DataType dataType,
        double mean,
        double standardDeviation,
        RandomGenerator randomGenerator,
        Optional<String> label,
        boolean requiresGrad)
```

It must have no fields/public/protected members/state/[cache](../../../../glossary.md#term-cache)/[registry](../../../../glossary.md#term-registry)/[seed](../../../../glossary.md#term-seed); one private zero-arg
constructor; no retained [arguments](../../../../glossary.md#term-argument); and private methods only for [descriptor](../../../../glossary.md#term-descriptor) [validation](../../../../glossary.md#term-validation) and typed
[sampling](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters)/import. Use ordinary loops without streams, reflection, parallel work, per-[element](../../../../glossary.md#term-element) boxing,
or temporary `double[]` conversion for [FLOAT32](../../../../glossary.md#term-float32)/[BFLOAT16](../../../../glossary.md#term-bfloat16). Allocate one source [carrier](../../../../glossary.md#term-storage-carrier), delegate once
to flat import, and never directly construct [Tensor](../../../../glossary.md#tensor)/storage or allocate IDs.

## Valid and invalid scenarios

| Scenario | Result |
|---|---|
| [FLOAT64](../../../../glossary.md#term-float64) `[2,2]`, scripted source | Four transformed [binary64](../../../../glossary.md#floating-point-precision-and-rounding) values in order |
| [FLOAT32](../../../../glossary.md#term-float32) | Binary64 formula, then one [narrowing](../../../../glossary.md#term-narrowing) |
| [BFLOAT16](../../../../glossary.md#term-bfloat16) | Binary64 formula, [binary32](../../../../glossary.md#floating-point-precision-and-rounding) narrowing, BFLOAT16 conversion |
| [Scalar](../../../../glossary.md#term-scalar) | One source call and one value |
| `[2,0,3]` | Empty dense [tensor](../../../../glossary.md#tensor) and zero source calls |
| Same seeded implementation/state/[arguments](../../../../glossary.md#term-argument) | Equivalent result within stated [scope](../../../../glossary.md#term-scope) |
| Integral/[BOOL](../../../../glossary.md#data-type--datatype) type | Rejected before [sampling](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters) |
| Dynamic or over-limit [shape](../../../../glossary.md#shape) | Rejected before sampling |
| Non-[finite](../../../../glossary.md#term-finite) [mean](../../../../glossary.md#term-mean) | Rejected before sampling |
| Negative/non-finite deviation | Rejected before sampling |
| [Signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) deviation | Accepted; one call per [element](../../../../glossary.md#term-element) still occurs |
| Blank label | All draws and one ID consumed before failure |
| Generator throws | No destination/ID; prior calls remain consumed |
| ID exhaustion | All draws consumed; no copy or [rollback](../../../../glossary.md#term-rollback) |

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorFactory.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorRandoms.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryRandomTest.java`

Documentation/[planning](../../../../glossary.md#term-planning) during implementation:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum [scope](../../../../glossary.md#term-scope)

At most two production files, one existing and one new test, and the five documentation/[planning](../../../../glossary.md#term-planning)
files above: nine paths total.

Do not modify other Java/tests, [Gradle](../../../../glossary.md#term-gradle), `AGENTS.md`, `ARCHITECTURE.md`, focused architecture docs,
architecture tests, capabilities, another [module](../../../../glossary.md#term-module), or unrelated docs. Do not create task 0013.
Stop if another file/public type/source policy/default generator/storage or architecture decision
is required.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Update TensorFactory type Javadoc for explicit caller-owned normal generation.
- Fully document the public method: type/shape/count/layout, formula/conversions, finite rules,
  call order/count, ownership/non-retention/threading/reproducibility, label/gradient, result,
  failures, and source/allocation/ID side effects.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [finite glossary](../../../../glossary.md#term-finite); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [reproducibility glossary](../../../../glossary.md#artifact-manifest-and-source-provenance); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Document helper, entry, and private methods to the same [precision](../../../../glossary.md#term-precision).
- Explain two-stage [BFLOAT16](../../../../glossary.md#term-bfloat16) [narrowing](../../../../glossary.md#term-narrowing) and absence of default source/[seed](../../../../glossary.md#term-seed)/service.
- Do not claim other distributions, seeded Synaptik API, random Operations, access/export, or
  runtime/backend execution.<span data-glossary-navigation="true"> ([distributions glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend))</span>
- Review existing related Javadocs and record why unchanged contracts remain accurate or stop on
  an out-of-scope discrepancy.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>

## Acceptance criteria

- Exactly one new public `randomNormal(...)`; no other public [API](../../../../glossary.md#term-application-programming-interface-api).
- Existing fields, factory methods/helpers, allocator, behavior, and tests remain unchanged.
- Only FLOAT64/FLOAT32/BFLOAT16, static shape, finite mean, finite non-negative deviation, explicit
  source/label/gradient are accepted.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [shape glossary](../../../../glossary.md#shape); [finite glossary](../../../../glossary.md#term-finite); [mean glossary](../../../../glossary.md#term-mean); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Exactly one source call per non-empty logical [element](../../../../glossary.md#term-element) in order; empty output makes zero calls.
- Formula and three [carrier](../../../../glossary.md#term-storage-carrier) conversions are exact.
- Source is used transiently and never stored/substituted/synchronized/seeded/reset/split/closed.
- One source carrier and one matching flat-import delegation; no direct [Tensor](../../../../glossary.md#tensor)/storage/ID work.
- Exact validation, messages, source/allocation/ID effects, source exceptions, blank label, and
  permanent exhaustion are tested.
- Scripted conversion and seeded repeatability are proven without a production hook.
- API/helper/[bytecode](../../../../glossary.md#term-bytecode)/import/static-state/[scope](../../../../glossary.md#term-scope) checks pass.
- Javadocs, Tensor API, glossary, task/master/roadmap, and separate documentation review complete.
- 0012F becomes Complete only after all [validation](../../../../glossary.md#term-validation); 0013 stays Draft without a spec.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) before and after documentation review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryRandomTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Focused tests cover helper/[API](../../../../glossary.md#term-application-programming-interface-api) [shape](../../../../glossary.md#shape), exact source use, scripted three-type conversion, [scalar](../../../../glossary.md#term-scalar)/
empty shapes, all nulls in order, dynamic/count failures, non-floating types, invalid [parameters](../../../../glossary.md#term-parameter)
and [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) acceptance, [gradients](../../../../glossary.md#term-gradient), labels/[descriptors](../../../../glossary.md#term-descriptor), call count/order, equivalent seeded
sources, non-retention, source exception, safely observable side-effect order, blank label,
permanent exhaustion, and allocator restoration. Do not force unsafe [OOME](../../../../glossary.md#term-oome) tests.

Manually verify `javap -p -c -s`, imports, validation order, ordinary multiply/add, conversions,
one flat-import delegation, no direct [Tensor](../../../../glossary.md#tensor)/storage/ID construction, no source [lookup](../../../../glossary.md#term-lookup)/streams/
reflection/[synchronization](../../../../glossary.md#term-synchronization)/static [RNG](../../../../glossary.md#graph-rng-state--graphrngstate)/[seed](../../../../glossary.md#term-seed)/architecture-layer imports, exact nine-path [scope](../../../../glossary.md#term-scope),
documentation links/anchors/fences/whitespace/status, and no task-0013 spec.

## Dependencies

- 0012B supplies typed copied flat import.
- 0012 supplies [identity](../../../../glossary.md#term-identity-meanings) allocation/public construction.
- 0001/0002/0003/0007/0010/0011 supply type, [shape](../../../../glossary.md#shape), [layout](../../../../glossary.md#layout), [descriptor](../../../../glossary.md#term-descriptor), storage, and [Tensor](../../../../glossary.md#tensor).
- 0012D/0012E supply descriptor/[carrier](../../../../glossary.md#term-storage-carrier)/side-effect precedents.

## Follow-up tasks

- Subsequent user-approved planning inserts 0012G uniform, 0012H integral, and 0012I Bernoulli
  initialization before provenance. These are independent follow-ups, not incomplete 0012F work.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [Bernoulli glossary](../../../../glossary.md#term-bernoulli-distribution))</span>
- 0013 defines minimal [Tensor](../../../../glossary.md#tensor) [provenance](../../../../glossary.md#provenance) after the expanded initializer sequence.
- Later tasks own random Operation semantics, training initialization, typed access/export,
  mutation/versioning, native/runtime/backend generation, compiler capture, and execution.<span data-glossary-navigation="true"> ([Operation glossary](../../../../glossary.md#operation); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings))</span>

Do not create task 0013 or later here.

## Architecture impact

Expected impact: None.

`RandomGenerator` is transient method input, not a retained service or locator. Work remains [eager](../../../../glossary.md#term-eager-evaluation)
copied [model](../../../../glossary.md#term-model) [leaf](../../../../glossary.md#term-leaf) data and changes no [module](../../../../glossary.md#term-module) boundary, [lifecycle](../../../../glossary.md#lifecycle), [ownership](../../../../glossary.md#term-ownership-meanings), [graph](../../../../glossary.md#graph)/training policy,
or [runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) rule. Stop before architecture edits if implementation reveals otherwise.

## Implementation prompt

Use this prompt in a separate clean-context agentic task/thread:

```text
You are a clean-context implementation agent working in the Synaptik repository.

Read AGENTS.md, ARCHITECTURE.md, current focused architecture docs, documentation rules, planning
guide/roadmap, model capabilities/master plan, tasks 0001/0002/0003/0007/0010/0011/0012/0012A/
0012B/0012D/0012E/0012F, Tensor API, glossary, current factory/model source and all factory tests,
and root/model Gradle only to confirm Java 26.

Implement task 0012F exactly. Modify TensorFactory.java, add package-private TensorRandoms.java,
update TensorFactoryTest only for exact API shape, and add TensorFactoryRandomTest.java. Add exactly
one public randomNormal(...) method. Preserve all existing factory APIs/helpers/tests.

Require caller RandomGenerator, static shape, FLOAT64/FLOAT32/BFLOAT16, finite mean, and finite
non-negative deviation. Consume one nextGaussian() per element, apply ordinary binary64
mean + gaussian * deviation, convert exactly, and delegate once to flat import. Never retain,
replace, seed, split, synchronize, close, or look up the generator. Follow exact validation,
reproducibility, ownership, source/allocation/ID effects, and messages.

Do not add defaults/seeds/global randomness/other distributions/non-floating output/conversion/
descriptor input/random Operations/access/export/views/storage/native/provenance/compiler/training/
runtime/backend/dependencies/build changes/follow-up specs. Stop beyond nine paths or on
architecture uncertainty.

Run every specified test, Javadoc, bytecode/import/manual, documentation/example/link/whitespace/
scope/status check. Then hand the actual diff and all source/reproducibility/conversion/side-effect
evidence to a separate clean-context documentation agent in the same change. It must independently
inspect source/tests/generated Javadoc, finalize permitted Javadocs/Tensor API/glossary/planning,
record existing-Javadoc and architecture no-change conclusions, and rerun validation.

Update only this task, master plan, and roadmap for status/evidence. Do not mark Complete before
both passes. Leave 0013 Draft without a spec. Do not commit or push.
```

## Local decisions

- Caller-supplied `RandomGenerator` keeps algorithm/seeding/state explicit without [model](../../../../glossary.md#term-model) services.
- One method uses explicit floating [`DataType`](../../../../glossary.md#data-type--datatype); samples originate in binary64 and conversion is an
  explicit requested result format.<span data-glossary-navigation="true"> ([samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>
- Both distribution parameters must be finite. A non-finite mean does not describe the selected
  finite-parameter normal distribution and is rejected rather than silently filling infinities or
  NaNs; signed-zero deviation remains a valid degenerate distribution.<span data-glossary-navigation="true"> ([distribution glossary](../../../../glossary.md#term-distribution); [parameters glossary](../../../../glossary.md#term-parameter); [finite glossary](../../../../glossary.md#term-finite); [normal distribution glossary](../../../../glossary.md#norms-elementary-functions-and-probability-distributions); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- Reproducibility is bounded to equivalent source implementation/state and arguments; no seed-only
  or algorithm convenience implies a wider promise.<span data-glossary-navigation="true"> ([Reproducibility glossary](../../../../glossary.md#artifact-manifest-and-source-provenance); [arguments glossary](../../../../glossary.md#term-argument); [seed glossary](../../../../glossary.md#term-seed))</span>
- One draw per element is consumed even for signed-zero deviation, keeping source advancement
  independent of constant-distribution optimization.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [constant glossary](../../../../glossary.md#term-constant))</span>
- Ordinary multiply/add matches selected legacy transformation; no [mean](../../../../glossary.md#term-mean)/deviation overload or [FMA](../../../../glossary.md#term-fused-multiply-add-fma).
- [FLOAT32](../../../../glossary.md#term-float32) narrows once; [BFLOAT16](../../../../glossary.md#term-bfloat16) narrows to [binary32](../../../../glossary.md#floating-point-precision-and-rounding) then uses established conversion.
- Sampling completes before import, avoiding public partial tensors while making late failure
  consumption explicit.<span data-glossary-navigation="true"> ([Sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters); [tensors glossary](../../../../glossary.md#tensor))</span>

## Known limitations

- Only [eager](../../../../glossary.md#term-eager-evaluation) independent normal [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) for three floating types.
- Static Java-array-sized [shapes](../../../../glossary.md#shape) only; practical [heap](../../../../glossary.md#term-heap) limits may be lower.
- No generator [synchronization](../../../../glossary.md#term-synchronization) or [atomic](../../../../glossary.md#resource-transaction-and-rollback) source [snapshot](../../../../glossary.md#term-snapshot).
- No [reproducibility](../../../../glossary.md#artifact-manifest-and-source-provenance) promise across algorithms/providers/Java versions/states/interference.
- Generated non-[finite](../../../../glossary.md#term-finite) values are not post-validated.
- No seeded convenience, default source, other distribution, random Operation, access/export,
  training initializer, or runtime/backend random generation.<span data-glossary-navigation="true"> ([distribution glossary](../../../../glossary.md#term-distribution); [Operation glossary](../../../../glossary.md#operation); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) read architecture/planning/documentation rules, capability baseline, completed [model](../../../../glossary.md#term-model) and
factory tasks, current helpers/tests, Java 26 `RandomGenerator`, and read-only legacy randn source
and [Tensor](../../../../glossary.md#tensor) conveniences. The plan introduced no architecture, [dependency](../../../../glossary.md#term-dependency), package, build, or
retained-service decision.

Implementation validation:

- `./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryTest`
  passed with 7 tests, 0 failures, 0 errors, and 0 skipped.
- `./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryRandomTest`
  passed with 9 tests, 0 failures, 0 errors, and 0 skipped.
- `./gradlew :modules:model:test` passed with 214 tests, 0 failures, 0 errors, and 0 skipped.
- `./gradlew :modules:model:javadoc` passed with 2 actionable tasks.
- `./gradlew test` passed with 36 actionable tasks.
- The complete normal-random example in `docs/api/tensor-api.md` was copied without semantic
  changes to a temporary `NormalRandomExample.java`, then compiled with
  `javac -cp modules/model/build/classes/java/main -d /private/tmp/synaptik-normal-example NormalRandomExample.java`.
  Compilation passed with no diagnostics. Running<span data-glossary-navigation="true"> ([Compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>
  `java -cp modules/model/build/classes/java/main:/private/tmp/synaptik-normal-example NormalRandomExample`
  passed and printed exactly:

  ```text
  [-1.0, 1.0, 3.0, 2.0]
  4
  DENSE_CONTIGUOUS
  ```

  The temporary source was removed after validation and is not part of the task [scope](../../../../glossary.md#term-scope).
- A Ruby local Markdown [target](../../../../glossary.md#term-target)-and-heading-slug checker ran over `docs/api/tensor-api.md`,
  `docs/glossary.md`, this task, the model master plan, and the roadmap. It checked 159 local
  links, including 58 anchor-bearing links, and reported `failures=0`. Two preliminary checker
  invocations were not accepted as evidence: the first used unavailable Ruby `filter_map`, and
  the second reported one false negative because it collapsed adjacent heading spaces; the final
  checker preserved GitHub-style repeated hyphens and passed.
- Two preliminary JShell example runs compiled the example and printed the expected output but
  failed while persisting JShell history because the environment's preferences backend could not
  synchronize. The passing isolated `javac` and `java` commands above are the final example
  evidence.
- `git diff --check` passed.
- `javap -p -c -s` confirmed exactly one public factory signature, one package-private helper
  entry, no helper fields, a private zero-argument constructor, one matching primitive array and
  flat-import call per branch, one <span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [argument glossary](../../../../glossary.md#term-argument))</span>`nextGaussian` call in each loop, ordinary `dmul` then `dadd`,
  `d2f` for [FLOAT32](../../../../glossary.md#term-float32), and `d2f` followed by `BFloat16Bits.fromFloat` for [BFLOAT16](../../../../glossary.md#term-bfloat16).
- Source/import/static-state inspection confirmed exact public null-check order, helper validation
  order and messages, no retained/default/static random source, seed/factory/source lookup,
  synchronization, streams, reflection, parallelism, direct Tensor/storage/ID creation, or
  compiler/runtime/backend/training imports.<span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed); [lookup glossary](../../../../glossary.md#term-lookup); [synchronization glossary](../../../../glossary.md#term-synchronization); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime))</span>

Documentation validation used clean context
`/root/implement_model_0012f/review_model_0012f_docs`. The selected primary profiles were [API](../../../../glossary.md#term-application-programming-interface-api) and
[Javadoc](../../../../glossary.md#term-javadoc) for `TensorFactory`, `TensorRandoms`, and the Tensor API, plus Planning for this task,
master plan, and roadmap; General style and Example format were also applied. The pass inspected
the implementation, focused and factory tests, generated Javadoc, related [data type](../../../../glossary.md#data-type--datatype)/BFLOAT16/
[shape](../../../../glossary.md#shape)/[layout](../../../../glossary.md#layout)/[descriptor](../../../../glossary.md#term-descriptor)/Tensor/storage contracts, architecture explanations, model capability
baseline, and final nine-path diff. It finalized factory/helper Javadocs, added current normal-
random behavior and a scripted verified example to the Tensor API, and updated existing Tensor and
[Tensor factory](../../../../glossary.md#tensor-factory) glossary entries. No new glossary term was warranted because caller-owned
`RandomGenerator` and [normal distribution](../../../../glossary.md#norms-elementary-functions-and-probability-distributions) are ordinary Java/mathematical vocabulary rather than
new reusable Synaptik domain concepts. Local Markdown targets and changed anchors, example
fences, generated Javadoc content, terminology, trailing whitespace, synchronized status, exact
nine-path scope, and absence of a task-0013 specification were checked.

Existing [`DataType`](../../../../glossary.md#data-type--datatype), `BFloat16Bits`, `Shape`, `LayoutDescriptor`, `TensorDescriptor`, `Tensor`,
`HostTensorStorage`, and `MemorySegmentStorage` Javadocs remain accurate without edits: this task
uses their existing [floating classification](../../../../glossary.md#floating-classification), conversion, static-count, dense-layout, descriptor,
[borrowed](../../../../glossary.md#ownership-borrowing-and-lease)-storage, and [identity](../../../../glossary.md#term-identity-meanings) contracts without changing them. Existing flat-import/allocation/
[constant](../../../../glossary.md#term-constant)/population helper contracts also remain unchanged. Focused architecture documents and
the capability baseline require no edit because the work implements their existing model-owned
factory capability without changing [ownership](../../../../glossary.md#term-ownership-meanings), [dependency direction](../../../../glossary.md#java-platform-and-build-tools), [lifecycle](../../../../glossary.md#lifecycle), or capability
scope. Architecture tests and [backend](../../../../glossary.md#backend)/integration tests require no focused update because no
[module](../../../../glossary.md#term-module) boundary, backend behavior, or end-to-end execution behavior changed.

## Implementation notes

- Added the single public explicit-source method and package-private stateless [sampling](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters) helper.
- Sampling validates all metadata before allocation, constructs canonical dense geometry, fills
  one exact requested carrier in row-major order, and delegates once to existing flat import.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [canonical glossary](../../../../glossary.md#term-canonical-form); [carrier glossary](../../../../glossary.md#term-storage-carrier); [row-major glossary](../../../../glossary.md#term-row-major))</span>
- Tests cover public/helper API shape, exact conversion and call order, empty/scalar behavior,
  validation/messages, signed zero, bounded seeded equivalence, source failure, label side effects,
  and permanent identifier exhaustion with allocator restoration.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [shape glossary](../../../../glossary.md#shape); [scalar glossary](../../../../glossary.md#term-scalar); [validation glossary](../../../../glossary.md#term-validation); [signed zero glossary](../../../../glossary.md#term-signed-zero))</span>
- Documentation was finalized independently in the required clean context before completion.

## Completion summary

- Completed changes: one explicit caller-source normal-random factory method, stateless typed
  sampling/import helper, focused API/behavior tests, finalized Javadocs and Tensor API example,
  glossary status, and synchronized planning frontier.<span data-glossary-navigation="true"> ([sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Files changed or created: `TensorFactory.java`, `TensorRandoms.java`, `TensorFactoryTest.java`,
  `TensorFactoryRandomTest.java`, `docs/api/tensor-api.md`, `docs/glossary.md`, this task, model
  master plan, and roadmap.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- Tests and validation: both focused suites, all 214 model tests, model Javadoc, root tests,
  bytecode/source/manual contract checks, Markdown/example/generated-doc checks, scope/status
  checks, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` passed.
- Documentation-agent review: clean context
  `/root/implement_model_0012f/review_model_0012f_docs` completed the independent pass using API
  and Javadoc, Planning, General style, and Example format profiles.
- Documentation impact: Tensor API now documents implemented normal-random behavior, ownership,
  reproducibility, conversion, failures, and a complete scripted example; architecture/focused
  capability documents remain accurate without changes.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [reproducibility glossary](../../../../glossary.md#artifact-manifest-and-source-provenance))</span>
- Javadoc review: affected factory and helper contracts were finalized; related existing model
  contracts remain accurate because their behavior did not change.
- Glossary impact: updated existing implementation-status, Tensor, and Tensor factory language;
  no new reusable project term was introduced.<span data-glossary-navigation="true"> ([Tensor factory glossary](../../../../glossary.md#tensor-factory))</span>
- Unresolved issues: None.
- Follow-up required: None for task 0012F. Subsequent [planning](../../../../glossary.md#term-planning) places 0012G–0012I before task 0013.

Status: Complete
