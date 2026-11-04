# Task 0012H: Integral Random [Tensor](../../../../glossary.md#tensor) Creation

**Editorial reading key (later context, not a status update).**

The application supplies integer bounds and its mutable RandomGenerator. The factory invokes the corresponding bounded integer method for each element, then imports independent leaf storage; it does not convert floating samples or retain the generator. Exclusive upper bounds explain why the full signed carrier domain cannot be requested with a same-width bound one above its maximum. One method call per element is not a promise about how many random bits the JDK consumes internally.

Focused current reading: [Current bounded-integer example](../../../../reference/tensor/random-initialization.md#complete-bounded-integral-example). The original record below retains its own scope, decisions and evidence.


## Status

Complete

## Goal

Add exact typed bounded random initialization for [`INT32`](../../../../glossary.md#data-type--datatype) and [`INT64`](../../../../glossary.md#data-type--datatype) [tensors](../../../../glossary.md#tensor). Two overloads infer
the result [data type](../../../../glossary.md#data-type--datatype) from primitive bounds, use the caller-owned `RandomGenerator`, and delegate
complete carriers to existing flat import.

The task extends the established random helper without data-type [arguments](../../../../glossary.md#term-argument), modulo reduction,
floating conversion, [gradient](../../../../glossary.md#term-gradient) [parameters](../../../../glossary.md#term-parameter), a new package, or changes to completed normal/uniform
behavior.

## [Scope](../../../../glossary.md#term-scope)

- Add one `randomInt(...)` overload with `int` bounds producing [`INT32`](../../../../glossary.md#data-type--datatype).
- Add one `randomInt(...)` overload with `long` bounds producing [`INT64`](../../../../glossary.md#data-type--datatype).
- Use inclusive origin and exclusive bound with `originInclusive < boundExclusive`.
- Invoke the exact bounded [JDK](../../../../glossary.md#term-java-development-kit-jdk) `nextInt(origin, bound)` or `nextLong(origin, bound)` once per logical
  element in row-major order.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [row-major glossary](../../../../glossary.md#term-row-major))</span>
- Require fully static Java-array-sized [shape](../../../../glossary.md#shape) and synthesize [canonical](../../../../glossary.md#term-canonical-form) dense [layout](../../../../glossary.md#layout) with
  `requiresGrad == false`.
- Preserve task 0012F source [ownership](../../../../glossary.md#term-ownership-meanings)/[reproducibility](../../../../glossary.md#artifact-manifest-and-source-provenance)/failure policy.
- Extend existing `TensorRandoms` with exactly two package-private overloads.
- Update the factory public API-shape test and only exact helper-surface assertions in both existing
  random focused tests; add one integral-random focused test.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Finalize Javadocs, Tensor API, glossary, task evidence, master plan, and roadmap through the
  independent documentation pass during implementation.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- floating or [BOOL](../../../../glossary.md#data-type--datatype) output, caller-selected [`DataType`](../../../../glossary.md#data-type--datatype), requires-grad input, rounding, narrowing,
  widening, modulo reduction, truthiness, or conversion<span data-glossary-navigation="true"> ([rounding glossary](../../../../glossary.md#term-rounding); [narrowing glossary](../../../../glossary.md#term-narrowing); [widening glossary](../../../../glossary.md#term-widening))</span>
- unbounded `nextInt()`/`nextLong()`, full-domain convenience, inclusive upper bound, equal/reversed
  bounds, automatic bound swapping, default origin/bound, or default labels
- default/global/thread-local generator, seed/algorithm overload, source factory, splitting/reset,
  synchronization, retention, registry, or service<span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed); [synchronization glossary](../../../../glossary.md#term-synchronization); [registry glossary](../../../../glossary.md#term-registry))</span>
- changing normal/uniform [signatures](../../../../glossary.md#term-signature), [sampling](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters), conversion, [validation](../../../../glossary.md#term-validation), [bytecode](../../../../glossary.md#term-bytecode), or behavior tests
- arbitrary distributions, Bernoulli behavior from 0012I, random ranges as graph Operations, typed
  access/export, provenance, storage/native allocation, compiler/training/runtime/backend behavior<span data-glossary-navigation="true"> ([distributions glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [Bernoulli glossary](../../../../glossary.md#term-bernoulli-distribution); [graph glossary](../../../../glossary.md#graph); [provenance glossary](../../../../glossary.md#provenance); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend))</span>
- dependencies, [Gradle](../../../../glossary.md#term-gradle)/architecture/package changes, another [module](../../../../glossary.md#term-module), or detailed later specs

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
- [Model capability baseline](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- [Model master plan](../master-plan.md)
- [Task 0001](0001-data-type-model.md)
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0003](0003-layout-descriptor-model.md)
- [Task 0007](0007-tensor-descriptor-model.md)
- [Task 0012](0012-tensor-factory.md)
- [Task 0012B](0012b-flat-typed-tensor-import.md)
- [Task 0012F](0012f-random-tensor-creation.md), which owns caller-source policy
- [Task 0012G](0012g-uniform-random-tensor-creation.md), which establishes helper-extension and
  exact-surface regression precedent<span data-glossary-navigation="true"> ([regression glossary](../../../../glossary.md#term-regression))</span>
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span> and [glossary](../../../../glossary.md)

## Capability origin

Bounded integral random initialization is an explicit user-approved addition beyond the legacy
minimum. It does not claim a matching legacy factory implementation and does not introduce a
random [graph](../../../../glossary.md#graph) [Operation](../../../../glossary.md#operation). The capability follows Java's bounded integer generator contracts and the
project's completed [eager](../../../../glossary.md#term-eager-evaluation) [leaf](../../../../glossary.md#term-leaf)-data factory architecture.

## Architecture constraints

- Work remains in `io.github.pho001.synaptik.model.tensor`; no `random`/`randoms` package or public
  distribution abstraction is added.<span data-glossary-navigation="true"> ([distribution glossary](../../../../glossary.md#term-distribution))</span>
- Public overloads belong to `TensorFactory`; mechanics remain package-private in `TensorRandoms`.
- Source is caller-owned transient input and is never stored, substituted, seeded, split, reset,
  synchronized, closed, registered, or looked up.
- Primitive overload selects exact type: `int` bounds -> [INT32](../../../../glossary.md#data-type--datatype), `long` bounds -> [INT64](../../../../glossary.md#data-type--datatype).
- Integral [descriptors](../../../../glossary.md#term-descriptor) are always non-[differentiable](../../../../glossary.md#derivative-and-chain-rule) and expose no meaningless `requiresGrad`.
- Bounded JDK generator methods provide uniform bounded sampling without project-owned modulo
  arithmetic. Synaptik counts one bounded method invocation per output; underlying source-bit
  consumption is generator implementation detail.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters))</span>
- Complete typed carrier precedes exactly one matching flat import; no partial Tensor or direct
  Tensor/storage/ID construction.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Existing normal and uniform APIs/helper logic/tests remain behaviorally unchanged.
- Stop if implementation needs another production type/package, public source/distribution type,
  custom unbiased algorithm, conversion, storage/dependency/architecture change, or another module.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module))</span>

## Package impact

Existing packages used:

- `model.tensor` — TensorFactory, existing TensorRandoms, [descriptor](../../../../glossary.md#term-descriptor) synthesis, tests.
- `model.datatype`, `model.shape`, `model.layout` — integral types, static [shape](../../../../glossary.md#shape), dense geometry.
- `java.util.random` — caller-owned bounded source methods.

Packages added or changed:

- No package is added. Only `model.tensor` changes.

Type placement:

- `TensorFactory` receives two public `randomInt` overloads.
- Existing package-private `TensorRandoms` receives two matching overloads and private typed loops.
- `TensorFactoryIntegralRandomTest` is the new focused same-package behavior test.
- Existing Factory/Random/Uniform tests change only exact public/helper [API](../../../../glossary.md#term-application-programming-interface-api)-shape assertions.

## Required contract

### Public methods

Add exactly:

```java
public static Tensor randomInt(
        Shape shape,
        int originInclusive,
        int boundExclusive,
        RandomGenerator randomGenerator,
        Optional<String> label)

public static Tensor randomInt(
        Shape shape,
        long originInclusive,
        long boundExclusive,
        RandomGenerator randomGenerator,
        Optional<String> label)
```

The first returns [INT32](../../../../glossary.md#data-type--datatype); the second returns [INT64](../../../../glossary.md#data-type--datatype). Both use `requiresGrad == false`. Do not add a
[`DataType`](../../../../glossary.md#data-type--datatype), [gradient](../../../../glossary.md#term-gradient), default-bound, unlabeled, [seed](../../../../glossary.md#term-seed)/factory, `randint`, or `randomLong` [API](../../../../glossary.md#term-application-programming-interface-api).

### Public null [validation](../../../../glossary.md#term-validation)

Each overload validates before helper delegation:

1. null [`shape`](../../../../glossary.md#shape): `NullPointerException`, message `shape`;
2. null `randomGenerator`: `NullPointerException`, message `randomGenerator`;
3. null `label`: `NullPointerException`, message `label`.

These failures consume no source call, allocation, or ID.

### Helper [validation](../../../../glossary.md#term-validation) order

For both overloads, validate before [carrier](../../../../glossary.md#term-storage-carrier) allocation/[sampling](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters):

1. dynamic [shape](../../../../glossary.md#shape): `IllegalArgumentException`,
   `integral random tensor creation requires a fully static shape: <shape>`;
2. read checked `shape.knownElementCount()`; [overflow](../../../../glossary.md#term-overflow) remains `ArithmeticException`;
3. count above `Integer.MAX_VALUE`: `IllegalArgumentException`,
   `integral random tensor element count exceeds Java array limit: required=<required>, maximum=2147483647`;
4. `originInclusive >= boundExclusive`: `IllegalArgumentException`,
   `integral random origin must be less than bound: origin=<origin>, bound=<bound>`;
5. construct [canonical](../../../../glossary.md#term-canonical-form) [contiguous](../../../../glossary.md#term-contiguous) [layout](../../../../glossary.md#layout) and exact [INT32](../../../../glossary.md#data-type--datatype)/[INT64](../../../../glossary.md#data-type--datatype) TensorDescriptor with false [gradient](../../../../glossary.md#term-gradient).

Bounds are validated even for zero-[element](../../../../glossary.md#term-element) output. [Scalar](../../../../glossary.md#term-scalar) consumes one bounded call; empty output
consumes zero.

### [Sampling](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters)

After [validation](../../../../glossary.md#term-validation):

- [INT32](../../../../glossary.md#data-type--datatype) allocates exactly one `int[]`; each [element](../../../../glossary.md#term-element) is the direct result of one
  `randomGenerator.nextInt(originInclusive, boundExclusive)` invocation.
- [INT64](../../../../glossary.md#data-type--datatype) allocates exactly one `long[]`; each element is the direct result of one
  `randomGenerator.nextLong(originInclusive, boundExclusive)` invocation.
- Preserve encounter order, then call exactly one matching `TensorFactory.fromFlatArray(...)`.
- Do not use unbounded draws, `%`, `floorMod`, floating multiplication, rejection logic, streams,
  batching, parallelism, boxing, or intermediate alternate carriers.

The [half-open](../../../../glossary.md#term-half-open-interval) guarantee comes from a conforming `RandomGenerator`. Custom non-conforming results
are not post-validated. Neither source nor [carrier](../../../../glossary.md#term-storage-carrier) is retained.

### Bound-domain limitation

Bounds use the result [carrier](../../../../glossary.md#term-storage-carrier) itself. Therefore the largest representable exclusive bound is
`Integer.MAX_VALUE` or `Long.MAX_VALUE`, and that value is never emitted. A single overload cannot
express the mathematical exclusive bound one greater than the carrier maximum; no unbounded/full-
domain alternative is added in this task. Negative and mixed-sign valid intervals are supported.

### [Ownership](../../../../glossary.md#term-ownership-meanings), [reproducibility](../../../../glossary.md#artifact-manifest-and-source-provenance), and failure effects

- Reuse task 0012F policy: caller owns source, no source management/synchronization, and equivalent
  results require equivalent implementation/state/arguments without interference.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization); [arguments glossary](../../../../glossary.md#term-argument))</span>
- One bounded method invocation per [element](../../../../glossary.md#term-element) is promised; internal random-bit consumption is not.
- Pre-[sampling](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters) failures and source-[carrier](../../../../glossary.md#term-storage-carrier) [OOME](../../../../glossary.md#term-oome) consume no calls/IDs.
- Generator exception propagates before destination/ID; previous calls remain consumed.
- Destination OOME follows all calls but precedes ID.
- Blank label consumes calls and one ID, then fails before copy.
- Exhaustion follows all calls/allocations and does not roll back.
- Unexpected copy failure consumes ID. No state is rolled back.

### Existing helper extension

Add exactly two package-private `TensorRandoms.randomInt(...)` overloads matching public [signatures](../../../../glossary.md#term-signature).
The helper remains final, package-private, stateless, field-free, and privately constructible.
It will expose exactly four package-private entries after this task: normal, uniform, [INT32](../../../../glossary.md#data-type--datatype) random,
and [INT64](../../../../glossary.md#data-type--datatype) random.

Private shared [validation](../../../../glossary.md#term-validation) is allowed only if exact [distribution](../../../../glossary.md#term-distribution)-specific messages/order remain and
normal/uniform [bytecode](../../../../glossary.md#term-bytecode) behavior is unchanged. No new helper type or production hook.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorFactory.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorRandoms.java`

Existing tests, [API](../../../../glossary.md#term-application-programming-interface-api)-[shape](../../../../glossary.md#shape) changes only:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryRandomTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryUniformRandomTest.java`

New focused test:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryIntegralRandomTest.java`

Documentation/[planning](../../../../glossary.md#term-planning) during implementation:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum [scope](../../../../glossary.md#term-scope)

At most two production files, three existing [API](../../../../glossary.md#term-application-programming-interface-api)/helper-[shape](../../../../glossary.md#shape) tests, one new focused test, and five
documentation/[planning](../../../../glossary.md#term-planning) files: eleven paths total. Existing random/normal/uniform behavior assertions
must not be weakened or changed. Do not modify capabilities, completed task specs, other Java/tests,
[Gradle](../../../../glossary.md#term-gradle), agent/architecture docs/tests, another [module](../../../../glossary.md#term-module), or unrelated files. Do not create detailed
0012I/0013 specs. Stop beyond eleven paths or on architecture uncertainty.

The eleven-path [atomic](../../../../glossary.md#resource-transaction-and-rollback) scope is intentional: the existing public and both package-private exact-
surface [regression](../../../../glossary.md#term-regression) tests must recognize the two new overloads. Splitting those expectation updates
from the API addition would leave the repository failing between tasks.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Update TensorFactory type Javadoc and document both overloads fully: inferred type, half-open
  bounds, static shape, source ownership/reproducibility/call count, no modulo bias, carrier-domain
  limitation, label/result, failures, allocation, source, and ID effects.<span data-glossary-navigation="true"> ([half-open glossary](../../../../glossary.md#term-half-open-interval); [shape glossary](../../../../glossary.md#shape); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [reproducibility glossary](../../../../glossary.md#artifact-manifest-and-source-provenance); [bias glossary](../../../../glossary.md#term-bias); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Finalize both helper entries/private loops and preserve existing normal/uniform Javadocs.
- Explain why there is no [DataType](../../../../glossary.md#data-type--datatype)/[gradient](../../../../glossary.md#term-gradient)/default/full-domain [API](../../../../glossary.md#term-application-programming-interface-api) and no new package.
- Review related type/shape/layout/descriptor/Tensor/storage/random Javadocs and record reasoned
  no-change conclusions or stop on out-of-scope discrepancy.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [descriptor glossary](../../../../glossary.md#term-descriptor); [Tensor glossary](../../../../glossary.md#tensor); [scope glossary](../../../../glossary.md#term-scope))</span>

## Acceptance criteria

- Exactly two new public and two matching package-private `randomInt` overloads.
- Existing normal/uniform public/helper behavior and tests unchanged except exact surface sets/counts.
- int -> [INT32](../../../../glossary.md#data-type--datatype), long -> [INT64](../../../../glossary.md#data-type--datatype), static [shape](../../../../glossary.md#shape), strict valid bounds, false [gradient](../../../../glossary.md#term-gradient).
- Exactly one matching bounded generator method invocation per [element](../../../../glossary.md#term-element); zero for empty output.
- No modulo/unbounded/floating [sampling](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters) or conversion.
- Source transient; one [carrier](../../../../glossary.md#term-storage-carrier); one flat import; no direct [Tensor](../../../../glossary.md#tensor)/storage/ID work.
- Exact messages/order/late effects and carrier limits documented/tested.
- Factory/normal/uniform/integral focused suites and aggregate [validation](../../../../glossary.md#term-validation) pass.
- [API](../../../../glossary.md#term-application-programming-interface-api)/helper/[bytecode](../../../../glossary.md#term-bytecode)/import/static-state/eleven-path checks pass.
- Independent documentation/[Javadoc](../../../../glossary.md#term-javadoc)/glossary/status pass completes in same change.
- 0012H becomes Complete only after validation; 0012I stays Draft without a spec.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) before and after documentation review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryRandomTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryUniformRandomTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryIntegralRandomTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Focused tests cover exact [API](../../../../glossary.md#term-application-programming-interface-api)/helper [shape](../../../../glossary.md#shape); [INT32](../../../../glossary.md#data-type--datatype)/[INT64](../../../../glossary.md#data-type--datatype) direct scripted values and exact bounded
[arguments](../../../../glossary.md#term-argument)/order/count; negative/mixed ranges; primitive boundary-adjacent ranges; [scalar](../../../../glossary.md#term-scalar)/empty;
all nulls; dynamic/count/bound validation/messages; dense [descriptors](../../../../glossary.md#term-descriptor)/labels/false [gradient](../../../../glossary.md#term-gradient);
seeded equivalence; non-retention; source exceptions; blank label/exhaustion; allocator restoration;
and absence of modulo/unbounded calls. Do not force [OOME](../../../../glossary.md#term-oome).

Manually inspect `javap -p -c -s`, imports, exact `(II)I` and `(JJ)J` calls, no arithmetic/modulo,
one flat import per overload, unchanged normal/uniform [bytecode](../../../../glossary.md#term-bytecode), exactly four helper entries, no [RNG](../../../../glossary.md#graph-rng-state--graphrngstate)
fields/services/forbidden layers, docs links/fences/whitespace/status, exact eleven paths, and no
later detailed specs.

## Dependencies

- 0012F owns caller-source/[reproducibility](../../../../glossary.md#artifact-manifest-and-source-provenance) policy.
- 0012G establishes safe helper extension and exact-surface update practice.
- 0012B supplies typed flat import; foundational [model](../../../../glossary.md#term-model) tasks remain complete.

## Follow-up tasks

- 0012I: [Bernoulli](../../../../glossary.md#term-bernoulli-distribution) [BOOL](../../../../glossary.md#data-type--datatype) initialization with explicit probability.
- 0013: minimal [Tensor](../../../../glossary.md#tensor) [provenance](../../../../glossary.md#provenance) after initialization sequence.

Do not create detailed follow-up specs here.

## Architecture impact

Expected impact: None. This is [eager](../../../../glossary.md#term-eager-evaluation) copied integral [model](../../../../glossary.md#term-model) [leaf](../../../../glossary.md#term-leaf) data using existing source/helper/
import contracts. No package, [module](../../../../glossary.md#term-module), [dependency](../../../../glossary.md#term-dependency), storage, [lifecycle](../../../../glossary.md#lifecycle), or [runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) change.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks through 0012H, Tensor API/glossary, current factory/random
helper and all factory tests, and Java 26 Gradle configuration.

Implement task 0012H exactly. Modify only TensorFactory.java and TensorRandoms.java for production.
Update TensorFactoryTest, TensorFactoryRandomTest, and TensorFactoryUniformRandomTest only for exact
public/helper surface. Add TensorFactoryIntegralRandomTest. Add exactly two public and two package-
private randomInt overloads; preserve normal/uniform behavior.

Use int bounds -> INT32 and long bounds -> INT64, caller RandomGenerator, static shape, strict
half-open bounds, false gradient, one matching bounded JDK call per element, one typed carrier, and
one flat import. Follow exact validation/messages, source ownership/reproducibility, carrier-domain
limitation, allocation/source/ID effects. Add no DataType/gradient/default/full-domain/modulo/
conversion/Bernoulli/package/service/access/storage/provenance/runtime/backend/build/later specs.
Stop beyond eleven paths or on architecture doubt.

Run all specified validation, then hand actual diff/evidence to a separate clean-context docs agent
to finalize Javadocs/Tensor API/glossary/planning and rerun validation. Update task/master/roadmap
only after both passes. Leave 0012I Draft without a spec. Do not commit or push.
```

## Local decisions

- Two primitive-bound overloads infer exact result type and avoid [DataType](../../../../glossary.md#data-type--datatype)/conversion ambiguity.
- `randomInt` is used as the Java-facing bounded-integer name for both int and long overloads,
  matching the project's overloaded `range` precedent.
- Bounded [JDK](../../../../glossary.md#term-java-development-kit-jdk) methods provide unbiased interval [sampling](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters); Synaptik adds no modulo implementation.
- No [gradient](../../../../glossary.md#term-gradient) [parameter](../../../../glossary.md#term-parameter) is exposed because integral [descriptors](../../../../glossary.md#term-descriptor) cannot require gradients.
- Carrier-maximum-inclusive full-domain sampling is not representable with same-carrier exclusive
  bounds and remains unsupported rather than adding unbounded/default APIs.<span data-glossary-navigation="true"> ([Carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Existing random exact-surface tests require an eleven-path [atomic](../../../../glossary.md#resource-transaction-and-rollback) change; behavior assertions stay.

## Known limitations

- Only bounded [INT32](../../../../glossary.md#data-type--datatype)/[INT64](../../../../glossary.md#data-type--datatype) output, static Java-array-sized [shapes](../../../../glossary.md#shape), and strict non-empty intervals.
- The exclusive bound cannot be mathematically one above [carrier](../../../../glossary.md#term-storage-carrier) maximum.
- One bounded method invocation is promised, not underlying random-bit consumption.
- No source synchronization/cross-provider reproducibility/defaults/full-domain/Bernoulli/random
  Operation/access/runtime generation.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization); [reproducibility glossary](../../../../glossary.md#artifact-manifest-and-source-provenance); [Bernoulli glossary](../../../../glossary.md#term-bernoulli-distribution); [Operation glossary](../../../../glossary.md#operation); [runtime glossary](../../../../glossary.md#term-runtime))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) reviewed architecture/planning rules, completed source policy, current public/helper exact-
surface tests, [JDK](../../../../glossary.md#term-java-development-kit-jdk) bounded int/long methods, and the user-approved capability. The 11-path [scope](../../../../glossary.md#term-scope) is
required to keep exact [API](../../../../glossary.md#term-application-programming-interface-api) [regression](../../../../glossary.md#term-regression) tests passing atomically; no architecture/package/[dependency](../../../../glossary.md#term-dependency)/
storage/build change is required.

Implementation validation before the documentation pass established the requested API, [sampling](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters),
and side-effect behavior. The independent documentation-focused pass then inspected the actual
source, tests, and diff and reran the complete validation [matrix](../../../../glossary.md#term-matrix-meanings) after finalizing Javadocs and the
public reference:

- `./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryTest`
  passed with 7 tests, 0 failures, 0 errors, and 0 skipped.
- `./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryRandomTest`
  passed with 9 tests, 0 failures, 0 errors, and 0 skipped; existing normal assertions remained
  unchanged except the exact helper-surface expectation.
- `./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryUniformRandomTest`
  passed with 11 tests, 0 failures, 0 errors, and 0 skipped; existing uniform assertions remained
  unchanged except the exact helper-surface expectation.
- `./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryIntegralRandomTest`
  passed with 10 tests, 0 failures, 0 errors, and 0 skipped.
- `./gradlew :modules:model:test` passed with 235 tests, 0 failures, 0 errors, and 0 skipped.
- `./gradlew :modules:model:javadoc` passed with 2 actionable tasks.
- `./gradlew test` passed; the final rerun reported 36 actionable tasks up-to-date after the
  preceding pass executed the rebuilt model JAR.<span data-glossary-navigation="true"> ([JAR glossary](../../../../glossary.md#term-jar))</span>
- `git diff --check` passed after the documentation edits.

The complete bounded-integral example in `docs/api/tensor-api.md` was evaluated against the built
[model](../../../../glossary.md#term-model) classes with its scripted source. It produced the documented values `[-3, -1, 0, 4]`, type
[`INT32`](../../../../glossary.md#data-type--datatype), [gradient](../../../../glossary.md#term-gradient) flag `false`, and call count `4`; the focused integral test independently
compiles and verifies the same direct bounded-call, [descriptor](../../../../glossary.md#term-descriptor), [carrier](../../../../glossary.md#term-storage-carrier), and order contract. The
interactive evaluator reported an environment preferences-history [persistence](../../../../glossary.md#term-persistence) error only after
printing `Goodbye`; that post-[evaluation](../../../../glossary.md#term-evaluation) tool-state limitation did not affect [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names),
execution, output, repository state, or the passing [Gradle](../../../../glossary.md#term-gradle) validation.

A Ruby local Markdown [target](../../../../glossary.md#term-target)-and-heading checker ran over the [Tensor](../../../../glossary.md#tensor) API, glossary, this task,
model master plan, and roadmap. It checked 163 local links, including 58 anchors, with 0 failures.
A separate fence check over the same five files reported 0 failures. `git diff --check` confirmed
no whitespace errors in tracked diffs, and a targeted trailing-whitespace [scan](../../../../glossary.md#term-cumulative-scan) of both new files
also passed.

`javap -p -s` confirmed exactly two public and two matching package-private `randomInt` overloads,
and exactly four package-private `TensorRandoms` entries. The helper remains final, package-private,
field-free, and privately constructible. `javap -p -c` confirmed the INT32 loop contains one
`RandomGenerator.nextInt:(II)I`, direct `iastore`, and one `fromFlatArray(...,[I)` call; the [INT64](../../../../glossary.md#data-type--datatype)
loop contains one `nextLong:(JJ)J`, direct `lastore`, and one `fromFlatArray(...,[J)` call. Neither
loop contains arithmetic, modulo, conversion, an unbounded draw, or an alternate carrier. The
normal and uniform loops retain their prior bounded calls, conversions, and single-import bodies;
their production bodies were not edited.

Source/import/static-state inspection confirmed exact public null-check and helper validation
order/messages, inferred INT32/INT64 descriptors with false gradient intent, one exact carrier,
one matching flat import, and no direct Tensor/storage/ID construction in the helper. It found no
stored/default generator, random service/factory/[registry](../../../../glossary.md#term-registry), [seed](../../../../glossary.md#term-seed) handling, [synchronization](../../../../glossary.md#term-synchronization), stream,
reflection, custom modulo/rejection algorithm, [Operation](../../../../glossary.md#operation), [compiler](../../../../glossary.md#term-compiler), planning, [prepare](../../../../glossary.md#prepare), [runtime](../../../../glossary.md#term-runtime),
[backend](../../../../glossary.md#backend), or training dependency. Final status inspection found exactly the eleven authorized
0012H paths after excluding the preexisting dirty task-0012G specification. No detailed 0012I or
0013 specification exists.

Documentation validation used the separate clean context
`/root/implement_model_0012h/review_model_0012h_docs`. The primary profiles were API and [Javadoc](../../../../glossary.md#term-javadoc)
for `TensorFactory`, `TensorRandoms`, and the Tensor API, plus Planning for this task, the model
master plan, and roadmap; General style and Example format were also applied. The pass finalized
the public/helper validation, carrier-domain, [ownership](../../../../glossary.md#term-ownership-meanings), [reproducibility](../../../../glossary.md#artifact-manifest-and-source-provenance), call-count, allocation,
source, and identifier contracts; added the current bounded-integral API and focused example; and
updated the existing Tensor and [Tensor factory](../../../../glossary.md#tensor-factory) glossary status. No new glossary entry was
warranted because [half-open interval](../../../../glossary.md#term-half-open-interval), bounded integer sampling, and `RandomGenerator` are ordinary
mathematical/JDK vocabulary rather than reusable Synaptik domain terms.

Existing [`DataType`](../../../../glossary.md#data-type--datatype), [`Shape`](../../../../glossary.md#shape), `LayoutDescriptor`, `TensorDescriptor`, `Tensor`,
`HostTensorStorage`, and `MemorySegmentStorage` Javadocs remain accurate without edits. Task 0012H
uses their existing integral classification, static-count, dense-[layout](../../../../glossary.md#layout), false-gradient,
[identity](../../../../glossary.md#term-identity-meanings)/label, and [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) [heap](../../../../glossary.md#term-heap)-storage contracts without changing them. Existing normal and
uniform Javadocs remain accurate and their method bodies are untouched. Focused architecture docs,
the capability baseline, architecture tests, backend [conformance](../../../../glossary.md#term-conformance) tests, and integration tests need
no update because the task implements the already planned model-owned [eager](../../../../glossary.md#term-eager-evaluation) [leaf](../../../../glossary.md#term-leaf) initializer
without changing [module](../../../../glossary.md#term-module) ownership, [dependency direction](../../../../glossary.md#java-platform-and-build-tools), [lifecycle](../../../../glossary.md#lifecycle), storage, backend behavior, or
end-to-end execution. Architecture impact is None.

## Implementation notes

- Added exactly two public `TensorFactory.randomInt(...)` overloads and two matching
  package-private `TensorRandoms.randomInt(...)` entries while preserving completed normal and
  uniform behavior.
- Primitive bounds infer exact result type. Validation completes before allocation; each typed
  loop stores one direct bounded JDK result per row-major element in one carrier and delegates once
  to flat import.<span data-glossary-navigation="true"> ([Validation glossary](../../../../glossary.md#term-validation); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [row-major glossary](../../../../glossary.md#term-row-major); [element glossary](../../../../glossary.md#term-element); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Focused tests cover exact API/helper shape, int/long arguments and direct values, scalar/empty and
  signed/boundary-adjacent intervals, validation order/messages, source ownership and failure,
  labels, descriptor facts, bounded reproducibility, late identifier effects, and allocator
  restoration.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [shape glossary](../../../../glossary.md#shape); [arguments glossary](../../../../glossary.md#term-argument); [scalar glossary](../../../../glossary.md#term-scalar); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [descriptor glossary](../../../../glossary.md#term-descriptor); [reproducibility glossary](../../../../glossary.md#artifact-manifest-and-source-provenance))</span>
- The independent documentation pass finalized Javadocs, the Tensor API example, glossary impact,
  validation evidence, and planning status before completion.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [planning glossary](../../../../glossary.md#term-planning))</span>

## Completion summary

- Completed changes: two explicit caller-source bounded-integral factory overloads, stateless exact
  typed sampling/import helper extensions, focused API and behavior tests, finalized Javadocs and
  Tensor API example, glossary status, and synchronized planning frontier.<span data-glossary-navigation="true"> ([sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Tensor glossary](../../../../glossary.md#tensor); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: `TensorFactory.java`, `TensorRandoms.java`, `TensorFactoryTest.java`,
  `TensorFactoryRandomTest.java`, `TensorFactoryUniformRandomTest.java`,
  `TensorFactoryIntegralRandomTest.java`, `docs/api/tensor-api.md`, `docs/glossary.md`, this task,
  model master plan, and roadmap.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- Tests and validation: all four focused suites, all 235 model tests, model Javadoc, root tests,
  bytecode/import/static-state checks, Markdown link/anchor/fence checks, exact eleven-path and
  later-spec checks, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode))</span>`git diff --check` passed. Scripted example evaluation produced the exact
  documented output; the recorded preferences-history limitation occurred only after evaluation.<span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation))</span>
- Documentation-agent review: clean context
  `/root/implement_model_0012h/review_model_0012h_docs` completed the independent pass using API
  and Javadoc, Planning, General style, and Example format profiles.
- Documentation impact: Tensor API and existing glossary entries now describe implemented bounded
  integral sampling, inferred type, strict half-open bounds, direct bounded calls, carrier-domain
  limitation, source policy, and failure effects; focused architecture and capability documents
  remain accurate without task-specific edits.<span data-glossary-navigation="true"> ([half-open glossary](../../../../glossary.md#term-half-open-interval); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Javadoc review: affected factory and helper contracts were finalized; related data type, shape,
  layout, descriptor, Tensor, storage, normal, and uniform contracts remain accurate unchanged.<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Glossary impact: updated existing implementation-status, Tensor, and Tensor factory language; no
  new reusable project term was introduced.<span data-glossary-navigation="true"> ([Tensor factory glossary](../../../../glossary.md#tensor-factory))</span>
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0012H. Subsequent planning promoted task 0012I to the Ready
  frontier with its own detailed specification.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>

Status: Complete
