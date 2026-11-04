# Task 0012I: [Bernoulli](../../../../glossary.md#term-bernoulli-distribution) Random [Tensor](../../../../glossary.md#tensor) Creation

**Editorial reading key (later context, not a status update).**

The caller supplies a probability and RandomGenerator; the factory draws immediately and compares each draw with that probability to populate BOOL storage. Bernoulli means a two-outcome distribution, not a floating tensor of probabilities or a dropout expression. Endpoint probabilities still consume source calls under this record's rule, so the generator's state advances even when all stored outcomes are predetermined. The resulting Tensor is a leaf with values, not the generator's resource owner or a deferred random producer.

Focused current reading: [Current Bernoulli example](../../../../reference/tensor/random-initialization.md#complete-bernoulli-random-example). The original record below retains its own scope, decisions and evidence.


## Status

Complete

## Goal

Add one explicit [Bernoulli](../../../../glossary.md#term-bernoulli-distribution) initializer that creates dense [BOOL](../../../../glossary.md#data-type--datatype) [tensors](../../../../glossary.md#tensor) from a caller-supplied
success probability and `RandomGenerator`. Each logical [element](../../../../glossary.md#term-element) independently compares one uniform
[binary64](../../../../glossary.md#floating-point-precision-and-rounding) draw with the probability and stores [canonical](../../../../glossary.md#term-canonical-form) false/true bytes through existing flat
import.

This finishes the user-approved random initializer sequence without adding numeric truthiness,
[gradient](../../../../glossary.md#term-gradient) [parameters](../../../../glossary.md#term-parameter), a new package, random services, or [graph](../../../../glossary.md#graph)/[runtime](../../../../glossary.md#term-runtime) random behavior.

## [Scope](../../../../glossary.md#term-scope)

- Add exactly one public `TensorFactory.randomBernoulli(...)` method.
- Accept one [finite](../../../../glossary.md#term-finite) probability in the closed interval `[0.0, 1.0]`.
- Require fully static Java-array-sized [shape](../../../../glossary.md#shape) and synthesize a [canonical](../../../../glossary.md#term-canonical-form) dense [BOOL](../../../../glossary.md#data-type--datatype) [descriptor](../../../../glossary.md#term-descriptor) with
  `requiresGrad == false`.
- Invoke the exact supplied `RandomGenerator.nextDouble()` once per logical element in row-major
  order, including probability zero and one.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [row-major glossary](../../../../glossary.md#term-row-major))</span>
- Store byte `1` when `draw < probability`, otherwise byte `0`, then delegate once to BOOL flat
  import.
- Extend existing `TensorRandoms` with one package-private entry and typed BOOL loop.
- Update Factory/Normal/Uniform exact-surface tests only; leave integral behavior tests unchanged;
  add one Bernoulli-focused test.<span data-glossary-navigation="true"> ([Bernoulli glossary](../../../../glossary.md#term-bernoulli-distribution))</span>
- Finalize Javadocs, Tensor API, glossary, task evidence, master plan, and roadmap through the
  required independent documentation pass during implementation.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- numeric [Bernoulli](../../../../glossary.md#term-bernoulli-distribution) output, caller-selected [`DataType`](../../../../glossary.md#data-type--datatype), requires-grad input, numeric truthiness,
  output casting, probability tensors, per-element probabilities, logits, or broadcasting<span data-glossary-navigation="true"> ([tensors glossary](../../../../glossary.md#tensor); [element glossary](../../../../glossary.md#term-element); [broadcasting glossary](../../../../glossary.md#broadcasting))</span>
- `nextBoolean()`, bounded `nextDouble(...)`, probability-specific sampling shortcuts, skipped
  calls at probability zero/one, vectorization, batching, streams, or parallel sampling<span data-glossary-navigation="true"> ([sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters))</span>
- default probability/source/label/shape, seed/algorithm overload, source lookup/factory/splitting/
  reset/synchronization/retention, registry, or service<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [seed glossary](../../../../glossary.md#term-seed); [lookup glossary](../../../../glossary.md#term-lookup); [synchronization glossary](../../../../glossary.md#term-synchronization); [registry glossary](../../../../glossary.md#term-registry))</span>
- changing normal/uniform/integral signatures, method bodies, sampling, validation, or behavior
  assertions<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [validation glossary](../../../../glossary.md#term-validation))</span>
- arbitrary distributions, random graph Operations, typed access/export, provenance, storage/
  native allocation, compiler/training/runtime/backend behavior, dependencies, build/architecture/
  package changes, another module, or detailed later specs<span data-glossary-navigation="true"> ([distributions glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [graph glossary](../../../../glossary.md#graph); [provenance glossary](../../../../glossary.md#provenance); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [module glossary](../../../../glossary.md#term-module))</span>

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
- [Task 0012G](0012g-uniform-random-tensor-creation.md), which defines binary64 uniform-source
  semantics<span data-glossary-navigation="true"> ([binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>
- [Task 0012H](0012h-integral-random-tensor-creation.md), which establishes the current helper
  surface and exact-surface regression scope<span data-glossary-navigation="true"> ([regression glossary](../../../../glossary.md#term-regression); [scope glossary](../../../../glossary.md#term-scope))</span>
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span> and [glossary](../../../../glossary.md)

## Capability origin

[Bernoulli](../../../../glossary.md#term-bernoulli-distribution) initialization is an explicit user-approved addition beyond the legacy minimum. It is an
[eager](../../../../glossary.md#term-eager-evaluation) [BOOL](../../../../glossary.md#data-type--datatype) [leaf](../../../../glossary.md#term-leaf)-data convenience, not evidence that legacy supported it and not a random [graph](../../../../glossary.md#graph)
[Operation](../../../../glossary.md#operation), training [dropout](../../../../glossary.md#dropout--inverted-dropout) policy, or [backend](../../../../glossary.md#backend) [kernel](../../../../glossary.md#kernel).

## Architecture constraints

- Work remains in `io.github.pho001.synaptik.model.tensor`; no new random package or public
  distribution/source type.<span data-glossary-navigation="true"> ([distribution glossary](../../../../glossary.md#term-distribution))</span>
- Public [API](../../../../glossary.md#term-application-programming-interface-api) belongs to `TensorFactory`; package-private [sampling](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters) remains in existing
  `TensorRandoms`.
- Generator is transient caller-owned input and is never stored, substituted, seeded, split,
  reset, synchronized, closed, registered, or looked up.
- Result type is always [BOOL](../../../../glossary.md#data-type--datatype) and non-[differentiable](../../../../glossary.md#derivative-and-chain-rule); no [DataType](../../../../glossary.md#data-type--datatype) or [gradient](../../../../glossary.md#term-gradient) [parameter](../../../../glossary.md#term-parameter) is exposed.
- Exactly one unbounded `nextDouble()` method invocation occurs per output element regardless of
  probability, so source advancement is independent of endpoint optimization.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element))</span>
- One complete canonical byte carrier precedes exactly one BOOL flat import; no partial Tensor or
  direct Tensor/storage/ID construction.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [carrier glossary](../../../../glossary.md#term-storage-carrier); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Existing normal/uniform/integral behavior and [bytecode](../../../../glossary.md#term-bytecode) must remain unchanged.
- Stop if implementation requires another production type/package, probability tensor/logit
  policy, source service, storage/dependency/architecture change, or another module.<span data-glossary-navigation="true"> ([logit glossary](../../../../glossary.md#logit); [dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module))</span>

## Package impact

Existing packages used:

- `model.tensor` — TensorFactory, TensorRandoms, [descriptor](../../../../glossary.md#term-descriptor) synthesis, tests.
- `model.datatype`, `model.shape`, `model.layout` — [BOOL](../../../../glossary.md#data-type--datatype), static [shape](../../../../glossary.md#shape), dense [layout](../../../../glossary.md#layout).
- `java.util.random` — transient caller-owned `nextDouble()` source.

Packages added or changed:

- No package is added. Only `model.tensor` changes.

Type placement:

- `TensorFactory` receives one public `randomBernoulli` method.
- Existing package-private `TensorRandoms` receives one matching entry/private byte loop.
- `TensorFactoryBernoulliRandomTest` is the new focused same-package test.
- Existing Factory/Random/Uniform tests change only exact public/helper surface assertions.
- `TensorFactoryIntegralRandomTest` is not modified because it does not own the exact helper set.

## Required contract

### Public method

Add exactly:

```java
public static Tensor randomBernoulli(
        Shape shape,
        double probability,
        RandomGenerator randomGenerator,
        Optional<String> label)
```

The result is [BOOL](../../../../glossary.md#data-type--datatype) with `requiresGrad == false`. Do not add [DataType](../../../../glossary.md#data-type--datatype)/[gradient](../../../../glossary.md#term-gradient)/default/unlabeled/
[seed](../../../../glossary.md#term-seed)/factory/algorithm/probability-array/[logit](../../../../glossary.md#logit)/[alias](../../../../glossary.md#aliasing-and-arena) overloads.

### Public null [validation](../../../../glossary.md#term-validation)

Validate before helper delegation:

1. null [`shape`](../../../../glossary.md#shape): `NullPointerException`, message `shape`;
2. null `randomGenerator`: `NullPointerException`, message `randomGenerator`;
3. null `label`: `NullPointerException`, message `label`.

These failures consume no source call, allocation, or ID.

### Helper [validation](../../../../glossary.md#term-validation) order

Before [carrier](../../../../glossary.md#term-storage-carrier) allocation/[sampling](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters):

1. dynamic [shape](../../../../glossary.md#shape): `IllegalArgumentException`,
   `bernoulli random tensor creation requires a fully static shape: <shape>`;
2. read checked `knownElementCount()`; arithmetic [overflow](../../../../glossary.md#term-overflow) remains `ArithmeticException`;
3. count above `Integer.MAX_VALUE`: `IllegalArgumentException`,
   `bernoulli random tensor element count exceeds Java array limit: required=<required>, maximum=2147483647`;
4. non-[finite](../../../../glossary.md#term-finite) or numerically outside `[0.0, 1.0]` probability: `IllegalArgumentException`,
   `bernoulli probability must be finite and in [0.0, 1.0]: <probability>`;
5. create [canonical](../../../../glossary.md#term-canonical-form) [contiguous](../../../../glossary.md#term-contiguous) [layout](../../../../glossary.md#layout) and [BOOL](../../../../glossary.md#data-type--datatype) TensorDescriptor with false [gradient](../../../../glossary.md#term-gradient).

Positive and [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero) are accepted as probability zero. Probability one is accepted. Bounds
are validated for empty output. [Scalar](../../../../glossary.md#term-scalar) consumes one draw; empty output consumes zero.

### [Sampling](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters) and [BOOL](../../../../glossary.md#data-type--datatype) representation

After [validation](../../../../glossary.md#term-validation):

1. Allocate exactly one `byte[]` of logical [element](../../../../glossary.md#term-element) count.
2. For each [logical index](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) call `double draw = randomGenerator.nextDouble()` exactly once.
3. Store `(byte) 1` if `draw < probability`; otherwise store `(byte) 0`.
4. Delegate once to `TensorFactory.fromFlatArray(descriptor, label, source)`.
5. Return that exact [Tensor](../../../../glossary.md#tensor).

Do not use `nextBoolean`, `nextDouble(bound)`, `nextDouble(origin,bound)`, skip endpoint draws,
invert the comparison, use `<=`, perform numeric conversion, or post-process through public access.
Flat import receives already [canonical](../../../../glossary.md#term-canonical-form) bytes and preserves them through its BOOL [normalization](../../../../glossary.md#term-normalization).

A conforming generator returns draws in `[0.0, 1.0)`. A custom non-conforming draw is not
post-validated; the specified strict comparison is still applied. Source and [carrier](../../../../glossary.md#term-storage-carrier) are not
retained.

### Probability endpoint semantics

- At probability `0.0` or `-0.0`, every conforming draw produces false, but one draw is still
  consumed per element.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element))</span>
- At probability `1.0`, every conforming draw produces true, but one draw is still consumed per
  element.
- Strict `<` implements the mathematical Bernoulli success rule and avoids including a hypothetical
  draw equal to the probability.<span data-glossary-navigation="true"> ([Bernoulli glossary](../../../../glossary.md#term-bernoulli-distribution))</span>

### [Ownership](../../../../glossary.md#term-ownership-meanings), [reproducibility](../../../../glossary.md#artifact-manifest-and-source-provenance), and failure effects

- Reuse task 0012F policy: caller owns source; no management/synchronization; equivalent results
  require equivalent implementation/state/arguments without interference.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization); [arguments glossary](../../../../glossary.md#term-argument))</span>
- Exactly one `nextDouble()` invocation per [element](../../../../glossary.md#term-element) is promised, not underlying bit consumption.
- Pre-[sampling](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters) failures and source-[carrier](../../../../glossary.md#term-storage-carrier) [OOME](../../../../glossary.md#term-oome) consume no draws/IDs.
- Generator exception occurs before destination/ID; previous draws remain consumed.
- Destination OOME follows all draws but precedes ID.
- Blank label consumes draws and one ID, then fails before copy.
- Exhaustion follows draws/allocations, before copy, without [rollback](../../../../glossary.md#term-rollback).
- Unexpected copy failure consumes ID. No state is rolled back.

### Existing helper extension

Add exactly one package-private `TensorRandoms.randomBernoulli(...)` entry matching the public
[signature](../../../../glossary.md#term-signature). The helper remains final/package-private/stateless/field-free with private constructor.
It exposes exactly five package-private entries afterward: normal, uniform, two integral overloads,
and [Bernoulli](../../../../glossary.md#term-bernoulli-distribution).

Private [validation](../../../../glossary.md#term-validation) sharing is allowed only if exact messages/order remain and existing method
bodies/[bytecode](../../../../glossary.md#term-bytecode) are unchanged. No helper type, strategy, enum, [registry](../../../../glossary.md#term-registry), or production hook.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorFactory.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorRandoms.java`

Existing tests, exact surface only:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryRandomTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryUniformRandomTest.java`

New focused test:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryBernoulliRandomTest.java`

Documentation/[planning](../../../../glossary.md#term-planning) during implementation:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum [scope](../../../../glossary.md#term-scope)

At most two production files, three existing exact-surface tests, one new test, and five docs/
[planning](../../../../glossary.md#term-planning) files: eleven paths total. Existing random [distribution](../../../../glossary.md#term-distribution)/integral behavior assertions and
`TensorFactoryIntegralRandomTest` must remain unchanged. Do not modify capabilities, completed task
specs, other Java/tests, [Gradle](../../../../glossary.md#term-gradle), agent/architecture docs/tests, another [module](../../../../glossary.md#term-module), or unrelated files.
Do not create task 0013 or later specs. Stop beyond eleven paths or on architecture uncertainty.

The [atomic](../../../../glossary.md#resource-transaction-and-rollback) scope is required because the factory and two existing helper-surface tests must accept
the fifth entry in the same buildable change.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Update TensorFactory type Javadoc and document `randomBernoulli` fully: BOOL/non-gradient result,
  probability interval/endpoints, strict comparison, exact source call count, static dense shape,
  ownership/reproducibility/threading, label/result, failures, allocation/source/ID effects.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [shape glossary](../../../../glossary.md#shape); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [reproducibility glossary](../../../../glossary.md#artifact-manifest-and-source-provenance))</span>
- Finalize helper entry/private loop; preserve all existing random Javadocs.
- Explain why endpoint draws are not skipped and why no [DataType](../../../../glossary.md#data-type--datatype)/[gradient](../../../../glossary.md#term-gradient)/package/source abstraction.
- Review related BOOL/type/shape/layout/descriptor/Tensor/storage/random Javadocs and record
  no-change conclusions or stop on out-of-scope discrepancy.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [descriptor glossary](../../../../glossary.md#term-descriptor); [Tensor glossary](../../../../glossary.md#tensor); [scope glossary](../../../../glossary.md#term-scope))</span>

## Acceptance criteria

- Exactly one new public and one package-private `randomBernoulli`.
- Existing normal/uniform/integral implementation behavior unchanged; only exact surface assertions
  change in three existing tests.
- [BOOL](../../../../glossary.md#data-type--datatype), false [gradient](../../../../glossary.md#term-gradient), static Java-array-sized [shape](../../../../glossary.md#shape), [finite](../../../../glossary.md#term-finite) `[0,1]` probability.
- One `nextDouble()` per [element](../../../../glossary.md#term-element) including endpoints; zero for empty output.
- [Canonical](../../../../glossary.md#term-canonical-form) byte result uses strict draw `<` probability; no nextBoolean/bounded draw/shortcut.
- Transient source, one byte [carrier](../../../../glossary.md#term-storage-carrier), one BOOL flat import, no direct [Tensor](../../../../glossary.md#tensor)/storage/ID work.
- Exact [validation](../../../../glossary.md#term-validation)/messages/late effects and endpoint semantics tested.
- All five focused suites and aggregate validation pass.
- [API](../../../../glossary.md#term-application-programming-interface-api)/helper/[bytecode](../../../../glossary.md#term-bytecode)/import/static-state/eleven-path checks pass.
- Independent documentation/[Javadoc](../../../../glossary.md#term-javadoc)/glossary/status pass completes in the same change.
- 0012I becomes Complete only after validation; 0013 remains Draft without a spec.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) before and after documentation review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryRandomTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryUniformRandomTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryIntegralRandomTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryBernoulliRandomTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Focused tests cover exact [API](../../../../glossary.md#term-application-programming-interface-api)/helper surface; scripted draws around probability; strict comparison;
[canonical](../../../../glossary.md#term-canonical-form) bytes; probability 0/-0/1 with calls; [scalar](../../../../glossary.md#term-scalar)/empty; all nulls; dynamic/count/probability
failures/messages; [descriptor](../../../../glossary.md#term-descriptor)/label; seeded equivalence; non-retention; generator exception; blank
label/exhaustion; allocator restoration; and absence of nextBoolean/bounded calls. Do not force [OOME](../../../../glossary.md#term-oome).

Manually inspect `javap -p -c -s`, imports, exact `nextDouble:()D`, `dcmpg`/strict branch and
canonical byte stores, one [BOOL](../../../../glossary.md#data-type--datatype) flat import, unchanged prior random [bytecode](../../../../glossary.md#term-bytecode), exactly five helper
entries, no [RNG](../../../../glossary.md#graph-rng-state--graphrngstate) fields/services/forbidden layers, docs links/fences/whitespace/status, exact eleven
paths, and no task-0013 spec.

## Dependencies

- 0012F owns caller-source/[reproducibility](../../../../glossary.md#artifact-manifest-and-source-provenance) policy.
- 0012G defines uniform [binary64](../../../../glossary.md#floating-point-precision-and-rounding) source semantics.
- 0012H establishes current helper surface and direct typed random extension practice.
- 0012B supplies [canonical](../../../../glossary.md#term-canonical-form) [BOOL](../../../../glossary.md#data-type--datatype) flat import; foundational [model](../../../../glossary.md#term-model) tasks remain complete.

## Follow-up tasks

- 0013: minimal [Tensor](../../../../glossary.md#tensor) [provenance](../../../../glossary.md#provenance) after the completed factory initializer sequence.
- Later work owns random graph Operations, per-element probability tensors, training dropout,
  access/export, compiler capture, and runtime/backend generation.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [element glossary](../../../../glossary.md#term-element); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend))</span>

Do not create detailed follow-up specs here.

## Architecture impact

Expected impact: None. This is [eager](../../../../glossary.md#term-eager-evaluation) copied [BOOL](../../../../glossary.md#data-type--datatype) [model](../../../../glossary.md#term-model) [leaf](../../../../glossary.md#term-leaf) data using existing source/helper/import
contracts. No package/[module](../../../../glossary.md#term-module)/[dependency](../../../../glossary.md#term-dependency)/storage/[lifecycle](../../../../glossary.md#lifecycle)/training/[runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) change.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks through 0012I, Tensor API/glossary, current factory/random
helper and all factory tests, and Java 26 Gradle configuration.

Implement task 0012I exactly. Modify only TensorFactory.java and TensorRandoms.java for production.
Update TensorFactoryTest, TensorFactoryRandomTest, and TensorFactoryUniformRandomTest only for exact
public/helper surface; do not modify IntegralRandomTest. Add TensorFactoryBernoulliRandomTest. Add
exactly one public and one package-private randomBernoulli; preserve all existing random behavior.

Use BOOL/false gradient, caller RandomGenerator, static shape, finite probability in [0,1], one
unbounded nextDouble per element including p=0/1, strict draw < probability, canonical bytes, and
one BOOL flat import. Follow exact validation/messages, ownership/reproducibility, endpoint,
allocation/source/ID effects. Add no DataType/gradient/default/shortcut/nextBoolean/bounded draw/
numeric output/probability tensor/package/service/access/storage/provenance/training/runtime/backend/
build/later specs. Stop beyond eleven paths or on architecture doubt.

Run all specified validation, then hand actual diff/evidence to a separate clean-context docs agent
to finalize Javadocs/Tensor API/glossary/planning and rerun validation. Update task/master/roadmap
only after both passes. Leave 0013 Draft without a spec. Do not commit or push.
```

## Local decisions

- One [BOOL](../../../../glossary.md#data-type--datatype)-only method avoids numeric truthiness, [DataType](../../../../glossary.md#data-type--datatype), conversion, and [gradient](../../../../glossary.md#term-gradient) ambiguity.
- Probability is [finite](../../../../glossary.md#term-finite) and closed `[0,1]`; [signed zero](../../../../glossary.md#term-signed-zero) is accepted as zero.
- One `nextDouble()` is always consumed per element, even at endpoints, for stable source
  advancement independent of optimization.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element))</span>
- Strict `<` is the [Bernoulli](../../../../glossary.md#term-bernoulli-distribution) success rule; [canonical](../../../../glossary.md#term-canonical-form) bytes are created before flat import.
- Existing helper/package remains sufficient; no [distribution](../../../../glossary.md#term-distribution) abstraction or random subpackage.
- Exact-surface tests require an eleven-path [atomic](../../../../glossary.md#resource-transaction-and-rollback) change; behavior assertions remain unchanged.

## Known limitations

- [Scalar](../../../../glossary.md#term-scalar) probability only; [BOOL](../../../../glossary.md#data-type--datatype) output; static Java-array-sized [shape](../../../../glossary.md#shape).
- No per-[element](../../../../glossary.md#term-element) probabilities, logits, [broadcasting](../../../../glossary.md#broadcasting), numeric output, or [gradient](../../../../glossary.md#term-gradient).
- No source synchronization/cross-provider reproducibility/defaults/random Operation/dropout/runtime
  generation.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization); [reproducibility glossary](../../../../glossary.md#artifact-manifest-and-source-provenance); [Operation glossary](../../../../glossary.md#operation); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Custom non-conforming draw results are compared directly without post-[validation](../../../../glossary.md#term-validation).

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) reviewed architecture/planning rules, completed source/[distribution](../../../../glossary.md#term-distribution) policies, current
factory/helper exact-surface tests, [JDK](../../../../glossary.md#term-java-development-kit-jdk) `nextDouble()` semantics, [BOOL](../../../../glossary.md#data-type--datatype) flat import, and the user-
approved [Bernoulli](../../../../glossary.md#term-bernoulli-distribution) capability. Eleven paths were required atomically; no architecture/package/
[dependency](../../../../glossary.md#term-dependency)/storage/build change was needed.

Implementation validation before the documentation pass established the requested [API](../../../../glossary.md#term-application-programming-interface-api), [sampling](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters),
and side-effect behavior. The five focused suites passed together and separately: factory 7,
normal-random 9, uniform-random 11, integral-random 10, and Bernoulli-random 8 tests. One initial
parallel integral-suite invocation collided while [Gradle](../../../../glossary.md#term-gradle) packed the shared test-result [cache](../../../../glossary.md#term-cache); the
immediate sequential rerun passed, so this was a validation-invocation collision rather than a
product failure. The implementation-side aggregate [model](../../../../glossary.md#term-model) suite passed with 243 tests and zero
failures, errors, or skips; model [Javadoc](../../../../glossary.md#term-javadoc), root tests, and `git diff --check` also passed.

Independent documentation validation used clean context
`/root/implement_model_0012i/review_model_0012i_docs`. The pass applied General style plus API and
Javadoc style to `TensorFactory`, `TensorRandoms`, and the [Tensor](../../../../glossary.md#tensor) API; Planning style to this task,
the model master plan, and roadmap; and Example format to the new complete Bernoulli example. It
read the [architecture contract](../../../../glossary.md#architecture-contract) and focused explanations, documentation/planning rules, model
capability/master chain, completed flat-import and random tasks, actual production/test diff,
related model Javadocs, Java 26 build configuration, current API reference, and glossary before
editing only the authorized files.

Post-documentation validation reran exactly:

- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorFactoryTest` — passed 7 tests with zero failures,
  errors, or skips.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorFactoryRandomTest` — passed 9 tests with zero
  failures, errors, or skips; existing normal assertions remained behaviorally unchanged.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorFactoryUniformRandomTest` — passed 11 tests with
  zero failures, errors, or skips; existing uniform assertions remained behaviorally unchanged.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorFactoryIntegralRandomTest` — passed 10 tests with
  zero failures, errors, or skips; its source file remained untouched.
- `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.tensor.TensorFactoryBernoulliRandomTest` — passed 8 tests with
  zero failures, errors, or skips.
- `./gradlew :modules:model:test` — passed 243 tests with zero failures, errors, or skips by XML
  aggregation.
- `./gradlew :modules:model:javadoc` — passed without Javadoc errors or warnings; generated
  `TensorFactory.html` contains the public signature and rendered probability, endpoint-call,
  source-ownership, result, and failure contract.
- `./gradlew test` — passed for the repository. The first post-documentation run reported 36
  actionable tasks, 1 executed and 35 up-to-date; the final evidence-only rerun reported all 36
  up-to-date.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- `git diff --check` — passed.

Manual source and `javap -p -c -s` inspection confirmed exactly one public and one package-private
`randomBernoulli` [signature](../../../../glossary.md#term-signature); exactly five package-private helper entries; no helper fields or [RNG
state](../../../../glossary.md#graph-rng-state--graphrngstate); one new `byte[]`; one unbounded `RandomGenerator.nextDouble:()D` per loop iteration;
`dcmpg` plus the strict branch; [canonical](../../../../glossary.md#term-canonical-form) `iconst_1`/`iconst_0` byte stores; and one
`TensorFactory.fromFlatArray(...,[B)` call. The [descriptor](../../../../glossary.md#term-descriptor) is BOOL with false [gradient](../../../../glossary.md#term-gradient) intent.
Imports remain limited to existing model/JDK types, and source/diff inspection confirmed the prior
normal, uniform, and integral method bodies were not edited. No source [lookup](../../../../glossary.md#term-lookup), bounded double,
`nextBoolean`, stream, reflection, [synchronization](../../../../glossary.md#term-synchronization), direct Tensor/storage/ID construction,
[Operation](../../../../glossary.md#operation), [graph](../../../../glossary.md#graph), [compiler](../../../../glossary.md#term-compiler), training, [runtime](../../../../glossary.md#term-runtime), [backend](../../../../glossary.md#backend), [registry](../../../../glossary.md#term-registry), or service behavior was added.

The complete Bernoulli example in `docs/api/tensor-api.md` was evaluated against the built model
classes with its scripted source and produced canonical bytes `[1, 1, 0, 0]`, type `BOOL`, gradient
flag `false`, and call count `4`. The [host](../../../../glossary.md#term-host) JShell then reported its known macOS preferences-history
flush failure after [evaluation](../../../../glossary.md#term-evaluation); this environmental tool-state limitation did not affect
[compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names), execution, printed values, repository state, or Gradle validation. Two attempts to
reroute JShell preferences failed before evaluation because the platform could not instantiate or
link the alternate preferences implementation, so they supplied no product evidence. The focused
Bernoulli suite independently compiles and verifies the same call, [carrier](../../../../glossary.md#term-storage-carrier), descriptor, and output
contract.

A local Markdown [target](../../../../glossary.md#term-target) and heading-anchor checker passed for the five changed documentation/
planning files with 167 links checked, including 58 anchors, and zero failures. Preliminary
checker drafts failed before inspecting content because Ruby interpolated the heading-quantifier
syntax and the host Ruby lacks `Array#filter_map`; the compatible corrected invocation passed. A
separate fence check passed all five files, targeted trailing-whitespace search found no matches,
and terminology/status review found no stale planned
Bernoulli claim. Final [scope](../../../../glossary.md#term-scope) review found exactly the eleven task-0012I paths: two production
files, three existing exact-surface tests, the new Bernoulli test, Tensor API, glossary, this task,
model master plan, and roadmap. The preexisting dirty task-0012H specification remained untouched
and excluded from that count. No task-0013 specification exists; 0013 remains Draft.

Existing [`DataType`](../../../../glossary.md#data-type--datatype)/BOOL, [`Shape`](../../../../glossary.md#shape), `LayoutDescriptor`, `TensorDescriptor`, `Tensor`,
`HostTensorStorage`, and `MemorySegmentStorage` Javadocs remain accurate without edits. Task 0012I
uses their existing [boolean](../../../../glossary.md#data-type--datatype) classification, static-count, dense-[layout](../../../../glossary.md#layout), false-gradient,
[identity](../../../../glossary.md#term-identity-meanings)/label, canonical flat-import, and [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) [heap](../../../../glossary.md#term-heap)-storage contracts without changing them.
Existing normal, uniform, and integral Javadocs also remain accurate; only the factory/helper type
surface and additive Bernoulli contracts required revision. The Tensor API now treats Bernoulli
creation as current and documents probability, endpoint calls, strict comparison, canonical BOOL
storage, [ownership](../../../../glossary.md#term-ownership-meanings), [reproducibility](../../../../glossary.md#artifact-manifest-and-source-provenance), and failure effects.

The glossary's implementation status and existing Tensor/TensorFactory entries now include
Bernoulli creation. No new reusable glossary term was warranted: Bernoulli distribution,
probability, and `RandomGenerator` are ordinary mathematical/JDK vocabulary, while the behavior is
one factory operation rather than a new Synaptik domain concept. `ARCHITECTURE.md`, focused
architecture documentation, ADRs, architecture tests, backend-[conformance](../../../../glossary.md#term-conformance) tests, integration
tests, and `capabilities.md` remain unchanged because this task implements an already approved
model-owned [eager](../../../../glossary.md#term-eager-evaluation) [leaf](../../../../glossary.md#term-leaf) initializer and changes no [module](../../../../glossary.md#term-module)/package direction, dependency, [lifecycle](../../../../glossary.md#lifecycle),
storage ownership, backend behavior, or end-to-end execution contract.

## Implementation notes

- Added exactly one public `TensorFactory.randomBernoulli(...)` method and one package-private
  `TensorRandoms.randomBernoulli(...)` entry while preserving completed normal, uniform, and
  integral behavior.
- Validation completes before sampling. The helper creates one canonical dense BOOL descriptor,
  consumes one unbounded draw per element even at probability endpoints, stores the strict
  comparison as canonical bytes, and delegates once to BOOL flat import.<span data-glossary-navigation="true"> ([Validation glossary](../../../../glossary.md#term-validation); [sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters); [canonical glossary](../../../../glossary.md#term-canonical-form); [BOOL glossary](../../../../glossary.md#data-type--datatype); [descriptor glossary](../../../../glossary.md#term-descriptor); [element glossary](../../../../glossary.md#term-element))</span>
- Focused tests cover exact public/helper surface, strict equality and non-conforming draws,
  endpoints including signed zero, canonical bytes, scalar/empty shapes, exact validation and
  messages, source ownership and failure, bounded reproducibility, labels, identifier effects,
  allocator restoration, and absence of bounded-double or boolean calls.<span data-glossary-navigation="true"> ([signed zero glossary](../../../../glossary.md#term-signed-zero); [scalar glossary](../../../../glossary.md#term-scalar); [shapes glossary](../../../../glossary.md#shape); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [reproducibility glossary](../../../../glossary.md#artifact-manifest-and-source-provenance); [boolean glossary](../../../../glossary.md#data-type--datatype))</span>
- The independent documentation pass finalized Javadocs, the Tensor API example, glossary impact,
  validation evidence, and synchronized planning status before completion.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>

## Completion summary

- Completed changes: One explicit caller-source Bernoulli factory method, stateless canonical BOOL
  sampling/import helper extension, focused API and behavior tests, finalized Javadocs and Tensor
  API example, glossary status, and synchronized planning frontier.<span data-glossary-navigation="true"> ([Bernoulli glossary](../../../../glossary.md#term-bernoulli-distribution); [canonical glossary](../../../../glossary.md#term-canonical-form); [BOOL glossary](../../../../glossary.md#data-type--datatype); [sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Tensor glossary](../../../../glossary.md#tensor); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: `TensorFactory.java`, `TensorRandoms.java`, `TensorFactoryTest.java`,
  `TensorFactoryRandomTest.java`, `TensorFactoryUniformRandomTest.java`,
  `TensorFactoryBernoulliRandomTest.java`, `docs/api/tensor-api.md`, `docs/glossary.md`, this task,
  model master plan, and roadmap.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- Tests and validation: All five focused suites, all 243 model tests, model Javadoc, root tests,
  bytecode/import/static-state checks, executable example, Markdown link/anchor/fence checks,
  exact eleven-path and later-spec checks, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode))</span>`git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0012i/review_model_0012i_docs` completed the independent pass using API
  and Javadoc, Planning, General style, and Example format profiles.
- Documentation impact: Tensor API and existing glossary entries now describe implemented
  Bernoulli sampling, finite closed probability, endpoint source calls, strict comparison,
  canonical BOOL storage, caller-source policy, and failure effects; architecture and capability
  documents remain accurate without task-specific edits.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite))</span>
- Javadoc review: Affected factory and helper contracts were finalized; related BOOL/data type,
  shape, layout, descriptor, Tensor, storage, normal, uniform, and integral contracts remain
  accurate unchanged for the reasons recorded above.<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Glossary impact: Existing implementation-status, Tensor, and TensorFactory language was updated;
  no new reusable project term was introduced.
- Architecture impact: None. No architecture document, ADR, architecture test, dependency rule,
  module boundary, package direction, or capability-baseline decision changed.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [architecture test glossary](../../../../glossary.md#term-architecture-test); [dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module))</span>
- Unresolved issues: None.
- Follow-up required: None for task 0012I. Task 0013 is the next separate Draft planning frontier
  without a detailed specification.

Status: Complete
