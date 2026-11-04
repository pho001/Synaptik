# Task 0012G: Uniform Random [Tensor](../../../../glossary.md#tensor) Creation

## Status

Complete

## Goal

Extend the completed explicit-source random factory with one method for dense floating [tensors](../../../../glossary.md#tensor)
sampled from a continuous [uniform distribution](../../../../glossary.md#term-uniform-distribution) over a caller-supplied [half-open interval](../../../../glossary.md#term-half-open-interval).

The task reuses task 0012F's caller-owned `RandomGenerator`, static dense [descriptor](../../../../glossary.md#term-descriptor), [carrier](../../../../glossary.md#term-storage-carrier),
flat-import, [reproducibility](../../../../glossary.md#artifact-manifest-and-source-provenance), and failure-side-effect policies without adding random services or
another package.

## [Scope](../../../../glossary.md#term-scope)

- Add exactly one `TensorFactory.randomUniform(...)` method.
- Support [`FLOAT64`](../../../../glossary.md#term-float64), [`FLOAT32`](../../../../glossary.md#term-float32), and [`BFLOAT16`](../../../../glossary.md#term-bfloat16) output.
- Require [finite](../../../../glossary.md#term-finite) lower/upper bounds with `lowerBoundInclusive < upperBoundExclusive`.
- Require fully static Java-array-sized shape, canonical dense layout, explicit label/gradient,
  and caller-owned <span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [canonical glossary](../../../../glossary.md#term-canonical-form); [layout glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient))</span>`RandomGenerator`.
- Invoke `randomGenerator.nextDouble(lowerBoundInclusive, upperBoundExclusive)` exactly once per
  logical element in row-major order.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [row-major glossary](../../../../glossary.md#term-row-major))</span>
- Store [binary64](../../../../glossary.md#floating-point-precision-and-rounding) [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) directly, narrow once for FLOAT32, and narrow then use
  `BFloat16Bits.fromFloat(...)` for BFLOAT16.
- Modify the existing `TensorRandoms` rather than introduce another helper or package.
- Update the factory public API-shape test, update only the existing random helper-entry shape
  assertion required by the second package-private entry, and add one focused uniform-random test.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Finalize Javadoc, Tensor API, glossary, task evidence, master plan, and roadmap through the
  required independent documentation pass during implementation.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Tensor glossary](../../../../glossary.md#tensor))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- normal behavior changes, integer or Bernoulli sampling, other distributions, or arbitrary
  sampler callbacks<span data-glossary-navigation="true"> ([Bernoulli glossary](../../../../glossary.md#term-bernoulli-distribution); [sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters); [distributions glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>
- default/global/thread-local source, seed/algorithm overloads, source creation/splitting/reset,
  synchronization, retention, registry, or service lookup<span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed); [synchronization glossary](../../../../glossary.md#term-synchronization); [registry glossary](../../../../glossary.md#term-registry); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- non-floating output, implicit category conversion, infinite bounds, equal/reversed bounds, or
  closed-upper-bound guarantees
- promising that narrowed FLOAT32/BFLOAT16 stored values remain strictly below the binary64 upper
  bound after rounding<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [rounding glossary](../../../../glossary.md#term-rounding))</span>
- defaults for bounds/type/[shape](../../../../glossary.md#shape)/label/[gradient](../../../../glossary.md#term-gradient)/source or convenience overloads
- dynamic shape, caller descriptor/layout, view/scatter population, typed access/export, storage
  changes, native allocation, provenance, random Operations, compiler/training/runtime/backend
  behavior, dependencies, build changes, architecture changes, or another module<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [layout glossary](../../../../glossary.md#layout); [view glossary](../../../../glossary.md#term-view-meanings); [provenance glossary](../../../../glossary.md#provenance); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [module glossary](../../../../glossary.md#term-module))</span>
- detailed task 0012H, 0012I, 0013, or later specifications

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
- [Task 0012F](0012f-random-tensor-creation.md), authoritative prerequisite for random-source and
  reproducibility policy<span data-glossary-navigation="true"> ([reproducibility glossary](../../../../glossary.md#artifact-manifest-and-source-provenance))</span>
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span> and [glossary](../../../../glossary.md)

## Architecture constraints

- Work remains in `io.github.pho001.synaptik.model.tensor`; do not add `random`, `randoms`, or
  another package.
- Public [API](../../../../glossary.md#term-application-programming-interface-api) remains on `TensorFactory`; package-private mechanics remain in existing
  `TensorRandoms` so package-private visibility is preserved without a public implementation type.
- The generator is transient caller-owned input and is never stored, selected, seeded, split,
  reset, synchronized, closed, or looked up.
- Uniform creation is [eager](../../../../glossary.md#term-eager-evaluation) copied [leaf](../../../../glossary.md#term-leaf) data, not [Operation](../../../../glossary.md#operation)/[compiler](../../../../glossary.md#term-compiler)/[runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) behavior.
- Only floating types are accepted; integral and [Bernoulli](../../../../glossary.md#term-bernoulli-distribution) policies belong to tasks 0012H/0012I.
- A complete exact typed carrier is sampled before exactly one matching flat import; no partial
  Tensor is exposed and no direct Tensor/storage/ID construction is added.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Existing normal-random API and helper behavior must remain [bytecode](../../../../glossary.md#term-bytecode)/API compatible.
- Stop if implementation needs another production type, public distribution/source abstraction,
  retained service, package, storage/dependency/architecture change, or another module.<span data-glossary-navigation="true"> ([distribution glossary](../../../../glossary.md#term-distribution); [dependency glossary](../../../../glossary.md#term-dependency); [module glossary](../../../../glossary.md#term-module))</span>

## Package impact

Existing packages used:

- `model.tensor` — TensorFactory and existing package-private random helper.
- `model.datatype`, `model.shape`, and `model.layout` — type, static [shape](../../../../glossary.md#shape), and dense [descriptor](../../../../glossary.md#term-descriptor).
- `java.util.random` — transient caller-owned source.

Packages added or changed:

- No package is added. Only `model.tensor` changes.

Type placement:

- `TensorFactory` receives one public method.
- Existing package-private `TensorRandoms` receives one package-private entry and private typed
  sampling dispatch.<span data-glossary-navigation="true"> ([sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- `TensorFactoryUniformRandomTest` is the focused same-package test.

## Required contract

### Public method

Add exactly:

```java
public static Tensor randomUniform(
        Shape shape,
        DataType dataType,
        double lowerBoundInclusive,
        double upperBoundExclusive,
        RandomGenerator randomGenerator,
        Optional<String> label,
        boolean requiresGrad)
```

Do not add [aliases](../../../../glossary.md#aliasing-and-arena), defaults, unlabeled/[seed](../../../../glossary.md#term-seed)/factory/algorithm overloads, or a [distribution](../../../../glossary.md#term-distribution) enum.

### Public null [validation](../../../../glossary.md#term-validation)

Validate before helper delegation in exact order:

1. null [`shape`](../../../../glossary.md#shape): `NullPointerException`, message `shape`;
2. null `dataType`: `NullPointerException`, message `dataType`;
3. null `randomGenerator`: `NullPointerException`, message `randomGenerator`;
4. null `label`: `NullPointerException`, message `label`.

No null failure consumes source calls, allocation, or ID.

### Helper [validation](../../../../glossary.md#term-validation) order

Before [carrier](../../../../glossary.md#term-storage-carrier) allocation or [sampling](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters):

1. dynamic [shape](../../../../glossary.md#shape): `IllegalArgumentException`,
   `uniform random tensor creation requires a fully static shape: <shape>`;
2. read checked `knownElementCount()`; arithmetic [overflow](../../../../glossary.md#term-overflow) remains `ArithmeticException`;
3. count above `Integer.MAX_VALUE`: `IllegalArgumentException`,
   `uniform random tensor element count exceeds Java array limit: required=<required>, maximum=2147483647`;
4. non-floating type: `IllegalArgumentException`,
   `uniform random creation requires floating data type: <dataType>`;
5. non-[finite](../../../../glossary.md#term-finite) lower bound: `IllegalArgumentException`,
   `uniform random lower bound must be finite: <lowerBoundInclusive>`;
6. non-finite upper bound: `IllegalArgumentException`,
   `uniform random upper bound must be finite: <upperBoundExclusive>`;
7. lower not strictly less than upper: `IllegalArgumentException`,
   `uniform random lower bound must be less than upper bound: lower=<lower>, upper=<upper>`;
8. create [canonical](../../../../glossary.md#term-canonical-form) [contiguous](../../../../glossary.md#term-contiguous) [layout](../../../../glossary.md#layout) and `TensorDescriptor` with explicit [gradient](../../../../glossary.md#term-gradient) intent.

[Scalar](../../../../glossary.md#term-scalar) consumes one call. Zero-[element](../../../../glossary.md#term-element) shape is valid and consumes zero calls.

### [Sampling](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters) and conversion

After [validation](../../../../glossary.md#term-validation), allocate exactly one matching source [carrier](../../../../glossary.md#term-storage-carrier). For each [logical index](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) call
`randomGenerator.nextDouble(lowerBoundInclusive, upperBoundExclusive)` exactly once and store:

| Type | Carrier | Conversion |
|---|---|---|
| [FLOAT64](../../../../glossary.md#term-float64) | `double[]` | exact returned [binary64](../../../../glossary.md#floating-point-precision-and-rounding) [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) |
| [FLOAT32](../../../../glossary.md#term-float32) | `float[]` | Java binary64-to-[binary32](../../../../glossary.md#floating-point-precision-and-rounding) [narrowing](../../../../glossary.md#term-narrowing) |
| [BFLOAT16](../../../../glossary.md#term-bfloat16) | `short[]` | binary32 narrowing then `BFloat16Bits.fromFloat(...)` |

Delegate once to matching `TensorFactory.fromFlatArray(...)` and return its exact [Tensor](../../../../glossary.md#tensor). Do not
replace the bounded generator call with `nextDouble()` plus arithmetic, streams, batching,
parallelism, or [FMA](../../../../glossary.md#term-fused-multiply-add-fma).

The [half-open](../../../../glossary.md#term-half-open-interval) `[lower, upper)` promise applies to the conforming generator's binary64 sample.
FLOAT32/BFLOAT16 [rounding](../../../../glossary.md#term-rounding) may produce a stored value equal to the corresponding narrowed upper
bound or lower-rounded representable value. This is conversion behavior, not a closed interval.
A custom non-conforming generator result is not post-validated. The source and carrier are not
retained.

### [Ownership](../../../../glossary.md#term-ownership-meanings), [reproducibility](../../../../glossary.md#artifact-manifest-and-source-provenance), and failure effects

- Reuse task 0012F policy: caller owns/configures source; no cross-algorithm/provider/JDK promise;
  no synchronization; equivalent results require equivalent source state and no interference.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- Pre-[sampling](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters) [validation](../../../../glossary.md#term-validation) consumes no calls or IDs. Source-[carrier](../../../../glossary.md#term-storage-carrier) [OOME](../../../../glossary.md#term-oome) consumes neither.
- A generator exception propagates before destination/ID; earlier calls remain consumed.
- After all [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters), flat import allocates destination then ID.
- Destination OOME consumes all draws but no ID.
- Blank label consumes all draws and one ID, then fails before copy.
- Exhaustion consumes all draws and allocations before failing; no [rollback](../../../../glossary.md#term-rollback).
- Unexpected copy failure consumes ID. No source/array/storage/ID rollback is attempted.

### Existing helper extension

Add exactly one new package-private `TensorRandoms.randomUniform(...)` entry with the same
[signature](../../../../glossary.md#term-signature). Preserve the helper's final/package-private/stateless [shape](../../../../glossary.md#shape), private constructor,
existing `randomNormal(...)`, and absence of fields/public surface.

Private implementation may share [descriptor](../../../../glossary.md#term-descriptor)/count [validation](../../../../glossary.md#term-validation) only when exact [distribution](../../../../glossary.md#term-distribution)-specific
validation order/messages remain visible and normal behavior is unchanged. Use typed loops without
reflection, streams, per-[element](../../../../glossary.md#term-element) boxing, or temporary double carriers for narrower types.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorFactory.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorRandoms.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryRandomTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryUniformRandomTest.java`

Documentation/[planning](../../../../glossary.md#term-planning) during implementation:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum [scope](../../../../glossary.md#term-scope)

At most two production files, two existing tests, one new test, and five documentation/[planning](../../../../glossary.md#term-planning)
files: ten paths. `TensorFactoryRandomTest` may change only its exact package-private helper-surface
expectation from the completed one-entry [shape](../../../../glossary.md#shape) to the required two-entry shape; normal behavior
assertions must remain unchanged. Do not modify other Java/tests, task 0012F implementation
evidence, [Gradle](../../../../glossary.md#term-gradle),
`AGENTS.md`, architecture/focused docs/tests, capabilities, another [module](../../../../glossary.md#term-module), or unrelated docs. Do
not create detailed task 0012H/0012I/0013 specs. Stop if scope or architecture must expand.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Update TensorFactory type Javadoc and fully document `randomUniform(...)` bounds, binary64
  half-open semantics, narrowing caveat, types, shape, source ownership/reproducibility/threading,
  call count, label/gradient, result, failures, allocation, and ID effects.<span data-glossary-navigation="true"> ([binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [half-open glossary](../../../../glossary.md#term-half-open-interval); [narrowing glossary](../../../../glossary.md#term-narrowing); [shape glossary](../../../../glossary.md#shape); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [reproducibility glossary](../../../../glossary.md#artifact-manifest-and-source-provenance); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Finalize the new helper entry/private implementation Javadocs without changing normal contracts.
- Explain why no new package, source abstraction, [seed](../../../../glossary.md#term-seed) [API](../../../../glossary.md#term-application-programming-interface-api), or [distribution](../../../../glossary.md#term-distribution) enum is introduced.
- Review existing normal/BFLOAT16/type/shape/layout/descriptor/Tensor/storage Javadocs and record
  reasoned no-change conclusions or stop on out-of-scope discrepancy.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [layout glossary](../../../../glossary.md#layout); [descriptor glossary](../../../../glossary.md#term-descriptor); [Tensor glossary](../../../../glossary.md#tensor); [scope glossary](../../../../glossary.md#term-scope))</span>

## Acceptance criteria

- Exactly one new public `randomUniform(...)` and one matching package-private helper entry.
- Existing normal [API](../../../../glossary.md#term-application-programming-interface-api)/helper and all prior factory behavior/tests remain unchanged.
- Only three floating types, static array-sized shape, finite ordered bounds, explicit source/label/
  gradient accepted.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [finite glossary](../../../../glossary.md#term-finite); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Exactly one bounded `nextDouble(lower, upper)` call per [element](../../../../glossary.md#term-element); zero for empty result.
- Exact conversions and narrowed-bound caveat documented/tested.
- Source is transient and never substituted/stored/synchronized/managed.
- One [carrier](../../../../glossary.md#term-storage-carrier), one matching flat import, no direct [Tensor](../../../../glossary.md#tensor)/storage/ID work.
- Exact [validation](../../../../glossary.md#term-validation)/messages and late failure effects pass focused tests.
- API/helper/[bytecode](../../../../glossary.md#term-bytecode)/import/static-state/ten-path checks pass.
- Independent documentation/[Javadoc](../../../../glossary.md#term-javadoc)/glossary/status pass completes in the same change.
- 0012G becomes Complete only after validation; 0012H remains Draft without a spec.

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) before and after documentation review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryRandomTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryUniformRandomTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Focused tests cover helper/[API](../../../../glossary.md#term-application-programming-interface-api) [shape](../../../../glossary.md#shape); exact bounded-call [arguments](../../../../glossary.md#term-argument)/order/count; three conversions;
[scalar](../../../../glossary.md#term-scalar)/empty; all nulls; dynamic/count/type/bound failures; [signed zero](../../../../glossary.md#term-signed-zero) and very narrow valid
intervals; [gradients](../../../../glossary.md#term-gradient)/label/dense [descriptor](../../../../glossary.md#term-descriptor); equivalent seeded source; non-retention; generator
exception; blank label/exhaustion and allocator restoration. Do not force [OOME](../../../../glossary.md#term-oome).

Manually inspect `javap -p -c -s`, imports, exact bounded call, conversions/delegation, unchanged
normal [bytecode](../../../../glossary.md#term-bytecode)/API, no stored/default/static source or forbidden layers, docs links/fences/
whitespace/status, exact ten paths, and absence of later detailed specs.

## Dependencies

- Task 0012F defines and implements source/[reproducibility](../../../../glossary.md#artifact-manifest-and-source-provenance)/random helper policy.
- Task 0012B supplies typed copied flat import.
- Foundational datatype/[shape](../../../../glossary.md#shape)/[layout](../../../../glossary.md#layout)/[descriptor](../../../../glossary.md#term-descriptor)/storage/[Tensor](../../../../glossary.md#tensor) tasks remain complete.

## Follow-up tasks

- 0012H: typed integral random [tensors](../../../../glossary.md#tensor) with exclusive bounds.
- 0012I: [BOOL](../../../../glossary.md#data-type--datatype) [Bernoulli](../../../../glossary.md#term-bernoulli-distribution) tensors with explicit probability.
- 0013: minimal Tensor [provenance](../../../../glossary.md#provenance) after factory initialization work.

Do not create detailed follow-up specs here.

## Architecture impact

Expected impact: None. This is another [eager](../../../../glossary.md#term-eager-evaluation) copied [model](../../../../glossary.md#term-model) [leaf](../../../../glossary.md#term-leaf) initializer using the existing
transient source and helper. No package/[module](../../../../glossary.md#term-module)/[lifecycle](../../../../glossary.md#lifecycle)/storage/[runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) rule changes.

## Implementation prompt

Use this prompt in a separate clean-context implementation thread:

```text
Read AGENTS.md, ARCHITECTURE.md, focused architecture docs, documentation/planning rules, roadmap,
model capabilities/master plan, tasks through 0012G, Tensor API/glossary, current factory/random
helper and all factory tests, and Java 26 Gradle configuration.

Implement task 0012G exactly. Modify only TensorFactory.java and TensorRandoms.java for production,
update TensorFactoryTest for the public API shape, update only the package-private helper-surface
expectation in TensorFactoryRandomTest from one entry to two, and add
TensorFactoryUniformRandomTest. Add exactly one public and one package-private randomUniform entry.
Preserve randomNormal and all prior behavior/tests.

Use caller RandomGenerator, static shape, three floating types, finite strictly ordered bounds, one
nextDouble(lower, upper) per element, exact carrier conversion, and one flat import. Follow exact
validation, ownership, reproducibility, rounding caveat, allocation, source, and ID effects. Add no
package/default/seed/source service/other distribution/integral/Bernoulli/access/storage/provenance/
compiler/runtime/backend/build/follow-up specs. Stop beyond ten paths or on architecture doubt.

Run all specified validation, then hand actual diff/evidence to a separate clean-context docs agent
to finalize Javadocs/Tensor API/glossary/planning and rerun validation. Update task/master/roadmap
status only after both passes. Leave 0012H Draft without a spec. Do not commit or push.
```

## Local decisions

- Public and helper types stay in `model.tensor`; current two-method cohesion does not justify a
  subpackage or public distribution abstraction.<span data-glossary-navigation="true"> ([distribution glossary](../../../../glossary.md#term-distribution))</span>
- The bounded [JDK](../../../../glossary.md#term-java-development-kit-jdk) `nextDouble(origin, bound)` call defines the binary64 half-open sample directly
  and fixes one source-method call per element.<span data-glossary-navigation="true"> ([binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [half-open glossary](../../../../glossary.md#term-half-open-interval); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [element glossary](../../../../glossary.md#term-element))</span>
- Narrowed stored values may round to a representation equal to a narrowed upper bound; avoiding
  that would require distribution-biased resampling or clamping and is not introduced.
- [Finite](../../../../glossary.md#term-finite) strict bounds avoid undefined infinite-width or degenerate intervals.
- Normal behavior remains closed and unchanged; uniform is a separate focused task sharing only
  stable source/descriptor/import mechanics.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor))</span>

## Known limitations

- Floating uniform output only, static Java-array-sized [shapes](../../../../glossary.md#shape) only.
- [Half-open](../../../../glossary.md#term-half-open-interval) guarantee is on [binary64](../../../../glossary.md#floating-point-precision-and-rounding) source [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) before [narrowing](../../../../glossary.md#term-narrowing).
- No source [synchronization](../../../../glossary.md#term-synchronization) or cross-algorithm/provider/[JDK](../../../../glossary.md#term-java-development-kit-jdk) [reproducibility](../../../../glossary.md#artifact-manifest-and-source-provenance).
- No defaults, [seeds](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings), other [distributions](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings), random Operations, typed access, or [runtime](../../../../glossary.md#term-runtime) generation.

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) reviewed architecture/planning rules, completed random task/source policy, [JDK](../../../../glossary.md#term-java-development-kit-jdk)
`RandomGenerator.nextDouble(origin, bound)`, current helper/tests, and the user-approved capability
extension. No architecture, package, [dependency](../../../../glossary.md#term-dependency), storage, or build change is required.

Implementation validation:

- `./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryTest`
  passed with 7 tests, 0 failures, 0 errors, and 0 skipped.
- `./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryRandomTest`
  passed with 9 tests, 0 failures, 0 errors, and 0 skipped; existing normal behavior assertions
  remained unchanged.
- `./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryUniformRandomTest`
  passed with 11 tests, 0 failures, 0 errors, and 0 skipped.
- `./gradlew :modules:model:test` passed with 225 tests, 0 failures, 0 errors, and 0 skipped.
- `./gradlew :modules:model:javadoc` passed with 2 actionable tasks.
- `./gradlew test` passed with 36 actionable tasks, 1 executed and 35 up-to-date.
- `git diff --check` passed.
- The complete uniform example in `docs/api/tensor-api.md` was copied without semantic changes to
  `/private/tmp/synaptik-uniform-example/UniformRandomExample.java`, compiled with
  `javac -cp modules/model/build/classes/java/main -d /private/tmp/synaptik-uniform-example/classes
  /private/tmp/synaptik-uniform-example/UniformRandomExample.java`, and [run](../../../../glossary.md#term-run) with
  `java -cp modules/model/build/classes/java/main:/private/tmp/synaptik-uniform-example/classes
  UniformRandomExample`. [Compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) passed without diagnostics, and execution printed exactly:

  ```text
  [-1.0, 0.0, 0.5, 1.0]
  4
  DENSE_CONTIGUOUS
  ```

  The temporary source is outside the repository and is not part of task [scope](../../../../glossary.md#term-scope).
- A Ruby local Markdown [target](../../../../glossary.md#term-target)-and-GitHub-heading-slug checker ran over `docs/api/tensor-api.md`,
  `docs/glossary.md`, this task, the model master plan, and the roadmap. It checked 159 local
  links, including 58 anchor-bearing links, and reported `failures=0`. A separate fence check over
  the same five files reported `unclosed_fences=0`.
- `javap -p -c -s` confirmed exactly two package-private helper entries, no helper fields, and a
  private zero-argument constructor. Every uniform typed loop invokes<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument))</span>
  `RandomGenerator.nextDouble:(DD)D` once per [element](../../../../glossary.md#term-element), followed by direct [binary64](../../../../glossary.md#floating-point-precision-and-rounding) storage,
  `d2f`, or `d2f` plus `BFloat16Bits.fromFloat`, and then one matching
  `TensorFactory.fromFlatArray` call. Existing normal loops still invoke `nextGaussian`, use
  ordinary `dmul` then `dadd`, retain their prior conversions, and delegate once.
- `javap -p -c -s` on `TensorFactory` confirmed exactly the existing `randomNormal` and new
  `randomUniform` public random entries, public null-check order, direct helper delegation, and no
  random field. Source/import inspection confirmed only model-foundation and JDK imports, exactly
  the two existing allocator fields, and no default/static source, source factory/lookup, seed,
  registry, service, synchronization, stream, reflection, Operation, compiler, training, runtime,
  or backend behavior.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [seed glossary](../../../../glossary.md#term-seed); [registry glossary](../../../../glossary.md#term-registry); [synchronization glossary](../../../../glossary.md#term-synchronization); [Operation glossary](../../../../glossary.md#operation); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Final status and scope inspection found exactly the ten authorized 0012G paths. The separate
  preexisting preparatory changes to `capabilities.md` and task 0012F remained untouched by the
  implementation and documentation passes. No task 0012H, 0012I, or 0013 specification exists.

Documentation validation used clean context
`/root/implement_model_0012g/review_model_0012g_docs`. The selected primary profiles were [API](../../../../glossary.md#term-application-programming-interface-api) and
[Javadoc](../../../../glossary.md#term-javadoc) for `TensorFactory`, `TensorRandoms`, and the [Tensor](../../../../glossary.md#tensor) API, plus Planning for this task,
master plan, and roadmap; General style and Example format were also applied. The pass read the
[architecture contract](../../../../glossary.md#architecture-contract) and focused [model](../../../../glossary.md#term-model)-boundary explanations, inspected the final implementation
and all affected tests directly, and finalized bounds, binary64 [half-open](../../../../glossary.md#term-half-open-interval) semantics, [narrowing](../../../../glossary.md#term-narrowing),
source [ownership](../../../../glossary.md#term-ownership-meanings), threading, [reproducibility](../../../../glossary.md#artifact-manifest-and-source-provenance), call order/count, validation messages, allocation,
and ID effects in Javadocs and the public reference. It also made uniform creation current in the
existing Tensor and [Tensor factory](../../../../glossary.md#tensor-factory) glossary entries. No new glossary term was warranted because
continuous [uniform distribution](../../../../glossary.md#term-uniform-distribution) and caller-owned `RandomGenerator` are ordinary mathematical and
JDK vocabulary rather than reusable Synaptik domain concepts.

Existing `randomNormal`, `BFloat16Bits`, [`DataType`](../../../../glossary.md#data-type--datatype), [`Shape`](../../../../glossary.md#shape), `LayoutDescriptor`,
`TensorDescriptor`, `Tensor`, `HostTensorStorage`, and `MemorySegmentStorage` Javadocs remain
accurate without edits. Uniform creation reuses their existing [floating classification](../../../../glossary.md#floating-classification),
[binary32](../../../../glossary.md#floating-point-precision-and-rounding)-to-[BFLOAT16](../../../../glossary.md#term-bfloat16) conversion, static-count, dense-[layout](../../../../glossary.md#layout), [descriptor](../../../../glossary.md#term-descriptor) eligibility, Tensor
[identity](../../../../glossary.md#term-identity-meanings)/label, and [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) [heap](../../../../glossary.md#term-heap)-storage contracts without changing them. Focused architecture
documents, the capability baseline, architecture tests, [backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance) tests, and integration
tests require no update because this task implements the already planned model-owned [eager](../../../../glossary.md#term-eager-evaluation) [leaf](../../../../glossary.md#term-leaf)
initializer without changing [module](../../../../glossary.md#term-module) ownership, [dependency direction](../../../../glossary.md#java-platform-and-build-tools), [lifecycle](../../../../glossary.md#lifecycle), storage, backend,
or end-to-end execution behavior. Architecture impact is therefore None.

## Implementation notes

- Added exactly one public `TensorFactory.randomUniform(...)` entry and one package-private
  `TensorRandoms.randomUniform(...)` entry while preserving the completed normal API and bytecode
  behavior.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- Uniform validation constructs only a canonical dense descriptor for static Java-array-sized
  shapes and the three floating types, then samples one exact typed carrier with one bounded source
  call per row-major element and delegates once to matching flat import.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [canonical glossary](../../../../glossary.md#term-canonical-form); [descriptor glossary](../../../../glossary.md#term-descriptor); [shapes glossary](../../../../glossary.md#shape); [samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [carrier glossary](../../../../glossary.md#term-storage-carrier); [row-major glossary](../../../../glossary.md#term-row-major); [element glossary](../../../../glossary.md#term-element))</span>
- Focused tests cover API/helper shape, exact bounded arguments/order/count, all three conversions,
  the narrowed-upper-bound caveat, scalar and empty shapes, validation order/messages, label and
  identifier side effects, bounded reproducibility, non-retention, and source exceptions.<span data-glossary-navigation="true"> ([arguments glossary](../../../../glossary.md#term-argument); [scalar glossary](../../../../glossary.md#term-scalar); [reproducibility glossary](../../../../glossary.md#artifact-manifest-and-source-provenance))</span>
- The independent clean-context documentation pass finalized Javadocs, the Tensor API example,
  glossary impact, and synchronized planning status before completion.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [planning glossary](../../../../glossary.md#term-planning))</span>

## Completion summary

- Completed changes: one explicit caller-source bounded continuous-uniform factory method,
  stateless typed sampling/import helper extension, focused API and behavior tests, finalized
  Javadocs and Tensor API example, glossary status, and synchronized planning frontier.<span data-glossary-navigation="true"> ([sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Tensor glossary](../../../../glossary.md#tensor); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: `TensorFactory.java`, `TensorRandoms.java`, `TensorFactoryTest.java`,
  `TensorFactoryRandomTest.java`, `TensorFactoryUniformRandomTest.java`, `docs/api/tensor-api.md`,
  `docs/glossary.md`, this task, [model](../../../../glossary.md#term-model) master plan, and roadmap.
- Tests and validation: all three focused suites, all 225 model tests, model Javadoc, root tests,
  runnable example, bytecode/import/static-state checks, Markdown link/anchor/fence checks,
  ten-path scope and later-spec checks, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [bytecode glossary](../../../../glossary.md#term-bytecode); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` passed.
- Documentation-agent review: clean context
  `/root/implement_model_0012g/review_model_0012g_docs` completed the independent pass using API
  and Javadoc, Planning, General style, and Example format profiles.
- Documentation impact: Tensor API and existing glossary entries now describe implemented uniform
  sampling, binary64 half-open semantics, narrowing, source policy, and failure effects; focused
  architecture and capability documents remain accurate without task-specific edits.<span data-glossary-navigation="true"> ([binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [half-open glossary](../../../../glossary.md#term-half-open-interval); [narrowing glossary](../../../../glossary.md#term-narrowing))</span>
- Javadoc review: affected factory and helper contracts were finalized; related normal, data type,
  BFLOAT16, shape, layout, descriptor, Tensor, and storage contracts remain accurate unchanged.<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Glossary impact: existing implementation-status, Tensor, and Tensor factory language was
  updated; no new reusable project term was introduced.<span data-glossary-navigation="true"> ([Tensor factory glossary](../../../../glossary.md#tensor-factory))</span>
- Unresolved issues: None.
- Follow-up required: None for task 0012G. Subsequent planning promoted task 0012H to the Ready
  frontier with its own detailed specification.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>

Status: Complete
