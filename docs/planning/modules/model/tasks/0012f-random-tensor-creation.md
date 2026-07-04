# Task 0012F: Random Tensor Creation

## Status

Complete

## Goal

Add one explicit public factory method for creating a dense floating tensor from a normal
distribution. The caller supplies the exact `RandomGenerator`, distribution parameters, floating
`DataType`, shape, optional label, and gradient intent.

This completes the selected TensorFactory population baseline without hidden global randomness,
generator retention, a service locator, default metadata, or runtime/backend behavior.

## Scope

- Add one `TensorFactory.randomNormal(...)` method using a caller-supplied
  `java.util.random.RandomGenerator`.
- Support exactly `FLOAT64`, `FLOAT32`, and `BFLOAT16` output.
- Require finite mean, finite non-negative standard deviation, and a fully static shape whose
  logical count fits a Java array.
- Synthesize canonical dense-contiguous layout and preserve explicit label/gradient intent.
- Consume exactly one `RandomGenerator.nextGaussian()` call per logical element in row-major order.
- Calculate `mean + gaussian * standardDeviation` with ordinary binary64 multiplication followed
  by addition, without fused multiply-add substitution.
- Store binary64 directly for FLOAT64, narrow to binary32 for FLOAT32, and narrow to binary32 then
  use `BFloat16Bits.fromFloat(...)` for BFLOAT16.
- Create one exact typed source carrier and delegate once to matching flat import.
- Define source ownership, reproducibility, threading, failure, allocation, and ID side effects.
- Modify `TensorFactory`, add one package-private `TensorRandoms`, update only the factory API-shape
  test, and add one focused random test.
- During implementation, update Tensor API, glossary, task evidence, master plan, and roadmap
  through the required separate clean-context documentation pass.

## Out of scope

- factory-owned/static/global/default/thread-local randomness, `RandomGenerator.getDefault()`,
  registry, service, dependency injection, retained source, or lifecycle ownership
- seed-only, algorithm-name, `RandomGeneratorFactory`, split/reset/copy/close, source-creation, or
  synchronization APIs
- cross-algorithm, cross-provider, cross-Java-version, or concurrent-use reproducibility promises
- uniform, Bernoulli, integer, boolean, categorical, truncated/multivariate normal, permutation,
  random-range, or arbitrary-distribution creation
- `INT32`, `INT64`, or `BOOL` random output and their rounding/truthiness policies
- default mean, deviation, dtype, label, gradient flag, source, shape, or convenience overloads
- dynamic binding, descriptor/layout input, non-dense destinations, view/scatter population, or
  Tensor input
- typed access/export, backing-array exposure, mutation/version tracking, or zero-copy transfer
- provenance, random operations, graph/compiler/planning/prepare/runtime/backend/training behavior,
  execution, or ONNX mapping
- new storage, native/off-heap allocation, `Arena`, deterministic close, pooling, or residency
- dependencies, preview/incubator features, Gradle/architecture changes, or another module
- a detailed task-0013 or later specification

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Architecture overview](../../../../architecture/overview.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md), especially normal random population and the
  requirement to decide source/reproducibility without live model services
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
- [Tensor API](../../../../api/tensor-api.md) and [glossary](../../../../glossary.md)

## Legacy evidence and rejected coupling

Read-only inspection of legacy `TensorDataFactory.randn(...)` and Tensor conveniences confirms
eager normally distributed floating tensors parameterized by shape, mean, standard deviation,
data type, and label. Legacy sampled `mean + Random.nextGaussian() * stdDev`, rejected
non-floating output and invalid deviation, and created an unseeded generator internally.

The new design rejects hidden per-call nondeterminism, default source selection, nullable labels,
positive-dimension-only shapes, implicit double-array conversion, mutable metadata, and direct
Tensor construction. It retains the useful distribution through an explicit caller-owned source
and current descriptor/import paths.

## Architecture constraints

- Production remains in `io.github.pho001.synaptik.model.tensor`. Public API belongs to
  `TensorFactory`; sampling mechanics belong to one package-private helper beside it.
- Package direction remains toward existing datatype/shape/layout/storage foundations plus stable
  JDK `java.util.random`; no cross-module dependency is introduced.
- `RandomGenerator` is transient caller-owned method input. No factory, helper, Tensor, descriptor,
  storage, registry, or service retains it.
- The factory performs no source lookup, default selection, algorithm choice, registration,
  seeding, splitting, reset, close, or synchronization.
- Random creation is eager leaf-data construction, not an Operation, graph node, compiler pass,
  runtime generator, backend kernel, or training policy.
- Only floating types are accepted. Integral/BOOL output would require separate policy.
- Every result has explicit shape/type/gradient/label, canonical dense layout, fresh storage, and a
  fresh factory ID.
- Sampling completes in one exact typed source carrier before flat import; no partial Tensor is
  observable.
- Success delegates exactly once to matching flat import and does not directly construct
  Tensor/storage or allocate an ID.
- Stop if implementation needs a public random abstraction, stored/default source, another helper,
  storage-contract change, architecture change, dependency, or another module.

## Package impact

Existing packages used:

- `model.tensor` — public construction, descriptor synthesis, and eager leaf population.
- `model.datatype` — floating types and BFLOAT16 conversion.
- `model.shape` — fully static shapes.
- `model.layout` — canonical dense-contiguous geometry.
- `java.util.random` — caller-owned Java 26 `RandomGenerator` contract.

Packages added or changed:

- No package is added. Only the existing `model.tensor` package changes.

Type placement:

- `io.github.pho001.synaptik.model.tensor.TensorFactory` — one public normal-random method.
- `io.github.pho001.synaptik.model.tensor.TensorRandoms` — package-private descriptor validation,
  sampling, conversion, and flat-import dispatch.
- `io.github.pho001.synaptik.model.tensor.TensorFactoryRandomTest` — same-package focused source,
  conversion, reproducibility, validation, and side-effect tests.

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

Do not add `randn`, default/unlabeled, seed, factory, algorithm-name, `Random`-specific,
descriptor/layout, or other-distribution overloads.

### Public null validation

Validate before helper delegation in exact order:

1. null `shape`: `NullPointerException`, message `shape`;
2. null `dataType`: `NullPointerException`, message `dataType`;
3. null `randomGenerator`: `NullPointerException`, message `randomGenerator`;
4. null `label`: `NullPointerException`, message `label`.

These failures consume no random calls, allocation, or Tensor ID.

### Shape, type, distribution, and descriptor validation

The helper validates before carrier allocation or sampling:

1. Dynamic shape: `IllegalArgumentException` with
   `random tensor creation requires a fully static shape: <shape>`.
2. Read `shape.knownElementCount()`; checked multiplication overflow remains
   `ArithmeticException`.
3. Count above `Integer.MAX_VALUE`: `IllegalArgumentException` with
   `random tensor element count exceeds Java array limit: required=<required>, maximum=2147483647`.
4. Non-floating type: `IllegalArgumentException` with
   `random normal creation requires floating data type: <dataType>`.
5. Non-finite mean: `IllegalArgumentException` with
   `random normal mean must be finite: <mean>`.
6. Non-finite or numerically negative deviation: `IllegalArgumentException` with
   `random normal standard deviation must be finite and non-negative: <standardDeviation>`.
   Positive and negative zero are accepted.
7. Create `LayoutDescriptor.contiguous(shape)`.
8. Create `TensorDescriptor(dataType, shape, Optional.of(layout), requiresGrad)`.

Scalar shape consumes one sample. A fully static zero-element shape is valid, creates an empty
tensor, and consumes zero samples.

### Sampling and carrier conversion

After validation:

1. Allocate exactly one matching source array of checked logical count.
2. For every index, call the exact supplied `randomGenerator.nextGaussian()` once.
3. Evaluate `double sample = mean + gaussian * standardDeviation` with ordinary multiplication
   then addition. Do not use `Math.fma`, streams, parallelism, batching, or
   `nextGaussian(mean, standardDeviation)`.
4. Store exactly:

| Data type | Carrier | Stored value |
|---|---|---|
| `FLOAT64` | `double[]` | binary64 formula result |
| `FLOAT32` | `float[]` | Java binary64-to-binary32 narrowing |
| `BFLOAT16` | `short[]` | binary32 narrowing, then `BFloat16Bits.fromFloat(...)` |

5. Delegate once to matching `TensorFactory.fromFlatArray(...)` with synthesized descriptor and
   caller label.
6. Return that exact Tensor.

Generated overflow, underflow, signed zero, infinity, or NaN is not post-validated. A custom
generator may override normal behavior; Synaptik still applies the specified formula/conversion.
BFLOAT16 retains existing canonical-NaN conversion. Generator and carrier are not retained.

### Source ownership and reproducibility

- The caller creates, configures, seeds, owns, and advances the generator.
- TensorFactory never selects an algorithm/default, stores a seed, resets/splits/closes, or retains
  the source.
- Non-empty success advances it by exactly one `nextGaussian()` method call per output element in
  row-major order; empty success advances it zero times.
- Equivalent results require equivalent generator implementation and initial state, identical
  arguments, and no interfering source use.
- No sequence promise exists across algorithms, providers, Java versions, seed expansion, source
  states, or concurrent access.
- The method does not synchronize caller state. Callers provide exclusive or otherwise safe use.
- A newly seeded JDK generator may support repeatable tests, but Synaptik exposes no seed contract.

### Failure side effects

- Null, shape, count, type, distribution, layout, and descriptor failures occur before source
  carrier/sampling and consume no ID.
- Source-carrier `OutOfMemoryError` occurs before sampling and consumes neither draws nor ID.
- If `nextGaussian()` throws, completed earlier calls stay consumed; the throwing call's effect is
  generator-defined. No destination or ID exists.
- After all samples, flat import allocates destination and then ID.
- Destination `OutOfMemoryError` occurs after all draws but before ID.
- Blank label consumes all draws, both carriers, and one ID, then fails before flat copy.
- Exhaustion occurs after all draws and both carrier allocations, before copy; no rollback occurs.
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

It must have no fields/public/protected members/state/cache/registry/seed; one private zero-arg
constructor; no retained arguments; and private methods only for descriptor validation and typed
sampling/import. Use ordinary loops without streams, reflection, parallel work, per-element boxing,
or temporary `double[]` conversion for FLOAT32/BFLOAT16. Allocate one source carrier, delegate once
to flat import, and never directly construct Tensor/storage or allocate IDs.

## Valid and invalid scenarios

| Scenario | Result |
|---|---|
| FLOAT64 `[2,2]`, scripted source | Four transformed binary64 values in order |
| FLOAT32 | Binary64 formula, then one narrowing |
| BFLOAT16 | Binary64 formula, binary32 narrowing, BFLOAT16 conversion |
| Scalar | One source call and one value |
| `[2,0,3]` | Empty dense tensor and zero source calls |
| Same seeded implementation/state/arguments | Equivalent result within stated scope |
| Integral/BOOL type | Rejected before sampling |
| Dynamic or over-limit shape | Rejected before sampling |
| Non-finite mean | Rejected before sampling |
| Negative/non-finite deviation | Rejected before sampling |
| Signed-zero deviation | Accepted; one call per element still occurs |
| Blank label | All draws and one ID consumed before failure |
| Generator throws | No destination/ID; prior calls remain consumed |
| ID exhaustion | All draws consumed; no copy or rollback |

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorFactory.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorRandoms.java`

Tests:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryRandomTest.java`

Documentation/planning during implementation:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum scope

At most two production files, one existing and one new test, and the five documentation/planning
files above: nine paths total.

Do not modify other Java/tests, Gradle, `AGENTS.md`, `ARCHITECTURE.md`, focused architecture docs,
architecture tests, capabilities, another module, or unrelated docs. Do not create task 0013.
Stop if another file/public type/source policy/default generator/storage or architecture decision
is required.

## Javadoc requirements

- Update TensorFactory type Javadoc for explicit caller-owned normal generation.
- Fully document the public method: type/shape/count/layout, formula/conversions, finite rules,
  call order/count, ownership/non-retention/threading/reproducibility, label/gradient, result,
  failures, and source/allocation/ID side effects.
- Document helper, entry, and private methods to the same precision.
- Explain two-stage BFLOAT16 narrowing and absence of default source/seed/service.
- Do not claim other distributions, seeded Synaptik API, random Operations, access/export, or
  runtime/backend execution.
- Review existing related Javadocs and record why unchanged contracts remain accurate or stop on
  an out-of-scope discrepancy.

## Acceptance criteria

- Exactly one new public `randomNormal(...)`; no other public API.
- Existing fields, factory methods/helpers, allocator, behavior, and tests remain unchanged.
- Only FLOAT64/FLOAT32/BFLOAT16, static shape, finite mean, finite non-negative deviation, explicit
  source/label/gradient are accepted.
- Exactly one source call per non-empty logical element in order; empty output makes zero calls.
- Formula and three carrier conversions are exact.
- Source is used transiently and never stored/substituted/synchronized/seeded/reset/split/closed.
- One source carrier and one matching flat-import delegation; no direct Tensor/storage/ID work.
- Exact validation, messages, source/allocation/ID effects, source exceptions, blank label, and
  permanent exhaustion are tested.
- Scripted conversion and seeded repeatability are proven without a production hook.
- API/helper/bytecode/import/static-state/scope checks pass.
- Javadocs, Tensor API, glossary, task/master/roadmap, and separate documentation review complete.
- 0012F becomes Complete only after all validation; 0013 stays Draft without a spec.

## Tests / validation

Run before and after documentation review:

```bash
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryTest
./gradlew :modules:model:test --tests io.github.pho001.synaptik.model.tensor.TensorFactoryRandomTest
./gradlew :modules:model:test
./gradlew :modules:model:javadoc
./gradlew test
git diff --check
```

Focused tests cover helper/API shape, exact source use, scripted three-type conversion, scalar/
empty shapes, all nulls in order, dynamic/count failures, non-floating types, invalid parameters
and signed-zero acceptance, gradients, labels/descriptors, call count/order, equivalent seeded
sources, non-retention, source exception, safely observable side-effect order, blank label,
permanent exhaustion, and allocator restoration. Do not force unsafe OOME tests.

Manually verify `javap -p -c -s`, imports, validation order, ordinary multiply/add, conversions,
one flat-import delegation, no direct Tensor/storage/ID construction, no source lookup/streams/
reflection/synchronization/static RNG/seed/architecture-layer imports, exact nine-path scope,
documentation links/anchors/fences/whitespace/status, and no task-0013 spec.

## Dependencies

- 0012B supplies typed copied flat import.
- 0012 supplies identity allocation/public construction.
- 0001/0002/0003/0007/0010/0011 supply type, shape, layout, descriptor, storage, and Tensor.
- 0012D/0012E supply descriptor/carrier/side-effect precedents.

## Follow-up tasks

- Subsequent user-approved planning inserts 0012G uniform, 0012H integral, and 0012I Bernoulli
  initialization before provenance. These are independent follow-ups, not incomplete 0012F work.
- 0013 defines minimal Tensor provenance after the expanded initializer sequence.
- Later tasks own random Operation semantics, training initialization, typed access/export,
  mutation/versioning, native/runtime/backend generation, compiler capture, and execution.

Do not create task 0013 or later here.

## Architecture impact

Expected impact: None.

`RandomGenerator` is transient method input, not a retained service or locator. Work remains eager
copied model leaf data and changes no module boundary, lifecycle, ownership, graph/training policy,
or runtime/backend rule. Stop before architecture edits if implementation reveals otherwise.

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

- Caller-supplied `RandomGenerator` keeps algorithm/seeding/state explicit without model services.
- One method uses explicit floating `DataType`; samples originate in binary64 and conversion is an
  explicit requested result format.
- Both distribution parameters must be finite. A non-finite mean does not describe the selected
  finite-parameter normal distribution and is rejected rather than silently filling infinities or
  NaNs; signed-zero deviation remains a valid degenerate distribution.
- Reproducibility is bounded to equivalent source implementation/state and arguments; no seed-only
  or algorithm convenience implies a wider promise.
- One draw per element is consumed even for signed-zero deviation, keeping source advancement
  independent of constant-distribution optimization.
- Ordinary multiply/add matches selected legacy transformation; no mean/deviation overload or FMA.
- FLOAT32 narrows once; BFLOAT16 narrows to binary32 then uses established conversion.
- Sampling completes before import, avoiding public partial tensors while making late failure
  consumption explicit.

## Known limitations

- Only eager independent normal samples for three floating types.
- Static Java-array-sized shapes only; practical heap limits may be lower.
- No generator synchronization or atomic source snapshot.
- No reproducibility promise across algorithms/providers/Java versions/states/interference.
- Generated non-finite values are not post-validated.
- No seeded convenience, default source, other distribution, random Operation, access/export,
  training initializer, or runtime/backend random generation.

## Validation evidence

Planning read architecture/planning/documentation rules, capability baseline, completed model and
factory tasks, current helpers/tests, Java 26 `RandomGenerator`, and read-only legacy randn source
and Tensor conveniences. The plan introduced no architecture, dependency, package, build, or
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
  Compilation passed with no diagnostics. Running
  `java -cp modules/model/build/classes/java/main:/private/tmp/synaptik-normal-example NormalRandomExample`
  passed and printed exactly:

  ```text
  [-1.0, 1.0, 3.0, 2.0]
  4
  DENSE_CONTIGUOUS
  ```

  The temporary source was removed after validation and is not part of the task scope.
- A Ruby local Markdown target-and-heading-slug checker ran over `docs/api/tensor-api.md`,
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
  flat-import call per branch, one `nextGaussian` call in each loop, ordinary `dmul` then `dadd`,
  `d2f` for FLOAT32, and `d2f` followed by `BFloat16Bits.fromFloat` for BFLOAT16.
- Source/import/static-state inspection confirmed exact public null-check order, helper validation
  order and messages, no retained/default/static random source, seed/factory/source lookup,
  synchronization, streams, reflection, parallelism, direct Tensor/storage/ID creation, or
  compiler/runtime/backend/training imports.

Documentation validation used clean context
`/root/implement_model_0012f/review_model_0012f_docs`. The selected primary profiles were API and
Javadoc for `TensorFactory`, `TensorRandoms`, and the Tensor API, plus Planning for this task,
master plan, and roadmap; General style and Example format were also applied. The pass inspected
the implementation, focused and factory tests, generated Javadoc, related data type/BFLOAT16/
shape/layout/descriptor/Tensor/storage contracts, architecture explanations, model capability
baseline, and final nine-path diff. It finalized factory/helper Javadocs, added current normal-
random behavior and a scripted verified example to the Tensor API, and updated existing Tensor and
Tensor factory glossary entries. No new glossary term was warranted because caller-owned
`RandomGenerator` and normal distribution are ordinary Java/mathematical vocabulary rather than
new reusable Synaptik domain concepts. Local Markdown targets and changed anchors, example
fences, generated Javadoc content, terminology, trailing whitespace, synchronized status, exact
nine-path scope, and absence of a task-0013 specification were checked.

Existing `DataType`, `BFloat16Bits`, `Shape`, `LayoutDescriptor`, `TensorDescriptor`, `Tensor`,
`HostTensorStorage`, and `MemorySegmentStorage` Javadocs remain accurate without edits: this task
uses their existing floating classification, conversion, static-count, dense-layout, descriptor,
borrowed-storage, and identity contracts without changing them. Existing flat-import/allocation/
constant/population helper contracts also remain unchanged. Focused architecture documents and
the capability baseline require no edit because the work implements their existing model-owned
factory capability without changing ownership, dependency direction, lifecycle, or capability
scope. Architecture tests and backend/integration tests require no focused update because no
module boundary, backend behavior, or end-to-end execution behavior changed.

## Implementation notes

- Added the single public explicit-source method and package-private stateless sampling helper.
- Sampling validates all metadata before allocation, constructs canonical dense geometry, fills
  one exact requested carrier in row-major order, and delegates once to existing flat import.
- Tests cover public/helper API shape, exact conversion and call order, empty/scalar behavior,
  validation/messages, signed zero, bounded seeded equivalence, source failure, label side effects,
  and permanent identifier exhaustion with allocator restoration.
- Documentation was finalized independently in the required clean context before completion.

## Completion summary

- Completed changes: one explicit caller-source normal-random factory method, stateless typed
  sampling/import helper, focused API/behavior tests, finalized Javadocs and Tensor API example,
  glossary status, and synchronized planning frontier.
- Files changed or created: `TensorFactory.java`, `TensorRandoms.java`, `TensorFactoryTest.java`,
  `TensorFactoryRandomTest.java`, `docs/api/tensor-api.md`, `docs/glossary.md`, this task, model
  master plan, and roadmap.
- Tests and validation: both focused suites, all 214 model tests, model Javadoc, root tests,
  bytecode/source/manual contract checks, Markdown/example/generated-doc checks, scope/status
  checks, and `git diff --check` passed.
- Documentation-agent review: clean context
  `/root/implement_model_0012f/review_model_0012f_docs` completed the independent pass using API
  and Javadoc, Planning, General style, and Example format profiles.
- Documentation impact: Tensor API now documents implemented normal-random behavior, ownership,
  reproducibility, conversion, failures, and a complete scripted example; architecture/focused
  capability documents remain accurate without changes.
- Javadoc review: affected factory and helper contracts were finalized; related existing model
  contracts remain accurate because their behavior did not change.
- Glossary impact: updated existing implementation-status, Tensor, and Tensor factory language;
  no new reusable project term was introduced.
- Unresolved issues: None.
- Follow-up required: None for task 0012F. Subsequent planning places 0012G–0012I before task 0013.

Status: Complete
