# Task 0012H: Integral Random Tensor Creation

## Status

Complete

## Goal

Add exact typed bounded random initialization for `INT32` and `INT64` tensors. Two overloads infer
the result data type from primitive bounds, use the caller-owned `RandomGenerator`, and delegate
complete carriers to existing flat import.

The task extends the established random helper without data-type arguments, modulo reduction,
floating conversion, gradient parameters, a new package, or changes to completed normal/uniform
behavior.

## Scope

- Add one `randomInt(...)` overload with `int` bounds producing `INT32`.
- Add one `randomInt(...)` overload with `long` bounds producing `INT64`.
- Use inclusive origin and exclusive bound with `originInclusive < boundExclusive`.
- Invoke the exact bounded JDK `nextInt(origin, bound)` or `nextLong(origin, bound)` once per logical
  element in row-major order.
- Require fully static Java-array-sized shape and synthesize canonical dense layout with
  `requiresGrad == false`.
- Preserve task 0012F source ownership/reproducibility/failure policy.
- Extend existing `TensorRandoms` with exactly two package-private overloads.
- Update the factory public API-shape test and only exact helper-surface assertions in both existing
  random focused tests; add one integral-random focused test.
- Finalize Javadocs, Tensor API, glossary, task evidence, master plan, and roadmap through the
  independent documentation pass during implementation.

## Out of scope

- floating or BOOL output, caller-selected `DataType`, requires-grad input, rounding, narrowing,
  widening, modulo reduction, truthiness, or conversion
- unbounded `nextInt()`/`nextLong()`, full-domain convenience, inclusive upper bound, equal/reversed
  bounds, automatic bound swapping, default origin/bound, or default labels
- default/global/thread-local generator, seed/algorithm overload, source factory, splitting/reset,
  synchronization, retention, registry, or service
- changing normal/uniform signatures, sampling, conversion, validation, bytecode, or behavior tests
- arbitrary distributions, Bernoulli behavior from 0012I, random ranges as graph Operations, typed
  access/export, provenance, storage/native allocation, compiler/training/runtime/backend behavior
- dependencies, Gradle/architecture/package changes, another module, or detailed later specs

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
- [Model capability baseline](../capabilities.md)
- [Model master plan](../master-plan.md)
- [Task 0001](0001-data-type-model.md)
- [Task 0002](0002-shape-and-dimension-model.md)
- [Task 0003](0003-layout-descriptor-model.md)
- [Task 0007](0007-tensor-descriptor-model.md)
- [Task 0012](0012-tensor-factory.md)
- [Task 0012B](0012b-flat-typed-tensor-import.md)
- [Task 0012F](0012f-random-tensor-creation.md), which owns caller-source policy
- [Task 0012G](0012g-uniform-random-tensor-creation.md), which establishes helper-extension and
  exact-surface regression precedent
- [Tensor API](../../../../api/tensor-api.md) and [glossary](../../../../glossary.md)

## Capability origin

Bounded integral random initialization is an explicit user-approved addition beyond the legacy
minimum. It does not claim a matching legacy factory implementation and does not introduce a
random graph Operation. The capability follows Java's bounded integer generator contracts and the
project's completed eager leaf-data factory architecture.

## Architecture constraints

- Work remains in `io.github.pho001.synaptik.model.tensor`; no `random`/`randoms` package or public
  distribution abstraction is added.
- Public overloads belong to `TensorFactory`; mechanics remain package-private in `TensorRandoms`.
- Source is caller-owned transient input and is never stored, substituted, seeded, split, reset,
  synchronized, closed, registered, or looked up.
- Primitive overload selects exact type: `int` bounds -> INT32, `long` bounds -> INT64.
- Integral descriptors are always non-differentiable and expose no meaningless `requiresGrad`.
- Bounded JDK generator methods provide uniform bounded sampling without project-owned modulo
  arithmetic. Synaptik counts one bounded method invocation per output; underlying source-bit
  consumption is generator implementation detail.
- Complete typed carrier precedes exactly one matching flat import; no partial Tensor or direct
  Tensor/storage/ID construction.
- Existing normal and uniform APIs/helper logic/tests remain behaviorally unchanged.
- Stop if implementation needs another production type/package, public source/distribution type,
  custom unbiased algorithm, conversion, storage/dependency/architecture change, or another module.

## Package impact

Existing packages used:

- `model.tensor` — TensorFactory, existing TensorRandoms, descriptor synthesis, tests.
- `model.datatype`, `model.shape`, `model.layout` — integral types, static shape, dense geometry.
- `java.util.random` — caller-owned bounded source methods.

Packages added or changed:

- No package is added. Only `model.tensor` changes.

Type placement:

- `TensorFactory` receives two public `randomInt` overloads.
- Existing package-private `TensorRandoms` receives two matching overloads and private typed loops.
- `TensorFactoryIntegralRandomTest` is the new focused same-package behavior test.
- Existing Factory/Random/Uniform tests change only exact public/helper API-shape assertions.

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

The first returns INT32; the second returns INT64. Both use `requiresGrad == false`. Do not add a
`DataType`, gradient, default-bound, unlabeled, seed/factory, `randint`, or `randomLong` API.

### Public null validation

Each overload validates before helper delegation:

1. null `shape`: `NullPointerException`, message `shape`;
2. null `randomGenerator`: `NullPointerException`, message `randomGenerator`;
3. null `label`: `NullPointerException`, message `label`.

These failures consume no source call, allocation, or ID.

### Helper validation order

For both overloads, validate before carrier allocation/sampling:

1. dynamic shape: `IllegalArgumentException`,
   `integral random tensor creation requires a fully static shape: <shape>`;
2. read checked `shape.knownElementCount()`; overflow remains `ArithmeticException`;
3. count above `Integer.MAX_VALUE`: `IllegalArgumentException`,
   `integral random tensor element count exceeds Java array limit: required=<required>, maximum=2147483647`;
4. `originInclusive >= boundExclusive`: `IllegalArgumentException`,
   `integral random origin must be less than bound: origin=<origin>, bound=<bound>`;
5. construct canonical contiguous layout and exact INT32/INT64 TensorDescriptor with false gradient.

Bounds are validated even for zero-element output. Scalar consumes one bounded call; empty output
consumes zero.

### Sampling

After validation:

- INT32 allocates exactly one `int[]`; each element is the direct result of one
  `randomGenerator.nextInt(originInclusive, boundExclusive)` invocation.
- INT64 allocates exactly one `long[]`; each element is the direct result of one
  `randomGenerator.nextLong(originInclusive, boundExclusive)` invocation.
- Preserve encounter order, then call exactly one matching `TensorFactory.fromFlatArray(...)`.
- Do not use unbounded draws, `%`, `floorMod`, floating multiplication, rejection logic, streams,
  batching, parallelism, boxing, or intermediate alternate carriers.

The half-open guarantee comes from a conforming `RandomGenerator`. Custom non-conforming results
are not post-validated. Neither source nor carrier is retained.

### Bound-domain limitation

Bounds use the result carrier itself. Therefore the largest representable exclusive bound is
`Integer.MAX_VALUE` or `Long.MAX_VALUE`, and that value is never emitted. A single overload cannot
express the mathematical exclusive bound one greater than the carrier maximum; no unbounded/full-
domain alternative is added in this task. Negative and mixed-sign valid intervals are supported.

### Ownership, reproducibility, and failure effects

- Reuse task 0012F policy: caller owns source, no source management/synchronization, and equivalent
  results require equivalent implementation/state/arguments without interference.
- One bounded method invocation per element is promised; internal random-bit consumption is not.
- Pre-sampling failures and source-carrier OOME consume no calls/IDs.
- Generator exception propagates before destination/ID; previous calls remain consumed.
- Destination OOME follows all calls but precedes ID.
- Blank label consumes calls and one ID, then fails before copy.
- Exhaustion follows all calls/allocations and does not roll back.
- Unexpected copy failure consumes ID. No state is rolled back.

### Existing helper extension

Add exactly two package-private `TensorRandoms.randomInt(...)` overloads matching public signatures.
The helper remains final, package-private, stateless, field-free, and privately constructible.
It will expose exactly four package-private entries after this task: normal, uniform, INT32 random,
and INT64 random.

Private shared validation is allowed only if exact distribution-specific messages/order remain and
normal/uniform bytecode behavior is unchanged. No new helper type or production hook.

## Affected files

Production:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorFactory.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorRandoms.java`

Existing tests, API-shape changes only:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryRandomTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryUniformRandomTest.java`

New focused test:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryIntegralRandomTest.java`

Documentation/planning during implementation:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum scope

At most two production files, three existing API/helper-shape tests, one new focused test, and five
documentation/planning files: eleven paths total. Existing random/normal/uniform behavior assertions
must not be weakened or changed. Do not modify capabilities, completed task specs, other Java/tests,
Gradle, agent/architecture docs/tests, another module, or unrelated files. Do not create detailed
0012I/0013 specs. Stop beyond eleven paths or on architecture uncertainty.

The eleven-path atomic scope is intentional: the existing public and both package-private exact-
surface regression tests must recognize the two new overloads. Splitting those expectation updates
from the API addition would leave the repository failing between tasks.

## Javadoc requirements

- Update TensorFactory type Javadoc and document both overloads fully: inferred type, half-open
  bounds, static shape, source ownership/reproducibility/call count, no modulo bias, carrier-domain
  limitation, label/result, failures, allocation, source, and ID effects.
- Finalize both helper entries/private loops and preserve existing normal/uniform Javadocs.
- Explain why there is no DataType/gradient/default/full-domain API and no new package.
- Review related type/shape/layout/descriptor/Tensor/storage/random Javadocs and record reasoned
  no-change conclusions or stop on out-of-scope discrepancy.

## Acceptance criteria

- Exactly two new public and two matching package-private `randomInt` overloads.
- Existing normal/uniform public/helper behavior and tests unchanged except exact surface sets/counts.
- int -> INT32, long -> INT64, static shape, strict valid bounds, false gradient.
- Exactly one matching bounded generator method invocation per element; zero for empty output.
- No modulo/unbounded/floating sampling or conversion.
- Source transient; one carrier; one flat import; no direct Tensor/storage/ID work.
- Exact messages/order/late effects and carrier limits documented/tested.
- Factory/normal/uniform/integral focused suites and aggregate validation pass.
- API/helper/bytecode/import/static-state/eleven-path checks pass.
- Independent documentation/Javadoc/glossary/status pass completes in same change.
- 0012H becomes Complete only after validation; 0012I stays Draft without a spec.

## Tests / validation

Run before and after documentation review:

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

Focused tests cover exact API/helper shape; INT32/INT64 direct scripted values and exact bounded
arguments/order/count; negative/mixed ranges; primitive boundary-adjacent ranges; scalar/empty;
all nulls; dynamic/count/bound validation/messages; dense descriptors/labels/false gradient;
seeded equivalence; non-retention; source exceptions; blank label/exhaustion; allocator restoration;
and absence of modulo/unbounded calls. Do not force OOME.

Manually inspect `javap -p -c -s`, imports, exact `(II)I` and `(JJ)J` calls, no arithmetic/modulo,
one flat import per overload, unchanged normal/uniform bytecode, exactly four helper entries, no RNG
fields/services/forbidden layers, docs links/fences/whitespace/status, exact eleven paths, and no
later detailed specs.

## Dependencies

- 0012F owns caller-source/reproducibility policy.
- 0012G establishes safe helper extension and exact-surface update practice.
- 0012B supplies typed flat import; foundational model tasks remain complete.

## Follow-up tasks

- 0012I: Bernoulli BOOL initialization with explicit probability.
- 0013: minimal Tensor provenance after initialization sequence.

Do not create detailed follow-up specs here.

## Architecture impact

Expected impact: None. This is eager copied integral model leaf data using existing source/helper/
import contracts. No package, module, dependency, storage, lifecycle, or runtime/backend change.

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

- Two primitive-bound overloads infer exact result type and avoid DataType/conversion ambiguity.
- `randomInt` is used as the Java-facing bounded-integer name for both int and long overloads,
  matching the project's overloaded `range` precedent.
- Bounded JDK methods provide unbiased interval sampling; Synaptik adds no modulo implementation.
- No gradient parameter is exposed because integral descriptors cannot require gradients.
- Carrier-maximum-inclusive full-domain sampling is not representable with same-carrier exclusive
  bounds and remains unsupported rather than adding unbounded/default APIs.
- Existing random exact-surface tests require an eleven-path atomic change; behavior assertions stay.

## Known limitations

- Only bounded INT32/INT64 output, static Java-array-sized shapes, and strict non-empty intervals.
- The exclusive bound cannot be mathematically one above carrier maximum.
- One bounded method invocation is promised, not underlying random-bit consumption.
- No source synchronization/cross-provider reproducibility/defaults/full-domain/Bernoulli/random
  Operation/access/runtime generation.

## Validation evidence

Planning reviewed architecture/planning rules, completed source policy, current public/helper exact-
surface tests, JDK bounded int/long methods, and the user-approved capability. The 11-path scope is
required to keep exact API regression tests passing atomically; no architecture/package/dependency/
storage/build change is required.

Implementation validation before the documentation pass established the requested API, sampling,
and side-effect behavior. The independent documentation-focused pass then inspected the actual
source, tests, and diff and reran the complete validation matrix after finalizing Javadocs and the
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
  preceding pass executed the rebuilt model JAR.
- `git diff --check` passed after the documentation edits.

The complete bounded-integral example in `docs/api/tensor-api.md` was evaluated against the built
model classes with its scripted source. It produced the documented values `[-3, -1, 0, 4]`, type
`INT32`, gradient flag `false`, and call count `4`; the focused integral test independently
compiles and verifies the same direct bounded-call, descriptor, carrier, and order contract. The
interactive evaluator reported an environment preferences-history persistence error only after
printing `Goodbye`; that post-evaluation tool-state limitation did not affect compilation,
execution, output, repository state, or the passing Gradle validation.

A Ruby local Markdown target-and-heading checker ran over the Tensor API, glossary, this task,
model master plan, and roadmap. It checked 163 local links, including 58 anchors, with 0 failures.
A separate fence check over the same five files reported 0 failures. `git diff --check` confirmed
no whitespace errors in tracked diffs, and a targeted trailing-whitespace scan of both new files
also passed.

`javap -p -s` confirmed exactly two public and two matching package-private `randomInt` overloads,
and exactly four package-private `TensorRandoms` entries. The helper remains final, package-private,
field-free, and privately constructible. `javap -p -c` confirmed the INT32 loop contains one
`RandomGenerator.nextInt:(II)I`, direct `iastore`, and one `fromFlatArray(...,[I)` call; the INT64
loop contains one `nextLong:(JJ)J`, direct `lastore`, and one `fromFlatArray(...,[J)` call. Neither
loop contains arithmetic, modulo, conversion, an unbounded draw, or an alternate carrier. The
normal and uniform loops retain their prior bounded calls, conversions, and single-import bodies;
their production bodies were not edited.

Source/import/static-state inspection confirmed exact public null-check and helper validation
order/messages, inferred INT32/INT64 descriptors with false gradient intent, one exact carrier,
one matching flat import, and no direct Tensor/storage/ID construction in the helper. It found no
stored/default generator, random service/factory/registry, seed handling, synchronization, stream,
reflection, custom modulo/rejection algorithm, Operation, compiler, planning, prepare, runtime,
backend, or training dependency. Final status inspection found exactly the eleven authorized
0012H paths after excluding the preexisting dirty task-0012G specification. No detailed 0012I or
0013 specification exists.

Documentation validation used the separate clean context
`/root/implement_model_0012h/review_model_0012h_docs`. The primary profiles were API and Javadoc
for `TensorFactory`, `TensorRandoms`, and the Tensor API, plus Planning for this task, the model
master plan, and roadmap; General style and Example format were also applied. The pass finalized
the public/helper validation, carrier-domain, ownership, reproducibility, call-count, allocation,
source, and identifier contracts; added the current bounded-integral API and focused example; and
updated the existing Tensor and Tensor factory glossary status. No new glossary entry was
warranted because half-open interval, bounded integer sampling, and `RandomGenerator` are ordinary
mathematical/JDK vocabulary rather than reusable Synaptik domain terms.

Existing `DataType`, `Shape`, `LayoutDescriptor`, `TensorDescriptor`, `Tensor`,
`HostTensorStorage`, and `MemorySegmentStorage` Javadocs remain accurate without edits. Task 0012H
uses their existing integral classification, static-count, dense-layout, false-gradient,
identity/label, and borrowed heap-storage contracts without changing them. Existing normal and
uniform Javadocs remain accurate and their method bodies are untouched. Focused architecture docs,
the capability baseline, architecture tests, backend conformance tests, and integration tests need
no update because the task implements the already planned model-owned eager leaf initializer
without changing module ownership, dependency direction, lifecycle, storage, backend behavior, or
end-to-end execution. Architecture impact is None.

## Implementation notes

- Added exactly two public `TensorFactory.randomInt(...)` overloads and two matching
  package-private `TensorRandoms.randomInt(...)` entries while preserving completed normal and
  uniform behavior.
- Primitive bounds infer exact result type. Validation completes before allocation; each typed
  loop stores one direct bounded JDK result per row-major element in one carrier and delegates once
  to flat import.
- Focused tests cover exact API/helper shape, int/long arguments and direct values, scalar/empty and
  signed/boundary-adjacent intervals, validation order/messages, source ownership and failure,
  labels, descriptor facts, bounded reproducibility, late identifier effects, and allocator
  restoration.
- The independent documentation pass finalized Javadocs, the Tensor API example, glossary impact,
  validation evidence, and planning status before completion.

## Completion summary

- Completed changes: two explicit caller-source bounded-integral factory overloads, stateless exact
  typed sampling/import helper extensions, focused API and behavior tests, finalized Javadocs and
  Tensor API example, glossary status, and synchronized planning frontier.
- Files changed or created: `TensorFactory.java`, `TensorRandoms.java`, `TensorFactoryTest.java`,
  `TensorFactoryRandomTest.java`, `TensorFactoryUniformRandomTest.java`,
  `TensorFactoryIntegralRandomTest.java`, `docs/api/tensor-api.md`, `docs/glossary.md`, this task,
  model master plan, and roadmap.
- Tests and validation: all four focused suites, all 235 model tests, model Javadoc, root tests,
  bytecode/import/static-state checks, Markdown link/anchor/fence checks, exact eleven-path and
  later-spec checks, and `git diff --check` passed. Scripted example evaluation produced the exact
  documented output; the recorded preferences-history limitation occurred only after evaluation.
- Documentation-agent review: clean context
  `/root/implement_model_0012h/review_model_0012h_docs` completed the independent pass using API
  and Javadoc, Planning, General style, and Example format profiles.
- Documentation impact: Tensor API and existing glossary entries now describe implemented bounded
  integral sampling, inferred type, strict half-open bounds, direct bounded calls, carrier-domain
  limitation, source policy, and failure effects; focused architecture and capability documents
  remain accurate without task-specific edits.
- Javadoc review: affected factory and helper contracts were finalized; related data type, shape,
  layout, descriptor, Tensor, storage, normal, and uniform contracts remain accurate unchanged.
- Glossary impact: updated existing implementation-status, Tensor, and Tensor factory language; no
  new reusable project term was introduced.
- Architecture impact: None.
- Unresolved issues: None.
- Follow-up required: None for task 0012H. Draft task 0012I is the next active planning frontier;
  its detailed specification remains intentionally uncreated.

Status: Complete
