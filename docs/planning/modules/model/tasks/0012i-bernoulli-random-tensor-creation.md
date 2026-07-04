# Task 0012I: Bernoulli Random Tensor Creation

## Status

Complete

## Goal

Add one explicit Bernoulli initializer that creates dense BOOL tensors from a caller-supplied
success probability and `RandomGenerator`. Each logical element independently compares one uniform
binary64 draw with the probability and stores canonical false/true bytes through existing flat
import.

This finishes the user-approved random initializer sequence without adding numeric truthiness,
gradient parameters, a new package, random services, or graph/runtime random behavior.

## Scope

- Add exactly one public `TensorFactory.randomBernoulli(...)` method.
- Accept one finite probability in the closed interval `[0.0, 1.0]`.
- Require fully static Java-array-sized shape and synthesize a canonical dense BOOL descriptor with
  `requiresGrad == false`.
- Invoke the exact supplied `RandomGenerator.nextDouble()` once per logical element in row-major
  order, including probability zero and one.
- Store byte `1` when `draw < probability`, otherwise byte `0`, then delegate once to BOOL flat
  import.
- Extend existing `TensorRandoms` with one package-private entry and typed BOOL loop.
- Update Factory/Normal/Uniform exact-surface tests only; leave integral behavior tests unchanged;
  add one Bernoulli-focused test.
- Finalize Javadocs, Tensor API, glossary, task evidence, master plan, and roadmap through the
  required independent documentation pass during implementation.

## Out of scope

- numeric Bernoulli output, caller-selected `DataType`, requires-grad input, numeric truthiness,
  output casting, probability tensors, per-element probabilities, logits, or broadcasting
- `nextBoolean()`, bounded `nextDouble(...)`, probability-specific sampling shortcuts, skipped
  calls at probability zero/one, vectorization, batching, streams, or parallel sampling
- default probability/source/label/shape, seed/algorithm overload, source lookup/factory/splitting/
  reset/synchronization/retention, registry, or service
- changing normal/uniform/integral signatures, method bodies, sampling, validation, or behavior
  assertions
- arbitrary distributions, random graph Operations, typed access/export, provenance, storage/
  native allocation, compiler/training/runtime/backend behavior, dependencies, build/architecture/
  package changes, another module, or detailed later specs

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
- [Task 0012G](0012g-uniform-random-tensor-creation.md), which defines binary64 uniform-source
  semantics
- [Task 0012H](0012h-integral-random-tensor-creation.md), which establishes the current helper
  surface and exact-surface regression scope
- [Tensor API](../../../../api/tensor-api.md) and [glossary](../../../../glossary.md)

## Capability origin

Bernoulli initialization is an explicit user-approved addition beyond the legacy minimum. It is an
eager BOOL leaf-data convenience, not evidence that legacy supported it and not a random graph
Operation, training dropout policy, or backend kernel.

## Architecture constraints

- Work remains in `io.github.pho001.synaptik.model.tensor`; no new random package or public
  distribution/source type.
- Public API belongs to `TensorFactory`; package-private sampling remains in existing
  `TensorRandoms`.
- Generator is transient caller-owned input and is never stored, substituted, seeded, split,
  reset, synchronized, closed, registered, or looked up.
- Result type is always BOOL and non-differentiable; no DataType or gradient parameter is exposed.
- Exactly one unbounded `nextDouble()` method invocation occurs per output element regardless of
  probability, so source advancement is independent of endpoint optimization.
- One complete canonical byte carrier precedes exactly one BOOL flat import; no partial Tensor or
  direct Tensor/storage/ID construction.
- Existing normal/uniform/integral behavior and bytecode must remain unchanged.
- Stop if implementation requires another production type/package, probability tensor/logit
  policy, source service, storage/dependency/architecture change, or another module.

## Package impact

Existing packages used:

- `model.tensor` — TensorFactory, TensorRandoms, descriptor synthesis, tests.
- `model.datatype`, `model.shape`, `model.layout` — BOOL, static shape, dense layout.
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

The result is BOOL with `requiresGrad == false`. Do not add DataType/gradient/default/unlabeled/
seed/factory/algorithm/probability-array/logit/alias overloads.

### Public null validation

Validate before helper delegation:

1. null `shape`: `NullPointerException`, message `shape`;
2. null `randomGenerator`: `NullPointerException`, message `randomGenerator`;
3. null `label`: `NullPointerException`, message `label`.

These failures consume no source call, allocation, or ID.

### Helper validation order

Before carrier allocation/sampling:

1. dynamic shape: `IllegalArgumentException`,
   `bernoulli random tensor creation requires a fully static shape: <shape>`;
2. read checked `knownElementCount()`; arithmetic overflow remains `ArithmeticException`;
3. count above `Integer.MAX_VALUE`: `IllegalArgumentException`,
   `bernoulli random tensor element count exceeds Java array limit: required=<required>, maximum=2147483647`;
4. non-finite or numerically outside `[0.0, 1.0]` probability: `IllegalArgumentException`,
   `bernoulli probability must be finite and in [0.0, 1.0]: <probability>`;
5. create canonical contiguous layout and BOOL TensorDescriptor with false gradient.

Positive and negative zero are accepted as probability zero. Probability one is accepted. Bounds
are validated for empty output. Scalar consumes one draw; empty output consumes zero.

### Sampling and BOOL representation

After validation:

1. Allocate exactly one `byte[]` of logical element count.
2. For each logical index call `double draw = randomGenerator.nextDouble()` exactly once.
3. Store `(byte) 1` if `draw < probability`; otherwise store `(byte) 0`.
4. Delegate once to `TensorFactory.fromFlatArray(descriptor, label, source)`.
5. Return that exact Tensor.

Do not use `nextBoolean`, `nextDouble(bound)`, `nextDouble(origin,bound)`, skip endpoint draws,
invert the comparison, use `<=`, perform numeric conversion, or post-process through public access.
Flat import receives already canonical bytes and preserves them through its BOOL normalization.

A conforming generator returns draws in `[0.0, 1.0)`. A custom non-conforming draw is not
post-validated; the specified strict comparison is still applied. Source and carrier are not
retained.

### Probability endpoint semantics

- At probability `0.0` or `-0.0`, every conforming draw produces false, but one draw is still
  consumed per element.
- At probability `1.0`, every conforming draw produces true, but one draw is still consumed per
  element.
- Strict `<` implements the mathematical Bernoulli success rule and avoids including a hypothetical
  draw equal to the probability.

### Ownership, reproducibility, and failure effects

- Reuse task 0012F policy: caller owns source; no management/synchronization; equivalent results
  require equivalent implementation/state/arguments without interference.
- Exactly one `nextDouble()` invocation per element is promised, not underlying bit consumption.
- Pre-sampling failures and source-carrier OOME consume no draws/IDs.
- Generator exception occurs before destination/ID; previous draws remain consumed.
- Destination OOME follows all draws but precedes ID.
- Blank label consumes draws and one ID, then fails before copy.
- Exhaustion follows draws/allocations, before copy, without rollback.
- Unexpected copy failure consumes ID. No state is rolled back.

### Existing helper extension

Add exactly one package-private `TensorRandoms.randomBernoulli(...)` entry matching the public
signature. The helper remains final/package-private/stateless/field-free with private constructor.
It exposes exactly five package-private entries afterward: normal, uniform, two integral overloads,
and Bernoulli.

Private validation sharing is allowed only if exact messages/order remain and existing method
bodies/bytecode are unchanged. No helper type, strategy, enum, registry, or production hook.

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

Documentation/planning during implementation:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- this task
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum scope

At most two production files, three existing exact-surface tests, one new test, and five docs/
planning files: eleven paths total. Existing random distribution/integral behavior assertions and
`TensorFactoryIntegralRandomTest` must remain unchanged. Do not modify capabilities, completed task
specs, other Java/tests, Gradle, agent/architecture docs/tests, another module, or unrelated files.
Do not create task 0013 or later specs. Stop beyond eleven paths or on architecture uncertainty.

The atomic scope is required because the factory and two existing helper-surface tests must accept
the fifth entry in the same buildable change.

## Javadoc requirements

- Update TensorFactory type Javadoc and document `randomBernoulli` fully: BOOL/non-gradient result,
  probability interval/endpoints, strict comparison, exact source call count, static dense shape,
  ownership/reproducibility/threading, label/result, failures, allocation/source/ID effects.
- Finalize helper entry/private loop; preserve all existing random Javadocs.
- Explain why endpoint draws are not skipped and why no DataType/gradient/package/source abstraction.
- Review related BOOL/type/shape/layout/descriptor/Tensor/storage/random Javadocs and record
  no-change conclusions or stop on out-of-scope discrepancy.

## Acceptance criteria

- Exactly one new public and one package-private `randomBernoulli`.
- Existing normal/uniform/integral implementation behavior unchanged; only exact surface assertions
  change in three existing tests.
- BOOL, false gradient, static Java-array-sized shape, finite `[0,1]` probability.
- One `nextDouble()` per element including endpoints; zero for empty output.
- Canonical byte result uses strict draw `<` probability; no nextBoolean/bounded draw/shortcut.
- Transient source, one byte carrier, one BOOL flat import, no direct Tensor/storage/ID work.
- Exact validation/messages/late effects and endpoint semantics tested.
- All five focused suites and aggregate validation pass.
- API/helper/bytecode/import/static-state/eleven-path checks pass.
- Independent documentation/Javadoc/glossary/status pass completes in the same change.
- 0012I becomes Complete only after validation; 0013 remains Draft without a spec.

## Tests / validation

Run before and after documentation review:

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

Focused tests cover exact API/helper surface; scripted draws around probability; strict comparison;
canonical bytes; probability 0/-0/1 with calls; scalar/empty; all nulls; dynamic/count/probability
failures/messages; descriptor/label; seeded equivalence; non-retention; generator exception; blank
label/exhaustion; allocator restoration; and absence of nextBoolean/bounded calls. Do not force OOME.

Manually inspect `javap -p -c -s`, imports, exact `nextDouble:()D`, `dcmpg`/strict branch and
canonical byte stores, one BOOL flat import, unchanged prior random bytecode, exactly five helper
entries, no RNG fields/services/forbidden layers, docs links/fences/whitespace/status, exact eleven
paths, and no task-0013 spec.

## Dependencies

- 0012F owns caller-source/reproducibility policy.
- 0012G defines uniform binary64 source semantics.
- 0012H establishes current helper surface and direct typed random extension practice.
- 0012B supplies canonical BOOL flat import; foundational model tasks remain complete.

## Follow-up tasks

- 0013: minimal Tensor provenance after the completed factory initializer sequence.
- Later work owns random graph Operations, per-element probability tensors, training dropout,
  access/export, compiler capture, and runtime/backend generation.

Do not create detailed follow-up specs here.

## Architecture impact

Expected impact: None. This is eager copied BOOL model leaf data using existing source/helper/import
contracts. No package/module/dependency/storage/lifecycle/training/runtime/backend change.

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

- One BOOL-only method avoids numeric truthiness, DataType, conversion, and gradient ambiguity.
- Probability is finite and closed `[0,1]`; signed zero is accepted as zero.
- One `nextDouble()` is always consumed per element, even at endpoints, for stable source
  advancement independent of optimization.
- Strict `<` is the Bernoulli success rule; canonical bytes are created before flat import.
- Existing helper/package remains sufficient; no distribution abstraction or random subpackage.
- Exact-surface tests require an eleven-path atomic change; behavior assertions remain unchanged.

## Known limitations

- Scalar probability only; BOOL output; static Java-array-sized shape.
- No per-element probabilities, logits, broadcasting, numeric output, or gradient.
- No source synchronization/cross-provider reproducibility/defaults/random Operation/dropout/runtime
  generation.
- Custom non-conforming draw results are compared directly without post-validation.

## Validation evidence

Planning reviewed architecture/planning rules, completed source/distribution policies, current
factory/helper exact-surface tests, JDK `nextDouble()` semantics, BOOL flat import, and the user-
approved Bernoulli capability. Eleven paths were required atomically; no architecture/package/
dependency/storage/build change was needed.

Implementation validation before the documentation pass established the requested API, sampling,
and side-effect behavior. The five focused suites passed together and separately: factory 7,
normal-random 9, uniform-random 11, integral-random 10, and Bernoulli-random 8 tests. One initial
parallel integral-suite invocation collided while Gradle packed the shared test-result cache; the
immediate sequential rerun passed, so this was a validation-invocation collision rather than a
product failure. The implementation-side aggregate model suite passed with 243 tests and zero
failures, errors, or skips; model Javadoc, root tests, and `git diff --check` also passed.

Independent documentation validation used clean context
`/root/implement_model_0012i/review_model_0012i_docs`. The pass applied General style plus API and
Javadoc style to `TensorFactory`, `TensorRandoms`, and the Tensor API; Planning style to this task,
the model master plan, and roadmap; and Example format to the new complete Bernoulli example. It
read the architecture contract and focused explanations, documentation/planning rules, model
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
  up-to-date.
- `git diff --check` — passed.

Manual source and `javap -p -c -s` inspection confirmed exactly one public and one package-private
`randomBernoulli` signature; exactly five package-private helper entries; no helper fields or RNG
state; one new `byte[]`; one unbounded `RandomGenerator.nextDouble:()D` per loop iteration;
`dcmpg` plus the strict branch; canonical `iconst_1`/`iconst_0` byte stores; and one
`TensorFactory.fromFlatArray(...,[B)` call. The descriptor is BOOL with false gradient intent.
Imports remain limited to existing model/JDK types, and source/diff inspection confirmed the prior
normal, uniform, and integral method bodies were not edited. No source lookup, bounded double,
`nextBoolean`, stream, reflection, synchronization, direct Tensor/storage/ID construction,
Operation, graph, compiler, training, runtime, backend, registry, or service behavior was added.

The complete Bernoulli example in `docs/api/tensor-api.md` was evaluated against the built model
classes with its scripted source and produced canonical bytes `[1, 1, 0, 0]`, type `BOOL`, gradient
flag `false`, and call count `4`. The host JShell then reported its known macOS preferences-history
flush failure after evaluation; this environmental tool-state limitation did not affect
compilation, execution, printed values, repository state, or Gradle validation. Two attempts to
reroute JShell preferences failed before evaluation because the platform could not instantiate or
link the alternate preferences implementation, so they supplied no product evidence. The focused
Bernoulli suite independently compiles and verifies the same call, carrier, descriptor, and output
contract.

A local Markdown target and heading-anchor checker passed for the five changed documentation/
planning files with 167 links checked, including 58 anchors, and zero failures. Preliminary
checker drafts failed before inspecting content because Ruby interpolated the heading-quantifier
syntax and the host Ruby lacks `Array#filter_map`; the compatible corrected invocation passed. A
separate fence check passed all five files, targeted trailing-whitespace search found no matches,
and terminology/status review found no stale planned
Bernoulli claim. Final scope review found exactly the eleven task-0012I paths: two production
files, three existing exact-surface tests, the new Bernoulli test, Tensor API, glossary, this task,
model master plan, and roadmap. The preexisting dirty task-0012H specification remained untouched
and excluded from that count. No task-0013 specification exists; 0013 remains Draft.

Existing `DataType`/BOOL, `Shape`, `LayoutDescriptor`, `TensorDescriptor`, `Tensor`,
`HostTensorStorage`, and `MemorySegmentStorage` Javadocs remain accurate without edits. Task 0012I
uses their existing boolean classification, static-count, dense-layout, false-gradient,
identity/label, canonical flat-import, and borrowed heap-storage contracts without changing them.
Existing normal, uniform, and integral Javadocs also remain accurate; only the factory/helper type
surface and additive Bernoulli contracts required revision. The Tensor API now treats Bernoulli
creation as current and documents probability, endpoint calls, strict comparison, canonical BOOL
storage, ownership, reproducibility, and failure effects.

The glossary's implementation status and existing Tensor/TensorFactory entries now include
Bernoulli creation. No new reusable glossary term was warranted: Bernoulli distribution,
probability, and `RandomGenerator` are ordinary mathematical/JDK vocabulary, while the behavior is
one factory operation rather than a new Synaptik domain concept. `ARCHITECTURE.md`, focused
architecture documentation, ADRs, architecture tests, backend-conformance tests, integration
tests, and `capabilities.md` remain unchanged because this task implements an already approved
model-owned eager leaf initializer and changes no module/package direction, dependency, lifecycle,
storage ownership, backend behavior, or end-to-end execution contract.

## Implementation notes

- Added exactly one public `TensorFactory.randomBernoulli(...)` method and one package-private
  `TensorRandoms.randomBernoulli(...)` entry while preserving completed normal, uniform, and
  integral behavior.
- Validation completes before sampling. The helper creates one canonical dense BOOL descriptor,
  consumes one unbounded draw per element even at probability endpoints, stores the strict
  comparison as canonical bytes, and delegates once to BOOL flat import.
- Focused tests cover exact public/helper surface, strict equality and non-conforming draws,
  endpoints including signed zero, canonical bytes, scalar/empty shapes, exact validation and
  messages, source ownership and failure, bounded reproducibility, labels, identifier effects,
  allocator restoration, and absence of bounded-double or boolean calls.
- The independent documentation pass finalized Javadocs, the Tensor API example, glossary impact,
  validation evidence, and synchronized planning status before completion.

## Completion summary

- Completed changes: One explicit caller-source Bernoulli factory method, stateless canonical BOOL
  sampling/import helper extension, focused API and behavior tests, finalized Javadocs and Tensor
  API example, glossary status, and synchronized planning frontier.
- Files changed or created: `TensorFactory.java`, `TensorRandoms.java`, `TensorFactoryTest.java`,
  `TensorFactoryRandomTest.java`, `TensorFactoryUniformRandomTest.java`,
  `TensorFactoryBernoulliRandomTest.java`, `docs/api/tensor-api.md`, `docs/glossary.md`, this task,
  model master plan, and roadmap.
- Tests and validation: All five focused suites, all 243 model tests, model Javadoc, root tests,
  bytecode/import/static-state checks, executable example, Markdown link/anchor/fence checks,
  exact eleven-path and later-spec checks, and `git diff --check` passed.
- Documentation-agent review: Clean context
  `/root/implement_model_0012i/review_model_0012i_docs` completed the independent pass using API
  and Javadoc, Planning, General style, and Example format profiles.
- Documentation impact: Tensor API and existing glossary entries now describe implemented
  Bernoulli sampling, finite closed probability, endpoint source calls, strict comparison,
  canonical BOOL storage, caller-source policy, and failure effects; architecture and capability
  documents remain accurate without task-specific edits.
- Javadoc review: Affected factory and helper contracts were finalized; related BOOL/data type,
  shape, layout, descriptor, Tensor, storage, normal, uniform, and integral contracts remain
  accurate unchanged for the reasons recorded above.
- Glossary impact: Existing implementation-status, Tensor, and TensorFactory language was updated;
  no new reusable project term was introduced.
- Architecture impact: None. No architecture document, ADR, architecture test, dependency rule,
  module boundary, package direction, or capability-baseline decision changed.
- Unresolved issues: None.
- Follow-up required: None for task 0012I. Task 0013 is the next separate Draft planning frontier
  without a detailed specification.

Status: Complete
