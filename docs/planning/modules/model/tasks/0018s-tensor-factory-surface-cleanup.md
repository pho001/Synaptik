# Task 0018S: Tensor Factory Surface Cleanup

## Status

Complete

## Goal

Make the eager tensor-construction API small enough to remain understandable as the model grows.
`TensorFactory` will keep model identity allocation, descriptor-based construction, heap
allocation, copied imports, constants, and integer ranges. Explicit-source random sampling will
move to the already cohesive `TensorRandoms` owner, and strict/cyclic prefix population will leave
production code because it is fixture preparation rather than a foundational tensor capability.

The intended ownership after this task is:

```text
TensorFactory
  identity + descriptor construction + allocation + import + constants + integer ranges

TensorRandoms
  explicit caller-owned RandomGenerator + distribution parameters -> eager leaf Tensor

TensorTestData (test source only)
  strict/cyclic prefix preparation -> TensorFactory.fromFlatArray(...)
```

This is an intentional pre-stabilization public API cleanup. It does not retain deprecated or
transitional aliases.

## Scope

- Remove all twelve public `TensorFactory.fromStrictFlatPrefix(...)` and
  `TensorFactory.fromCyclicFlatPrefix(...)` overloads.
- Remove strict/cyclic prefix implementation from production source.
- Preserve the useful prefix-fixture behavior in one package-private test-source
  `TensorTestData` helper so repository tests can prepare repeated or truncated typed carriers
  without making that behavior a production contract.
- Remove the five public random entries from `TensorFactory`:
  `randomNormal`, `randomUniform`, the two `randomInt` overloads, and `randomBernoulli`.
- Promote the existing stateless `TensorRandoms` class to the focused public random-initialization
  owner and make those same five signatures public without changing names or semantics.
- Preserve every random distribution, validation order and message, caller-owned-generator rule,
  source-call count, conversion rule, allocation order, label behavior, and tensor-ID side effect.
- Preserve both public integer `TensorFactory.range(...)` overloads and move only their
  package-private mechanics from broad `TensorPopulations` to narrow `TensorRanges`.
- Delete production `TensorPopulations` after its range and prefix responsibilities have moved to
  their selected owners.
- Keep every existing `TensorFactory` construction, allocation, import, scalar, zero/one,
  full-value, identity-matrix, derived-output, and ID-allocation contract unchanged.
- Do not add `ScalarValue` overloads to eager tensor construction. `ScalarValue` remains semantic
  operation state; the existing exact primitive scalar/full APIs remain the eager storage API.
- Update the exact public-surface tests, focused behavior tests, Tensor API, glossary, capability
  baseline, task status, model master plan, and roadmap.
- Close the recorded public-surface cleanup checkpoint after all executable and documentation
  validation passes.

## Out of scope

- a default, global, thread-local, engine-owned, runtime-owned, or service-located random source
- seed-only random overloads, generator selection, source splitting, source synchronization,
  source retention, source serialization, or cross-generator reproducibility
- graph RNG state, random `Operation` kinds, dropout, state-producing tensors, compiler capture,
  or task-0019B behavior
- new distributions, changed distribution formulas, changed parameter types, changed output types,
  or renamed random methods
- production strict-prefix, cyclic-prefix, repeat, tile, fill, fixture, dataset, or test-data APIs
- compatibility aliases, deprecated bridges, forwarding methods, or reflective adapters for any
  removed `TensorFactory` method
- removal, renaming, or semantic changes to integer `TensorFactory.range(...)`
- changes to flat or nested import, heap allocation, constants, `identityMatrix`, `eye`, ID
  allocation, provenance, `TensorProducer`, or derived-output construction
- `ScalarValue`-to-Tensor conversion, generic scalar/full overloads, boxed values, implicit
  conversion, or new data types
- typed tensor read/export APIs, zero-copy import, backing-array exposure, storage mutation,
  native storage, deterministic lifetime, or ownership changes
- operation semantics, graph state, gradients, autograd, compiler, planning, prepare, runtime,
  engine, backend, ONNX, or training behavior
- dependencies, Gradle changes, preview/incubator features, architecture changes, another module,
  or another production package
- a detailed specification for task 0018T or any later task

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially `modules/model` ownership
  of public tensor construction and the prohibition on runtime/backend state
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md), especially the prefix-population and large
  factory-surface decisions
- [Model master plan](../master-plan.md)
- [Task 0012](0012-tensor-factory.md) and [tasks 0012A–0012I](../master-plan.md#task-list), which
  established the current construction, population, and random behavior
- [Task 0013A](0013a-full-value-and-identity-matrix-tensor-creation.md), which established the
  current full-value and identity contracts
- [Task 0018N](0018n-typed-scalar-value-contract.md), which deliberately kept eager primitive
  scalar/full APIs separate from semantic `ScalarValue`
- [Tensor API](../../../../api/tensor-api.md) and [glossary](../../../../glossary.md)

## Architecture constraints

- All production work remains in `modules/model` and the existing
  `io.github.pho001.synaptik.model.tensor` package.
- `TensorFactory` remains the sole model identity allocator and the package-private construction
  seam used by current expression helpers. Moving public random entry points must not duplicate,
  expose, reset, or relocate its allocator.
- Random results remain eager provenance-free leaf tensors. `TensorRandoms` must build a validated
  dense descriptor and delegate once to the matching public `TensorFactory.fromFlatArray(...)`;
  it must not construct `Tensor`, storage, provenance, or identifiers directly.
- `TensorRandoms` is a stateless public namespace, not a random source, registry, service,
  distribution hierarchy, or lifecycle owner. It retains no generator or result.
- The exact caller-owned `RandomGenerator` remains mandatory for every random method.
- Prefix behavior becomes test code only. No production source or generated production Javadoc
  may expose strict/cyclic prefix creation after this task.
- `TensorTestData` may call only public model construction APIs. Production code must not import or
  otherwise depend on test source.
- Integer range remains eager model-owned leaf-data construction and continues to use
  `TensorFactory` identity, flat import, and exact INT32/INT64 carrier rules.
- Removed public APIs receive no alias or deprecation bridge. The repository has not stabilized a
  released compatibility contract for these provisional methods.
- If implementation requires a new production package, a random-source abstraction, a generic
  initializer hierarchy, storage changes, another module, or an architecture/dependency change,
  stop and report the conflict.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — owns public eager Tensor construction, random leaf
  initialization, package-private mechanics, and same-package tests.
- `io.github.pho001.synaptik.model.datatype` — supplies exact output data types and BFLOAT16
  conversion.
- `io.github.pho001.synaptik.model.shape` — supplies fully static eager result shapes.
- `io.github.pho001.synaptik.model.layout` — supplies canonical dense-contiguous descriptors.

Packages added or changed:

- No package is added. The existing `model.tensor` package remains the cohesive boundary.

Type placement:

- `io.github.pho001.synaptik.model.tensor.TensorFactory` — narrowed public identity,
  construction, allocation, import, constant, and integer-range boundary.
- `io.github.pho001.synaptik.model.tensor.TensorRandoms` — existing helper promoted to a public
  final stateless owner for the five explicit-source random methods.
- `io.github.pho001.synaptik.model.tensor.TensorRanges` — new package-private final production
  helper containing only the two integer-range entries and their private sizing/descriptor logic.
- `io.github.pho001.synaptik.model.tensor.TensorPopulations` — removed because its broad mixed
  production responsibility no longer exists.
- `io.github.pho001.synaptik.model.tensor.TensorTestData` — package-private test-source helper for
  strict/cyclic prefix fixture preparation; it is not part of the production artifact.

## Required contract

### Final `TensorFactory` boundary

`TensorFactory` remains a public final non-record static utility with exactly its existing two
private identity atomics and private zero-argument constructor. Its public methods after cleanup
are exactly these categories:

| Category | Public entries |
|---|---:|
| Descriptor construction | two `create` overloads |
| Resolved heap allocation | two `allocate` overloads |
| Exact flat import | six `fromFlatArray` overloads |
| Rectangular nested import | one `fromNestedArray` method |
| Typed scalar constants | five primitive `scalar` overloads plus `scalarBFloat16` |
| Zero/one constants | `zeros`, `ones`, `zerosLike`, `onesLike` |
| Typed full constants | five primitive `full` overloads plus `fullBFloat16` |
| Identity constants | `identityMatrix` and exact alias `eye` |
| Integer ranges | INT32 and INT64 `range` overloads |

That is exactly 31 public static methods. The existing package-private `createDerived` and
`createDerivedOutputs` and private `importFlat` and `nextTensorId` remain, for exactly 35 declared
methods total. No random or prefix method remains declared by `TensorFactory`.

The factory class Javadoc must explain this narrower ownership without referring to removed
methods. Existing construction/import/constant/range signatures and behavior remain unchanged.

### Public `TensorRandoms` boundary

Promote the existing class in place. It must be a public final non-record class with:

- no fields;
- no implemented interfaces;
- one private zero-argument constructor; and
- exactly these five public static entry signatures:

```java
public static Tensor randomNormal(
        Shape shape,
        DataType dataType,
        double mean,
        double standardDeviation,
        RandomGenerator randomGenerator,
        Optional<String> label,
        boolean requiresGrad)

public static Tensor randomUniform(
        Shape shape,
        DataType dataType,
        double lowerBoundInclusive,
        double upperBoundExclusive,
        RandomGenerator randomGenerator,
        Optional<String> label,
        boolean requiresGrad)

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

public static Tensor randomBernoulli(
        Shape shape,
        double probability,
        RandomGenerator randomGenerator,
        Optional<String> label)
```

Do not rename these methods in this cleanup. Existing call sites change only the declaring class
from `TensorFactory` to `TensorRandoms`.

The public entries inherit the exact former `TensorFactory` null checks rather than relying on
package-private callers:

- normal/uniform check `shape`, `dataType`, `randomGenerator`, `label` in that order;
- integral overloads check `shape`, `randomGenerator`, `label` in that order; and
- Bernoulli checks `shape`, `randomGenerator`, `label` in that order.

Every later validation, exception type/message, sample call, conversion, descriptor, allocation,
blank-label effect, exhaustion effect, and no-rollback rule remains observable as before except
for stack-trace declaring class. No forwarding method remains in `TensorFactory`.

### Integer range ownership

Both existing public `TensorFactory.range(...)` overloads remain unchanged. Their package-private
implementation moves from `TensorPopulations` to `TensorRanges` without changing:

- inclusive-start/exclusive-end behavior;
- non-empty and non-zero-step requirements;
- positive or negative direction validation;
- `BigInteger`-confined overflow-safe sizing;
- INT32/INT64 carrier mapping;
- canonical dense descriptors;
- label, allocation, and ID side effects; or
- exact validation messages and ordering.

`TensorRanges` contains no prefix or random entry and no public surface.

### Test-only prefix fixtures

Production removes every strict/cyclic prefix symbol. The existing useful fixture mechanics move
to `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTestData.java`.

`TensorTestData` is package-private, final, field-free, non-record, and non-instantiable. It may
retain twelve package-private static overloads named `fromStrictFlatPrefix` and
`fromCyclicFlatPrefix` for the six primitive carriers. It must:

- infer the same exact carrier data type;
- require a fully static Java-array-sized shape;
- preserve raw numeric and BFLOAT16 values;
- let public flat import normalize BOOL;
- copy strict prefixes and repeat cyclic prefixes without retaining the caller array;
- accept empty cyclic input only for an empty result; and
- delegate final construction to the matching public `TensorFactory.fromFlatArray(...)`.

This helper exists only to prepare repository test data. It is absent from `src/main`, production
bytecode, production generated Javadoc, Tensor API, and glossary public inventories. Existing
focused prefix behavior tests may target it so useful fixture coverage is not lost.

### Typed scalar decision

Do not add `TensorFactory.scalar(ScalarValue, ...)`, `full(Shape, ScalarValue, ...)`, or a generic
typed value overload. `ScalarValue` preserves semantic operation attributes and is not an eager
storage carrier. The current primitive methods already express exact eager values, including the
explicit BFLOAT16 conversion entries. A later concrete use case may propose a separate task.

### No transitional state

The migration is atomic:

- `TensorFactory` no longer declares random or prefix methods;
- `TensorRandoms` is the only production public random owner;
- prefix helpers exist only in test source; and
- no deprecated, forwarding, alias, reflection, or service-discovery bridge is introduced.

## Affected files

Expected production Java:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorFactory.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorRandoms.java`
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorPopulations.java`
  (delete)
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorRanges.java` (create)

Expected test Java:

- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTestData.java` (create)
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryPopulationTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryRandomTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryUniformRandomTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryIntegralRandomTest.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorFactoryBernoulliRandomTest.java`

Expected documentation and planning:

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- `docs/planning/modules/model/tasks/0018s-tensor-factory-surface-cleanup.md`
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum scope

This task may create, modify, move, or delete at most the exact 17 paths listed above.

File deletion and creation for the `TensorPopulations` to `TensorRanges` responsibility rename
count as separate paths. Do not rename the four historical focused random-test files merely to
change their filenames; their contents may identify `TensorRandoms` as the final public owner.

If another production type, test file, API document, build file, module, or package is needed,
stop and propose a follow-up or request explicit scope expansion before editing it.

## Javadoc requirements

- Finalize complete public type and method Javadocs for `TensorRandoms`, including every parameter,
  result, validation failure, source ownership rule, source advancement, conversion, allocation,
  ID side effect, and reproducibility boundary.
- Rewrite `TensorFactory` type Javadoc to describe only its retained responsibilities.
- Review retained `TensorFactory` methods and change their Javadocs only when they mention removed
  random/prefix ownership or broad factory claims.
- Document package-private `TensorRanges` and `TensorTestData` as narrow stateless implementation
  and test-fixture contracts respectively.
- Do not describe test-only prefix utilities as public product capabilities.
- Review `Tensor`, `TensorDescriptor`, `TensorProducer`, storage, import, constants, and
  `ScalarValue` Javadocs. Record a reasoned no-change conclusion unless the final diff reveals a
  factual stale reference; do not expand scope silently.

## Acceptance criteria

- `TensorFactory` has exactly the retained 31-method public surface and no random/prefix method.
- `TensorRandoms` is the sole public production owner of exactly the five existing random
  signatures, with no fields or public constructor.
- Every existing random behavior and failure/side-effect contract remains covered and passing
  after changing the declaring class.
- `TensorRanges` preserves both range paths exactly and contains no prefix responsibility.
- No production source, production class, generated Javadoc page, Tensor API inventory, or
  glossary public entry exposes `fromStrictFlatPrefix` or `fromCyclicFlatPrefix`.
- Test-only `TensorTestData` preserves useful exact-carrier strict/cyclic fixture behavior and is
  absent from the production artifact.
- No compatibility alias or deprecated bridge remains.
- Existing construction, allocation, import, constant, range, identity, ID, provenance, and
  derived-output tests remain passing.
- No `ScalarValue` eager factory overload is added.
- Public random documentation names the exact caller-owned source and makes clear that model owns
  no random service or RNG lifecycle.
- Tensor API and glossary distinguish product construction, focused random initialization, and
  test-only fixture preparation in newcomer-readable language.
- `ARCHITECTURE.md`, focused architecture documentation, Compile API, Training API, Gradle files,
  architecture tests, other modules, and later task specifications remain unchanged, with
  reasoned no-change conclusions recorded.
- A separate clean-context documentation-focused agent has inspected final source/tests and
  finalized Javadocs, Tensor API, glossary, capability/task/master/roadmap status, examples,
  terminology, links, and formatting in the same overall change.
- The public-surface cleanup capability checkpoint passes.
- Task 0018S is `Complete` consistently only after all validation passes. Task 0018T and every
  later task remain `Draft`, and no detailed later specification is created.

## Tests / validation

During implementation, use the focused suites as needed:

```bash
./gradlew :modules:model:test \
  --tests io.github.pho001.synaptik.model.tensor.TensorFactoryTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorFactoryPopulationTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorFactoryRandomTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorFactoryUniformRandomTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorFactoryIntegralRandomTest \
  --tests io.github.pho001.synaptik.model.tensor.TensorFactoryBernoulliRandomTest
```

After executable Java stabilizes, run the public-surface cleanup checkpoint once:

```bash
./gradlew test
```

This root command is the final Java validation for the task and must include the model and
architecture-test projects in its task graph. Do not additionally rerun the successful model suite
in the documentation pass unless executable Java changes afterward or a concrete risk is recorded.

Documentation-focused pass:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

The documentation pass must also:

- compile and run at least one Java 26 Tensor API example using public `TensorRandoms`;
- validate local Markdown links and changed anchors;
- verify generated production Javadoc contains `TensorRandoms` and contains no prefix API;
- verify production bytecode/source has exactly one public random owner and no prefix symbols;
- verify `TensorFactory` has 31 public and 35 total declared methods;
- verify `TensorTestData` exists only under test source and no production code imports it;
- verify exactly the authorized 17 paths changed;
- verify `ARCHITECTURE.md`, focused architecture docs, Compile API, Training API, Gradle files,
  architecture tests, and other modules are unchanged;
- verify 0018S status synchronization, 0018T Draft status, and absence of a detailed 0018T-or-later
  task; and
- check fences, final newlines, trailing whitespace, terminology, and `git diff --check`.

## Dependencies

- Tasks 0012 through 0012I: current factory construction, imports, populations, random behavior,
  identity allocation, and failure side effects.
- Task 0013A: final current full-value and identity-matrix construction surface.
- Task 0018N: exact separation between semantic `ScalarValue` and eager primitive storage values.

## Follow-up tasks

- Task 0018T remains the next Draft frontier for core scalar and unary numeric gaps after this
  cleanup checkpoint.
- Task 0019B remains the owner of explicit graph RNG state and dropout. Public `TensorRandoms`
  creates eager leaf data and does not satisfy graph-random semantics.
- No detailed follow-up specification is created by this task.

## Architecture impact

Expected impact: None.

This task narrows and relocates model-owned public eager construction without changing module
ownership, dependency direction, lifecycle responsibilities, or runtime/backend behavior. If
implementation requires any architecture change, stop and report it before editing
`ARCHITECTURE.md` or focused architecture documentation.

## Implementation prompt

Use this prompt in a separate agentic task/thread with clean context:

```text
You are working in the Synaptik repository.

Read AGENTS.md, ARCHITECTURE.md, documentation/planning rules, roadmap, model capabilities/master
plan, tasks 0012–0012I/0013A/0018N/0018S, Tensor API, glossary, and every affected or review-only
source/test named by task 0018S in full.

Implement task 0018S exactly. Atomically narrow TensorFactory to construction/import/constants/
range, promote TensorRandoms as the sole public explicit-source random owner, move strict/cyclic
prefix preparation to test-only TensorTestData, and split range mechanics into TensorRanges.
Preserve all retained behavior, failure ordering, source/ownership rules, allocation and ID side
effects. Add no aliases, generic initializer abstraction, random service, ScalarValue factory
overload, cross-layer behavior, dependencies, build, architecture changes, or later specs. Stay
within the exact seventeen paths and stop on scope or architecture conflict. Do not commit or push.

Run focused tests as needed and one final root test checkpoint after executable Java stabilizes.
Then hand the actual diff and exact evidence to a separate clean-context documentation-focused
agent in the same overall change. That agent must inspect final source/tests, finalize permitted
Javadocs/Tensor API/glossary/capability/planning documents, run model Javadoc and documentation/
scope checks, and must not repeat successful Java tests unless executable behavior changes or it
records a concrete reason.

Mark 0018S Complete only after both passes and the checkpoint succeed. Leave 0018T and every later
task Draft without a detailed specification.
```

## Local decisions

- Planning selected promotion of the existing `TensorRandoms` type rather than a new package,
  source abstraction, distribution enum, or second broad factory.
- Public random method names and signatures remain unchanged; only their declaring owner changes.
- Integer ranges remain in `TensorFactory` because they are a common exact eager construction,
  while strict/cyclic prefix shaping is retained only for repository fixture preparation.
- No eager `ScalarValue` overload is added because primitive carriers already express exact stored
  values and `ScalarValue` has a different semantic-attribute responsibility.

Implementation-local decisions:

- Retained the five established random method names and signatures exactly while moving the public
  declaring owner; no compatibility forwarding path remains in TensorFactory.
- Kept the complete source-ownership, source-advancement, conversion, allocation, ID-side-effect,
  and bounded-reproducibility contract in public TensorRandoms Javadoc rather than duplicating
  random behavior in TensorFactory documentation.
- Kept prefix terminology only where needed to explain the removed public boundary and the
  package-private test-source fixture; it is absent from product API inventories and generated
  production Javadoc.

## Known limitations

- This is a source-incompatible cleanup for provisional random and prefix call sites; no
  compatibility bridge is intentionally provided.
- `TensorRandoms` still requires fully static Java-array-sized shapes and caller-owned generators.
- Prefix preparation is not a product capability after this task.
- Eager random leaf construction is not graph RNG state and does not make dropout representable.
- Integer range remains eager, non-empty, INT32/INT64-only construction.

## Validation evidence

Planning context: clean planning task that created this specification and synchronized the initial
Ready frontier.

- Read the architecture contract, architecture index, documentation rules, planning guide,
  roadmap, model capability baseline/master plan, completed factory task history, current factory
  helpers, exact API tests, Tensor API, and glossary impact.
- Confirmed the current surface has 48 public `TensorFactory` methods: the selected final 31 plus
  five random and twelve prefix methods.
- Confirmed the existing `TensorRandoms` already contains the exact five cohesive random entries
  and no fields, so promotion requires no new package or abstraction.
- Confirmed `TensorPopulations` currently mixes exactly two range entries with twelve prefix
  entries, allowing an atomic split into narrow production range mechanics and test-only prefix
  fixtures.
- Confirmed no detailed task 0018S or later task specification existed before this planning step.
- Implementation, Java tests, Javadoc, checkpoint, and documentation validation were deferred from
  planning to the implementation and documentation contexts below.

Implementation context: clean task `/root/task_0018s_implementation`.

- The exact focused command in this task passed with `BUILD SUCCESSFUL`: 58 tests across
  `TensorFactoryTest`, `TensorFactoryPopulationTest`, `TensorFactoryRandomTest`,
  `TensorFactoryUniformRandomTest`, `TensorFactoryIntegralRandomTest`, and
  `TensorFactoryBernoulliRandomTest`; zero failures, errors, or skips.
- After executable Java stabilized, the one required root checkpoint `./gradlew test` passed with
  `BUILD SUCCESSFUL`: 715 model tests across 88 suites, zero failures, errors, or skips, and
  `testing:architecture-tests` included in the task graph.
- No executable Java changed after those successful runs. The documentation context changed only
  Javadocs/comments in permitted Java paths and the six permitted documentation/planning paths, so
  it did not repeat either Java test command.

Documentation-agent context: clean task
`/root/task_0018s_implementation/task_0018s_documentation`, applying the General,
API/Javadoc, Planning, and Example profiles.

- Independently read the repository instructions, full architecture contract and current
  architecture index, documentation/planning rules, roadmap, capability baseline, master plan,
  this task, the actual diff, affected source/tests, Tensor API, and glossary.
- Finalized the narrowed TensorFactory type Javadoc; complete public TensorRandoms type and method
  Javadocs for all parameters, results, failures, caller source ownership/advancement, conversions,
  allocation and ID effects, and reproducibility limits; and the narrow package-private
  TensorRanges and test-only TensorTestData contracts. Retained TensorFactory member Javadocs were
  reviewed and remained accurate without further edits.
- Finalized `docs/api/tensor-api.md` and `docs/glossary.md`: TensorFactory owns construction,
  allocation, import, constants, and integer ranges; TensorRandoms is the sole focused public
  explicit-source eager-random owner; prefix preparation is test-fixture mechanics rather than
  product inventory. Capabilities, task, master plan, and roadmap are synchronized to Complete.
- `./gradlew :modules:model:javadoc` passed with `BUILD SUCCESSFUL`; two tasks executed. Generated
  production Javadoc contains the public TensorRandoms page and all five public entries, and no
  strict/cyclic prefix symbol.
- A Java 26 public example compiled and ran with
  `javac -cp modules/model/build/classes/java/main -d /tmp/tensor-randoms-doc-example
  /tmp/TensorRandomsExample.java` followed by `java`; it printed `BOOL`, `Shape[2, 2]`, and
  `[1, 0, 1, 0]` for four scripted Bernoulli draws.
- The revised integer-range documentation example also compiled and ran; it printed `INT32`,
  `Shape[3]`, and `[1, 4, 7]`, matching the documented inclusive-start/exclusive-end result.
- Reflection and `javap -public` confirmed TensorFactory has exactly 31 public and 35 total
  declared methods; TensorRandoms has exactly five public methods and no fields. Source and
  generated-Javadoc scans found exactly one public production random owner and no production
  prefix symbol or compatibility bridge.
- Source/artifact scans confirmed TensorTestData exists only under test source, no production code
  imports it, and no main-class artifact contains it. The exact changed-path inventory is the
  authorized seventeen paths: four production paths including one deletion and one creation,
  seven test paths including one creation, and six documentation/planning paths.
- Targeted Markdown validation resolved all 473 local links and 139 GitHub-style fragments in the
  six changed Markdown files. All six have final newlines, no trailing whitespace, and balanced
  backtick and tilde fences.
- The first local anchor-checker draft falsely rejected four existing glossary anchors because it
  collapsed adjacent hyphens; correcting the checker to GitHub-style space replacement made those
  exact anchors pass before the final 473-link/139-fragment result.
- Status and task-inventory scans found 0018S Complete in this task, master plan, and roadmap;
  0018T and every later task remain Draft, with no detailed 0018T-or-later specification.
- `git diff --check` passed with no output after the final combined edits. New untracked paths also
  passed the explicit trailing-whitespace, final-newline, and fence checks.
- Reviewed Tensor, TensorDescriptor, TensorProducer, storage, flat/nested import, constant, and
  ScalarValue Javadocs. They remain accurate because this task changes only public ownership of
  random initialization and removes prefix convenience; it does not change those types,
  construction/storage/import/constant behavior, or the semantic-value/eager-carrier boundary.
- Architecture documentation needs no change because module ownership, dependency direction,
  lifecycle boundaries, and the no-service rule are unchanged. Compile API needs no change because
  eager leaves still have no compiler capture or random Operation. Training API needs no change
  because no gradient rule, parameter, optimizer, dropout, or training workflow changed. Gradle,
  architecture tests, backend conformance, integration tests, other modules, and later task specs
  likewise need no change because there is no dependency, build, cross-layer, or executable
  contract expansion.

## Implementation notes

Implemented atomically within the exact seventeen-path scope. TensorFactory now exposes only the
retained 31-method construction/import/constant/range surface; TensorRandoms is the sole public
explicit-source random owner; TensorRanges owns package-private range mechanics; TensorPopulations
is deleted; and TensorTestData owns test-only prefix fixture preparation.

## Completion summary

- Completed changes: narrowed TensorFactory, promoted TensorRandoms, moved range mechanics to
  TensorRanges, removed TensorPopulations, and moved prefix fixture preparation to TensorTestData.
- Files changed or created: exactly the seventeen production, test, documentation, and planning
  paths authorized by this task; no additional repository path changed.
- Tests and validation: reused the passing 58-test focused command and 715-test root checkpoint;
  model Javadoc, Java 26 example, reflection/javap, generated-page/source/artifact, Markdown,
  exact-scope, status, terminology, newline/fence/whitespace, and diff checks passed.
- Documentation-agent review: completed independently in a clean documentation-focused context.
- Documentation impact: Tensor API, glossary, capabilities, task, master plan, and roadmap now
  describe the final narrowed ownership and test-fixture boundary.
- Javadoc review: TensorRandoms, TensorFactory, TensorRanges, and TensorTestData finalized; retained
  factory members and related Tensor/storage/import/constant/scalar contracts reviewed unchanged.
- Glossary impact: added focused Tensor random initialization ownership and removed prefix creation
  from the public product inventory.
- Unresolved issues: None.
- Follow-up required: None.

Status: Complete
