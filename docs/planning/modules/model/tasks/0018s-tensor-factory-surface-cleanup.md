# Task 0018S: [Tensor Factory](../../../../glossary.md#tensor-factory) Surface Cleanup

**Editorial reading key (current context, not a status update).**

This record captures the historical narrowing of `TensorFactory`, promotion of `TensorRandoms`, and
removal of production prefix helpers; its method counts and completion evidence are historical, not
current inventories. Here “owner” means API responsibility, not storage ownership. The caller
supplies a `RandomGenerator`; stateless `TensorRandoms` samples values and delegates to
`TensorFactory.fromFlatArray`, which copies the carrier into independent storage and allocates
identity. Test-only `TensorTestData` prepares prefixes; `TensorRanges` supplies private
integer-range mechanics. A leaf has no producing expression, not necessarily stored values; eager
random leaves do not represent graph RNG state.

For current distinctions, follow [allocation, import and constant
contracts](../../../../reference/tensor/creation.md#allocation-import-and-constant-contracts):
`create` retains a descriptor and optional borrowed storage, `allocate` reserves storage, and import
copies values. For example, importing BOOL bytes zero and seven stores zero and one without
retaining the source array. Primitive scalar/full inputs supply eager values; `ScalarValue`
describes semantic operation state. Model semantics defines meaning; execution providers realize
supported routes rather than redefine it. Storage-free expressions neither execute nor own result
buffers.

## Status

Complete

## Goal

Make the [eager](../../../../glossary.md#term-eager-evaluation) [tensor](../../../../glossary.md#tensor)-construction [API](../../../../glossary.md#term-application-programming-interface-api) small enough to remain understandable as the [model](../../../../glossary.md#term-model) grows.
`TensorFactory` will keep model [identity](../../../../glossary.md#term-identity-meanings) allocation, [descriptor](../../../../glossary.md#term-descriptor)-based construction, [heap](../../../../glossary.md#term-heap)
allocation, copied imports, [constants](../../../../glossary.md#term-constant), and integer ranges. Explicit-source random [sampling](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters) will
move to the already cohesive `TensorRandoms` [owner](../../../../glossary.md#term-owner), and strict/cyclic prefix population will leave
production code because it is [fixture](../../../../glossary.md#term-fixture) preparation rather than a foundational tensor capability.

The intended [ownership](../../../../glossary.md#term-ownership-meanings) after this task is:

```text
TensorFactory
  identity + descriptor construction + allocation + import + constants + integer ranges

TensorRandoms
  explicit caller-owned RandomGenerator + distribution parameters -> eager leaf Tensor

TensorTestData (test source only)
  strict/cyclic prefix preparation -> TensorFactory.fromFlatArray(...)
```

This is an intentional pre-stabilization public API cleanup. It does not retain deprecated or
transitional [aliases](../../../../glossary.md#aliasing-and-arena).

## [Scope](../../../../glossary.md#term-scope)

- Remove all twelve public `TensorFactory.fromStrictFlatPrefix(...)` and
  `TensorFactory.fromCyclicFlatPrefix(...)` overloads.
- Remove strict/cyclic prefix implementation from production source.
- Preserve the useful prefix-[fixture](../../../../glossary.md#term-fixture) behavior in one package-private test-source
  `TensorTestData` helper so repository tests can prepare repeated or truncated typed carriers
  without making that behavior a production contract.<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare))</span>
- Remove the five public random entries from `TensorFactory`:
  `randomNormal`, `randomUniform`, the two `randomInt` overloads, and `randomBernoulli`.
- Promote the existing stateless `TensorRandoms` class to the focused public random-initialization
  owner and make those same five signatures public without changing names or semantics.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [signatures glossary](../../../../glossary.md#term-signature))</span>
- Preserve every random distribution, validation order and message, caller-owned-generator rule,
  source-call count, conversion rule, allocation order, label behavior, and tensor-ID side effect.<span data-glossary-navigation="true"> ([distribution glossary](../../../../glossary.md#term-distribution); [validation glossary](../../../../glossary.md#term-validation); [side effect glossary](../../../../glossary.md#term-side-effect))</span>
- Preserve both public integer `TensorFactory.range(...)` overloads and move only their
  package-private mechanics from broad `TensorPopulations` to narrow `TensorRanges`.
- Delete production `TensorPopulations` after its range and prefix responsibilities have moved to
  their selected owners.
- Keep every existing `TensorFactory` construction, allocation, import, scalar, zero/one,
  full-value, identity-matrix, derived-output, and ID-allocation contract unchanged.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [identity glossary](../../../../glossary.md#term-identity-meanings); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Do not add [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) overloads to [eager](../../../../glossary.md#term-eager-evaluation) [tensor](../../../../glossary.md#tensor) construction. `ScalarValue` remains semantic
  operation state; the existing exact primitive scalar/full APIs remain the eager storage API.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Update the exact public-surface tests, focused behavior tests, Tensor API, glossary, capability
  baseline, task status, model master plan, and roadmap.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
- Close the recorded public-surface cleanup checkpoint after all executable and documentation
  validation passes.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- a default, global, thread-local, [engine](../../../../glossary.md#term-engine)-owned, [runtime](../../../../glossary.md#term-runtime)-owned, or service-located random source
- seed-only random overloads, generator selection, source splitting, source synchronization,
  source retention, source serialization, or cross-generator reproducibility<span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed); [synchronization glossary](../../../../glossary.md#term-synchronization); [serialization glossary](../../../../glossary.md#term-serialization); [reproducibility glossary](../../../../glossary.md#artifact-manifest-and-source-provenance))</span>
- [graph RNG state](../../../../glossary.md#graph-rng-state--graphrngstate), random [`Operation`](../../../../glossary.md#operation) kinds, dropout, state-producing tensors, compiler capture,
  or task-0019B behavior<span data-glossary-navigation="true"> ([dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [tensors glossary](../../../../glossary.md#tensor); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- new distributions, changed distribution formulas, changed parameter types, changed output types,
  or renamed random methods<span data-glossary-navigation="true"> ([distributions glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [distribution glossary](../../../../glossary.md#term-distribution); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- production strict-prefix, cyclic-prefix, repeat, tile, fill, [fixture](../../../../glossary.md#term-fixture), [dataset](../../../../glossary.md#term-dataset), or test-data APIs
- compatibility aliases, deprecated bridges, forwarding methods, or reflective adapters for any
  removed <span data-glossary-navigation="true"> ([aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>`TensorFactory` method
- removal, renaming, or semantic changes to integer `TensorFactory.range(...)`
- changes to flat or nested import, [heap](../../../../glossary.md#term-heap) allocation, [constants](../../../../glossary.md#term-constant), `identityMatrix`, `eye`, ID
  allocation, provenance, <span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>[`TensorProducer`](../../../../glossary.md#producer--tensorproducer), or derived-output construction
- [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue)-to-Tensor conversion, generic scalar/full overloads, boxed values, implicit
  conversion, or new data types<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>
- typed tensor read/export APIs, zero-copy import, backing-array exposure, storage mutation,
  native storage, deterministic lifetime, or ownership changes<span data-glossary-navigation="true"> ([lifetime glossary](../../../../glossary.md#term-lifetime); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- operation semantics, graph state, gradients, autograd, compiler, planning, prepare, runtime,
  engine, backend, ONNX, or training behavior<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [gradients glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [planning glossary](../../../../glossary.md#term-planning); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [ONNX glossary](../../../../glossary.md#term-onnx))</span>
- dependencies, Gradle changes, preview/incubator features, architecture changes, another module,
  or another production package<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [features glossary](../../../../glossary.md#term-software-feature); [module glossary](../../../../glossary.md#term-module))</span>
- a detailed specification for task 0018T or any later task

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, especially `modules/model` ownership
  of public tensor construction and the prohibition on runtime/backend state<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [tensor glossary](../../../../glossary.md#tensor); [runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend))</span>
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Implementation roadmap](../../../roadmap.md)
- [Model capability baseline](../capabilities.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>, especially the prefix-population and large
  factory-surface decisions
- [Model master plan](../master-plan.md)
- [Task 0012](0012-tensor-factory.md) and [tasks 0012A–0012I](../master-plan.md#task-list), which
  established the current construction, population, and random behavior
- [Task 0013A](0013a-full-value-and-identity-matrix-tensor-creation.md), which established the
  current full-value and identity contracts<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- [Task 0018N](0018n-typed-scalar-value-contract.md), which deliberately kept eager primitive
  scalar/full APIs separate from semantic <span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [scalar glossary](../../../../glossary.md#term-scalar))</span>[`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue)
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span> and [glossary](../../../../glossary.md)

## Architecture constraints

- All production work remains in `modules/model` and the existing
  `io.github.pho001.synaptik.model.tensor` package.
- `TensorFactory` remains the sole model identity allocator and the package-private construction
  seam used by current expression helpers. Moving public random entry points must not duplicate,
  expose, reset, or relocate its allocator.
- Random results remain [eager](../../../../glossary.md#term-eager-evaluation) [provenance](../../../../glossary.md#provenance)-free [leaf](../../../../glossary.md#term-leaf) [tensors](../../../../glossary.md#tensor). `TensorRandoms` must build a validated
  dense descriptor and delegate once to the matching public <span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor))</span>`TensorFactory.fromFlatArray(...)`;
  it must not construct `Tensor`, storage, provenance, or identifiers directly.
- `TensorRandoms` is a stateless public namespace, not a random source, registry, service,
  distribution hierarchy, or lifecycle owner. It retains no generator or result.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [distribution glossary](../../../../glossary.md#term-distribution); [lifecycle glossary](../../../../glossary.md#lifecycle); [owner glossary](../../../../glossary.md#term-owner))</span>
- The exact caller-owned `RandomGenerator` remains mandatory for every random method.
- Prefix behavior becomes test code only. No production source or generated production Javadoc
  may expose strict/cyclic prefix creation after this task.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- `TensorTestData` may call only public model construction APIs. Production code must not import or
  otherwise depend on test source.
- Integer range remains eager [model](../../../../glossary.md#term-model)-owned leaf-data construction and continues to use
  `TensorFactory` [identity](../../../../glossary.md#term-identity-meanings), flat import, and exact [INT32](../../../../glossary.md#data-type--datatype)/[INT64](../../../../glossary.md#data-type--datatype) [carrier](../../../../glossary.md#term-storage-carrier) rules.
- Removed public APIs receive no alias or deprecation bridge. The repository has not stabilized a
  released compatibility contract for these provisional methods.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- If implementation requires a new production package, a random-source abstraction, a generic
  initializer hierarchy, storage changes, another module, or an architecture/dependency change,
  stop and report the conflict.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.model.tensor` — owns public eager Tensor construction, random leaf
  initialization, package-private mechanics, and same-package tests.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [leaf glossary](../../../../glossary.md#term-leaf))</span>
- `io.github.pho001.synaptik.model.datatype` — supplies exact output data types and BFLOAT16
  conversion.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>
- `io.github.pho001.synaptik.model.shape` — supplies fully static [eager](../../../../glossary.md#term-eager-evaluation) result [shapes](../../../../glossary.md#shape).
- `io.github.pho001.synaptik.model.layout` — supplies [canonical](../../../../glossary.md#term-canonical-form) dense-[contiguous](../../../../glossary.md#term-contiguous) [descriptors](../../../../glossary.md#term-descriptor).

Packages added or changed:

- No package is added. The existing `model.tensor` package remains the cohesive boundary.

Type placement:

- `io.github.pho001.synaptik.model.tensor.TensorFactory` — narrowed public identity,
  construction, allocation, import, constant, and integer-range boundary.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [constant glossary](../../../../glossary.md#term-constant))</span>
- `io.github.pho001.synaptik.model.tensor.TensorRandoms` — existing helper promoted to a public
  final stateless owner for the five explicit-source random methods.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- `io.github.pho001.synaptik.model.tensor.TensorRanges` — new package-private final production
  helper containing only the two integer-range entries and their private sizing/descriptor logic.
- `io.github.pho001.synaptik.model.tensor.TensorPopulations` — removed because its broad mixed
  production responsibility no longer exists.
- `io.github.pho001.synaptik.model.tensor.TensorTestData` — package-private test-source helper for
  strict/cyclic prefix fixture preparation; it is not part of the production artifact.<span data-glossary-navigation="true"> ([fixture glossary](../../../../glossary.md#term-fixture); [artifact glossary](../../../../glossary.md#term-artifact))</span>

## Required contract

### Final `TensorFactory` boundary

`TensorFactory` remains a public final non-record static utility with exactly its existing two
private [identity](../../../../glossary.md#term-identity-meanings) atomics and private zero-[argument](../../../../glossary.md#term-argument) constructor. Its public methods after cleanup
are exactly these categories:

| Category | Public entries |
|---|---:|
| [Descriptor](../../../../glossary.md#term-descriptor) construction | two `create` overloads |
| Resolved [heap](../../../../glossary.md#term-heap) allocation | two `allocate` overloads |
| Exact flat import | six `fromFlatArray` overloads |
| Rectangular nested import | one `fromNestedArray` method |
| Typed [scalar](../../../../glossary.md#term-scalar) [constants](../../../../glossary.md#term-constant) | five primitive `scalar` overloads plus `scalarBFloat16` |
| Zero/one constants | `zeros`, `ones`, `zerosLike`, `onesLike` |
| Typed full constants | five primitive `full` overloads plus `fullBFloat16` |
| Identity constants | `identityMatrix` and exact [alias](../../../../glossary.md#aliasing-and-arena) `eye` |
| Integer ranges | [INT32](../../../../glossary.md#data-type--datatype) and [INT64](../../../../glossary.md#data-type--datatype) `range` overloads |

That is exactly 31 public static methods. The existing package-private `createDerived` and
`createDerivedOutputs` and private `importFlat` and `nextTensorId` remain, for exactly 35 declared
methods total. No random or prefix method remains declared by `TensorFactory`.

The factory class [Javadoc](../../../../glossary.md#term-javadoc) must explain this narrower [ownership](../../../../glossary.md#term-ownership-meanings) without referring to removed
methods. Existing construction/import/constant/range [signatures](../../../../glossary.md#term-signature) and behavior remain unchanged.

### Public `TensorRandoms` boundary

Promote the existing class in place. It must be a public final non-record class with:

- no fields;
- no implemented interfaces;
- one private zero-[argument](../../../../glossary.md#term-argument) constructor; and
- exactly these five public static entry [signatures](../../../../glossary.md#term-signature):

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

- normal/uniform check [`shape`](../../../../glossary.md#shape), `dataType`, `randomGenerator`, `label` in that order;
- integral overloads check `shape`, `randomGenerator`, `label` in that order; and
- [Bernoulli](../../../../glossary.md#term-bernoulli-distribution) checks `shape`, `randomGenerator`, `label` in that order.

Every later [validation](../../../../glossary.md#term-validation), exception type/message, [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) call, conversion, [descriptor](../../../../glossary.md#term-descriptor), allocation,
blank-label effect, exhaustion effect, and no-[rollback](../../../../glossary.md#term-rollback) rule remains observable as before except
for stack-[trace](../../../../glossary.md#trace) declaring class. No forwarding method remains in `TensorFactory`.

### Integer range [ownership](../../../../glossary.md#term-ownership-meanings)

Both existing public `TensorFactory.range(...)` overloads remain unchanged. Their package-private
implementation moves from `TensorPopulations` to `TensorRanges` without changing:

- inclusive-start/exclusive-end behavior;
- non-empty and non-zero-step requirements;
- positive or negative direction [validation](../../../../glossary.md#term-validation);
- `BigInteger`-confined [overflow](../../../../glossary.md#term-overflow)-safe sizing;
- [INT32](../../../../glossary.md#data-type--datatype)/[INT64](../../../../glossary.md#data-type--datatype) [carrier](../../../../glossary.md#term-storage-carrier) mapping;
- [canonical](../../../../glossary.md#term-canonical-form) dense [descriptors](../../../../glossary.md#term-descriptor);
- label, allocation, and ID side effects; or
- exact validation messages and ordering.

`TensorRanges` contains no prefix or random entry and no public surface.

### Test-only prefix fixtures

Production removes every strict/cyclic prefix [symbol](../../../../glossary.md#term-symbol). The existing useful [fixture](../../../../glossary.md#term-fixture) mechanics move
to `modules/model/src/test/java/io/github/pho001/synaptik/model/tensor/TensorTestData.java`.

`TensorTestData` is package-private, final, field-free, non-record, and non-instantiable. It may
retain twelve package-private static overloads named `fromStrictFlatPrefix` and
`fromCyclicFlatPrefix` for the six primitive carriers. It must:

- infer the same exact [carrier](../../../../glossary.md#term-storage-carrier) [data type](../../../../glossary.md#data-type--datatype);
- require a fully static Java-array-sized [shape](../../../../glossary.md#shape);
- preserve raw numeric and [BFLOAT16](../../../../glossary.md#term-bfloat16) values;
- let public flat import normalize [BOOL](../../../../glossary.md#data-type--datatype);
- copy strict prefixes and repeat cyclic prefixes without retaining the caller array;
- accept empty cyclic input only for an empty result; and
- delegate final construction to the matching public `TensorFactory.fromFlatArray(...)`.

This helper exists only to [prepare](../../../../glossary.md#prepare) repository test data. It is absent from `src/main`, production
[bytecode](../../../../glossary.md#term-bytecode), production generated [Javadoc](../../../../glossary.md#term-javadoc), [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api), and glossary public inventories. Existing
focused prefix behavior tests may [target](../../../../glossary.md#term-target) it so useful fixture coverage is not lost.

### Typed [scalar](../../../../glossary.md#term-scalar) decision

Do not add `TensorFactory.scalar(ScalarValue, ...)`, `full(Shape, ScalarValue, ...)`, or a generic
typed value overload. [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) preserves semantic [operation](../../../../glossary.md#operation) attributes and is not an [eager](../../../../glossary.md#term-eager-evaluation)
[storage carrier](../../../../glossary.md#term-storage-carrier). The current primitive methods already express exact eager values, including the
explicit [BFLOAT16](../../../../glossary.md#term-bfloat16) conversion entries. A later concrete use case may propose a separate task.

### No transitional state

The migration is [atomic](../../../../glossary.md#resource-transaction-and-rollback):

- `TensorFactory` no longer declares random or prefix methods;
- `TensorRandoms` is the only production public random [owner](../../../../glossary.md#term-owner);
- prefix helpers exist only in test source; and
- no deprecated, forwarding, [alias](../../../../glossary.md#aliasing-and-arena), reflection, or service-discovery bridge is introduced.

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

Expected documentation and [planning](../../../../glossary.md#term-planning):

- `docs/api/tensor-api.md`
- `docs/glossary.md`
- `docs/planning/modules/model/capabilities.md`
- `docs/planning/modules/model/tasks/0018s-tensor-factory-surface-cleanup.md`
- `docs/planning/modules/model/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create, modify, move, or delete at most the exact 17 paths listed above.

File deletion and creation for the `TensorPopulations` to `TensorRanges` responsibility rename
count as separate paths. Do not rename the four historical focused random-test files merely to
change their filenames; their contents may identify `TensorRandoms` as the final public [owner](../../../../glossary.md#term-owner).

If another production type, test file, [API](../../../../glossary.md#term-application-programming-interface-api) document, build file, [module](../../../../glossary.md#term-module), or package is needed,
stop and propose a follow-up or request explicit scope expansion before editing it.

## [Javadoc](../../../../glossary.md#term-javadoc) requirements

- Finalize complete public type and method Javadocs for `TensorRandoms`, including every parameter,
  result, validation failure, source ownership rule, source advancement, conversion, allocation,
  ID side effect, and reproducibility boundary.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [validation glossary](../../../../glossary.md#term-validation); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [side effect glossary](../../../../glossary.md#term-side-effect); [reproducibility glossary](../../../../glossary.md#artifact-manifest-and-source-provenance))</span>
- Rewrite `TensorFactory` type Javadoc to describe only its retained responsibilities.
- Review retained `TensorFactory` methods and change their Javadocs only when they mention removed
  random/prefix ownership or broad factory claims.
- Document package-private `TensorRanges` and `TensorTestData` as narrow stateless implementation
  and test-fixture contracts respectively.<span data-glossary-navigation="true"> ([fixture glossary](../../../../glossary.md#term-fixture))</span>
- Do not describe test-only prefix utilities as public product capabilities.
- Review [`Tensor`](../../../../glossary.md#tensor), `TensorDescriptor`, [`TensorProducer`](../../../../glossary.md#producer--tensorproducer), storage, import, [constants](../../../../glossary.md#term-constant), and
  [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) Javadocs. Record a reasoned no-change conclusion unless the final diff reveals a
  factual stale reference; do not expand scope silently.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>

## Acceptance criteria

- `TensorFactory` has exactly the retained 31-method public surface and no random/prefix method.
- `TensorRandoms` is the sole public production owner of exactly the five existing random
  signatures, with no fields or public constructor.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [signatures glossary](../../../../glossary.md#term-signature))</span>
- Every existing random behavior and failure/side-effect contract remains covered and passing
  after changing the declaring class.
- `TensorRanges` preserves both range paths exactly and contains no prefix responsibility.
- No production source, production class, generated Javadoc page, Tensor API inventory, or
  glossary public entry exposes <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>`fromStrictFlatPrefix` or `fromCyclicFlatPrefix`.
- Test-only `TensorTestData` preserves useful exact-carrier strict/cyclic fixture behavior and is
  absent from the production artifact.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [fixture glossary](../../../../glossary.md#term-fixture); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- No compatibility [alias](../../../../glossary.md#aliasing-and-arena) or deprecated bridge remains.
- Existing construction, allocation, import, constant, range, identity, ID, provenance, and
  derived-output tests remain passing.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [identity glossary](../../../../glossary.md#term-identity-meanings); [provenance glossary](../../../../glossary.md#provenance))</span>
- No [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) [eager](../../../../glossary.md#term-eager-evaluation) factory overload is added.
- Public random documentation names the exact caller-owned source and makes clear that model owns
  no random service or RNG lifecycle.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Tensor API and glossary distinguish product construction, focused random initialization, and
  test-only fixture preparation in newcomer-readable language.
- `ARCHITECTURE.md`, focused architecture documentation, Compile API, Training API, Gradle files,
  architecture tests, other modules, and later task specifications remain unchanged, with
  reasoned no-change conclusions recorded.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>
- A separate clean-context documentation-focused agent has inspected final source/tests and
  finalized Javadocs, Tensor API, glossary, capability/task/master/roadmap status, examples,
  terminology, links, and formatting in the same overall change.
- The public-surface cleanup capability [checkpoint](../../../../glossary.md#term-checkpoint) passes.
- Task 0018S is `Complete` consistently only after all validation passes. Task 0018T and every
  later task remain <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>`Draft`, and no detailed later specification is created.

## Tests / [validation](../../../../glossary.md#term-validation)

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

After executable Java stabilizes, [run](../../../../glossary.md#term-run) the public-surface cleanup [checkpoint](../../../../glossary.md#term-checkpoint) once:

```bash
./gradlew test
```

This root command is the final Java validation for the task and must include the [model](../../../../glossary.md#term-model) and
architecture-test projects in its task [graph](../../../../glossary.md#graph). Do not additionally rerun the successful model suite
in the documentation pass unless executable Java changes afterward or a concrete risk is recorded.

Documentation-focused pass:

```bash
./gradlew :modules:model:javadoc
git diff --check
```

The documentation pass must also:

- [compile](../../../../glossary.md#compile) and run at least one Java 26 [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api) example using public `TensorRandoms`;
- validate local Markdown links and changed anchors;
- verify generated production [Javadoc](../../../../glossary.md#term-javadoc) contains `TensorRandoms` and contains no prefix API;
- verify production [bytecode](../../../../glossary.md#term-bytecode)/source has exactly one public random [owner](../../../../glossary.md#term-owner) and no prefix symbols;
- verify `TensorFactory` has 31 public and 35 total declared methods;
- verify `TensorTestData` exists only under test source and no production code imports it;
- verify exactly the authorized 17 paths changed;
- verify `ARCHITECTURE.md`, focused architecture docs, Compile API, Training API, Gradle files,
  architecture tests, and other modules are unchanged;<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>
- verify 0018S status synchronization, 0018T Draft status, and absence of a detailed 0018T-or-later
  task; and<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- check fences, final newlines, trailing whitespace, terminology, and `git diff --check`.

## Dependencies

- Tasks 0012 through 0012I: current factory construction, imports, populations, random behavior,
  identity allocation, and failure side effects.
- Task 0013A: final current full-value and [identity](../../../../glossary.md#term-identity-meanings)-[matrix](../../../../glossary.md#term-matrix-meanings) construction surface.
- Task 0018N: exact separation between semantic [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) and [eager](../../../../glossary.md#term-eager-evaluation) primitive storage values.

## Follow-up tasks

- Task 0018T remains the next Draft frontier for core scalar and unary numeric gaps after this
  cleanup checkpoint.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [unary glossary](../../../../glossary.md#term-unary); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Task 0019B remains the [owner](../../../../glossary.md#term-owner) of explicit [graph RNG state](../../../../glossary.md#graph-rng-state--graphrngstate) and [dropout](../../../../glossary.md#dropout--inverted-dropout). Public `TensorRandoms`
  creates eager leaf data and does not satisfy graph-random semantics.<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [leaf glossary](../../../../glossary.md#term-leaf); [graph glossary](../../../../glossary.md#graph))</span>
- No detailed follow-up specification is created by this task.

## Architecture impact

Expected impact: None.

This task narrows and relocates [model](../../../../glossary.md#term-model)-owned public [eager](../../../../glossary.md#term-eager-evaluation) construction without changing [module](../../../../glossary.md#term-module)
[ownership](../../../../glossary.md#term-ownership-meanings), [dependency direction](../../../../glossary.md#java-platform-and-build-tools), [lifecycle](../../../../glossary.md#lifecycle) responsibilities, or [runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) behavior. If
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

- [Planning](../../../../glossary.md#term-planning) selected promotion of the existing `TensorRandoms` type rather than a new package,
  source abstraction, distribution enum, or second broad factory.<span data-glossary-navigation="true"> ([distribution glossary](../../../../glossary.md#term-distribution))</span>
- Public random method names and [signatures](../../../../glossary.md#term-signature) remain unchanged; only their declaring [owner](../../../../glossary.md#term-owner) changes.
- Integer ranges remain in `TensorFactory` because they are a common exact eager construction,
  while strict/cyclic prefix shaping is retained only for repository fixture preparation.<span data-glossary-navigation="true"> ([fixture glossary](../../../../glossary.md#term-fixture))</span>
- No [eager](../../../../glossary.md#term-eager-evaluation) [`ScalarValue`](../../../../glossary.md#typed-scalar-value--scalarvalue) overload is added because primitive carriers already express exact stored
  values and `ScalarValue` has a different semantic-attribute responsibility.

Implementation-local decisions:

- Retained the five established random method names and signatures exactly while moving the public
  declaring owner; no compatibility forwarding path remains in TensorFactory.
- Kept the complete source-ownership, source-advancement, conversion, allocation, ID-side-effect,
  and bounded-reproducibility contract in public TensorRandoms Javadoc rather than duplicating
  random behavior in TensorFactory documentation.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [reproducibility glossary](../../../../glossary.md#artifact-manifest-and-source-provenance); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Kept prefix terminology only where needed to explain the removed public boundary and the
  package-private test-source fixture; it is absent from product API inventories and generated
  production Javadoc.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Known limitations

- This is a source-incompatible cleanup for provisional random and prefix call sites; no
  compatibility bridge is intentionally provided.
- `TensorRandoms` still requires fully static Java-array-sized [shapes](../../../../glossary.md#shape) and caller-owned generators.
- Prefix preparation is not a product capability after this task.
- [Eager](../../../../glossary.md#term-eager-evaluation) random [leaf](../../../../glossary.md#term-leaf) construction is not [graph RNG state](../../../../glossary.md#graph-rng-state--graphrngstate) and does not make [dropout](../../../../glossary.md#dropout--inverted-dropout) representable.
- Integer range remains eager, non-empty, [INT32](../../../../glossary.md#data-type--datatype)/[INT64](../../../../glossary.md#data-type--datatype)-only construction.

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) context: clean planning task that created this specification and synchronized the initial
[Ready](../../../../glossary.md#term-ready) frontier.

- Read the architecture contract, architecture index, documentation rules, planning guide,
  roadmap, model capability baseline/master plan, completed factory task history, current factory
  helpers, exact API tests, Tensor API, and glossary impact.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Confirmed the current surface has 48 public `TensorFactory` methods: the selected final 31 plus
  five random and twelve prefix methods.
- Confirmed the existing `TensorRandoms` already contains the exact five cohesive random entries
  and no fields, so promotion requires no new package or abstraction.
- Confirmed `TensorPopulations` currently mixes exactly two range entries with twelve prefix
  entries, allowing an atomic split into narrow production range mechanics and test-only prefix
  fixtures.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>
- Confirmed no detailed task 0018S or later task specification existed before this planning step.
- Implementation, Java tests, Javadoc, checkpoint, and documentation validation were deferred from
  planning to the implementation and documentation contexts below.

Implementation context: clean task `/root/task_0018s_implementation`.

- The exact focused command in this task passed with `BUILD SUCCESSFUL`: 58 tests across
  `TensorFactoryTest`, `TensorFactoryPopulationTest`, `TensorFactoryRandomTest`,
  `TensorFactoryUniformRandomTest`, `TensorFactoryIntegralRandomTest`, and
  `TensorFactoryBernoulliRandomTest`; zero failures, errors, or skips.
- After executable Java stabilized, the one required root [checkpoint](../../../../glossary.md#term-checkpoint) `./gradlew test` passed with
  `BUILD SUCCESSFUL`: 715 [model](../../../../glossary.md#term-model) tests across 88 suites, zero failures, errors, or skips, and
  `testing:architecture-tests` included in the task [graph](../../../../glossary.md#graph).
- No executable Java changed after those successful runs. The documentation context changed only
  Javadocs/comments in permitted Java paths and the six permitted documentation/planning paths, so
  it did not repeat either Java test command.

Documentation-agent context: clean task
`/root/task_0018s_implementation/task_0018s_documentation`, applying the General,
[API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc), Planning, and Example profiles.

- Independently read the repository instructions, full architecture contract and current
  architecture index, documentation/planning rules, roadmap, capability baseline, master plan,
  this task, the actual diff, affected source/tests, Tensor API, and glossary.
- Finalized the narrowed TensorFactory type Javadoc; complete public TensorRandoms type and method
  Javadocs for all parameters, results, failures, caller source ownership/advancement, conversions,
  allocation and ID effects, and reproducibility limits; and the narrow package-private
  TensorRanges and test-only TensorTestData contracts. Retained TensorFactory member Javadocs were
  reviewed and remained accurate without further edits.<span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [reproducibility glossary](../../../../glossary.md#artifact-manifest-and-source-provenance))</span>
- Finalized `docs/api/tensor-api.md` and `docs/glossary.md`: TensorFactory owns construction,
  allocation, import, constants, and integer ranges; TensorRandoms is the sole focused public
  explicit-source eager-random owner; prefix preparation is test-fixture mechanics rather than
  product inventory. Capabilities, task, master plan, and roadmap are synchronized to Complete.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [eager glossary](../../../../glossary.md#term-eager-evaluation); [owner glossary](../../../../glossary.md#term-owner); [fixture glossary](../../../../glossary.md#term-fixture))</span>
- `./gradlew :modules:model:javadoc` passed with `BUILD SUCCESSFUL`; two tasks executed. Generated
  production Javadoc contains the public TensorRandoms page and all five public entries, and no
  strict/cyclic prefix symbol.<span data-glossary-navigation="true"> ([symbol glossary](../../../../glossary.md#term-symbol))</span>
- A Java 26 public example compiled and ran with
  `javac -cp modules/model/build/classes/java/main -d /tmp/tensor-randoms-doc-example
  /tmp/TensorRandomsExample.java` followed by `java`; it printed [`BOOL`](../../../../glossary.md#data-type--datatype), `Shape[2, 2]`, and
  `[1, 0, 1, 0]` for four scripted [Bernoulli](../../../../glossary.md#term-bernoulli-distribution) draws.
- The revised integer-range documentation example also compiled and ran; it printed [`INT32`](../../../../glossary.md#data-type--datatype),
  `Shape[3]`, and `[1, 4, 7]`, matching the documented inclusive-start/exclusive-end result.
- Reflection and `javap -public` confirmed TensorFactory has exactly 31 public and 35 total
  declared methods; TensorRandoms has exactly five public methods and no fields. Source and
  generated-Javadoc scans found exactly one public production random owner and no production
  prefix symbol or compatibility bridge.
- Source/artifact scans confirmed TensorTestData exists only under test source, no production code
  imports it, and no main-class artifact contains it. The exact changed-path inventory is the
  authorized seventeen paths: four production paths including one deletion and one creation,
  seven test paths including one creation, and six documentation/planning paths.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact))</span>
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
  construction/storage/import/constant behavior, or the semantic-value/eager-carrier boundary.<span data-glossary-navigation="true"> ([TensorProducer glossary](../../../../glossary.md#producer--tensorproducer); [ScalarValue glossary](../../../../glossary.md#typed-scalar-value--scalarvalue); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Architecture documentation needs no change because module ownership, dependency direction,
  lifecycle boundaries, and the no-service rule are unchanged. Compile API needs no change because
  eager leaves still have no compiler capture or random Operation. Training API needs no change
  because no gradient rule, parameter, optimizer, dropout, or training workflow changed. Gradle,
  architecture tests, backend conformance, integration tests, other modules, and later task specs
  likewise need no change because there is no dependency, build, cross-layer, or executable
  contract expansion.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [lifecycle glossary](../../../../glossary.md#lifecycle); [Compile glossary](../../../../glossary.md#compile); [compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [Operation glossary](../../../../glossary.md#operation); [gradient glossary](../../../../glossary.md#term-gradient); [optimizer glossary](../../../../glossary.md#term-optimizer); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [Gradle glossary](../../../../glossary.md#term-gradle); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Implementation notes

Implemented atomically within the exact seventeen-path [scope](../../../../glossary.md#term-scope). TensorFactory now exposes only the
retained 31-method construction/import/[constant](../../../../glossary.md#term-constant)/range surface; TensorRandoms is the sole public
explicit-source random [owner](../../../../glossary.md#term-owner); TensorRanges owns package-private range mechanics; TensorPopulations
is deleted; and TensorTestData owns test-only prefix [fixture](../../../../glossary.md#term-fixture) preparation.

## Completion summary

- Completed changes: narrowed TensorFactory, promoted TensorRandoms, moved range mechanics to
  TensorRanges, removed TensorPopulations, and moved prefix fixture preparation to TensorTestData.<span data-glossary-navigation="true"> ([fixture glossary](../../../../glossary.md#term-fixture))</span>
- Files changed or created: exactly the seventeen production, test, documentation, and planning
  paths authorized by this task; no additional repository path changed.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: reused the passing 58-test focused command and 715-test root checkpoint;
  model Javadoc, Java 26 example, reflection/javap, generated-page/source/artifact, Markdown,
  exact-scope, status, terminology, newline/fence/whitespace, and diff checks passed.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [artifact glossary](../../../../glossary.md#term-artifact); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: completed independently in a clean documentation-focused context.
- Documentation impact: Tensor API, glossary, capabilities, task, master plan, and roadmap now
  describe the final narrowed ownership and test-fixture boundary.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Javadoc review: TensorRandoms, TensorFactory, TensorRanges, and TensorTestData finalized; retained
  factory members and related Tensor/storage/import/constant/scalar contracts reviewed unchanged.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Glossary impact: added focused Tensor random initialization ownership and removed prefix creation
  from the public product inventory.<span data-glossary-navigation="true"> ([Tensor random initialization glossary](../../../../glossary.md#tensor-random-initialization))</span>
- Unresolved issues: None.
- Follow-up required: None.

Status: Complete
