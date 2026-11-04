# Task 0004A: [Parameter](../../../../glossary.md#term-parameter) Update and Traversal Hardening

## Status

Complete

## Goal

Close the three confirmed findings from the post-[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)-0004 code review before the first layer is
added. A downstream generic training consumer must be able to install a compatible replacement
[Tensor](../../../../glossary.md#tensor) through a discovered [`Parameter`](../../../../glossary.md#term-parameter); recursive [module](../../../../glossary.md#term-module)-tree operations must remain correct for
deep valid trees and fail on repeated [identities](../../../../glossary.md#term-identity-meanings) without relying on the Java call stack; and the
four fan-based initializer contracts must document and test their existing Java-array-size
failure.

## [Scope](../../../../glossary.md#term-scope)

- Make the existing final `io.github.pho001.synaptik.nn.module.Parameter` the public update
  capability consumed downstream. Add exactly:

  ```java
  public void replace(Tensor value)
  ```

  [`Parameter`](../../../../glossary.md#term-parameter) remains final, so the method is non-overridable without adding an interface,
  updater service, token, or training dependency. <span data-glossary-navigation="true"> ([token glossary](../../../../glossary.md#term-token); [dependency glossary](../../../../glossary.md#term-dependency))</span>`value()` remains the current exact [binding](../../../../glossary.md#term-binding).
- Preserve [`Module`](../../../../glossary.md#term-module) as the owner of parameter declaration, registration, naming, traversal, and
  the existing protected direct-name convenience <span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>`replaceParameter(String, Tensor)`. That method
  delegates to the same `Parameter.replace(Tensor)` validation after resolving the direct local
  parameter. Preserve the existing protected direct `replaceBuffer(String, Tensor)` convenience.
  [`Buffer`](../../../../glossary.md#term-buffer) receives no public replacement method.
- Define a parameter declaration [schema](../../../../glossary.md#term-schema). The package-private
  `Parameter(String name, Tensor value)` constructor remains package-private with the same
  signature; no public constructor is added. That constructor validates non-null <span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>`name`, non-null
  `value`, floating `value.descriptor().dataType()`, then
  `value.descriptor().requiresGrad() == true`, in that order. The existing protected
  `Module.parameter(name, value)` keeps its current name/namespace preflight before invoking the
  constructor. Failed declaration installs no registry entry. The parameter permanently retains
  the declaration-time exact <span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [registry glossary](../../../../glossary.md#term-registry))</span>[`DataType`](../../../../glossary.md#data-type--datatype) and [immutable](../../../../glossary.md#term-immutable) [`Shape`](../../../../glossary.md#shape) as its replacement schema.
- Define replacement [validation](../../../../glossary.md#term-validation) exactly. `Parameter.replace(value)` first rejects a null value,
  then requires the replacement's exact `DataType` to equal the declaration-time type, its Shape
  to be structurally equal to the declaration-time Shape, and `requiresGrad == true`, in that
  order. Validation failure is an `IllegalArgumentException` and leaves the current binding
  unchanged. Successful replacement retains the exact supplied Tensor reference while preserving
  the <span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>`Parameter` wrapper and local name [identity](../../../../glossary.md#term-identity-meanings).
- Remove the package-private `Parameter.replaceValue(Tensor)` mutation primitive. Module and
  downstream callers use the one validated public `replace(Tensor)` entry so no internal bypass
  can drift the declaration schema.
- Do not freeze or compare layout, host storage, provenance, label, Tensor identity, or any other
  descriptor or lifecycle fact. A compatible replacement may therefore have different resolved
  versus unresolved layout, storage presence, expression provenance, label, and Tensor identity.
  The declaration schema is not a public descriptor/schema API.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [host storage glossary](../../../../glossary.md#host-storage); [provenance glossary](../../../../glossary.md#provenance); [descriptor glossary](../../../../glossary.md#term-descriptor); [lifecycle glossary](../../../../glossary.md#lifecycle); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Replace recursive Java calls in `Module.parametersRecursively()`,
  `Module.buffersRecursively()`, and the shared `train()`/`eval()` traversal with iterative
  depth-first search using an explicit stack. Preserve the existing exact preorder and child
  registration order. Parameter discovery visits each module's direct parameters before its
  children; buffer discovery visits each module's direct buffers before its children; mode
  propagation collects the receiver before descendants.
- Apply identity-repeat defense to both recursive discovery operations as well as mode traversal.
  A repeated `Module` identity, whether from a cycle or a shared-child corruption, throws
  `IllegalStateException`. Discovery builds only a local result and returns nothing on failure.
  `train()` and `eval()` retain their existing two-phase all-or-nothing contract: complete
  identity preflight/collection succeeds before any mode assignment occurs.
- Use no arbitrary depth limit. Avoid constructing and retaining a complete prefix String at
  every empty level of a deep chain. Prefer stack frames plus a mutable path-segment stack, or an
  equivalently bounded iterative representation, and construct a dot-qualified String only when
  emitting an actual parameter or buffer entry. The unavoidable final qualified key is not an
  intermediate-prefix allocation requirement.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable))</span>
- Add one deep valid-chain regression, constructed without quadratic ancestor checking, that
  exercises `parametersRecursively()`, `buffersRecursively()`, `train()`, and `eval()` beyond a
  practical recursive-call-stack depth and proves no `StackOverflowError`. The test must also
  verify exact root-relative paths, wrapper identities, and final modes. It must not establish a
  supported maximum depth.
- Extend defensive corruption coverage so both recursive discovery methods reject repeated
  identity. Preserve and extend the existing assertion that a failed mode change leaves every
  already visited module in its prior mode.
- Add a black-box test in a test package different from
  `io.github.pho001.synaptik.nn.module`. It obtains only a root `Module`, traverses
  `parametersRecursively()`, and replaces every discovered parameter through public
  `Parameter.replace(Tensor)` without knowing the concrete module or layer type. The test proves
  stable qualified-name order, exact wrapper identity, compatible replacement, and rejection of
  data-type, Shape, gradient-eligibility, and null mismatches without partial mutation.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient))</span>
- Update the four fan initializer Javadocs (`glorotNormal`, `glorotUniform`,
  `kaimingReluNormal`, and `kaimingReluUniform`) so `@throws IllegalArgumentException` explicitly
  covers a positive fully static rank-two Shape whose element count exceeds<span data-glossary-navigation="true"> ([rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [element glossary](../../../../glossary.md#term-element))</span>
  `Integer.MAX_VALUE`. Add a [regression](../../../../glossary.md#term-regression) using `Shape.of(46_341, 46_341)` for all four methods.
  Each call must fail before a random-source draw, destination allocation, or Tensor-ID
  allocation, preserving the existing Model failure order and side effects.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- Update affected NN Javadoc, the training API status text, glossary, task record, and NN master
  plan. After executable implementation and focused validation, use a separate clean
  documentation-focused context to inspect the final diff and finalize those artifacts.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Javadoc glossary](../../../../glossary.md#term-javadoc); [artifacts glossary](../../../../glossary.md#term-artifact))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- An optimizer algorithm, optimizer base type, SGD, Adam, AdamW, parameter group, training
  session, training step, gradient publication, or update orchestration.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer); [SGD glossary](../../../../glossary.md#term-stochastic-gradient-descent-sgd); [Adam, AdamW glossary](../../../../glossary.md#adaptive-optimizer--adam-adamw); [parameter group glossary](../../../../glossary.md#parameter-group--parametergroup); [training session glossary](../../../../glossary.md#training-session--trainingsession); [gradient glossary](../../../../glossary.md#term-gradient); [publication glossary](../../../../glossary.md#term-publication))</span>
- A batch, tree-wide, transactional, versioned, compare-and-set, rollback, or checkpoint update
  API; state dictionaries, serialization, or file formats.<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [compare-and-set glossary](../../../../glossary.md#term-compare-and-set-cas); [rollback glossary](../../../../glossary.md#term-rollback); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [serialization glossary](../../../../glossary.md#term-serialization))</span>
- Thread safety, locking, volatile fields, atomic references, unsynchronized visibility promises,
  concurrent snapshots, or a multi-parameter consistency protocol. Callers still coordinate
  replacement with forward construction, traversal, and other mutation.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [parameter glossary](../../../../glossary.md#term-parameter); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Public replacement of [`Buffer`](../../../../glossary.md#term-buffer), a buffer update service, a public module parent, detach,
  rename, reparenting, shared children, or a supported corruption mechanism.
- Public parameter-schema accessors, layout/storage/provenance/label compatibility, shape
  coercion, data-type conversion, gradient creation, or Tensor mutation.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance); [shape glossary](../../../../glossary.md#shape); [Tensor glossary](../../../../glossary.md#tensor))</span>
- A recursive path replacement [API](../../../../glossary.md#term-application-programming-interface-api) on [`Module`](../../../../glossary.md#term-module); the existing protected Module conveniences remain
  direct-name only.
- A [`Linear`](../../../../glossary.md#term-linear-meanings) layer, new initializer, Xavier alias, configurable fan policy, numerical execution,
  compiler, runtime, prepare, Engine, or backend behavior.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend))</span>
- Gradle, included-project, dependency, architecture-contract, ADR, architecture-test, global
  roadmap, CPU, or unrelated refactoring changes.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [CPU glossary](../../../../glossary.md#term-cpu))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>: immutable Tensor identity/provenance,
  NN ownership, downstream training consumption, extension dependency direction, and testing
  discipline.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [Tensor glossary](../../../../glossary.md#tensor); [identity glossary](../../../../glossary.md#term-identity-meanings); [provenance glossary](../../../../glossary.md#provenance); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools))</span>
- [ADR 0007: Neural-network module and training boundary](../../../../design/decisions/0007-neural-network-module-and-training-boundary.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [module glossary](../../../../glossary.md#term-module))</span>.
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>.
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>.
- [Training API](../../../../api/training-api.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>.
- [NN master plan](../master-plan.md).
- [Task 0002](0002-module-tree-ownership-and-recursive-mode-propagation.md),
  [Task 0003](0003-validated-parameter-and-buffer-binding-replacement.md), and
  [Task 0004](0004-explicit-eager-parameter-initializers.md).
- [Planning Guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>.
- [Documentation rules](../../../../developer-guide/documentation-rules.md), the
  [General profile](../../../../developer-guide/documentation/general-style.md), the
  [API/Javadoc profile](../../../../developer-guide/documentation/api-and-javadoc-style.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>, and
  the [Planning profile](../../../../developer-guide/documentation/planning-style.md).

## Architecture constraints

- `extensions/nn` continues to depend only on `modules/model`; it must not import training,
  compiler, runtime, prepare, Engine, or a concrete backend.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
- NN owns the declared parameter capability. A downstream training consumer may invoke that
  capability, but optimizer mathematics and update orchestration remain owned by<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
  `extensions/training`. No reverse [dependency](../../../../glossary.md#term-dependency) or training-owned [parameter](../../../../glossary.md#term-parameter) type is introduced.
- Tensor identity, descriptor, and provenance remain immutable. Replacement changes only which
  exact Tensor a stable NN-owned <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [immutable glossary](../../../../glossary.md#term-immutable))</span>`Parameter` returns on later `value()` calls. A Tensor obtained
  earlier and expressions already built from it remain unchanged.
- Parameter declaration is trainable by contract: the declared [Tensor](../../../../glossary.md#tensor) is floating and has
  `requiresGrad == true`. Replacement retains that trainable schema and exact declaration-time
  data type and Shape while deliberately ignoring execution/storage facts.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [data type glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape))</span>
- Module discovery and mode propagation remain deterministic structural operations over the
  exclusively owned tree. Iterative traversal is an implementation hardening, not a new tree,
  naming, ordering, concurrency, or execution contract.<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- The authorized NN parallel exception is implementation-order only. Do not touch the dirty CPU
  work or global roadmap, and do not reinterpret this remediation as an architecture change.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.nn.module` — owns [`Module`](../../../../glossary.md#term-module), [`Parameter`](../../../../glossary.md#term-parameter), [`Buffer`](../../../../glossary.md#term-buffer), declaration
  schema, replacement, traversal, and mode propagation.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>
- `io.github.pho001.synaptik.nn.initialization` — owns the existing fan initializer Javadocs and
  their focused regression.<span data-glossary-navigation="true"> ([regression glossary](../../../../glossary.md#term-regression))</span>
- `io.github.pho001.synaptik.model.tensor`, `.datatype`, and `.shape` — supply immutable Tensor
  descriptor facts and existing eager random failure behavior; Model source does not change.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [Tensor descriptor glossary](../../../../glossary.md#tensor-descriptor); [eager glossary](../../../../glossary.md#term-eager-evaluation); [Model glossary](../../../../glossary.md#term-model))</span>

Packages added or changed:

- `io.github.pho001.synaptik.nn.consumer` under NN test sources only — black-box downstream-style
  use of the public Module/Parameter surface. No production package is added.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>

Type placement:

- `io.github.pho001.synaptik.nn.module.Parameter` — existing final public capability whose
  package-private constructor captures the declaration schema and whose public <span data-glossary-navigation="true"> ([captures glossary](../../../../glossary.md#term-capture-meanings))</span>`replace` installs
  one compatible exact Tensor.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- `io.github.pho001.synaptik.nn.module.Module` — existing owner of declarations and iterative
  tree traversal; its protected direct parameter convenience delegates to <span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>`Parameter.replace`.
- `io.github.pho001.synaptik.nn.initialization.ParameterInitializers` — existing initializer
  surface whose four fan-method failure Javadocs become complete.
- `io.github.pho001.synaptik.nn.consumer.ParameterUpdateContractTest` — black-box proof that a
  generic consumer needs neither package-private access nor concrete layer knowledge.
- `io.github.pho001.synaptik.nn.module.ParameterAndBufferTest` — declaration-schema, public API,
  wrapper-identity, and Buffer-no-public-replacement coverage.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `io.github.pho001.synaptik.nn.module.ModuleTreeTest` — iterative order, repeated-identity, deep
  traversal, and all-or-nothing mode regressions.
- `io.github.pho001.synaptik.nn.initialization.LinearWeightInitializersTest` — positive rank-two
  over-limit no-draw/no-ID regression for all four fan methods.<span data-glossary-navigation="true"> ([rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>

## Affected files

Expected production files:

- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Parameter.java`.
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Module.java`.
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/initialization/ParameterInitializers.java`.

Expected test files:

- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/module/ParameterAndBufferTest.java`.
- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/module/ModuleTreeTest.java`.
- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/initialization/LinearWeightInitializersTest.java`.
- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/consumer/ParameterUpdateContractTest.java`.

Expected documentation and [planning](../../../../glossary.md#term-planning) files:

- `docs/api/training-api.md` — replace the now-stale claim that no parameter mutation rule is
  stable with the bounded NN-owned public replacement capability; do not invent an optimizer API.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- `docs/glossary.md` — update the existing NN parameter entry from protected Module-only
  replacement to the public compatible Parameter replacement and declaration schema.<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module); [schema glossary](../../../../glossary.md#term-schema))</span>
- `docs/planning/extensions/nn/master-plan.md`.
- this task specification.

Explicit review/no-change candidates:

- `ARCHITECTURE.md`, [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0007, [dependency](../../../../glossary.md#term-dependency) rules, [training graph](../../../../glossary.md#training-graph), architecture tests, [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api),
  `Buffer.java`, NN package documentation, Gradle/build files, and the global roadmap. They already
  establish the correct ownership and dependency boundary or are unaffected by the bounded API.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most:

- three production Java files;
- four [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) test files; and
- the four documentation/[planning](../../../../glossary.md#term-planning) files listed above.

No other file may enter the task diff. If implementation needs another production type/package,
a new [module](../../../../glossary.md#term-module) edge, public [schema](../../../../glossary.md#term-schema) [carrier](../../../../glossary.md#term-storage-carrier), [checkpoint](../../../../glossary.md#term-checkpoint)/[transaction](../../../../glossary.md#term-transaction)/[concurrency](../../../../glossary.md#term-concurrency) mechanism,
[architecture test](../../../../glossary.md#term-architecture-test), [Model](../../../../glossary.md#term-model) change, or more files, stop and propose a separate follow-up task.

## Acceptance criteria

- The existing final [`Parameter`](../../../../glossary.md#term-parameter) exposes exactly one public [binding](../../../../glossary.md#term-binding) mutation method with [signature](../../../../glossary.md#term-signature)
  `void replace(Tensor value)`. It retains `name()` and `value()`, adds no public constructor or
  schema accessor, and remains neither a Tensor subtype nor an optimizer type.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [Tensor glossary](../../../../glossary.md#tensor); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- The package-private `Parameter(String, Tensor)` signature and current Module declaration entry
  remain source-compatible. The constructor checks null name, null value, floating type, then
  `requiresGrad == true`; Module's existing local-name/namespace validation still precedes those
  constructor checks. Schema failures are <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>`IllegalArgumentException` and install no state. Every
  current valid NN declaration and initializer test remains valid.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- A parameter [captures](../../../../glossary.md#term-capture-meanings) the declaration-time exact [`DataType`](../../../../glossary.md#data-type--datatype) and immutable Shape. Replacement
  checks null, exact data type, structural Shape equality, then <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [Shape glossary](../../../../glossary.md#shape); [data type glossary](../../../../glossary.md#data-type--datatype))</span>`requiresGrad == true`; each
  failure leaves the exact old Tensor current. A successful call retains and returns the exact new
  Tensor while preserving wrapper/name identity and discovery order.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Replacement accepts differences in Tensor identity, layout resolution/geometry, host storage,
  provenance, and label. No compatibility check or promise is introduced for those facts.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [host storage glossary](../../../../glossary.md#host-storage); [provenance glossary](../../../../glossary.md#provenance))</span>
- `Module.replaceParameter(String, Tensor)` stays protected, final, direct-name only, and delegates
  compatible replacement through the resolved direct `Parameter`. `replaceBuffer` retains its
  current protected direct behavior. [`Buffer`](../../../../glossary.md#term-buffer) has no public/protected `replace`, `update`, or
  [`rebind`](../../../../glossary.md#capture-graph-ports-and-reconstruction) method.
- A black-box test in `io.github.pho001.synaptik.nn.consumer` updates discovered parameters from a
  root [`Module`](../../../../glossary.md#term-module) without concrete layer knowledge or access to the module package. It proves the
  downstream architecture use case and the complete replacement validation matrix.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Recursive parameter and buffer discovery use iterative depth-first traversal, preserve exact
  child-registration/path order, return insertion-ordered unmodifiable structural snapshots, and
  reject any repeated module identity with `IllegalStateException` before returning a result.
- `train()` and `eval()` use an iterative identity-preflight collection and assign no mode until
  that collection is complete. Their preorder and existing immutable <span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight))</span>`ForwardContext` behavior
  remain unchanged.
- One bottom-up-constructed valid chain beyond a practical call-stack depth executes parameter
  discovery, buffer discovery, `eval()`, and `train()` without `StackOverflowError` or an arbitrary
  depth limit. It verifies leaf qualified paths and all modes. Empty intermediate levels do not
  require one retained qualified-prefix String per depth.<span data-glossary-navigation="true"> ([leaf glossary](../../../../glossary.md#term-leaf))</span>
- Existing reflective malformed-tree coverage proves repeated-identity failure for parameters,
  buffers, `train()`, and `eval()`; failed mode propagation leaves all prior modes unchanged.
- All four fan methods document the Java-array-limit `IllegalArgumentException`. For
  `Shape.of(46_341, 46_341)`, each method fails before a source draw or Tensor-ID allocation; the
  test proves both side effects remain unchanged.
- Javadoc accurately documents declaration schema, compatible replacement, exact-reference and
  old-expression behavior, ignored Tensor facts, validation order/failures, direct Module
  delegation, iterative traversal ordering, identity defense, all-or-nothing mode behavior, and
  lack of thread safety, versioning, transactions, checkpoints, or optimizer algorithms.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [checkpoints glossary](../../../../glossary.md#term-checkpoint))</span>
- The training API and glossary describe only the now-stable NN-owned replacement capability and
  keep optimizer signatures/orchestration planned. Architecture, dependency, Tensor API, and
  Buffer documentation receive explicit reasoned no-change conclusions.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- A separate documentation-focused clean-context pass finalizes affected Javadoc, API text,
  glossary, planning evidence, links, terminology, and generated Javadoc in the same overall
  change.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- No architecture, dependency, Gradle, Model, CPU, global-roadmap, layer, optimizer, checkpoint,
  execution, or backend behavior changes.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [Model glossary](../../../../glossary.md#term-model); [CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation pass runs one final affected-[module](../../../../glossary.md#term-module) command after executable Java stabilizes:

```bash
./gradlew :extensions:nn:test
```

The test report must record suite/test counts and zero failures/errors/skips, including the
black-box consumer, declaration/replacement validation, iterative deep-tree and corruption
coverage, and fan over-limit regressions.

Documentation pass runs after final [Javadoc](../../../../glossary.md#term-javadoc) and explanatory text edits:

```bash
./gradlew :extensions:nn:javadoc
git diff --check
```

The documentation pass also validates local Markdown links and heading anchors in the changed
task/master/[API](../../../../glossary.md#term-application-programming-interface-api)/glossary files, inspects the generated `Module`, [`Parameter`](../../../../glossary.md#term-parameter), and initializer
pages, confirms exact package/type placement and the eleven-file maximum, and checks that no dirty
[CPU](../../../../glossary.md#term-cpu) or global-roadmap path entered this task's diff. It reuses the successful implementation test
unless executable Java changes afterward; if it changes executable behavior, it reruns the
focused [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) test once and records the reason.

Repository-wide and architecture-test validation are deferred to the NN capability [checkpoint](../../../../glossary.md#term-checkpoint) or
[CI](../../../../glossary.md#term-continuous-integration-ci). This task changes one module's API and implementation without changing a [Gradle](../../../../glossary.md#term-gradle) edge,
architecture boundary, or shared cross-module implementation. The black-box test automates the
cross-package API risk that motivated the review finding.

## Dependencies

- [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0001, 0002, 0003, and 0004 are Complete.
- [Model](../../../../glossary.md#term-model) [`Tensor`](../../../../glossary.md#tensor), `TensorDescriptor`, [`Shape`](../../../../glossary.md#shape), [`DataType`](../../../../glossary.md#data-type--datatype), `TensorFactory`, and `TensorRandoms`
  contracts are Complete and provide every schema and failure fact used here.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>
- Accepted [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0007 requires generic downstream [optimizer](../../../../glossary.md#term-optimizer) consumption of NN-owned [parameters](../../../../glossary.md#term-parameter).
- The user-authorized NN parallel exception in the NN master plan remains in force and the task's
  files do not overlap the active dirty CPU work.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>

## Follow-up tasks

- [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0005: add the first [`Linear`](../../../../glossary.md#term-linear-meanings) layer only after this remediation is Complete.
- `extensions/training`: define optimizer algorithms, update sequencing, gradient/result mapping,
  and synchronization when its own frontier is reached; it should consume <span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer); [gradient glossary](../../../../glossary.md#term-gradient); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>`Parameter.replace`
  rather than introduce another mutation owner.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- A later checkpoint/state-dictionary task must independently define multi-binding validation,
  atomicity, persistence, and file format. It must not infer transaction semantics from this
  individual replacement capability.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [binding glossary](../../../../glossary.md#term-binding); [validation glossary](../../../../glossary.md#term-validation); [atomicity glossary](../../../../glossary.md#term-atomicity); [persistence glossary](../../../../glossary.md#term-persistence); [transaction glossary](../../../../glossary.md#term-transaction))</span>

## Architecture impact

Expected impact: None.

This task closes an implementation gap in the already accepted [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0007 boundary: [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) owns a
[parameter](../../../../glossary.md#term-parameter) capability that downstream training can consume generically. It changes no [module](../../../../glossary.md#term-module) edge,
[owner](../../../../glossary.md#term-owner), [dependency](../../../../glossary.md#term-dependency) rule, or [Tensor](../../../../glossary.md#tensor) [invariant](../../../../glossary.md#term-invariant). Iterative traversal and initializer [Javadoc](../../../../glossary.md#term-javadoc) are
implementation/contract hardening. If implementation requires architecture or dependency changes,
stop and report the conflict instead of editing architecture documents.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are working in the Synaptik repository.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/developer-guide/documentation-rules.md, and
docs/planning/extensions/nn/tasks/0004a-parameter-update-and-traversal-hardening.md in full.
Implement that task exactly as specified. Preserve unrelated dirty CPU and global-roadmap work;
do not implement NN 0005 or any out-of-scope behavior, commit, or push. Stop and report an
architecture or maximum-scope conflict.

After executable implementation and the single final focused NN test run, hand the complete diff
and exact test evidence to a separate documentation-focused clean context. That pass must inspect
the actual source/tests, follow the documentation rules and selected profiles, finalize affected
Javadoc, training API text, glossary impact, planning evidence, and documentation validation in
the same overall change, and avoid repeating successful Java tests unless executable behavior
changes or a concrete risk is recorded.

Update this task with local decisions, known limitations, exact validation evidence,
implementation notes, completion summary, and final status only after that pass. Do not mark it
Complete before every acceptance criterion passes.
```

## Local decisions

- The existing final [`Parameter`](../../../../glossary.md#term-parameter), not a new updater abstraction, is the capability handed to a
  downstream consumer by recursive discovery. Public replacement therefore follows the exact
  small `void replace(Tensor)` surface.
- Declaration-time floating type and `requiresGrad == true` define what makes a `Parameter`
  trainable. Exact data type and structural Shape become stable private replacement schema;
  layout, storage, provenance, label, and Tensor identity deliberately remain replaceable.<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape); [schema glossary](../../../../glossary.md#term-schema); [layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance); [Tensor glossary](../../../../glossary.md#tensor); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- The existing protected `Module.replaceParameter` remains for a concrete module's own direct
  state transitions and delegates to the same public capability. This preserves source
  compatibility with task 0003 while removing its downstream-consumer limitation.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>
- The former package-private `replaceValue` entry is removed so both Module and downstream
  callers pass through one validation and mutation path.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Buffers remain module-owned persistent state with no public replacement capability. An optimizer
  discovers and updates parameters only.<span data-glossary-navigation="true"> ([Buffers glossary](../../../../glossary.md#term-buffer); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- Iterative traversal preserves the current externally observed preorder and adds identity
  defense to discovery. Mode propagation remains two-phase so malformed trees cannot expose a
  partial requested mode.
- Qualified path text is produced only for emitted state entries. A deep empty chain uses explicit
  traversal frames/path segments rather than allocating a String for every intermediate prefix.

## Known limitations

- Replacement updates one parameter at a time and is not thread-safe. It has no version,
  transaction, rollback, checkpoint, or cross-parameter consistency guarantee.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [thread-safe glossary](../../../../glossary.md#term-thread-safe); [transaction glossary](../../../../glossary.md#term-transaction); [rollback glossary](../../../../glossary.md#term-rollback); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Exact declaration-time data type and structural Shape cannot be changed through replacement.
  Callers must construct a new module/parameter if a layer's trainable schema changes.<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [Shape glossary](../../../../glossary.md#shape); [module glossary](../../../../glossary.md#term-module); [schema glossary](../../../../glossary.md#term-schema))</span>
- The public capability does not compute an optimizer update, map gradients, execute a graph, or
  coordinate with concurrent forward construction.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer); [gradients glossary](../../../../glossary.md#term-gradient); [graph glossary](../../../../glossary.md#graph); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Iterative traversal removes call-stack depth dependence but does not promise a maximum tree
  size; heap capacity and final qualified-path size remain ordinary environmental limits.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap))</span>
- Buffer replacement remains available only to a declaring subclass through the existing
  protected direct-name Module method.<span data-glossary-navigation="true"> ([Buffer glossary](../../../../glossary.md#term-buffer))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

Implementation context `/root/nn_0004a_implementation` completed the executable change. Its first
sandboxed `./gradlew :extensions:nn:test` attempt could not open the existing [Gradle](../../../../glossary.md#term-gradle) wrapper [cache](../../../../glossary.md#term-cache)
lock under `/Users/phujka/.gradle`; rerunning the same focused command with approved cache access
passed. After the final executable and test edits stabilized, the authoritative final
`./gradlew :extensions:nn:test` passed on 2026-08-13 (`BUILD SUCCESSFUL`, 5 actionable tasks: 2
executed and 3 up-to-date). The XML reports contain 7 suites and 33 tests with zero failures, zero
errors, and zero skips. They include the cross-package consumer suite, declaration/replacement
[schema](../../../../glossary.md#term-schema) and [API](../../../../glossary.md#term-application-programming-interface-api)-surface coverage, 20,000-level iterative traversal and malformed-repeat coverage,
and all four fan over-limit regressions.

Implementation `git diff --check` passed with no output. Because the current [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0004/0004A task
and initialization files are still untracked in the shared working tree, the implementation pass
also ran `git diff --no-index --check /dev/null <file>` for the 0004A task, the changed initializer
source and test, and the new consumer test; every check passed. Source scans confirmed that
`Parameter.replace(Tensor)` is the sole public [parameter](../../../../glossary.md#term-parameter) mutation entry, [Buffer](../../../../glossary.md#term-buffer) retains only its
package-private [Module](../../../../glossary.md#term-module)-owned primitive, and no recursive helper call remains in the three module-
tree operations.

Documentation context `/root/nn_0004a_docs` independently reviewed the complete final change
against `AGENTS.md`, `ARCHITECTURE.md`, the current architecture index, [planning](../../../../glossary.md#term-planning) guide,
documentation rules, General/API-and-[Javadoc](../../../../glossary.md#term-javadoc)/Planning/Example profiles, NN master plan and tasks
0001–0004A, [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0007, [training graph](../../../../glossary.md#training-graph), [dependency](../../../../glossary.md#term-dependency) rules, training and [Tensor](../../../../glossary.md#tensor) API references,
glossary, all final NN source/tests, and the relevant [Model](../../../../glossary.md#term-model) [Tensor descriptor](../../../../glossary.md#tensor-descriptor), [Shape](../../../../glossary.md#shape), data-type,
[constant](../../../../glossary.md#term-constant), and random contracts. It found no executable behavior defect and changed only Javadoc,
training API, glossary, and NN planning records. No executable Java changed after the final
focused test [run](../../../../glossary.md#term-run), so that successful 7-suite/33-test evidence was reused rather than repeated.

The documentation pass finalized the declaration/replacement schema and generic downstream
capability, exact wrapper/name and old-expression behavior, deliberately unfrozen Tensor facts,
single-[binding](../../../../glossary.md#term-binding)/threading/[checkpoint](../../../../glossary.md#term-checkpoint) boundaries, Buffer distinction, iterative traversal order and
[identity](../../../../glossary.md#term-identity-meanings) defense, two-phase mode assignment, and all four fan initializer Java-array-limit
failures. `./gradlew :extensions:nn:javadoc` passed on 2026-08-13 (`BUILD SUCCESSFUL`, 3 actionable
tasks: 2 executed and 1 up-to-date). Generated `Parameter`, `Module`, and
`ParameterInitializers` pages were inspected and rendered the required contracts, including all
four over-limit `IllegalArgumentException` entries.

A targeted repository-local Markdown validator passed the training API, glossary, NN master
plan, and this task: 4 files, 333 local links, 291 heading anchors, balanced backtick/tilde fences,
and terminal newlines. `javap -public` confirmed that final `Parameter` exposes exactly `name`,
`value`, and `replace(Tensor)`; `Buffer` exposes only `name` and `value`; Module and the eight
initializer methods retain their planned public surfaces. Production-import scans found only
Model, NN-module, and [JDK](../../../../glossary.md#term-java-development-kit-jdk) imports and no training/[compiler](../../../../glossary.md#term-compiler)/[runtime](../../../../glossary.md#term-runtime)/[prepare](../../../../glossary.md#prepare)/[Engine](../../../../glossary.md#term-engine)/[backend](../../../../glossary.md#backend) edge.

Exact [scope](../../../../glossary.md#term-scope) inspection confirmed the planned eleven task-owned paths: three production Java
files, four NN tests, training API, glossary, NN master plan, and this task. The dirty [CPU](../../../../glossary.md#term-cpu) 0007
work and global roadmap remained unrelated and untouched by this pass. Final `git diff --check`
passed with no output, and a read-only check of all 14 untracked text files in the shared tree
found no trailing whitespace or missing terminal newline.

Repository-wide and architecture tests remain deferred exactly as specified because no Gradle
edge, architecture boundary, or cross-module implementation changed. `ARCHITECTURE.md`, ADR 0007,
training graph, dependency rules, architecture tests, Tensor API, Model API/source, `Buffer`
Javadoc/source, NN build/package documentation, and Gradle files require no change: they already
express the same [ownership](../../../../glossary.md#term-ownership-meanings)/direction or are unaffected by this bounded public capability and
iterative implementation. The unrelated modified global roadmap is explicitly outside this task.

## Implementation notes

The clean implementation context added the public [schema](../../../../glossary.md#term-schema)-validated `Parameter.replace(Tensor)`
capability and removed the former [parameter](../../../../glossary.md#term-parameter) package-internal mutation bypass. Parameter
declaration now requires a floating [gradient](../../../../glossary.md#term-gradient)-eligible [Tensor](../../../../glossary.md#tensor) and privately retains its exact [data
type](../../../../glossary.md#data-type--datatype) plus [immutable](../../../../glossary.md#term-immutable) [Shape](../../../../glossary.md#shape). Protected direct [Module](../../../../glossary.md#term-module) replacement delegates through the same public
[validation](../../../../glossary.md#term-validation); [Buffer](../../../../glossary.md#term-buffer) remains unchanged and has no public replacement surface.

Module parameter discovery, buffer discovery, and mode [preflight](../../../../glossary.md#term-preflight) now use explicit-stack DFS with
[identity](../../../../glossary.md#term-identity-meanings) tracking and the existing preorder. State paths are assembled only when an entry is
emitted. Focused tests cover a bottom-up 20,000-level chain and defensive repeated-identity
failures. The implementation drafted the four delegated positive Java-array-limit failure
contracts and the documentation pass finalized them; their [regression](../../../../glossary.md#term-regression) proves no random draw or
Tensor-ID allocation for `Shape.of(46_341, 46_341)`.

The complete task stayed inside the planned three production, four test, and four documentation/
[planning](../../../../glossary.md#term-planning) files. The distinct clean documentation context finalized the two explanatory files,
all affected Javadocs, and planning evidence without changing executable Java.

## Review-finding closure

- Finding 1 — downstream generic training cannot install updates: closed by public schema-
  validated <span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>`Parameter.replace`, the black-box consumer suite, and finalized downstream-boundary
  documentation.
- Finding 2 — deep valid trees can overflow the Java call stack: executable closure passed for all
  four iterative operations plus deep and corruption regressions; generated Javadoc records the
  explicit-stack, identity-defense, and two-phase mode contracts.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow); [Javadoc glossary](../../../../glossary.md#term-javadoc); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Finding 3 — fan initializer Javadocs omit Java-array-size failure: closed by all four finalized
  failure contracts and the passing positive rank-two no-draw/no-ID regression.<span data-glossary-navigation="true"> ([rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [regression glossary](../../../../glossary.md#term-regression))</span>

## Completion summary

- Completed changes: Implemented the bounded parameter declaration/replacement schema, public
  downstream update capability, iterative identity-defended discovery and mode traversal, and fan
  initializer Java-array-limit regression and finalized contracts.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [identity glossary](../../../../glossary.md#term-identity-meanings); [regression glossary](../../../../glossary.md#term-regression))</span>
- Files changed or created: The planned three production Java files, four NN test files, training
  API, glossary, NN master plan, and this task record.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Tests and [validation](../../../../glossary.md#term-validation): Final `./gradlew :extensions:nn:test` passed 7 suites/33 tests with no
  failures, errors, or skips; NN Javadoc, generated-page inspection, Markdown links/anchors/fences,
  public surface, imports, exact eleven-path scope, tracked diff whitespace, and all untracked text
  whitespace/newline checks passed.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: `/root/nn_0004a_docs` independently finalized affected Javadoc,
  API reference, glossary, and planning evidence under the selected documentation profiles.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: Updated the training API and existing NN glossary entry. Architecture,
  ADR, training graph, dependency rules/tests, Tensor and Model APIs, Buffer, build files, CPU, and
  global roadmap need no task-owned change for the reasoned boundaries recorded above.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [training graph glossary](../../../../glossary.md#training-graph); [dependency glossary](../../../../glossary.md#term-dependency); [Tensor glossary](../../../../glossary.md#tensor); [Model glossary](../../../../glossary.md#term-model); [Buffer glossary](../../../../glossary.md#term-buffer); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: Finalized [`Module`](../../../../glossary.md#term-module), [`Parameter`](../../../../glossary.md#term-parameter), and all four fan-initializer failure contracts;
  generated Javadoc passed and was inspected.
- Glossary impact: Replaced stale Module-only unvalidated parameter wording with the public
  declaration-schema capability and documented iterative discovery/mode failure behavior. No new
  glossary heading was needed.
- Unresolved issues: None.
- Follow-up required: None.

Status: Complete
