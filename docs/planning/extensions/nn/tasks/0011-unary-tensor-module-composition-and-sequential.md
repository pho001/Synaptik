# Task 0011: Unary Tensor Module Composition and Sequential

## Status

Complete

## Goal

Add the smallest type-safe NN composition contract justified by one real consumer: an immutable
ordered `Sequential` module that owns unary Tensor-to-Tensor child modules and forwards one Tensor
through them from left to right.

The shared contract must preserve both facts at compile time: every accepted element is a
`Module`, so existing exclusive child ownership, recursive mode propagation, state discovery, and
state-dictionary paths apply; and every accepted element has exactly `Tensor forward(Tensor)`, so
composition needs no cast, reflection, bridge accessor, method handle, or erased generic facade.

This task is declarative Tensor-expression composition. It does not evaluate a Tensor, calculate a
numerical result, infer a whole pipeline ahead of each child, capture or optimize a graph, define a
gradient, compile, prepare, lower, select a backend, allocate runtime storage, or execute work.

## Scope

- Add public abstract
  `io.github.pho001.synaptik.nn.module.UnaryTensorModule extends Module` with one protected
  constructor and one public abstract `Tensor forward(Tensor input)` contract.
- Change only the three current mode-insensitive Tensor-to-Tensor layers to extend that base:
  `Linear`, `LayerNorm`, and `Embedding`. Their constructors, declared fields, state, accessors,
  validation, Model delegation, expression ordering, failure effects, and final class status remain
  unchanged.
- Add final public `io.github.pho001.synaptik.nn.module.Sequential extends UnaryTensorModule`.
  Its sole constructor accepts `List<? extends UnaryTensorModule>`.
- Snapshot the constructor list in encounter order, permanently register each exact child under its
  zero-based canonical decimal index (`"0"`, `"1"`, and so on), and retain one private immutable
  typed list for forward traversal. The inherited `children()` map is the sole public structural
  accessor.
- Extend `Module` with one package-private, final indexed-child registration primitive used only by
  `Sequential`. It validates the complete candidate snapshot before changing the receiving
  module's child map or any candidate's parent, then installs all children in numeric order. Refactor
  private validation shared with `child(name, child)` only as needed; do not change the existing
  protected method's contract.
- Permit an empty list. An empty `Sequential.forward(input)` validates non-null input and returns
  that exact input reference without creating a Tensor, producer, wrapper, or identifier.
- Define non-empty forward as an exact left fold: call child zero with the supplied input, call each
  later child with the exact prior output, and return the exact final output. `Sequential` itself
  creates no Tensor or expression wrapper.
- Preserve ordinary partial-expression semantics. If a later child throws, earlier successful child
  calls and any Tensor expressions or identifiers they created remain; `Sequential` performs no
  rollback. A child that violates the non-null result contract is rejected at that position before a
  later child is called.
- Cover construction atomicity, empty identity, ownership and numeric paths, nested composition,
  actual participating layers, forward order/reference flow, mode propagation, replacement and
  state-dictionary behavior, and failure effects with focused NN tests.
- Update affected public and package Javadocs, the Training API's current NN description, the NN
  glossary entry, and the NN planning records in the same overall implementation change. Finalize
  substantive documentation in a separate clean documentation-focused context.

## Out of scope

- Adding `forward` to `Module`, making every module unary, or renaming the existing base.
- A public forward interface, sealed interface, adapter, wrapper, method-reference registry,
  service/facade, reflective dispatch, or unsafe cast.
- A generic `Layer`, `Block`, `Container`, `Pipeline`, `Chain`, `Manager`, `Service`, or functional
  namespace.
- Generic input/output type parameters, tuples, multiple inputs or outputs, optional values,
  heterogeneous result carriers, automatic adaptation, or a context/state erasure mechanism.
- Changing `BatchNorm`. Its exact `forward(Tensor, ForwardContext)` contract selects evaluation or
  training and training installs two next-statistic buffer bindings; it is not unary Tensor-only.
- Changing `Dropout` or `DropoutForwardResult`. Dropout requires explicit `GraphRngState` and
  `ForwardContext` and returns output plus next state; it must remain outside this contract.
- A contextual or state-threading sequential container. Do not implicitly call
  `Module.forwardContext()`, synthesize a context, retain graph RNG state, discard a result carrier,
  or hide buffer transitions.
- Adding `BatchNorm`, `Dropout`, arbitrary `Module`, or non-`Module` functions to the accepted child
  type.
- Varargs, array, stream, iterable, builder, factory, named-child map, add/remove/replace/move,
  public module-list accessor, index accessor, size method, iterator, or mutable structure API.
- Rejecting empty composition or manufacturing an identity operation for it.
- Constructor-time shape, data-type, rank, layout, gradient, provenance, storage, or adjacent-child
  compatibility checking. Each child validates the exact Tensor it receives under its existing
  contract.
- Child detachment, renaming, reparenting, sharing, cloning, copying, or state duplication.
- Fusing, flattening, canonicalizing, optimizing, compiling, lowering, scheduling, executing, or
  benchmarking a sequence.
- Numerical values, numerical algorithms, backend support, kernels, devices, ONNX, serialization,
  persistent checkpoint formats, optimizer behavior, or training-session orchestration.
- Any Model, training-extension, compiler, runtime, prepare, Engine, backend, build-dependency,
  architecture-contract, global-roadmap, or CPU implementation change.
- Detailed planning for a later NN task.

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md): NN stateful composition ownership,
  Model Tensor identity, and one-way dependency rules.
- [Current architecture plan](../../../../architecture/current-architecture-plan.md).
- [ADR 0007: Neural-network module and training boundary](../../../../design/decisions/0007-neural-network-module-and-training-boundary.md).
- [Dependency rules](../../../../architecture/dependency-rules.md).
- [Training graph](../../../../architecture/training-graph.md).
- [Planning guide](../../../planning-guide.md).
- [NN master plan](../master-plan.md).
- [Task 0002](0002-module-tree-ownership-and-recursive-mode-propagation.md): exclusive child
  ownership, registration, numeric-path prerequisites, and recursive mode behavior.
- [Task 0005](0005-linear-layer.md), [Task 0006](0006-layer-normalization-layer.md), and
  [Task 0007](0007-embedding-layer.md): the three current unary Tensor-forward consumers.
- [Task 0008](0008-batch-normalization-layer.md) and [Task 0009](0009-dropout-layer.md): explicit
  evidence against a universal signature.
- [Task 0010](0010-state-dictionary-and-checkpoint-contract.md): deterministic owned-tree state
  export and strict path-keyed load.

## Architecture constraints

- `extensions/nn` owns this stateful module-composition contract and continues to depend only on
  `modules/model`.
- `Module` remains the general state/tree/mode owner and deliberately retains no universal forward
  method. Typed subclasses define only signatures that are truthful for them.
- `UnaryTensorModule` is a narrow nominal intersection of existing Module ownership and one unary
  Tensor-forward shape. It does not own Tensor semantics, execution, autograd, context synthesis,
  or explicit state threading.
- Model remains the sole owner of generic Tensor operations, descriptors, identities, provenance,
  expression factories, and local expression validation. NN delegates to those contracts.
- Tensor objects remain immutable identities. A sequence passes exact references between child
  calls; it neither mutates nor copies them.
- Existing child ownership remains exclusive and permanent. Construction must complete all
  ordinary candidate validation before installing any child because a failed constructor cannot
  expose a usable parent from which a partially attached prefix could be detached.
- Numeric child names are stable local path segments derived from the immutable constructor order.
  They share the same namespace and dot-path rules as every other Module child.
- `train()` and `eval()` remain the existing recursive structural mode operations. `Sequential`
  neither overrides them nor changes their preflight/assignment semantics.
- Current participating children are mode-insensitive. A future mode-sensitive or explicitly
  stateful module does not qualify merely because a container wants to call it.
- No new project dependency, module boundary, architecture rule, ADR, architecture test, or build
  structure is required.

## Type-design decision

| Candidate | Compile-time ownership | Compile-time unary forward | Decision |
|---|---|---|---|
| Public interface implemented by Module subclasses | No: an interface cannot extend the `Module` class; accepting the interface loses child ownership, while accepting `Module` loses forward | Yes in isolation | Reject. Restoring both facts requires a cast, class-level intersection generic, sealed implementation list, or a leaky `module()` bridge. |
| Abstract `Module` subclass | Yes | Yes | Select. It is the smallest nominal type accepted by a heterogeneous list and nested sequence without casts or extra accessors. |
| Package-owned adapter around Module plus forward function | Only indirectly | Only indirectly | Reject. It creates a second identity, obscures which object owns state, and has no current consumer beyond avoiding the direct type contract. |

Do not replace the selected class with a generic class such as
`Sequential<M extends Module & UnaryForward>`. That surface is awkward for a heterogeneous list of
final layer types, leaks an implementation constraint into every caller, and provides no benefit
over the narrow abstract base.

## Public API

| Package/type | Visibility and shape | Exact task-owned contract |
|---|---|---|
| `io.github.pho001.synaptik.nn.module.UnaryTensorModule` | `public abstract class UnaryTensorModule extends Module` | Nominal base only for Modules whose complete public forward signature is one non-null Tensor to one non-null Tensor. No fields, interfaces, nested types, factories, or other methods. |
| `UnaryTensorModule()` | `protected` | Creates the ordinary empty training-mode Module base by calling `super()`; creates no state or Tensor. |
| `UnaryTensorModule.forward(Tensor input)` | `public abstract Tensor forward(Tensor input)` | Requires non-null input and promises a non-null exact expression/reference result under the concrete child's documented contract. It defines no shared shape, type, freshness, mode, state-mutation, or numerical rule. |
| `io.github.pho001.synaptik.nn.module.Sequential` | `public final class Sequential extends UnaryTensorModule` | Structural ordered module owner. No interfaces, nested types, or public/protected fields. |
| `Sequential(List<? extends UnaryTensorModule> modules)` | `public` | Takes one non-null ordered list, permits empty, validates/snapshots every exact child before ownership mutation, and registers child index `i` under `Integer.toString(i)`. |
| `Sequential.forward(Tensor input)` | `public final` override | Non-null validation followed by exact left-to-right forwarding; empty returns exact input. It returns the exact final child output and creates no wrapper of its own. |

The implementation may retain exactly one private final `List<UnaryTensorModule>` field. The field
contains the immutable registration snapshot and is used directly for ordered forward traversal.
There is no public list accessor because inherited `children()` already exposes an immutable
ordered structural snapshot, and inherited recursive state APIs expose the owned state.

`Module` gains one package-private final support method with exact source signature:

```java
final <T extends Module> List<T> registerIndexedChildren(List<? extends T> modules)
```

This is not public or protected API. It snapshots and validates the complete list, then registers
the exact candidates under canonical decimal indices and returns the immutable typed snapshot.
It exists in `Module` because only that class can inspect and install private parent ownership
atomically. `Sequential` is placed in the same `module` package so this current need does not widen
the subclass API. The existing protected `child(String, T)` signature and behavior remain exact.

`Linear`, `LayerNorm`, and `Embedding` keep their existing final public types and declared forward
signatures, but their direct superclass becomes `UnaryTensorModule` and their forward methods gain
an ordinary `@Override`. `BatchNorm` and `Dropout` continue to extend `Module` directly.

## Construction, ownership, and validation

Validation and side effects are fixed in this order:

1. Reject a null list with `NullPointerException` and message `modules`.
2. Traverse the supplied list once in encounter order into an independent immutable structural
   snapshot. Reject the first null element with `NullPointerException` and a diagnostic identifying
   `modules[index]`. Caller list identity is not retained or promised.
3. Reject the first repeated module identity with `IllegalArgumentException`. Equality overrides
   are irrelevant; duplicate detection is by object identity. An empty list succeeds.
4. Preflight every candidate in index order using canonical names `"0"` through
   `Integer.toString(size - 1)`: shared-name availability, self/ancestor exclusion, and absence of
   an existing parent. The ordinary public constructor cannot supply its not-yet-created receiver
   as a child, but the ownership primitive keeps the established defensive cycle check.
5. Only after all candidates pass, install child-map entries and parent links in increasing index
   order. Under caller-coordinated use, ordinary validation failure changes neither the receiver's
   registry nor any candidate's parent.
6. Retain the same immutable typed snapshot for forward traversal.

An already-owned candidate fails with the existing `IllegalStateException` ownership category.
A self/ancestor candidate fails with the existing `IllegalArgumentException` cycle category.
There is no detach or rollback API. Like all Module mutation, constructor registration is not
thread-safe or linearizable; validation-before-install does not promise atomic visibility to racing
threads or recovery from JVM-fatal failure.

A duplicate or late-invalid candidate must not strand a valid prefix under an unreachable failed
constructor. Focused tests must prove this by successfully attaching an earlier candidate to a
different sequence after the first construction fails.

## Forward composition and failure effects

- `forward` first rejects null input with `NullPointerException` and message `input`, before any
  child is called.
- Empty composition returns the exact input reference. It creates no Model occurrence and consumes
  no Tensor identifier.
- Non-empty composition starts with `current = input`. For every retained child in numeric order,
  it calls `child.forward(current)` exactly once and assigns the exact returned reference to
  `current`.
- A null child result violates `UnaryTensorModule`'s contract. `Sequential` rejects it immediately
  with `NullPointerException` identifying the child output/index; no later child is called.
- The returned value is the exact final `current` reference. `Sequential` adds no identity,
  producer, provenance, label, storage, descriptor, or expression node.
- The abstract base makes no blanket freshness promise. The three current participating layer
  calls each retain their existing fresh Model-expression behavior; empty composition is identity,
  and a future truthful unary Module may itself document identity-preserving behavior.
- If child `i` throws or returns null, calls `0` through `i - 1` have already occurred. Their
  expressions, Tensor identifiers, retained references, and any child-local side effects remain.
  The failing child controls its own validation and partial-construction semantics. `Sequential`
  catches nothing and performs no expression, identifier, state, or mode rollback.
- No whole-sequence prevalidation is attempted. For example, a `Linear` result is handed to a
  following `LayerNorm`, and that layer/Model contract validates the actual intermediate when its
  call occurs.

## Mode, state, and path behavior

- `Sequential` starts in `TRAINING` through `Module`. Every registered child retains its own local
  initial mode after construction; construction does not normalize mode.
- Calling `train()` or `eval()` on the sequence uses the existing Module-tree preflight and then
  assigns the requested mode to the sequence and every descendant in deterministic preorder.
- `Sequential.forward` does not read `mode()`, call `forwardContext()`, or alter a mode. `Linear`,
  `LayerNorm`, and `Embedding` are mode-insensitive, so their local mode does not select a branch.
  Nested sequences behave the same way.
- Child names and state paths are stable numeric paths. A sequence containing `Linear` then
  `LayerNorm` exposes child names `0`, `1` and state paths such as `0.weight`, `0.bias`, `1.scale`,
  and `1.bias`. A nested sequence at index one produces paths such as `1.0.weight`.
- Inherited `parametersRecursively()`, `buffersRecursively()`, `stateDictionary()`, and
  `loadStateDictionary(...)` require no special case. They traverse numeric children in order.
  Strict load remains path-keyed and validate-before-install under task 0010.
- A compatible replacement through a discovered `Parameter` or state-dictionary load changes the
  binding observed by a later child forward call. An expression already constructed from an older
  binding remains unchanged. Sequence forward is not a joint state snapshot or transaction.
- No child replacement is possible. Numeric path stability follows immutable permanent ownership.

## Package impact

```text
io.github.pho001.synaptik.nn/
  module/
    Module.java                    existing owner; package-private atomic indexed registration
    UnaryTensorModule.java         new public narrow forward/ownership base
    Sequential.java                new public structural composition module
    package-info.java              updated ownership and unary-composition explanation
  layers/
    Linear.java                    same API/behavior; new direct superclass
    LayerNorm.java                 same API/behavior; new direct superclass
    Embedding.java                 same API/behavior; new direct superclass
    BatchNorm.java                 unchanged direct Module subclass and explicit context API
    Dropout.java                   unchanged direct Module subclass and explicit state/context API
    package-info.java              identifies participating and excluded layer signatures
```

`Sequential` belongs under `module/`, not `layers/`: it owns no operation semantics, parameters,
buffers, initialization policy, or numerical layer behavior. Its sole responsibility is structural
Module ownership and ordered invocation. A new `container/` or `composition/` package for one type
would add unnecessary package surface.

## Affected files

Implementation-owned production and test paths:

- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Module.java`
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/UnaryTensorModule.java`
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Sequential.java`
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/package-info.java`
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/Linear.java`
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/LayerNorm.java`
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/Embedding.java`
- `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/layers/package-info.java`
- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/module/SequentialTest.java`
- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/LinearTest.java`
- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/LayerNormTest.java`
- `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/EmbeddingTest.java`

Documentation-focused finalization paths:

- `docs/api/training-api.md`
- `docs/glossary.md`
- `docs/planning/extensions/nn/master-plan.md`
- `docs/planning/extensions/nn/tasks/0011-unary-tensor-module-composition-and-sequential.md`

An implementation may reduce this set when a listed Javadoc/test remains exact without an edit,
but it must record the review. Adding a production type, public member, test helper in production,
dependency/build path, architecture document, or unrelated documentation path requires stopping
and justifying a revised plan before implementation.

## Maximum scope

- Two new production classes, one package-private `Module` support method plus private validation
  refactoring, three direct-superclass changes, one new focused test class, necessary exact-surface
  updates to three existing tests, two package Javadoc files, Training API, glossary, and the two NN
  planning records.
- No other Java, test, documentation, Gradle, architecture, roadmap, module, or backend file.

## Acceptance criteria

- `UnaryTensorModule` has exactly the selected public/protected surface and extends `Module`
  directly. It introduces no interface, field, nested type, generic parameter, default behavior,
  or shared mode/state semantics.
- `Linear`, `LayerNorm`, and `Embedding` extend it and retain every prior constructor, field,
  declared public method, validation, Javadoc contract, and forward-expression behavior except the
  intentional new superclass/override relationship.
- `BatchNorm` and `Dropout` remain direct `Module` subclasses and are not assignable to
  `UnaryTensorModule`.
- `Sequential` is final, extends `UnaryTensorModule`, has exactly one public List constructor and
  one declared public final forward override, and has no varargs/accessor/mutation surface.
- Constructor null-list, null-element, identity-duplicate, already-owned, and defensive cycle
  categories are deterministic and validate before any registration. Empty construction succeeds.
- Caller list mutation after successful construction cannot change child membership, order, names,
  or forwarding. Child instances themselves are retained exactly.
- Children are permanently registered in encounter order under canonical names `0`, `1`, ...;
  inherited direct/recursive snapshots and state dictionaries expose the corresponding deterministic
  numeric paths.
- Failed construction leaves every valid candidate unattached and reusable under the ordinary
  caller-coordinated contract.
- Empty forward validates input, returns the exact input, and creates no expression or identifier.
- Non-empty forward calls each child exactly once left to right with exact reference handoff and
  returns the exact final child result. It performs no ahead-of-child compatibility check.
- Null input calls no child. A throwing/null-returning later child calls no suffix child; successful
  prefix work is not rolled back.
- Nested sequences preserve numeric paths and left-to-right flattening behavior through ordinary
  calls without structurally flattening the owned tree.
- Existing recursive `train()`/`eval()` behavior reaches the entire sequence tree. Sequence forward
  neither reads nor synthesizes context; current participating children remain mode-insensitive.
- Compatible parameter replacement and state-dictionary load are visible to later forward calls at
  stable numeric paths; earlier expressions remain unchanged.
- No Model, BatchNorm, Dropout/result-carrier, ForwardContext, graph RNG, training, compiler,
  runtime, prepare, Engine, backend, dependency, architecture, or build contract changes.
- Javadocs fully document inputs, outputs, nullability, ownership, order, identity, failure,
  partial-expression, mode, mutation, and thread-safety boundaries without claiming execution or
  numerical results.
- The Training API and glossary describe the implemented unary/Sequential boundary after code
  lands and explicitly preserve the BatchNorm/Dropout exclusions.

## Tests / validation

Implementation validation tier: normal affected-module validation plus documentation and exact
surface checks. Repository-wide validation is not required because no dependency, architecture,
shared build, or multi-module executable contract changes.

Focused tests must cover:

- reflection of the exact abstract-base and final-container type/member surfaces;
- existing layer superclass assertions updated only for `Linear`, `LayerNorm`, and `Embedding`,
  with `BatchNorm`/`Dropout` exclusion asserted;
- null list, late null element, repeated exact instance, and late already-owned candidate with no
  partial ownership installation;
- empty construction, immutable caller-list snapshot, numeric child names/order, exact references,
  and absent public list/varargs/mutation APIs;
- empty exact-reference identity and no Tensor-identifier allocation;
- custom test unary modules proving exact left-to-right inputs, one call per child, final output
  identity, null-result handling, exception propagation, suffix suppression, and retained prefix
  effects;
- one real `Linear`/`LayerNorm` chain and one `Embedding`-starting chain using valid current Tensor
  contracts, proving reachable provenance rather than numerical output;
- nested sequence order, child maps, recursive parameter paths, state-dictionary paths/order, and
  strict load compatibility;
- parameter replacement after sequence construction affecting a later expression while an earlier
  expression retains its old exact binding;
- recursive train/eval propagation through nested sequences with unchanged mode-insensitive forward
  composition;
- regression of all existing NN layer, ownership, traversal, replacement, and state-dictionary
  tests.

Implementation context commands:

```text
./gradlew :extensions:nn:test --tests io.github.pho001.synaptik.nn.module.SequentialTest
./gradlew :extensions:nn:test
git diff --check
```

The independent documentation context must review the unchanged final executable diff and reuse
the successful NN suite evidence unless it changes Java behavior or finds a concrete reason to
rerun. It must run:

```text
./gradlew :extensions:nn:javadoc
git diff --check
```

It must also inspect generated Javadoc for `Module`, `UnaryTensorModule`, `Sequential`, and the
three changed layer hierarchy pages; verify exact public/protected surface with reflection or
`javap`; compile one external-package import/subclass/composition example; check Markdown local
paths/anchors, balanced fences, one terminal newline, and trailing whitespace; and confirm the
final exact path set. No numerical, compiler, backend-conformance, or integration test is required.

## Dependencies

- NN tasks 0001–0010 are Complete and provide the Module tree, mode, state, concrete layer, explicit
  context/state, and state-dictionary contracts this task composes.
- Completed Model Tensor expression contracts provide `linear`, `layerNorm`, and `embedding` and
  their exact validation/provenance behavior.
- ADR 0007 and the architecture contract already assign stateful neural-network composition to NN.
- CPU, compiler, training, runtime, prepare, Engine, backend, and persistent checkpoint work are
  not prerequisites.

## Follow-up tasks

- A contextual/state-threading container may be planned only when a concrete consumer can preserve
  `ForwardContext`, BatchNorm buffer transitions, graph RNG state, and result carriers explicitly.
  This task does not reserve its name or signature.
- Stateless NN functional conveniences remain a concise future master-plan concern only when a
  concrete API is selected.
- Persistent checkpoint encoding and training-session/optimizer coordination remain future work
  owned by their concrete consumers.
- Do not create another detailed NN task during implementation of 0011.

## Documentation and no-change review

- `UnaryTensorModule`, `Sequential`, the package contract, and changed layer Javadocs require an
  independent documentation-focused final pass under the General and API/Javadoc profiles.
- `docs/api/training-api.md` must update its implementation-status/current-NN explanation because
  downstream training consumers now see stable numeric composition paths and a narrow public
  module subtype. It must not invent optimizer or training-session use of Sequential.
- `docs/glossary.md` must update the existing neural-network module entry and layer counts, define
  unary Tensor module and Sequential at first use, show one concise numeric-path/left-fold example,
  and preserve the explicit BatchNorm/Dropout distinction. It must not imply numerical evaluation.
- `docs/api/tensor-api.md` requires no change: no generic Tensor operation or Model contract changes.
- `docs/api/compile-api.md`, `docs/api/runtime-api.md`, training graph, user training/autograd guides,
  and backend guides require no change because capture, gradient, execution, and backend behavior
  remain untouched.
- `ARCHITECTURE.md`, current architecture plan, module boundaries, dependency rules, and ADR 0007
  require no change because this realizes the already-authorized NN composition responsibility.
- Architecture tests and Gradle files require no change because `extensions/nn` retains its sole
  Model project dependency.
- Backend conformance and integration tests require no change because this task constructs only NN
  ownership and Model expressions.
- Preserve every unrelated/concurrent CPU planning, source, test, guide, roadmap, and glossary
  change exactly. Documentation work must merge its targeted glossary edit with the live file.

## Architecture impact

Expected impact: None.

This task realizes the existing NN ownership/composition boundary with a narrow type needed by a
real container. It does not alter module direction, public lifecycle ownership, Tensor semantics,
or execution boundaries and therefore needs no architecture-contract, ADR, dependency, or
architecture-test change.

## Implementation prompt

```text
You are the clean-context implementation agent for Synaptik NN task 0011. Work in the existing
shared worktree. Do not use GSD. Do not commit or push. Read root AGENTS.md, ARCHITECTURE.md,
current architecture plan, planning guide/roadmap, NN master plan and tasks 0001–0011, ADR 0007,
final Module/state/mode/state-dictionary APIs and tests, final Linear/LayerNorm/Embedding/
BatchNorm/Dropout APIs and tests, Model Tensor contracts/result carriers, Training API, glossary,
documentation rules, and dependency architecture tests. Preserve all unrelated/concurrent CPU
changes exactly.

Implement task 0011 exactly. Add public abstract UnaryTensorModule and final module-package
Sequential with only the specified APIs. Change only Linear, LayerNorm, and Embedding to the new
base. Keep BatchNorm and Dropout outside it. Add Module's package-private validate-before-install
indexed-child primitive; do not widen Module's public/protected surface. Accept one immutable List
snapshot, allow empty identity, use canonical numeric names, forward exact references left to
right, and preserve prefix effects on later failure. Add no cast, reflection, interface, adapter,
varargs, accessor, generic facade, shape pipeline validation, context/RNG erasure, Model behavior,
dependency, or execution claim.

Run the focused Sequential test, the complete NN module suite once after the final executable
change, and git diff --check. Then hand the unchanged diff and exact evidence to a distinct clean
documentation-focused agent. That agent must finalize Javadocs, package docs, Training API,
glossary, planning records, and no-change reasoning; run NN Javadoc and documentation/surface/
scope/whitespace checks; and must not repeat successful Java tests unless it changes executable
behavior. Do not mark Complete until implementation, documentation, and all required validation
pass. If a type-safe design requires a cast/reflection/leaky bridge, if construction cannot avoid
stranding partially attached children on ordinary validation failure, or if architecture/scope
must expand, stop and report the blocker rather than inventing behavior.
```

## Documentation-agent handoff

The implementation context must provide the clean documentation context with:

- the exact final executable diff and changed-path list;
- focused and full NN test commands, counts, and results;
- final exact public/protected surface and constructor validation order;
- proof that constructor validation precedes ownership installation;
- participating/excluded layer list and unchanged behavioral contracts;
- empty/non-empty/failure expression identity evidence;
- numeric child/state-path, replacement/load, and recursive-mode evidence;
- unresolved issues or explicit confirmation that none remain.

The documentation context must independently read the final source/tests and directly relevant
contracts. It may edit only task-owned documentation/Javadocs and must preserve live concurrent
glossary work. It records reasoned no-change conclusions for Tensor, Compile, Runtime, architecture,
training graph, dependency tests, conformance/integration, Gradle, and other modules.

## Local decisions

- Select an abstract Module subclass because it alone gives a heterogeneous list both ownership
  and forward capabilities at compile time without an adapter or cast.
- Place Sequential in `nn.module` because it is structural composition with no layer operation or
  state policy. This also keeps atomic registration support package-private.
- Admit exactly Linear, LayerNorm, and Embedding because their complete current public forward
  shape is Tensor-to-Tensor without explicit context or state. Embedding's integral-index input is
  still a Tensor and its own Model contract retains exact type validation.
- Exclude BatchNorm and Dropout rather than hide their essential explicit context/state/result
  semantics.
- Accept only a List. It covers fixed and programmatic construction while avoiding redundant array
  validation and overload precedence.
- Allow empty as a useful immutable identity composition. Identity is an exact-reference bypass,
  not a manufactured Model operation.
- Expose no module-list accessor. Inherited children/state snapshots are the established structural
  inspection surface.
- Preflight the whole list before installing parent links so failed construction cannot permanently
  strand a prefix under an unreachable object.
- Keep numeric registration stable and nested rather than flattening, preserving Module ownership
  and state paths.

## Known limitations

- Only unary Tensor-to-Tensor Modules compose. BatchNorm, Dropout, multiple inputs/outputs, explicit
  contexts, and explicit threaded state require a different future design.
- The base contract does not prove adjacent Shape/data-type compatibility. Failure may occur after
  earlier children have created expressions.
- Forward, ownership mutation during construction, mode changes, replacement, traversal, and load
  are not thread-safe or jointly transactional.
- Numeric names are structural positions, not caller-defined semantic names. Reordering constructor
  elements changes state paths by design.
- Empty identity and a future identity-preserving unary child may return an existing Tensor; the
  abstract base provides no universal fresh-result guarantee.
- Sequential does not flatten nested containers, fuse operations, or reduce call overhead.

## Validation evidence

- Clean planning context inspected the complete authoritative architecture and planning contracts,
  documentation rules/profiles, ADR 0007, NN master plan and tasks 0001–0010, final NN source/tests
  for Module state/tree/mode/dictionary and all five concrete layers, Model Tensor expression and
  explicit BatchNorm/Dropout/RNG result contracts, Training API, glossary, NN build dependency, and
  the NN/training architecture test.
- Repository search found no existing Sequential/container API or accepted universal forward
  interface. Earlier layer tasks deliberately deferred a shared contract until this exact consumer;
  BatchNorm and Dropout tasks explicitly require preserving their context/state/result signatures.
- Type analysis rejected a public interface because Java interfaces cannot extend `Module`, and
  rejected an adapter because it would split state ownership from invocation. The selected abstract
  subclass is compile-time safe for heterogeneous and nested composition without casts/reflection.
- Planning inspection found that one-at-a-time `Module.child` registration is atomic per child but
  cannot prevent a failed multi-child constructor from stranding an installed prefix. The selected
  package-private whole-list preflight/installation primitive closes that construction-only need
  without changing public/protected Module API.
- This planning pass changes only this new task specification and the NN master plan. It runs no
  Java, Gradle, Javadoc, numerical, compiler, backend, or repository-wide test because no executable
  or explanatory API documentation is changed during planning.
- Planning validation passed local Markdown path resolution, required/canonical heading presence,
  balanced fences, one terminal newline, trailing-whitespace checks, exactly one NN Ready row/task,
  and exactly the two permitted NN planning paths. Whole-worktree `git diff --check` passed with no
  output. The new-file `git diff --no-index --check` produced no whitespace diagnostic; its exit
  code was the expected `1` because the compared file is newly added rather than identical to
  `/dev/null`.
- Clean implementation context `/root/nn_0011_implementation` found no architecture, final-API,
  dependency, package-placement, or task-scope conflict. It preserved all unrelated live CPU,
  roadmap, glossary, and probe changes.
- `./gradlew :extensions:nn:test --tests io.github.pho001.synaptik.nn.module.SequentialTest` passed
  after executable implementation stabilized: one suite, 12 tests, zero failures, errors, or
  skips. Coverage locks exact type/member surfaces; participating and excluded superclasses;
  whole-list null/identity/name/cycle/ownership preflight and prefix reusability; immutable list
  snapshot; numeric/nested paths; exact empty and non-empty reference flow; null/throw failure
  effects; actual Linear/LayerNorm/Embedding provenance; state load/replacement snapshots; and
  recursive mode behavior.
- The affected-layer command selecting `LinearTest`, `LayerNormTest`, and `EmbeddingTest` passed:
  three suites and 20 tests with zero failures, errors, or skips. The only intended existing-layer
  contract change is their direct superclass and ordinary forward override relationship.
- The sole authoritative final `./gradlew :extensions:nn:test` passed after all executable source,
  tests, and draft Javadocs/package edits stabilized (`BUILD SUCCESSFUL`; five actionable tasks,
  one executed and four up-to-date). XML reports contain 17 suites and 110 tests with zero
  failures, errors, or skips. No executable Java or test changed afterward in this context.
- Preliminary `./gradlew :extensions:nn:javadoc` passed (`BUILD SUCCESSFUL`; three actionable
  tasks, one executed and two up-to-date). Final Javadoc editing, generation, and rendered-page
  inspection remain owned by the separate documentation context.
- `javap -public` confirmed the exact selected abstract/final types, constructor/method surfaces,
  three participating direct superclasses, and direct-Module BatchNorm/Dropout exclusions.
  Private inspection confirmed Sequential retains only one final typed List field, while Module
  exposes `registerIndexedChildren(List)` only at package visibility and no universal forward.
- An external-package Java 26 example importing `Sequential` and subclassing
  `UnaryTensorModule` compiled successfully against the final NN and Model class directories. It
  constructed the sole public List form and called the unary forward surface without casts,
  reflection, adapters, or bridge accessors.
- Production import and Gradle scans found only Model, existing NN, and JDK dependencies;
  `extensions/nn` retains exactly its sole `implementation(project(":modules:model"))` edge. The
  new composition types import no ForwardContext, graph RNG/result carrier, training, compiler,
  runtime, prepare, Engine, or backend contract.
- Implementation scope inspection found exactly 14 current NN implementation/planning paths:
  eight production/Javadoc paths, four tests, this task, and the NN master plan. The final
  documentation context is expected to add only the two remaining authorized explanatory paths,
  Training API and glossary. Whole-worktree `git diff --check`, explicit new-file no-index checks,
  trailing-whitespace scans, and terminal-newline checks passed with no diagnostics. Unrelated
  live CPU, roadmap, glossary, and probe paths remain preserved outside this implementation diff.
- The mandatory clean documentation context `/root/nn_0011_docs` independently reviewed the final
  executable diff, focused tests, all affected and excluded public APIs, generated pages, directly
  relevant architecture/ADR/planning contracts, documentation profiles, Tensor and Training APIs,
  glossary, and NN tasks 0001–0010. It found no behavior, API, architecture, dependency, or scope
  blocker and changed no executable Java or test.
- The documentation context finalized the affected type, constructor, method, and package
  Javadocs. The rendered contracts preserve Module's lack of a universal forward method, the
  narrow unary participation boundary, whole-list ownership preflight, exact empty/non-empty
  reference behavior, prefix failure effects, inherited numeric state/mode behavior, and absent
  execution and concurrency guarantees. `./gradlew :extensions:nn:javadoc` then passed
  (`BUILD SUCCESSFUL`; three actionable tasks, two executed and one up-to-date), and inspection of
  the generated Module, UnaryTensorModule, Sequential, Linear, LayerNorm, Embedding, and package
  pages found the intended hierarchy, links, signatures, parameters, returns, and failure terms.
- Final `javap -public` and private-member inspection reconfirmed that UnaryTensorModule is public
  abstract with only its protected constructor and public abstract forward declaration;
  Sequential is public final with only its List constructor and public final forward declaration;
  Module's indexed registration method is package-private final; the three participating layers
  directly extend UnaryTensorModule; and BatchNorm/Dropout directly extend Module. An independent
  external-package Java probe compiled and ran successfully, including protected subclassing,
  composition, superclass/finality/member reflection, and numeric-child discovery.
- Training API and glossary finalization now explain the implemented composition boundary without
  turning it into training, context/RNG erasure, numerical evaluation, compilation, or execution.
  The glossary gives concise empty and two-child declarative examples and stable numeric state
  paths. Automated checking passed every local Markdown path and anchor, balanced fence, and
  terminal newline in the four task-owned Markdown documents.
- Final import and dependency scans found only JDK, Model, and existing NN references and retained
  exactly `implementation(project(":modules:model"))`. Scope checking found exactly the 16
  authorized task paths: eight production/Javadoc paths, four tests, Training API, glossary, task,
  and NN master plan. All NN tasks 0001–0011 now read Complete with no Ready task or later detailed
  NN specification. Trailing-whitespace, terminal-newline, new-file no-index, and whole-worktree
  `git diff --check` gates passed. Unrelated concurrent CPU source/test/documentation/planning,
  roadmap, and glossary hunks remain unchanged by this documentation context.
- The successful Sequential 1-suite/12-test, affected-layer 3-suite/20-test, and authoritative NN
  17-suite/110-test evidence was reused as required. No Java behavior or test changed afterward,
  and no concrete reason appeared to repeat those suites in the documentation context.
- Tensor API, Compile API, Runtime API, architecture contract/current plan/module boundaries,
  dependency rules, ADR 0007, training graph, architecture tests, backend conformance, integration
  tests, Gradle, and every other module require no change: this task adds only NN-owned declarative
  composition over existing Tensor expressions and the already-authorized Model-only dependency.

## Implementation notes

- Clean implementation context `/root/nn_0011_implementation` added the narrow abstract base,
  immutable numeric-child sequence, package-private whole-list registration primitive, three
  superclass changes, focused coverage, and draft Javadocs/package documentation. No Model,
  context/RNG, execution, dependency, Gradle, architecture, backend, or unrelated behavior changed.
- `registerIndexedChildren` traverses the caller list into an independent snapshot, rejects nulls,
  then rejects repeated identities, then preflights every numeric local name, cycle, and existing
  parent before installing any child-map entry or parent link. Focused late-invalid tests prove an
  earlier valid candidate remains reusable after null, duplicate, ownership, cycle, and name
  failures.
- `Sequential.forward` null-checks the input, performs one exact-reference left fold over the
  retained immutable list, rejects a null result at its index, and preserves completed prefix work
  on a later null or exception. Empty identity consumes no Tensor identifier.
- Preserve the live unrelated CPU source/test/guide/planning/roadmap/glossary work during the
  mandatory later documentation pass. Executable source and tests are stable; the separate pass
  must not repeat the successful Java suites unless it changes executable Java behavior.

## Completion summary

- Completed changes: added the exact UnaryTensorModule/Sequential contract, atomic indexed child
  registration, the three intended superclass overrides, focused tests, finalized Javadocs/package
  documentation, Training API/glossary explanations, and synchronized NN planning evidence.
- Files changed or created: exactly the 16 authorized paths listed under Affected files; unrelated
  concurrent paths were preserved.
- Tests and validation: reused passing focused Sequential 1-suite/12-test, affected-layer
  3-suite/20-test, and authoritative NN 17-suite/110-test evidence with zero failures, errors, or
  skips. Final Javadoc generation/page inspection, `javap`, independent reflection and external
  compilation/run, Markdown, dependency/import, exact-scope, status, newline/whitespace, no-index,
  and `git diff --check` gates passed.
- Documentation impact: affected public/package Javadocs, Training API, glossary, task, and NN
  master plan are final. Tensor/Compile/Runtime APIs, architecture/training graph, dependency tests,
  conformance/integration, Gradle, and other modules were reviewed and correctly remain unchanged.
- Unresolved issues: None.
- Required follow-up: None for task 0011.

Status: Complete
