# NN Extension Master Plan

## Goal

Define the stateful neural-network composition layer: module ownership, trainable parameters,
persistent buffers, train/eval behavior, forward context, layers, blocks, and functional
conveniences built from `modules/model` Tensor semantics.

## Architecture references

- [Architecture contract](../../../../ARCHITECTURE.md)
- [ADR 0007: Neural-network module and training boundary](../../../design/decisions/0007-neural-network-module-and-training-boundary.md)
- [Module boundaries](../../../architecture/module-boundaries.md)
- [Dependency rules](../../../architecture/dependency-rules.md)
- [Training graph](../../../architecture/training-graph.md)

## Scope

- `Module`, `Parameter`, and `Buffer` contracts
- module-tree parameter and buffer discovery
- train/eval mode propagation and forward context
- explicit parameter-initialization conveniences over caller-owned sources
- neural-network layers, blocks, and functional conveniences that compose model operations

## Out of scope

- generic Tensor or operation semantics
- compiler autograd construction
- optimizer algorithms, update orchestration, and training sessions
- backend storage, lowering, kernel selection, and backend-specific optimizer execution

## Module invariants

- NN owns module-declared `Parameter` and `Buffer` values.
- A discovered `Parameter` is the downstream capability for installing one schema-compatible
  trainable Tensor binding; NN retains declaration, schema, and validation ownership.
- NN owns train/eval forward behavior and forward context.
- NN composes `modules/model`; it does not make model depend on nn.
- NN must not depend on `extensions/training`, compiler, runtime, prepare, engine, or concrete backends.
- Optimizers and training-step orchestration belong to `extensions/training` downstream of nn.

## Allowed dependencies

- modules/model

## Forbidden dependencies

- extensions/training
- modules/compiler
- modules/runtime
- modules/prepare
- modules/engine
- backends/cpu
- backends/metal
- backends/cuda

## Package structure

```text
io.github.pho001.synaptik.nn/
  module/       public module ownership, named state, and forward-context contracts
  initialization/ explicit eager parameter-initialization conveniences and policies
  layers/       public stateful neural-network layers
  functional/   stateless NN-oriented conveniences over model operations
```

The implemented foundation remains deliberately small, and task 0005 adds only the first
concrete type under `layers/`:

- `Module` owns direct named state declarations and its local train/eval mode; it has no universal
  forward signature.
- `Parameter` and `Buffer` retain a named current `Tensor` binding without becoming `Tensor`
  subtypes.
- `ForwardMode` and immutable `ForwardContext` provide the mode snapshot supplied to a concrete
  layer's own typed forward method.

Task 0004 adds only the public stateless `ParameterInitializers` namespace and package
documentation under `initialization/`. It creates eager gradient-enabled unlabeled Tensor leaves;
it does not change `Parameter`, retain a random source, or introduce an initializer object model.
Task 0005 adds final public `Linear` under `layers/` without a generic layer
abstraction. `functional/` remains future work. State dictionaries remain later work.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| [0001](tasks/0001-module-parameter-buffer-and-forward-context-foundation.md) | Module, parameter, buffer, and forward-context foundation | Complete | Completed `modules/model` Tensor contracts; ADR 0007 | Created the NN Gradle module and its minimal model-only state/mode API, activated the required one-way training build edge, and enforced it with architecture tests; it adds no layers or execution behavior. |
| [0002](tasks/0002-module-tree-ownership-and-recursive-mode-propagation.md) | Module-tree ownership and recursive mode propagation | Complete | 0001 | Added exclusively owned child modules, deterministic collision-free dot-path parameter/buffer snapshots, and atomic recursive `train()`/`eval()` propagation without checkpoint or optimizer behavior. |
| [0003](tasks/0003-validated-parameter-and-buffer-binding-replacement.md) | Validated parameter and buffer binding replacement | Complete | 0002 | Added Module-owned replacement of one direct current Tensor binding with exact wrapper identity, structural snapshot, validation, and explicit no-concurrency-guarantee semantics; it adds no optimizer or checkpoint behavior. |
| [0004](tasks/0004-explicit-eager-parameter-initializers.md) | Explicit eager parameter initializers | Complete | 0001–0003; completed Model eager constant/random contracts | Added one stateless eight-method parameter-initializer surface: floating zero/one and explicit-source normal/uniform leaves plus fixed rank-two `[outFeatures, inFeatures]` Glorot and Kaiming-ReLU policies, with no label, hidden RNG, or Parameter policy. |
| [0004A](tasks/0004a-parameter-update-and-traversal-hardening.md) | Parameter update and traversal hardening | Complete | 0001–0004; post-0004 code review | Closed the three review findings with a public schema-validated `Parameter.replace`, iterative identity-defended deep-tree traversal, and complete fan-initializer Java-array-limit contracts before adding a layer. |
| [0005](tasks/0005-linear-layer.md) | Linear layer | Complete | 0001, 0004, 0004A; completed model `Tensor.linear` | Added the first stateful layer with caller-supplied or explicit-source Glorot-uniform `weight`, optional caller-supplied or zero `bias`, stable parameter handles, and exact delegation to visible Model linear composition without execution behavior. |

## Milestones

- Module, parameter, buffer, and forward-mode contracts
- Explicit eager initialization and first `Linear` layer
- Initial stateful layers and functional conveniences
- Training-extension integration over the stable parameter contract

## Current status

NN 0001 is Complete as an explicitly authorized parallel exception while CPU 0006D remains the
global active frontier. The user authorized this exception because the bounded foundation depends
only on completed `modules/model` Tensor contracts, does not require CPU, Engine, compiler,
runtime, or prepare behavior, and its planned files do not overlap the active CPU task. The
exception is implementation-order only; it does not change the global roadmap, architecture, or
dependency direction. NN work resumes in this master-plan order after 0001.

NN 0004 is intentionally placed immediately before the first `Linear` layer. It is not a
`Parameter` feature: a parameter retains one supplied Tensor; initializer conveniences create a
Tensor that a caller or layer then supplies to that parameter. They must reuse the existing eager
model construction boundary and caller-owned random source rather than add a default generator,
hidden seed, mutable module RNG, or graph-execution requirement.

NN 0002 is Complete under the same authorized parallel exception. It remains model-only and
touched only the NN module plus its own task record and master-plan row; it did not alter the
active CPU work, the global roadmap, Gradle dependencies, or architecture rules. It establishes
exclusive permanent child ownership, collision-free dot-separated recursive paths, immutable
discovery snapshots, and identity-preflighted recursive mode propagation.

NN 0003 is Complete under that same authorized parallel exception. Its bounded replacement API stays
inside the existing NN module and existing model-only dependency edge; it changes no Gradle,
global-roadmap, architecture, CPU, or Engine files. The task deliberately makes replacement a
direct-state operation of the declaring `Module`, rather than a public `Parameter`/`Buffer`
setter, and records the absent checkpoint and concurrency guarantees before a downstream training
extension can consume the stable binding contract.

Detailed NN 0004 is Complete under the same authorized parallel exception. It added only the
planned NN `initialization/` package, focused NN tests, the glossary, and synchronized NN planning
records. Its clean documentation pass finalized the eager ownership, exact fan formula, random-
source lifecycle, `Parameter`, and graph-RNG boundaries and passed generated-Javadoc, Markdown,
scope, and whitespace gates while reusing the stable focused test evidence. Those seven task-owned
files do not overlap the dirty CPU 0007 task, CPU master plan, or global roadmap. Existing Model
constant/random APIs provide the complete eager leaf, caller-owned RNG, floating conversion, and
failure semantics, so NN 0004 adds no Model, build, dependency, architecture, compiler, runtime,
Engine, or backend work.

Detailed NN 0004A is Complete as a bounded review-remediation insertion before NN 0005. The
post-0004 review proved that task 0003's protected direct Module replacement cannot be consumed by
a generic downstream optimizer, that the recursive discovery and mode implementation relies on
the Java call stack, and that the four fan-method Javadocs omit their delegated positive
Java-array-limit failure. Task 0004A closes those findings inside the existing NN and Model-only
boundary: `Parameter` becomes the public schema-validated replacement capability, Module traversal
becomes iterative and identity-defended, and initializer documentation/tests lock the existing
failure side effects. It changes no dependency, architecture, Gradle, CPU, or global-roadmap file.
The isolated implementation pass passed the final focused NN suite with 7 suites and 33 tests.
The independent documentation pass finalized Javadoc, training API, glossary, and planning
evidence and passed generated-Javadoc, Markdown, public-surface, scope, import, and whitespace
checks without repeating the stable executable suite.

Detailed NN 0005 is Complete under the same bounded parallel exception. It adds final public
`Linear`, its `layers` package contract, focused state/initialization/forward tests, the glossary,
and synchronized NN planning in exactly seven paths. Supplied state keeps exact parameter Tensor
identity and strict positive static schemas; initialized state uses explicit-source Glorot uniform
and deterministic zero bias. Forward is mode-insensitive and delegates exactly to the current
Model `Tensor.linear` composition using the bindings observed by that call. The implementation
pass passed the focused 2-suite/11-test selection and final 9-suite/44-test NN suite. The
independent documentation pass finalized glossary and planning wording, reviewed the drafted
Javadocs unchanged, and passed generated-Javadoc, Markdown, public-surface, dependency/import,
scope, and whitespace checks without repeating stable executable tests. There is no later detailed
NN task; the next NN capability remains to be specified at its frontier.

## Open questions

- Define checkpoint/state-dictionary semantics after stable recursive traversal exists.
- Decide whether a concrete future consumer justifies configurable gain, activation, fan mode,
  convolution fan geometry, or an initializer object abstraction. NN 0004 deliberately fixes only
  unit-gain Glorot and fan-in/ReLU Kaiming for positive rank-two Linear weights.

## Decisions made

- `extensions/nn` owns `Module`, `Parameter`, `Buffer`, train/eval behavior, and forward context.
- `extensions/training` depends on nn and owns optimizer algorithms and training orchestration.
- Generic Tensor and operation semantics remain in `modules/model`.
- NN 0001 may run in parallel with CPU 0006D under the recorded isolated-file and model-only
  dependency exception above.
- NN 0002 uses one shared local namespace for direct parameters, buffers, and child modules;
  child attachment is exclusive and permanent, and depth-first declaration order produces
  dot-separated state paths. Local names reserve `.` for the path separator, so recursive state
  paths cannot collide. Its detailed task specifies the resulting APIs and failure behavior.
- NN 0003 replaces the value of exactly one direct binding through protected named `Module`
  methods. The `Parameter` or `Buffer` wrapper and its local name retain identity; a successful
  replacement changes only its current Tensor reference. There is no descriptor-freezing policy,
  version counter, recursive/path replacement, batch atomicity, or thread-safety guarantee in
  this first mutation contract.
- Initializers create eager model leaf Tensors through existing model APIs and receive any random
  source explicitly from the caller. They do not use `GraphRngState`, which represents deferred
  graph-random expressions rather than eager host-data initialization.
- NN 0004 exposes one field-free `ParameterInitializers` class. All eight methods require an
  explicit floating data type and fully static Shape, create fresh unlabeled leaves with
  `requiresGrad == true`, and retain no RNG. Fan-based methods accept only positive rank-two
  `[outFeatures, inFeatures]` weights: Glorot uses unit-gain fan average, and Kaiming uses fixed
  fan-in/ReLU gain. There are no Xavier aliases or ambiguous configurable Kaiming names.
- NN 0004A supersedes only task 0003's assumption that all binding replacement remains protected
  behind the declaring Module. The existing final `Parameter` exposes one public
  `void replace(Tensor)` capability so a generic downstream consumer can update recursively
  discovered parameters. Declaration requires a floating Tensor with `requiresGrad == true`;
  replacement preserves exact declaration-time data type and structural Shape while allowing a
  different Tensor identity, layout, storage, provenance, and label. Buffer replacement remains
  protected and direct through Module. No optimizer algorithm, batch transaction, checkpoint,
  version, or thread-safety contract is introduced.
- NN 0004A preserves recursive discovery and mode preorder while replacing recursive calls with
  explicit-stack depth-first traversal. Parameter and buffer discovery gain the same repeated-
  identity defense as mode propagation, and `train()`/`eval()` retain complete preflight before
  assignment. Qualified prefix text is built only when state is emitted so empty deep chains do
  not retain one String per intermediate level.
- NN 0005 fixes the first `Linear` surface to two caller-Tensor constructors plus one initialized
  constructor with explicit feature counts, bias presence, floating data type, and caller-owned
  random source. Initialized weight delegates to fixed Glorot uniform and optional bias delegates
  to deterministic zeros. The layer exposes stable weight and optional bias `Parameter` handles
  and has one mode-insensitive `forward(Tensor)` that delegates to the existing Model linear
  convenience; it adds no generic layer interface or execution behavior.

## Risks

- Coupling layer composition to optimizer APIs would reverse the architecture dependency.
- Letting train/eval mode become backend residency or per-run execution state would blur the lifecycle boundary.
- Giving NN a default or retained random source would obscure reproducibility, ownership, and
  concurrent-use policy; initializer tasks must keep those responsibilities with the caller.
- A public parameter replacement that omits declaration schema could silently change a layer's
  expected type or Shape; NN 0004A freezes only those logical facts and keeps execution/storage
  facts replaceable.
- Recursive module-tree algorithms must not use Java call-stack depth as a hidden model-size
  limit or accept repeated identities during discovery.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and
follow [the planning guide](../../planning-guide.md).
