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
- optimizer algorithms, parameter updates, and training sessions
- backend storage, lowering, kernel selection, and backend-specific optimizer execution

## Module invariants

- NN owns module-declared `Parameter` and `Buffer` values.
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

The initial public surface is deliberately confined to `module/`:

- `Module` owns direct named state declarations and its local train/eval mode; it has no universal
  forward signature.
- `Parameter` and `Buffer` retain a named current `Tensor` binding without becoming `Tensor`
  subtypes.
- `ForwardMode` and immutable `ForwardContext` provide the mode snapshot supplied to a concrete
  layer's own typed forward method.

`layers/` and `functional/` remain empty until their own ready tasks. Module-tree traversal,
recursive mode propagation, state dictionaries, binding replacement, and initialization remain
later work. The `initialization/` package is reserved for a focused task rather than making
`Parameter` responsible for creating its own value.

## Task list

| ID | Task | Status | Depends on | Summary |
|---|---|---|---|---|
| [0001](tasks/0001-module-parameter-buffer-and-forward-context-foundation.md) | Module, parameter, buffer, and forward-context foundation | Complete | Completed `modules/model` Tensor contracts; ADR 0007 | Created the NN Gradle module and its minimal model-only state/mode API, activated the required one-way training build edge, and enforced it with architecture tests; it adds no layers or execution behavior. |
| [0002](tasks/0002-module-tree-ownership-and-recursive-mode-propagation.md) | Module-tree ownership and recursive mode propagation | Complete | 0001 | Added exclusively owned child modules, deterministic collision-free dot-path parameter/buffer snapshots, and atomic recursive `train()`/`eval()` propagation without checkpoint or optimizer behavior. |
| 0003 | Validated parameter and buffer binding replacement | Draft | 0002 | Define controlled replacement of a current Tensor binding, including validation and concurrency/checkpoint interaction, without optimizer algorithms. |
| 0004 | Explicit eager parameter initializers | Draft | 0001; completed model `TensorRandoms` contracts | Add small zero/one and floating uniform/normal initializer conveniences over fully static Shapes and caller-owned `RandomGenerator` sources; add fan-in/fan-out Xavier/Glorot and Kaiming policies only when their type, shape, and numerical conventions are fully specified. |
| 0005 | Linear layer | Draft | 0001, 0004; completed model `Tensor.linear` | Add the first stateful layer with explicitly initialized or caller-supplied `weight` and optional `bias`; validate only Tensor-expression ownership/provenance until numerical execution coverage is available. |

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

## Open questions

- Define controlled parameter/buffer binding replacement with training, checkpoint, and concurrency
  contracts before exposing mutation.
- Define checkpoint/state-dictionary semantics after stable recursive traversal exists.
- Select the minimum initializer public surface and the precise fan-in/fan-out rules for rank-two
  linear weights before NN 0004 becomes Ready; convolution and other layout-specific policies are
  later work.

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
- Initializers create eager model leaf Tensors through existing model APIs and receive any random
  source explicitly from the caller. They do not use `GraphRngState`, which represents deferred
  graph-random expressions rather than eager host-data initialization.

## Risks

- Coupling layer composition to optimizer APIs would reverse the architecture dependency.
- Letting train/eval mode become backend residency or per-run execution state would blur the lifecycle boundary.
- Giving NN a default or retained random source would obscure reproducibility, ownership, and
  concurrent-use policy; initializer tasks must keep those responsibilities with the caller.

## Notes

Keep this master plan concise. Put executable work in small task specifications under `tasks/` and
follow [the planning guide](../../planning-guide.md).
