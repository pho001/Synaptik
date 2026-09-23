# Synaptik documentation

[`ARCHITECTURE.md`](../ARCHITECTURE.md) is the authoritative architecture root and sole authority
index. It incorporates exactly six scoped normative contracts under
[`architecture/contracts/`](architecture/contracts/). All other documentation in this directory
explains the architecture, APIs, workflows, and implementation plans without overriding that
contract.

## Start here

- [Getting started](getting-started.md)
- [Glossary](glossary.md)
- [Architecture overview](architecture/overview.md)
- [Current architecture documentation index](architecture/current-architecture-plan.md)
- [Implementation plans](planning/README.md)

The current implementation includes a runnable public lifecycle. `Engine.standard()` supplies
fixed CPU ownership, while `Engine.builder()` explicitly owns opened CPU and/or Metal integrations
and executes each complete plan through one registered owner. Tensor expressions can be compiled,
prepared once, run repeatedly with isolated invocation state, and materialized as detached host
values. One-shot forward computation and a bounded scalar-objective backward convenience are also
current. Metal executes supported static `FLOAT32` negation partitions through MPSGraph or a
custom singleton route. A standard-Metal convenience, mixed-owner execution, generic plugin
registration/discovery, broader Metal coverage, CUDA, training orchestration, persistence, and
generic graph/plan tuning remain planned. Each focused page distinguishes current contracts from
those future capabilities.

## Contributor guides

- [Documentation rules](developer-guide/documentation-rules.md)
- [Documentation style profiles](developer-guide/documentation/README.md)
- [Coding rules](developer-guide/coding-rules.md)

## Documentation areas

- **Architecture:** [overview](architecture/overview.md), [lifecycle](architecture/lifecycle.md), [module boundaries](architecture/module-boundaries.md), [dependency rules](architecture/dependency-rules.md), [partition scoring](architecture/partition-scoring.md), [training graph](architecture/training-graph.md), [tracing](architecture/tracing.md), and [runtime/prepare/backend boundary](architecture/runtime-prepare-backend-boundary.md).
- **API reference:** [public API status](api/public-api.md), [tensor model](api/tensor-api.md), [compile](api/compile-api.md), [runtime](api/runtime-api.md), and [training](api/training-api.md).
- **User guides:** [tensors](user-guide/tensors.md), [compiling graphs](user-guide/compiling-graphs.md), [backend selection](user-guide/backend-selection.md), [preparing execution](user-guide/preparing-execution.md), [running models](user-guide/running-models.md), [autograd](user-guide/autograd.md), and [training](user-guide/training.md).
- **Backend guides:** [writing a backend](backend-guide/writing-a-backend.md), [capabilities](backend-guide/capability-provider.md), [partition preparation](backend-guide/partition-preparer.md), [kernel routes](backend-guide/kernel-routes.md), [CPU](backend-guide/cpu-backend.md), [Metal](backend-guide/metal-backend.md), and [CUDA](backend-guide/cuda-backend.md).
- **Developer guides:** [repository layout](developer-guide/repository-layout.md), [coding rules](developer-guide/coding-rules.md), [architecture tests](developer-guide/architecture-tests.md), [debugging with traces](developer-guide/debugging-trace.md), [benchmarking](developer-guide/benchmarking.md), [release process](developer-guide/release-process.md), and [documentation rules](developer-guide/documentation-rules.md).
- **Design records:** the [design index](design/README.md) links every architecture decision record and strategy note.
- **Planning:** the [planning index](planning/README.md), [planning guide](planning/planning-guide.md), and [roadmap](planning/roadmap.md) lead to every master plan and executable task specification.
