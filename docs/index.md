# Synaptik documentation

The authoritative architecture contract is [`ARCHITECTURE.md`](../ARCHITECTURE.md). Documentation in this directory explains the architecture, APIs, workflows, and implementation plans without overriding that contract.

## Start here

- [Getting started](getting-started.md)
- [Glossary](glossary.md)
- [Architecture overview](architecture/overview.md)
- [Current architecture documentation index](architecture/current-architecture-plan.md)
- [Implementation plans](planning/README.md)

The current implementation contains model value foundations only. Pages about compilation, preparation, runtime, backends, tracing, and training explain planned architecture unless they explicitly say an API is implemented.

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
