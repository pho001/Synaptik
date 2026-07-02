# Repository layout

The repository is organized around the module boundaries defined by [`ARCHITECTURE.md`](../../ARCHITECTURE.md):

- `modules/` contains shared model, compiler, planning, prepare, runtime, engine, configuration, backend contracts, and tracing modules;
- `backends/` contains concrete backend implementations and low-level providers;
- `extensions/` contains optional functionality such as training and ONNX integration;
- `tools/` contains tuning, benchmark, and command-line tooling;
- `native/` contains platform-specific native integration;
- `testing/` contains architecture, backend-conformance, and integration test modules; and
- `docs/` contains explanatory documentation and working implementation plans.

Implementation plans live under [`docs/planning/`](../planning/README.md). They coordinate non-trivial work but do not override the root architecture contract. Production Java packages use the `io.github.pho001.synaptik.*` namespace.
