# Public compile output values

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileArtifacts.java)

> **Scope:** Model construction, Compiler admission, backend execution, and checkpoint persistence are separate gates. Execution-status sentences retained in this detailed material record earlier qualification slices; they are not an exhaustive current support matrix. The [current backend matrix](../../backend-guide/production-support.md) and its incorporated contracts supersede narrower historical CPU/Metal lists. Package-private implementation sketches are not supported application API.

The public `modules:compiler` contract surface now contains:

- `GraphCompilationPort`, the narrow constant-free cross-module compile SPI used by Engine
  composition, not an ordinary-user lifecycle facade;
- `FunctionalGradientRequest`, the immutable one/two-stage reverse-mode input with aligned
  cotangent seeds and an explicit disconnected-target policy;
- `GradientPublicationBinding`, the derivative-order and stage-local
  target-to-gradient-publication association;
- `DerivativeGraphMetadata`, the exact graph-adjacent per-node derivative-order sidecar, including
  `scanActivityByNode()` associations from containing Scan IDs to exact local activity plans;
- `CompileArtifacts`, the exact eight-component immutable recipe containing mode, final graph,
  maximal partitions, logical memory, publication, constants, diagnostics, and derivative
  metadata;
- `PublicationPlan`, an output-only exact-graph context exposing separate immutable ordered
  `forwardBindings()` of `ForwardPublicationBinding` values and `gradientBindings()` of
  `GradientPublicationBinding` values;
- `CompileConstantPlan`, an output-only immutable graph-input classification with nested
  `BindableInput(TensorId, ValueId)` associations and `ConstantSource(ValueId, ScalarValue)`
  values; and
- `CompileDiagnostics`, an output-only immutable successful-compile projection with nested
  `DeferredConstraintDiagnostic(NodeId, subject, predicate)` values.

The three output-only plan classes have package-private constructors. `CompileArtifacts` is a public
record whose canonical constructor cross-validates the supplied components and snapshots
partition membership. These types are public so later prepare and engine work can consume stable
compile-time data. `GraphCompilationPort.compile(...)` makes complete constant-free compilation
callable for cross-module integration while both `GraphCompiler` entries and `GraphCompilation`
remain package-private. The port always treats reachable provenance-free forward leaves as
caller-bindable, retains no provider or availability snapshot, and neither prepares nor executes
its result.

The artifact result retains no provider, availability snapshot, selected device, route, kernel,
physical buffer, transfer, prepared schedule, executable, residency, or mutable run state.
`PublicationPlan` validates graph membership and boundary order but adds no delivery policy.
`CompileConstantPlan.bindableInputBindings()` associates every final caller-bindable graph input
with its originating immutable Tensor identity in graph-input order. Its existing
`bindableInputs()` method remains the ordered `ValueId` projection for compatibility, and its
constant entries carry logical splats rather than dense data or storage. No source entry retains a
live Tensor, descriptor copy, provenance, or host storage.
`CompileDiagnostics` carries deterministic text projections rather than a public predicate
language, trace schema, or serialization.


---

[Previous topic](planning-values.md) · [Next topic](../engine/cpu-integration.md)
