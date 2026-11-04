# Public [compile](../../glossary.md#compile) output values

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileArtifacts.java)

These values are [compilation](../../glossary.md#compiler-engine-and-execution-stage-names) results for other [modules](../../glossary.md#term-module) to read. An **[artifact](../../glossary.md#term-artifact)** here is an
[immutable](../../glossary.md#term-immutable) description of the final computation and its plans, not a generated [binary](../../glossary.md#term-binary-descriptor-and-metadata) or saved
[checkpoint](../../glossary.md#term-checkpoint) file. A **port** is the narrow [API](../../glossary.md#term-application-programming-interface-api) another module calls across this boundary; [SPI](../../glossary.md#term-service-provider-interface-spi) means
service-provider interface, an integration contract rather than the usual application entry point.

The public `modules:compiler` contract surface now contains:

- `GraphCompilationPort`, the narrow constant-free cross-module compile SPI used by Engine
  composition, not an ordinary-user lifecycle facade;<span data-glossary-navigation="true"> ([lifecycle glossary](../../glossary.md#lifecycle); [facade glossary](../../glossary.md#term-facade))</span>
- `FunctionalGradientRequest`, the immutable one/two-stage reverse-mode input with aligned
  cotangent seeds and an explicit disconnected-target policy;<span data-glossary-navigation="true"> ([reverse-mode glossary](../../glossary.md#primal-adjoint-and-differentiation-seed); [cotangent glossary](../../glossary.md#cotangent); [seeds glossary](../../glossary.md#commit-release-distribution-log-and-seed-meanings); [target glossary](../../glossary.md#term-target))</span>
- `GradientPublicationBinding`, the derivative-order and stage-local
  target-to-gradient-publication association;<span data-glossary-navigation="true"> ([gradient glossary](../../glossary.md#term-gradient); [publication glossary](../../glossary.md#term-publication))</span>
- `DerivativeGraphMetadata`, the exact [graph](../../glossary.md#graph)-adjacent per-[node](../../glossary.md#node) [derivative](../../glossary.md#term-derivative)-order sidecar, including
  `scanActivityByNode()` associations from containing [Scan](../../glossary.md#term-cumulative-scan) IDs to exact local activity plans;
- `CompileArtifacts`, the exact eight-component immutable recipe containing mode, final graph,
  maximal partitions, logical memory, publication, constants, diagnostics, and derivative
  metadata;<span data-glossary-navigation="true"> ([metadata glossary](../../glossary.md#term-metadata))</span>
- `PublicationPlan`, an output-only exact-graph context exposing separate immutable ordered
  `forwardBindings()` of `ForwardPublicationBinding` values and `gradientBindings()` of
  `GradientPublicationBinding` values;
- `CompileConstantPlan`, an output-only immutable graph-input classification with nested
  `BindableInput(TensorId, ValueId)` associations and `ConstantSource(ValueId, ScalarValue)`
  values; and
- `CompileDiagnostics`, an output-only immutable successful-compile projection with nested
  `DeferredConstraintDiagnostic(NodeId, subject, predicate)` values.

The three output-only plan classes have package-private constructors. `CompileArtifacts` is a public
record whose [canonical](../../glossary.md#term-canonical-form) constructor cross-validates the supplied components and snapshots
[partition](../../glossary.md#partition) membership. These types are public so later [prepare](../../glossary.md#prepare) and [engine](../../glossary.md#term-engine) work can consume stable
compile-time data. `GraphCompilationPort.compile(...)` makes complete [constant](../../glossary.md#term-constant)-free compilation
callable for cross-module integration while both `GraphCompiler` entries and `GraphCompilation`
remain package-private. The port always treats reachable [provenance](../../glossary.md#provenance)-free forward leaves as
caller-bindable, retains no provider or availability [snapshot](../../glossary.md#term-snapshot), and neither prepares nor executes
its result.

The artifact result retains no provider, availability snapshot, selected [device](../../glossary.md#term-device), route, [kernel](../../glossary.md#kernel),
physical [buffer](../../glossary.md#term-buffer), transfer, [prepared schedule](../../glossary.md#prepared-schedule--preparedschedule), executable, [residency](../../glossary.md#residency), or [mutable](../../glossary.md#term-mutable) [run state](../../glossary.md#run-state--runstate).
`PublicationPlan` validates graph membership and boundary order but adds no delivery policy.
`CompileConstantPlan.bindableInputBindings()` associates every final caller-bindable graph input
with its originating immutable [Tensor](../../glossary.md#tensor) [identity](../../glossary.md#term-identity-meanings) in graph-input order. Its existing
`bindableInputs()` method remains the ordered [`ValueId`](../../glossary.md#logical-valueid-versus-physical-memory-slot) projection for compatibility, and its
constant entries [carry](../../glossary.md#term-carry) logical splats rather than dense data or storage. No source entry retains a
live Tensor, [descriptor](../../glossary.md#term-descriptor) copy, provenance, or [host storage](../../glossary.md#host-storage).
`CompileDiagnostics` carries deterministic text projections rather than a public predicate
language, [trace](../../glossary.md#trace) [schema](../../glossary.md#term-schema), or [serialization](../../glossary.md#term-serialization).


---

[Previous topic](planning-values.md) · [Next topic](../engine/cpu-integration.md)
