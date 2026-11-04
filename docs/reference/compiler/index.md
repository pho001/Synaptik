# [Compiler](../../glossary.md#term-compiler) contracts and boundaries

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileArtifacts.java)

[Compilation](../../glossary.md#compiler-engine-and-execution-stage-names) turns an expression's meaning into a verified logical recipe. **[Capture](../../glossary.md#term-capture-meanings)** records
its dependencies, **[inference](../../glossary.md#term-inference)** derives expected [shapes](../../glossary.md#shape) and types, and **[planning](../../glossary.md#term-planning)** assigns [backend](../../glossary.md#backend)
[owners](../../glossary.md#term-owner) and groups consecutive owned work. **[Lowering](../../glossary.md#lowering)** is the later backend-specific translation
into executable operations; it is not the same as capture or logical planning.

<a id="purpose-and-implementation-status"></a>
## Purpose and implementation status

This reference separates the [compile](../../glossary.md#compile)-time contracts implemented today from later [Engine](../../glossary.md#term-engine)
conveniences. Ordinary `Engine` exposes forward-only and explicitly seeded first-order compile
overloads. `Engine.standard()` supplies fixed [CPU](../../glossary.md#term-cpu) [ownership](../../glossary.md#term-ownership-meanings), while `Engine.builder()` compiles
against an explicit frozen CPU/[Metal](../../glossary.md#term-metal) inventory. The [compiler](../../glossary.md#term-compiler) [module](../../glossary.md#term-module) contains package-private
`GraphCompiler`. Its five-[argument](../../glossary.md#term-argument) entry compiles a forward-only or combined one/two-stage
functional [derivative](../../glossary.md#term-derivative) [Tensor](../../glossary.md#tensor) expression into package-private [immutable](../../glossary.md#term-immutable) `GraphCompilation`. A
second package-private nine-argument entry completes [publication](../../glossary.md#term-publication) and [backend](../../glossary.md#backend)-neutral [planning](../../glossary.md#term-planning) and
returns public immutable `CompileArtifacts`. Advanced Engine exposes that complete [artifact](../../glossary.md#term-artifact) path
as an explicit CPU-only composition seam.

The current stages are [fail-closed](../../glossary.md#term-fail-closed) [autograd](../../glossary.md#autograd) [preflight](../../glossary.md#term-preflight), formula construction through public Tensor
operations, one phase-aware [capture](../../glossary.md#term-capture-meanings), [binding](../../glossary.md#term-binding)-free captured-[graph](../../glossary.md#graph) verification with retained
[occurrence](../../glossary.md#term-occurrence)-local [Shape](../../glossary.md#shape) predicates, mandatory dense
[canonicalization](../../glossary.md#occurrence-identity-and-canonical-form), explicit logical-splat facts, one bounded exact whole-[graph optimization](../../glossary.md#forward-graph-optimization)
pipeline, three bounded final logical-[descriptor](../../glossary.md#term-descriptor) closures, publication-role [validation](../../glossary.md#term-validation), one [owner](../../glossary.md#term-owner)
selection per final graph [node](../../glossary.md#node), maximal same-owner partitioning, logical-memory derivation, and
immutable artifact assembly. Public
`GraphCompilationPort` now exposes the complete [constant](../../glossary.md#term-constant)-free pipeline as a narrow cross-module
service-provider interface ([SPI](../../glossary.md#term-service-provider-interface-spi)). It is not the recommended application [API](../../glossary.md#term-application-programming-interface-api) and does not provide
the still-planned `CompileConfig` aggregate. The same port also provides bounded cold raw source
[capture](../../glossary.md#term-structural-capture) and reconstruction, separate from complete artifact [compilation](../../glossary.md#compiler-engine-and-execution-stage-names). `AdvancedEngine` uses
this port and returns an [opaque](../../glossary.md#term-opaque) owner-bound `AdvancedCompiledGraph`, but does not expose the
underlying `CompileArtifacts`. The compiler also provides the public immutable
`FunctionalGradientRequest` input value used by this integration boundary. Ordinary Engine
callers do not construct that request: the three-list overload maps their explicit outputs,
aligned [cotangent](../../glossary.md#cotangent) [seeds](../../glossary.md#commit-release-distribution-log-and-seed-meanings), and ordered [targets](../../glossary.md#term-target) to one fixed first-order stage.
Final [compile artifacts](../../glossary.md#compile-artifacts) now associate every caller-bindable graph input with the immutable
[`TensorId`](../../glossary.md#tensorid) of its originating logical caller Tensor. The ordinary Engine uses that [metadata](../../glossary.md#term-metadata) for
current logical binding; the association itself retains no Tensor, descriptor copy, [host storage](../../glossary.md#host-storage),
or [runtime](../../glossary.md#term-runtime) representation.
The current config module provides four immutable standalone values that a later compile
configuration aggregate can contain: [`BackendIntent`](../../glossary.md#backend-intent--backendintent), [`CompileMode`](../../glossary.md#compile-mode--compilemode),
[`GraphOptimizationConfig`](../../glossary.md#graph-optimization-configuration--graphoptimizationconfig), and [`PartitionScoringConfig`](../../glossary.md#partition-scoring-configuration--partitionscoringconfig). [Model](../../glossary.md#term-model) alone owns the one profile-free
family/[dtype](../../glossary.md#data-type--datatype) semantic contract. Compiler-generated [gradients](../../glossary.md#term-gradient) use ordinary public Tensor
operations under that contract, with exact guards and saved values and no gradient-only output
envelope. [Finite](../../glossary.md#term-finite) tolerances are test-only and never license a Compiler rewrite.
The current planning module provides the immutable occurrence-specific
[`OperationCapabilityQuery`](../../glossary.md#operation-capability-query--operationcapabilityquery) and the explicitly supplied [`BackendCapabilityProvider`](../../glossary.md#backend-capability-provider--backendcapabilityprovider) collaboration.
It keeps per-query hard eligibility and baseline comparison package-private and exposes one public
`BackendOwnerPlanning.selectOwner(...)` collaboration that composes them without exposing the
intermediate. It also exposes the existing
`MaximalSameOwnerPartitioning.partition(...)` and `LogicalMemoryPlanning.plan(...)` operations in
their owning packages. The module still provides no reusable/public capability [matrix](../../glossary.md#term-review-matrix), public
graph-wide planner workflow, general cost scoring, or owner-map assembly. Compiler owns the
graph-wide loop and invokes those three Planning operations directly.

Compilation answers two questions: what the computation means, and which [backend identity](../../glossary.md#backend-identity--backendid) owns
each planned region. It will not create physical [buffers](../../glossary.md#term-buffer), choose concrete [kernels](../../glossary.md#kernel), or construct
prepared executables.


---

[Previous topic](batch-binding.md) · [Next topic](entry-points.md)

## Topic navigation

- [Named-batch metadata binding](batch-binding.md)<span data-glossary-navigation="true"> ([batch glossary](../../glossary.md#term-batch-machine-learning-and-hyperparameters); [metadata glossary](../../glossary.md#term-metadata); [binding glossary](../../glossary.md#term-binding))</span>
- [Ordinary and advanced compile entry points](entry-points.md)<span data-glossary-navigation="true"> ([compile glossary](../../glossary.md#compile))</span>
- [Structural capture and model contracts](capture.md)<span data-glossary-navigation="true"> ([Structural capture glossary](../../glossary.md#term-structural-capture); [model glossary](../../glossary.md#term-model))</span>
- [Cold source capture and reconstruction](cold-source.md)
- [Verification inference and scoped Scan](verification-scan.md)<span data-glossary-navigation="true"> ([inference glossary](../../glossary.md#term-inference); [Scan glossary](../../glossary.md#term-cumulative-scan))</span>
- [Canonicalization and whole-graph optimization](optimization.md)<span data-glossary-navigation="true"> ([Canonicalization glossary](../../glossary.md#occurrence-identity-and-canonical-form); [graph optimization glossary](../../glossary.md#forward-graph-optimization))</span>
- [Pre-capture autograd and functional gradients](autograd.md)<span data-glossary-navigation="true"> ([capture glossary](../../glossary.md#term-capture-meanings))</span><span data-glossary-navigation="true"> ([autograd glossary](../../glossary.md#autograd); [gradients glossary](../../glossary.md#term-gradient))</span>
- [Complete artifact compilation and integration port](artifacts.md)<span data-glossary-navigation="true"> ([artifact glossary](../../glossary.md#term-artifact); [compilation glossary](../../glossary.md#compiler-engine-and-execution-stage-names); [integration port glossary](../../glossary.md#term-integration-port))</span>
- [Compile-configuration inputs](configuration.md)
- [Capability queries and planned partitions](planning.md)
- [Logical memory and compile outputs](memory-output.md)
- [Internal lifecycle, failures and interpretation](lifecycle.md)<span data-glossary-navigation="true"> ([lifecycle glossary](../../glossary.md#lifecycle))</span>
- [Declarative compilation and tuning values](configuration-values.md)
- [Public capability and planning values](planning-values.md)<span data-glossary-navigation="true"> ([planning glossary](../../glossary.md#term-planning))</span>
- [Public compile output values](artifact-values.md)
