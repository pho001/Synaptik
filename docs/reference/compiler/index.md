# Compiler contracts and boundaries

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileArtifacts.java)

> **Scope:** Model construction, Compiler admission, backend execution, and checkpoint persistence are separate gates. Execution-status sentences retained in this detailed material record earlier qualification slices; they are not an exhaustive current support matrix. The [current backend matrix](../../backend-guide/production-support.md) and its incorporated contracts supersede narrower historical CPU/Metal lists. Package-private implementation sketches are not supported application API.

<a id="purpose-and-implementation-status"></a>
## Purpose and implementation status

This reference separates the compile-time contracts implemented today from later Engine
conveniences. Ordinary `Engine` exposes forward-only and explicitly seeded first-order compile
overloads. `Engine.standard()` supplies fixed CPU ownership, while `Engine.builder()` compiles
against an explicit frozen CPU/Metal inventory. The compiler module contains package-private
`GraphCompiler`. Its five-argument entry compiles a forward-only or combined one/two-stage
functional derivative Tensor expression into package-private immutable `GraphCompilation`. A
second package-private nine-argument entry completes publication and backend-neutral planning and
returns public immutable `CompileArtifacts`. Advanced Engine exposes that complete artifact path
as an explicit CPU-only composition seam.

The current stages are fail-closed autograd preflight, formula construction through public Tensor
operations, one phase-aware capture, binding-free captured-graph verification with retained
occurrence-local Shape predicates, mandatory dense
canonicalization, explicit logical-splat facts, one bounded exact whole-graph optimization
pipeline, three bounded final logical-descriptor closures, publication-role validation, one owner
selection per final graph node, maximal same-owner partitioning, logical-memory derivation, and
immutable artifact assembly. Public
`GraphCompilationPort` now exposes the complete constant-free pipeline as a narrow cross-module
service-provider interface (SPI). It is not the recommended application API and does not provide
the still-planned `CompileConfig` aggregate. The same port also provides bounded cold raw source
capture and reconstruction, separate from complete artifact compilation. `AdvancedEngine` uses
this port and returns an opaque owner-bound `AdvancedCompiledGraph`, but does not expose the
underlying `CompileArtifacts`. The compiler also provides the public immutable
`FunctionalGradientRequest` input value used by this integration boundary. Ordinary Engine
callers do not construct that request: the three-list overload maps their explicit outputs,
aligned cotangent seeds, and ordered targets to one fixed first-order stage.
Final compile artifacts now associate every caller-bindable graph input with the immutable
`TensorId` of its originating logical caller Tensor. The ordinary Engine uses that metadata for
current logical binding; the association itself retains no Tensor, descriptor copy, host storage,
or runtime representation.
The current config module provides four immutable standalone values that a later compile
configuration aggregate can contain: `BackendIntent`, `CompileMode`,
`GraphOptimizationConfig`, and `PartitionScoringConfig`. Model alone owns the one profile-free
family/dtype semantic contract. Compiler-generated gradients use ordinary public Tensor
operations under that contract, with exact guards and saved values and no gradient-only output
envelope. Finite tolerances are test-only and never license a Compiler rewrite.
The current planning module provides the immutable occurrence-specific
`OperationCapabilityQuery` and the explicitly supplied `BackendCapabilityProvider` collaboration.
It keeps per-query hard eligibility and baseline comparison package-private and exposes one public
`BackendOwnerPlanning.selectOwner(...)` collaboration that composes them without exposing the
intermediate. It also exposes the existing
`MaximalSameOwnerPartitioning.partition(...)` and `LogicalMemoryPlanning.plan(...)` operations in
their owning packages. The module still provides no reusable/public capability matrix, public
graph-wide planner workflow, general cost scoring, or owner-map assembly. Compiler owns the
graph-wide loop and invokes those three Planning operations directly.

Compilation answers two questions: what the computation means, and which backend identity owns
each planned region. It will not create physical buffers, choose concrete kernels, or construct
prepared executables.


---

[Previous topic](batch-binding.md) · [Next topic](entry-points.md)

## Topic navigation

- [Named-batch metadata binding](batch-binding.md)
- [Ordinary and advanced compile entry points](entry-points.md)
- [Structural capture and model contracts](capture.md)
- [Cold source capture and reconstruction](cold-source.md)
- [Verification inference and scoped Scan](verification-scan.md)
- [Canonicalization and whole-graph optimization](optimization.md)
- [Pre-capture autograd and functional gradients](autograd.md)
- [Complete artifact compilation and integration port](artifacts.md)
- [Operation capture and numerical qualification notes](operation-qualification.md)
- [Recurrent and geometry qualification notes](geometry-qualification.md)
- [Compile-configuration inputs](configuration.md)
- [Capability queries and planned partitions](planning.md)
- [Logical memory and compile outputs](memory-output.md)
- [Internal lifecycle, failures and interpretation](lifecycle.md)
- [Declarative compilation and tuning values](configuration-values.md)
- [Public capability and planning values](planning-values.md)
- [Public compile output values](artifact-values.md)
