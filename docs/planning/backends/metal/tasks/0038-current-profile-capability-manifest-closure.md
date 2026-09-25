# Task 0038: Current Metal Profile Capability Manifest Closure

## Status

Complete

Readiness was verified from Metal 0022–0025 Complete, every later evidence task through 0037
finalized Blocked without capability, and a provider/schema audit that found documentation summary
drift only. Execution preserved that boundary and leaves no Ready Metal frontier.

## Change class

Class A — documentation and audit only. The implemented capability, public API, native ABI, schema,
identities, lifecycle, and observable behavior do not change. Public and incorporated architecture
summaries receive a targeted documentation review because they describe a supported boundary.

## Goal

Freeze and synchronize the honest current `STRICT_IEEE` and `ACCELERATOR` Metal occurrence
matrices. Record that strict capability is an accelerator subset, every operation not in the exact
implemented lists remains fail-closed, and no Blocked task becomes capability. Close expansion
until a separately authorized task supplies an exact route/custom kernel or a preceding Model
profile change.

## Exact implemented manifest

The common matrix under both profiles contains only:

- parameterless canonical `FLOAT32` `NEG` and `ABS`;
- `RESHAPE`, `EXPAND`, `PERMUTE`, `EXPAND_DIMS`, and `SQUEEZE` affine transforms;
- the explicit `CONTIGUOUS` canonicalization barrier;
- bounded canonical `FLOAT32` `UNFOLD_AXIS`;
- canonical positive-rank `FLOAT32` data `GATHER` with canonical `INT32` indices;
- canonical positive-rank `INT32`-to-`BOOL` `ONE_HOT`; and
- canonical positive-rank `FLOAT32`/`INT32`/`FLOAT32`
  `SCATTER_ELEMENTS/ScatterReduction.NONE`.

`ACCELERATOR` additionally contains only:

- canonical tensor `FLOAT32` `ADD`, `SUB`, `MUL`, and `DIV`;
- canonical `FLOAT32` `SUM`, `MEAN`, and binding-resolved `SUM_TO_SHAPE`; and
- positive static rank-two `FLOAT32` `MATMUL` with canonical or authenticated exact local
  rank-two-transpose operands.

Every common occurrence has the identical result contract in both profiles. Every strict-positive
query is accelerator-positive. The profile table grants possible operation-local result freedoms;
it does not require Metal support for a listed Model family.

## Exact occurrence boundary

- `MetalCapabilityProvider.supports(OperationCapabilityQuery)` is the public programmatic
  inspection API. `MetalBackendIntegration.capabilityProvider()` exposes the stable immutable
  provider. No enumerable manifest API or operation registry is added.
- Every admitted occurrence has exactly one output. Shapes are fully static and positive rank
  `1..16`, except a locally produced reduction result may be scalar.
- Floating roles are exact `FLOAT32`; indexing roles are canonical `INT32`; `ONE_HOT` produces
  local canonical `BOOL`. Operation-specific Shape, layout, and `requiresGrad` relationships remain
  those already enforced by the provider.
- Only affine/`CONTIGUOUS` inputs may be resolved zero-offset views at occurrence query time;
  partition analysis authenticates local affine provenance. MATMUL transpose candidates likewise
  require exact local-producer authentication.
- All other Model families, types, layouts, ranks, attributes, arities, and profile/operation pairs
  remain false before route selection. No fallback, host repair, or capability inference occurs.

## Frozen package and native state

Current production remains ABI v4 with thirteen exports, the fixed 160-byte node record, node
schema 11, operation wires `1..19`, attribute wires `0..6`, and workload, exact-policy, candidate,
compatibility, route-policy, and codec identities 12. Schema 12, wire 20, attribute 7, INT64,
ABI/export changes, and identity 13 remain conditional, unimplemented, and unreserved.

`MetalMpsGraphProgram.NodeKind` has no blocked-operation row. Java native preflight accepts only the
common node kinds for `STRICT_IEEE` and the current complete node vocabulary for `ACCELERATOR`.
Native code receives only already-authenticated schema records and gains no profile or operation.

## Completed and Blocked evidence boundary

Complete Tasks 0014, 0015, and 0019–0025 account for every implemented row. Historical strict
binary/reduction/MATMUL work remains withdrawn or Blocked; separately proven accelerator Tasks
0015, 0020, and 0021 own those current rows. Task 0019 owns common exact `ABS`; the remaining unary
work stays Blocked.

Tasks 0006–0007, 0009–0013, 0016–0018, 0026–0027, and 0030–0037 add no current capability.
Specifically, Task 0027 BatchNorm remains Blocked. This task closes only the separate
capability-profile audit and must not mark, imply, wire, or document BatchNorm as implemented.

## Non-goals

- Adding, removing, widening, or narrowing any capability answer.
- Production, native, test, build-configuration, generated, ABI, schema, identity, or route changes.
- A second public manifest, enumerable operation registry, reflection scan, source-text test, or
  duplicate capability implementation.
- Reopening a Blocked task, transferring Model profile freedom to another operation, or inferring
  implementation from MPSGraph/API availability.
- Device probes, numerical or Shape matrices, benchmarks, native builds, or a full project build.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) keeps
  capability truthful and Model as semantic owner.
- [Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
  owns strict-subset monotonicity and the sole operation-family result-freedom table.
- [Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend) and
  [Numerical-profile backend identity](../../../../architecture/contracts/backend-execution.md#numerical-profile-backend-identity)
  own the implemented backend boundary.
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
  keeps profile-qualified route selection cold and Runtime profile-free.

## Dependencies and integration

- Depends on: Metal 0022–0025 Complete; finalized planning evidence through Metal 0037; Model
  0027–0029; Config 0006; Engine 0018.
- Conflicts with: every concurrent Metal capability/schema/native/candidate/codec documentation
  scope and edits to the shared README, architecture, API, backend, user, master, or roadmap files.
- Parallel group: None.
- Common base revision: `ec4499bd9e18ed863c9091baa9cfe82dbcf06a9b`.
- Integration order: planning authorization commit, then one documentation/audit closure commit.
- Integration validation: focused existing Metal capability/conformance checks, targeted architecture
  checks, Markdown/link/terminology validation, `git diff --check`, and documentation-only scope.
- Shared-document integration owner: Task 0038 executor.

## Files and documentation ownership

Create and complete this brief; synchronize:

- `README.md`;
- `docs/architecture/contracts/backend-execution.md`;
- `docs/architecture/current-architecture-plan.md`;
- `docs/architecture/lifecycle.md`;
- `docs/architecture/runtime-prepare-backend-boundary.md`;
- `docs/api/tensor-api.md`;
- `docs/backend-guide/capability-provider.md`;
- both stale profile summaries in `docs/user-guide/compiling-graphs.md`;
- `docs/planning/backends/metal/master-plan.md`; and
- `docs/planning/roadmap.md`.

The already-exact provider, package Javadoc, schema/native code and guide, public API guide, Metal
backend guide, module-boundary summary, partition-preparer guide, tests, and normative profile table
remain unchanged.

## Acceptance criteria

- Every changed current-facing summary names the complete common and accelerator-only lists above,
  states strict-subset monotonicity, and leaves all unlisted occurrences fail-closed.
- Root README corrects both the omitted Tasks 0023–0025 rows and the false strict-only custom NEG
  wording; an eligible singleton NEG route is available under either profile.
- The architecture summary matches the already-correct detailed Metal contract without converting
  profile permission into capability.
- ABI/schema/identity freeze and the absence of every Blocked row are explicit.
- Task 0027 BatchNorm and all other Blocked tasks remain Blocked. Task 0038 becomes Complete and no
  Metal task remains Ready.
- No production, native, test, build, generated, schema, identity, or probe artifact changes.

## Validation

Run only:

```bash
./gradlew :backends:metal:test \
  --tests 'io.github.pho001.synaptik.backend.metal.MetalCapabilityProviderTest' \
  --tests 'io.github.pho001.synaptik.backend.metal.MetalAffineCapabilityTest'
./gradlew :testing:backend-conformance:test \
  --tests 'io.github.pho001.synaptik.testing.conformance.MetalNegCapabilityPartitionConformanceTest'
./gradlew :testing:architecture-tests:test \
  --tests 'io.github.pho001.synaptik.testing.architecture.BackendConformanceDependencyContractTest' \
  --tests 'io.github.pho001.synaptik.testing.architecture.EngineCompositionContractTest'
```

Then validate changed Markdown links/anchors/fences/final newlines, exact terminology, final diff,
and documentation-only scope. Run no device probe, native build, formatter, full test suite, or full
build.

## Result

Completed as a documentation/audit-only closure from planning commit `1bb2a0e5`. Root, architecture,
API, backend, user, master, and roadmap summaries now publish the exact common and accelerator-only
occurrence matrices, strict-subset monotonicity, fail-closed remainder, and frozen ABI/schema/
identity boundary. Task 0027 BatchNorm and every other Blocked row remain unsupported. The three
focused Gradle invocations passed; Markdown links, anchors, fences, final newlines, terminology,
final diff, and documentation-only scope passed. No device probe, native build, full suite, or full
build ran.
