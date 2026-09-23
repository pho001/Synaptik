# Task 0006B11: Static-Result Logical-Layout Closure for Public CPU Execution

## Status

Complete

## Authorization and classification

- User-authorized benchmark-blocker fix from exact main commit
  `aba8eaa8d2c0cfdf9b3fbcddde508485ffecde3f`.
- Class B behavior change: compiler artifacts presented to capability providers change for eligible
  fully static operation outputs, and CPU complete-partition validation corrects one fail-closed
  structural comparison.
- Independent review is required before the local commit.

## Goal

Make ordinary fully static public Tensor expressions reachable through `Engine.compute(...)` and
reusable `compile -> prepare -> run -> materialize` on CPU without package-private provenance or
manually injected derived layouts, while retaining semantic and complete-partition fail-closed
checks.

## Diagnosis

1. Model inference intentionally leaves ordinary derived Tensor layouts unresolved.
2. `CpuCapabilityProvider` correctly rejects every occurrence whose input or output layout is not
   statically resolved, so ownership selection rejected pointwise, matmul, reduction, and
   normalization expressions before CPU preparation.
3. Existing published-constant, convolution, and NEG-specific compiler passes showed that final
   logical descriptor completion belongs before capability queries, but the NEG pass covered only
   one operation family.
4. Appending public `contiguous()` did not solve the root cause because the inner operation
   occurrence remained unresolved.
5. Once general closure made both occurrences eligible, a second existing defect rejected valid
   multi-unit pointwise-to-affine plans: general topology validation compared affine generated
   address-table access plans with physical boundary bindings even though those are intentionally
   different validated domains.

## Contracts

- `ARCHITECTURE.md`: compile owns logical inference/canonicalization; backends own physical
  representation, resource preparation, and execution.
- `docs/architecture/contracts/compiler-autograd.md`: Compiler may complete logical descriptors but
  may not allocate physical buffers or create backend executables.
- `docs/architecture/contracts/backend-execution.md`: CPU capability and complete-partition lowering
  remain responsible for exact semantic, shape, layout, topology, carrier, and resource checks.
- `docs/architecture/contracts/runtime-prepare-engine.md`: proof must use the public Engine lifecycle
  rather than package-private graph seams.

## Implementation

- Replace the NEG-only final pass with a backend-neutral, explicit allowlist of operation families
  whose results are newly materialized logical values, plus the semantic `CONTIGUOUS` request and
  explicit compile-time splat graph inputs. Unknown future kinds fail closed.
- Exclude affine/view results (`RESHAPE`, `EXPAND`, `PERMUTE`, `EXPAND_DIMS`, `SQUEEZE`, slice
  extraction, and scalar `SELECT`) so their input-derived offsets and non-contiguous or zero
  strides are preserved or remain unresolved.
- Preserve caller-bindable inputs, dynamic values, resolved layouts, graph topology, identities,
  sidecars, and all operation references.
- Keep specialized convolution closure before the operation-aware pass because Conv1d's direct
  squeeze view needs exact semantic propagation from its convolution producer.
- Keep capability providers strict; logical completion grants no operation, dtype, shape, semantic,
  or backend support.
- Exempt affine-copy units from the invalid general access-plan equality check. Affine lowering and
  plan construction continue to validate address-table bounds and physical geometry; only the
  comparison between generated address-table and boundary-binding domains is removed.
- Exercise direct compute and a twice-reused prepared lifecycle for pointwise, matmul, reduction,
  and normalization expressions whose direct derived outputs are publicly observable as unresolved.

## Affected files

- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/StaticResultLogicalLayoutClosure.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/GraphCompiler.java`
- removed `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/NegLogicalLayoutClosure.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/RecurrentScanCompilerTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/StaticResultLogicalLayoutClosureTest.java`
- removed `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/NegLogicalLayoutClosureTest.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/EngineTypedLifecycleIntegrationTest.java`
- `docs/api/compile-api.md`
- `docs/backend-guide/cpu-backend.md`
- this task and the Compiler master plan

## Acceptance

- Direct public Engine compute succeeds with exact values for representative pointwise, matmul,
  reduction, and normalization expressions whose derived layouts start unresolved.
- Public compile, prepare, two runs of the same prepared artifact, and materialization succeed with
  exact values for the same families.
- Caller-bindable and dynamic descriptors remain unchanged; affine/view results retain exact
  resolved geometry or stay unresolved, and unsupported operations remain fail-closed.
- A focused multi-unit pointwise-to-affine plan proves stable unit ordering, dependencies, boundary
  ownership, and address-table ordering and bounds.
- Compiler, CPU, and integration module tests pass.
- Java and explanatory API/backend documentation describe the new invariant and responsibility
  split.
- Independent Class B review reports no unresolved blocker.

## Verification to date

- `./gradlew :testing:integration-tests:test --tests io.github.pho001.synaptik.testing.integration.EngineTypedLifecycleIntegrationTest`
  passed, exercising public direct compute and reusable compile/prepare/run/materialize scenarios.
- Focused operation-aware Compiler closure/graph tests passed, including materialized canonical
  results; affine/view exclusions; dynamic, bindable, and splat values; and offset/zero-stride
  preservation.
- Focused CPU partition-preparation tests passed, including multi-unit affine address-table bounds,
  unit ordering, dependencies, and boundary ownership.
- `./gradlew :modules:compiler:test :backends:cpu:test :testing:integration-tests:test` passed
  after the operation-aware correction.
- `./gradlew :modules:compiler:javadoc :backends:cpu:javadoc` passed; it reported only the existing
  missing-`@param` warnings in compatibility constructors.

## Independent review

The corrected Class B re-review returned `APPROVED` with zero blockers. It confirmed the explicit
materialized-result allowlist, affine/view and unknown-kind exclusions, focused closure proof,
narrow `CpuAffineCopyIr` validation exemption with multi-unit topology evidence, planning task, and
0006B7 supersession note. The reviewer also reported focused Compiler and CPU tests plus
`git diff --check` passing.

## Documentation impact

`docs/api/compile-api.md` now defines the operation-aware final logical-descriptor closure, its
ordering, its explicit materialized-result proof, and its affine/view exclusions.
`docs/backend-guide/cpu-backend.md` explains that completion precedes ownership but does not grant
CPU capability. No architecture-contract change is required because logical versus physical
responsibility is unchanged.
