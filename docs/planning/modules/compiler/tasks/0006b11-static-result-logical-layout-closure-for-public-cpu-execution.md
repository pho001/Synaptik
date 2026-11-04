# Task 0006B11: Static-Result Logical-[Layout](../../../../glossary.md#layout) Closure for Public [CPU](../../../../glossary.md#term-cpu) Execution

## Status

Complete

## Authorization and classification

- User-authorized benchmark-blocker fix from exact main [commit](../../../../glossary.md#term-commit)
  `aba8eaa8d2c0cfdf9b3fbcddde508485ffecde3f`.
- Class B behavior change: compiler artifacts presented to capability providers change for eligible
  fully static operation outputs, and CPU complete-partition validation corrects one fail-closed
  structural comparison.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [artifacts glossary](../../../../glossary.md#term-artifact); [operation glossary](../../../../glossary.md#operation); [CPU glossary](../../../../glossary.md#term-cpu); [partition glossary](../../../../glossary.md#partition); [validation glossary](../../../../glossary.md#term-validation); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- Independent review is required before the local commit.

## Goal

Make ordinary fully static public [Tensor](../../../../glossary.md#tensor) expressions reachable through `Engine.compute(...)` and
reusable `compile -> prepare -> run -> materialize` on [CPU](../../../../glossary.md#term-cpu) without package-private [provenance](../../../../glossary.md#provenance) or
manually injected derived [layouts](../../../../glossary.md#layout), while retaining semantic and complete-[partition](../../../../glossary.md#partition) [fail-closed](../../../../glossary.md#term-fail-closed)
checks.

## Diagnosis

1. [Model](../../../../glossary.md#term-model) [inference](../../../../glossary.md#term-inference) intentionally leaves ordinary derived [Tensor](../../../../glossary.md#tensor) [layouts](../../../../glossary.md#layout) unresolved.
2. `CpuCapabilityProvider` correctly rejects every occurrence whose input or output layout is not
   statically resolved, so ownership selection rejected pointwise, matmul, reduction, and
   normalization expressions before CPU preparation.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [pointwise glossary](../../../../glossary.md#term-pointwise); [normalization glossary](../../../../glossary.md#term-normalization); [CPU glossary](../../../../glossary.md#term-cpu))</span>
3. Existing published-constant, convolution, and NEG-specific compiler passes showed that final
   logical descriptor completion belongs before capability queries, but the NEG pass covered only
   one operation family.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [compiler glossary](../../../../glossary.md#term-compiler); [descriptor glossary](../../../../glossary.md#term-descriptor); [operation glossary](../../../../glossary.md#operation))</span>
4. Appending public `contiguous()` did not solve the root cause because the inner operation
   occurrence remained unresolved.
5. Once general closure made both occurrences eligible, a second existing defect rejected valid
   multi-unit pointwise-to-affine plans: general topology validation compared affine generated
   address-table access plans with physical boundary bindings even though those are intentionally
   different validated domains.<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [validation glossary](../../../../glossary.md#term-validation); [bindings glossary](../../../../glossary.md#term-binding))</span>

## Contracts

- `ARCHITECTURE.md`: compile owns logical inference/canonicalization; backends own physical
  representation, resource preparation, and execution.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [inference glossary](../../../../glossary.md#term-inference); [canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [backends glossary](../../../../glossary.md#backend))</span>
- `docs/architecture/contracts/compiler-autograd.md`: Compiler may complete logical descriptors but
  may not allocate physical buffers or create backend executables.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [descriptors glossary](../../../../glossary.md#term-descriptor); [buffers glossary](../../../../glossary.md#term-buffer))</span>
- `docs/architecture/contracts/backend-execution.md`: CPU capability and complete-partition lowering
  remain responsible for exact semantic, shape, layout, topology, carrier, and resource checks.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [partition glossary](../../../../glossary.md#partition); [lowering glossary](../../../../glossary.md#lowering); [shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [topology glossary](../../../../glossary.md#term-topology); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- `docs/architecture/contracts/runtime-prepare-engine.md`: proof must use the public Engine lifecycle
  rather than package-private graph seams.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [lifecycle glossary](../../../../glossary.md#lifecycle); [graph glossary](../../../../glossary.md#graph))</span>

## Implementation

- Replace the NEG-only final pass with a backend-neutral, explicit allowlist of operation families
  whose results are newly materialized logical values, plus the semantic <span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [operation glossary](../../../../glossary.md#operation))</span>[`CONTIGUOUS`](../../../../glossary.md#term-contiguous) request and
  explicit compile-time splat graph inputs. Unknown future kinds fail closed.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [graph glossary](../../../../glossary.md#graph); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
- Exclude affine/[view](../../../../glossary.md#term-view-meanings) results (`RESHAPE`, `EXPAND`, `PERMUTE`, `EXPAND_DIMS`, `SQUEEZE`, slice
  extraction, and scalar <span data-glossary-navigation="true"> ([slice glossary](../../../../glossary.md#slice); [scalar glossary](../../../../glossary.md#term-scalar))</span>`SELECT`) so their input-derived offsets and non-contiguous or zero
  strides are preserved or remain unresolved.<span data-glossary-navigation="true"> ([strides glossary](../../../../glossary.md#term-stride))</span>
- Preserve caller-bindable inputs, dynamic values, resolved layouts, graph topology, identities,
  sidecars, and all operation references.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout); [topology glossary](../../../../glossary.md#term-topology))</span>
- Keep specialized convolution closure before the operation-aware pass because Conv1d's direct
  squeeze view needs exact semantic propagation from its convolution producer.<span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [producer glossary](../../../../glossary.md#term-producer))</span>
- Keep capability providers strict; logical completion grants no operation, dtype, shape, semantic,
  or backend support.<span data-glossary-navigation="true"> ([dtype glossary](../../../../glossary.md#data-type--datatype); [shape glossary](../../../../glossary.md#shape))</span>
- Exempt affine-copy units from the invalid general access-plan equality check. Affine lowering and
  plan construction continue to validate address-table bounds and physical geometry; only the
  comparison between generated address-table and boundary-binding domains is removed.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [binding glossary](../../../../glossary.md#term-binding))</span>
- Exercise direct compute and a twice-reused prepared lifecycle for pointwise, matmul, reduction,
  and normalization expressions whose direct derived outputs are publicly observable as unresolved.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [pointwise glossary](../../../../glossary.md#term-pointwise); [normalization glossary](../../../../glossary.md#term-normalization))</span>

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
- this task and the [Compiler](../../../../glossary.md#term-compiler) master plan

## Acceptance

- Direct public Engine compute succeeds with exact values for representative pointwise, matmul,
  reduction, and normalization expressions whose derived layouts start unresolved.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [pointwise glossary](../../../../glossary.md#term-pointwise); [normalization glossary](../../../../glossary.md#term-normalization); [layouts glossary](../../../../glossary.md#layout))</span>
- Public compile, prepare, two runs of the same prepared artifact, and materialization succeed with
  exact values for the same families.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [prepare glossary](../../../../glossary.md#prepare); [artifact glossary](../../../../glossary.md#term-artifact); [materialization glossary](../../../../glossary.md#materialization))</span>
- Caller-bindable and dynamic descriptors remain unchanged; affine/view results retain exact
  resolved geometry or stay unresolved, and unsupported operations remain fail-closed.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [view glossary](../../../../glossary.md#term-view-meanings); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- A focused multi-unit pointwise-to-affine plan proves stable unit ordering, dependencies, boundary
  ownership, and address-table ordering and bounds.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- [Compiler](../../../../glossary.md#term-compiler), [CPU](../../../../glossary.md#term-cpu), and integration [module](../../../../glossary.md#term-module) tests pass.
- Java and explanatory API/backend documentation describe the new invariant and responsibility
  split.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [backend glossary](../../../../glossary.md#backend); [invariant glossary](../../../../glossary.md#term-invariant))</span>
- Independent Class B review reports no unresolved blocker.

## Verification to date

- `./gradlew :testing:integration-tests:test --tests io.github.pho001.synaptik.testing.integration.EngineTypedLifecycleIntegrationTest`
  passed, exercising public direct compute and reusable compile/prepare/run/materialize scenarios.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [prepare glossary](../../../../glossary.md#prepare); [run glossary](../../../../glossary.md#term-run))</span>
- Focused operation-aware Compiler closure/graph tests passed, including materialized canonical
  results; affine/view exclusions; dynamic, bindable, and splat values; and offset/zero-stride
  preservation.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [Compiler glossary](../../../../glossary.md#term-compiler); [graph glossary](../../../../glossary.md#graph); [canonical glossary](../../../../glossary.md#term-canonical-form); [view glossary](../../../../glossary.md#term-view-meanings); [stride glossary](../../../../glossary.md#term-stride))</span>
- Focused CPU partition-preparation tests passed, including multi-unit affine address-table bounds,
  unit ordering, dependencies, and boundary ownership.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [partition glossary](../../../../glossary.md#partition); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- `./gradlew :modules:compiler:test :backends:cpu:test :testing:integration-tests:test` passed
  after the operation-aware correction.
- `./gradlew :modules:compiler:javadoc :backends:cpu:javadoc` passed; it reported only the existing
  missing-`@param` warnings in compatibility constructors.

## Independent review

The corrected Class B re-review returned `APPROVED` with zero blockers. It confirmed the explicit
materialized-result allowlist, affine/[view](../../../../glossary.md#term-view-meanings) and unknown-kind exclusions, focused closure proof,
narrow `CpuAffineCopyIr` [validation](../../../../glossary.md#term-validation) exemption with multi-unit [topology](../../../../glossary.md#term-topology) evidence, [planning](../../../../glossary.md#term-planning) task, and
0006B7 supersession note. The reviewer also reported focused [Compiler](../../../../glossary.md#term-compiler) and [CPU](../../../../glossary.md#term-cpu) tests plus
`git diff --check` passing.

## Documentation impact

`docs/api/compile-api.md` now defines the [operation](../../../../glossary.md#operation)-aware final logical-[descriptor](../../../../glossary.md#term-descriptor) closure, its
ordering, its explicit materialized-result proof, and its affine/[view](../../../../glossary.md#term-view-meanings) exclusions.
`docs/backend-guide/cpu-backend.md` explains that completion precedes [ownership](../../../../glossary.md#term-ownership-meanings) but does not grant
[CPU](../../../../glossary.md#term-cpu) capability. No architecture-contract change is required because logical versus physical
responsibility is unchanged.
