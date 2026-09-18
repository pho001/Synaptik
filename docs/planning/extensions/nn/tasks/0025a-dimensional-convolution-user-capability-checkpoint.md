# Task 0025A: Dimensional-Convolution User-Capability Checkpoint

## Status

Complete

## Goal

Close the public user workflow for the completed channels-first `Conv1d`, `Conv2d`, and `Conv3d`
layers by exercising their real module state through `Engine.standard()` and the CPU backend.
Prove known forward values for NCW, grouped NCHW, and grouped NCDHW expressions, exact strict
state-dictionary ownership, both reusable and one-shot Engine paths where they establish distinct
contracts, explicit publication/result cleanup, and the current fail-closed Conv3d training
boundary.

This is an integration checkpoint over existing production APIs:

```text
known host-backed input and state Tensor leaves
  -> strict StateDictionary load
  -> Conv1d / Conv2d / Conv3d forward expression
  -> Engine.standard()
  -> compile -> prepare -> run -> materialize
     or one-shot compute
  -> detached HostTensorValue checked against an independent Java oracle
```

The checkpoint adds no execution facade to NN and no layer knowledge to Engine. Compiler task
0006C remains Draft, so this task must not claim Conv3d gradients or training support.

## Scope

- Add one black-box integration test class under `testing/integration-tests` that imports only
  public Model, NN, and Engine APIs.
- Add the test project's narrow `testImplementation(project(":extensions:nn"))` dependency. This
  is test composition only; it creates no production dependency or transitive runtime facade.
- Update `EngineCompositionContractTest.APPROVED_INTEGRATION` with exactly one trailing ordered
  line, `testImplementation(project(":extensions:nn"))`, matching the integration build file.
  Preserve the exact-list equality assertion, ordering, Engine inventory, reverse-dependency scan,
  and every other test line; do not weaken, generalize, or replace the inventory guard.
- Construct the three existing concrete layer types through a deliberately small mix of direct
  construction and `ModuleFactory.standard()`:
  - direct `Conv1d`, proving the concrete public constructor without bias;
  - factory-created grouped `Conv2d`, proving the standard recipe in the main reusable workflow;
  - direct grouped `Conv3d`, proving the concrete forward-only type with bias.
  Existing NN unit tests already prove direct/factory parity for every rank, so duplicating both
  construction paths for all three ranks is unnecessary.
- Give every layer nontrivial, rank-specific kernel/stride/padding/dilation geometry. Use groups
  greater than one for Conv2d and Conv3d. Across the three fixtures cover both bias-disabled and
  bias-enabled state schemas without multiplying otherwise equivalent cases.
- Bind every layer through a complete strict `StateDictionary` before its first forward call.
  Use exact, caller-created, gradient-eligible, host-backed weight and optional bias Tensors with
  deterministic literal values. Before load, demonstrate on the direct Conv1d fixture that
  `weight()` and state export fail closed while the reservation is unbound. Do not invoke
  automatic initialization in this checkpoint.
- For each loaded layer assert the exact exported state paths, order, `StateKind.PARAMETER`,
  candidate Tensor reference identity, parameter wrapper order, weight Shape, optional-bias
  presence, and absence of unexpected state.
- Create each runtime input as a separate non-gradient caller-owned host-backed Tensor. The input,
  loaded weight, and loaded bias where present are the only provenance-free value leaves that
  carry host storage. The layer output is a storage-free expression. Record that `compute` first
  inventories reachable leaves by exact Tensor identity, while final `CompiledGraph.inputs()`
  remains authoritative for membership and binding order.
- Exercise `Engine.compute(output)` for Conv1d and Conv3d. This proves automatic input discovery,
  fresh compile/prepare/run/materialize/cleanup, and detached host values without repeating the
  explicit lifecycle for every rank.
- Exercise grouped Conv2d through `compile(List.of(output))`, `prepare(compiled)`, two
  `run(prepared, inputs)` calls, and explicit `RunResult.materialize`. Supply the exact required
  input/state Tensors in an order different from `compiled.inputs()` and assert the prepared
  handle is reusable while each run has a distinct result and publication occurrence.
- Compare every materialized output with an independent clean-Java channels-first grouped
  cross-correlation oracle implemented in the integration test. The oracle must index primitive
  arrays directly, apply stride, symmetric padding, dilation, groups, optional bias, and
  out-of-bounds zero padding, and must not call `Tensor.conv1d`, `conv2d`, `conv3d`, the layer
  under test, Compiler, CPU reference code, or another production convolution helper.
- Check the public lifecycle contract observably: result/publication are open before close and
  closed after close; post-close materialization fails; caller-owned input and state storage stay
  alive after result close while their arena remains open; detached values remain readable after
  result and Engine close; `Engine.isClosed()` changes after close; and new Engine work fails
  after close. These assertions prove only the documented ownership/lifecycle behavior, not
  absence of every possible leak.
- Verify Conv3d backward rejection through the ordinary public `Engine.backward(...)` boundary
  using a scalar reduction of the layer output and the exact loaded weight as explicit target.
  Assert `IllegalArgumentException` identifies `CONV3D` and ends with
  `Conv3d is forward-only until Compiler task 0006C closes its gradients`. Also assert the Engine
  remains open and a subsequent Conv3d forward compute still succeeds. This complements, rather
  than duplicates, Compiler unit evidence: it proves that an actual NN layer expression reaches
  the fail-closed inventory through the supported public Engine workflow.
- Add a focused layer-to-Engine example to `docs/api/training-api.md`. The existing convolution
  section stops at expression construction and the Runtime API explains generic Engine mechanics;
  neither currently connects strict layer state, caller host storage, `forward`, and `compute` in
  one user workflow. Keep the example concise and link to the Runtime API for the full reusable
  lifecycle.
- Finalize the documentation edit, current-boundary wording, links, and planning evidence through
  a distinct clean documentation-focused context in the same overall change.

## Out of scope

- Any production Java, public API, Javadoc, Model operation, Compiler rule, Engine behavior,
  Runtime/Prepare contract, CPU lowering, or backend change.
- Automatic/random parameter initialization, seed reproducibility, retry, concurrent first
  forward, strict-load rollback, or direct/factory parity matrices already covered by NN 0025.
- Conv1d or Conv2d gradient execution, optimizer/session behavior, a general training example, or
  any broader gradient claim. Their existing Compiler contracts remain unchanged; this checkpoint
  is forward-only apart from proving Conv3d rejection.
- Positive Conv3d backward, gradient publication, or training support. Compiler 0006C must first
  close its exact grouped NCDHW gradient inventory before a later task may make that claim.
- Exhaustive geometry/type/layout coverage, performance evidence, generated-code inspection,
  backend conformance expansion, or repeating raw Tensor convolution coverage already owned by
  `EngineConvolutionIntegrationTest`.
- A new NN execution facade, `Module.execute`, Tensor execution method, implicit state binding,
  durable checkpoint format, host-storage ownership transfer, or facade over Engine.
- Architecture-contract, focused architecture-documentation, ADR, backend-conformance, glossary,
  Javadoc, Gradle convention/plugin, or other module changes. No architecture-test change is
  allowed beyond the exact one-line ordered integration inventory update specified above.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [NN master plan](../master-plan.md)
- [NN 0025 channels-first layers](0025-channels-first-conv1d-conv2d-conv3d-layers.md)
- [Engine 0004 host materialization](../../../modules/engine/tasks/0004-explicit-host-materialization-boundary.md)
- [Engine 0005 one-shot forward convenience](../../../modules/engine/tasks/0005-one-shot-forward-convenience.md)
- [Engine 0005A automatic input discovery](../../../modules/engine/tasks/0005a-automatic-input-discovery-and-compute-convenience.md)
- [Engine 0006 scalar-objective backward](../../../modules/engine/tasks/0006-one-shot-scalar-objective-backward-convenience.md)
- [Engine 0008 lifecycle checkpoint](../../../modules/engine/tasks/0008-engine-lifecycle-capability-checkpoint.md)
- [Compiler 0006B Conv3d forward boundary](../../../modules/compiler/tasks/0006b-conv3d-forward-adoption-and-explicit-gradient-boundary.md)
- [Compiler 0006B6 convolution layout closure](../../../modules/compiler/tasks/0006b6-final-convolution-logical-layout-closure.md)
- [CPU 0008A dimensional-convolution closure](../../../backends/cpu/tasks/0008a-portable-channels-first-dimensional-convolution-closure.md)
- [Training API](../../../../api/training-api.md) and [Runtime API](../../../../api/runtime-api.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)

## Architecture constraints

- NN continues to own layer construction, trainable parameter wrappers, state paths, and strict
  in-memory state loading while depending only on Model. This integration test may compose NN with
  Engine because `testing/integration-tests` is an outward test project, not a production owner.
- Model remains the sole owner of Tensor identity, provenance, descriptors, host-storage
  association, and convolution semantics. Loaded parameter values remain ordinary Tensor leaves.
- Engine remains the composition root. It owns the public CPU-only compile, prepare, run,
  publication, materialization, automatic-discovery, cleanup, and close workflows. Do not bypass
  it with Compiler, Runtime, Prepare, or CPU internals.
- Caller input and loaded state storage are borrowed for a run and remain caller-owned. A
  `RunResult` owns its publication lease; a `HostTensorValue` is detached and owns no closeable
  resource.
- Prepared recipes remain immutable and reusable, while each run receives isolated mutable
  Runtime state. The test may observe separate result/publication identities but must not infer
  physical buffer identity or leak freedom.
- Compiler alone owns autograd. Conv3d remains rejected by every backward-capable request before
  derivative construction until Compiler 0006C completes. Gradient-eligible NN parameters do not
  imply a supported gradient rule.
- CPU remains the only standard Engine backend and accepts the existing fully static,
  resolved-layout dimensional-convolution domain. The test adds no capability or fallback.
- If implementation needs a production edit, new API, changed ownership/dependency rule, or
  broader execution topology, stop and report the conflict.

## Current-state evidence

- `Conv1d`, `Conv2d`, and `Conv3d` are final public `UnaryTensorModule` types with exact
  rank-specific scalar constructors, `weight()`, optional `bias()`, and `forward(Tensor)`.
- `ModuleFactory.standard()` already exposes matching stateless recipes and retains no layer or
  configuration state.
- `Module.loadStateDictionary(...)` validates the complete target tree before installing exact
  candidate Tensor references. A reserved convolution group can therefore bind without sampling,
  allocation, or another Tensor ID. Export order is `weight`, then optional `bias`.
- `Engine.standard()` supplies the current CPU-only composition. `compile`, `prepare`, `run`,
  `RunResult.materialize`, and `compute` are public. Explicit runs bind current host associations
  by Tensor ID; `compute` inventories reachable provenance-free leaves and then follows final
  Compiler input metadata.
- `EngineConvolutionIntegrationTest` already proves raw Tensor-level public forward execution for
  all three ranks. It does not instantiate NN layers, load their state dictionaries, or connect
  layer-owned state to Engine discovery/binding.
- The current uncommitted implementation adds
  `testImplementation(project(":extensions:nn"))` after the existing CPU test dependency in
  `testing/integration-tests/build.gradle.kts`; the new integration test consumes that exact edge.
  This test-only project edge is necessary and compatible with the outward integration role.
- The first `./gradlew test --rerun-tasks` checkpoint reached the architecture suite and exposed
  one planned inventory mismatch, not a product-test failure:
  `EngineCompositionContractTest.engineHasOnlyTheApprovedOrderedDirectDependencies()` still
  expected the pre-NN six-line `APPROVED_INTEGRATION` list while the build file correctly reported
  the new NN line seventh. Its XML records one test, one failure, and zero errors. This diagnosis
  authorized only the matching one-line expected-inventory addition and does not constitute final
  checkpoint evidence. After synchronizing that exact inventory, the implementation context's
  fresh complete repository run succeeded and is the controlling checkpoint evidence recorded
  below.
- `docs/api/training-api.md` explains the convolution layer/state contract and stops after
  expression construction. `docs/api/runtime-api.md` explains generic Engine lifecycle and
  storage ownership, but has no layer-state workflow.
- Compiler 0006C has no detailed task and remains Draft. Current `AutogradPreflight` emits the
  exact Conv3d forward-only failure named above.

## Exact fixture matrix and oracle contract

Use `FLOAT32` throughout and these exact fixtures so the clean implementation context does not
silently broaden or simplify geometry:

| Fixture | Construction | Input | Weight | Bias | Geometry | Output |
|---|---|---|---|---|---|---|
| Conv1d | direct, groups `1`, bias `false` | `[1,2,6]` | `[2,2,3]` | absent | kernel `3`, stride `2`, padding `1`, dilation `2` | `[1,2,2]` |
| Conv2d | `ModuleFactory.standard()`, groups `2`, bias `true` | `[1,4,4,5]` | `[4,2,2,3]` | `[4]` | kernel `2x3`, stride `2x1`, padding `1x1`, dilation `1x2` | `[1,4,3,3]` |
| Conv3d | direct, groups `2`, bias `true` | `[1,4,3,4,5]` | `[4,2,2,2,2]` | `[4]` | kernel `2x2x2`, stride `1x2x1`, padding `1x0x1`, dilation `1x1x2` | `[1,4,4,2,5]` |

Construct each layer with `ParameterInitialization.zeros()` and a fixed seed, then bypass that
unused automatic policy by loading complete known state before `forward`. Populate inputs and
weights from small deterministic primitive-array patterns defined by the test, for example signed
fractions derived from the flat index; use the exact bias values
`{0.5f, -0.25f, 1.0f, -1.0f}` for both biased fixtures. The test must preserve those arrays as the
oracle inputs rather than reading parameter storage back through production code.

Implement separate private rank-specific oracle entry points or one private dimension-parameterized
loop whose indexing remains explicit. For every output coordinate, derive the group, visit only
that group's input channels, map kernel coordinates through stride/padding/dilation, skip
out-of-range input coordinates as zero, accumulate products in `float`, and add the selected bias
once. Compare complete row-major outputs with a documented small floating tolerance and assert the
exact expected Shapes above. The test names should make the three obligations visible, such as:

- `executesStrictLoadedConv1dThroughOneShotCompute`;
- `reusesPreparedGroupedConv2dAndObservesPublicationCleanup`; and
- `executesStrictLoadedGroupedConv3dAndRejectsBackward`.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.testing.integration` — black-box cross-module integration tests.
- `io.github.pho001.synaptik.nn.layers` and `.module` — unchanged public layers, factory, and state
  dictionary APIs.
- `io.github.pho001.synaptik.engine` — unchanged ordinary Engine lifecycle.
- Model tensor/storage/datatype/layout/Shape packages — unchanged leaf and host-storage creation.

Packages added or changed:

- None. One test class is added to the existing integration-test package.

Type placement:

- `io.github.pho001.synaptik.testing.integration.NnConvolutionEngineIntegrationTest` — package-
  private JUnit integration owner for the layer-to-Engine checkpoint and its private primitive-
  array oracle/helpers.

## Affected files

Expected implementation/test/documentation paths:

1. `testing/integration-tests/build.gradle.kts`
2. `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/NnConvolutionEngineIntegrationTest.java` (new)
3. `testing/architecture-tests/src/test/java/io/github/pho001/synaptik/testing/architecture/EngineCompositionContractTest.java`
4. `docs/api/training-api.md`
5. this task specification
6. `docs/planning/extensions/nn/master-plan.md`
7. `docs/planning/roadmap.md`

Review without modification: NN convolution production/Javadocs and unit tests; Module,
ModuleFactory, StateDictionary, and Parameter APIs; Engine production/Javadocs and existing
integration fixtures; Tensor/Compile/Runtime APIs; architecture contracts/tests; backend
conformance; glossary; Gradle settings/conventions; every other module and plan. For architecture
tests, review all files but modify only the one exact `APPROVED_INTEGRATION` line above.

## Maximum scope

At most the exact seven paths above: one test-only dependency edit, one new integration test, one
one-line ordered architecture inventory update, one focused public API-documentation edit, and
three synchronized planning records. If implementation needs another helper, production file,
test owner, build file, documentation page, or architecture artifact, stop and amend this Draft
specification before proceeding.

## Acceptance criteria

- Direct Conv1d, factory grouped Conv2d, and direct grouped Conv3d construct only their existing
  APIs and produce usable layer forward expressions after strict state load.
- Conv1d is bias-free; Conv2d and Conv3d are biased; both higher-rank fixtures use groups greater
  than one. Every fixture uses nontrivial rank-specific geometry, and expected Shape calculations
  agree with the public layer result and final publication.
- Every state dictionary contains exactly `weight` followed by optional `bias`, every entry is a
  parameter, exact loaded Tensor identities are retained/exported, wrapper order matches, and the
  loaded numerical values demonstrably determine Engine output. The selected pre-load Conv1d
  access/export checks reject incomplete state, and strict load makes its first forward usable.
- Only the input and loaded state leaves carry caller-owned host storage. Explicit compilation
  reports exactly those bindable Tensor identities; run accepts them in a different order.
- Conv1d and Conv3d pass one-shot `compute` with detached expected values. Grouped Conv2d passes
  one compile/prepare and two runs through the same prepared handle, with distinct run/publication
  objects and correct explicit materialization.
- All three results match independent primitive-array Java oracles that implement grouped
  channels-first cross-correlation directly. No expected value is obtained from a Tensor
  convolution, NN layer, Compiler, CPU helper, or the operation under test.
- Observable ownership/lifecycle assertions pass: result/publication close state, post-close
  rejection, caller storage survival through result close, detached post-close readability,
  Engine close state, and rejection of new work after Engine close. No broader leak claim appears.
- Public `Engine.backward` on a scalar depending on Conv3d fails closed with the exact current
  `CONV3D`/Compiler-0006C diagnostic; the Engine stays open and remains capable of a later valid
  forward compute. No Conv3d gradient or training success is claimed.
- No Conv1d/Conv2d gradient test is added because this task does not establish a training workflow
  and their existing Compiler contracts are unchanged.
- The Training API gains one concise layer-state-to-Engine example, accurately distinguishes
  expression construction from execution, identifies which leaves own host storage, explains
  detached results and cleanup, links to the reusable Runtime workflow, and keeps Conv3d
  forward-only wording explicit.
- `EngineCompositionContractTest` gains exactly the new NN test dependency as the seventh ordered
  `APPROVED_INTEGRATION` entry. Its strict equality check and every other assertion remain intact.
- No production/Javadoc, glossary, architecture contract/documentation/ADR, conformance, settings,
  or other module path changes. A separate clean documentation-focused pass finalizes the API
  text and reasoned no-change conclusions without repeating stable Java tests.
- Focused integration validation and one repository-wide capability-checkpoint run pass. Links,
  anchors, fences, terminology, exact seven-path scope, empty staging, LF/final newlines, status
  agreement, and `git diff --check` pass before the task can become Complete.

## Tests / validation

During implementation, run the focused new integration class while stabilizing executable work:

```bash
./gradlew :testing:integration-tests:test \
  --tests io.github.pho001.synaptik.testing.integration.NnConvolutionEngineIntegrationTest
```

After Java, the test dependency, and the exact architecture inventory line stabilize, rerun the
complete capability checkpoint. The earlier root run failed the stale expected inventory and is
diagnostic only; it cannot be reused as the final repository result:

```bash
./gradlew test --rerun-tasks
```

Record exact test totals from the fresh final Gradle XML and confirm execution of the new
integration class,
`EngineConvolutionIntegrationTest`, the NN convolution/state/factory suites, Engine lifecycle
suites, and architecture tests, including the passing strict ordered Engine composition inventory.
The repository-wide command is required because this is a named capability checkpoint and changes
a test-project dependency; do not separately rerun those stable suites afterward.

Documentation-focused pass, after it finalizes the Training API and planning evidence:

```bash
git diff --check
git status --short -uall
git diff --cached --check
```

Also validate local Markdown file links and heading anchors, unique headings, balanced fences,
current-versus-planned terminology, exact state/failure names in the example, LF/final newlines,
trailing whitespace, exact seven-path scope, and status/dependency agreement. Confirm the index is
empty. No Javadoc generation is required because no production Java or Javadoc changes; review
the affected public Javadocs against the example and record that they remain accurate. Do not run
Java tests in the documentation context unless it changes executable Java or identifies a
specific stale-evidence risk.

## Dependencies

- [NN 0025](0025-channels-first-conv1d-conv2d-conv3d-layers.md) — Complete; supplies the three
  layers, recipes, state lifecycle, unit tests, and documentation boundary.
- [Engine 0004](../../../modules/engine/tasks/0004-explicit-host-materialization-boundary.md) —
  Complete; supplies explicit detached host materialization and result lease behavior.
- Engine 0005 and 0005A — Complete; supply one-shot compute and authoritative automatic leaf
  discovery/input selection.
- Engine 0006 and 0008 — Complete; supply public backward failure traversal and the closed public
  lifecycle checkpoint.
- [CPU 0008A](../../../backends/cpu/tasks/0008a-portable-channels-first-dimensional-convolution-closure.md)
  — Complete; supplies truthful Conv1d composition and grouped Conv3d forward execution.
- Compiler 0006B and 0006B6 — Complete; supply Conv3d forward adoption, exact rejection, final
  convolution layouts, and positive raw Tensor Engine fixtures.
- Compiler 0006C — Draft and therefore not a satisfied dependency. It is required only before a
  task claims positive Conv3d training. This checkpoint instead locks the current public
  fail-closed boundary.

The global roadmap explicitly selected NN 0025 and 0025A as an isolated dimensional-convolution
sequence after Engine 0008. Unrelated NN 0021B–0024 remain Draft; their recurrent/Data contracts
and files do not overlap this checkpoint.

## Follow-up tasks

- Return to the next frontier already recorded by the current roadmap after this checkpoint;
  current planning points to tools/tuning 0002 rather than inventing another NN convolution task.
- Compiler 0006C remains the independent future owner of Conv3d adjoint expressibility and
  gradient closure. If it completes, a separately selected user-training checkpoint may replace
  the failure assertion with positive evidence; this task must not anticipate that result.
- Additional convolution padding modes, layouts, transposed/depthwise-specialized layers,
  performance work, or durable checkpoint transport require their own future tasks.

## Documentation, Javadoc, architecture, conformance, and integration impact

- Integration: add exactly one layer-level public Engine class because existing raw Tensor
  integration cannot prove module state ownership/load/discovery.
- Documentation: add a focused Training API example because the present construction-only example
  and generic Runtime workflow leave the layer-to-Engine handoff implicit. Do not duplicate the
  full Runtime lifecycle guide.
- Javadoc: no change. Existing layer, state-dictionary, Engine, result, and host-value Javadocs
  already describe the contracts consumed here; review them in the documentation pass.
- Glossary: no change. The task introduces no new reusable term or changes the meaning of layer,
  state dictionary, convolution, publication, or host materialization.
- Architecture contract, focused architecture documentation, and ADRs: no change. The outward
  integration project composes established NN and Engine APIs without changing ownership or
  dependency direction.
- Architecture tests: update only `EngineCompositionContractTest.APPROVED_INTEGRATION` with the
  exact trailing NN test dependency already present in the integration build. Retain strict
  ordered-list equality and all inward-dependency assertions. This records an approved outward
  test composition edge, not a production NN-to-Engine dependency or architecture-rule change.
- Backend conformance: no change. CPU behavior and raw operation conformance are already covered;
  this task verifies only public cross-module composition.
- Production modules and Javadocs: no change because no defect or missing public seam is present.

## Architecture impact

Expected architecture-contract and dependency-direction impact: None. This task composes public
NN, Model, and Engine contracts only from the outward integration-test project. Nevertheless,
`EngineCompositionContractTest.APPROVED_INTEGRATION` must gain the exact one trailing
`testImplementation(project(":extensions:nn"))` entry because that source inventory enforces the
outward test project's approved ordered dependencies. Synchronizing this strict expected inventory
is not an architecture rule, production dependency, module-ownership, or dependency-direction
change. If implementation reveals a missing production seam or a need for another dependency
direction, stop and report the exact conflict rather than editing around it.

## Implementation prompt

```text
You are the clean-context implementation agent for Synaptik NN task 0025A in
/Users/phujka/IdeaProjects/Synaptik. Do not use GSD. Do not stage, commit, or push.

Read AGENTS.md, ARCHITECTURE.md, the current architecture plan, planning guide, roadmap, NN master
plan, and this task specification in full. Read NN 0025 and the directly referenced Engine,
Compiler, and CPU prerequisites. Inspect the current public layers, ModuleFactory/state APIs,
Engine lifecycle, host Tensor construction, existing convolution/lifecycle integration tests,
build file, and affected API documentation.

Implement exactly this Draft specification within its seven-path ceiling. Add only the test-only
NN dependency, one black-box integration test with independent primitive-array oracles, the exact
one-line trailing `APPROVED_INTEGRATION` inventory entry without weakening its test, and the
focused Training API example. Preserve the explicit Conv3d fail-closed boundary and add no
production API or behavior. Stop on architecture uncertainty or scope overflow.

Run the focused integration class, synchronize the exact architecture inventory, then rerun one
final repository-wide checkpoint after executable work stabilizes; the earlier inventory failure
is not final evidence. Hand the frozen diff and exact evidence to a distinct clean
documentation-focused context. That pass must follow the General, API/Javadoc, Planning, and
Example profiles; finalize the Training API and planning/no-change evidence; reuse stable Java
evidence; and validate links, scope, statuses, index state, and whitespace. Update this task,
master plan, and roadmap only after all gates pass; keep Compiler 0006C and unrelated NN
0021B–0024 Draft.
```

## Local decisions

- Strict-load literal state is selected instead of fresh automatic initialization because it
  proves exact ownership and Engine consumption with deterministic numerical evidence; automatic
  initialization behavior is already exhaustively covered by NN 0025 unit tests.
- One-shot execution covers Conv1d and Conv3d; reusable execution covers grouped Conv2d twice.
  This assigns each public workflow a distinct proof instead of repeating both for every rank.
- The independent oracle stays private in the one integration class. A shared production/test
  convolution utility would either duplicate a semantic owner or weaken independence.
- Conv3d backward rejection is in scope at the public Engine boundary because it is a visible user
  capability limit and confirms the layer expression reaches Compiler preflight. Positive
  Conv1d/Conv2d gradient execution is not needed to close this forward checkpoint.
- The Training API receives the focused example; the Runtime API already owns the complete Engine
  lifecycle explanation and should be linked, not duplicated.
- The architecture guard remains strict. Adding the exact seventh expected integration dependency
  is preferable to excluding NN lines, sorting dynamically, weakening equality, or broadening an
  allow pattern because the test intentionally locks both membership and order.

## Known limitations

- Standard Engine execution is CPU-only and limited to fully static supported layouts.
- The fixtures are representative, not exhaustive across all types, groups, or geometry.
- State dictionaries are in-memory Tensor-binding snapshots, not durable checkpoint bytes.
- Conv3d remains forward-only until Compiler 0006C completes and a later user-facing claim is
  selected and validated.
- The lifecycle assertions prove the documented close/ownership observations, not universal leak
  detection or performance.

## Validation evidence

- Planning context `01a0b5f5-1cc0-7d42-b497-82c148a6d01b` selected this bounded seven-path
  checkpoint. Implementation context `01a0b600-a3a8-79c3-a17a-00a41cb3f1d8` added the test-only
  NN dependency, the public-API-only integration class, and the exact trailing strict architecture
  inventory entry. Documentation context `01a0b611-62b6-70b0-a9a7-d3b6291b0fa6` independently
  reviewed the frozen executable diff and finalized the Training API and planning records.
- The implementation context's focused
  `./gradlew :testing:integration-tests:test --tests io.github.pho001.synaptik.testing.integration.NnConvolutionEngineIntegrationTest`
  run passed 3 tests with zero failures, errors, or skips. After the inventory amendment, the
  focused `EngineCompositionContractTest` run passed its one test with zero failures, errors, or
  skips.
- The first repository-wide `./gradlew test --rerun-tasks` run is retained as diagnostic history:
  it reached the strict architecture inventory and failed only because
  `APPROVED_INTEGRATION` still described the prior six dependencies while the integration build
  correctly contained the new seventh NN test dependency. The authorized one-line expected-list
  amendment resolved that mismatch without weakening equality or changing another assertion.
- The implementation context then ran a fresh controlling `./gradlew test --rerun-tasks` after
  executable work stabilized. It completed successfully in 2 minutes 49 seconds with 70 tasks
  executed. Fresh XML contained 498 suites and 3,217 tests: zero failures, zero errors, and 28
  skips; all 498 XML files were rewritten by that final run.
- The controlling XML confirms the new NN integration class passed 3/3 and the existing
  `EngineConvolutionIntegrationTest` passed 3/3. NN `Conv1dTest` passed 3, `Conv2dTest` 4,
  `Conv3dTest` 3, `ConvolutionInitializationTest` 8,
  `ConvolutionStateDictionaryTest` 7, and `ModuleFactoryTest` 10, without skips. Engine advanced
  lifecycle passed 6, typed lifecycle 26, standard composition 6, representative execution 24,
  and the integration lifecycle/composition suites passed. The seven architecture suites passed
  all nine tests without skips, including the strict ordered Engine composition inventory.
- Structural review confirmed exactly the three executable paths, public Model/NN/Engine imports
  only, independent primitive Java 1D/2D/3D oracles, no forbidden inward helper or import, and the
  exact one-line architecture inventory change. LF/final-newline, empty-index, and executable diff
  checks passed in the implementation context.
- The documentation pass added one strict-loaded factory `Conv2d`-to-Engine example. It uses
  caller-owned host-backed input, weight, and bias leaves; complete ordered parameter state;
  storage-free `forward` expression construction; one-shot `Engine.compute`; and a detached
  `[1,1,2,2]` host result whose four values are `-3.5f`. It links to Runtime's reusable lifecycle
  and explicitly preserves the forward-only Conv3d/Compiler-0006C-Draft boundary.
- Documentation validation checked every local Markdown file target and relevant heading anchor,
  unique headings, balanced fences, glossary terminology, current-versus-planned wording, example
  public names/signatures/state order/failure boundaries, status/dependency/frontier agreement,
  LF/final newlines, trailing whitespace, exact seven-path combined scope, and empty staging.
  `git diff --check` and `git diff --cached --check` passed. Java tests and Javadoc were not rerun:
  this pass changed no executable Java or Javadoc, so it reused the fresh implementation evidence
  as required by the documentation workflow.
- SHA-256 hashes of the frozen executable paths were identical before and after documentation:
  `e3305bd7b0fde58a987f34f62f08de9f4db6731dc3195bbab45c098fa40b3747` for the integration
  build, `cac176b2eca027a348ac3109acfdb01c86c44cc72ae9baf137dac6ab57658611` for the new
  integration class, and `1f32e81c5015b3c7de496d11f3330d8b84125b594cf5426bccba137a158c03fd`
  for the architecture inventory test.
- Final status checks confirm NN 0025 and 0025A are Complete, Compiler 0006C and NN 0021B–0024
  remain Draft, tools/tuning 0002 is the next recorded frontier, and no detailed tools/tuning
  0002 task was created.

## Implementation notes

- Added the integration project's narrow test-only dependency on `extensions:nn` and recorded it
  as the seventh ordered integration dependency in the unchanged strict architecture guard.
- Added one public black-box integration class covering direct Conv1d, factory grouped Conv2d,
  direct grouped Conv3d, complete strict state loading, exact state identity/order, automatic and
  reusable Engine paths, detached materialization and cleanup, independent numerical oracles, and
  the exact public Conv3d backward rejection. No production behavior or facade changed.
- Added the focused Training API example and synchronized this task, the NN master plan, and the
  global roadmap after the fresh full checkpoint passed.
- Existing production Javadocs for `Conv1d`, `Conv2d`, `Conv3d`, `ModuleFactory`,
  `StateDictionary`, `StateEntry`, `StateKind`, Tensor/host-storage APIs, and Engine compile/run/
  materialization types remain accurate: the change composes their documented state, expression,
  borrowed-storage, execution, and detached-value contracts without changing any signature or
  behavior. No production Javadoc edit or generation is required.
- The glossary already defines channels-first convolution layers, strict in-memory state,
  standard module factory, host storage, publication occurrence, and host tensor value with the
  meanings used here. The checkpoint introduces no reusable term or changed definition, so no
  glossary edit is warranted.
- No architecture contract, current architecture page, or ADR changed because the new edge is
  outward test composition only and production dependency direction is unchanged. No backend-
  conformance change is needed because backend operation behavior is unchanged. No production
  Java, settings, convention, or other-module change is needed.

## Completion summary

- Completed changes: Closed the dimensional-convolution user checkpoint with strict-loaded
  Conv1d/Conv2d/Conv3d public Engine execution, independent expected values, lifecycle and
  fail-closed Conv3d backward evidence, plus one concise user example.
- Files changed or created: Exactly the seven paths listed in this task—three frozen executable
  paths and four finalized documentation/planning paths.
- Tests and validation: Reused the controlling 3-test focused integration pass, 1-test focused
  architecture pass, and successful 3,217-test/498-suite repository checkpoint from implementation
  context `01a0b600-a3a8-79c3-a17a-00a41cb3f1d8`; the documentation context passed all required
  Markdown, scope, status, index, hash, newline, whitespace, and diff checks without running Java.
- Documentation-agent review: Complete in clean context
  `01a0b611-62b6-70b0-a9a7-d3b6291b0fa6` under the General, API/Javadoc, Planning, and Example
  profiles.
- Documentation impact: The Training API now connects strict NN state, caller host storage,
  expression construction, standard Engine execution, detached results, and the reusable Runtime
  workflow while preserving all current limitations.
- Javadoc review: Existing affected public contracts remain accurate; no source change or Javadoc
  run was required.
- Glossary impact: No new or changed reusable terminology; existing entries remain accurate.
- Architecture and conformance impact: None beyond the exact strict inventory record for the
  outward integration-test dependency.
- Unresolved issues: None.
- Follow-up required: tools/tuning 0002 is the next recorded planning frontier; Compiler 0006C
  remains the separate Draft prerequisite for any future positive Conv3d-gradient claim.

Status: Complete
