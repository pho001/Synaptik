# Task 0025A: Dimensional-Convolution User-Capability Checkpoint

## Status

Draft

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
- Architecture-contract, ADR, architecture-test, backend-conformance, glossary, Javadoc, Gradle
  convention/plugin, or other module changes.

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
- `testing/integration-tests` currently has no dependency on `extensions/nn`; adding one test-only
  project edge is necessary and compatible with its outward integration role.
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
3. `docs/api/training-api.md`
4. this task specification
5. `docs/planning/extensions/nn/master-plan.md`
6. `docs/planning/roadmap.md`

Review without modification: NN convolution production/Javadocs and unit tests; Module,
ModuleFactory, StateDictionary, and Parameter APIs; Engine production/Javadocs and existing
integration fixtures; Tensor/Compile/Runtime APIs; architecture contracts/tests; backend
conformance; glossary; Gradle settings/conventions; every other module and plan.

## Maximum scope

At most the exact six paths above: one test-only dependency edit, one new integration test, one
focused public API-documentation edit, and three synchronized planning records. If implementation
needs another helper, production file, test owner, build file, documentation page, or architecture
artifact, stop and amend this Draft specification before proceeding.

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
- No production/Javadoc, glossary, architecture, conformance, settings, or other module path
  changes. A separate clean documentation-focused pass finalizes the API text and reasoned
  no-change conclusions without repeating stable Java tests.
- Focused integration validation and one repository-wide capability-checkpoint run pass. Links,
  anchors, fences, terminology, exact six-path scope, empty staging, LF/final newlines, status
  agreement, and `git diff --check` pass before the task can become Complete.

## Tests / validation

During implementation, run the focused new integration class while stabilizing executable work:

```bash
./gradlew :testing:integration-tests:test \
  --tests io.github.pho001.synaptik.testing.integration.NnConvolutionEngineIntegrationTest
```

After Java and the test dependency stabilize, run the capability checkpoint once:

```bash
./gradlew test --rerun-tasks
```

Record exact test totals from Gradle XML and confirm execution of the new integration class,
`EngineConvolutionIntegrationTest`, the NN convolution/state/factory suites, Engine lifecycle
suites, and architecture tests. The repository-wide command is required because this is a named
capability checkpoint and changes a test-project dependency; do not separately rerun those stable
suites afterward.

Documentation-focused pass, after it finalizes the Training API and planning evidence:

```bash
git diff --check
git status --short -uall
git diff --cached --check
```

Also validate local Markdown file links and heading anchors, unique headings, balanced fences,
current-versus-planned terminology, exact state/failure names in the example, LF/final newlines,
trailing whitespace, exact six-path scope, and status/dependency agreement. Confirm the index is
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
- Architecture contract/ADR: no change. The outward integration project composes established NN
  and Engine APIs without changing ownership or dependency direction.
- Architecture tests: no source change. The new edge is test-only in the integration-test module,
  not a production NN-to-Engine dependency; the repository-wide checkpoint still executes the
  existing architecture suite.
- Backend conformance: no change. CPU behavior and raw operation conformance are already covered;
  this task verifies only public cross-module composition.
- Production modules and Javadocs: no change because no defect or missing public seam is present.

## Architecture impact

Expected impact: None. This task composes public NN, Model, and Engine contracts only from the
outward integration-test project. If implementation reveals a missing production seam or a need
for another dependency direction, stop and report the exact conflict rather than editing around
it.

## Implementation prompt

```text
You are the clean-context implementation agent for Synaptik NN task 0025A in
/Users/phujka/IdeaProjects/Synaptik. Do not use GSD. Do not stage, commit, or push.

Read AGENTS.md, ARCHITECTURE.md, the current architecture plan, planning guide, roadmap, NN master
plan, and this task specification in full. Read NN 0025 and the directly referenced Engine,
Compiler, and CPU prerequisites. Inspect the current public layers, ModuleFactory/state APIs,
Engine lifecycle, host Tensor construction, existing convolution/lifecycle integration tests,
build file, and affected API documentation.

Implement exactly this Draft specification within its six-path ceiling. Add only the test-only NN
dependency, one black-box integration test with independent primitive-array oracles, and the
focused Training API example. Preserve the explicit Conv3d fail-closed boundary and add no
production API or behavior. Stop on architecture uncertainty or scope overflow.

Run the focused integration class, then one final repository-wide checkpoint after executable work
stabilizes. Hand the frozen diff and exact evidence to a distinct clean documentation-focused
context. That pass must follow the General, API/Javadoc, Planning, and Example profiles; finalize
the Training API and planning/no-change evidence; reuse stable Java evidence; and validate links,
scope, statuses, index state, and whitespace. Update this task, master plan, and roadmap only after
all gates pass; keep Compiler 0006C and unrelated NN 0021B–0024 Draft.
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

## Known limitations

- Standard Engine execution is CPU-only and limited to fully static supported layouts.
- The fixtures are representative, not exhaustive across all types, groups, or geometry.
- State dictionaries are in-memory Tensor-binding snapshots, not durable checkpoint bytes.
- Conv3d remains forward-only until Compiler 0006C completes and a later user-facing claim is
  selected and validated.
- The lifecycle assertions prove the documented close/ownership observations, not universal leak
  detection or performance.

## Validation evidence

Not yet executed.

## Implementation notes

Not yet executed.

## Completion summary

Not yet executed.
