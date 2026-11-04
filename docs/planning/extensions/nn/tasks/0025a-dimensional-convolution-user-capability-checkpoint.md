# Task 0025A: Dimensional-[Convolution](../../../../glossary.md#term-channels-first-convolution-layers) User-Capability [Checkpoint](../../../../glossary.md#term-checkpoint)

## Status

Complete

## Goal

Close the public user workflow for the completed channels-first `Conv1d`, [`Conv2d`](../../../../glossary.md#convolution--conv2d), and [`Conv3d`](../../../../glossary.md#conv3d--ncdhw)
layers by exercising their real [module](../../../../glossary.md#term-module) state through `Engine.standard()` and the [CPU](../../../../glossary.md#term-cpu) [backend](../../../../glossary.md#backend).
Prove known forward values for [NCW](../../../../glossary.md#term-ncw), grouped [NCHW](../../../../glossary.md#term-nchw), and grouped [NCDHW](../../../../glossary.md#conv3d--ncdhw) expressions, exact strict
state-dictionary [ownership](../../../../glossary.md#term-ownership-meanings), both reusable and one-shot [Engine](../../../../glossary.md#term-engine) paths where they establish distinct
contracts, explicit [publication](../../../../glossary.md#term-publication)/result cleanup, and the current [fail-closed](../../../../glossary.md#term-fail-closed) Conv3d training
boundary.

This is an integration [checkpoint](../../../../glossary.md#term-checkpoint) over existing production APIs:

```text
known host-backed input and state Tensor leaves
  -> strict StateDictionary load
  -> Conv1d / Conv2d / Conv3d forward expression
  -> Engine.standard()
  -> compile -> prepare -> run -> materialize
     or one-shot compute
  -> detached HostTensorValue checked against an independent Java oracle
```

The checkpoint adds no execution [facade](../../../../glossary.md#term-facade) to [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) and no layer knowledge to Engine. [Compiler](../../../../glossary.md#term-compiler) task
0006C remains Draft, so this task must not claim Conv3d [gradients](../../../../glossary.md#term-gradient) or training support.

## [Scope](../../../../glossary.md#term-scope)

- Add one black-box [integration test](../../../../glossary.md#term-integration-test) class under `testing/integration-tests` that imports only
  public Model, NN, and Engine APIs.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- Add the test project's narrow `testImplementation(project(":extensions:nn"))` dependency. This
  is test composition only; it creates no production dependency or transitive runtime facade.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [runtime glossary](../../../../glossary.md#term-runtime); [facade glossary](../../../../glossary.md#term-facade))</span>
- Update `EngineCompositionContractTest.APPROVED_INTEGRATION` with exactly one trailing ordered
  line, `testImplementation(project(":extensions:nn"))`, matching the integration build file.
  Preserve the exact-list equality assertion, ordering, Engine inventory, reverse-dependency scan,
  and every other test line; do not weaken, generalize, or replace the inventory guard.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [guard glossary](../../../../glossary.md#term-guard))</span>
- Construct the three existing concrete layer types through a deliberately small mix of direct
  construction and `ModuleFactory.standard()`:
  - direct `Conv1d`, proving the concrete public constructor without [bias](../../../../glossary.md#term-bias);
  - factory-created grouped [`Conv2d`](../../../../glossary.md#convolution--conv2d), proving the standard recipe in the main reusable workflow;
  - direct grouped [`Conv3d`](../../../../glossary.md#conv3d--ncdhw), proving the concrete forward-only type with bias.
  Existing NN unit tests already prove direct/factory parity for every rank, so duplicating both
  construction paths for all three ranks is unnecessary.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank))</span>
- Give every layer nontrivial, rank-specific kernel/stride/padding/dilation geometry. Use groups
  greater than one for Conv2d and Conv3d. Across the three fixtures cover both bias-disabled and
  bias-enabled state schemas without multiplying otherwise equivalent cases.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [stride glossary](../../../../glossary.md#term-stride); [padding glossary](../../../../glossary.md#padding); [dilation glossary](../../../../glossary.md#term-dilation); [schemas glossary](../../../../glossary.md#term-schema))</span>
- Bind every layer through a complete strict `StateDictionary` before its first forward call.
  Use exact, caller-created, gradient-eligible, host-backed weight and optional bias Tensors with
  deterministic literal values. Before load, demonstrate on the direct Conv1d fixture that<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [host glossary](../../../../glossary.md#term-host); [weight glossary](../../../../glossary.md#term-weight); [Tensors glossary](../../../../glossary.md#tensor); [fixture glossary](../../../../glossary.md#term-fixture))</span>
  `weight()` and state export fail closed while the reservation is unbound. Do not invoke
  automatic initialization in this checkpoint.<span data-glossary-navigation="true"> ([fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- For each loaded layer assert the exact exported state paths, order, `StateKind.PARAMETER`,
  candidate Tensor reference identity, parameter wrapper order, weight Shape, optional-bias
  presence, and absence of unexpected state.<span data-glossary-navigation="true"> ([reference identity glossary](../../../../glossary.md#term-reference-identity); [parameter glossary](../../../../glossary.md#term-parameter); [Shape glossary](../../../../glossary.md#shape))</span>
- Create each runtime input as a separate non-gradient caller-owned host-backed Tensor. The input,
  loaded weight, and loaded bias where present are the only provenance-free value leaves that
  carry host storage. The layer output is a storage-free expression. Record that <span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [carry glossary](../../../../glossary.md#term-carry); [host storage glossary](../../../../glossary.md#host-storage))</span>`compute` first
  inventories reachable leaves by exact Tensor identity, while final <span data-glossary-navigation="true"> ([leaves glossary](../../../../glossary.md#term-leaf))</span>`CompiledGraph.inputs()`
  remains authoritative for membership and binding order.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- Exercise `Engine.compute(output)` for Conv1d and Conv3d. This proves automatic input discovery,
  fresh compile/prepare/run/materialize/cleanup, and detached host values without repeating the
  explicit lifecycle for every rank.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [prepare glossary](../../../../glossary.md#prepare); [run glossary](../../../../glossary.md#term-run); [detached glossary](../../../../glossary.md#detached-model-results); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Exercise grouped Conv2d through `compile(List.of(output))`, `prepare(compiled)`, two
  `run(prepared, inputs)` calls, and explicit `RunResult.materialize`. Supply the exact required
  input/state Tensors in an order different from `compiled.inputs()` and assert the prepared
  handle is reusable while each run has a distinct result and publication occurrence.<span data-glossary-navigation="true"> ([publication occurrence glossary](../../../../glossary.md#publication-occurrence))</span>
- Compare every materialized output with an independent clean-Java channels-first grouped
  cross-correlation oracle implemented in the integration test. The oracle must index primitive
  arrays directly, apply stride, symmetric padding, dilation, groups, optional bias, and
  out-of-bounds zero padding, and must not call <span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle))</span>`Tensor.conv1d`, `conv2d`, `conv3d`, the layer
  under test, Compiler, CPU reference code, or another production convolution helper.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [CPU glossary](../../../../glossary.md#term-cpu); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers))</span>
- Check the public lifecycle contract observably: result/publication are open before close and
  closed after close; post-close materialization fails; caller-owned input and state storage stay
  alive after result close while their arena remains open; detached values remain readable after
  result and Engine close; <span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [materialization glossary](../../../../glossary.md#materialization); [arena glossary](../../../../glossary.md#term-arena))</span>`Engine.isClosed()` changes after close; and new Engine work fails
  after close. These assertions prove only the documented ownership/lifecycle behavior, not
  absence of every possible leak.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Verify Conv3d backward rejection through the ordinary public `Engine.backward(...)` boundary
  using a scalar reduction of the layer output and the exact loaded weight as explicit target.
  Assert <span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [target glossary](../../../../glossary.md#term-target))</span>`IllegalArgumentException` identifies `CONV3D` and ends with
  `Conv3d is forward-only until Compiler task 0006C closes its gradients`. Also assert the Engine
  remains open and a subsequent Conv3d forward compute still succeeds. This complements, rather
  than duplicates, Compiler unit evidence: it proves that an actual NN layer expression reaches
  the fail-closed inventory through the supported public Engine workflow.<span data-glossary-navigation="true"> ([fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- Add a focused layer-to-[Engine](../../../../glossary.md#term-engine) example to `docs/api/training-api.md`. The existing convolution
  section stops at expression construction and the Runtime API explains generic Engine mechanics;
  neither currently connects strict layer state, caller host storage, <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>`forward`, and `compute` in
  one user workflow. Keep the example concise and link to the Runtime API for the full reusable
  lifecycle.
- Finalize the documentation edit, current-boundary wording, links, and planning evidence through
  a distinct clean documentation-focused context in the same overall change.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- Any production Java, public API, Javadoc, Model operation, Compiler rule, Engine behavior,
  Runtime/Prepare contract, CPU lowering, or backend change.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Model glossary](../../../../glossary.md#term-model); [operation glossary](../../../../glossary.md#operation); [Compiler glossary](../../../../glossary.md#term-compiler); [Engine glossary](../../../../glossary.md#term-engine); [Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [CPU glossary](../../../../glossary.md#term-cpu); [lowering glossary](../../../../glossary.md#lowering); [backend glossary](../../../../glossary.md#backend))</span>
- Automatic/random parameter initialization, seed reproducibility, retry, concurrent first
  forward, strict-load rollback, or direct/factory parity matrices already covered by NN 0025.<span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [seed glossary](../../../../glossary.md#term-seed); [reproducibility glossary](../../../../glossary.md#artifact-manifest-and-source-provenance); [rollback glossary](../../../../glossary.md#term-rollback); [matrices glossary](../../../../glossary.md#term-matrix-meanings))</span>
- Conv1d or Conv2d gradient execution, optimizer/session behavior, a general training example, or
  any broader gradient claim. Their existing Compiler contracts remain unchanged; this checkpoint
  is forward-only apart from proving Conv3d rejection.<span data-glossary-navigation="true"> ([Conv2d glossary](../../../../glossary.md#convolution--conv2d); [gradient glossary](../../../../glossary.md#term-gradient); [optimizer glossary](../../../../glossary.md#term-optimizer); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw))</span>
- Positive Conv3d backward, gradient publication, or training support. Compiler 0006C must first
  close its exact grouped NCDHW gradient inventory before a later task may make that claim.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [NCDHW glossary](../../../../glossary.md#conv3d--ncdhw))</span>
- Exhaustive geometry/type/layout coverage, performance evidence, generated-code inspection,
  backend conformance expansion, or repeating raw Tensor convolution coverage already owned by<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [conformance glossary](../../../../glossary.md#term-conformance); [Tensor glossary](../../../../glossary.md#tensor); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers))</span>
  `EngineConvolutionIntegrationTest`.
- A new [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) execution [facade](../../../../glossary.md#term-facade), `Module.execute`, Tensor execution method, implicit state binding,
  durable checkpoint format, host-storage ownership transfer, or facade over Engine.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [host glossary](../../../../glossary.md#term-host); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Architecture-contract, focused architecture-documentation, ADR, backend-conformance, glossary,
  Javadoc, Gradle convention/plugin, or other module changes. No architecture-test change is
  allowed beyond the exact one-line ordered integration inventory update specified above.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Gradle glossary](../../../../glossary.md#term-gradle); [plugin glossary](../../../../glossary.md#term-plugin); [module glossary](../../../../glossary.md#term-module))</span>

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Roadmap](../../../roadmap.md)
- [NN master plan](../master-plan.md)<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- [NN 0025 channels-first layers](0025-channels-first-conv1d-conv2d-conv3d-layers.md)
- [Engine 0004 host materialization](../../../modules/engine/tasks/0004-explicit-host-materialization-boundary.md)<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [host glossary](../../../../glossary.md#term-host); [materialization glossary](../../../../glossary.md#materialization))</span>
- [Engine 0005 one-shot forward convenience](../../../modules/engine/tasks/0005-one-shot-forward-convenience.md)
- [Engine 0005A automatic input discovery](../../../modules/engine/tasks/0005a-automatic-input-discovery-and-compute-convenience.md)
- [Engine 0006 scalar-objective backward](../../../modules/engine/tasks/0006-one-shot-scalar-objective-backward-convenience.md)<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [objective glossary](../../../../glossary.md#term-objective))</span>
- [Engine 0008 lifecycle checkpoint](../../../modules/engine/tasks/0008-engine-lifecycle-capability-checkpoint.md)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- [Compiler 0006B Conv3d forward boundary](../../../modules/compiler/tasks/0006b-conv3d-forward-adoption-and-explicit-gradient-boundary.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw))</span>
- [Compiler 0006B6 convolution layout closure](../../../modules/compiler/tasks/0006b6-final-convolution-logical-layout-closure.md)<span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [layout glossary](../../../../glossary.md#layout))</span>
- [CPU 0008A dimensional-convolution closure](../../../backends/cpu/tasks/0008a-portable-channels-first-dimensional-convolution-closure.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- [Training API](../../../../api/training-api.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span> and [Runtime API](../../../../api/runtime-api.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)

## Architecture constraints

- NN continues to own layer construction, trainable parameter wrappers, state paths, and strict
  in-memory state loading while depending only on Model. This integration test may compose NN with
  Engine because <span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [trainable parameter glossary](../../../../glossary.md#trainable-parameter); [Model glossary](../../../../glossary.md#term-model); [integration test glossary](../../../../glossary.md#term-integration-test); [Engine glossary](../../../../glossary.md#term-engine))</span>`testing/integration-tests` is an outward test project, not a production [owner](../../../../glossary.md#term-owner).
- Model remains the sole owner of Tensor identity, provenance, descriptors, host-storage
  association, and convolution semantics. Loaded parameter values remain ordinary Tensor leaves.<span data-glossary-navigation="true"> ([leaves glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [identity glossary](../../../../glossary.md#term-identity-meanings); [provenance glossary](../../../../glossary.md#provenance); [descriptors glossary](../../../../glossary.md#term-descriptor); [host glossary](../../../../glossary.md#term-host); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- Engine remains the composition root. It owns the public CPU-only compile, prepare, run,
  publication, materialization, automatic-discovery, cleanup, and close workflows. Do not bypass
  it with Compiler, Runtime, Prepare, or CPU internals.<span data-glossary-navigation="true"> ([composition root glossary](../../../../glossary.md#term-composition-root); [CPU glossary](../../../../glossary.md#term-cpu); [compile glossary](../../../../glossary.md#compile); [prepare glossary](../../../../glossary.md#prepare); [materialization glossary](../../../../glossary.md#materialization); [Compiler glossary](../../../../glossary.md#term-compiler); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Caller input and loaded state storage are [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) for a [run](../../../../glossary.md#term-run) and remain caller-owned. A
  [`RunResult`](../../../../glossary.md#run-result--runresult) owns its [publication](../../../../glossary.md#term-publication) [lease](../../../../glossary.md#term-lease); a [`HostTensorValue`](../../../../glossary.md#host-tensor-value--hosttensorvalue) is detached and owns no closeable
  resource.<span data-glossary-navigation="true"> ([detached glossary](../../../../glossary.md#detached-model-results))</span>
- Prepared recipes remain immutable and reusable, while each run receives isolated mutable
  Runtime state. The test may observe separate result/publication identities but must not infer
  physical buffer identity or leak freedom.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [mutable glossary](../../../../glossary.md#term-mutable); [buffer glossary](../../../../glossary.md#term-buffer))</span>
- Compiler alone owns autograd. Conv3d remains rejected by every backward-capable request before
  derivative construction until Compiler 0006C completes. Gradient-eligible NN parameters do not
  imply a supported gradient rule.<span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [derivative glossary](../../../../glossary.md#term-derivative); [Gradient glossary](../../../../glossary.md#term-gradient))</span>
- CPU remains the only standard Engine backend and accepts the existing fully static,
  resolved-layout dimensional-convolution domain. The test adds no capability or fallback.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [layout glossary](../../../../glossary.md#layout))</span>
- If implementation needs a production edit, new API, changed ownership/dependency rule, or
  broader execution topology, stop and report the conflict.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [dependency glossary](../../../../glossary.md#term-dependency); [topology glossary](../../../../glossary.md#term-topology))</span>

## Current-state evidence

- `Conv1d`, [`Conv2d`](../../../../glossary.md#convolution--conv2d), and [`Conv3d`](../../../../glossary.md#conv3d--ncdhw) are final public `UnaryTensorModule` types with exact
  rank-specific scalar constructors, <span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [scalar glossary](../../../../glossary.md#term-scalar))</span>`weight()`, optional `bias()`, and `forward(Tensor)`.
- `ModuleFactory.standard()` already exposes matching stateless recipes and retains no layer or
  configuration state.
- `Module.loadStateDictionary(...)` validates the complete target tree before installing exact
  candidate Tensor references. A reserved convolution group can therefore bind without sampling,
  allocation, or another Tensor ID. Export order is <span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [Tensor glossary](../../../../glossary.md#tensor); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters))</span>[`weight`](../../../../glossary.md#term-weight), then optional [`bias`](../../../../glossary.md#term-bias).
- `Engine.standard()` supplies the current [CPU](../../../../glossary.md#term-cpu)-only composition. [`compile`](../../../../glossary.md#compile), [`prepare`](../../../../glossary.md#prepare), [`run`](../../../../glossary.md#term-run),
  `RunResult.materialize`, and `compute` are public. Explicit runs bind current host associations
  by Tensor ID; <span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host))</span>`compute` inventories reachable provenance-free leaves and then follows final
  Compiler input metadata.<span data-glossary-navigation="true"> ([leaves glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- `EngineConvolutionIntegrationTest` already proves raw Tensor-level public forward execution for
  all three ranks. It does not instantiate NN layers, load their state dictionaries, or connect
  layer-owned state to Engine discovery/binding.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Engine glossary](../../../../glossary.md#term-engine); [binding glossary](../../../../glossary.md#term-binding))</span>
- The current uncommitted implementation adds
  `testImplementation(project(":extensions:nn"))` after the existing CPU test [dependency](../../../../glossary.md#term-dependency) in
  `testing/integration-tests/build.gradle.kts`; the new integration test consumes that exact edge.
  This test-only project edge is necessary and compatible with the outward integration role.<span data-glossary-navigation="true"> ([integration test glossary](../../../../glossary.md#term-integration-test))</span>
- The first `./gradlew test --rerun-tasks` checkpoint reached the architecture suite and exposed
  one planned inventory mismatch, not a product-test failure:<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
  `EngineCompositionContractTest.engineHasOnlyTheApprovedOrderedDirectDependencies()` still
  expected the pre-NN six-line `APPROVED_INTEGRATION` list while the build file correctly reported
  the new NN line seventh. Its XML records one test, one failure, and zero errors. This diagnosis
  authorized only the matching one-line expected-inventory addition and does not constitute final
  checkpoint evidence. After synchronizing that exact inventory, the implementation context's
  fresh complete repository run succeeded and is the controlling checkpoint evidence recorded
  below.
- `docs/api/training-api.md` explains the convolution layer/state contract and stops after
  expression construction. `docs/api/runtime-api.md` explains generic Engine lifecycle and
  storage ownership, but has no layer-state workflow.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- [Compiler](../../../../glossary.md#term-compiler) 0006C has no detailed task and remains Draft. Current `AutogradPreflight` emits the
  exact Conv3d forward-only failure named above.

## Exact [fixture](../../../../glossary.md#term-fixture) [matrix](../../../../glossary.md#term-matrix-meanings) and [oracle](../../../../glossary.md#term-test-oracle) contract

Use [`FLOAT32`](../../../../glossary.md#term-float32) throughout and these exact fixtures so the clean implementation context does not
silently broaden or simplify geometry:

| Fixture | Construction | Input | [Weight](../../../../glossary.md#term-weight) | [Bias](../../../../glossary.md#term-bias) | Geometry | Output |
|---|---|---|---|---|---|---|
| Conv1d | direct, groups `1`, bias `false` | `[1,2,6]` | `[2,2,3]` | absent | [kernel](../../../../glossary.md#kernel) `3`, [stride](../../../../glossary.md#term-stride) `2`, [padding](../../../../glossary.md#padding) `1`, [dilation](../../../../glossary.md#term-dilation) `2` | `[1,2,2]` |
| [Conv2d](../../../../glossary.md#convolution--conv2d) | `ModuleFactory.standard()`, groups `2`, bias `true` | `[1,4,4,5]` | `[4,2,2,3]` | `[4]` | kernel `2x3`, stride `2x1`, padding `1x1`, dilation `1x2` | `[1,4,3,3]` |
| [Conv3d](../../../../glossary.md#conv3d--ncdhw) | direct, groups `2`, bias `true` | `[1,4,3,4,5]` | `[4,2,2,2,2]` | `[4]` | kernel `2x2x2`, stride `1x2x1`, padding `1x0x1`, dilation `1x1x2` | `[1,4,4,2,5]` |

Construct each layer with `ParameterInitialization.zeros()` and a fixed [seed](../../../../glossary.md#term-seed), then bypass that
unused automatic policy by loading complete known state before `forward`. Populate inputs and
weights from small deterministic primitive-array patterns defined by the test, for example signed
fractions derived from the flat index; use the exact bias values
`{0.5f, -0.25f, 1.0f, -1.0f}` for both biased fixtures. The test must preserve those arrays as the
oracle inputs rather than reading [parameter](../../../../glossary.md#term-parameter) storage back through production code.

Implement separate private [rank](../../../../glossary.md#term-rank)-specific oracle entry points or one private [dimension](../../../../glossary.md#dimension)-parameterized
loop whose indexing remains explicit. For every output [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), derive the group, visit only
that group's input channels, map kernel coordinates through stride/padding/dilation, skip
out-of-range input coordinates as zero, accumulate products in `float`, and add the selected bias
once. Compare complete [row-major](../../../../glossary.md#term-row-major) outputs with a documented small floating [tolerance](../../../../glossary.md#term-tolerance) and assert the
exact expected [Shapes](../../../../glossary.md#shape) above. The test names should make the three obligations visible, such as:

- `executesStrictLoadedConv1dThroughOneShotCompute`;
- `reusesPreparedGroupedConv2dAndObservesPublicationCleanup`; and
- `executesStrictLoadedGroupedConv3dAndRejectsBackward`.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.testing.integration` — black-box cross-[module](../../../../glossary.md#term-module) integration tests.
- `io.github.pho001.synaptik.nn.layers` and `.module` — unchanged public layers, factory, and state
  dictionary APIs.
- `io.github.pho001.synaptik.engine` — unchanged ordinary [Engine](../../../../glossary.md#term-engine) [lifecycle](../../../../glossary.md#lifecycle).
- [Model](../../../../glossary.md#term-model) [tensor](../../../../glossary.md#tensor)/storage/datatype/[layout](../../../../glossary.md#layout)/[Shape](../../../../glossary.md#shape) packages — unchanged [leaf](../../../../glossary.md#term-leaf) and [host](../../../../glossary.md#term-host)-storage creation.

Packages added or changed:

- None. One test class is added to the existing integration-test package.

Type placement:

- `io.github.pho001.synaptik.testing.integration.NnConvolutionEngineIntegrationTest` — package-
  private JUnit integration owner for the layer-to-Engine checkpoint and its private primitive-
  array oracle/helpers.<span data-glossary-navigation="true"> ([JUnit glossary](../../../../glossary.md#term-junit); [owner glossary](../../../../glossary.md#term-owner); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>

## Affected files

Expected implementation/test/documentation paths:

1. `testing/integration-tests/build.gradle.kts`
2. `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/NnConvolutionEngineIntegrationTest.java` (new)
3. `testing/architecture-tests/src/test/java/io/github/pho001/synaptik/testing/architecture/EngineCompositionContractTest.java`
4. `docs/api/training-api.md`
5. this task specification
6. `docs/planning/extensions/nn/master-plan.md`
7. `docs/planning/roadmap.md`

Review without modification: [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [convolution](../../../../glossary.md#term-channels-first-convolution-layers) production/Javadocs and unit tests; [Module](../../../../glossary.md#term-module),
ModuleFactory, StateDictionary, and [Parameter](../../../../glossary.md#term-parameter) APIs; [Engine](../../../../glossary.md#term-engine) production/Javadocs and existing
integration fixtures; [Tensor](../../../../glossary.md#tensor)/[Compile](../../../../glossary.md#compile)/[Runtime](../../../../glossary.md#term-runtime) APIs; architecture contracts/tests; [backend](../../../../glossary.md#backend)
[conformance](../../../../glossary.md#term-conformance); glossary; [Gradle](../../../../glossary.md#term-gradle) settings/conventions; every other module and plan. For architecture
tests, review all files but modify only the one exact `APPROVED_INTEGRATION` line above.

## Maximum [scope](../../../../glossary.md#term-scope)

At most the exact seven paths above: one test-only [dependency](../../../../glossary.md#term-dependency) edit, one new [integration test](../../../../glossary.md#term-integration-test), one
one-line ordered architecture inventory update, one focused public [API](../../../../glossary.md#term-application-programming-interface-api)-documentation edit, and
three synchronized [planning](../../../../glossary.md#term-planning) records. If implementation needs another helper, production file,
test [owner](../../../../glossary.md#term-owner), build file, documentation page, or architecture [artifact](../../../../glossary.md#term-artifact), stop and amend this Draft
specification before proceeding.

## Acceptance criteria

- Direct Conv1d, factory grouped Conv2d, and direct grouped Conv3d construct only their existing
  APIs and produce usable layer forward expressions after strict state load.<span data-glossary-navigation="true"> ([Conv2d glossary](../../../../glossary.md#convolution--conv2d))</span>
- Conv1d is bias-free; Conv2d and Conv3d are biased; both higher-rank fixtures use groups greater
  than one. Every fixture uses nontrivial rank-specific geometry, and expected Shape calculations
  agree with the public layer result and final publication.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [fixture glossary](../../../../glossary.md#term-fixture); [Shape glossary](../../../../glossary.md#shape); [publication glossary](../../../../glossary.md#term-publication))</span>
- Every state dictionary contains exactly [`weight`](../../../../glossary.md#term-weight) followed by optional [`bias`](../../../../glossary.md#term-bias), every entry is a
  parameter, exact loaded Tensor identities are retained/exported, wrapper order matches, and the
  loaded numerical values demonstrably determine Engine output. The selected pre-load Conv1d
  access/export checks reject incomplete state, and strict load makes its first forward usable.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter); [Tensor glossary](../../../../glossary.md#tensor); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Only the input and loaded state leaves carry caller-owned host storage. Explicit compilation
  reports exactly those bindable Tensor identities; run accepts them in a different order.<span data-glossary-navigation="true"> ([carry glossary](../../../../glossary.md#term-carry); [host storage glossary](../../../../glossary.md#host-storage); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [run glossary](../../../../glossary.md#term-run))</span>
- Conv1d and [Conv3d](../../../../glossary.md#conv3d--ncdhw) pass one-shot `compute` with detached expected values. Grouped Conv2d passes
  one compile/prepare and two runs through the same prepared handle, with distinct run/publication
  objects and correct explicit materialization.<span data-glossary-navigation="true"> ([detached glossary](../../../../glossary.md#detached-model-results); [compile glossary](../../../../glossary.md#compile); [prepare glossary](../../../../glossary.md#prepare); [materialization glossary](../../../../glossary.md#materialization))</span>
- All three results match independent primitive-array Java oracles that implement grouped
  channels-first cross-correlation directly. No expected value is obtained from a Tensor
  convolution, NN layer, Compiler, CPU helper, or the operation under test.<span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [Compiler glossary](../../../../glossary.md#term-compiler); [CPU glossary](../../../../glossary.md#term-cpu); [operation glossary](../../../../glossary.md#operation))</span>
- Observable ownership/lifecycle assertions pass: result/publication close state, post-close
  rejection, caller storage survival through result close, detached post-close readability,
  Engine close state, and rejection of new work after Engine close. No broader leak claim appears.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Public `Engine.backward` on a [scalar](../../../../glossary.md#term-scalar) depending on Conv3d fails closed with the exact current
  `CONV3D`/Compiler-0006C diagnostic; the Engine stays open and remains capable of a later valid
  forward compute. No Conv3d gradient or training success is claimed.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient))</span>
- No Conv1d/Conv2d gradient test is added because this task does not establish a training workflow
  and their existing Compiler contracts are unchanged.
- The Training API gains one concise layer-state-to-Engine example, accurately distinguishes
  expression construction from execution, identifies which leaves own host storage, explains
  detached results and cleanup, links to the reusable Runtime workflow, and keeps Conv3d
  forward-only wording explicit.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- `EngineCompositionContractTest` gains exactly the new [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) test [dependency](../../../../glossary.md#term-dependency) as the seventh ordered
  `APPROVED_INTEGRATION` entry. Its strict equality check and every other assertion remain intact.
- No production/Javadoc, glossary, architecture contract/documentation/ADR, conformance, settings,
  or other module path changes. A separate clean documentation-focused pass finalizes the API
  text and reasoned no-change conclusions without repeating stable Java tests.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [architecture contract glossary](../../../../glossary.md#architecture-contract); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [conformance glossary](../../../../glossary.md#term-conformance); [module path glossary](../../../../glossary.md#term-module-path))</span>
- Focused integration validation and one repository-wide capability-checkpoint run pass. Links,
  anchors, fences, terminology, exact seven-path scope, empty staging, LF/final newlines, status
  agreement, and <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` pass before the task can become Complete.

## Tests / [validation](../../../../glossary.md#term-validation)

During implementation, [run](../../../../glossary.md#term-run) the focused new integration class while stabilizing executable work:

```bash
./gradlew :testing:integration-tests:test \
  --tests io.github.pho001.synaptik.testing.integration.NnConvolutionEngineIntegrationTest
```

After Java, the test [dependency](../../../../glossary.md#term-dependency), and the exact architecture inventory line stabilize, rerun the
complete capability [checkpoint](../../../../glossary.md#term-checkpoint). The earlier root run failed the stale expected inventory and is
diagnostic only; it cannot be reused as the final repository result:

```bash
./gradlew test --rerun-tasks
```

Record exact test totals from the fresh final [Gradle](../../../../glossary.md#term-gradle) XML and confirm execution of the new
integration class,
`EngineConvolutionIntegrationTest`, the [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [convolution](../../../../glossary.md#term-channels-first-convolution-layers)/state/factory suites, [Engine](../../../../glossary.md#term-engine) [lifecycle](../../../../glossary.md#lifecycle)
suites, and architecture tests, including the passing strict ordered Engine composition inventory.
The repository-wide command is required because this is a named capability checkpoint and changes
a test-project dependency; do not separately rerun those stable suites afterward.

Documentation-focused pass, after it finalizes the Training [API](../../../../glossary.md#term-application-programming-interface-api) and [planning](../../../../glossary.md#term-planning) evidence:

```bash
git diff --check
git status --short -uall
git diff --cached --check
```

Also validate local Markdown file links and heading anchors, unique headings, balanced fences,
current-versus-planned terminology, exact state/failure names in the example, LF/final newlines,
trailing whitespace, exact seven-path [scope](../../../../glossary.md#term-scope), and status/dependency agreement. Confirm the index is
empty. No [Javadoc](../../../../glossary.md#term-javadoc) generation is required because no production Java or Javadoc changes; review
the affected public Javadocs against the example and record that they remain accurate. Do not run
Java tests in the documentation context unless it changes executable Java or identifies a
specific stale-evidence risk.

## Dependencies

- [NN 0025](0025-channels-first-conv1d-conv2d-conv3d-layers.md) — Complete; supplies the three
  layers, recipes, state lifecycle, unit tests, and documentation boundary.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Engine 0004](../../../modules/engine/tasks/0004-explicit-host-materialization-boundary.md) —
  Complete; supplies explicit detached host materialization and result lease behavior.<span data-glossary-navigation="true"> ([detached glossary](../../../../glossary.md#detached-model-results); [host glossary](../../../../glossary.md#term-host); [materialization glossary](../../../../glossary.md#materialization); [lease glossary](../../../../glossary.md#term-lease))</span>
- Engine 0005 and 0005A — Complete; supply one-shot compute and authoritative automatic leaf
  discovery/input selection.<span data-glossary-navigation="true"> ([leaf glossary](../../../../glossary.md#term-leaf))</span>
- Engine 0006 and 0008 — Complete; supply public backward failure traversal and the closed public
  lifecycle checkpoint.
- [CPU 0008A](../../../backends/cpu/tasks/0008a-portable-channels-first-dimensional-convolution-closure.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
  — Complete; supplies truthful Conv1d composition and grouped Conv3d forward execution.<span data-glossary-navigation="true"> ([Conv1d composition glossary](../../../../glossary.md#conv1d-composition); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw))</span>
- Compiler 0006B and 0006B6 — Complete; supply Conv3d forward adoption, exact rejection, final
  convolution layouts, and positive raw Tensor Engine fixtures.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [layouts glossary](../../../../glossary.md#layout); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Compiler 0006C — Draft and therefore not a satisfied dependency. It is required only before a
  task claims positive Conv3d training. This checkpoint instead locks the current public
  fail-closed boundary.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>

The global roadmap explicitly selected [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) 0025 and 0025A as an isolated dimensional-[convolution](../../../../glossary.md#term-channels-first-convolution-layers)
sequence after [Engine](../../../../glossary.md#term-engine) 0008. Unrelated NN 0021B–0024 remain Draft; their recurrent/Data contracts
and files do not overlap this [checkpoint](../../../../glossary.md#term-checkpoint).

## Follow-up tasks

- Return to the next frontier already recorded by the current roadmap after this checkpoint;
  current planning points to tools/tuning 0002 rather than inventing another NN convolution task.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [planning glossary](../../../../glossary.md#term-planning); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers))</span>
- Compiler 0006C remains the independent future owner of Conv3d adjoint expressibility and
  gradient closure. If it completes, a separately selected user-training checkpoint may replace
  the failure assertion with positive evidence; this task must not anticipate that result.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [owner glossary](../../../../glossary.md#term-owner); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [adjoint glossary](../../../../glossary.md#term-adjoint); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Additional convolution padding modes, layouts, transposed/depthwise-specialized layers,
  performance work, or durable checkpoint transport require their own future tasks.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [layouts glossary](../../../../glossary.md#layout))</span>

## Documentation, [Javadoc](../../../../glossary.md#term-javadoc), architecture, [conformance](../../../../glossary.md#term-conformance), and integration impact

- Integration: add exactly one layer-level public Engine class because existing raw Tensor
  integration cannot prove module state ownership/load/discovery.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [Tensor glossary](../../../../glossary.md#tensor); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Documentation: add a focused Training API example because the present construction-only example
  and generic Runtime workflow leave the layer-to-Engine handoff implicit. Do not duplicate the
  full Runtime lifecycle guide.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Runtime glossary](../../../../glossary.md#term-runtime); [handoff glossary](../../../../glossary.md#term-handoff); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Javadoc: no change. Existing layer, state-dictionary, Engine, result, and host-value Javadocs
  already describe the contracts consumed here; review them in the documentation pass.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host))</span>
- Glossary: no change. The task introduces no new reusable term or changes the meaning of layer,
  state dictionary, convolution, publication, or host materialization.<span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [publication glossary](../../../../glossary.md#term-publication); [materialization glossary](../../../../glossary.md#materialization))</span>
- Architecture contract, focused architecture documentation, and ADRs: no change. The outward
  integration project composes established NN and Engine APIs without changing ownership or
  dependency direction.<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools))</span>
- Architecture tests: update only `EngineCompositionContractTest.APPROVED_INTEGRATION` with the
  exact trailing NN test dependency already present in the integration build. Retain strict
  ordered-list equality and all inward-dependency assertions. This records an approved outward
  test composition edge, not a production NN-to-Engine dependency or architecture-rule change.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
- Backend conformance: no change. CPU behavior and raw operation conformance are already covered;
  this task verifies only public cross-module composition.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [CPU glossary](../../../../glossary.md#term-cpu); [operation glossary](../../../../glossary.md#operation))</span>
- Production [modules](../../../../glossary.md#term-module) and Javadocs: no change because no defect or missing public seam is present.

## Architecture impact

Expected architecture-contract and [dependency](../../../../glossary.md#term-dependency)-direction impact: None. This task composes public
[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context), [Model](../../../../glossary.md#term-model), and [Engine](../../../../glossary.md#term-engine) contracts only from the outward integration-test project. Nevertheless,
`EngineCompositionContractTest.APPROVED_INTEGRATION` must gain the exact one trailing
`testImplementation(project(":extensions:nn"))` entry because that source inventory enforces the
outward test project's approved ordered dependencies. Synchronizing this strict expected inventory
is not an architecture rule, production dependency, [module](../../../../glossary.md#term-module)-[ownership](../../../../glossary.md#term-ownership-meanings), or dependency-direction
change. If implementation reveals a missing production seam or a need for another [dependency
direction](../../../../glossary.md#java-platform-and-build-tools), stop and report the exact conflict rather than editing around it.

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
  initialization behavior is already exhaustively covered by NN 0025 unit tests.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [Engine glossary](../../../../glossary.md#term-engine); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- One-shot execution covers Conv1d and Conv3d; reusable execution covers grouped Conv2d twice.
  This assigns each public workflow a distinct proof instead of repeating both for every rank.<span data-glossary-navigation="true"> ([Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [rank glossary](../../../../glossary.md#term-rank))</span>
- The independent oracle stays private in the one integration class. A shared production/test
  convolution utility would either duplicate a semantic owner or weaken independence.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [owner glossary](../../../../glossary.md#term-owner))</span>
- Conv3d backward rejection is in scope at the public Engine boundary because it is a visible user
  capability limit and confirms the layer expression reaches Compiler preflight. Positive
  Conv1d/Conv2d gradient execution is not needed to close this forward checkpoint.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [Compiler glossary](../../../../glossary.md#term-compiler); [preflight glossary](../../../../glossary.md#term-preflight); [gradient glossary](../../../../glossary.md#term-gradient); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- The Training API receives the focused example; the Runtime API already owns the complete Engine
  lifecycle explanation and should be linked, not duplicated.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Runtime glossary](../../../../glossary.md#term-runtime); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- The architecture guard remains strict. Adding the exact seventh expected integration dependency
  is preferable to excluding NN lines, sorting dynamically, weakening equality, or broadening an
  allow pattern because the test intentionally locks both membership and order.<span data-glossary-navigation="true"> ([guard glossary](../../../../glossary.md#term-guard); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Known limitations

- Standard [Engine](../../../../glossary.md#term-engine) execution is [CPU](../../../../glossary.md#term-cpu)-only and limited to fully static supported [layouts](../../../../glossary.md#layout).
- The fixtures are representative, not exhaustive across all types, groups, or geometry.
- State dictionaries are in-memory [Tensor](../../../../glossary.md#tensor)-[binding](../../../../glossary.md#term-binding) snapshots, not durable [checkpoint](../../../../glossary.md#term-checkpoint) bytes.
- Conv3d remains forward-only until Compiler 0006C completes and a later user-facing claim is
  selected and validated.<span data-glossary-navigation="true"> ([Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
- The lifecycle assertions prove the documented close/ownership observations, not universal leak
  detection or performance.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- [Planning](../../../../glossary.md#term-planning) context `01a0b5f5-1cc0-7d42-b497-82c148a6d01b` selected this bounded seven-path
  checkpoint. Implementation context <span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>`01a0b600-a3a8-79c3-a17a-00a41cb3f1d8` added the test-only
  NN dependency, the public-API-only integration class, and the exact trailing strict architecture
  inventory entry. Documentation context <span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>`01a0b611-62b6-70b0-a9a7-d3b6291b0fa6` independently
  reviewed the frozen executable diff and finalized the Training API and planning records.
- The implementation context's focused
  `./gradlew :testing:integration-tests:test --tests io.github.pho001.synaptik.testing.integration.NnConvolutionEngineIntegrationTest`
  run passed 3 tests with zero failures, errors, or skips. After the inventory amendment, the
  focused <span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>`EngineCompositionContractTest` run passed its one test with zero failures, errors, or
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
- The controlling XML confirms the new [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) integration class passed 3/3 and the existing
  `EngineConvolutionIntegrationTest` passed 3/3. NN `Conv1dTest` passed 3, `Conv2dTest` 4,
  `Conv3dTest` 3, `ConvolutionInitializationTest` 8,
  `ConvolutionStateDictionaryTest` 7, and `ModuleFactoryTest` 10, without skips. Engine advanced
  lifecycle passed 6, typed lifecycle 26, standard composition 6, representative execution 24,
  and the integration lifecycle/composition suites passed. The seven architecture suites passed
  all nine tests without skips, including the strict ordered Engine composition inventory.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Structural review confirmed exactly the three executable paths, public Model/NN/Engine imports
  only, independent primitive Java 1D/2D/3D oracles, no forbidden inward helper or import, and the
  exact one-line architecture inventory change. LF/final-newline, empty-index, and executable diff
  checks passed in the implementation context.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- The documentation pass added one strict-loaded factory [`Conv2d`](../../../../glossary.md#convolution--conv2d)-to-Engine example. It uses
  caller-owned host-backed input, weight, and bias leaves; complete ordered parameter state;
  storage-free <span data-glossary-navigation="true"> ([weight glossary](../../../../glossary.md#term-weight); [bias glossary](../../../../glossary.md#term-bias); [parameter glossary](../../../../glossary.md#term-parameter))</span>`forward` expression construction; one-shot `Engine.compute`; and a [detached](../../../../glossary.md#detached-model-results)
  `[1,1,2,2]` [host](../../../../glossary.md#term-host) result whose four values are `-3.5f`. It links to Runtime's reusable lifecycle
  and explicitly preserves the forward-only Conv3d/Compiler-0006C-Draft boundary.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
- Documentation validation checked every local Markdown file target and relevant heading anchor,
  unique headings, balanced fences, glossary terminology, current-versus-planned wording, example
  public names/signatures/state order/failure boundaries, status/dependency/frontier agreement,
  LF/final newlines, trailing whitespace, exact seven-path combined scope, and empty staging.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [signatures glossary](../../../../glossary.md#term-signature); [scope glossary](../../../../glossary.md#term-scope))</span>
  `git diff --check` and `git diff --cached --check` passed. Java tests and Javadoc were not rerun:
  this pass changed no executable Java or Javadoc, so it reused the fresh implementation evidence
  as required by the documentation workflow.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- [SHA-256](../../../../glossary.md#term-sha-256) [hashes](../../../../glossary.md#term-cryptographic-hash) of the frozen executable paths were identical before and after documentation:
  `e3305bd7b0fde58a987f34f62f08de9f4db6731dc3195bbab45c098fa40b3747` for the integration
  build, `cac176b2eca027a348ac3109acfdb01c86c44cc72ae9baf137dac6ab57658611` for the new
  integration class, and `1f32e81c5015b3c7de496d11f3330d8b84125b594cf5426bccba137a158c03fd`
  for the architecture inventory test.
- Final status checks confirm NN 0025 and 0025A are Complete, Compiler 0006C and NN 0021B–0024
  remain Draft, tools/tuning 0002 is the next recorded frontier, and no detailed tools/tuning
  0002 task was created.

## Implementation notes

- Added the integration project's narrow test-only [dependency](../../../../glossary.md#term-dependency) on `extensions:nn` and recorded it
  as the seventh ordered integration dependency in the unchanged strict architecture guard.<span data-glossary-navigation="true"> ([guard glossary](../../../../glossary.md#term-guard))</span>
- Added one public black-box integration class covering direct Conv1d, factory grouped Conv2d,
  direct grouped Conv3d, complete strict state loading, exact state identity/order, automatic and
  reusable Engine paths, detached materialization and cleanup, independent numerical oracles, and
  the exact public Conv3d backward rejection. No production behavior or facade changed.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [Engine glossary](../../../../glossary.md#term-engine); [detached glossary](../../../../glossary.md#detached-model-results); [materialization glossary](../../../../glossary.md#materialization); [facade glossary](../../../../glossary.md#term-facade))</span>
- Added the focused Training API example and synchronized this task, the NN master plan, and the
  global roadmap after the fresh full checkpoint passed.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Existing production Javadocs for `Conv1d`, [`Conv2d`](../../../../glossary.md#convolution--conv2d), [`Conv3d`](../../../../glossary.md#conv3d--ncdhw), `ModuleFactory`,
  `StateDictionary`, `StateEntry`, `StateKind`, Tensor/host-storage APIs, and Engine compile/run/
  materialization types remain accurate: the change composes their documented state, expression,
  borrowed-storage, execution, and detached-value contracts without changing any signature or
  behavior. No production Javadoc edit or generation is required.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [host glossary](../../../../glossary.md#term-host); [compile glossary](../../../../glossary.md#compile); [run glossary](../../../../glossary.md#term-run); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [signature glossary](../../../../glossary.md#term-signature); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- The glossary already defines channels-first convolution layers, strict in-memory state,
  standard module factory, host storage, publication occurrence, and host tensor value with the
  meanings used here. The checkpoint introduces no reusable term or changed definition, so no
  glossary edit is warranted.<span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [standard module factory glossary](../../../../glossary.md#standard-module-factory); [host storage glossary](../../../../glossary.md#host-storage); [publication occurrence glossary](../../../../glossary.md#publication-occurrence); [host tensor value glossary](../../../../glossary.md#host-tensor-value--hosttensorvalue))</span>
- No architecture contract, current architecture page, or ADR changed because the new edge is
  outward test composition only and production dependency direction is unchanged. No backend-
  conformance change is needed because backend operation behavior is unchanged. No production
  Java, settings, convention, or other-module change is needed.<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [operation glossary](../../../../glossary.md#operation); [module glossary](../../../../glossary.md#term-module))</span>

## Completion summary

- Completed changes: Closed the dimensional-convolution user checkpoint with strict-loaded
  Conv1d/Conv2d/Conv3d public Engine execution, independent expected values, lifecycle and
  fail-closed Conv3d backward evidence, plus one concise user example.<span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [Engine glossary](../../../../glossary.md#term-engine); [lifecycle glossary](../../../../glossary.md#lifecycle); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- Files changed or created: Exactly the seven paths listed in this task—three frozen executable
  paths and four finalized documentation/planning paths.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: Reused the controlling 3-test focused integration pass, 1-test focused
  architecture pass, and successful 3,217-test/498-suite repository checkpoint from implementation
  context <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>`01a0b600-a3a8-79c3-a17a-00a41cb3f1d8`; the documentation context passed all required
  Markdown, scope, status, index, hash, newline, whitespace, and diff checks without running Java.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [hash glossary](../../../../glossary.md#term-hash))</span>
- Documentation-agent review: Complete in clean context
  `01a0b611-62b6-70b0-a9a7-d3b6291b0fa6` under the General, API/Javadoc, Planning, and Example
  profiles.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Documentation impact: The Training API now connects strict NN state, caller host storage,
  expression construction, standard Engine execution, detached results, and the reusable Runtime
  workflow while preserving all current limitations.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [host storage glossary](../../../../glossary.md#host-storage); [detached glossary](../../../../glossary.md#detached-model-results); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Javadoc review: Existing affected public contracts remain accurate; no source change or Javadoc
  run was required.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- Glossary impact: No new or changed reusable terminology; existing entries remain accurate.
- Architecture and conformance impact: None beyond the exact strict inventory record for the
  outward integration-test dependency.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Unresolved issues: None.
- Follow-up required: tools/tuning 0002 is the next recorded planning frontier; Compiler 0006C
  remains the separate Draft prerequisite for any future positive Conv3d-gradient claim.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [gradient glossary](../../../../glossary.md#term-gradient))</span>

Status: Complete
