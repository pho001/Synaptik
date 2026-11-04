# Task Checkpoint0012: General-scan in-memory copy and [rebind](../../../../glossary.md#capture-graph-ports-and-reconstruction)

> **Editorial reading context — not a status update.** A general scan contains a step body
> with its own local identities. Checkpoint preserves that attachment while renumbering the
> outer graph; Compiler, not a second Checkpoint body interpreter, reconstructs it. Copy reads
> named state into private storage; rebind substitutes exact-descriptor inputs symbolically
> and reads no values. These in-memory operations are independent of disk encoding and backend
> execution. Today's [general-scan copy/rebind explanation](../../../../reference/checkpoint/graph.md#general-scan-in-memory-copy-and-rebind)
> includes later capability context; the temporary NN guards and native-environment exception
> recorded below remain facts of this task's own stage.

## Status

Complete — main accepted implementation, independent review and scoped [validation](../../../../glossary.md#term-validation) on 2026-10-23,
with the unrelated native-environment root exception recorded below; unrestricted root is NOT green.
Historical [Ready](../../../../glossary.md#term-ready)/launch verification: approved NN0030, Complete Compiler0012/Checkpoint0008–0011
and low-level join, accepted 274-path base/CPU0036, matching HEAD/origin, empty index and inactive
conflicts verified before the clean executor launched. [Checkpoint](../../../../glossary.md#term-checkpoint)/[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/[Compiler](../../../../glossary.md#term-compiler) frontiers: None.

## Change class

Class C — scoped-body/[ownership](../../../../glossary.md#term-ownership-meanings) and cross-consumer boundary; clean executor and independent clean code/docs review mandatory.

## Goal and [scope](../../../../glossary.md#term-scope)

Make existing `GraphCheckpoint.copy(Source)` and `rebind(Source,List<Tensor>)` preserve [general
SCAN](../../../../glossary.md#general-differentiable-scan) body attachments while normalizing outer IDs. Body IDs remain independently scoped.
Compiler0012 already owns complete [cold reconstruction](../../../../glossary.md#term-cold-reconstruction); [Checkpoint](../../../../glossary.md#term-checkpoint) must not clone bodies or
maintain a body [opcode](../../../../glossary.md#term-opcode) inventory/reconstruction table. Preserve applicable [Model](../../../../glossary.md#term-model) [dtypes](../../../../glossary.md#data-type--datatype) for exact
[Compiler](../../../../glossary.md#term-compiler)-reconstructible [signatures](../../../../glossary.md#term-signature), independently of [CPU](../../../../glossary.md#term-cpu) support; not blanket all-[PURE](../../../../glossary.md#term-pure) [admission](../../../../glossary.md#term-admission).

Keep disk save/load closed. Add only minimal compatibility guards in existing [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [owners](../../../../glossary.md#term-owner) so
Checkpoint [widening](../../../../glossary.md#term-widening) cannot accidentally open general-scan acquisition/compute/train. Reject through
ordinary public Model [provenance](../../../../glossary.md#provenance) and `ScanKind.SCAN` before copy/setup; no new Compiler edge.

## Non-goals

No new production type/package/public [API](../../../../glossary.md#term-application-programming-interface-api), [dependency](../../../../glossary.md#term-dependency), wire/version, architecture policy or [ADR](../../../../glossary.md#term-architecture-decision-record-adr);
no [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [qualification](../../../../glossary.md#term-qualification)/workflow [widening](../../../../glossary.md#term-widening), fixed-scan [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt), [padding](../../../../glossary.md#padding), [backend](../../../../glossary.md#backend)/numerical/[runtime](../../../../glossary.md#term-runtime) change,
T-[unrolling](../../../../glossary.md#term-unrolling), per-step/per-compute/train source reconstruction, benchmark or native/[device](../../../../glossary.md#term-device) opt-in.
Do not reopen Compiler0012 or require CPU0036 performance qualification for cold copying.
STOP if a new type, policy, missing contract or architecture decision becomes necessary.

## Contracts

Read AGENTS.md, affected source/tests and these exact headings via the root [scope](../../../../glossary.md#term-scope) index, not predecessor histories.

- [Root](../../../../../ARCHITECTURE.md): `Authority, incorporation, and precedence`, `Core lifecycle`,
  `Core invariants`, `Scope-indexed normative contracts`, `Module-ownership routing`, `Dependency rules`
  (actual Checkpoint/NN edges), <span data-glossary-navigation="true"> ([Checkpoint glossary](../../../../glossary.md#term-checkpoint); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>`Testing requirements`.
- [Extensions owner](../../../../architecture/contracts/extensions-training.md#extensionscheckpoint)<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>:
  `extensions/checkpoint`, `Generic graph source and values`, `Canonical graph-model version 1`,
  `Canonical graph-model version 2`, `Canonical graph-model version 3`; also
  [Owned tensor Models](../../../../architecture/contracts/extensions-training.md#owned-tensor-models)<span data-glossary-navigation="true"> ([tensor glossary](../../../../glossary.md#tensor))</span>
  for the independently closed NN acquisition gate and unchanged fixed-scan training rejection.
- [Scan owner](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan):
  [`General differentiable scan`](../../../../glossary.md#general-differentiable-scan), `Source body, attachments and scoped identities`,
  `Canonical residual ports and demand`, `Traversal, empty domains and declared qualification`.
- [Compiler owner](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>:
  `Compile lifecycle`, `Bounded general-scan cold reconstruction` — exact flat/body signatures,
  complete preflight and sole reconstruction ownership, not backend eligibility.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [preflight glossary](../../../../glossary.md#term-preflight); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [backend glossary](../../../../glossary.md#backend))</span>

## Dependencies and integration

- Depends on: approved NN0030 decision; Complete Compiler0012 and Checkpoint0008–0011;
  accepted Model/Planning/Compiler/Prepare/CPU low-level scan join. CPU0036 is accepted baseline,
  not an additional performance gate. Required predecessor reviews/validation are recorded Complete.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Planning glossary](../../../../glossary.md#term-planning); [Compiler glossary](../../../../glossary.md#term-compiler); [Prepare glossary](../../../../glossary.md#prepare); [CPU glossary](../../../../glossary.md#term-cpu); [validation glossary](../../../../glossary.md#term-validation))</span>
- Conflicts with: Checkpoint graph normalization/wire/API, Compiler cold reconstruction, scan
  contracts, NN acquisition/lifecycle and shared-document writers; inactive and serialized.<span data-glossary-navigation="true"> ([Checkpoint glossary](../../../../glossary.md#term-checkpoint); [graph glossary](../../../../glossary.md#graph); [normalization glossary](../../../../glossary.md#term-normalization); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [cold reconstruction glossary](../../../../glossary.md#term-cold-reconstruction); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Parallel group: None; source and documentation writers serialize. Read-only review may overlap tests.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus
  274 accepted dirty paths, including CPU0036 Complete; preserve all unrelated work.
  Baseline: `/private/tmp/synaptik-cp0012-baseline.P4pqr7/accepted` and `snapshot.json`;
  evidence root: `/private/tmp/synaptik-cp0012-evidence.bMKlxo`.
- Integration order: clean implementation → independent review/docs finalization → final affected
  module gates → root attempts with recorded environment exception → main scoped acceptance.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>
- Integration validation: combined root command below, including architecture and Engine consumers;
  no successful standalone filtered main suites first, and no duplicate module runs.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>
- Shared-document integration owner: main; exclusively delegates targeted current-fact/Javadoc
  finalization to the independent reviewer. Main alone owns master/roadmap/result integration.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

## Files and symbols

- [Checkpoint](../../../../glossary.md#term-checkpoint) production directory: `modules/checkpoint/src/main/java/io/github/pho001/synaptik/checkpoint/graph`.
  `GraphSource.java`: [`capture`](../../../../glossary.md#term-capture-meanings) preserves attachments with five-component `CompiledNode`,
  remapping only outer ports.
- Same directory: `GraphArtifactCodec.java` explicit independent wire eligibility; `GraphCheckpoint.java`
  affected facade Javadoc. <span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`GraphValueStorage.java` read-only [ownership](../../../../glossary.md#term-ownership-meanings)/storage reference.
- Same package under `modules/checkpoint/src/test/java`: `GraphCheckpointTest.java`,
  `GraphArtifactCodecTest.java`, `GraphArtifactV3Test.java`; new focused
  `GraphCheckpointGeneralScanTest.java` in this same test package.
- [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) production directory: `extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module`.
  `Model.java`: [`graph`](../../../../glossary.md#graph) before `GraphCheckpoint.copy`;
  `OwnedModelSource.java`: `acquireDefinition`, `acquireComposition` and provenance inventory.
  Minimal early guards only; adoption after copy is too late. Do not broaden fixed-only<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
  `containsScan`/`rejectScanTraining`. `ModelExecutionState.java` is read-only [lifecycle](../../../../glossary.md#lifecycle) reference.
- Same package under `extensions/nn/src/test/java`: `OwnedModelSourceTest.java` and focused existing
  lifecycle/training/runtime-length tests as needed: preserve every affected acquisition route.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime))</span>
- Integration package `io.github.pho001.synaptik.testing.integration` under
  `testing/integration-tests/src/test/java`: `GraphCheckpointEngineIntegrationTest.java`,
  bounded copy/rebind-to-public-Engine regression, not NN workflow qualification.<span data-glossary-navigation="true"> ([rebind glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [Engine glossary](../../../../glossary.md#term-engine); [regression glossary](../../../../glossary.md#term-regression); [qualification glossary](../../../../glossary.md#term-qualification))</span>
- Read-only [Model](../../../../glossary.md#term-model) `CompiledNode`, `CompiledScanBody`, `ScanBody`/`Scan`; [Compiler](../../../../glossary.md#term-compiler)
  `GraphCompilationPort`, `SourceGraphReconstruction` and its focused tests. Do not edit Compiler.

## Acceptance criteria

- Capture once; preserve full descriptors, occurrences, sharing, declared unused inputs/formals,
  repeated actual-role positions, role aliases, canonical final-carry/emission/residual ports and
  root order. Outer NodeId/ValueId normalization never rewrites/resolves local body IDs.<span data-glossary-navigation="true"> ([Capture glossary](../../../../glossary.md#term-capture-meanings); [descriptors glossary](../../../../glossary.md#term-descriptor); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [canonical glossary](../../../../glossary.md#term-canonical-form); [carry glossary](../../../../glossary.md#term-carry); [residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand); [NodeId glossary](../../../../glossary.md#nodeid); [ValueId glossary](../../../../glossary.md#logical-valueid-versus-physical-memory-slot); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- Compiler reconstructs fresh formals/body per occurrence and every canonical port once. Reusing
  an immutable template never merges scopes or retains original body/formals in copied provenance.
  Both traversal flags, T=0 and constituent empty-domain semantics survive; no time unrolling.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [occurrence glossary](../../../../glossary.md#term-occurrence); [immutable glossary](../../../../glossary.md#term-immutable); [provenance glossary](../../../../glossary.md#provenance); [unrolling glossary](../../../../glossary.md#term-unrolling))</span>
- Copy owns detached heap state once per named identity, preserves raw bits/roles/aliases and
  storage-free prototypes, outlives original native state and mutates no caller binding/storage.
  Rebind is metadata-only, retains named state, permits repeated/produced exact-descriptor inputs
  with opaque upstream provenance, and returns immutable ordinary roots under existing guards.<span data-glossary-navigation="true"> ([detached glossary](../../../../glossary.md#detached-model-results); [heap glossary](../../../../glossary.md#term-heap); [identity glossary](../../../../glossary.md#term-identity-meanings); [raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [prototypes glossary](../../../../glossary.md#term-prototype); [binding glossary](../../../../glossary.md#term-binding); [Rebind glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [metadata glossary](../../../../glossary.md#term-metadata); [opaque glossary](../../../../glossary.md#term-opaque))</span>
- Test outer remapping with coincident independent body IDs, multiple carries/emissions/residuals,
  tied roles, multiple occurrences, unused formals, detached ownership and template isolation.
  Include valid low dtype and CPU-unsupported reconstructible bodies (e.g. low TANH/F32 RELU);
  reject unsupported exact body signatures without importing backend predicates or all-PURE support.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [dtype glossary](../../../../glossary.md#data-type--datatype); [CPU glossary](../../../../glossary.md#term-cpu); [TANH glossary](../../../../glossary.md#term-tanh); [F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [signatures glossary](../../../../glossary.md#term-signature); [backend glossary](../../../../glossary.md#backend); [PURE glossary](../../../../glossary.md#term-pure))</span>
- Full unsupported-body preflight precedes host access/copy, including late failures and T=0;
  retain existing source/API, geometry, alias, storage, null/replacement and failure-order guards.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [host glossary](../../../../glossary.md#term-host); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- `GraphArtifactCodec` independently rejects general bodies and malformed attachments before
  host admission/import/copy or file effects. Keep explicit opcodes 1..22 and byte-compatible
  v1/v2/v3 selection/readers. No general-body save/load via widened Compiler admission.<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission))</span>
- Split obsolete `generalScanConsumersRejectMissingNormalizedAttachmentBeforeAnyHostReadOrFileEffect`:
  positive copy/metadata-only rebind tests plus independent SAVE rejection using its real host-read
  spy. Test unreadable/absent/dead state, unchanged existing target, absent target and no staging
  files/host effects; retain positive spy control and malformed/unknown wire rejection coverage.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- NN early GENERAL rejection covers graph overloads, defineTensor/acquireDefinition and
  acquireComposition as applicable, before copy/storage/setup effects using Model-only provenance.
  No general-scan owned Model reaches compute/configure/train; fixed inference/training guard and
  static routes remain unchanged. Test final-state/residual-only roots, heads, T=0 and unreadable state.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [graph glossary](../../../../glossary.md#graph); [Model glossary](../../../../glossary.md#term-model); [owned Model glossary](../../../../glossary.md#model-module-buffer-and-scan-distinctions); [inference glossary](../../../../glossary.md#term-inference); [guard glossary](../../../../glossary.md#term-guard))</span>
- Existing Engine consumers still work; new supported F32/F64 copy/rebind regression matches direct
  ordinary scan results, while low/CPU-unsupported cases remain cold-only rather than execution promises.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [regression glossary](../../../../glossary.md#term-regression))</span>

## [Validation](../../../../glossary.md#term-validation)

Recorded final [module](../../../../glossary.md#term-module) gate after independent [Javadoc](../../../../glossary.md#term-javadoc) finalization; outcomes are in Result:

```bash
./gradlew :extensions:checkpoint:test :extensions:checkpoint:javadoc :extensions:nn:test :extensions:nn:javadoc --max-workers=2 --console=plain
```

Recorded normal root command; a remaining-gate attempt appended `-x :backends:metal:test`.
Both exited 1 on missing native configuration, not CP0012 failures; no third attempt was [run](../../../../glossary.md#term-run).

```bash
./gradlew test javadoc --max-workers=2 --console=plain -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false
```

Required consumer XML, fresh/cached outcomes and root limitations are recorded below.
No timings, [benchmarking](../../../../glossary.md#term-benchmarking), native/[device](../../../../glossary.md#term-device) opt-in or build/[dependency](../../../../glossary.md#term-dependency) edits; future [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) native join stays separate.
Check links/anchors, size, whitespace, baseline preservation and `git diff --check`.

## Documentation and review impact

Independent clean reviewer inspects actual stabilized diff/tests and finalizes affected Javadocs,
`docs/api/graph-checkpoint-api.md`, and targeted root/scoped current-implementation facts and
explanations under main's exclusive delegation. No architecture policy changes/[ADR](../../../../glossary.md#term-architecture-decision-record-adr) are expected;
stop on a new decision. Follow documentation-rules plus general/[API](../../../../glossary.md#term-application-programming-interface-api)/architecture profiles as
applicable; targeted glossary search, no new terminology expected. Source/doc writers serialize;
read-only review may overlap executor tests. Reuse successful tests unless executable code changes.
Completion summary: changes/files, validations, documentation/[Javadoc](../../../../glossary.md#term-javadoc)/glossary, unresolved issues/follow-up;
`Status: Complete` or `Status: Incomplete` with specific follow-up.

## Result

Status: Complete — implementation/scoped acceptance only; unrestricted repository root is NOT green.

- Delivered attachment-preserving outer normalization, Compiler-owned cold reconstruction,
  independent closed v1/v2/v3 wire admission and NN early GENERAL rejection; no new API/type/edge.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [Compiler glossary](../../../../glossary.md#term-compiler); [cold reconstruction glossary](../../../../glossary.md#term-cold-reconstruction); [admission glossary](../../../../glossary.md#term-admission); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Changed 20 task paths: [checkpoint](../../../../glossary.md#term-checkpoint).[graph](../../../../glossary.md#graph) `GraphSource`, `GraphArtifactCodec`, `GraphCheckpoint`,
  `GraphCheckpointTest`, new `GraphCheckpointGeneralScanTest`; nn.[module](../../../../glossary.md#term-module) [`Model`](../../../../glossary.md#term-model), `OwnedModelSource`,
  `OwnedModelSourceTest`; `GraphCheckpointEngineIntegrationTest`; `GraphCompilationPort` Javadoc;
  root, graph-checkpoint API, compiler-autograd/extensions-training contracts, module-boundaries,
  general-differentiable-scan explanation, glossary; this brief, Checkpoint master and roadmap.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [autograd glossary](../../../../glossary.md#autograd); [differentiable glossary](../../../../glossary.md#derivative-and-chain-rule))</span>
- Independent clean code/docs review Complete, pass/no findings (2026-10-23 11:29:53 UTC).
  Seven targeted documents and affected Javadocs finalized; existing glossary distinctions clarified,
  no new terminology/policy/ADR. Compiler executable inventory/logic and CPU0036 stayed unchanged.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- Final module command above: 52 Checkpoint + 526 NN = 578 fresh tests, zero failures/errors/skips;
  both fresh Javadocs had zero warnings. Development evidence reused; no duplicate filtered reruns.
- Normal root command above: exit 1, 5m8s, only four unchanged `MetalExpNativeTest`/
  `MetalSigmoidNativeTest` cases failed at `Path.of(null)` with unset `SYNAPTIK_METAL_TEST_LIBRARY`,
  before library opening. Fresh CPU: 1214 reported, 1176 executed, 38 skipped, zero failures.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- Same root command plus `-x :backends:metal:test`: exit 1, 19s; 15 unchanged cases in
  `EngineScalarMetalIntegrationTest`, `EngineExpMetalIntegrationTest`, `EngineSigmoidMetalIntegrationTest`
  failed on the same missing environment before native opening. Normal unit tasks, architecture
  and conformance passed/reused; all Javadocs passed/reused, with two benchmark incubator warnings.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance))</span>
- Fresh required XML: `CheckpointDependencyContractTest` 1 pass; `GraphCheckpointEngineIntegrationTest`
  7 passes; architecture 25 and conformance 59 passes, no failures/skips in those suites.
- Current reports: 4762 reported = 4507 passed + 236 skipped + 19 environment failures, zero errors;
  not all fresh. Root reused module/CPU successes; 13 stale pre-relocation reports excluded.
  No third root, benchmarks, native/device qualification, unrelated source fixes or build changes.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device); [qualification glossary](../../../../glossary.md#term-qualification))</span>
- Evidence: `/private/tmp/synaptik-cp0012-evidence.bMKlxo/{final-validation.json,root-test-results.json}`;
  exact commands/tasks/logs and fresh/cached breakdown retained there. Source freeze unchanged;
  274-path baseline preserved outside 20 authorized task paths, HEAD=origin at base, index empty.<span data-glossary-navigation="true"> ([logs glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>
- No unresolved CP0012 acceptance issue. Separate follow-up: native test harness configuration/gating
  for the 19 unrelated failures; no fix/qualification here. NN workflow/native join remains separate,
  body persistence stays closed, Checkpoint/NN/Compiler frontiers None; no successor task or launch.<span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence))</span>
