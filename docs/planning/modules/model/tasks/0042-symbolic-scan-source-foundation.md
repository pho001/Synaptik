# Task Model0042: Static and symbolic Scan source foundation

## Status

Complete — Model source, consumer guards, documentation and source-matched validation accepted.
PlanningAuthorized=true; ImplementationAuthorized=true. Implementation frontier: None.
Authority frontier: None. No Compiler symbolic inference/AD or executable Scan support is authorized.

## Change class

Class C: supported source-shape invariants, scoped Scan metadata, canonical output allocation and
the unchanged fail-closed handoff to Compiler. Use a clean implementation and independent review.

## Goal and scope

Implement the accepted Model portion of [symbolic Scan](../../../../architecture/contracts/recurrent-scan.md#accepted-symbolic-batch-scan)
through existing `ScanBody` and all-valid `Scan.apply`. Preserve fully static construction as an
equal first-class path; never silently convert static shapes into symbolic B or change its header,
descriptors, identities or execution route. User explicitly requested both static and symbolic forms.

Allow fixed extents or one consistently named batch dimension per value, with fixed sequence time
T and fixed remaining axes. Examples: static X `[10,32,16]`, h `[32,64]`; symbolic X `[10,B,16]`,
h `[B,64]`. Keep public signatures and source/structural IR inventory unchanged. Symbolic formals,
actual source bindings and ports preserve exact named shapes with unresolved numeric layouts;
fixed/shared ports retain their static canonical geometry. Source construction does not bind B.

## Non-goals

No Compiler/Prepare/CPU/Engine production changes, symbolic execution/AD/cold/persistence,
length-mode symbolic combination, new API/type/session/graph, dynamic T/features, numerics,
dependency/build/native changes, benchmarks, commits or push. No body whitelist copied from CPU.
No rewrite of completed briefs or unrelated accepted dirty work. Report any required consumer
production guard fix to MAIN before widening the explicit scope.

## Contracts

Read root [Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle),
[Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
[Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts),
[Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules) and
[Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
Owning Scan headings: [Source body, attachments and scoped identities](../../../../architecture/contracts/recurrent-scan.md#source-body-attachments-and-scoped-identities),
[Canonical residual ports and demand](../../../../architecture/contracts/recurrent-scan.md#canonical-residual-ports-and-demand),
[Traversal, empty domains and declared qualification](../../../../architecture/contracts/recurrent-scan.md#traversal-empty-domains-and-declared-qualification),
[Accepted symbolic-batch scan](../../../../architecture/contracts/recurrent-scan.md#accepted-symbolic-batch-scan)
and [Header, bindings and source admission](../../../../architecture/contracts/recurrent-scan.md#header-bindings-and-source-admission)
for retained static length-mode rejection.
For consumer tests only, read Compiler [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation),
[Accepted symbolic-batch metadata binding](../../../../architecture/contracts/compiler-autograd.md#accepted-symbolic-batch-metadata-binding)
and [Bounded general-scan cold reconstruction](../../../../architecture/contracts/compiler-autograd.md#bounded-general-scan-cold-reconstruction).
These contracts are already accepted. Update current support status, not their semantic decision;
stop on a missing/ambiguous contract or a necessary new production type/signature.

## Dependencies and integration

- Depends on: Complete Model0041 authority/review; retained Complete Model0038 and Compiler0013;
  existing static Scan/source/dtype/read/length contracts and tests.
- Conflicts with: Scan source/API/descriptor/attachment and scope/AD/cold admission semantics;
  all permitted write paths below. No competing writer; serialize these concerns.
- Parallel group: None.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus accepted
  uncommitted baseline, including Complete Model0041; index empty/unchanged.
- Integration order: sole clean writer → frozen independent code/Javadoc/docs review → MAIN's
  single configured source-matched repository test/Javadoc gate → final metadata acceptance.
- Integration validation: root command below, retained native hash, documentation/status/link and
  permitted-path baseline audit; reuse successful worker suites through Gradle up-to-date evidence.
- Shared-document integration owner: MAIN; the sixteen permitted paths are exclusively delegated
  to this writer until freeze. Reviewer is read-only. No other task is launched.

Admission: MAIN verified predecessor Status/Result Complete, accepted authority and unchanged
baseline on 2026-10-25, unused Model0042 ID and inactive conflicts before launch. User “ok, pust se
do prace” authorizes this exact source task; the added static-dimension request is included above.
Snapshot: `/private/tmp/synaptik-dynamic-batch-cutover.Of4Q8D/pre-model-symbolic-source.json`.

## Files and symbols

Production writes (existing `io.github.pho001.synaptik.model.tensor`, no new/moved types):

- `modules/model/src/main/java/.../model/tensor/ScanBody.java` — static/bounded-symbolic formals,
  complete contained/sibling shape validation and unchanged ordinary family semantic checks.
- `modules/model/src/main/java/.../model/tensor/Scan.java` — `validateBindings`, `validatePort`,
  `portDescriptor` and both construction/attachment seams; retain static length-mode boundaries.

Test writes: existing Model `ScanBodyTest.java`, `ScanExpressionTest.java`; new Model
`model/tensor/ScanSymbolicBatchSourceTest.java`; existing Compiler `GeneralScanCompilerTest.java`.
Resolve the paths under the normal `src/test/java/io/github/pho001/synaptik/` trees.
Other production owners are read-only: DynamicDimension/Shape/TensorDescriptor, TensorProducer,
CompiledScanBody, ScanAttrs, ScanBodyInference/ScanGradientRules/GraphCompilationPort.

Documentation/planning writes: `ARCHITECTURE.md` and owning `recurrent-scan.md` for current-status
clarification only; `docs/api/tensor-api.md` general-scan construction; focused
`docs/architecture/general-differentiable-scan.md`; targeted `docs/glossary.md`; this brief;
Model master plan/proposal, `docs/planning/dynamic-batch-execution.md`, `docs/planning/roadmap.md`.
These are sixteen permitted paths, not a requirement to manufacture unnecessary edits.

## Acceptance criteria

- Static and symbolic variants use the same existing API. Static layouts/header/flags/ports remain
  unchanged, with tests for T=0, concrete B=0/1 and ordinary B; no implicit symbolization or fake B.
- Derive one canonical named symbol from declared formals, including unused positions; compare
  DynamicDimension by its canonical name/value, not object identity or a hard-coded name "B".
  Reject multiple symbols, anonymous/derived unsupported dimensions, repeated B axes, output-only
  symbols and unproved B transformations across the complete body, including unused siblings.
- Keep known PURE operation guards/applicable Model dtypes independent of AD/backend qualification;
  include symbolic F16 TANH/F32 RELU examples and retained static F32/F64 source, not a CPU whitelist.
- Formal leaves remain identity-distinct, storage/provenance-free; actuals remain disjoint and match
  exact source shapes/dtypes/gradient flags. Static/symbolic mismatches reject; no runtime binding at
  source construction. Sequence axis zero is a common static T, never a symbolic time extent.
- Symbolic descriptors have absent numeric layouts; static/shared ports have canonical layouts.
  Final/emission/residual port order, independent wrappers, repeated roles, deterministic retained
  inventory and conservative differentiable gradient OR remain correct, including unused inputs/T=0.
- Validate all canonical siblings and direct attachment descriptors before output-ID suppliers;
  preserve no-ID validation, malformed/host-backed/free/effectful/nested-body rejection and storage
  rechecks. Do not construct a second graph or synthetic Tensor merely to validate metadata.
- Length overload and direct length attachment explicitly reject all symbolic body/actual geometry
  with a deliberate contract failure before IDs, not accidental `staticSize().orElseThrow` failure.
  Preserve all current static length semantics, roles, masks and no-value-read behavior.
- Public Compiler forward/backward/cold entry points still reject unsupported symbolic Scan;
  test standard/disabled optimization and pre-derivative-ID rejection where applicable. Internal
  structural capture alone is not executable/inference/cold qualification. No consumer enablement.
- Javadoc documents new shape/layout/failure behavior and all affected inputs/results. API/explanation
  contain a complete construction-only static/symbolic example; explicitly distinguish supported
  source from still-static Compiler/Prepare/CPU/cold/NN/training paths and future memoization.
- Independent review has zero open blocking findings; worker evidence and MAIN's final integration
  pass match the frozen source, and all unrelated files/HEAD/origin/index remain unchanged.

## Validation

Worker focused tests during development, then once after final source stabilization:

```bash
./gradlew --no-daemon :modules:model:test :modules:model:javadoc :modules:compiler:test
```

MAIN, after frozen review (no concurrent Gradle/source writer):

```bash
SYNAPTIK_METAL_TEST_LIBRARY=/Users/phujka/IdeaProjects/Synaptik/modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew --no-daemon -Dsynaptik.cpu.generalScan.performance=false test javadoc
```

Use the existing native library, no native build; SHA-256 before/after:
`03cba3ebfa0bbb78c1bff369ff3451c591846fd975c0034fa7f67fdd2209985a`.
All performance flags stay default-off/explicit false. Record actual source-matched test XML counts,
failures/skips and cached task evidence; exclude stale build directories of relocated modules.
No generated-code timing campaign: emitters/hot paths are unchanged. Final docs/Javadocs after edits,
local links/anchors, `git diff --check`, brief <=200 lines/15 KB and permitted-scope hash audit.
MAIN supplies a read-only scope/document validator while the writer implements.

## Documentation and review impact

Supported source behavior/Javadocs and status wording change: independent clean targeted Class C
code/documentation review is mandatory. Apply general, API/Javadoc, architecture, planning and
conceptual-example profiles; targeted glossary review only. Preserve old semantic authority and
completed briefs; no new ADR or numerical/extension contract is selected.

## Result

MAIN accepted 2026-10-25 after independent APPROVE, zero open P1/P2; the P2 family-proof gap is resolved.
No downstream enablement. The correction changes only ScanBody.java, ScanSymbolicBatchSourceTest.java,
Tensor API, focused explanation and this live brief relative to the first frozen handoff.

- Production: only existing ScanBody.java and Scan.java; named-axis/source-family proofs, exact
  binding, canonical static/unresolved symbolic ports, complete sibling and explicit static length guards.
- Tests: ScanBodyTest updated; new ScanSymbolicBatchSourceTest; GeneralScanCompilerTest expanded.
  Existing ScanExpressionTest unchanged and reused. Static F32/F64 T=0, B=0/1/32, symbolic low TANH/
  F32 RELU, retained/sibling/role/gradient/no-ID/storage checks pass. Public forward/backward/cold
  reject symbolic Scan, including B only in unused declarations with entirely static public ports,
  standard/disabled optimization and unrelated-original-branch preflight. No consumer production fix required.
- Documentation: affected Javadocs, root/owning-contract current status, Tensor API, focused explanation,
  glossary, Model master/proposal, dynamic-batch coordination and roadmap synchronized; semantics unchanged.
- P2 correction: exact attention batch/embedding/sequence/mask proofs, normalized-shape attribute/suffix
  validation, Conv2d/3d grouped-channel/bias, every BatchNorm parameter and dense/index loss equalities.
  All eight dispatched shape-attribute forms are checked before the static fast path. Bounded sweep
  reuses strict gather/scatter/fold/composition/broadcast guards; no general numeric-domain prover.
  Six added tests include no-ID negatives, low-type symbolic positives and retained ordinary guards.
- Focused development command `./gradlew --no-daemon :modules:model:test --tests '*Scan*Test'
  :modules:compiler:test --tests '*GeneralScanCompilerTest'` passed after stabilization.
- Corrected-source final worker command above PASS: 16 tasks, 3 executed, 13 up-to-date;
  Model 147 suites/1,193 tests; Compiler 49 suites/366 tests; zero failures/errors/skips. Javadoc PASS.
- Both complete documentation programs compiled with javac and ran against final Model classes;
  exact printed static/symbolic shapes/layout presence matched the text. Native SHA-256 unchanged.
- Evidence: `/private/tmp/synaptik-model0042-p2-evidence.sId9ip/worker-final.log`, focused.log,
  worker-test-counts.json, source-match.json, family-sweep.json and scope-review.json; git diff --check PASS.
  Prior freeze evidence remains under `/private/tmp/synaptik-model0042-evidence.yUK8KV/`, superseded.
- Frozen ScanBody SHA-256 `2c64e4448af96a684c8248f29710e20021cc0ca1322cb59ff06854f293e9c6a0`;
  Scan SHA-256 `99e18905d231a2e88addf849873721747ea2f756b81bcb8d7512b74c74fd6509`.
- MAIN's configured root command above PASS in 5m45s: 119 tasks, 28 executed, 91 up-to-date;
  695 suites/4,902 tests, zero failures/errors, 51 skips (39 CPU opt-ins, one Engine timing,
  ten Metal observer opt-ins, one host-fixture integration assumption). Javadoc PASS with existing warnings.
  Evidence: `/private/tmp/synaptik-model-source.tlvy79/root-gate.log` and `root-test-results.json`.
- No new production types or public signatures; no native build, benchmark, stage/commit/push.
  Source matched frozen v2 through the root gate; native hash, HEAD/origin/index and outside scope unchanged.

Status: Complete
Follow-up: separately admit Compiler symbolic inference/actual AD/all-port binding; no automatic successor.
