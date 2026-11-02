# Task Compiler0015: Symbolic-length Scan inference, first-order AD and scoped binding

## Status

Complete — MAIN accepted the exact Compiler-only scope on 2026-10-26; all required gates passed.

## Change class

Class C — scoped validation, automatic differentiation (AD) and binding cross a durable Compiler/Scan
contract. Clean implementation and independent source/docs review are mandatory within this task.

## Goal

Combine one named batch dimension B with discrete runtime INT64[B] lengths for general Scan:
fixed time T/features, homogeneous FLOAT32/FLOAT64 inference, actual first-order opposite-traversal
sibling AD and complete fresh scoped all-port BatchBindingPlan admission, without execution support.

## Scope

Extend existing inference/row proofs and sibling construction; binder edits require a demonstrated gap.
Preserve static length/all-valid support and APIs/header/body/packages; added/moved production types: None.
The [proposal](../../../length-aware-general-scan.md#proposed-symbolic-b-plus-lengths-integration) is context.
Size exception above 15 KB: keep this capability's inference, actual AD, binding and evidence atomic.

## Non-goals

No Model executable/semantic change, new API/type/protocol, Runtime/Prepare/CPU/Engine enablement,
geometry memo, resource/history allocation, cold capture/reconstruction/copy, persistence/v5,
NN/Training/fluent migration, fixed-scan BPTT, variable T/features, low/mixed types, higher-order,
native/backend/emitter changes or performance campaign. Do not copy legacy implementation.

## Contracts

- Root: [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts),
  [Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle), [Core invariants](../../../../../ARCHITECTURE.md#core-invariants)
  and [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules).
- Compiler owner: [Accepted symbolic-batch metadata binding](../../../../architecture/contracts/compiler-autograd.md#accepted-symbolic-batch-metadata-binding),
  [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation),
  [`modules/compiler`](../../../../architecture/contracts/compiler-autograd.md#modulescompiler) and
  [Compile lifecycle](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle), including
  `Bounded general-scan cold reconstruction` (still closed for length mode).
- Sole Scan semantic owner: [General differentiable scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan),
  specifically `Source body, attachments and scoped identities`, `Canonical residual ports and
  demand`, `Reverse scan and exact primal bindings` (independent capture/saved-site obligations).
- Symbolic intersection: [Accepted symbolic-batch scan](../../../../architecture/contracts/recurrent-scan.md#accepted-symbolic-batch-scan),
  [Empty domains and numeric resources](../../../../architecture/contracts/recurrent-scan.md#empty-domains-and-numeric-resources),
  [Qualification and mechanism gates](../../../../architecture/contracts/recurrent-scan.md#qualification-and-mechanism-gates),
  including `Symbolic templates, roles and ports` and `Actual first-order closure and arithmetic`.
- Length intersection: [Length-aware whole-batch scan](../../../../architecture/contracts/recurrent-scan.md#length-aware-whole-batch-scan),
  [Header, bindings and source admission](../../../../architecture/contracts/recurrent-scan.md#header-bindings-and-source-admission),
  [Bounded row-role analysis](../../../../architecture/contracts/recurrent-scan.md#bounded-row-role-analysis),
  [First-order length-mode reverse](../../../../architecture/contracts/recurrent-scan.md#first-order-length-mode-reverse),
  [Runtime preflight and implementation gates](../../../../architecture/contracts/recurrent-scan.md#runtime-preflight-and-implementation-gates),
  including `Activity, raw identity and retained histories`.
- Required executor reading is root + Compiler/Scan headings above. Unchanged downstream contracts
  are read-only gate references, not default reading: consult only focused rejection clauses for a
  concrete consumer issue. No downstream production/contract edits; MAIN owns the three guard tests.

### Accepted qualification and Ready receipt

MAIN accepted the bounded support gate before Java on 2026-10-26 after Meitner pre-Java APPROVE.
Header/row/activity/AD rules, algorithms, APIs/dependencies stay unchanged; ADR0049/0051 suffice.

## Dependencies and integration

- Depends on: Complete Model0041/0044, Compiler0011/0013/0014, Planning0008 and the retained
  Prepare0013 ordinary binding baseline; frozen pre-Java Meitner APPROVE and MAIN acceptance satisfied.
- Conflicts with: all writers to Scan semantics/status, Compiler inference/AD/binding/capture,
  Model Scan Javadoc, or shared planning/docs; none active under MAIN's sole-serial-writer handoff.
- Parallel group: None; serial contexts, no concurrent write worktree required.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus ALL
  accepted dirty work, not merged predecessors; preserve the baseline described below.
- Integration order: frozen pre-Java APPROVE/MAIN acceptance and Ready → clean Compiler implementation →
  independent Class C source/docs APPROVE → named compatibility checkpoint → MAIN acceptance.
- Integration validation: MAIN's single **Compiler0015 closed-consumer compatibility checkpoint** below.
- Shared-document integration owner: MAIN; sole serial owner of authority/explanations/ADR/glossary,
  this brief, Compiler master-plan and roadmap, with explicit final-doc delegation if needed.

MAIN's 2026-10-26 Ready receipt verified seven Complete predecessors, base/index and inactive conflicts;
evidence: `/private/tmp/synaptik-compiler0015-main.TDKRDV/main-planning-verification.txt`. Worker now closed.
Protect accepted dirty work against `/private/tmp/synaptik-dynamic-batch-cutover.Of4Q8D/compiler0015-baseline.json`
and its 2480-path copies, not HEAD-only diffs; independent review uses the source freeze.

## Files and symbols

Production prefix: `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/`.

- `ScanBodyInference.java`: `infer`, `validateRows`, `rowResult`, `checkRow`, fixed-feature checks;
  replace static numeric row comparisons with independently proved Dimension provenance.
- `ScanGradientRules.java`: `preflight`, `sourceRoles`, `rowRole`, `apply`, `stackedShape` and
  existing row-constant/zero closures; remove only the admitted symbolic-length restriction.
- `BatchBindingPlan.java`: `prepare`, `indexValidity`, `bind`, `Binding.descriptor`; first test
  the generic implementation. No production edit is required if all scoped/all-port evidence passes.
- Existing seams only for demonstrated gaps: `AutogradPreflight.java` (`preflightBody`, whole-original
  preflight), `FirstOrderAutograd.java` (`expandBody`, `DerivativeConstants`),
  `CapturedGraphInference.java` (`inferAndValidate`), `GraphCapture.java` (`captureBody`),
  `GraphCompiler.java` (`compile`), `ScanBodyInference.refreshDemand`; no second rule algebra.
- Tests in the matching Compiler test package: extend `LengthAwareScanBodyInferenceTest`,
  `LengthAwareScanGradientRulesTest`, `SymbolicScanInferenceTest`, `SymbolicScanGradientRulesTest`,
  `SymbolicScanCompilerTest`, `GeneralScanCompilerTest`, `BatchBindingPlanTest` as relevant.
  Extend existing `ScanGradientRulesTest.Oracle` for test-local length evaluation; no new framework.
  Test-only `SymbolicLengthScanCompilerTest` in that package is allowed if clearer than overloading
  existing symbolic tests; no new production type or test framework.
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Scan.java`: final docs-context
  Javadoc-only status correction at type line 46/overload lines 145–147; no executable/semantic/API change.

## Acceptance criteria

- Both dtypes/traversals independently infer combined scans, including broader forward mixed
  ROW(0)/rank-two ROW(1)/SHARED sequences and terminal SHARED carries. Final non-gradient INT64[B]
  lengths has no formal/target/cotangent: checked counts c+s+i+1 actuals, c+s+i formals.
  Preserve positional ties and conservative outer gradient OR.
- Compare Dimensions by canonical value/name, including distinct objects with the same non-`B` name.
  Only row axes contain it, exactly once. Reject foreign/repeated/derived/anonymous symbols, variable
  T/features, sampled layouts, unproved transforms and malformed unused declarations/attributes/siblings.
- Prove the full bounded row algebra: right-aligned singleton SHARED broadcasting; transpose and
  allowed MATMUL roles; feature SUM/sumToShape restoration; explicit row SUM stays terminal even
  at B=1; equal-axis sumToShape preserves ROW. No terminal feedback to rows/emissions, SHARED-to-ROW
  invention or shared-carry leakage. Each suffix carry is its own distinct exact old.add(contribution).
- Before derivative Tensor IDs, whole-original preflight retains all-carry/all-original-sequence
  ROW(0) restriction, even unused/non-gradient/unrelated branches and T=0. Preserve residual-root,
  unsupported-rule, fixed-scan, generated-shared-carry differentiation and higher-order/createGraph
  rejection. Broader forward admission does not relax those guards. Preserve Model validation
  before canonical output IDs; do not promise ID rollback for later capture/inference failures.
- Actual opposite-traversal siblings cover final-only/emission-only/joint seeds, two carries,
  false-flag flow, disconnected zeros, tied roles/roots and cross-occurrence sums. Load exact saved
  TANH/SIGMOID, never replay; normalize each SHARED-target term before accumulation. No B×parameter banks.
- Generated seeds/histories/constants retain explicit mixed roles, including ROW(1) literal/zero
  sequences [T,F,B], shared transpose histories [T,F,H], row histories [T,B,H], and pruned axes in
  lockstep with used sequences. Capture/rewrites/demand/final queries preserve scopes and saved sites;
  neither coincident local IDs nor equal numeric F=B/H=B nor B=0/1 may infer/erase role identity.
- One BatchBindingPlan fresh-binds B=0,1,F,H,ordinary and 32→32→7→1→32 with fresh actual descriptors,
  including changed admitted isView at repeated B. Check unused/tied actuals including L, scoped formal/value/
  typed obligation, both siblings and all latent final/emission/residual/gradient ports. Existing lazy
  outer indexing suffices; no public local-descriptor API. Ports become canonical non-view; templates
  stay unresolved. Preserve current input-backed descriptors/isView and earlier Bindings; reject bad
  metadata before returning Binding. No memo substitutes for complete fresh scoped admission.
  Changed storage/L payloads at repeated B are numerical ORACLE evidence; bind sees descriptors only.
- Check fixed-feature products independently of empty T/B, including unused/dead local values,
  ports and shape attributes; preserve legitimate zero-feature domains. Raw IR/descriptor paths
  must independently reject invalid metadata, not rely solely on Model constructors or incidental
  Optional/index failures. Same-scope obligations cannot disappear in equal-ID neighboring scopes.
- Evaluate the actual generated Tensor graph/sibling with the existing test-local helper against
  independent nonuniform active-prefix ordinary unrolling, not handwritten reverse. Cover zero/full/
  mixed/changed L, B=0/1/F/H/ordinary, separate T=0 and T=0/B=0; raw signed-zero/NaN carry/seed
  identity, inactive +0, poisoned padding/seeds, saved sites and contribution-then-distinct-time-ADD.
  Use explicit dtype test oracles/tolerances, not cross-grouping bit equality. Outer MEAN counts B×H,
  not L/prototype B: B=0 is NaN with empty row gradients/nonempty typed-zero parameter sums.
  All-zero L performs no body work.
- Production compile/bind reads no input/L values, allocates no history/resources and runs no graph.
  Only the test oracle checks L values, including invalid-last/T=0. Direct L and existing floating-to-
  INT64 producer graphs are separate metadata witnesses, not CPU grants. Keep downstream/cold/wire/
  NN/Training gates and static length/all-valid compatibility under disabled/standard optimization.
  Body/capture size stays O(body), not T-fold unrolling.

## Validation

Worker: focused checks during development; once after executable freeze:

```bash
./gradlew :modules:compiler:test :modules:compiler:javadoc
```

MAIN alone owns this one metadata-guard compatibility checkpoint after Compiler admission:

```bash
./gradlew :modules:prepare:test --tests '*PrepareContextTest.directSymbolicLengthProjectionAndEveryIsolatedSymbolRejectDeliberatelyBeforeExtents' \
  :backends:cpu:test --tests '*CpuGeneralScanCapabilityTest.providerDeliberatelyRejectsRawSymbolicLengthQueriesIncludingZeroTime' \
  :extensions:checkpoint:test --tests '*GraphCheckpointGeneralScanTest.symbolicLengthSourceDoesNotOpenCopyRebindOrSave'
```

No root run by default: no dependency/build/runtime/backend behavior change. No emitter/native/
performance runs. Review reuses successful tests unless executable changes/evidence gaps require
a rerun. Generate final Compiler Javadoc after final edits; allowed Scan comment edits require Model
Javadoc once, not Model tests. Validate links/anchors/terms/whitespace and `git diff --check`;
compare final ownership to the dirty baseline, not HEAD alone.

## Documentation and review impact

Independent review reconciles 13 delegated Markdown paths, Model Scan status comments and helper Javadoc;
production worker Javadocs stay unchanged. General/architecture/planning/API/example profiles apply; no new ADR.

## Result

- Implemented named-B lengths inference/actual first-order AD/fresh scoped all-port binding. Binder unchanged.
- Worker changed two production and seven test Java paths (one new); exact paths/diff and detailed
  coverage: `/private/tmp/synaptik-compiler0015-main.TDKRDV/worker-handoff.md` and `worker-baseline-relative.patch`.
- Reused `./gradlew :modules:compiler:test :modules:compiler:javadoc`: PASS, 399 tests/53 suites,
  zero failures/errors/skips; 5s, 14 tasks (4 executed/10 up-to-date). Retained XML independently counted.
- Earlier sandbox lock denial ran no tests; an earlier noIds fixture allocated badLengths inside
  its measured lambda. Moving fixture allocation outside fixed the test; final run supersedes it.
- Independent Meitner source/docs review: APPROVE, 0 open P1/P2. Actual siblings match independent
  active-prefix ordinary AD, including ROW(1), exact saved overrides, ties/false flags and zeros.
- Documentation/Javadoc/glossary status and scope/hash/link/whitespace/brief checks PASS; no semantic/API change.
- MAIN's exact three-consumer command above: PASS, exactly 3 tests (1 Prepare/1 CPU/1 Checkpoint),
  zero failures/errors/skips; 2s, 27 tasks (6 executed/21 up-to-date). Evidence: `main-compatibility-counts.json`.
- Final `./gradlew :modules:compiler:javadoc :modules:model:javadoc`: PASS, both regenerated after comments;
  2s, 8 tasks (2 executed/6 up-to-date). Logs: `main-compatibility.log`, `main-final-javadoc.log` in the evidence directory.
- MAIN accepted 2026-10-26; no unresolved issues or follow-up. Source freeze/dirty base remain preserved.
- No Compiler-test or review rerun, root/native/performance run, Git mutation, commit or push.
- Execution/cold/copy/wire/v5/NN/Training/higher-order/memo gates remain closed. No successor launched.
- Status: Complete
