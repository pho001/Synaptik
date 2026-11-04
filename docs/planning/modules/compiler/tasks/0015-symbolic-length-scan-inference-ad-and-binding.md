# Task Compiler0015: Symbolic-length [Scan](../../../../glossary.md#term-cumulative-scan) [inference](../../../../glossary.md#term-inference), first-order [AD](../../../../glossary.md#autograd) and scoped [binding](../../../../glossary.md#term-binding)

> **Editorial reading key — not part of the historical record.**
>
> There are two independent inputs to this admission: a named batch dimension in descriptions and
> integer lengths supplied as values only at execution. Compiler and BatchBindingPlan inspect
> metadata, so binding the same batch size again does not validate new lengths payloads. Explicit
> ROW/SHARED roles survive reverse construction even when batch and feature extents coincide. The
> test-local numerical oracle below exercises generated expressions; it is not a production
> executor or a widened downstream gate.
>
> Current reading: [scoped Scan capture, roles and first-order differentiation](../../../../reference/compiler/verification-scan.md#current-scoped-scan-capture-and-first-order-ad).

## Status

Complete — MAIN accepted the exact [Compiler](../../../../glossary.md#term-compiler)-only [scope](../../../../glossary.md#term-scope) on 2026-10-26; all required gates passed.

## Change class

Class C — scoped [validation](../../../../glossary.md#term-validation), [automatic differentiation](../../../../glossary.md#autograd) ([AD](../../../../glossary.md#autograd)) and [binding](../../../../glossary.md#term-binding) cross a durable [Compiler](../../../../glossary.md#term-compiler)/[Scan](../../../../glossary.md#term-cumulative-scan)
contract. Clean implementation and independent source/docs review are mandatory within this task.

## Goal

Combine one named [batch dimension](../../../../glossary.md#batch-dimension) B with discrete [runtime](../../../../glossary.md#term-runtime) [INT64](../../../../glossary.md#data-type--datatype)[B] lengths for [general Scan](../../../../glossary.md#general-differentiable-scan):
fixed time T/[features](../../../../glossary.md#term-software-feature), [homogeneous](../../../../glossary.md#term-homogeneous) [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) [inference](../../../../glossary.md#term-inference), actual first-order opposite-traversal
sibling [AD](../../../../glossary.md#autograd) and complete fresh scoped all-port BatchBindingPlan [admission](../../../../glossary.md#term-admission), without execution support.

## [Scope](../../../../glossary.md#term-scope)

Extend existing [inference](../../../../glossary.md#term-inference)/row proofs and sibling construction; binder edits require a demonstrated gap.
Preserve static length/all-valid support and APIs/header/body/packages; added/moved production types: None.
The [proposal](../../../length-aware-general-scan.md#proposed-symbolic-b-plus-lengths-integration) is context.
Size exception above 15 KB: keep this capability's inference, actual [AD](../../../../glossary.md#autograd), [binding](../../../../glossary.md#term-binding) and evidence [atomic](../../../../glossary.md#resource-transaction-and-rollback).

## Non-goals

No [Model](../../../../glossary.md#term-model) executable/semantic change, new [API](../../../../glossary.md#term-application-programming-interface-api)/type/protocol, [Runtime](../../../../glossary.md#term-runtime)/[Prepare](../../../../glossary.md#prepare)/[CPU](../../../../glossary.md#term-cpu)/[Engine](../../../../glossary.md#term-engine) enablement,
geometry [memo](../../../../glossary.md#cache-and-memoization), resource/history allocation, cold [capture](../../../../glossary.md#term-capture-meanings)/reconstruction/copy, [persistence](../../../../glossary.md#term-persistence)/v5,
[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training/fluent migration, fixed-[scan](../../../../glossary.md#term-cumulative-scan) [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt), variable T/[features](../../../../glossary.md#term-feature), low/mixed types, higher-order,
native/[backend](../../../../glossary.md#backend)/emitter changes or performance campaign. Do not copy legacy implementation.

## Contracts

- Root: [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [normative glossary](../../../../glossary.md#term-normative))</span>,
  [Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>, [Core invariants](../../../../../ARCHITECTURE.md#core-invariants)
  and [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>.
- [Compiler](../../../../glossary.md#term-compiler) [owner](../../../../glossary.md#term-owner): [Accepted symbolic-batch metadata binding](../../../../architecture/contracts/compiler-autograd.md#accepted-symbolic-batch-metadata-binding)<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [metadata glossary](../../../../glossary.md#term-metadata); [binding glossary](../../../../glossary.md#term-binding))</span>,
  [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)<span data-glossary-navigation="true"> ([automatic differentiation glossary](../../../../glossary.md#autograd))</span>,
  [`modules/compiler`](../../../../architecture/contracts/compiler-autograd.md#modulescompiler) and
  [Compile lifecycle](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>, including
  `Bounded general-scan cold reconstruction` (still closed for length mode).
- Sole [Scan](../../../../glossary.md#term-cumulative-scan) semantic owner: [General differentiable scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan)<span data-glossary-navigation="true"> ([General differentiable scan glossary](../../../../glossary.md#general-differentiable-scan))</span>,
  specifically `Source body, attachments and scoped identities`, `Canonical residual ports and
  demand`, `Reverse scan and exact primal bindings` (independent [capture](../../../../glossary.md#term-capture-meanings)/saved-site obligations).
- Symbolic intersection: [Accepted symbolic-batch scan](../../../../architecture/contracts/recurrent-scan.md#accepted-symbolic-batch-scan),
  [Empty domains and numeric resources](../../../../architecture/contracts/recurrent-scan.md#empty-domains-and-numeric-resources),
  [Qualification and mechanism gates](../../../../architecture/contracts/recurrent-scan.md#qualification-and-mechanism-gates)<span data-glossary-navigation="true"> ([Qualification glossary](../../../../glossary.md#term-qualification))</span>,
  including `Symbolic templates, roles and ports` and `Actual first-order closure and arithmetic`.
- Length intersection: [Length-aware whole-batch scan](../../../../architecture/contracts/recurrent-scan.md#length-aware-whole-batch-scan),
  [Header, bindings and source admission](../../../../architecture/contracts/recurrent-scan.md#header-bindings-and-source-admission)<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission))</span>,
  [Bounded row-role analysis](../../../../architecture/contracts/recurrent-scan.md#bounded-row-role-analysis),
  [First-order length-mode reverse](../../../../architecture/contracts/recurrent-scan.md#first-order-length-mode-reverse),
  [Runtime preflight and implementation gates](../../../../architecture/contracts/recurrent-scan.md#runtime-preflight-and-implementation-gates)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [preflight glossary](../../../../glossary.md#term-preflight))</span>,
  including `Activity, raw identity and retained histories`.
- Required executor reading is root + Compiler/Scan headings above. Unchanged downstream contracts
  are read-only gate references, not default reading: consult only focused rejection clauses for a
  concrete consumer issue. No downstream production/contract edits; MAIN owns the three guard tests.<span data-glossary-navigation="true"> ([guard glossary](../../../../glossary.md#term-guard))</span>

### Accepted [qualification](../../../../glossary.md#term-qualification) and [Ready](../../../../glossary.md#term-ready) receipt

MAIN accepted the bounded support gate before Java on 2026-10-26 after Meitner pre-Java APPROVE.
Header/row/activity/[AD](../../../../glossary.md#autograd) rules, algorithms, APIs/dependencies stay unchanged; ADR0049/0051 suffice.

## Dependencies and integration

- Depends on: Complete Model0041/0044, Compiler0011/0013/0014, Planning0008 and the retained
  Prepare0013 ordinary binding baseline; frozen pre-Java Meitner APPROVE and MAIN acceptance satisfied.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- Conflicts with: all writers to Scan semantics/status, Compiler inference/AD/binding/capture,
  Model Scan Javadoc, or shared planning/docs; none active under MAIN's sole-serial-writer handoff.<span data-glossary-navigation="true"> ([Scan glossary](../../../../glossary.md#term-cumulative-scan); [Compiler glossary](../../../../glossary.md#term-compiler); [inference glossary](../../../../glossary.md#term-inference); [AD glossary](../../../../glossary.md#autograd); [capture glossary](../../../../glossary.md#term-capture-meanings); [Model glossary](../../../../glossary.md#term-model); [Javadoc glossary](../../../../glossary.md#term-javadoc); [planning glossary](../../../../glossary.md#term-planning); [handoff glossary](../../../../glossary.md#term-handoff))</span>
- Parallel group: None; serial contexts, no concurrent write worktree required.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus ALL
  accepted dirty work, not merged predecessors; preserve the baseline described below.
- Integration order: frozen pre-Java APPROVE/MAIN acceptance and Ready → clean Compiler implementation →
  independent Class C source/docs APPROVE → named compatibility checkpoint → MAIN acceptance.
- Integration [validation](../../../../glossary.md#term-validation): MAIN's single **Compiler0015 closed-consumer compatibility [checkpoint](../../../../glossary.md#term-checkpoint)** below.
- Shared-document integration owner: MAIN; sole serial owner of authority/explanations/ADR/glossary,
  this brief, Compiler master-plan and roadmap, with explicit final-doc delegation if needed.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>

MAIN's 2026-10-26 [Ready](../../../../glossary.md#term-ready) receipt verified seven Complete predecessors, base/index and inactive conflicts;
evidence: `/private/tmp/synaptik-compiler0015-main.TDKRDV/main-planning-verification.txt`. Worker now closed.
Protect accepted dirty work against `/private/tmp/synaptik-dynamic-batch-cutover.Of4Q8D/compiler0015-baseline.json`
and its 2480-path copies, not HEAD-only diffs; independent review uses the source freeze.

## Files and symbols

Production prefix: `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/`.

- `ScanBodyInference.java`: `infer`, `validateRows`, `rowResult`, `checkRow`, fixed-feature checks;
  replace static numeric row comparisons with independently proved Dimension provenance.<span data-glossary-navigation="true"> ([feature glossary](../../../../glossary.md#term-software-feature); [Dimension glossary](../../../../glossary.md#dimension); [provenance glossary](../../../../glossary.md#provenance))</span>
- `ScanGradientRules.java`: [`preflight`](../../../../glossary.md#term-preflight), `sourceRoles`, `rowRole`, `apply`, `stackedShape` and
  existing row-constant/zero closures; remove only the admitted symbolic-length restriction.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant))</span>
- `BatchBindingPlan.java`: [`prepare`](../../../../glossary.md#prepare), `indexValidity`, `bind`, `Binding.descriptor`; first test
  the generic implementation. No production edit is required if all scoped/all-port evidence passes.
- Existing seams only for demonstrated gaps: `AutogradPreflight.java` (`preflightBody`, whole-original
  preflight), `FirstOrderAutograd.java` (`expandBody`, `DerivativeConstants`),
  `CapturedGraphInference.java` (`inferAndValidate`), `GraphCapture.java` (`captureBody`),
  `GraphCompiler.java` ([`compile`](../../../../glossary.md#compile)), `ScanBodyInference.refreshDemand`; no second rule algebra.
- Tests in the matching [Compiler](../../../../glossary.md#term-compiler) test package: extend `LengthAwareScanBodyInferenceTest`,
  `LengthAwareScanGradientRulesTest`, `SymbolicScanInferenceTest`, `SymbolicScanGradientRulesTest`,
  `SymbolicScanCompilerTest`, `GeneralScanCompilerTest`, `BatchBindingPlanTest` as relevant.
  Extend existing `ScanGradientRulesTest.Oracle` for test-local length evaluation; no new framework.
  Test-only <span data-glossary-navigation="true"> ([evaluation glossary](../../../../glossary.md#term-evaluation))</span>`SymbolicLengthScanCompilerTest` in that package is allowed if clearer than overloading
  existing symbolic tests; no new production type or test framework.
- `modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Scan.java`: final docs-context
  Javadoc-only status correction at type line 46/overload lines 145–147; no executable/semantic/API change.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Acceptance criteria

- Both dtypes/traversals independently infer combined scans, including broader forward mixed
  ROW(0)/rank-two ROW(1)/SHARED sequences and terminal SHARED carries. Final non-gradient INT64[B]
  lengths has no formal/target/cotangent: checked counts c+s+i+1 actuals, c+s+i formals.
  Preserve positional ties and conservative outer gradient OR.<span data-glossary-navigation="true"> ([dtypes glossary](../../../../glossary.md#data-type--datatype); [rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [gradient glossary](../../../../glossary.md#term-gradient); [INT64 glossary](../../../../glossary.md#data-type--datatype); [target glossary](../../../../glossary.md#term-target); [cotangent glossary](../../../../glossary.md#cotangent))</span>
- Compare [Dimensions](../../../../glossary.md#dimension) by [canonical](../../../../glossary.md#term-canonical-form) value/name, including distinct objects with the same non-`B` name.
  Only row axes contain it, exactly once. Reject foreign/repeated/derived/anonymous symbols, variable
  T/features, sampled layouts, unproved transforms and malformed unused declarations/attributes/siblings.<span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis); [features glossary](../../../../glossary.md#term-feature); [layouts glossary](../../../../glossary.md#layout))</span>
- Prove the full bounded row algebra: right-aligned singleton SHARED broadcasting; transpose and
  allowed MATMUL roles; feature SUM/sumToShape restoration; explicit row SUM stays terminal even
  at B=1; equal-axis sumToShape preserves ROW. No terminal feedback to rows/emissions, SHARED-to-ROW
  invention or shared-carry leakage. Each suffix carry is its own distinct exact old.add(contribution).<span data-glossary-navigation="true"> ([feature glossary](../../../../glossary.md#term-software-feature))</span><span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton); [broadcasting glossary](../../../../glossary.md#broadcasting); [transpose glossary](../../../../glossary.md#term-transpose); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [carry glossary](../../../../glossary.md#term-carry))</span>
- Before derivative Tensor IDs, whole-original preflight retains all-carry/all-original-sequence
  ROW(0) restriction, even unused/non-gradient/unrelated branches and T=0. Preserve residual-root,
  unsupported-rule, fixed-scan, generated-shared-carry differentiation and higher-order/createGraph
  rejection. Broader forward admission does not relax those guards. Preserve Model validation
  before canonical output IDs; do not promise ID rollback for later capture/inference failures.<span data-glossary-navigation="true"> ([derivative glossary](../../../../glossary.md#term-derivative); [Tensor glossary](../../../../glossary.md#tensor); [preflight glossary](../../../../glossary.md#term-preflight); [residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand); [scan glossary](../../../../glossary.md#term-cumulative-scan); [admission glossary](../../../../glossary.md#term-admission); [Model glossary](../../../../glossary.md#term-model); [validation glossary](../../../../glossary.md#term-validation); [rollback glossary](../../../../glossary.md#term-rollback); [capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference))</span>
- Actual opposite-traversal siblings cover final-only/emission-only/joint seeds, two carries,
  false-flag flow, disconnected zeros, tied roles/roots and cross-occurrence sums. Load exact saved
  TANH/SIGMOID, never replay; normalize each SHARED-target term before accumulation. No B×parameter banks.<span data-glossary-navigation="true"> ([seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [occurrence glossary](../../../../glossary.md#term-occurrence); [TANH glossary](../../../../glossary.md#term-tanh); [SIGMOID glossary](../../../../glossary.md#term-sigmoid); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- Generated seeds/histories/constants retain explicit mixed roles, including ROW(1) literal/zero
  sequences [T,F,B], shared transpose histories [T,F,H], row histories [T,B,H], and pruned axes in
  lockstep with used sequences. Capture/rewrites/demand/final queries preserve scopes and saved sites;
  neither coincident local IDs nor equal numeric F=B/H=B nor B=0/1 may infer/erase role identity.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- One BatchBindingPlan fresh-binds B=0,1,F,H,ordinary and 32→32→7→1→32 with fresh actual descriptors,
  including changed admitted isView at repeated B. Check unused/tied actuals including L, scoped formal/value/
  typed obligation, both siblings and all latent final/emission/residual/gradient ports. Existing lazy
  outer indexing suffices; no public local-descriptor API. Ports become canonical non-view; templates
  stay unresolved. Preserve current input-backed descriptors/isView and earlier Bindings; reject bad
  metadata before returning Binding. No memo substitutes for complete fresh scoped admission.
  Changed storage/L payloads at repeated B are numerical ORACLE evidence; bind sees descriptors only.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [lazy glossary](../../../../glossary.md#term-lazy-evaluation); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [view glossary](../../../../glossary.md#term-view-meanings); [Bindings glossary](../../../../glossary.md#term-binding); [metadata glossary](../../../../glossary.md#term-metadata); [memo glossary](../../../../glossary.md#cache-and-memoization); [numerical ORACLE glossary](../../../../glossary.md#adr-ci-and-qualification))</span>
- Check fixed-feature products independently of empty T/B, including unused/dead local values,
  ports and shape attributes; preserve legitimate zero-feature domains. Raw IR/descriptor paths
  must independently reject invalid metadata, not rely solely on Model constructors or incidental
  Optional/index failures. Same-scope obligations cannot disappear in equal-ID neighboring scopes.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [scope glossary](../../../../glossary.md#term-scope))</span>
- Evaluate the actual generated Tensor graph/sibling with the existing test-local helper against
  independent nonuniform active-prefix ordinary unrolling, not handwritten reverse. Cover zero/full/
  mixed/changed L, B=0/1/F/H/ordinary, separate T=0 and T=0/B=0; raw signed-zero/NaN carry/seed
  identity, inactive +0, poisoned padding/seeds, saved sites and contribution-then-distinct-time-ADD.
  Use explicit dtype test oracles/tolerances, not cross-grouping bit equality. Outer MEAN counts B×H,
  not L/prototype B: B=0 is NaN with empty row gradients/nonempty typed-zero parameter sums.
  All-zero L performs no body work.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [unrolling glossary](../../../../glossary.md#term-unrolling); [signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN glossary](../../../../glossary.md#term-nan); [seed glossary](../../../../glossary.md#term-seed); [padding glossary](../../../../glossary.md#padding); [MEAN glossary](../../../../glossary.md#term-mean); [prototype glossary](../../../../glossary.md#term-prototype))</span>
- Production compile/bind reads no input/L values, allocates no history/resources and runs no graph.
  Only the test oracle checks L values, including invalid-last/T=0. Direct L and existing floating-to-
  INT64 producer graphs are separate metadata witnesses, not CPU grants. Keep downstream/cold/wire/
  NN/Training gates and static length/all-valid compatibility under disabled/standard optimization.
  Body/capture size stays O(body), not T-fold unrolling.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [test oracle glossary](../../../../glossary.md#term-test-oracle); [producer glossary](../../../../glossary.md#term-producer); [CPU glossary](../../../../glossary.md#term-cpu); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>

## [Validation](../../../../glossary.md#term-validation)

Worker: focused checks during development; once after executable freeze:

```bash
./gradlew :modules:compiler:test :modules:compiler:javadoc
```

MAIN alone owns this one [metadata](../../../../glossary.md#term-metadata)-[guard](../../../../glossary.md#term-guard) compatibility [checkpoint](../../../../glossary.md#term-checkpoint) after [Compiler](../../../../glossary.md#term-compiler) [admission](../../../../glossary.md#term-admission):

```bash
./gradlew :modules:prepare:test --tests '*PrepareContextTest.directSymbolicLengthProjectionAndEveryIsolatedSymbolRejectDeliberatelyBeforeExtents' \
  :backends:cpu:test --tests '*CpuGeneralScanCapabilityTest.providerDeliberatelyRejectsRawSymbolicLengthQueriesIncludingZeroTime' \
  :extensions:checkpoint:test --tests '*GraphCheckpointGeneralScanTest.symbolicLengthSourceDoesNotOpenCopyRebindOrSave'
```

No root [run](../../../../glossary.md#term-run) by default: no [dependency](../../../../glossary.md#term-dependency)/build/[runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) behavior change. No emitter/native/
performance runs. Review reuses successful tests unless executable changes/evidence gaps require
a rerun. Generate final Compiler [Javadoc](../../../../glossary.md#term-javadoc) after final edits; allowed [Scan](../../../../glossary.md#term-cumulative-scan) comment edits require [Model](../../../../glossary.md#term-model)
Javadoc once, not Model tests. Validate links/anchors/terms/whitespace and `git diff --check`;
compare final [ownership](../../../../glossary.md#term-ownership-meanings) to the dirty baseline, not HEAD alone.

## Documentation and review impact

Independent review reconciles 13 delegated Markdown paths, [Model](../../../../glossary.md#term-model) [Scan](../../../../glossary.md#term-cumulative-scan) status comments and helper [Javadoc](../../../../glossary.md#term-javadoc);
production worker Javadocs stay unchanged. General/architecture/[planning](../../../../glossary.md#term-planning)/[API](../../../../glossary.md#term-application-programming-interface-api)/example profiles apply; no new [ADR](../../../../glossary.md#term-architecture-decision-record-adr).

## Result

- Implemented named-B lengths [inference](../../../../glossary.md#term-inference)/actual first-order [AD](../../../../glossary.md#autograd)/fresh scoped all-port [binding](../../../../glossary.md#term-binding). Binder unchanged.
- Worker changed two production and seven test Java paths (one new); exact paths/diff and detailed
  coverage: `/private/tmp/synaptik-compiler0015-main.TDKRDV/worker-handoff.md` and `worker-baseline-relative.patch`.
- Reused `./gradlew :modules:compiler:test :modules:compiler:javadoc`: PASS, 399 tests/53 suites,
  zero failures/errors/skips; 5s, 14 tasks (4 executed/10 up-to-date). Retained XML independently counted.
- Earlier sandbox lock denial ran no tests; an earlier noIds fixture allocated badLengths inside
  its measured lambda. Moving fixture allocation outside fixed the test; final run supersedes it.<span data-glossary-navigation="true"> ([fixture glossary](../../../../glossary.md#term-fixture))</span>
- Independent Meitner source/docs review: APPROVE, 0 open P1/P2. Actual siblings match independent
  active-prefix ordinary AD, including ROW(1), exact saved overrides, ties/false flags and zeros.
- Documentation/[Javadoc](../../../../glossary.md#term-javadoc)/glossary status and [scope](../../../../glossary.md#term-scope)/[hash](../../../../glossary.md#term-hash)/link/whitespace/brief checks PASS; no semantic/[API](../../../../glossary.md#term-application-programming-interface-api) change.
- MAIN's exact three-consumer command above: PASS, exactly 3 tests (1 Prepare/1 CPU/1 Checkpoint),
  zero failures/errors/skips; 2s, 27 tasks (6 executed/21 up-to-date). Evidence: <span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [CPU glossary](../../../../glossary.md#term-cpu); [Checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>`main-compatibility-counts.json`.
- Final `./gradlew :modules:compiler:javadoc :modules:model:javadoc`: PASS, both regenerated after comments;
  2s, 8 tasks (2 executed/6 up-to-date). Logs: <span data-glossary-navigation="true"> ([Logs glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>`main-compatibility.log`, `main-final-javadoc.log` in the evidence directory.
- MAIN accepted 2026-10-26; no unresolved issues or follow-up. Source freeze/dirty base remain preserved.
- No [Compiler](../../../../glossary.md#term-compiler)-test or review rerun, root/native/performance [run](../../../../glossary.md#term-run), Git mutation, [commit](../../../../glossary.md#term-commit) or push.
- Execution/cold/copy/wire/v5/[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training/higher-order/[memo](../../../../glossary.md#cache-and-memoization) gates remain closed. No successor launched.
- Status: Complete
