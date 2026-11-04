# Task CPU0076: Generic row-[invariant](../../../../glossary.md#term-invariant) external-cell load staging

## Reader context

Staging means loading a proved unchanged external cell into a generated method local earlier,
then reusing that value while it remains valid. A matrix supplied as a Scan invariant may stay
unchanged throughout one invocation; a SHARED history instead supplies a different value at
each original time position even if its shape is identical. The proof must distinguish those
cases before moving any read.

`CpuRowMapReductionLowering.invariantLoadStaging` supplies the common cold proof used by
preparation and the ordinary/Scan emitters. It proves both the caller's generated entry and a
worker helper's local lifetimes; “caller” here can mean the generated Scan entry that calls its
helper, not the application. The locals belong to that invocation/helper call and cannot cache
weights across later runs with changed storage. Inactive work must not read a source merely
because the source would be invariant if used.

The [current staging explanation](../../../../backend-guide/cpu-backend.md#cpu0076-implemented-scalar-invariant-staging)
contrasts a fixed matrix with a time-indexed history and explains checked fallback when the
proof fails. Its [within-row predecessor](../../../../backend-guide/cpu-backend.md#implemented-scalar-loadaddress-reuse-cutover)
has a narrower reuse lifetime. The finite experiment, recorded percentages and JIT limits below
are unchanged historical observations, not a new measurement or universal benefit claim.

## Status

Complete — 2026-10-31. MAIN accepted source, bounded result and final [CPU](../../../../glossary.md#term-cpu)/[Engine](../../../../glossary.md#term-engine) gates;
independent Class C source/protocol/result/documentation review APPROVE, 0 P1/P2.

## Change class and goal

Class C — generated [dataflow](../../../../glossary.md#term-dataflow)/[lifetime](../../../../glossary.md#term-lifetime) and [artifact](../../../../glossary.md#term-artifact) [identity](../../../../glossary.md#term-identity-meanings). Port generic external-cell staging:
invariants once per invocation, SHARED cells only at consuming active original t.
Primary witness: real generated [F32](../../../../glossary.md#floating-point-precision-and-rounding) [scalar](../../../../glossary.md#term-scalar) reverse [Scan](../../../../glossary.md#term-cumulative-scan), not a Java transcription.

## [Scope](../../../../glossary.md#term-scope) and non-goals

- Existing eligible ordinary/Scan F32/F64 scalar rows; ordinary arrays/checked segments, Scan segments.
  Static/named-B/length/combined admission unchanged, including symbolic F64 exclusions. Parallel-scalar
  stages only where proof fits the existing helper ABI; otherwise checked fallback, no capability shrink.<span data-glossary-navigation="true"> ([Scan glossary](../../../../glossary.md#term-cumulative-scan); [F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [scalar glossary](../../../../glossary.md#term-scalar); [admission glossary](../../../../glossary.md#term-admission); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
- Implemented staging requires complete actually consumed fixed-feature views, rank at most two,
  and complete regional-body lifetimes. ROW/carry/bank/scratch and already-hoisted scalar sources
  are excluded. Partial/unused/empty domains, unrelated producers and over-budget proofs remain checked.<span data-glossary-navigation="true"> ([producers glossary](../../../../glossary.md#term-producer))</span><span data-glossary-navigation="true"> ([feature glossary](../../../../glossary.md#term-software-feature); [views glossary](../../../../glossary.md#term-view-meanings); [rank glossary](../../../../glossary.md#term-rank); [carry glossary](../../../../glossary.md#term-carry))</span>
- No RNN/F3/H2/root9/10/value special case, carry promotion, math/store reorder, SIMD rewrite,
  new capability/supported API/type/framework, larger resources, runtime/worker/ownership/ABI change or rollout.
  No old/new emitter pair, stage/commit/push, historical timing rerun or task cascade.<span data-glossary-navigation="true"> ([RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn); [SIMD glossary](../../../../glossary.md#simd); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [runtime glossary](../../../../glossary.md#term-runtime); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [commit glossary](../../../../glossary.md#term-commit))</span>
- Private helpers allowed. ONLY shared exception: existing [lowering](../../../../glossary.md#lowering) [owner](../../../../glossary.md#term-owner)'s
  [`invariantLoadStaging`](../../../../architecture/contracts/backend-execution.md#cpu0076-shared-cold-proof), exact declaration/result contract there;
  use its selection, not competing private eligibility or allocation-derived facts.

## Contracts

Read AGENTS, current sources/focused tests, the root [scope](../../../../glossary.md#term-scope) index and these exact headings:

- Root [Core invariants](../../../../../ARCHITECTURE.md#core-invariants), [Module-ownership routing](../../../../../ARCHITECTURE.md#module-ownership-routing)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>,
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>,
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- [Backend](../../../../glossary.md#backend) [CPU0073 accepted row-map and multiple-terminal reduction fusion](../../../../architecture/contracts/backend-execution.md#cpu0073-accepted-row-map-and-multiple-terminal-reduction-fusion)<span data-glossary-navigation="true"> ([fusion glossary](../../../../glossary.md#term-fusion))</span>:
  `Physical region and admission`, `Cold proof, resources and range`,
  `Exact types, signatures and generation lifetime`, `Selective compatibility and retention`.
- Backend [CPU0074 selected row-region execution exception](../../../../architecture/contracts/backend-execution.md#cpu0074-selected-row-region-execution-exception):
  `Bounded selection and numerical boundary`, `Exact new internal seams`,
  `Resources, dispatch lifetime and compatibility`; [CPU0075 selective scalar load and address reuse](../../../../architecture/contracts/backend-execution.md#cpu0075-selective-scalar-load-and-address-reuse)<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>;
  [CPU0076 invariant external-cell load staging](../../../../architecture/contracts/backend-execution.md#cpu0076-prospective-invariant-external-cell-load-staging)<span data-glossary-navigation="true"> ([invariant glossary](../../../../glossary.md#term-invariant))</span>.
- Backend [Accepted ordinary symbolic-batch CPU contract](../../../../architecture/contracts/backend-execution.md#accepted-ordinary-symbolic-batch-cpu-contract)<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [CPU glossary](../../../../glossary.md#term-cpu))</span>.
- [Scan](../../../../glossary.md#term-cumulative-scan) [Reverse scan and exact primal bindings](../../../../architecture/contracts/recurrent-scan.md#reverse-scan-and-exact-primal-bindings)<span data-glossary-navigation="true"> ([primal glossary](../../../../glossary.md#term-primal); [bindings glossary](../../../../glossary.md#term-binding))</span>,
  [Traversal, empty domains and declared qualification](../../../../architecture/contracts/recurrent-scan.md#traversal-empty-domains-and-declared-qualification)<span data-glossary-navigation="true"> ([qualification glossary](../../../../glossary.md#term-qualification))</span>,
  [CPU0074 selected terminal grouping exception](../../../../architecture/contracts/recurrent-scan.md#cpu0074-selected-terminal-grouping-exception),
  [Activity, raw identity and retained histories](../../../../architecture/contracts/recurrent-scan.md#activity-raw-identity-and-retained-histories)<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>,
  [First-order length-mode reverse](../../../../architecture/contracts/recurrent-scan.md#first-order-length-mode-reverse),
  [Runtime preflight and implementation gates](../../../../architecture/contracts/recurrent-scan.md#runtime-preflight-and-implementation-gates)<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight))</span>.
- [Runtime](../../../../glossary.md#term-runtime) [CPU0074 run-owned synchronous reservation](../../../../architecture/contracts/runtime-prepare-engine.md#cpu0074-run-owned-synchronous-reservation)<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [synchronous glossary](../../../../glossary.md#term-synchronous))</span>.

## Dependencies and integration

- Depends on: CPU0073/0074/0075 Complete, retained baseline and explicit user CPU0076 approval.
- Conflicts with: CPU row proof/emission/cache/authority writers; historical campaigns closed;
  CPU0048–0052 held/inactive. MAIN read-only in this documentation scope.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [cache glossary](../../../../glossary.md#term-cache); [scope glossary](../../../../glossary.md#term-scope))</span>
- Parallel group: None; serial source writer, independent review, exclusive final docs.
- Common base revision: `07a281edd3086efde2853211b71b48288db027e1` plus accepted dirty tree;
  `/private/tmp/synaptik-cpu0076-run.TVyPgh/baseline/accepted-inventory.json` records 2578 file
  hashes/head/empty index: inventory files, not Git-dirty paths or merged predecessors.<span data-glossary-navigation="true"> ([hashes glossary](../../../../glossary.md#term-hash))</span>
- Integration order: authority/packet → independent review + MAIN GO → clean implementation,
  untimed source/equality preflight and freeze → independent source/protocol review → MAIN release
  → one finite observation → confirmed-improvement review → final owning gate/docs → MAIN acceptance.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [release glossary](../../../../glossary.md#term-release); [finite glossary](../../../../glossary.md#term-finite))</span>
- Integration validation: focused checks/final CPU test/Javadoc, justified named boundary checks;
  no root campaign or repeated gate.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Shared-document integration owner: MAIN; this planning/authority writer owns the nine listed
  Markdown files; MAIN delegated exclusive final documentation to the independent reviewer.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [planning glossary](../../../../glossary.md#term-planning))</span>

MAIN verified [Ready](../../../../glossary.md#term-ready)/dependencies/frontier/base and accepted packet/exact-seam review before Java GO.

## Files and symbols

Production root: `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/`:

- `codegen/emit/CpuGeneralScanEmitter.java`: regional stage/load/budget and original-time/range [binding](../../../../glossary.md#term-binding).
- `codegen/emit/CpuClassFileKernelGenerator.java`: `emitRowRegion`, `regionLoad`, typed/address budget.
- `lowering/CpuRowMapReductionLowering.java`: `prove`, shared `invariantLoadStaging` before cuts;
  `lowering/CpuGeneralScanLowering.java`: regional pre-cut scopes/bases and `shapedResources`.
- `prepare/CpuPartitionPreparer.java`: `rowEmission`, `scanRowHelperSlots`, `analyzeGeneralScan`.
- `cache/CpuKernelSpecialization.java`: `compatibilityBytes`, `classIdentityBytes`,
  private staging identity; no new record component.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Changed tests under the matching [CPU](../../../../glossary.md#term-cpu) test root: `codegen/emit/CpuRowScalarReuseTest.java`,
  `lowering/CpuRowMapReductionLoweringTest.java`, `cache/CpuKernelSpecializationTest.java`.
  Reused comparison/builders/original Java oracle and focused suites below; historical resources immutable.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [immutable glossary](../../../../glossary.md#term-immutable))</span>

## Acceptance criteria

1. Prove source/storage/view origin, all-B affine equality/zero row dependence, temporal/formal/
   bank binding, readiness, first-load/branch dominance, nonempty consumer and complete lifetime.
   Include retained/public/saved/scratch/carry writes and inclusive cuts. No equal-size/sampled-B/
   nominal-root shortcut; preserve actual alias/span/alignment/overflow checks and store order.<span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#term-view-meanings); [binding glossary](../../../../glossary.md#term-binding); [lifetime glossary](../../../../glossary.md#term-lifetime); [carry glossary](../../../../glossary.md#term-carry); [nominal glossary](../../../../glossary.md#term-nominal); [alias glossary](../../../../glossary.md#aliasing-and-arena); [alignment glossary](../../../../glossary.md#term-alignment); [overflow glossary](../../../../glossary.md#term-overflow))</span>
2. True invariants stage once per generated entry invocation under nonempty/any-active guards;
   SHARED temporal cells stage after producers at each consuming active original t, never from
   an adjacent time. No source read at T=0/B=0/all-inactive/empty feature domain. Worker staging
   needs an active consumer in its range, not just global active(t); otherwise checked fallback.
   Do not label per-helper staging once-per-whole-Scan.<span data-glossary-navigation="true"> ([producers glossary](../../../../glossary.md#term-producer))</span><span data-glossary-navigation="true"> ([feature glossary](../../../../glossary.md#term-software-feature); [Scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
3. Before cuts/resources/slots prove BOTH caller/helper whole-lifetime bounds, even without exec-v1.
   Include all demanded history LONG bases (forwarded sources/prepublication too), duplicate
   producing wrappers' carrier/LONG bases in BOTH phases, controls/addresses, producers/terminals,
   staged cells, flags and remaining CPU0075 pools. Keep 128 slots (DOUBLE/LONG two), <=48000 bytes,
   unchanged scratch/frame; optimization failure must not shrink capability/strategy.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
4. Real F32 scalar reverse witness must show the lab's ten W/U cells loaded once per invocation
   outside active rows/time loop, not just reused within rows. Preserve row 24 MUL/24 ADD/2 SUB,
   ordered five row stores and twelve distinct memory-backed after-row carry ADDs per active t.
   No across-T carry locals/FMA; inspect ClassFile control/dataflow, not just counts. No hidden
   arithmetic/dispatch, allocation, boxing or reflection.<span data-glossary-navigation="true"> ([F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [scalar glossary](../../../../glossary.md#term-scalar); [FMA glossary](../../../../glossary.md#term-fused-multiply-add-fma); [dataflow glossary](../../../../glossary.md#term-dataflow); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
5. Generic witnesses: non-square/renumbered/computed rows, F64/named-B, genuine differing SHARED
   original-time histories in both traversals against a matching oracle, ROW(1), aliases/distinct
   origins/writes and helper-local inactive ranges. Original R is not the temporal-layout oracle.<span data-glossary-navigation="true"> ([F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [oracle glossary](../../../../glossary.md#term-test-oracle); [layout glossary](../../../../glossary.md#layout))</span>
6. Reuse raw/special/empty/activity/length/rejection/concurrency/mixed-B/fresh-storage and exact-SUM
   coverage. Changed weights are reread per invocation; no inactive/inaccessible source read,
   seed/padding corruption, alias/bounds regression or write before invalid-length rejection.<span data-glossary-navigation="true"> ([concurrency glossary](../../../../glossary.md#term-concurrency); [weights glossary](../../../../glossary.md#term-weight); [seed glossary](../../../../glossary.md#term-seed); [padding glossary](../../../../glossary.md#padding); [regression glossary](../../../../glossary.md#term-regression))</span>
7. Append `|ordinary-row-invariant-load-staging-v1=` / `|scan-row-invariant-load-staging-v1=`
   in BOTH identity streams ONLY where emitted scalar placement changes. Bind all deterministic
   proof/scope/budget choices; no actual B/L/addresses/slots. Canonical keys, envelope71 and
   projections72/71 unchanged. No-staging/SIMD/nonregional keys/bytes/resources stay literal.
   Immutable <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [scope glossary](../../../../glossary.md#term-scope); [Canonical glossary](../../../../glossary.md#term-canonical-form); [SIMD glossary](../../../../glossary.md#simd); [Immutable glossary](../../../../glossary.md#term-immutable))</span>`.tsv`/`.gz.b64` stay untouched; test stale rejection, generation/load/reuse and
   B/L-independent identities. No blanket scalar marker bump.

## [Validation](../../../../glossary.md#term-validation) and [finite](../../../../glossary.md#term-finite) performance grant

Untimed source/ClassFile/equality [preflight](../../../../glossary.md#term-preflight) FIRST; freeze source/arms/harness/reset/[binding](../../../../glossary.md#term-binding)/
protocol/[hashes](../../../../glossary.md#term-hash) for review/MAIN GO. Focused checks (no timing flag):

```bash
./gradlew :backends:cpu:test --tests '*CpuRowScalarReuseTest' --tests '*CpuRowMapReductionGeneratedKernelTest' --tests '*CpuRowFusionSimdParallelTest' --tests '*CpuRowMapReductionLoweringTest' --tests '*CpuGeneralScanLoweringTest' --tests '*CpuKernelSpecializationTest' --tests '*CpuGeneratedKernelArtifactStoreTest' --offline
```

Consumed budget: ONE [JVM](../../../../glossary.md#term-java-virtual-machine-jvm), external 90s watchdog, actual [F32](../../../../glossary.md#floating-point-precision-and-rounding) reverse [Scan](../../../../glossary.md#term-cumulative-scan) B32/B256, T4/F3/H2.
A=literal accepted CPU0075 current G, SHA256
`d8333ed3f20e2663b455411d45eda6a8295edb56346ff959b9d5823c60611a6e`, from
`baseline/weight-lab-frozen/G.class`; B=actual new generator, C=unchanged original Java R.
NOT CPU0074 `authentic-old.class`, lab J0/J1 or rewritten R. Exact shared [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) inputs,
distinct outputs, saved primals/joint [seeds](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings), L[row]=row%5; common reset and staging INSIDE each call.
Per arm/case: three 10000-call warm blocks then three 5000-call [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) blocks, ABC/BCA/CAB each
phase;36 windows total. Five fresh symmetric raw-five-output checks/case, not every timed call.
Samples/events/exit and same-JVM primary/caller/wrapper/outer-loop [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) [logs](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) retained,
including normal [C2](../../../../glossary.md#term-c2)/[OSR](../../../../glossary.md#term-on-stack-replacement-osr)/[C1](../../../../glossary.md#term-c1), [inlining](../../../../glossary.md#term-inlining), failures and retirement. Counts do not prove readiness.
Grant spent; no retry, diagnostic JVM, extra [fork](../../../../glossary.md#term-fork-meanings)/case or historical-budget reuse.

After confirmed improvement and MAIN [release](../../../../glossary.md#term-release), ONE final owning gate:

```bash
./gradlew :backends:cpu:test :backends:cpu:javadoc --offline
```

MAIN's named changed-boundary join (actual reverse/empty-time/empty-row [Engine](../../../../glossary.md#term-engine) [publication](../../../../glossary.md#term-publication)):

```bash
./gradlew :testing:integration-tests:test --tests '*EngineCpuGeneralScanIntegrationTest.nonSquareMatmulAndActualReverseMatchUnrolledAndEmptyTimeThroughEngine' --tests '*EngineCpuGeneralScanIntegrationTest.emptyRowMatmulActualReversePublishesNonemptyTypedZeroParameterGradient' --offline
```

## Documentation and review impact

Independent reviewer finalized exactly nine packet Markdown files: root, [backend](../../../../glossary.md#backend)/[Scan](../../../../glossary.md#term-cumulative-scan) contracts,
boundary explanation, [CPU](../../../../glossary.md#term-cpu) guide, ADR0055, this brief, CPU master and roadmap. General/architecture/
backend/decision/[planning](../../../../glossary.md#term-planning)/example/[API](../../../../glossary.md#term-application-programming-interface-api)-[Javadoc](../../../../glossary.md#term-javadoc) profiles applied; local links/anchors/whitespace checked.
Javadoc implementation contracts reviewed in the six frozen production [owners](../../../../glossary.md#term-owner); no final Java edits,
successful owning generation reused. Targeted Scan/formal/row-role glossary anchors suffice; no new term.

## Result

- Six existing production owners and three tests changed; no new type, public/supported API, ABI or resource.
  The approved shared internal cold method is the only new seam, as specified in the owning contract.
  Both resource P2s closed: full helper duplicate allowance and all caller history lifetimes;
  focused 25-wrapper/60-forwarded-history regressions preserve strategy/output while declining staging.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
- Frozen source/protocol approval binds `FROZEN.json` SHA256
  `3364b02d1f543503506ea1e45236e12c0c25aabd656951cd41df9c2f830fa2ef` under
  `/private/tmp/synaptik-cpu0076-run.TVyPgh/`; exact nine source [hashes](../../../../glossary.md#term-hash) are in that [manifest](../../../../glossary.md#term-manifest).
- Focused command above PASS: 95 tests, 94 passed/1 historical opt-in skipped, 0 fail/error.
  Actual preflight: five gradients raw-equal; row gets22→12, ten invariant gets before time,
  math24/24/2, five ordered stores and twelve distinct memory-backed after-row carry ADDs retained.
  A/B code bytes2011/2072, max_locals128/138, actual CFG-live51/59 (not hardware registers).<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [gradients glossary](../../../../glossary.md#term-gradient); [invariant glossary](../../../../glossary.md#term-invariant); [carry glossary](../../../../glossary.md#term-carry); [CFG glossary](../../../../glossary.md#term-control-flow-graph-cfg))</span>
- Literal proof: 355 hashes unchanged, nine changed hashes in exactly three scalar cases;
  eight SIMD classes and immutable historical resources literal. Both markers affected-only;
  canonical keys/envelope71/projections72/71 and B/L-independent reuse retained.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [SIMD glossary](../../../../glossary.md#simd); [immutable glossary](../../../../glossary.md#term-immutable); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Sole observation exit0, uncensored3.141464s,36windows/72edges, five fresh raw receipts/case.
  B256 median A/B/R5301.4/4360.15/3013.225ns: observed17.7547% lower B time; B/R1.447004.
  Logged primary/Entry/window C2 stable during2893–3088ms. B32 medians699.3166/584/431.2332ns,
  observed16.4899% lower but window C1/C2OSR transitions during1819–1851ms confound causality.<span data-glossary-navigation="true"> ([median glossary](../../../../glossary.md#term-median); [C2 glossary](../../../../glossary.md#term-c2); [C1 glossary](../../../../glossary.md#term-c1))</span>
  `ANALYSIS.json`, [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters)/events/[JIT](../../../../glossary.md#term-just-in-time-compilation-jit)/classes/[identities](../../../../glossary.md#term-identity-meanings)/raw receipts retain exact evidence.
- MAIN final CPU/Javadoc command above PASS once:233 XMLs,1450 tests,1394 executed/56 skipped,
  0 fail/error;94 baseline Javadoc warnings and existing native-access warnings unchanged.
  Named Engine command above PASS once:1 suite,4 parameterized F64/F32 cases,0 skip/fail/error;
  logs <span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Engine glossary](../../../../glossary.md#term-engine); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [logs glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>`final-cpu-gate.log`/`final-engine-join.log`, archived `final-*-test-results/` in the same root.
- Independent Class C source/protocol/result/Javadoc/documentation APPROVE,0 P1/P2; no review reruns.
  Complete-body/rank/large combined-F64 fallback and per-helper lifetime limits remain explicit.
  No universal support/stationarity/speed/parity/C1 repair, assembly cause, defaults or rollout claim.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- Documentation/glossary/Javadoc impact reviewed as above; no unresolved acceptance issue or new
  successor. CPU frontier None; measurement grant closed. Status: Complete.
