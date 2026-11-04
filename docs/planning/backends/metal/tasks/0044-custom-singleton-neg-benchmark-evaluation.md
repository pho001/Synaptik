# Task 0044: Custom [Singleton](../../../../glossary.md#term-singleton)-NEG Benchmark [Evaluation](../../../../glossary.md#term-evaluation) and No-Change Closure

## Reader context

This was an evidence-limits decision with no new measurement or implementation. Retaining the
existing custom-singleton heuristic did not declare it the faster route. The recorded timing
samples came from one independent JVM process and one frozen workload, narrower than the
heuristic's full element-count domain.

CUSTOM_SINGLE_NEG and MPSGRAPH named backend choices; CUSTOM_KERNEL and GRAPH_EXECUTABLE were
their trace descriptions. Neither a favorable timing summary nor exact outputs for that corpus
authorized a general production policy change. Metal's existing analysis and authenticated
session decisions therefore remained unchanged.
The [current report-only benchmark boundary](../../../../backend-guide/metal-backend.md#fixed-report-only-route-benchmark)
explains why attestation, descriptive timing and production selection are separate claims.
The historical timing arrays below are retained observations, not newly reproduced results.

## Status

Complete

Executed from exact clean [planning](../../../../glossary.md#term-planning) revision
`3d458b7a1d356f32fcd2c04de408ae9aa22355b0` after Task 0043 completed at remediation
`77e6091b2a452faa04fa2b674bc295ddc7be88b7` and the same independent Class C reviewer returned
`APPROVE` with zero remaining findings.

## Change class

Class A — documentation and audit only. This task interprets already-recorded report-only evidence
and changes no production behavior, benchmark, test, [API](../../../../glossary.md#term-application-programming-interface-api), native source or [application binary
interface](../../../../glossary.md#term-application-binary-interface-abi) ([ABI](../../../../glossary.md#application-binary-interface-and-native-bridge)), [schema](../../../../glossary.md#term-schema), [identity](../../../../glossary.md#term-identity-meanings), route, candidate order, [cache](../../../../glossary.md#term-cache), threshold, or policy. Its bounded
no-change conclusion can become Complete directly after the documentation [validation](../../../../glossary.md#term-validation) below; no
code review is required.

## Goal

Close the custom-[kernel](../../../../glossary.md#kernel) benchmark [evaluation](../../../../glossary.md#term-evaluation) with the smallest honest decision supported by the
reviewed Task 0043 evidence:

- retain both current [singleton](../../../../glossary.md#term-singleton)-`NEG` route candidates;
- retain `CUSTOM_SINGLE_NEG` as the no-selected-decision safe heuristic for its existing exact
  domain; and<span data-glossary-navigation="true"> ([heuristic glossary](../../../../glossary.md#term-heuristic))</span>
- make no performance endorsement or other production change.

This is a conservative no-change conclusion from insufficient production-decision evidence, not a
claim that the custom kernel won.

## Evidence source and [integrity](../../../../glossary.md#term-integrity)

Use only Task 0043's updated [metadata](../../../../glossary.md#term-metadata)-complete baseline produced after P2 remediation. The original
pre-remediation report is stale and must not be cited. The updated report was validated in memory
and was not retained in the repository.

The accepted report identifies:

- top-level [schema](../../../../glossary.md#term-schema) `1`, generator schema `2`, benchmark class [SHA-256](../../../../glossary.md#term-sha-256)
  `5aedbaf2e731974fa0c4e4797ef47aacb7cbe835a5b87ab6d38d649ebe70d28a`;
- remediation base `c3e12e7de949b8d3f2c5dad5f45b8086cfc8fedb`, staged executable tree
  `1eeede489969c95b969ba91660ec8d06163b41fd`, and harness-source SHA-256
  `df6d3b514ff5d791632fea1daaae1a903881fcc545c785972bca8173a34d2989`;
- the exact configured [dylib](../../../../glossary.md#term-dylib) SHA-256
  `8aebd8773d33e12b0348cf07b5d3789fc3631ab1d3bbf522691e8b97eb329190`;
- one MacBook Pro `Mac15,9`, Apple M3 Max with 40-core GPU/Metal 4, 64 GB memory and 16
  processors, macOS 26.6.2 build 25G83, Java 26.0.1, the fixed baseline JVM arguments, AC power,
  and no recorded thermal or performance warning; and<span data-glossary-navigation="true"> ([GPU glossary](../../../../glossary.md#term-gpu); [Metal glossary](../../../../glossary.md#term-metal); [JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [arguments glossary](../../../../glossary.md#term-argument))</span>
- [fork](../../../../glossary.md#term-fork-meanings) `1` only.

The frozen workload is exactly one [canonical](../../../../glossary.md#term-canonical-form) [contiguous](../../../../glossary.md#term-contiguous) no-grad `FLOAT32 NEG`, `STRICT_IEEE`,
[graph optimization](../../../../glossary.md#forward-graph-optimization) disabled, caller-input [Shape](../../../../glossary.md#shape) `[1_048_576]`, using generator schema 2. Its input
SHA-256 is `d8643264bb46cbdfca78a87012059336a94d402c798e259bbd1e19f0046d412e`; its raw sign-bit-flipped
[oracle](../../../../glossary.md#term-test-oracle) SHA-256 is `d95c57cdfc2aca246e868f7f050233768417ba2c1e5967c799afa05ea6c8bb02`.

Both candidates attested successful [PREPARE](../../../../glossary.md#prepare) and [RUN](../../../../glossary.md#term-run) with native `SUCCESS`, `NOT_QUERIED`, and
`SESSION` compatibility. Every ordinary precheck and all eight post-[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) checks per route matched
the exact oracle bytes. This proves both routes are correct and executable for this frozen case on
this one environment. It does not prove comparative production performance.

## Exact descriptive timing facts

The report explicitly says `eligibleForProductionDecision=false`, `autotuningEvidence=false`, and
`selectionMode=fixed-candidate-enumeration-no-winner`. One fresh [JVM](../../../../glossary.md#term-java-virtual-machine-jvm) invocation is the independent
unit; its eight paired [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) are correlated. No timing below is a winner, recommendation, gate,
or confidence statement.

Shared descriptive setup durations were:

- [graph](../../../../glossary.md#graph) [compile](../../../../glossary.md#compile): `1,070,875` ns;
- candidate enumeration: `541,209` ns; and
- complete command elapsed: `1,116,815,000` ns.

Cold preparation occurred once per route in fixed candidate order and is not a comparative
[statistic](../../../../glossary.md#term-statistic):

- `CUSTOM_KERNEL`: `1,855,208` ns;
- `GRAPH_EXECUTABLE`: `3,939,250` ns.

The retained hot boundary was only
`PreparedExecutionRunner.run(prepared, borrowedInput) + RunResult.close()`. It excluded compile,
enumeration, preparation, upload, tracing, output download, [checksum](../../../../glossary.md#term-checksum), [JSON](../../../../glossary.md#term-json) construction, and
integration cleanup.

`CUSTOM_KERNEL` retained:

- iterations: `[42,42,42,53,53,53,53,53]`;
- raw [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) ns:
  `[30470583,29592542,28429917,27422166,28893583,30914583,29003833,27109542]`;
- integer normalized [run](../../../../glossary.md#term-run) ns:
  `[725490,704584,676902,517399,545161,583294,547242,511500]`; and
- descriptive minimum/[median](../../../../glossary.md#term-median)/maximum: `511,500 / 583,294 / 725,490` ns.

`GRAPH_EXECUTABLE` retained:

- iterations: `[36,36,45,45,57,57,57,57]`;
- raw batch ns:
  `[26174250,27241000,25979209,46055500,32272458,33115542,32774917,34941833]`;
- integer normalized run ns:
  `[727062,756694,577315,1023455,566183,580974,574998,613014]`; and
- descriptive minimum/median/maximum: `566,183 / 613,014 / 1,023,455` ns.

All sixteen raw batches met the declared 25 ms floor. The alternating within-process order and
retention of every sample make these facts reproducible observations; they do not turn correlated
samples from one [fork](../../../../glossary.md#term-fork-meanings) into independent evidence.

## Current route policy and domain mismatch

Current [Metal](../../../../glossary.md#term-metal) analysis uses `CUSTOM_SINGLE_NEG` without a selected decision only for exactly one
`NEG`, one unique feed, one unique [target](../../../../glossary.md#term-target), and a checked [element](../../../../glossary.md#term-element) count in `1..UINT32_MAX`; every
other supported [partition](../../../../glossary.md#partition) uses [`MPSGRAPH`](../../../../glossary.md#term-mpsgraph). The eligible candidate order is
`[CUSTOM_SINGLE_NEG, MPSGRAPH]`. A compatible session-local selected decision can choose the other
candidate.

That [heuristic](../../../../glossary.md#term-heuristic) covers both numerical profiles, the complete positive element-count interval, and
caller-input or exact splat feeds. The report covers only `STRICT_IEEE`, one caller-input [Shape](../../../../glossary.md#shape),
one corpus, one M3 Max environment, one native/[JVM](../../../../glossary.md#term-java-virtual-machine-jvm)/OS build, and one [fork](../../../../glossary.md#term-fork-meanings). It does not cover the
heuristic's size domain, splat form, both profiles, other Apple-silicon targets, OS/native/JVM
strata, or end-to-end preparation costs. It also has no preregistered production [objective](../../../../glossary.md#term-objective),
uncertainty calculation, [regression](../../../../glossary.md#term-regression) ceiling, representative weighting, or decision threshold.

Changing the default globally would therefore generalize beyond every independent [dimension](../../../../glossary.md#dimension) the
report fixed. Adding a `[1_048_576]` special case or threshold would instead manufacture a new
policy from one observation. Neither is authorized.

## Decision

Retain both implemented routes and the existing safe-[heuristic](../../../../glossary.md#term-heuristic)-first order. With no compatible
selected decision, retain `CUSTOM_SINGLE_NEG` for its existing exact eligible domain and
[`MPSGRAPH`](../../../../glossary.md#term-mpsgraph) for every other supported [partition](../../../../glossary.md#partition).

This decision preserves reviewed behavior because the available report cannot justify changing it.
It does **not** say that `CUSTOM_KERNEL` is faster, preferred, lower [variance](../../../../glossary.md#term-variance), or generally better;
does not select a winner; and does not change a default, order, route policy, [cache](../../../../glossary.md#term-cache), candidate,
[kernel](../../../../glossary.md#kernel), threshold, [Shape](../../../../glossary.md#shape) special case, or later tuning decision. It also does not remove either
route.

## Measurement boundary

No additional measurement is authorized or necessary for this Class A no-change closure. Repeating
the same baseline or adding another bounded single case would not close the independent-[fork](../../../../glossary.md#term-fork-meanings),
[target](../../../../glossary.md#term-target), workload-domain, or statistical-protocol gaps and would risk treating pseudoreplication as
evidence.

A future production route/default change would require a separately planned and reviewed
multi-fork, representative-target/workload, statistical protocol with a declared [objective](../../../../glossary.md#term-objective),
process-order design, uncertainty method, correctness gates, and [regression](../../../../glossary.md#term-regression) limits. The current
user rule prohibits that [matrix](../../../../glossary.md#term-matrix-meanings), so no such task or measurement is authorized here.

## Files and exclusions

Change only:

- this task;
- `docs/planning/backends/metal/master-plan.md`; and
- `docs/planning/roadmap.md`.

Do not edit production code, [native code](../../../../glossary.md#native-numerical-library-and-instruction-architecture), tests, benchmark code, public [API](../../../../glossary.md#term-application-programming-interface-api) documentation, [backend](../../../../glossary.md#backend)
or developer guides, architecture documents, tuning tools, [schemas](../../../../glossary.md#term-schema), [identities](../../../../glossary.md#term-identity-meanings), or generated
files. Do not add or retain a report, [manifest](../../../../glossary.md#term-manifest), [cache](../../../../glossary.md#term-cache), baseline [artifact](../../../../glossary.md#term-artifact), script, or probe.

## [Validation](../../../../glossary.md#term-validation) and completion

Validate the three changed Markdown files for links, anchors, fences, final newlines, and trailing
whitespace. Search the changed text for accidental winner, speedup, superiority, recommendation,
or production-change language; [run](../../../../glossary.md#term-run) `git diff --check`; verify the exact documentation-only path
[scope](../../../../glossary.md#term-scope) and a clean [commit](../../../../glossary.md#term-commit).

Run no [Gradle](../../../../glossary.md#term-gradle) task, test, build, Javadocs, native build, [device](../../../../glossary.md#term-device) probe, [smoke](../../../../glossary.md#term-smoke-test), baseline, or other
measurement. After those documentation checks, record the exact [planning](../../../../glossary.md#term-planning) base and validation in
this brief, set Task 0044 to Complete directly, and leave no invented [Ready](../../../../glossary.md#term-ready) [Metal](../../../../glossary.md#term-metal) implementation
task.

## Acceptance

- Every fact above matches the updated reviewed Task 0043 report, not the stale original report.
- Both routes are described as exact and correct only for the frozen case.
- All timing arrays and [distributions](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) are labeled descriptive and correlated.
- The no-change decision retains both routes and the existing no-selected-decision heuristic without
  performance endorsement.<span data-glossary-navigation="true"> ([heuristic glossary](../../../../glossary.md#term-heuristic))</span>
- The broader production domain and every missing inference [dimension](../../../../glossary.md#dimension) are explicit.
- No additional measurement, production conclusion, report [artifact](../../../../glossary.md#term-artifact), or out-of-[scope](../../../../glossary.md#term-scope) file exists.
- Task 0044 becomes Complete after documentation validation; Metal has no newly invented Ready
  implementation frontier.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Metal glossary](../../../../glossary.md#term-metal); [Ready glossary](../../../../glossary.md#term-ready))</span>

## Result

Complete as a Class A documentation/audit no-change closure. The updated reviewed Task 0043 facts
were transcribed exactly without retaining its [JSON](../../../../glossary.md#term-json) report. Both routes remain exact and correct
only for the frozen `STRICT_IEEE` caller-input `[1_048_576]` case on the one recorded M3 Max
environment/[fork](../../../../glossary.md#term-fork-meanings); all timing arrays and [distributions](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) remain descriptive correlated observations.

Production retains both routes and the existing `CUSTOM_SINGLE_NEG` no-selected-decision safe
[heuristic](../../../../glossary.md#term-heuristic). No winner, performance endorsement, default/order/policy/[cache](../../../../glossary.md#term-cache)/removal/threshold/
[Shape](../../../../glossary.md#shape)-special-case change, additional measurement, or new [Ready](../../../../glossary.md#term-ready) implementation task exists.

The task, [Metal](../../../../glossary.md#term-metal) master, and roadmap passed Markdown link/anchor/fence/final-newline/
trailing-whitespace [validation](../../../../glossary.md#term-validation), the targeted forbidden-claim and status searches, exact
documentation-only path-[scope](../../../../glossary.md#term-scope) review, and `git diff --check`. No build, test, Javadocs, native
action, [device](../../../../glossary.md#term-device) probe, [smoke](../../../../glossary.md#term-smoke-test), baseline, or other measurement ran.
