# Declarative compilation and tuning values

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileArtifacts.java)

The implemented `modules:config` surface contains four standalone compile-configuration values:

- `BackendIntent` records whether planning has one hard backend eligibility target;
- `CompileMode` records the requested compile-time graph scope;
- `GraphOptimizationConfig` permits or suppresses optional semantics-preserving compiler work;
- `PartitionScoringConfig` records an optional soft `DeviceClass` preference for later ranking of
  already eligible ownership candidates.

Model owns one family/dtype numerical contract; these configuration values contain no numerical
selector. Metal uses occurrence-specific provider capability and its fixed custom program for
admitted BFLOAT16/FLOAT16 partitions. Finite arithmetic tolerances are test-only route
qualification, not Engine or Runtime policy.

They are immutable requests, not a runnable compiler configuration aggregate. For example:

```java
import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.backend.contract.BackendIdRequirement;
import io.github.pho001.synaptik.backend.contract.DeviceClass;
import io.github.pho001.synaptik.config.compile.BackendIntent;
import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;

BackendId cuda = new BackendId("cuda");
BackendIntent unconstrained = BackendIntent.unconstrained();
BackendIntent requireCuda =
        BackendIntent.requiring(new BackendIdRequirement(cuda));
CompileMode graphScope = CompileMode.FORWARD_AND_BACKWARD;
GraphOptimizationConfig optimization = GraphOptimizationConfig.standard();
PartitionScoringConfig neutralRanking = PartitionScoringConfig.neutral();
PartitionScoringConfig preferAccelerator =
        PartitionScoringConfig.preferring(DeviceClass.ACCELERATOR);
```

`unconstrained.hardRequirement()` is empty. That absence means only that no hard eligibility
target constrains planning; it does not select a default backend or promise discovery,
fallback, availability, capability, or successful ownership. `requireCuda.hardRequirement()`
contains the exact requirement reference supplied to `requiring`. Direct construction with an
`Optional<BackendRequirement>` retains that exact optional reference.

The canonical constructor rejects a null optional with message `hardRequirement`, and
`requiring(null)` rejects null with message `requirement`. Each factory returns a fresh record.
The record evaluates no requirement and contains no preference, scoring, numerical policy, service,
preparation, run, publication, or execution behavior.

`graphScope` requests current internal compiler autograd expansion and combined
forward/backward compile-time graph work. The other exact values are `FORWARD_ONLY` and
`TRAINING_STEP`. The latter records the
architecture's training-step direction but does not add an optimizer, optimizer-update graph,
training session, schedule, or execution behavior.

`optimization.optionalOptimizationsEnabled()` is `true`, so the current internal compiler may apply its
standard optional semantics-preserving pipeline. `GraphOptimizationConfig.disabled()` returns a
fresh false value that requests skipping only optional optimization. It cannot disable capture,
ordering, inference, validation, mandatory canonical representation, mode-required autograd,
publication binding, planning, preparation, or execution. Neither value exposes a pass list,
pass order, numerical relaxation, backend fusion switch, or execution policy.

`neutralRanking.preferredDeviceClass()` is empty. That means only that this value supplies no
explicit coarse device-class preference; it does not choose a default, promise fallback, or imply
equal candidate scores. `preferAccelerator` contains the exact `DeviceClass.ACCELERATOR` reference.
The preference is soft and applies only after hard eligibility, so it neither makes an eligible
CPU candidate ineligible nor weakens a conflicting hard requirement or guarantees accelerator
ownership. Direct construction retains the exact non-null `Optional<DeviceClass>` reference and
rejects null with message `preferredDeviceClass`; `preferring(null)` rejects null with message
`deviceClass`. Both factories return fresh values.

`PartitionScoringConfig` itself does not enumerate or evaluate candidates, calculate or compare
scores, contain profile measurements, choose ownership or a device, select a route or kernel, or
perform compiler, prepare, runtime, or execution work. Current Planning interprets only its
optional class preference through a cost-free provider-order baseline, and package-private
Compiler supplies that value per final graph node. There is no graph-wide numerical selection.
`CompileConfig` and public graph-wide planning remain planned; the current Engine passes the four
operational standalone compile inputs without a numerical selector.

The implemented `config.tuning` package contains one separate declarative facade,
`ModelAutotuningConfig`. Possessing this value means that model autotuning was requested; there is
no `enabled` component, disabled sentinel, implicit default, or default cache location. Its seven
ordered inputs are:

- the sole current objective, `MIN_MEDIAN_ELAPSED_NANOS`;
- Phase-1 positive cache-miss and per-miss candidate maxima plus non-negative warmup and positive
  odd timed-sample counts;
- one caller-defined, schema-versioned representative-profile identity whose opaque bytes are
  snapshotted on construction and access;
- either `REQUIRE_TUNED_RESULT` or `ALLOW_SAFE_HEURISTIC`; and
- one explicit workload-cache `Path`, retained exactly without normalization or I/O;
- one independent Phase-2 `CompletePlanBudget` containing a positive complete-plan candidate
  maximum, non-negative warmup count, positive odd timed-sample count, positive whole-transaction
  execution ceiling, and non-negative aggregate correctness ceiling in bytes; and
- one explicit non-empty-string model-plan-cache `Path`, retained exactly without normalization or
  I/O.

For example, this current code constructs request data only:

```java
import io.github.pho001.synaptik.config.tuning.ModelAutotuningConfig;
import java.nio.file.Path;

ModelAutotuningConfig request =
        new ModelAutotuningConfig(
                ModelAutotuningConfig.Objective.MIN_MEDIAN_ELAPSED_NANOS,
                new ModelAutotuningConfig.Budget(8, 16, 2, 5),
                new ModelAutotuningConfig.RepresentativeProfileIdentity(
                        1, new byte[] {0x2a, 0x11}),
                ModelAutotuningConfig.FallbackPolicy.REQUIRE_TUNED_RESULT,
                Path.of("cache", "workloads.bin"),
                new ModelAutotuningConfig.CompletePlanBudget(
                        4, 1, 3, 20L, 1_048_576L),
                Path.of("cache", "model-plan.bin"));
```

The concrete inputs permit at most eight distinct workload-cache misses, at most sixteen complete
candidates for each Phase-1 miss, two Phase-1 untimed warmup executions, and five Phase-1 timed
executions per candidate. Independently, Phase 2 permits at most four actual complete-plan
candidates, one warmup and three timed samples per candidate, at most twenty total executions, and
at most 1,048,576 aggregate canonical-publication correctness bytes. Later Phase-2 preflight uses
the actual candidate count `N` and checks `N * (1 + W + S)` against the total-execution ceiling.
The profile bytes identify the caller's representative profile; they are not representative input
values. Construction validates and retains policy data but does not inspect either path, run
tuning, read or write a cache, enumerate candidates, perform correctness execution, prepare an
executable, or perform Engine or Runtime work.

Current `Engine.prepareTuned(...)` maps both budgets and the objective one-for-one, supplies the
request's caller-defined model and representative-profile identities, binds its live
representative inputs, and passes both explicit cache paths to `tools/tuning`. Phase 1 exposes
exactly one backend-local occurrence at occurrence index 0, partition index 0, and weight 1. CPU
retains its current local and complete-plan alternatives. Metal accepts only an exact singleton
NEG and exposes exactly the ordered local candidates custom kernel then MPSGraph. Its authenticated
selected decision then passes unchanged into a one-candidate complete-plan batch fixed to that
route. Phase 2 checks every candidate by exact canonical publication bytes before any warmup or
timed execution, measures the bounded candidates, authenticates the winner, cleans the
representative session, and prepares exactly one fresh selected production recipe.

Current CPU and Metal complete-plan reuse is `SESSION`, so Phase 2 retains but does not inspect,
create, or publish the supplied model-plan path. Metal local compatibility is also `SESSION`, so
the supplied workload-cache path can neither provide a Metal hit nor receive a Metal publication.
The same generic translation remains valid for a future producer that can honestly declare
`PERSISTENT`. Strict mode reports a failed or unavailable complete tuning result, while
safe-heuristic mode may return one fresh ordinary safe preparation through the already-selected
owner with no evidence. Fallback grants no candidate eligibility, numerical relaxation, cache
compatibility, partial-result acceptance, owner substitution, or suppression of an unrelated
preparation failure. Multiple-occurrence extraction, multiple partitions, mixed ownership, and
broader Compiler/Planning plan alternatives remain planned.


---

[Previous topic](../backend-identities.md) · [Next topic](planning-values.md)
