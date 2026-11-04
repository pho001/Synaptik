# Declarative [compilation](../../glossary.md#compiler-engine-and-execution-stage-names) and tuning values

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileArtifacts.java)

These configuration values express requests rather than perform work. A **hard requirement**
excludes unacceptable [backend](../../glossary.md#backend) choices; a **soft preference** ranks choices that already qualify.
An [autotuning](../../glossary.md#autotuning--model-autotuning) request asks later preparation to compare eligible execution alternatives within
explicit budgets, rather than granting an otherwise unsupported alternative permission to [run](../../glossary.md#term-run).

The implemented `modules:config` surface contains four standalone [compile](../../glossary.md#compile)-configuration values:

- [`BackendIntent`](../../glossary.md#backend-intent--backendintent) records whether [planning](../../glossary.md#term-planning) has one hard backend eligibility [target](../../glossary.md#term-target);
- [`CompileMode`](../../glossary.md#compile-mode--compilemode) records the requested compile-time [graph](../../glossary.md#graph) [scope](../../glossary.md#term-scope);
- [`GraphOptimizationConfig`](../../glossary.md#graph-optimization-configuration--graphoptimizationconfig) permits or suppresses optional semantics-preserving [compiler](../../glossary.md#term-compiler) work;
- [`PartitionScoringConfig`](../../glossary.md#partition-scoring-configuration--partitionscoringconfig) records an optional soft [`DeviceClass`](../../glossary.md#device-class--deviceclass) preference for later ranking of
  already eligible ownership candidates.

[Model](../../glossary.md#term-model) owns one family/[dtype](../../glossary.md#data-type--datatype) numerical contract; these configuration values contain no numerical
selector. [Metal](../../glossary.md#term-metal) uses [occurrence](../../glossary.md#term-occurrence)-specific provider capability and its fixed custom program for
admitted [BFLOAT16](../../glossary.md#term-bfloat16)/[FLOAT16](../../glossary.md#term-float16) partitions. [Finite](../../glossary.md#term-finite) arithmetic tolerances are test-only route
[qualification](../../glossary.md#term-qualification), not [Engine](../../glossary.md#term-engine) or [Runtime](../../glossary.md#term-runtime) policy.

They are [immutable](../../glossary.md#term-immutable) requests, not a runnable compiler configuration aggregate. For example:

```java
// Constructing policy records performs no compilation, discovery or allocation of tensor storage.
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
fallback, availability, capability, or successful [ownership](../../glossary.md#term-ownership-meanings). `requireCuda.hardRequirement()`
contains the exact requirement reference supplied to `requiring`. Direct construction with an
`Optional<BackendRequirement>` retains that exact optional reference.

The [canonical](../../glossary.md#term-canonical-form) constructor rejects a null optional with message `hardRequirement`, and
`requiring(null)` rejects null with message `requirement`. Each factory returns a fresh record.
The record evaluates no requirement and contains no preference, scoring, numerical policy, service,
preparation, run, [publication](../../glossary.md#term-publication), or execution behavior.

`graphScope` requests current internal compiler [autograd](../../glossary.md#autograd) expansion and combined
forward/backward compile-time graph work. The other exact values are `FORWARD_ONLY` and
`TRAINING_STEP`. The latter records the
architecture's training-step direction but does not add an [optimizer](../../glossary.md#term-optimizer), optimizer-update graph,
[training session](../../glossary.md#training-session--trainingsession), schedule, or execution behavior.

`optimization.optionalOptimizationsEnabled()` is `true`, so the current internal compiler may apply its
standard optional semantics-preserving pipeline. `GraphOptimizationConfig.disabled()` returns a
fresh false value that requests skipping only optional optimization. It cannot disable [capture](../../glossary.md#term-capture-meanings),
ordering, [inference](../../glossary.md#term-inference-meanings), [validation](../../glossary.md#term-validation), mandatory canonical representation, mode-required autograd,
publication [binding](../../glossary.md#term-binding), planning, preparation, or execution. Neither value exposes a pass list,
pass order, numerical relaxation, backend [fusion](../../glossary.md#term-fusion) switch, or execution policy.

`neutralRanking.preferredDeviceClass()` is empty. That means only that this value supplies no
explicit coarse [device](../../glossary.md#term-device)-class preference; it does not choose a default, promise fallback, or imply
equal candidate scores. `preferAccelerator` contains the exact `DeviceClass.ACCELERATOR` reference.
The preference is soft and applies only after hard eligibility, so it neither makes an eligible
[CPU](../../glossary.md#term-cpu) candidate ineligible nor weakens a conflicting hard requirement or guarantees [accelerator](../../glossary.md#term-accelerator)
ownership. Direct construction retains the exact non-null `Optional<DeviceClass>` reference and
rejects null with message `preferredDeviceClass`; `preferring(null)` rejects null with message
`deviceClass`. Both factories return fresh values.

`PartitionScoringConfig` itself does not enumerate or evaluate candidates, calculate or compare
scores, contain profile measurements, choose ownership or a device, select a route or [kernel](../../glossary.md#kernel), or
perform compiler, [prepare](../../glossary.md#prepare), runtime, or execution work. Current Planning interprets only its
optional class preference through a cost-free provider-order baseline, and package-private
Compiler supplies that value per final graph [node](../../glossary.md#node). There is no graph-wide numerical selection.
`CompileConfig` and public graph-wide planning remain planned; the current Engine passes the four
operational standalone compile inputs without a numerical selector.

The implemented `config.tuning` package contains one separate declarative [facade](../../glossary.md#term-facade),
[`ModelAutotuningConfig`](../../glossary.md#model-autotuning-request-facade--modelautotuningconfig). Possessing this value means that [model autotuning](../../glossary.md#autotuning--model-autotuning) was requested; there is
no `enabled` component, disabled sentinel, implicit default, or default [cache](../../glossary.md#term-cache) location. Its seven
ordered inputs are:

- the sole current [objective](../../glossary.md#term-objective), `MIN_MEDIAN_ELAPSED_NANOS`;
- Phase-1 positive cache-miss and per-miss candidate maxima plus non-negative warmup and positive
  odd timed-sample counts;
- one caller-defined, schema-versioned representative-profile identity whose opaque bytes are
  snapshotted on construction and access;
- either `REQUIRE_TUNED_RESULT` or `ALLOW_SAFE_HEURISTIC`; and
- one explicit workload-cache `Path`, retained exactly without [normalization](../../glossary.md#term-normalization) or I/O;
- one independent Phase-2 `CompletePlanBudget` containing a positive complete-plan candidate
  maximum, non-negative warmup count, positive odd timed-sample count, positive whole-transaction
  execution ceiling, and non-negative aggregate correctness ceiling in bytes; and
- one explicit non-empty-string model-plan-cache `Path`, retained exactly without normalization or
  I/O.

For example, this current code constructs request data only:

```java
// Paths and profile identity label a tuning request; construction does not read caches or run trials.
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

The concrete inputs permit at most eight distinct Phase-1 workload-cache misses, sixteen
candidates per miss, two untimed warmups and five timed samples per candidate. Independently,
Phase 2 permits four complete-plan candidates, one warmup and three timed samples per candidate,
twenty total executions, and 1,048,576 aggregate correctness bytes. Let $N$ be the actual
candidate count, $W$ the warmups per candidate and $S$ the timed samples per candidate, all
non-negative integers (with $S$ positive and odd). The execution count is
$E=N(1+W+S)$: one correctness run plus warmups and samples for each candidate. At
$N=4$, $W=1$, $S=3$, $E=4(1+1+3)=20$, exactly the example's ceiling. Profile identity bytes
label the caller's representative workload, not input values. Constructing the request neither
reads paths nor performs tuning, cache I/O, correctness execution or preparation.

Current `Engine.prepareTuned(...)` maps both budgets and the objective one-for-one, supplies the
request's caller-defined model and representative-profile [identities](../../glossary.md#term-identity-meanings), binds its live
representative inputs, and passes both explicit cache paths to `tools/tuning`. Phase 1 exposes
exactly one backend-local occurrence at occurrence index 0, [partition](../../glossary.md#partition) index 0, and [weight](../../glossary.md#term-weight) 1. CPU
retains its current local and complete-plan alternatives. Metal accepts only an exact [singleton](../../glossary.md#term-singleton)
NEG and exposes exactly the ordered local candidates custom kernel then [MPSGraph](../../glossary.md#term-mpsgraph). Its authenticated
selected decision then passes unchanged into a one-candidate complete-plan [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) fixed to that
route. Phase 2 checks every candidate by exact canonical publication bytes before any warmup or
timed execution, measures the bounded candidates, authenticates the winner, cleans the
representative session, and prepares exactly one fresh selected production recipe.

Current CPU and Metal complete-plan reuse is `SESSION`, so Phase 2 retains but does not inspect,
create, or publish the supplied model-plan path. Metal local compatibility is also `SESSION`, so
the supplied workload-cache path can neither provide a Metal hit nor receive a Metal publication.
The same generic translation remains valid for a future [producer](../../glossary.md#term-producer) that can honestly declare
`PERSISTENT`. Strict mode reports a failed or unavailable complete tuning result, while
safe-[heuristic](../../glossary.md#term-heuristic) mode may return one fresh ordinary safe preparation through the already-selected
[owner](../../glossary.md#term-owner) with no evidence. Fallback grants no candidate eligibility, numerical relaxation, cache
compatibility, partial-result acceptance, owner substitution, or suppression of an unrelated
preparation failure. Multiple-occurrence extraction, multiple partitions, mixed ownership, and
broader Compiler/Planning plan alternatives remain planned.


---

[Previous topic](../backend-identities.md) · [Next topic](planning-values.md)
## Which caller interprets this data?

The application or integration author constructs request values; Engine forwards compile requests
to Compiler/Planning and tuned-preparation requests to the tuning collaboration. Constructors
validate and snapshot configuration, not representative tensor contents. In particular, profile
identity bytes label the application's representative workload; they are not a numerical-policy
profile or stored sample inputs. Tuning needs separately live values under
[Engine's tuned-preparation lifetime](../engine/lifecycle.md#current-ordinary-explicit-composition-and-advanced-lifecycle).
Read [compile configuration](configuration.md) for the ordinary/advanced policy distinction.

