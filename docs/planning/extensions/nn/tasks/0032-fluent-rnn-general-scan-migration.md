# Task NN0032: Fluent RNN general-scan migration compatibility decision

## Status

Draft — **blocked on a confirmed advanced second-order CPU compile regression**. Implementation frontier:
**None**. MAIN paused authority/cutover edits on 2026-10-24; do not mark Ready, launch Java work or
redirect `.rnn(hiddenSize)`. No authority or implementation change has been made by this assessment.

The subsequently authorized [Model0036](../../../modules/model/tasks/0036-length-aware-general-scan-design.md)
is Complete for [length-aware Scan proposal delivery](../../../length-aware-general-scan.md) only,
after independent final APPROVE and MAIN acceptance; no planning/source/authority frontier follows.
Its new row/sequence binding metadata, length overload and v5 remain PROPOSED; it neither resumes
this migration nor resolves/waives the second-order compile, CONCAT-copy or mixed-engine gates.

## Change class

Class C — changing fluent recurrence affects observable provenance, autograd admission,
persistence and recurrent lifecycle/hot-path compatibility. This context is factual planning only;
any resumed implementation requires a clean executor and independent code/documentation review.

## Goal and preserved scope

The user approved proceeding with an F32/F64 (FLOAT32/FLOAT64) fluent RNN migration using existing
general Scan concepts, **while preserving existing API, state and supported capabilities**.
That preservation condition is not a first-order-only waiver. Current ordinary `.rnn` remains
unchanged pending the user's scope decision below. No approval to lose advanced second-order
compile capability or to extend Scan second order is assumed.

The original pause assessment updated only this factual brief. Source/tests and normative contracts
are read-only. No new production type, wrapper, API, execution/AD layer, provider inspection,
request-sensitive retry, dtype conversion, fallback or auxiliary execution DAG is authorized.
No commit, push, successor, Java execution, Gradle run or performance campaign belongs to this pass.

## Contracts

Read the root scope index, then the exact applicable headings; planning cannot override them:

- [Root](../../../../../ARCHITECTURE.md): `Authority, incorporation, and precedence`,
  `Core lifecycle`, `Core invariants`, `Scope-indexed normative contracts`, `Dependency rules`,
  `Testing requirements`, `Documentation authority and architecture updates`.
- [Extensions/training](../../../../architecture/contracts/extensions-training.md#extensionsnn):
  `extensions/nn`, [Owned tensor Models](../../../../architecture/contracts/extensions-training.md#owned-tensor-models),
  [Generic graph source and values](../../../../architecture/contracts/extensions-training.md#generic-graph-source-and-values),
  [Canonical graph-model version 3](../../../../architecture/contracts/extensions-training.md#canonical-graph-model-version-3)
  and [version 4](../../../../architecture/contracts/extensions-training.md#canonical-graph-model-version-4).
- [Recurrent scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan):
  `Performance and migration invariants`, `General differentiable scan`,
  `Source body, attachments and scoped identities`, `Canonical residual ports and demand`,
  `Reverse scan and exact primal bindings`, `Traversal, empty domains and declared qualification`,
  `Qualification, performance and migration gates`.
- [Compiler](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle):
  `Compile lifecycle`, `Bounded general-scan cold reconstruction`; the existing bounded
  higher-order request contract does not grant general Scan higher order.

Root + owning extensions/training and the recurrent owner's migration gate would need coordinated
review before a cutover. Those edits are frozen, not enacted. No module-edge rule, native ABI,
wire schema or generator change is proposed; no architecture-test change is required for this
factual documentation update. Stop rather than invent an interpretation of conflicting contracts.

## Dependencies and integration

- Depends on: NN0029, NN0030/0031, Compiler0012, CP0010–0013, CPU0030–0037 and accepted low-level
  join — Complete; their Result/status/review evidence was checked, not rerun.
- Conflicts with: fluent recurrence, NN initialization/ownership, general-body autograd/wire,
  scan capability, root/scoped authority and shared planning documents; serialize these scopes.
- Parallel group: None.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus all
  295 accepted dirty paths; empty index. Accepted baseline lists 2423 files at
  `/private/tmp/synaptik-nn0032.VvJDWe/baseline.json`, archive `accepted.tar` beside it.
- Integration order: factual confirmed-blocker handoff → explicit user scope decision coordinated
  by MAIN → independently reviewed authority/Ready packet only after the conflict is resolved.
- Integration validation: documentation link/size/whitespace checks now; no executable checkpoint
  authorized while blocked. Any resumed packet must name worker/module and MAIN integration gates.
- Shared-document integration owner: MAIN; this executor is sole writer of this brief during the
  pause. No other documentation or normative edits; no concurrent code writer.

Verification 2026-10-24: current user approval includes preservation; the later explicit pause
holds cutover. HEAD/origin and empty index were checked. MAIN reports full archive extraction
matched all 2423 baseline hashes, with zero mismatch. Declared code conflicts are inactive;
independent source review identified the blocker now confirmed by MAIN's compile-only probe below.
Complete prerequisites do not satisfy compatibility: **Ready/frontier verification is not satisfied**. MAIN must accept
final independent planning review before any eventual code launch.

## Confirmed facts and evidence limits

### Advanced second-order CPU compile regression is confirmed

Current `ModelBuilder.RnnStep.forward` calls `RnnSequence.forward` and packs its ordinary
MATMUL/ADD/TANH expressions with CONCAT. It inserts no general Scan.
The [general-scan contract](../../../../architecture/contracts/recurrent-scan.md#reverse-scan-and-exact-primal-bindings)
explicitly rejects every scan higher-order/createGraph request before derivative allocation.
Independent source review additionally reports the two-stage rejection at
`modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradPreflight.java:234`.

MAIN's tiny Java diagnostic confirms an **old-pass/new-reject advanced CPU compile capability**,
not merely a source-level risk. Evidence, both read by this documentation executor:

- Source: `/private/tmp/synaptik-nn0032.VvJDWe/RnnHigherOrderProbe.java`.
- Output: `/private/tmp/synaptik-nn0032.VvJDWe/higher-order-probe.txt`.
- Final reproducible evidence: `/private/tmp/synaptik-nn0032.VvJDWe/higher-order-probe-final.json`;
  exit 0 and the same results for both dtypes. Its classpath uses only CURRENT settings projectDirs,
  excluding obsolete relocated Training/Checkpoint classes; Java 26 source launch uses incubator Vector.

MAIN's final reproducible diagnostic command, recorded here but not rerun by this executor:

```bash
node /private/tmp/synaptik-nn0032.VvJDWe/run-higher-order-probe.cjs
```

For each dtype it builds actual `Model.builder().input(type,2,1,1).seed(5).rnn(1)`, giving
`[T,B,F,H]=[2,1,1,1]`. The old root is `model.forward(x)`; the candidate uses exactly the same
owned W/U/b in an explicit general Scan plus singleton CONCAT, with non-gradient zero input and
initial h. Each objective is the corresponding root's `sum()`. The public
`GraphCompilationPort.compile` uses actual `CpuCapabilityProvider` and explicit `cpu/host` CPU
availability, standard graph optimization and neutral scoring, against current already-built classes.

The first stage differentiates the objective with respect to inputWeight. For the two-stage
request it sets `createGraph=true`, then differentiates that first-stage gradient with respect to
the same inputWeight using a typed-one cotangent; both stages use disconnected policy ERROR.
The table reports compiled graph node counts, not timing or numerical derivative results.

| Dtype | Old first order | Scan + CONCAT first order | Old two-stage | Scan + CONCAT two-stage |
|---|---|---|---|---|
| FLOAT32 | PASS, 54 nodes | PASS, 12 nodes | PASS, 100 nodes; derivative order 2 present | REJECT, first-order only |
| FLOAT64 | PASS, 54 nodes | PASS, 12 nodes | PASS, 100 nodes; derivative order 2 present | REJECT, first-order only |

Both new two-stage requests throw `IllegalArgumentException` with
`general scan supports first-order only; createGraph is unsupported`. No preparation, execution
or numerical Hessian validation was performed. This proves the bounded advanced CPU **compile**
regression, not runtime Hessian correctness, a timing comparison, or every old higher-order regime.
Ordinary Model.configureTraining being first-order does not waive this capability: cold
`Model.forward` exposes expressions to callers' supported Compiler/Engine requests.

CP0013 proves bounded persistence; NN0031 and CPU0037 prove their recorded first-order workflows.
None qualifies Scan higher order. CPU0037's actual-reverse gradients are not second derivatives.
Replacing this confirmed old supported compile request with the present Scan source loses that
capability; singleton CONCAT does not hide the contained Scan from autograd inventory.

### Singleton CONCAT preserves the cold descriptor, but copies

The existing-operation descriptor candidate is
`Tensor.concat(0, scan.outputs().getFirst())`: one axis-zero, rank-three CONCAT around the
stacked emission, not a new wrapper/type/API. `TensorCompositionExpressions.concatChecked` returns
a fresh unresolved-layout descriptor with the same dtype/shape/gradient OR, whereas raw Scan
emission is canonical. This preserves the old cold `[T,B,H]` output descriptor, including layout
absence, and therefore the metadata expected by exact-descriptor downstream substitutions.

`OwnedModelSource.acquireComposition/inventory/forward` retain the fixed owned source, traverse
outer actual inputs rather than body formals, and accept exact-descriptor produced substitutions
without cold upstream capture. W/U/b must remain actual invariant bindings; canonical storage-free
body formals must not become named state/constants. No new source/binding representation is needed.

This CONCAT is **not a no-copy alias** in the current pipeline. Existing
`CpuNonAffineMovementLowering` lowers it as a movement unit and
`CpuDataMovementEmitter.emitBoundedConcat` loads and stores the output. Relative to publishing raw
Scan emission, it adds one full-output movement pass over `T*B*H` represented elements, plus the
ordinary output materialization; exact peak physical memory is not asserted here.
Do not add CONTIGUOUS, normalize Tensor descriptors, invent a compiler bypass, or claim that this
pass disappears. CPU0037 qualifies its dense-view reverse seed numerically; it does not remove
the forward copy or resolve higher-order compatibility.

## User scope decision required

MAIN must obtain the user's choice: accept loss of this advanced second-order compile capability,
or separately qualify Scan second order before migrating fluent RNN. Neither choice is assumed
approved. Keep current fluent unrolling while this decision is outstanding. Extending Scan higher
order requires separate architecture/implementation authorization, not hidden NN0032 work.
Do not infer a first-order-only waiver, silently narrow support, or create a speculative successor.
No `.rnn` authority cutover or executable Ready packet is possible under the current unresolved grant.

Any later preservation packet must still cover initialization without a T-step bootstrap graph;
W/U/b paths/order/seed bytes/wrappers/ownership; all-valid positive static T/B/F; non-square outputs,
actual gradients and repeated SGD; stacks, Conv1d, heads and lastTimeStep; cold substitutions,
save/load/current-weight re-save; preparation reuse and O(body) object/resource structure.
Preserve public RnnSequence/compact outputs, FP16/BF16 static behavior, fixed runtime-length APIs
and GRU/LSTM. Mixed CPU/Metal compatibility requires truthful focused old/new validation; earlier
Metal-only old/new rejections do not establish mixed-engine parity. No new performance claim follows.

## Documentation, review and planning result

### Historical pause-assessment result (before Model0036)

The completed pause-assessment pass changed only this brief; its evidence below is historical.
The later Model0036 planning pass adds only the proposal cross-link/status above, not a gate waiver.
No normative, explanatory, source, test or Javadoc edit was applied by either planning pass.
ADR0049 was free when checked; it was not created or reserved. Targeted glossary search found no
new term meaning requiring an edit. Planning/general style applies to this factual handoff.
This documentation executor ran no Java/Gradle/benchmark/native command; MAIN's supplied compile-only
probe/result is reused above. Independent clean final factual planning review: APPROVE (2026-10-24),
no findings; the frozen brief/diff, probe source/runner/JSON and current rejection contract agree.
This approves the documented disposition only, not migration. The explicit user scope decision
and remaining compatibility gates remain outstanding. No new task or ADR is authorized or created.
Migration status remains Draft/blocked, not implementation Complete or Ready.
Documentation hygiene: scoped `git diff --check` passed; targeted local check passed all 9 links
and 8 fragments, with no trailing whitespace. Brief is within 200 lines/15 KB.
