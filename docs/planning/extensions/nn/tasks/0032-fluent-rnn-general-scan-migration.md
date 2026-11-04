# Task NN0032: Fluent [RNN](../../../../glossary.md#term-recurrent-neural-network-rnn) general-[scan](../../../../glossary.md#term-cumulative-scan) migration compatibility decision

## Status

Draft — **blocked on a confirmed advanced second-order [CPU](../../../../glossary.md#term-cpu) [compile](../../../../glossary.md#compile) [regression](../../../../glossary.md#term-regression)**. Implementation frontier:
**None**. MAIN paused authority/cutover edits on 2026-10-24; do not mark [Ready](../../../../glossary.md#term-ready), launch Java work or
redirect `.rnn(hiddenSize)`. No authority or implementation change has been made by this assessment.

The subsequently authorized [Model0036](../../../modules/model/tasks/0036-length-aware-general-scan-design.md)
is Complete for [length-aware Scan proposal delivery](../../../length-aware-general-scan.md)<span data-glossary-navigation="true"> ([Scan glossary](../../../../glossary.md#term-cumulative-scan))</span> only,
after independent final APPROVE and MAIN acceptance; no [planning](../../../../glossary.md#term-planning)/source/authority frontier follows.
Its new row/sequence [binding](../../../../glossary.md#term-binding) [metadata](../../../../glossary.md#term-metadata), length overload and v5 remain PROPOSED; it neither resumes
this migration nor resolves/waives the second-order compile, CONCAT-copy or mixed-[engine](../../../../glossary.md#term-engine) gates.

## Change class

Class C — changing fluent recurrence affects observable [provenance](../../../../glossary.md#provenance), [autograd](../../../../glossary.md#autograd) [admission](../../../../glossary.md#term-admission),
[persistence](../../../../glossary.md#term-persistence) and recurrent [lifecycle](../../../../glossary.md#lifecycle)/hot-path compatibility. This context is factual [planning](../../../../glossary.md#term-planning) only;
any resumed implementation requires a clean executor and independent code/documentation review.

## Goal and preserved [scope](../../../../glossary.md#term-scope)

The user approved proceeding with an [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding) ([FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64)) fluent [RNN](../../../../glossary.md#term-recurrent-neural-network-rnn) migration using existing
[general Scan](../../../../glossary.md#general-differentiable-scan) concepts, **while preserving existing [API](../../../../glossary.md#term-application-programming-interface-api), state and supported capabilities**.
That preservation condition is not a first-order-only waiver. Current ordinary `.rnn` remains
unchanged pending the user's scope decision below. No approval to lose advanced second-order
[compile](../../../../glossary.md#compile) capability or to extend [Scan](../../../../glossary.md#term-cumulative-scan) second order is assumed.

The original pause assessment updated only this factual brief. Source/tests and [normative](../../../../glossary.md#term-normative) contracts
are read-only. No new production type, wrapper, API, execution/[AD](../../../../glossary.md#autograd) layer, provider inspection,
request-sensitive retry, [dtype](../../../../glossary.md#data-type--datatype) conversion, fallback or auxiliary execution [DAG](../../../../glossary.md#term-directed-acyclic-graph-dag) is authorized.
No [commit](../../../../glossary.md#term-commit), push, successor, Java execution, [Gradle](../../../../glossary.md#term-gradle) [run](../../../../glossary.md#term-run) or performance campaign belongs to this pass.

## Contracts

Read the root [scope](../../../../glossary.md#term-scope) index, then the exact applicable headings; [planning](../../../../glossary.md#term-planning) cannot override them:

- [Root](../../../../../ARCHITECTURE.md): `Authority, incorporation, and precedence`,
  `Core lifecycle`, `Core invariants`, `Scope-indexed normative contracts`, `Dependency rules`,
  `Testing requirements`, `Documentation authority and architecture updates`.
- [Extensions/training](../../../../architecture/contracts/extensions-training.md#extensionsnn):
  `extensions/nn`, [Owned tensor Models](../../../../architecture/contracts/extensions-training.md#owned-tensor-models)<span data-glossary-navigation="true"> ([tensor glossary](../../../../glossary.md#tensor))</span>,
  [Generic graph source and values](../../../../architecture/contracts/extensions-training.md#generic-graph-source-and-values)<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>,
  [Canonical graph-model version 3](../../../../architecture/contracts/extensions-training.md#canonical-graph-model-version-3)<span data-glossary-navigation="true"> ([Canonical glossary](../../../../glossary.md#term-canonical-form); [model glossary](../../../../glossary.md#term-model))</span>
  and [version 4](../../../../architecture/contracts/extensions-training.md#canonical-graph-model-version-4).
- [Recurrent scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan)<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan))</span>:
  `Performance and migration invariants`, [`General differentiable scan`](../../../../glossary.md#general-differentiable-scan),
  `Source body, attachments and scoped identities`, `Canonical residual ports and demand`,
  `Reverse scan and exact primal bindings`, `Traversal, empty domains and declared qualification`,
  `Qualification, performance and migration gates`.
- [Compiler](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>:
  `Compile lifecycle`, `Bounded general-scan cold reconstruction`; the existing bounded
  higher-order request contract does not grant general Scan higher order.<span data-glossary-navigation="true"> ([general Scan glossary](../../../../glossary.md#general-differentiable-scan))</span>

Root + owning extensions/training and the recurrent [owner](../../../../glossary.md#term-owner)'s migration gate would need coordinated
review before a cutover. Those edits are frozen, not enacted. No [module](../../../../glossary.md#term-module)-edge rule, native [ABI](../../../../glossary.md#term-application-binary-interface-abi),
wire [schema](../../../../glossary.md#term-schema) or generator change is proposed; no architecture-test change is required for this
factual documentation update. Stop rather than invent an interpretation of conflicting contracts.

## Dependencies and integration

- Depends on: NN0029, NN0030/0031, Compiler0012, CP0010–0013, CPU0030–0037 and accepted low-level
  join — Complete; their Result/status/review evidence was checked, not rerun.
- Conflicts with: fluent recurrence, NN initialization/ownership, general-body autograd/wire,
  scan capability, root/scoped authority and shared planning documents; serialize these scopes.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [autograd glossary](../../../../glossary.md#autograd); [scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- Parallel group: None.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus all
  295 accepted dirty paths; empty index. Accepted baseline lists 2423 files at
  `/private/tmp/synaptik-nn0032.VvJDWe/baseline.json`, archive `accepted.tar` beside it.
- Integration order: factual confirmed-blocker handoff → explicit user scope decision coordinated
  by MAIN → independently reviewed authority/Ready packet only after the conflict is resolved.<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff); [scope glossary](../../../../glossary.md#term-scope))</span>
- Integration validation: documentation link/size/whitespace checks now; no executable checkpoint
  authorized while blocked. Any resumed packet must name worker/module and MAIN integration gates.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [module glossary](../../../../glossary.md#term-module))</span>
- Shared-document integration owner: MAIN; this executor is sole writer of this brief during the
  pause. No other documentation or normative edits; no concurrent code writer.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [normative glossary](../../../../glossary.md#term-normative))</span>

Verification 2026-10-24: current user approval includes preservation; the later explicit pause
holds cutover. HEAD/origin and empty index were checked. MAIN reports full archive extraction
matched all 2423 baseline [hashes](../../../../glossary.md#term-hash), with zero mismatch. Declared code conflicts are inactive;
independent source review identified the blocker now confirmed by MAIN's [compile](../../../../glossary.md#compile)-only probe below.
Complete prerequisites do not satisfy compatibility: **[Ready](../../../../glossary.md#term-ready)/frontier verification is not satisfied**. MAIN must accept
final independent [planning](../../../../glossary.md#term-planning) review before any eventual code launch.

## Confirmed facts and evidence limits

### Advanced second-order [CPU](../../../../glossary.md#term-cpu) [compile](../../../../glossary.md#compile) [regression](../../../../glossary.md#term-regression) is confirmed

Current `ModelBuilder.RnnStep.forward` calls `RnnSequence.forward` and packs its ordinary
[MATMUL](../../../../glossary.md#matrix-multiplication--matmul)/ADD/[TANH](../../../../glossary.md#term-tanh) expressions with CONCAT. It inserts no [general Scan](../../../../glossary.md#general-differentiable-scan).
The [general-scan contract](../../../../architecture/contracts/recurrent-scan.md#reverse-scan-and-exact-primal-bindings)
explicitly rejects every [scan](../../../../glossary.md#term-cumulative-scan) higher-order/createGraph request before [derivative](../../../../glossary.md#term-derivative) allocation.
Independent source review additionally reports the two-stage rejection at
`modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradPreflight.java:234`.

MAIN's tiny Java diagnostic confirms an **old-pass/new-reject advanced CPU compile capability**,
not merely a source-level risk. Evidence, both read by this documentation executor:

- Source: `/private/tmp/synaptik-nn0032.VvJDWe/RnnHigherOrderProbe.java`.
- Output: `/private/tmp/synaptik-nn0032.VvJDWe/higher-order-probe.txt`.
- Final reproducible evidence: `/private/tmp/synaptik-nn0032.VvJDWe/higher-order-probe-final.json`;
  exit 0 and the same results for both dtypes. Its classpath uses only CURRENT settings projectDirs,
  excluding obsolete relocated Training/Checkpoint classes; Java 26 source launch uses incubator Vector.<span data-glossary-navigation="true"> ([classpath glossary](../../../../glossary.md#term-classpath); [Checkpoint glossary](../../../../glossary.md#term-checkpoint); [Vector glossary](../../../../glossary.md#term-vector))</span>

MAIN's final reproducible diagnostic command, recorded here but not rerun by this executor:

```bash
node /private/tmp/synaptik-nn0032.VvJDWe/run-higher-order-probe.cjs
```

For each [dtype](../../../../glossary.md#data-type--datatype) it builds actual `Model.builder().input(type,2,1,1).seed(5).rnn(1)`, giving
`[T,B,F,H]=[2,1,1,1]`. The old root is `model.forward(x)`; the candidate uses exactly the same
owned W/U/b in an explicit general Scan plus [singleton](../../../../glossary.md#term-singleton) CONCAT, with non-[gradient](../../../../glossary.md#term-gradient) zero input and
initial h. Each [objective](../../../../glossary.md#term-objective) is the corresponding root's `sum()`. The public
`GraphCompilationPort.compile` uses actual `CpuCapabilityProvider` and explicit `cpu/host` CPU
availability, standard [graph optimization](../../../../glossary.md#forward-graph-optimization) and neutral scoring, against current already-built classes.

The first stage differentiates the objective with respect to inputWeight. For the two-stage
request it sets `createGraph=true`, then differentiates that first-stage gradient with respect to
the same inputWeight using a typed-one [cotangent](../../../../glossary.md#cotangent); both stages use disconnected policy ERROR.
The table reports [compiled graph](../../../../glossary.md#compiled-graph--compiledgraphmodel) [node](../../../../glossary.md#node) counts, not timing or numerical derivative results.

| Dtype | Old first order | Scan + CONCAT first order | Old two-stage | Scan + CONCAT two-stage |
|---|---|---|---|---|
| [FLOAT32](../../../../glossary.md#term-float32) | PASS, 54 nodes | PASS, 12 nodes | PASS, 100 nodes; derivative order 2 present | REJECT, first-order only |
| [FLOAT64](../../../../glossary.md#term-float64) | PASS, 54 nodes | PASS, 12 nodes | PASS, 100 nodes; derivative order 2 present | REJECT, first-order only |

Both new two-stage requests throw `IllegalArgumentException` with
`general scan supports first-order only; createGraph is unsupported`. No preparation, execution
or numerical [Hessian](../../../../glossary.md#term-hessian) [validation](../../../../glossary.md#term-validation) was performed. This proves the bounded advanced CPU **compile**
regression, not [runtime](../../../../glossary.md#term-runtime) Hessian correctness, a timing comparison, or every old higher-order regime.
Ordinary [Model](../../../../glossary.md#term-model).configureTraining being first-order does not waive this capability: cold
`Model.forward` exposes expressions to callers' supported [Compiler](../../../../glossary.md#term-compiler)/[Engine](../../../../glossary.md#term-engine) requests.

CP0013 proves bounded [persistence](../../../../glossary.md#term-persistence); NN0031 and CPU0037 prove their recorded first-order workflows.
None qualifies Scan higher order. CPU0037's actual-reverse gradients are not second derivatives.
Replacing this confirmed old supported compile request with the present Scan source loses that
capability; singleton CONCAT does not hide the contained Scan from [autograd](../../../../glossary.md#autograd) inventory.

### [Singleton](../../../../glossary.md#term-singleton) CONCAT preserves the cold [descriptor](../../../../glossary.md#term-descriptor), but copies

The existing-[operation](../../../../glossary.md#operation) descriptor candidate is
`Tensor.concat(0, scan.outputs().getFirst())`: one [axis](../../../../glossary.md#term-axis)-zero, [rank](../../../../glossary.md#term-rank)-three CONCAT around the
stacked emission, not a new wrapper/type/[API](../../../../glossary.md#term-application-programming-interface-api). `TensorCompositionExpressions.concatChecked` returns
a fresh unresolved-[layout](../../../../glossary.md#layout) descriptor with the same [dtype](../../../../glossary.md#data-type--datatype)/[shape](../../../../glossary.md#shape)/[gradient](../../../../glossary.md#term-gradient) OR, whereas raw [Scan](../../../../glossary.md#term-cumulative-scan)
emission is [canonical](../../../../glossary.md#term-canonical-form). This preserves the old cold `[T,B,H]` output descriptor, including layout
absence, and therefore the [metadata](../../../../glossary.md#term-metadata) expected by exact-descriptor downstream substitutions.

`OwnedModelSource.acquireComposition/inventory/forward` retain the fixed owned source, traverse
outer actual inputs rather than body formals, and accept exact-descriptor produced substitutions
without cold upstream [capture](../../../../glossary.md#term-capture-meanings). W/U/b must remain actual [invariant](../../../../glossary.md#term-invariant) [bindings](../../../../glossary.md#term-binding); canonical storage-free
body formals must not become named state/[constants](../../../../glossary.md#term-constant). No new source/binding representation is needed.

This CONCAT is **not a no-copy [alias](../../../../glossary.md#aliasing-and-arena)** in the current pipeline. Existing
`CpuNonAffineMovementLowering` lowers it as a movement unit and
`CpuDataMovementEmitter.emitBoundedConcat` loads and stores the output. Relative to publishing raw
Scan emission, it adds one full-output movement pass over `T*B*H` represented [elements](../../../../glossary.md#term-element), plus the
ordinary output [materialization](../../../../glossary.md#materialization); exact peak physical memory is not asserted here.
Do not add [CONTIGUOUS](../../../../glossary.md#term-contiguous), normalize [Tensor](../../../../glossary.md#tensor) descriptors, invent a [compiler](../../../../glossary.md#term-compiler) bypass, or claim that this
pass disappears. CPU0037 qualifies its dense-[view](../../../../glossary.md#term-view-meanings) reverse [seed](../../../../glossary.md#term-seed) numerically; it does not remove
the forward copy or resolve higher-order compatibility.

## User [scope](../../../../glossary.md#term-scope) decision required

MAIN must obtain the user's choice: accept [loss](../../../../glossary.md#loss) of this advanced second-order [compile](../../../../glossary.md#compile) capability,
or separately qualify [Scan](../../../../glossary.md#term-cumulative-scan) second order before migrating fluent [RNN](../../../../glossary.md#term-recurrent-neural-network-rnn). Neither choice is assumed
approved. Keep current fluent [unrolling](../../../../glossary.md#term-unrolling) while this decision is outstanding. Extending Scan higher
order requires separate architecture/implementation authorization, not hidden NN0032 work.
Do not infer a first-order-only waiver, silently narrow support, or create a speculative successor.
No `.rnn` authority cutover or executable [Ready](../../../../glossary.md#term-ready) packet is possible under the current unresolved grant.

Any later preservation packet must still cover initialization without a T-step bootstrap [graph](../../../../glossary.md#graph);
W/U/b paths/order/[seed](../../../../glossary.md#term-seed) bytes/wrappers/[ownership](../../../../glossary.md#term-ownership-meanings); all-valid positive static T/B/F; non-square outputs,
actual [gradients](../../../../glossary.md#term-gradient) and repeated [SGD](../../../../glossary.md#term-stochastic-gradient-descent-sgd); stacks, Conv1d, heads and lastTimeStep; cold substitutions,
save/load/current-[weight](../../../../glossary.md#term-weight) re-save; preparation reuse and O(body) object/resource structure.
Preserve public RnnSequence/compact outputs, [FP16](../../../../glossary.md#term-fp16)/[BF16](../../../../glossary.md#floating-point-precision-and-rounding) static behavior, fixed [runtime](../../../../glossary.md#term-runtime)-length APIs
and [GRU](../../../../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../../../../glossary.md#term-long-short-term-memory-lstm). Mixed [CPU](../../../../glossary.md#term-cpu)/[Metal](../../../../glossary.md#term-metal) compatibility requires truthful focused old/new [validation](../../../../glossary.md#term-validation); earlier
Metal-only old/new rejections do not establish mixed-[engine](../../../../glossary.md#term-engine) parity. No new performance claim follows.

## Documentation, review and [planning](../../../../glossary.md#term-planning) result

### Historical pause-assessment result (before Model0036)

The completed pause-assessment pass changed only this brief; its evidence below is historical.
The later Model0036 [planning](../../../../glossary.md#term-planning) pass adds only the proposal cross-link/status above, not a gate waiver.
No [normative](../../../../glossary.md#term-normative), explanatory, source, test or [Javadoc](../../../../glossary.md#term-javadoc) edit was applied by either planning pass.
ADR0049 was free when checked; it was not created or reserved. Targeted glossary search found no
new term meaning requiring an edit. Planning/general style applies to this factual [handoff](../../../../glossary.md#term-handoff).
This documentation executor ran no Java/[Gradle](../../../../glossary.md#term-gradle)/benchmark/native command; MAIN's supplied [compile](../../../../glossary.md#compile)-only
probe/result is reused above. Independent clean final factual planning review: APPROVE (2026-10-24),
no findings; the frozen brief/diff, probe source/runner/[JSON](../../../../glossary.md#term-json) and current rejection contract agree.
This approves the documented disposition only, not migration. The explicit user [scope](../../../../glossary.md#term-scope) decision
and remaining compatibility gates remain outstanding. No new task or [ADR](../../../../glossary.md#term-architecture-decision-record-adr) is authorized or created.
Migration status remains Draft/blocked, not implementation Complete or [Ready](../../../../glossary.md#term-ready).
Documentation hygiene: scoped `git diff --check` passed; targeted local check passed all 9 links
and 8 fragments, with no trailing whitespace. Brief is within 200 lines/15 KB.
