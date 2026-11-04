# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0047: Bounded general-[scan](../../glossary.md#term-cumulative-scan) [graph](../../glossary.md#graph) [persistence](../../glossary.md#term-persistence)


## Editorial reading context

The application saves ordinary graph source and named current values. Checkpoint encodes one scoped body; Compiler reconstructs it on load. Time steps and run-specific activation history are not serialized copies of that body. Old body-free sources keep their smallest compatible format; successful load still requires separate qualification before a backend can execute the body.

Use the [bounded general-body persistence](../../api/graph-checkpoint-api.md#bounded-general-scan-persistence) for focused guidance. The original status, decision
and evidence below remain historical records; this reading aid does not amend them.

## Status

Accepted decision — 2026-10-23. The user approved selective [graph](../../glossary.md#graph)-[model](../../glossary.md#term-model).v4 and the existing bounded
cold inventory with “ok, pust se do toho”. Main accepted CP0013 [Ready](../../glossary.md#term-ready) and authorized the pre-code
authority/documentation cutover; independent pre-code approval and main acceptance then preceded
implementation. The [codec](../../glossary.md#term-codec) and focused tests are implemented; main accepted final independent review,
[module](../../glossary.md#term-module)/[Javadoc](../../glossary.md#term-javadoc), integration, architecture and configured root gates. CP0013 is Complete.

This record explains the [architecture root](../../../ARCHITECTURE.md), not a separate authority.
It follows [ADR 0046](0046-general-scan-cold-source-and-owned-model-direction.md)<span data-glossary-navigation="true"> ([ADR glossary](../../glossary.md#term-architecture-decision-record-adr))</span>, whose cold-source
and owned-workflow decisions remain unchanged, and extends the graph [persistence](../../glossary.md#term-persistence) direction of
[ADR 0039](0039-portable-graph-model-persistence.md).

## Context

[Compiler](../../glossary.md#term-compiler) now reconstructs bounded general-[scan](../../glossary.md#term-cumulative-scan) bodies, [Checkpoint](../../glossary.md#term-checkpoint) copy/[rebind](../../glossary.md#capture-graph-ports-and-reconstruction) preserves scoped
attachments and [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) supports the all-valid owned compute/first-order training workflow. Disk
save/load rejected bodies independently of those capabilities when this decision was made.
Published [graph](../../glossary.md#graph)-[model](../../glossary.md#term-model) v1/v2/v3
have no general-body field; silently changing v3 would break exact tuple/record compatibility.
Fluent `.rnn` migration cannot retain tested save/load merely because a [general scan](../../glossary.md#general-differentiable-scan) can execute.
That migration also has an independent [backend](../../glossary.md#backend)/[layout](../../glossary.md#layout) compatibility decision; this [ADR](../../glossary.md#term-architecture-decision-record-adr) does not
authorize it.

## Decision drivers

- Persist ordinary source and current raw values through existing APIs and [module](../../glossary.md#term-module) [owners](../../glossary.md#term-owner).
- Preserve exact old bytes/readers and [fail closed](../../glossary.md#admission-preflight-and-fail-closed-validation) before state import or [publication](../../glossary.md#term-publication).
- Match existing constructor-backed cold [admission](../../glossary.md#term-admission) without importing [CPU](../../glossary.md#term-cpu) execution predicates.
- Retain bounded [metadata](../../glossary.md#term-metadata), independent [scope](../../glossary.md#term-scope) [identity](../../glossary.md#term-identity-meanings), [aliases](../../glossary.md#aliasing-and-arena) and all [canonical](../../glossary.md#term-canonical-form) ports.
- Add no new public [API](../../glossary.md#term-application-programming-interface-api)/type, reconstruction layer, execution [owner](../../glossary.md#term-owner) or [runtime](../../glossary.md#term-runtime) history format.

## Options considered

| Option | Consequence |
|---|---|
| Keep body files rejected | Preserves the status quo but cannot deliver the selected [persistence](../../glossary.md#term-persistence) capability. |
| Encode only [RNN](../../glossary.md#term-recurrent-neural-network-rnn)-like bodies | Adds another [admission](../../glossary.md#term-admission) subset despite existing bounded cold constructors; does not address the same general owned-source capability. |
| Mutate v3 or export [unrolled](../../glossary.md#backpropagation-through-time-and-recurrent-state) [graphs](../../glossary.md#graph) | Breaks old exact records or substitutes T-fold [provenance](../../glossary.md#provenance) for the accepted scoped source; rejected. |
| Encode arbitrary [PURE](../../glossary.md#term-pure) bodies | Requires unsupported constructor coverage and broadens architecture beyond the selected inventory; rejected. |
| Selective v4 with the existing bounded cold inventory | Selected: explicit compatibility boundary with no second family reconstruction [owner](../../glossary.md#term-owner). |

The options differ in what can be saved and whether existing source/format boundaries remain intact.

## Decision and rationale

Use ArtifactFile v1 kind [graph](../../glossary.md#graph)-[model](../../glossary.md#term-model).v4 for sources containing [general SCAN](../../glossary.md#general-differentiable-scan), with one inline scoped
body per [occurrence](../../glossary.md#term-occurrence). The [extensions-training owner](../../architecture/contracts/extensions-training.md#canonical-graph-model-version-4)<span data-glossary-navigation="true"> ([owner glossary](../../glossary.md#term-owner))</span>
alone defines all wire fields, guards, bounds and failure order. Outer [admission](../../glossary.md#term-admission) is v3 plus [SCAN](../../glossary.md#term-cumulative-scan);
body admission is exactly [Compiler's existing cold inventory](../../architecture/contracts/compiler-autograd.md#bounded-general-scan-cold-reconstruction).
Model-valid [PURE](../../glossary.md#term-pure) is not automatically reconstructible, [differentiable](../../glossary.md#derivative-and-chain-rule) or executable.

Retain the smallest sufficient byte-identical v1/v2/v3 write for body-free sources and their exact
readers. Old readers reject v4. No selector, [registry](../../glossary.md#term-registry), [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [codec](../../glossary.md#term-codec), implicit cast or exporter fallback
is added. [Compiler](../../glossary.md#term-compiler) preflights and reconstructs ordinary source once; [Checkpoint](../../glossary.md#term-checkpoint) owns wire/bounds/
[host](../../glossary.md#term-host)-value [publication](../../glossary.md#term-publication), and NN consumes the same existing [facade](../../glossary.md#term-facade). [Residual](../../glossary.md#scan-residual-port-and-history-demand) [bindings](../../glossary.md#term-binding) are [metadata](../../glossary.md#term-metadata),
not [activation](../../glossary.md#term-activation)-history payloads or saved [derivative](../../glossary.md#term-derivative) demand.

For example, a three-step `tanh(carry + step * weight)` source stores one inline body and its
named [weight](../../glossary.md#term-weight) values, never three [unrolled](../../glossary.md#backpropagation-through-time-and-recurrent-state) bodies or input [batches](../../glossary.md#term-batch-machine-learning-and-hyperparameters). After load, fresh scoped formals
and ordinary [provenance](../../glossary.md#provenance) must use the restored weight. Compute/train still require actual body and
binder [qualification](../../glossary.md#term-qualification); cold support for a low [dtype](../../glossary.md#data-type--datatype) does not qualify its execution. This is an
implemented cold roundtrip; focused loaded-consumer tests also compare qualified [FLOAT32](../../glossary.md#term-float32)/[FLOAT64](../../glossary.md#term-float64)
compute/training against independent unrolled oracles. Final acceptance still requires main's gates.

## Consequences

### Benefits

One cohesive CP0013 capability covers authority, [codec](../../glossary.md#term-codec), tests/docs and consumer proof without a
mechanical successor cascade. Existing [ownership](../../glossary.md#term-ownership-meanings), APIs, limits, dependencies and stored [raw bits](../../glossary.md#storage-mapping-contiguity-and-injectivity)
remain. Current [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) acquisition documentation can reflect accepted NN0031 without inventing another
workflow. No [Engine](../../glossary.md#term-engine) is created by save/load and no native/[backend](../../glossary.md#backend)/generator change is needed.

### Costs and risks

The [codec](../../glossary.md#term-codec) must validate a larger scoped [schema](../../glossary.md#term-schema), conservative [residual](../../glossary.md#scan-residual-port-and-history-demand) [metadata](../../glossary.md#term-metadata) and aggregate
counts before allocation/import. Existing encoded limits are not peak-[heap](../../glossary.md#term-heap) guarantees. Some valid
[PURE](../../glossary.md#term-pure) bodies remain unsupported; exact cold inventory is intentionally narrower than [Model](../../glossary.md#term-model) validity.
Old readers cannot open v4. Exact [optimizer](../../glossary.md#term-optimizer)/[RNG](../../glossary.md#graph-rng-state--graphrngstate)/cursor continuation, [padding](../../glossary.md#padding) and `.rnn` migration
remain outside this decision.

### [Validation](../../glossary.md#term-validation) and migration

Independent pre-code contract review/main acceptance preceded clean [codec](../../glossary.md#term-codec) implementation.
Focused evidence covers exact old-format bytes, scoped roles/ports, malformed late records,
checked limits and rejection before copying/import or body construction. Consumer tests save before
compute and after training, close the source, cold-load/compose, compute/train fresh [FLOAT32](../../glossary.md#term-float32)/[FLOAT64](../../glossary.md#term-float64)
[batches](../../glossary.md#term-batch-machine-learning-and-hyperparameters) against independent [unrolled](../../glossary.md#backpropagation-through-time-and-recurrent-state) oracles and re-save. These tests followed implementation;
the authority-only cutover ran no executable validation. The [CP0013 brief](../../planning/extensions/checkpoint/tasks/0013-general-scan-graph-persistence-decision.md)
owns evidence and main's accepted final validation results. No fluent builder or [backend](../../glossary.md#backend) migration follows
automatically; published old files remain supported without conversion.

## Related documentation

- [Graph checkpoint API and bounded v4 persistence](../../api/graph-checkpoint-api.md#bounded-general-scan-persistence)<span data-glossary-navigation="true"> ([Graph glossary](../../glossary.md#graph); [checkpoint glossary](../../glossary.md#term-checkpoint); [API glossary](../../glossary.md#term-application-programming-interface-api); [persistence glossary](../../glossary.md#term-persistence))</span>
- [Cold source and scoped persistence explanation](../../architecture/general-differentiable-scan.md#bounded-cold-persistence)
- [Owned general-scan workflow](../../user-guide/running-models.md#train-an-all-valid-general-scan)<span data-glossary-navigation="true"> ([scan glossary](../../glossary.md#term-cumulative-scan))</span>
