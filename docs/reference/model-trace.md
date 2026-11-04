# [Model](../glossary.md#term-model) and diagnostic public boundaries

[Reference goals](index.md) · [Javadoc and exact signatures](javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../glossary.md#term-javadoc); [signatures glossary](../glossary.md#term-signature))</span>

Model describes what a [Tensor](../glossary.md#tensor) value means and how expressions depend on one another. [Trace](../glossary.md#trace)
instead describes observations about work, such as a preparation outcome, without carrying the
Tensor's numeric contents. Its event envelope is a common wrapper for an observation's [identity](../glossary.md#term-identity-meanings),
time and typed details; a [marker interface](../glossary.md#term-marker-interface) labels permitted details but does not emit events.

The implemented `modules:model` surface contains:

- [data type](../glossary.md#data-type--datatype) [metadata](../glossary.md#term-metadata), typed [scalar](../glossary.md#term-scalar) values, and [numeric promotion](../glossary.md#numeric-promotion);
- [BFLOAT16](../glossary.md#term-bfloat16), [FLOAT16](../glossary.md#term-float16) and both [OFP8](../glossary.md#term-ofp8) scalar bit conversions (see [low floating types](../user-guide/tensors.md#low-floating-types-and-raw-bytes));
- static, named dynamic, and expression [dimensions](../glossary.md#dimension), [immutable](../glossary.md#term-immutable) [Shapes](../glossary.md#shape), and local [broadcasting](../glossary.md#broadcasting);
- resolved static [layout](../glossary.md#layout) geometry and [host](../glossary.md#term-host)-storage contracts;
- public [mutable](../glossary.md#term-mutable) `Tensor`, eager leaf factories, explicit-source random construction, and
  backend-independent expression metadata;<span data-glossary-navigation="true"> ([eager glossary](../glossary.md#term-eager-evaluation); [leaf glossary](../glossary.md#term-leaf))</span>
- typed operation attributes and occurrence signatures, shared multi-output producer provenance,
  and operation-specific result carriers; and<span data-glossary-navigation="true"> ([operation glossary](../glossary.md#operation); [occurrence glossary](../glossary.md#term-occurrence); [provenance glossary](../glossary.md#provenance))</span>
- immutable graph values, nodes, compiled graph-model data, forward publication bindings, and
  distinct tensor/node/value identifiers.<span data-glossary-navigation="true"> ([graph glossary](../glossary.md#graph); [nodes glossary](../glossary.md#node); [compiled graph glossary](../glossary.md#compiled-graph--compiledgraphmodel); [publication glossary](../glossary.md#term-publication); [bindings glossary](../glossary.md#term-binding))</span>

The [Tensor API reference](tensor/index.md)<span data-glossary-navigation="true"> ([API glossary](../glossary.md#term-application-programming-interface-api))</span> documents these current contracts, inputs, results,
failures, and examples. The current model surface records meaning and metadata; it does not imply
[compiler](../glossary.md#term-compiler) [capture](../glossary.md#term-capture-meanings), [backend](../glossary.md#backend) support, [kernels](../glossary.md#kernel), [prepared execution](../glossary.md#prepared-execution--preparedexecution), [runtime](../glossary.md#term-runtime) [residency](../glossary.md#residency), or numerical
execution.

The implemented `modules:trace` surface contains:

- [producer](../glossary.md#term-producer)-assigned non-negative [`TraceEventId`](../glossary.md#trace-event-identity--traceeventid) values;
- [`TracePhase`](../glossary.md#trace-phase--tracephase) [lifecycle](../glossary.md#lifecycle) classification for [`COMPILE`](../glossary.md#compile), [`PREPARE`](../glossary.md#prepare), and [`RUN`](../glossary.md#term-run);
- [`TraceLevel`](../glossary.md#trace-level--tracelevel) detail and severity classification;
- the open method-free [`TracePayload`](../glossary.md#trace-payload--tracepayload) marker;
- the generic `TraceEvent<T extends TracePayload>` envelope with a producer-supplied monotonic
  nanosecond reading;
- [nominal](../glossary.md#term-nominal) non-negative [`TraceNodeId`](../glossary.md#trace-node-correlation-identity--tracenodeid), [`TraceValueId`](../glossary.md#trace-value-correlation-identity--tracevalueid), and [`TraceTensorId`](../glossary.md#trace-tensor-correlation-identity--tracetensorid) records for trace-local
  correlation with producer-owned model identities;<span data-glossary-navigation="true"> ([owned model glossary](../glossary.md#model-module-buffer-and-scan-distinctions))</span>
- nominal non-negative [`TraceBackendId`](../glossary.md#trace-backend-correlation-identity--tracebackendid), [`TraceDeviceId`](../glossary.md#trace-device-correlation-identity--tracedeviceid), [`TracePreparedUnitId`](../glossary.md#trace-prepared-unit-correlation-identity--tracepreparedunitid), and
  [`TraceInvocationId`](../glossary.md#trace-invocation-correlation-identity--traceinvocationid) records for producer-stream backend execution correlations;
- the closed `TraceOutcomeStatus`, `TraceRouteKind`, `TraceCacheStatus`,
  and `TraceNativeStatusKind` vocabulary plus exact-code [`TraceNativeStatus`](../glossary.md#trace-native-status--tracenativestatus); and
- immutable `BackendPreparationOutcome` and `BackendInvocationOutcome` payloads.

The outcome payloads [carry](../glossary.md#term-carry) only trace-owned IDs and closed neutral facts. Success requires present
native success; failure permits no native status or a present non-success status. `NOT_QUERIED`
records no [cache](../glossary.md#term-cache) [lookup](../glossary.md#term-lookup) and is not a miss. The [tracing explanation](../architecture/tracing.md)
documents exact component roles, correlation, and [ownership](../glossary.md#term-ownership-meanings) boundaries. The producer owns
correlation allocation, uniqueness, [lifetime](../glossary.md#term-lifetime), mapping, timestamps, and emission. Broader [payload](../glossary.md#artifact-envelope-and-atomic-publication)
families, [partition](../glossary.md#partition)/schedule/run IDs, typed backend attributes, [serialization](../glossary.md#term-serialization), sinks, and shared or
generic emission facilities remain planned. Backend is a producer role, not another lifecycle
phase.


---

[Previous topic](extensions/overview.md) · [Next topic](backend-identities.md)
