# [Model](../glossary.md#term-model) and diagnostic public boundaries

[Reference goals](index.md) · [Javadoc and exact signatures](javadoc.md)

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
  backend-independent expression metadata;
- typed operation attributes and occurrence signatures, shared multi-output producer provenance,
  and operation-specific result carriers; and
- immutable graph values, nodes, compiled graph-model data, forward publication bindings, and
  distinct tensor/node/value identifiers.

The [Tensor API reference](tensor/index.md) documents these current contracts, inputs, results,
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
  correlation with producer-owned model identities;
- nominal non-negative [`TraceBackendId`](../glossary.md#trace-backend-correlation-identity--tracebackendid), [`TraceDeviceId`](../glossary.md#trace-device-correlation-identity--tracedeviceid), [`TracePreparedUnitId`](../glossary.md#trace-prepared-unit-correlation-identity--tracepreparedunitid), and
  [`TraceInvocationId`](../glossary.md#trace-invocation-correlation-identity--traceinvocationid) records for producer-stream backend execution correlations;
- the closed `TraceOutcomeStatus`, `TraceRouteKind`, `TraceCacheStatus`,
  and `TraceNativeStatusKind` vocabulary plus exact-code [`TraceNativeStatus`](../glossary.md#trace-native-status--tracenativestatus); and
- immutable `BackendPreparationOutcome` and `BackendInvocationOutcome` payloads.

The outcome payloads carry only trace-owned IDs and closed neutral facts. Success requires present
native success; failure permits no native status or a present non-success status. `NOT_QUERIED`
records no [cache](../glossary.md#term-cache) [lookup](../glossary.md#term-lookup) and is not a miss. The [tracing explanation](../architecture/tracing.md)
documents exact component roles, correlation, and [ownership](../glossary.md#term-ownership-meanings) boundaries. The producer owns
correlation allocation, uniqueness, [lifetime](../glossary.md#term-lifetime), mapping, timestamps, and emission. Broader [payload](../glossary.md#trace-payload--tracepayload)
families, [partition](../glossary.md#partition)/schedule/run IDs, typed backend attributes, [serialization](../glossary.md#term-serialization), sinks, and shared or
generic emission facilities remain planned. Backend is a producer role, not another lifecycle
phase.


---

[Previous topic](extensions/overview.md) · [Next topic](backend-identities.md)
## An expression producer is not a diagnostic event producer

A Tensor producer is the ordinary expression object that describes one computation occurrence.
A trace producer is whichever component emits observations and assigns event IDs. The Metal
integration, for example, calls the application's observer with bounded outcome data; it does
not hand over Tensor storage or native-resource ownership. Closing the observer is not Engine's
cleanup responsibility. Keep [expression capture](compiler/capture.md) separate from
[Metal observer callbacks](engine/composition.md#current-explicit-cpu-and-metal-composition): the
first describes requested work, while the second observes preparation/execution outcomes.

