# Model and diagnostic public boundaries

[Reference goals](index.md) · [Javadoc and exact signatures](javadoc.md)

Model describes what a Tensor value means and how expressions depend on one another. Trace
instead describes observations about work, such as a preparation outcome, without carrying the
Tensor's numeric contents. Its event envelope is a common wrapper for an observation's identity,
time and typed details; a marker interface labels permitted details but does not emit events.

The implemented `modules:model` surface contains:

- data type metadata, typed scalar values, and numeric promotion;
- BFLOAT16, FLOAT16 and both OFP8 scalar bit conversions (see [low floating types](../user-guide/tensors.md#low-floating-types-and-raw-bytes));
- static, named dynamic, and expression dimensions, immutable Shapes, and local broadcasting;
- resolved static layout geometry and host-storage contracts;
- public mutable `Tensor`, eager leaf factories, explicit-source random construction, and
  backend-independent expression metadata;
- typed operation attributes and occurrence signatures, shared multi-output producer provenance,
  and operation-specific result carriers; and
- immutable graph values, nodes, compiled graph-model data, forward publication bindings, and
  distinct tensor/node/value identifiers.

The [Tensor API reference](tensor/index.md) documents these current contracts, inputs, results,
failures, and examples. The current model surface records meaning and metadata; it does not imply
compiler capture, backend support, kernels, prepared execution, runtime residency, or numerical
execution.

The implemented `modules:trace` surface contains:

- producer-assigned non-negative `TraceEventId` values;
- `TracePhase` lifecycle classification for `COMPILE`, `PREPARE`, and `RUN`;
- `TraceLevel` detail and severity classification;
- the open method-free `TracePayload` marker;
- the generic `TraceEvent<T extends TracePayload>` envelope with a producer-supplied monotonic
  nanosecond reading;
- nominal non-negative `TraceNodeId`, `TraceValueId`, and `TraceTensorId` records for trace-local
  correlation with producer-owned model identities;
- nominal non-negative `TraceBackendId`, `TraceDeviceId`, `TracePreparedUnitId`, and
  `TraceInvocationId` records for producer-stream backend execution correlations;
- the closed `TraceOutcomeStatus`, `TraceRouteKind`, `TraceCacheStatus`,
  and `TraceNativeStatusKind` vocabulary plus exact-code `TraceNativeStatus`; and
- immutable `BackendPreparationOutcome` and `BackendInvocationOutcome` payloads.

The outcome payloads carry only trace-owned IDs and closed neutral facts. Success requires present
native success; failure permits no native status or a present non-success status. `NOT_QUERIED`
records no cache lookup and is not a miss. The [tracing explanation](../architecture/tracing.md)
documents exact component roles, correlation, and ownership boundaries. The producer owns
correlation allocation, uniqueness, lifetime, mapping, timestamps, and emission. Broader payload
families, partition/schedule/run IDs, typed backend attributes, serialization, sinks, and shared or
generic emission facilities remain planned. Backend is a producer role, not another lifecycle
phase.


---

[Previous topic](extensions/overview.md) · [Next topic](backend-identities.md)
