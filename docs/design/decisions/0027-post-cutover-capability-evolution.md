# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0027: Post-cutover capability evolution

## Status

Accepted — 2026-10-09, for the capability-evolution rule only. This decision explains the
[authoritative root](../../../ARCHITECTURE.md#core-invariants) and
[backend capability contract](../../architecture/contracts/backend-execution.md#profile-free-backend-capability-and-identity)<span data-glossary-navigation="true"> ([backend capability glossary](../../glossary.md#backend-capability))</span>;
it does not accept Task 0074's route or grant provider support.

## Context

The [Model](../../glossary.md#term-model) 0032 profile-free cutover preserved pre-cutover [`ACCELERATOR`](../../glossary.md#term-accelerator) [CPU](../../glossary.md#term-cpu) and [Metal](../../glossary.md#term-metal) provider
answers. A *[capability ledger](../../glossary.md#capability-ledger)* is a checked-in [snapshot](../../glossary.md#term-snapshot) of actual [boolean](../../glossary.md#data-type--datatype) provider queries over
representative, fully described [operation](../../glossary.md#operation) occurrences. The v1 ledger is historical cutover
evidence, while v2 checks the current profile-free providers. Wording that requires current v2
answers to remain an identical v1 projection would prevent a qualified new route, such as a
[FLOAT32](../../glossary.md#term-float32)-only Metal [`EXP`](../../glossary.md#term-expx), even though the old v1 false answer must remain historically accurate.

## Decision drivers

- Preserve the exact cutover record, including false answers, without freezing future [backend](../../glossary.md#backend) work.
- Keep provider support [occurrence](../../glossary.md#term-occurrence)- and [dtype](../../glossary.md#data-type--datatype)-specific, especially for [BFLOAT16](../../glossary.md#term-bfloat16) and [FLOAT16](../../glossary.md#term-float16).
- Require qualified routes and test evidence before advertising support; keep [Runtime](../../glossary.md#term-runtime) policy-free.
- Reject incompatible prior program, generated, or tuning inputs when behavior changes.

## Options considered

| Option | Benefit | Cost or reason rejected |
|---|---|---|
| Permanently freeze v2 to the v1 [`ACCELERATOR`](../../glossary.md#term-accelerator) projection | Simple equality check | Forbids reviewed post-cutover capability additions. |
| Rewrite v1 when a provider improves | One apparent current ledger | Destroys the historical cutover witness and obscures what changed. |
| Freeze v1; evolve v2 with the reviewed current provider | Preserves history and permits qualified additions | Requires separate current-ledger and [conformance](../../glossary.md#term-conformance) review for each delta. |

The table compares evidence [ownership](../../glossary.md#term-ownership-meanings), not route performance.

## Decision

Keep v1 [immutable](../../glossary.md#term-immutable) as historical cutover evidence. Let v2 represent actual current provider
answers, including both true and false answers and separate [BFLOAT16](../../glossary.md#term-bfloat16)/[FLOAT16](../../glossary.md#term-float16) queries. A later
supported [occurrence](../../glossary.md#term-occurrence) needs an independently true query for its [dtype](../../glossary.md#data-type--datatype), a qualified [backend route](../../glossary.md#backend-route),
updated current-ledger and [conformance](../../glossary.md#term-conformance) tests, and an append-only compatibility [identity](../../glossary.md#term-identity-meanings) when
changed behavior makes earlier inputs incompatible. A new [FLOAT32](../../glossary.md#term-float32) true answer never implies
BFLOAT16 or FLOAT16 support, whether or not the provider uses a FLOAT32 [descriptor](../../glossary.md#term-descriptor) proxy for
existing low-baseline support. Newly admitted low-valued work must have a qualified custom step
and retain the fixed whole-[partition](../../glossary.md#partition) `CUSTOM_PROGRAM` route. Existing supported low answers remain
supported. No provider answer follows merely from a passing [sample](../../glossary.md#term-sample-machine-learning-and-hyperparameters) or from transfer coverage.

## Rationale

The two ledgers answer different questions: v1 says what the cutover preserved; v2 says what the
current provider reports on the same representative basis. A [FLOAT32](../../glossary.md#term-float32) [`EXP`](../../glossary.md#term-expx) addition can therefore
change only that current answer while [BFLOAT16](../../glossary.md#term-bfloat16)/[FLOAT16](../../glossary.md#term-float16) remain false until separately qualified.
The test gate protects [Model](../../glossary.md#term-model) semantics; the version gate protects compatibility. Neither is a
[runtime](../../glossary.md#term-runtime) numerical [certificate](../../glossary.md#formal-proof-certificate-and-trusted-assumptions) or [tolerance](../../glossary.md#term-tolerance) selector.

## Consequences

Future capability work must preserve v1, update v2 from real provider queries, qualify the new
route with ordinary [occurrence](../../glossary.md#term-occurrence)- and [dtype](../../glossary.md#data-type--datatype)-specific [conformance](../../glossary.md#term-conformance) tests, and review affected
[identities](../../glossary.md#term-identity-meanings). Current providers and their ledgers can diverge from historical false answers without
erasing history.
The cost is explicit review of each dtype and any necessary compatibility migration. Unsupported
occurrences still [fail closed](../../glossary.md#admission-preflight-and-fail-closed-validation); [Prepare](../../glossary.md#prepare) fixes routes before shared resource declaration, and
[Runtime](../../glossary.md#term-runtime) gains no [certificate](../../glossary.md#formal-proof-certificate-and-trusted-assumptions), retry, fallback, or policy [lookup](../../glossary.md#term-lookup). This [ADR](../../glossary.md#term-architecture-decision-record-adr) changes no provider,
ledger row, route, native [application binary interface](../../glossary.md#term-application-binary-interface-abi), or [test fixture](../../glossary.md#adr-ci-and-qualification) by itself.

## Related documentation

- [Root capability and low-partition invariants](../../../ARCHITECTURE.md#core-invariants)<span data-glossary-navigation="true"> ([partition glossary](../../glossary.md#partition))</span>
- [Backend capability and identity contract](../../architecture/contracts/backend-execution.md#profile-free-backend-capability-and-identity)<span data-glossary-navigation="true"> ([Backend capability glossary](../../glossary.md#backend-capability); [identity glossary](../../glossary.md#term-identity-meanings))</span>
- [Runtime/Prepare boundary explanation](../../architecture/runtime-prepare-backend-boundary.md#numerical-semantics-at-the-boundary)<span data-glossary-navigation="true"> ([Runtime glossary](../../glossary.md#term-runtime); [Prepare glossary](../../glossary.md#prepare))</span>
- [Task 0074A brief](../../planning/backends/metal/tasks/0074a-post-cutover-capability-evolution-contract.md)
- [Current-provider ledger test](../../../testing/backend-conformance/src/test/java/io/github/pho001/synaptik/testing/conformance/LowPrecisionCapabilityLedgerTest.java)
- [Historical v1 ledger](../../../testing/backend-conformance/src/test/resources/low-precision-capability-ledger-v1.tsv)
- [Current v2 ledger](../../../testing/backend-conformance/src/test/resources/low-precision-capability-ledger-v2.tsv)
