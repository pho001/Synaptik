# ADR 0027: Post-cutover capability evolution

## Status

Accepted — 2026-10-09, for the capability-evolution rule only. This decision explains the
[authoritative root](../../../ARCHITECTURE.md#core-invariants) and
[backend capability contract](../../architecture/contracts/backend-execution.md#profile-free-backend-capability-and-identity);
it does not accept Task 0074's route or grant provider support.

## Context

The Model 0032 profile-free cutover preserved pre-cutover `ACCELERATOR` CPU and Metal provider
answers. A *capability ledger* is a checked-in snapshot of actual boolean provider queries over
representative, fully described operation occurrences. The v1 ledger is historical cutover
evidence, while v2 checks the current profile-free providers. Wording that requires current v2
answers to remain an identical v1 projection would prevent a qualified new route, such as a
FLOAT32-only Metal `EXP`, even though the old v1 false answer must remain historically accurate.

## Decision drivers

- Preserve the exact cutover record, including false answers, without freezing future backend work.
- Keep provider support occurrence- and dtype-specific, especially for BFLOAT16 and FLOAT16.
- Require qualified routes and test evidence before advertising support; keep Runtime policy-free.
- Reject incompatible prior program, generated, or tuning inputs when behavior changes.

## Options considered

| Option | Benefit | Cost or reason rejected |
|---|---|---|
| Permanently freeze v2 to the v1 `ACCELERATOR` projection | Simple equality check | Forbids reviewed post-cutover capability additions. |
| Rewrite v1 when a provider improves | One apparent current ledger | Destroys the historical cutover witness and obscures what changed. |
| Freeze v1; evolve v2 with the reviewed current provider | Preserves history and permits qualified additions | Requires separate current-ledger and conformance review for each delta. |

The table compares evidence ownership, not route performance.

## Decision

Keep v1 immutable as historical cutover evidence. Let v2 represent actual current provider
answers, including both true and false answers and separate BFLOAT16/FLOAT16 queries. A later
supported occurrence needs an independently true query for its dtype, a qualified backend route,
updated current-ledger and conformance tests, and an append-only compatibility identity when
changed behavior makes earlier inputs incompatible. A new FLOAT32 true answer never implies
BFLOAT16 or FLOAT16 support, whether or not the provider uses a FLOAT32 descriptor proxy for
existing low-baseline support. Newly admitted low-valued work must have a qualified custom step
and retain the fixed whole-partition `CUSTOM_PROGRAM` route. Existing supported low answers remain
supported. No provider answer follows merely from a passing sample or from transfer coverage.

## Rationale

The two ledgers answer different questions: v1 says what the cutover preserved; v2 says what the
current provider reports on the same representative basis. A FLOAT32 `EXP` addition can therefore
change only that current answer while BFLOAT16/FLOAT16 remain false until separately qualified.
The test gate protects Model semantics; the version gate protects compatibility. Neither is a
runtime numerical certificate or tolerance selector.

## Consequences

Future capability work must preserve v1, update v2 from real provider queries, qualify the new
route with ordinary occurrence- and dtype-specific conformance tests, and review affected
identities. Current providers and their ledgers can diverge from historical false answers without
erasing history.
The cost is explicit review of each dtype and any necessary compatibility migration. Unsupported
occurrences still fail closed; Prepare fixes routes before shared resource declaration, and
Runtime gains no certificate, retry, fallback, or policy lookup. This ADR changes no provider,
ledger row, route, native application binary interface, or test fixture by itself.

## Related documentation

- [Root capability and low-partition invariants](../../../ARCHITECTURE.md#core-invariants)
- [Backend capability and identity contract](../../architecture/contracts/backend-execution.md#profile-free-backend-capability-and-identity)
- [Runtime/Prepare boundary explanation](../../architecture/runtime-prepare-backend-boundary.md#numerical-semantics-at-the-boundary)
- [Task 0074A brief](../../planning/backends/metal/tasks/0074a-post-cutover-capability-evolution-contract.md)
- [Current-provider ledger test](../../../testing/backend-conformance/src/test/java/io/github/pho001/synaptik/testing/conformance/LowPrecisionCapabilityLedgerTest.java)
- [Historical v1 ledger](../../../testing/backend-conformance/src/test/resources/low-precision-capability-ledger-v1.tsv)
- [Current v2 ledger](../../../testing/backend-conformance/src/test/resources/low-precision-capability-ledger-v2.tsv)
