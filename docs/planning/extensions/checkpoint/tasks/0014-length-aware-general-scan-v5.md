# Task 0014: Length-aware general Scan graph-model v5

## Status

Complete — MAIN accepted 2026-11-01 after independent final code review (APPROVE, zero P1/P2),
75 Checkpoint and 544 NN tests with no failures/errors/skips, affected Javadocs and public loaded
F32/F64 Model numerical proof. Continuous production delivery is user-authorized.

## Change class

Class C — durable wire format, untrusted decoding, cold reconstruction and consumer compatibility.
Use a clean implementation context and independent targeted code/documentation review.

## Goal and scope

Persist the static homogeneous F32/F64 length-aware general Scan cold intersection through the
existing GraphCheckpoint save/load APIs, retaining named values, independent body scopes, ordered
roles, positional aliases and every canonical port. Preserve byte-identical smallest-version
selection and exact kind/magic/version/opcode admission for sources needing no length mode.

## Non-goals

No new production type/package/API, reflection/registry, execution owner, backend dependency,
compiled/runtime/history persistence, saved batch/length values, symbolic cold source, wider
per-value body dtype or constructor admission, optimizer resume or ArtifactLimits change.

## Contracts

- [Root authority and scope](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts):
  Core lifecycle, Core invariants, Dependency rules, Testing requirements.
- [Compiler cold inventory](../../../../architecture/contracts/compiler-autograd.md#bounded-general-scan-cold-reconstruction):
  static homogeneous length intersection, preflight and substitution roles, no provider admission.
- [Scan header/roles](../../../../architecture/contracts/recurrent-scan.md#header-bindings-and-source-admission):
  c+s+i+1 actuals, c+s+i formals, final discrete lengths actual, ordered row roles.
- [Generic graph source](../../../../architecture/contracts/extensions-training.md#generic-graph-source-and-values),
  [v4](../../../../architecture/contracts/extensions-training.md#canonical-graph-model-version-4) and
  [v5](../../../../architecture/contracts/extensions-training.md#canonical-graph-model-version-5):
  codec-only wire ownership, dry sizing, bounds, complete preflight before import and publication.

## Exact v5 encoding

Use ArtifactFile v1 kind `graph-model.v5`, ASCII `SYNGRF05`, signed big-endian i32 version 5.
Only an outer length-aware Scan selects v5 and opcode 33. Opcode 23 retains its original all-valid
seven-field record in all versions; v5 may contain both 23 and 33. Bodies retain opcodes 1..19 and
24..32, never nested scans or opcode 33. New readers preserve exact v1-v4 record admission.

For opcode 33, retain ordinary phase/opcode, count-prefixed ordered actual IDs, then write the
existing Scan fields in order: i64 timeCount, i32 carryCount, sequenceCount, invariantCount,
emissionCount, residualCount, u8 reverse (0/1). Append i32 rowCarryCount, i32 sequence-axis count
(equal to sequenceCount), then one ordered u8 presence flag per sequence (0=SHARED, 1=ROW), with an
i32 step axis only when present. The existing role domain admits axes 0 and 1. The final entry of
the actual-ID list is the lengths reference; do not duplicate that reference or serialize its bytes.
Continue with the unchanged v4 inline-body record, Q and all canonical output descriptors.

Require positive rowCarryCount <= carryCount, checked actual/formal/output counts and exact role
geometry. Bound axis count by sequenceCount and remaining bytes before list allocation; reject
unknown flags, negative/out-of-domain axes, truncated optional fields, header/count/descriptor
mismatches and overflow. Complete raw/Compiler preflight precedes any host import or producer IDs,
including late-invalid/dead/T=0 bodies. Keep all existing aggregate, payload, file, manifest, entry,
key and cumulative ID budgets; dry Sink counts exactly the same fields before schema allocation.

## Dependencies and integration

- Depends on: Compiler0018 Complete; CP0013 and retained artifact/graph baseline Complete.
- Conflicts with: Compiler cold changes, CP wire/source/limits work and NN workflow/shared contract
  integration; serialize changed semantics.
- Parallel group: None.
- Common base revision: HEAD `07a281edd3086efde2853211b71b48288db027e1` plus the accepted
  post-Compiler0018 main source snapshot. MAIN records `checkpoint0014-base.json` in the existing
  production-run evidence directory before launching the sole isolated codec worker.
- Integration order: accepted exact wire/authority packet -> implementation/review -> MAIN consumer
  checkpoint -> later NN length workflow. No separate user prompt is required for this admitted
  successor under continuous production delivery.
- Integration validation: Checkpoint/NN module tests/Javadocs and public loaded generic Model
  proof completed; NN0033 owns the later recurrent-composition consumer and final release join.
- Shared-document integration owner: MAIN.

MAIN Ready verification: CP0013 and Compiler0018 are Complete; the exact v5 encoding is in the
owning contract and ADR0057, with independent design APPROVE. No competing Checkpoint wire or NN
writer is active. Freeze the full dirty source base, not HEAD alone. No build/lint/tests/formatters
mid-flight; MAIN owns subsequent verification and shared documentation.

## Files and symbols

- `modules/checkpoint/src/main/java/io/github/pho001/synaptik/checkpoint/graph/GraphArtifactCodec.java`:
  version selection, kind/magic, opcode/attribute writing, bounded Reader and scoped node admission.
- `GraphCheckpoint.java` in the same package: supported wire/cold boundary Javadoc only as needed.
- Matching Checkpoint tests: new focused `GraphArtifactV5Test`, extend codec/old-format tests where
  the old rejection contract changes; retain old independently authored byte fixtures.
- MAIN owns root/owning contract, master/roadmap, graph-checkpoint API and release status updates.

## Acceptance criteria

- Static F32/F64, both traversal directions, T=0/nonzero, B=0/1/nonzero, ROW(0)/ROW(1)/SHARED
  sequences and terminal accumulators round-trip without executing, reading lengths or unrolling.
- Preserve ordered input/state aliases, unused formals, repeated roles, all canonical sibling
  descriptors, raw attributes/state bits and separate occurrence/body scopes.
- Valid independent v5 bytes load; malformed role counts/flags/axes/length descriptors/references,
  old tuple with new opcode/header, nested opcode33 and trailing/truncated bytes fail before
  import or output IDs. A changed runtime lengths input is supplied after load, not restored.
- Exact/one-below encoded limits reject before state reads/schema allocation; existing atomic
  publication and failure ordering remain. A late malformed body cannot partially import state.
- Sources requiring no length mode retain byte-identical v1-v4 selection; unknown versions and
  mismatched tuples reject. A v5 image with only old records re-saves as the smallest old version.
- Final consumer evidence uses public loaded sources/Models in both dtypes, actual numerical
  forward/backward and fresh input/length bindings; source round-trip alone is not execution proof.

## Validation

No build/lint/tests/formatters mid-flight. MAIN runs focused `GraphArtifactV5Test`, old
`GraphArtifactV4Test`, `GraphArtifactCodecTest` and `GraphCheckpointGeneralScanTest` after source
and independent review stabilize, then one affected Checkpoint test/Javadoc gate. MAIN owns the
later public NN consumer smoke and combined integration checkpoint. Do not repeat prior successful
evidence or label historical failures PASS.

## Documentation and review impact

Independent review covers field order, bounds, failure-before-import/IDs, exact old version
admission and API/Javadoc accuracy. Update the owning v5 contract/status and graph-checkpoint
explanation in the same integrated change. No new glossary type is introduced.

## Result

Implemented in existing `GraphArtifactCodec`/`GraphCheckpoint` and four matching test classes,
including new `GraphArtifactV5Test`; no new production API/type/dependency. The isolated source
base has 2617 files, recorded in `checkpoint0014-base.json`; six integrated paths/hashes are in
`checkpoint0014-integrated-files.json`. Existing main counterparts matched that base before copy.

Version selection uses outer opcode 33 only for length mode; opcode 23 and smallest v1-v4 bytes
remain unchanged. Reused dry Sink, ArtifactLimits, state/alias handling, Compiler cold admission/
reconstruction, canonical topology and recapture checks. Body formal count excludes lengths.

Independent review first found two P2 blockers: raw encoding bypassed homogeneous cold admission,
and role allocation preceded scalar/cardinality checks. Both were corrected and independently
APPROVED: v5 encode reuses Compiler SourceCapture before sizing/storage; opcode33 reuses the
seven-field ScanAttrs header and checked arity before role parsing. Direct raw typed/symbolic
negatives and early malformed/overflow header cases remain behavioral regression evidence.

MAIN command with JDK 26: `:extensions:checkpoint:test :extensions:checkpoint:jar
:extensions:checkpoint:javadoc :extensions:nn:test :extensions:nn:javadoc` — PASS
(`artifact://81`, 75 Checkpoint + 544 NN tests; no failures, errors or skips).
Independent fixtures cover valid/corrupt fields, scopes/aliases/ports, static/empty domains,
smallest-version re-save and exact/one-below encoded budgets.

The unchanged public `LengthModelSmoke.java` before/after scenario first recorded the old save
rejection. After v5, both F32/F64 passed independent scalar forward/SGD expectations for fresh
`[3,1]`, `[0,2]`, `[0,0]` lengths and padded NaNs, rejected invalid lengths without weight change,
saved before compute/after train, closed, loaded, inferred with `[2,0]`, then configured and
trained a new optimizer. Log: `length-model-v5-after.log` in the production-run evidence directory.
No Vector API launcher flag was needed. Installed OpenBLAS qualification emitted the ordinary
JDK native-access warning when that permission flag was omitted; no warning was suppressed.

Symbolic/wider-typed cold source, optimizer resume and fluent RNN recipe delivery remain outside
this accepted wire scope. NN0033 is the separately admitted practical-composition successor.
