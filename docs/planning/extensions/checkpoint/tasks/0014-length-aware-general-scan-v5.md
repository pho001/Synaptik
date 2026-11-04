# Task 0014: Length-aware [general Scan](../../../../glossary.md#general-differentiable-scan) [graph](../../../../glossary.md#graph)-[model](../../../../glossary.md#term-model) v5

> **Editorial reading context — not a status update.** Length-aware source describes which
> carried values and sequence axes belong to each batch row. The lengths Tensor is a separate
> outer control input, not a step-body formal. With external lengths, the file preserves that
> role and descriptor, not the particular counts from a batch; the application supplies fresh
> compatible counts after load. Named length state would instead follow named-value rules.
> See today's [version-five role and persistence explanation](../../../../reference/checkpoint/graph-formats.md#length-aware-general-scan-persistence)
> and [runtime-length NN construction](../../../../reference/nn/recurrent.md#nn-runtime-length-recurrent-composition).
> Stored source/current weights still omit optimizer continuation and prepared execution.

## Status

Complete — MAIN accepted 2026-11-01 after independent final code review (APPROVE, zero P1/P2),
75 [Checkpoint](../../../../glossary.md#term-checkpoint) and 544 [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) tests with no failures/errors/skips, affected Javadocs and public loaded
[F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding) [Model](../../../../glossary.md#term-model) numerical proof. Continuous production delivery is user-authorized.

## Change class

Class C — durable [wire format](../../../../glossary.md#term-wire-format), untrusted decoding, [cold reconstruction](../../../../glossary.md#term-cold-reconstruction) and consumer compatibility.
Use a clean implementation context and independent targeted code/documentation review.

## Goal and [scope](../../../../glossary.md#term-scope)

Persist the static [homogeneous](../../../../glossary.md#term-homogeneous) [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding) length-aware [general Scan](../../../../glossary.md#general-differentiable-scan) cold intersection through the
existing GraphCheckpoint save/load APIs, retaining named values, independent body scopes, ordered
roles, positional [aliases](../../../../glossary.md#aliasing-and-arena) and every [canonical](../../../../glossary.md#term-canonical-form) port. Preserve byte-identical smallest-version
selection and exact kind/magic/version/[opcode](../../../../glossary.md#term-opcode) [admission](../../../../glossary.md#term-admission) for sources needing no length mode.

## Non-goals

No new production type/package/[API](../../../../glossary.md#term-application-programming-interface-api), reflection/[registry](../../../../glossary.md#term-registry), execution [owner](../../../../glossary.md#term-owner), [backend](../../../../glossary.md#backend) [dependency](../../../../glossary.md#term-dependency),
compiled/[runtime](../../../../glossary.md#term-runtime)/history [persistence](../../../../glossary.md#term-persistence), saved [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)/length values, symbolic cold source, wider
per-value body [dtype](../../../../glossary.md#data-type--datatype) or constructor [admission](../../../../glossary.md#term-admission), [optimizer](../../../../glossary.md#term-optimizer) [resume](../../../../glossary.md#schema-serialization-and-checkpoint) or ArtifactLimits change.

## Contracts

- [Root authority and scope](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>:
  Core lifecycle, Core invariants, Dependency rules, Testing requirements.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Compiler cold inventory](../../../../architecture/contracts/compiler-autograd.md#bounded-general-scan-cold-reconstruction)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>:
  static homogeneous length intersection, preflight and substitution roles, no provider admission.<span data-glossary-navigation="true"> ([homogeneous glossary](../../../../glossary.md#term-homogeneous); [preflight glossary](../../../../glossary.md#term-preflight); [admission glossary](../../../../glossary.md#term-admission))</span>
- [Scan header/roles](../../../../architecture/contracts/recurrent-scan.md#header-bindings-and-source-admission):
  c+s+i+1 actuals, c+s+i formals, final discrete lengths actual, ordered row roles.
- [Generic graph source](../../../../architecture/contracts/extensions-training.md#generic-graph-source-and-values)<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>,
  [v4](../../../../architecture/contracts/extensions-training.md#canonical-graph-model-version-4) and
  [v5](../../../../architecture/contracts/extensions-training.md#canonical-graph-model-version-5):
  codec-only wire ownership, dry sizing, bounds, complete preflight before import and publication.<span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [dry sizing glossary](../../../../glossary.md#term-dry-sizing); [publication glossary](../../../../glossary.md#term-publication))</span>

## Exact v5 encoding

Use ArtifactFile v1 kind `graph-model.v5`, [ASCII](../../../../glossary.md#term-ascii) `SYNGRF05`, signed [big-endian](../../../../glossary.md#text-encoding-and-file-boundaries) i32 version 5.
Only an outer length-aware Scan selects v5 and [opcode](../../../../glossary.md#term-opcode) 33. Opcode 23 retains its original all-valid
seven-field record in all versions; v5 may contain both 23 and 33. Bodies retain opcodes 1..19 and
24..32, never nested scans or opcode 33. New readers preserve exact v1-v4 record [admission](../../../../glossary.md#term-admission).

For opcode 33, retain ordinary phase/opcode, count-prefixed ordered actual IDs, then write the
existing Scan fields in order: i64 timeCount, i32 carryCount, sequenceCount, invariantCount,
emissionCount, residualCount, u8 reverse (0/1). Append i32 rowCarryCount, i32 sequence-[axis](../../../../glossary.md#term-axis) count
(equal to sequenceCount), then one ordered u8 presence flag per sequence (0=SHARED, 1=ROW), with an
i32 step axis only when present. The existing role domain admits axes 0 and 1. The final entry of
the actual-ID list is the lengths reference; do not duplicate that reference or serialize its bytes.
Continue with the unchanged v4 inline-body record, Q and all [canonical](../../../../glossary.md#term-canonical-form) output [descriptors](../../../../glossary.md#term-descriptor).

Require positive rowCarryCount <= carryCount, checked actual/formal/output counts and exact role
geometry. Bound axis count by sequenceCount and remaining bytes before list allocation; reject
unknown flags, negative/out-of-domain axes, truncated optional fields, header/count/descriptor
mismatches and [overflow](../../../../glossary.md#term-overflow). Complete raw/[Compiler](../../../../glossary.md#term-compiler) [preflight](../../../../glossary.md#term-preflight) precedes any [host](../../../../glossary.md#term-host) import or [producer](../../../../glossary.md#term-producer) IDs,
including late-invalid/dead/T=0 bodies. Keep all existing aggregate, [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication), file, [manifest](../../../../glossary.md#term-manifest), entry,
key and cumulative ID budgets; dry Sink counts exactly the same fields before [schema](../../../../glossary.md#term-schema) allocation.

## Dependencies and integration

- Depends on: Compiler0018 Complete; CP0013 and retained [artifact](../../../../glossary.md#term-artifact)/[graph](../../../../glossary.md#graph) baseline Complete.
- Conflicts with: Compiler cold changes, CP wire/source/limits work and NN workflow/shared contract
  integration; serialize changed semantics.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>
- Parallel group: None.
- Common base revision: HEAD `07a281edd3086efde2853211b71b48288db027e1` plus the accepted
  post-Compiler0018 main source snapshot. MAIN records <span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot))</span>`checkpoint0014-base.json` in the existing
  production-run evidence directory before launching the sole isolated codec worker.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [codec glossary](../../../../glossary.md#term-codec))</span>
- Integration order: accepted exact wire/authority packet -> implementation/review -> MAIN consumer
  checkpoint -> later NN length workflow. No separate user prompt is required for this admitted
  successor under continuous production delivery.
- Integration validation: Checkpoint/NN module tests/Javadocs and public loaded generic Model
  proof completed; NN0033 owns the later recurrent-composition consumer and final release join.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [module glossary](../../../../glossary.md#term-module); [Model glossary](../../../../glossary.md#term-model); [release glossary](../../../../glossary.md#term-release))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): MAIN.

MAIN [Ready](../../../../glossary.md#term-ready) verification: CP0013 and Compiler0018 are Complete; the exact v5 encoding is in the
owning contract and ADR0057, with independent design APPROVE. No competing [Checkpoint](../../../../glossary.md#term-checkpoint) wire or [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)
writer is active. Freeze the full dirty source base, not HEAD alone. No build/lint/tests/formatters
mid-flight; MAIN owns subsequent verification and shared documentation.

## Files and symbols

- `modules/checkpoint/src/main/java/io/github/pho001/synaptik/checkpoint/graph/GraphArtifactCodec.java`:
  version selection, kind/magic, opcode/attribute writing, bounded Reader and scoped node admission.<span data-glossary-navigation="true"> ([opcode glossary](../../../../glossary.md#term-opcode); [node glossary](../../../../glossary.md#node); [admission glossary](../../../../glossary.md#term-admission))</span>
- `GraphCheckpoint.java` in the same package: supported wire/cold boundary [Javadoc](../../../../glossary.md#term-javadoc) only as needed.
- Matching [Checkpoint](../../../../glossary.md#term-checkpoint) tests: new focused `GraphArtifactV5Test`, extend codec/old-format tests where
  the old rejection contract changes; retain old independently authored byte fixtures.<span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec))</span>
- MAIN owns root/owning contract, master/roadmap, [graph](../../../../glossary.md#graph)-checkpoint [API](../../../../glossary.md#term-application-programming-interface-api) and [release](../../../../glossary.md#term-release) status updates.

## Acceptance criteria

- Static F32/F64, both traversal directions, T=0/nonzero, B=0/1/nonzero, ROW(0)/ROW(1)/SHARED
  sequences and terminal accumulators round-trip without executing, reading lengths or unrolling.<span data-glossary-navigation="true"> ([F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [round-trip glossary](../../../../glossary.md#schema-serialization-and-checkpoint); [unrolling glossary](../../../../glossary.md#term-unrolling))</span>
- Preserve ordered input/state aliases, unused formals, repeated roles, all canonical sibling
  descriptors, raw attributes/state bits and separate occurrence/body scopes.<span data-glossary-navigation="true"> ([aliases glossary](../../../../glossary.md#aliasing-and-arena); [canonical glossary](../../../../glossary.md#term-canonical-form); [descriptors glossary](../../../../glossary.md#term-descriptor); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Valid independent v5 bytes load; malformed role counts/flags/axes/length descriptors/references,
  old tuple with new opcode/header, nested opcode33 and trailing/truncated bytes fail before
  import or output IDs. A changed runtime lengths input is supplied after load, not restored.<span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis); [opcode glossary](../../../../glossary.md#term-opcode); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Exact/one-below encoded limits reject before state reads/schema allocation; existing atomic
  publication and failure ordering remain. A late malformed body cannot partially import state.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [atomic publication glossary](../../../../glossary.md#term-atomic-publication))</span>
- Sources requiring no length mode retain byte-identical v1-v4 selection; unknown versions and
  mismatched tuples reject. A v5 image with only old records re-saves as the smallest old version.
- Final consumer evidence uses public loaded sources/Models in both dtypes, actual numerical
  forward/backward and fresh input/length bindings; source round-trip alone is not execution proof.<span data-glossary-navigation="true"> ([dtypes glossary](../../../../glossary.md#data-type--datatype); [bindings glossary](../../../../glossary.md#term-binding))</span>

## [Validation](../../../../glossary.md#term-validation)

No build/lint/tests/formatters mid-flight. MAIN runs focused `GraphArtifactV5Test`, old
`GraphArtifactV4Test`, `GraphArtifactCodecTest` and `GraphCheckpointGeneralScanTest` after source
and independent review stabilize, then one affected [Checkpoint](../../../../glossary.md#term-checkpoint) test/[Javadoc](../../../../glossary.md#term-javadoc) gate. MAIN owns the
later public [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) consumer [smoke](../../../../glossary.md#term-smoke-test) and combined integration checkpoint. Do not repeat prior successful
evidence or label historical failures PASS.

## Documentation and review impact

Independent review covers field order, bounds, failure-before-import/IDs, exact old version
[admission](../../../../glossary.md#term-admission) and [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc) accuracy. Update the owning v5 contract/status and [graph](../../../../glossary.md#graph)-[checkpoint](../../../../glossary.md#term-checkpoint)
explanation in the same integrated change. No new glossary type is introduced.

## Result

Implemented in existing `GraphArtifactCodec`/`GraphCheckpoint` and four matching test classes,
including new `GraphArtifactV5Test`; no new production [API](../../../../glossary.md#term-application-programming-interface-api)/type/[dependency](../../../../glossary.md#term-dependency). The isolated source
base has 2617 files, recorded in `checkpoint0014-base.json`; six integrated paths/[hashes](../../../../glossary.md#term-hash) are in
`checkpoint0014-integrated-files.json`. Existing main counterparts matched that base before copy.

Version selection uses outer [opcode](../../../../glossary.md#term-opcode) 33 only for length mode; opcode 23 and smallest v1-v4 bytes
remain unchanged. Reused dry Sink, ArtifactLimits, state/[alias](../../../../glossary.md#aliasing-and-arena) handling, [Compiler](../../../../glossary.md#term-compiler) cold [admission](../../../../glossary.md#term-admission)/
reconstruction, [canonical](../../../../glossary.md#term-canonical-form) [topology](../../../../glossary.md#term-topology) and recapture checks. Body formal count excludes lengths.

Independent review first found two P2 blockers: raw encoding bypassed [homogeneous](../../../../glossary.md#term-homogeneous) cold admission,
and role allocation preceded [scalar](../../../../glossary.md#term-scalar)/[cardinality](../../../../glossary.md#term-cardinality) checks. Both were corrected and independently
APPROVED: v5 encode reuses Compiler SourceCapture before sizing/storage; opcode33 reuses the
seven-field ScanAttrs header and checked [arity](../../../../glossary.md#term-arity) before role parsing. Direct raw typed/symbolic
negatives and early malformed/[overflow](../../../../glossary.md#term-overflow) header cases remain behavioral [regression](../../../../glossary.md#term-regression) evidence.

MAIN command with [JDK](../../../../glossary.md#term-java-development-kit-jdk) 26: `:extensions:checkpoint:test :extensions:checkpoint:jar
:extensions:checkpoint:javadoc :extensions:nn:test :extensions:nn:javadoc` — PASS
(`artifact://81`, 75 [Checkpoint](../../../../glossary.md#term-checkpoint) + 544 [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) tests; no failures, errors or skips).
Independent fixtures cover valid/corrupt fields, scopes/aliases/ports, static/empty domains,
smallest-version re-save and exact/one-below encoded budgets.

The unchanged public `LengthModelSmoke.java` before/after scenario first recorded the old save
rejection. After v5, both [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding) passed independent scalar forward/[SGD](../../../../glossary.md#term-stochastic-gradient-descent-sgd) expectations for fresh
`[3,1]`, `[0,2]`, `[0,0]` lengths and padded NaNs, rejected invalid lengths without [weight](../../../../glossary.md#term-weight) change,
saved before compute/after train, closed, loaded, inferred with `[2,0]`, then configured and
trained a new [optimizer](../../../../glossary.md#term-optimizer). [Log](../../../../glossary.md#term-log): `length-model-v5-after.log` in the production-[run](../../../../glossary.md#term-run) evidence directory.
No [Vector API](../../../../glossary.md#term-vector-api) launcher flag was needed. Installed [OpenBLAS qualification](../../../../glossary.md#openblas-qualification) emitted the ordinary
JDK native-access warning when that permission flag was omitted; no warning was suppressed.

Symbolic/wider-typed cold source, optimizer [resume](../../../../glossary.md#schema-serialization-and-checkpoint) and fluent [RNN](../../../../glossary.md#term-recurrent-neural-network-rnn) recipe delivery remain outside
this accepted wire [scope](../../../../glossary.md#term-scope). NN0033 is the separately admitted practical-composition successor.
