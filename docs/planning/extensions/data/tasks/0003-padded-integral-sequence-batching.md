# Task Data0003: Padded integral sequence batching

## Status

Complete — independently approved and accepted by main for the local Data scope, 2026-10-20.

## Change class

Class C — shared padding semantics, unpublished public API migration and Text/Data boundary.

## Goal and scope

Deliver the shared LEFT/RIGHT batch representation and fluent INT64 long-sequence batching as
one complete Data capability, including migration, focused tests, Javadoc and API documentation.
The user approved the exact slice; main approved enacted authority and verified HEAD, accepted
dirty Data0001/Data0002, inactive conflicts, no other writer and unchanged protected files.
Final independent review APPROVE and main's local acceptance are recorded below.

## Non-goals

No float/sample batchers, truncation/limit system, tokenization, NN changes, runtime-length binding,
scan/BPTT widening, native lifetime or Engine edge. Deferred Data0005/0006 retain broader work.

## Contracts

- [ARCHITECTURE.md — Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)
  routes reading; also read `Repository layout`, `Module-ownership routing`, `Dependency rules`
  and `Testing requirements` in that root.
- [Extensions/training — extensions/data](../../../../architecture/contracts/extensions-training.md#extensionsdata)
  owns exact API/geometry, allocation and mutation semantics; `extensions/text` identifies the
  downstream consumer. Read `extensions/nn` dependency paragraph only: no NN-to-Data/Text edge.
- [ADR 0043](../../../../design/decisions/0043-whitespace-text-and-padded-integral-batches.md)
  explains the selected migration. Stop on missing/ambiguous authority; do not invent architecture.

## Dependencies and integration

- Depends on: Data0001 and Data0002 Complete/accepted; main authority approval recorded above.
- Conflicts with: all Data/Text writers, shared authority/docs/build/architecture-test writers.
- Parallel group: None; serialized sole writer in the existing workspace.
- Common base revision: ef35f2ec001b0c161d1abe825ff3be725b9be49e plus accepted uncommitted Data0001/Data0002 and enacted authority, not merged predecessors.
- Integration order: approved authority → Data0003 → independent review/acceptance → Text0001 after main verification → main integration checkpoint.
- Integration validation: main's Text–Embedding checkpoint and final source-matched root `./gradlew test javadoc`, once after ordered integration.
- Shared-document integration owner: main; sole Data0003 worker owns module and assigned architecture-consumer/enforcement migration; independent reviewer finalizes shared docs after worker stops.

## Files and symbols

Production paths below are under `extensions/data/src/main/java/io/github/pho001/synaptik/data/`:

- Remove `RightPaddedSequenceBatch.java`; replace with `PaddedSequenceBatch.java` and its Builder.
- Add `PaddingSide.java` and `LongSequenceBatcher.java` with nested Builder, same flat package.
- Update `ValidSequenceLengths.java` Javadoc from prefixes to position-independent counts;
  signatures, copying, equality, failure behavior and O(B) implementation remain unchanged.
- Preserve `SequenceAxisOrder.java`; review Javadoc accuracy.

Tests are under `extensions/data/src/test/java/io/github/pho001/synaptik/data/`:

- Remove `RightPaddedSequenceBatchTest.java` only when migrated to `PaddedSequenceBatchTest.java`.
- Add `PaddingSideTest.java`, `LongSequenceBatcherTest.java`; update existing lengths tests as needed.
- Worker may draft `docs/api/data-api.md` for migration and honest examples; reviewer finalizes it.
- Sole Data0003 worker also migrates
  `testing/architecture-tests/src/dataApiConsumer/java/io/github/pho001/synaptik/testing/architecture/DataApiConsumer.java`
  and `testing/architecture-tests/src/test/java/io/github/pho001/synaptik/testing/architecture/DataDependencyContractTest.java`
  within this capability. Main coordinates but does not edit Class C Java.
- Never touch `THIRD_PARTY_NOTICES.md` or
  `modules/backend/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuOrdinaryPointwiseCastCleanJavaOracle.java`.
  Preserve accepted Data0001/0002 work and their completed briefs.

## Acceptance criteria

- Implement the exact owning contract; no old compatibility type/family. Shared values builder
  requires explicit counts, side and axes; record adds non-null `paddingSide` as fourth component.
- Retain every prior metadata behavior, including descriptor-only borrowing, all Model-valid
  dtypes/layouts/provenance, opaque trailing dimensions, zero extents and Long.MAX_VALUE metadata.
  Extend coverage to both sides, null order, side-sensitive equality and builder independence.
- `LongSequenceBatcher.builder().paddingValue(long).leftPadded()/rightPadded()` plus
  `.timeMajor()/batchMajor().build()` requires all choices; `batch(List<long[]>)` pads to batch max,
  preserves values/order, derives counts and never truncates. Verify all four side/order pairs,
  real padding-valued data, negative/long-extreme values, nulls and repeated independent outputs.
- Empty outer B=0,T=0 and all-empty rows work in both orders/sides. Source mutation after return
  cannot change output; caller coordinates mutation during the call, without transactional promises.
- Check B*currentMaxLength incrementally before B-sized snapshots/output, then checked long byte
  span. Element count must fit Integer.MAX_VALUE; bytes need not fit int. Virtual enormous
  AbstractList with first row long[2] or long[4] rejects immediately without giant allocations
  or reading later rows. Use bounded fixtures, no public bounds helper.
- Allocate one GC-owned destination with `TensorFactory.allocate`, fill its public writable Model
  host storage directly; no flat staging/import recopy. Output is contiguous offset-zero rank-two
  INT64, non-gradient, provenance-free, with no retained caller arrays or new lifetime/close owner.

## Validation

Read current Data source/tests and TensorFactory allocation before editing. Worker runs:

```bash
export JAVA_HOME=/Users/phujka/Library/Java/JavaVirtualMachines/openjdk-26.0.1/Contents/Home
./gradlew :extensions:data:test :extensions:data:javadoc
git diff --check
```

Use the existing JDK/toolchain and Gradle cache. Main's ordinary `./gradlew --version` failed
before any build because the sandbox denied access to the existing wrapper distribution's
`zip.lck` under `/Users/phujka/.gradle/wrapper/dists/gradle-9.6.1-bin/`. This is an environment
restriction, not a code/test failure. Main handles scoped `require_escalated` access to that cache;
do not change `GRADLE_USER_HOME`, download a replacement distribution or alter build configuration.

Main owns architecture consumer/enforcement validation and the final root checkpoint; no duplicate
successful worker suite in review. No benchmark/native campaign. Text numerical integration belongs
to the external integration-test project, not Data tests.

## Documentation and review impact

Implementer updates affected Javadoc and may draft the Data API guide, with every parameter/result/failure
documented under the repository API profile. Independent targeted Class C review finalizes these
against the actual diff/tests after the sole worker stops, including shared glossary/index/public
API finalization. Main coordinates review, integration validation and shared-document integration;
no parallel writers or separate mechanical plumbing tasks. No predecessor brief rewrite.

## Result

Accepted locally by main on 2026-10-20 after final independent APPROVE; zero executable/architecture blockers.

- Added PaddingSide, PaddedSequenceBatch/Builder and LongSequenceBatcher/Builder; removed
  RightPaddedSequenceBatch. Updated ValidSequenceLengths Javadoc; counts implementation unchanged.
- Migrated Data metadata tests and architecture DataApiConsumer/DataDependencyContractTest;
  added side/integral tests. Sequential foreach passes avoid quadratic LinkedList access;
  bounded regression covers all four side/order pairs and indexed null diagnostics.
- Worker: JDK 26.0.1, `./gradlew :extensions:data:test :extensions:data:javadoc` — final post-fix
  PASS, 48 tests/5 suites, zero failures/errors/skips; final Javadoc PASS.
- Main's focused DataDependencyContract + external DataApiConsumer gate — PASS, 7 tests;
  25 tasks, 3 executed/22 UP-TO-DATE. Reused from before the iteration-only fix because API,
  imports and enforcement were unchanged; the source-matched root join completed on 2026-10-21
  (final commands/counts in [Text0001 Result](../../text/tasks/0001-whitespace-tokenizer-and-data-batching.md#result)).
- Independent reviewer finalized docs/api/data-api.md, docs/api/public-api.md, docs/glossary.md,
  docs/index.md and docs/architecture/module-boundaries.md; links/whitespace PASS.
  Reviewed affected Javadoc; no reviewer Java edits or successful-suite reruns.
- Main verified unchanged HEAD ef35f2ec001b0c161d1abe825ff3be725b9be49e, accepted dirty
  Data0001/0002 + enacted authority + Data0003, inactive writer conflicts and protected hashes.
- No local acceptance gap remains. Oversized fixtures reject before large allocations;
  no multi-GB allocation or performance claim. The Text program join's final root test and Javadoc
  gates passed; Text0001 is Complete. No automatic successor is authorized.

Status: Complete
