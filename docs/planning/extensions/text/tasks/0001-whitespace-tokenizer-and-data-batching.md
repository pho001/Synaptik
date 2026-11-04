# Task Text0001: Whitespace [tokenizer](../../../../glossary.md#frozen-whitespace-tokenizer) and Data batching

> **Editorial reading context — not a status update.** Text copies the application's ordered
> vocabulary and encodes whitespace-separated runs into fresh integer arrays. TextSequenceBatcher
> delegates numeric allocation/padding to Data; returned counts are host metadata, not graph
> bindings. NN Embedding separately turns IDs into floating features, and Engine separately
> evaluates the expression. Unknown and padding IDs are vocabulary conventions, not special
> optimizer behavior or a global sequence mask. Read today's [frozen tokenization](../../../../reference/text/tokenization.md#frozen-token-ids),
> [batch delegation](../../../../reference/text/tokenization.md#explicit-batch-configuration) and
> [integration limits](../../../../reference/text/tokenization.md#boundaries-and-related-contracts)
> for the actual application-owned connections and missing fluent integer-input transition.

## Status

Complete — independently approved and accepted by main on 2026-10-21.

## Change class

Class C — new extension [dependency](../../../../glossary.md#term-dependency) boundary and user-visible text-to-numeric workflow.

## Goal and [scope](../../../../glossary.md#term-scope)

Deliver one concrete [frozen whitespace tokenizer](../../../../glossary.md#frozen-whitespace-tokenizer) and a fluent Data-delegating [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) adapter,
including [module](../../../../glossary.md#term-module) registration, focused tests, [Javadoc](../../../../glossary.md#term-javadoc) and [API](../../../../glossary.md#term-application-programming-interface-api) guide. The user approved this bounded
capability on 2026-10-20; main approved coordinated authority, exact sole Text API edge and dirty
base. Data0003 is locally Complete after final independent APPROVE and main acceptance.
Main's fresh [dependency](../../../../glossary.md#term-dependency)/base/conflict verification below authorizes the serial Text implementation.
Final independent Text review and the Text program join are complete; evidence is below.

## Non-goals

No [tokenizer](../../../../glossary.md#frozen-whitespace-tokenizer) interface, separate vocabulary/[artifact](../../../../glossary.md#term-artifact)/[fingerprint](../../../../glossary.md#term-fingerprint) family, [BPE](../../../../glossary.md#term-byte-pair-encoding-bpe), vocabulary training,
[persistence](../../../../glossary.md#term-persistence), decode, [normalization](../../../../glossary.md#term-normalization), [BOS](../../../../glossary.md#term-bos)/[EOS](../../../../glossary.md#term-eos), truncation/limit policy, predictor or corpus reader.
No [Embedding](../../../../glossary.md#embedding) edit, [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) Data/Text edge, Text [Engine](../../../../glossary.md#term-engine)/[backend](../../../../glossary.md#backend) edge, [runtime](../../../../glossary.md#term-runtime) [binding](../../../../glossary.md#term-binding) or scan/[BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt) change.

## Contracts

- [ARCHITECTURE.md — Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [normative glossary](../../../../glossary.md#term-normative))</span>,
  plus root `Repository layout`, `Module-ownership routing`, `Dependency rules`, `Testing requirements`.
- [Extensions/training — extensions/text](../../../../architecture/contracts/extensions-training.md#extensionstext)
  owns exact tokenizer/adapter semantics; <span data-glossary-navigation="true"> ([tokenizer glossary](../../../../glossary.md#frozen-whitespace-tokenizer))</span>`extensions/data` owns batching and host counts.
  Read <span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host))</span>`extensions/nn` [dependency](../../../../glossary.md#term-dependency) paragraph only: [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) has no Data/Text edge.
- [ADR 0043](../../../../design/decisions/0043-whitespace-text-and-padded-integral-batches.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>.
  Stop on missing/ambiguous authority rather than inventing architecture.

## Dependencies and integration

- Depends on: Data0003 Complete, independently approved and accepted by main; [Ready](../../../../glossary.md#term-ready) verification below.
- Conflicts with: all Data/Text writers and shared authority/build/architecture/integration/docs writers.
- Parallel group: None; serialized sole writer in the existing workspace.
- Common base revision: ef35f2ec001b0c161d1abe825ff3be725b9be49e plus accepted dirty Data0001/Data0002, enacted Data/Text authority and accepted Data0003 including the sequential-iteration fix and final documentation; uncommitted, not merged predecessors.
- Integration order: approved authority → Data0003 acceptance → main marks Text0001 Ready → Text0001 → independent review → main [checkpoint](../../../../glossary.md#term-checkpoint).
- Integration [validation](../../../../glossary.md#term-validation): `./gradlew :testing:architecture-tests:test :testing:integration-tests:test --tests '*TextEmbeddingIntegrationTest'`, then main's single source-matched `./gradlew test javadoc` checkpoint.
- Shared-document integration [owner](../../../../glossary.md#term-owner): main; sole Text0001 worker owns registration, architecture enforcement and numerical integration with the [module](../../../../glossary.md#term-module); reviewer finalizes shared docs after worker stops.

### Launch verification

Main verified on 2026-10-20: Data0001/0002 and Data0003 are Complete/accepted; final fixed-code
Data review is APPROVE with zero executable/architecture blockers, and the reviewer stopped.
Worker's post-fix 48 tests/5 suites and [Javadoc](../../../../glossary.md#term-javadoc) passed; main's 7-test architecture/consumer gate
passed and remains applicable to unchanged [API](../../../../glossary.md#term-application-programming-interface-api)/imports/enforcement. HEAD and the exact dirty base
above are unchanged. Enacted root/owning Data/Text contracts and the sole api(Data) edge are approved;
no architecture change is authorized. Declared writer conflicts are inactive; this [planning](../../../../glossary.md#term-planning) writer
stops before main launches the sole Text worker. Both protected file [hashes](../../../../glossary.md#term-hash) are unchanged.
Text0001 is the sole Text frontier; no parallel writers or further approval loop is required.
Main completed the source-matched root test/Javadoc gate at the Text program join; see Result.

## Files and symbols

- Add `extensions/text/build.gradle.kts`: exactly `api(project(":extensions:data"))`, no direct
  Model edge in any configuration. Model comes transitively through Data's API export.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- Add `extensions/text/src/main/java/io/github/pho001/synaptik/text/WhitespaceTokenizer.java`:
  final concrete type with private construction and immutable ordered copied vocabulary.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- Add adjacent `TextSequenceBatcher.java`: final concrete adapter with nested public static final
  Builder; no subpackages or additional public top-level types.
- Add `extensions/text/src/test/java/io/github/pho001/synaptik/text/WhitespaceTokenizerTest.java`
  and adjacent `TextSequenceBatcherTest.java`.
- Worker may draft `docs/api/text-api.md`; independent reviewer finalizes public [API](../../../../glossary.md#term-application-programming-interface-api)/shared docs.
- Sole Text0001 worker updates `settings.gradle.kts`, `testing/architecture-tests/build.gradle.kts`
  (source inputs and isolated Text consumer compilation), architecture dependency fixtures/tests
  under <span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [dependency glossary](../../../../glossary.md#term-dependency))</span>`testing/architecture-tests/src/test/java/io/github/pho001/synaptik/testing/architecture/`,
  and adds a Text-only consumer under `testing/architecture-tests/src/textApiConsumer/`.
- The same worker updates `testing/integration-tests/build.gradle.kts` with only a
  testImplementation Text edge and adds external
  `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/TextEmbeddingIntegrationTest.java`.
  These are part of this cohesive capability, not separate mechanical plumbing tasks.
- Read existing `extensions/nn/src/test/java/io/github/pho001/synaptik/nn/layers/EmbeddingTest.java`
  for unchanged ordinary row-zero/GATHER semantics. Do not modify Embedding.<span data-glossary-navigation="true"> ([GATHER glossary](../../../../glossary.md#gather); [Embedding glossary](../../../../glossary.md#embedding))</span>
- Never touch `THIRD_PARTY_NOTICES.md` or
  `modules/backend/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuOrdinaryPointwiseCastCleanJavaOracle.java`.
  Preserve accepted Data changes and completed predecessor briefs.

## Acceptance criteria

- `WhitespaceTokenizer.of(List<String>)`, `int vocabularySize()`, `long paddingId()`,
  `long unknownId()`, `long[] encode(String)` implement the owning contract: PAD=0, UNK=1,
  ordinary IDs from 2 in exact supplied order, immutable copy, fresh encode arrays even empty.<span data-glossary-navigation="true"> ([PAD glossary](../../../../glossary.md#frozen-whitespace-tokenizer); [UNK glossary](../../../../glossary.md#frozen-whitespace-tokenizer); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- Reject null list/token/text; reject empty, whitespace-containing and duplicate ordinary tokens.
  Empty vocabulary is valid. No magic spellings: listed "PAD"/"UNK" receive ordinary IDs.<span data-glossary-navigation="true"> ([token glossary](../../../../glossary.md#term-token))</span>
- Encode maximal nonempty code-point runs separated by Character.isWhitespace(int). Test tabs,
  repeated/leading/trailing whitespace, Unicode whitespace and non-whitespace distinctions,
  punctuation, case, composed/decomposed text, supplementary characters and unpaired surrogates.
  Unknown runs map to one UNK each; there is no normalization or source-character replacement.<span data-glossary-navigation="true"> ([Unicode glossary](../../../../glossary.md#term-unicode); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- `TextSequenceBatcher.builder(tokenizer)` fixes a non-null tokenizer; explicit left/right and
  time/batch-major selections are required, no defaults. Missing choices fail at build; setters
  return the same Builder, last selection wins and built configurations are independent.<span data-glossary-navigation="true"> ([tokenizer glossary](../../../../glossary.md#frozen-whitespace-tokenizer); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- `batch(List<String>)` rejects null list/rows, encodes each String once, then delegates to one
  configured LongSequenceBatcher exactly once using paddingId. No duplicated padding/allocation
  algorithm or manually synchronized lengths. Verify both sides/orders, unknowns, empty outer
  and all-empty rows, input-list independence and fresh outputs.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding))</span>
- Architecture enforcement proves sole api(Data), transitive Model consumer compilation and
  absence of forbidden imports/edges including NN→Data/Text; Text unit tests use no NN/Engine.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Main's true numerical integration uses a deterministic FLOAT32 Embedding table sized from the
  tokenizer vocabulary and <span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [Embedding glossary](../../../../glossary.md#embedding))</span>`engine.compute(embedding.forward(batch.values()))`. Compare every
  output against a direct table-row oracle for all four side/order pairs, known/unknown IDs and
  padded positions. Give PAD row distinct nonzero values and prove it is gathered normally;
  expression shape/provenance inspection alone is not acceptance. No recurrence is involved.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [shape glossary](../../../../glossary.md#shape); [provenance glossary](../../../../glossary.md#provenance))</span>

## [Validation](../../../../glossary.md#term-validation)

Worker owns serialized registration/plumbing and runs:

```bash
export JAVA_HOME=/Users/phujka/Library/Java/JavaVirtualMachines/openjdk-26.0.1/Contents/Home
./gradlew :extensions:text:test :extensions:text:javadoc
git diff --check
```

Use the existing [JDK](../../../../glossary.md#term-java-development-kit-jdk)/[toolchain](../../../../glossary.md#term-toolchain) and [Gradle](../../../../glossary.md#term-gradle) [cache](../../../../glossary.md#term-cache). Main's ordinary `./gradlew --version` failed
before any build because the sandbox denied access to the existing wrapper [distribution](../../../../glossary.md#term-distribution)'s
`zip.lck` under `/Users/phujka/.gradle/wrapper/dists/gradle-9.6.1-bin/`. This is an environment
restriction, not a code/test failure. Main handles scoped `require_escalated` access to that cache;
do not change `GRADLE_USER_HOME`, download a replacement distribution or alter build configuration.

Main owns the integration validation commands above and final root gate, reusing worker evidence;
main coordinates Class C code but does not implement Java/plumbing directly.
Architecture fixtures must include positive/negative boundary cases and a Text-only consumer.
No benchmark/native campaign or duplicate successful suite in documentation review.

## Documentation and review impact

Implementer drafts meaningful [parameter](../../../../glossary.md#term-parameter)/result/failure/[lifecycle](../../../../glossary.md#lifecycle) [Javadoc](../../../../glossary.md#term-javadoc) and focused Text [API](../../../../glossary.md#term-application-programming-interface-api)
examples, explicitly distinguishing Data counts from scan [runtime](../../../../glossary.md#term-runtime) inputs. Independent targeted
Class C review finalizes the actual diff/tests and guide under repository documentation profiles.
After the sole worker stops, the independent reviewer finalizes shared glossary/index/public API
documentation. Main owns review coordination, shared-document integration, [validation](../../../../glossary.md#term-validation) and final
integration status. Deferred training/[artifact](../../../../glossary.md#term-artifact) plans remain Draft, not silently removed.

## Result

Accepted by main on 2026-10-21 after independent code APPROVE and separate final documentation APPROVE.
- Implemented `WhitespaceTokenizer`, `TextSequenceBatcher`, module registration, isolated transitive
  Data/Model consumer, architecture enforcement and external <span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [Model glossary](../../../../glossary.md#term-model))</span>`TextEmbeddingIntegrationTest`.
- Reused worker `./gradlew :extensions:text:test :extensions:text:javadoc`: PASS, 12 unit tests;
  Data final module/Javadoc PASS, 48 tests/5 suites and independent APPROVE.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Focused gates PASS: 7 Data architecture, 5 Text architecture, 1 true Engine/Embedding numerical
  test against a deterministic FLOAT32 oracle for all 4 side/order combinations, including PAD.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [Embedding glossary](../../../../glossary.md#embedding); [FLOAT32 glossary](../../../../glossary.md#term-float32); [oracle glossary](../../../../glossary.md#term-test-oracle); [PAD glossary](../../../../glossary.md#padding))</span>
- Final source-matched root commands (existing [JDK](../../../../glossary.md#term-java-development-kit-jdk)/[cache](../../../../glossary.md#term-cache), configured [Metal](../../../../glossary.md#term-metal)):
  `JAVA_HOME=/Users/phujka/Library/Java/JavaVirtualMachines/openjdk-26.0.1/Contents/Home SYNAPTIK_METAL_TEST_LIBRARY=/Users/phujka/IdeaProjects/Synaptik/modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew test`
  PASS, 12 s, 94 tasks: 3 executed/91 UP-TO-DATE. Actual XML: architecture 25 tests/0 skips;
  integration 184 tests/1 skip (existing no-eligible-CPU-tuning-alternatives assumption); zero failures/errors.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
  `JAVA_HOME=/Users/phujka/Library/Java/JavaVirtualMachines/openjdk-26.0.1/Contents/Home ./gradlew javadoc`
  PASS, 539 ms, 51 tasks all UP-TO-DATE; own-module docs generated previously, no Java/Javadoc comments changed afterward.
- Initial root failure: stale EngineComposition integration-edge expectation fixed to the exact Text test edge; missing Metal environment corrected, no [backend](../../../../glossary.md#backend) code changes.
- Independent documentation APPROVE: `docs/api/text-api.md` (135 lines), shared API/index/glossary,
  boundary/dependency/layout explanations, root and ADR delivered wording reviewed; Javadoc accurate unchanged.
  Path/heading-anchor checks PASS, 22 links; scoped whitespace PASS; examples match source/numerical evidence.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [dependency glossary](../../../../glossary.md#term-dependency); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- HEAD `ef35f2ec001b0c161d1abe825ff3be725b9be49e` plus accepted uncommitted Data0001–0003/Text0001,
  not merged. No persistence/training/runtime-length/scan extension or performance claim; no acceptance gaps or follow-up.<span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence); [runtime glossary](../../../../glossary.md#term-runtime))</span>

Status: Complete
