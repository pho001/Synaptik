# Task 0005B: Whole-model save and automatic inference load

## Status

Complete — 2026-10-15. Frozen implementation on accepted packet base `7d65f660051becc9bc6adc6f08af155b08096b59`,
independent code/documentation APPROVE and recorded module/Javadoc/coordinator gates. No successor authorized.

## Change class

Class C — new persistent kind and automatic semantic reconstruction/publication boundary.

## Goal

Save an ordinary supported Sequential or DeclarativeModel and load a usable heap-backed evaluation
DeclarativeModel without manual reconstruction, callbacks, digests or Arena plumbing. Deliver codec,
existing-facade file/envelope APIs, tests, Javadoc and focused docs as one Checkpoint capability.

## Scope and frozen API

The [owning whole-model contract](../../../../architecture/contracts/extensions-training.md#whole-model-definition-and-composition)
is the sole detailed authority for API, failures, wire fields, limits, preflight and coherence.
Add these static methods to existing ModelCheckpoint; file calls throw IOException, envelope calls
ArtifactFormatException. Module roots are recognized ONLY as Sequential or DeclarativeModel.

```text
void save(Path, Module, Engine)
void save(Path, Module, Engine, ArtifactLimits)
void save(Path, Module, Engine, InputArtifactFingerprint, ArtifactLimits)
DeclarativeModel load(Path)
DeclarativeModel load(Path, ArtifactLimits)
DeclarativeModel load(Path, InputArtifactFingerprint, ArtifactLimits)
ArtifactEnvelope captureWholeModel(Module, Engine, ArtifactLimits)
ArtifactEnvelope captureWholeModel(Module, Engine, InputArtifactFingerprint, ArtifactLimits)
DeclarativeModel loadWholeModel(ArtifactEnvelope, ArtifactLimits)
DeclarativeModel loadWholeModel(ArtifactEnvelope, InputArtifactFingerprint, ArtifactLimits)
```

InputArtifactFingerprint is final: public InputArtifactFingerprint(byte[]) constructor, defensive byte[] bytes(), value
equals/hashCode, exactly 32 SHA-256 bytes. Explicit overloads require non-null owner-verified
identity. Others save absence and reject stored presence. Existing values-only APIs stay unchanged.
Nulls: NullPointerException; invalid source/limits: IllegalArgumentException; unbound/Engine contract:
IllegalStateException; capture arithmetic overflow: ArithmeticException. Stored malformed/unsupported/
incoherent/over-limit data and fingerprint mismatch: ArtifactFormatException, owner failures wrapped
with cause; file I/O/publication: IOException. Tensor-ID exhaustion stays IllegalStateException.

## Non-goals

No Training recipe/state/run, recurrence, branches/custom callback or record I/O, registry,
reflection/Java serialization, backend discovery/compile/device format, new state installer,
public image wrapper/validator/credential, zero-copy publication or generic transaction.
No topology guessing from model.v1; no changes to ArtifactEnvelope or old wire/methods.
No backend rank-16 limit, optimized RecurrentScan/backend BPTT/Data prerequisite or performance campaign.

## Contracts

- Root exact headings: [Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence),
  [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts),
  [Module-ownership routing](../../../../../ARCHITECTURE.md#module-ownership-routing),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules),
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- Scoped exact headings: [extensions/nn](../../../../architecture/contracts/extensions-training.md#extensionsnn),
  [Closed declarative models](../../../../architecture/contracts/extensions-training.md#closed-declarative-models),
  [extensions/checkpoint](../../../../architecture/contracts/extensions-training.md#extensionscheckpoint),
  [Whole-model definition and composition](../../../../architecture/contracts/extensions-training.md#whole-model-definition-and-composition),
  [Optimizer/training lifecycle](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)
  (quiescence/no installation into open Training session only).
- [ADR 0038 accepted CP scope](../../../../design/decisions/0038-declarative-model-and-training-resume.md#accepted-checkpoint-decision).

Stop on real conflict or ambiguity; plans and ADR do not override these contracts.

## Dependencies and integration

- Depends on: NN0026 and Checkpoint0002/0004/0004A/0005A Complete, including required review/docs/
  gates; accepted whole-model contract with final independent approval and coordinator verification.
- Conflicts with: NN/Checkpoint semantic/wire/API and shared authoritative/explanatory documents;
  NN0027, CP0005C, Training0006 and adapter0006 serialized behind this capability.
- Parallel group: None; separate clean implementation and independent review contexts.
- Common base revision: source `f892cd3fbde4f8b7722b7eb5b14f4b0a5f3b2669` plus accepted contract
  packet; coordinator records the final contract commit in the launcher after integration.
- Integration order: NN0026 → CP0005B → NN0027 semantic recurrent reconstruction
  → CP0005C versioned persistence → Training0006 → optional adapter CP0006; CP0005A also feeds adapter.
- Integration validation: architecture + WholeModelCheckpointEngineIntegrationTest after review; one combined
  source-matched test/Javadoc checkpoint at the program join after adapter integration.
- Shared-document integration owner: NN/Training/Checkpoint coordinator.
- Authorized frontier: None after CP0005B completion; successors remain Draft pending separate gates.

Verified 2026-10-14 by coordinator: stable source HEAD/base f892cd3fbde4f8b7722b7eb5b14f4b0a5f3b2669;
NN0026 Complete with 326 NN/2 CPU/10 architecture tests and final reviews; CP0002/0004/0004A/0005A
Complete with review/docs/gates. Exact root/scoped headings above and ADR0038 CP scope are accepted;
final independent contract review APPROVE on all 13 packet docs, no P1/P2, 276 links validated.
User-approved serial order and sole Checkpoint frontier verified; no conflicting code writers;
NN/Engine/Model are still exactly the three direct public project edges. Execution base is source
plus accepted packet; final contract commit belongs in the post-integration launcher, not guessed here.

## Files and symbols

All production names below are in `io.github.pho001.synaptik.checkpoint.model`:

- Existing `extensions/checkpoint/src/main/java/.../checkpoint/model/ModelCheckpoint.java`: overloads.
- New package-private final `WholeModelDefinitionCodec`: dry sizing, fixed wire IDs, encode/decode.
- New package-private final `WholeModelComposition`: outer preflight/projection and owner coherence.
- New public final `InputArtifactFingerprint`: the only new public production type.
- Existing public `checkpoint.artifact.{ArtifactFile,ArtifactEnvelope,ArtifactLimits}` reused unchanged;
  public ModelCheckpoint.capture/inspect/canonicalStateBytes/load are the component bridge.
- Read NN `nn.module.{ModelSpecification,DeclarativeModel,Sequential}` and their focused tests;
  read Model ScalarValue and Conv1d/2d/3dAttrs only for exact codec semantics.
- Extend Checkpoint model/artifact/API unit tests for codec/composition/limits/corruption.
- Add `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/WholeModelCheckpointEngineIntegrationTest.java`
  in existing integration-test package for the new end-to-end user workflow. Existing NN/CPU/
  Checkpoint test edges suffice; no build/whitelist change or NN production change.

## Acceptance criteria

- Reject custom/unbound source before Engine/filesystem work. Describe Sequential normally, or
  DeclarativeModel's current bindings, under whole-call caller quiescence. Description + public
  values capture may read state twice; promise neither one combined snapshot nor a transaction.
- Implement the exact owning canonical definition: SYNDEF01/i32 v1, prefix Sequence tree with
  eight closed kind IDs/complete attributes/guard/scalar raw bits, then canonical alias groups.
  Numeric traversal paths, not lexical order; no node IDs, edge tables or cycle fields.
- Keep DataType IDs 1 F64, 2 F32, 3 BF16, 4 I32, 5 I64, 6 BOOL, 7 F16; only floating parameters.
  Unknown recurrent/type/enum/version IDs fail closed; future adoption explicitly versions format.
- Outer whole-model.v1 = definition/schema + model/ prefixed UNCHANGED model.v1 entries.
  Configuration SHA-256 is exact definition bytes, automatically computed, not authentication.
- Default ArtifactLimits: (268435456L,4194304,16384,64,67108864,268435456L); fixed definition
  ceiling 1048576 bytes under advanced limits too. NN occurrence/depth 4096/64 remain fixed.
  Limits bound encoded file/aggregate/manifest/entries/keys/single payload, not peak heap.
- For n inner entries (states + schema), delta manifest = 67+6n; file delta = D+67+6n;
  aggregate delta D; entries delta 1; projected key limit outer minus 6. Check outer kind/definition
  keys and D separately; maxPayload stays unchanged. Checked residual file ≥48, manifest ≥8 and
  valid other minima before constructing projected limits. Public capture's preflight must ensure
  outer bounds BEFORE Engine/bulk value copies without duplicating private schema/ordinal parsing.
  Order: dry-size/check D and residuals → encode bounded ≤1 MiB metadata → SHA → public capture
  (actual component preflight) → recheck complete outer/coherence → prefixed payload copies.
  Test each outer-only boundary where the unprefixed component fits; rejection must precede
  Engine admission/bulk values (including residuals below constructor minima).
- Envelope-load outer preflight precedes copies/projection. Before decoded allocation, bound
  counts, paths and LayerNorm rank by remaining definition bytes (including scalar suffix), not
  backend rank 16. Reject strict-UTF8 errors, unknown fields, noncanonical flags/order, duplicates,
  extra/missing keys, trailing bytes, overflow, depth/count and every encoded-bound violation.
- Independently use public inspect/canonicalStateBytes to compare complete ordered descriptors,
  kinds/types/Shapes/byte lengths, config/input digests and every alias group's exact raw bytes.
  Reconstruct private NN descriptor skeleton, public target-load, return only after complete success.
  No private component parser, temp files, trusted inspection token or weakened defensive copy.
- Return independent heap-backed eval model; parameters retain gradient eligibility. Load does
  no Engine/autograd/compile/native ownership/RNG/throwaway initialization. Test after Engine/source
  Arena/file closure, alias joins/splits on re-save, all layer attrs/floating types/scalar bits,
  variable leading dimensions, Embedding integral inputs and direct CPU inference/re-save.
  Named integration fixture: ordinary Sequential save → close source Arena/Engine → automatic
  load → CPU compute with a fresh Engine; cover exact aliases, variable batches and loaded re-save.
- Empty/activation-only/identity semantic trees are allowed; empty state makes no Engine call.
  Do not claim generic zero-node admission. Old values-only load gives target-load diagnostic.
- Reuse ArtifactFile force/revalidate/one atomic move; failures return no partial model and do not
  mutate published models. Preserve existing file failure/cleanup semantics and wire compatibility.

## Validation

Worker: focused new codec/composition/public API tests, then once (completed):

```bash
./gradlew :extensions:checkpoint:test :extensions:checkpoint:javadoc --max-workers=2 --console=plain
```

Coordinator ran these distinct gates once after independent CP review (no duplicate NN/module
tests or performance campaign):

```bash
./gradlew :testing:architecture-tests:test --max-workers=2 --console=plain
./gradlew :testing:integration-tests:test --tests io.github.pho001.synaptik.testing.integration.WholeModelCheckpointEngineIntegrationTest --max-workers=2 --console=plain
```

Reuse worker evidence unless integration changed code. At program join:
`./gradlew test javadoc --max-workers=2 --console=plain`.
No dependency change: no new architecture edge test/build inclusion needed for CP0005B.

## Documentation and review impact

Independent Class C contract/API/wire/preflight and implementation/documentation reviews APPROVE.
Coordinator owns root/scoped contract, architecture explanations, ADR and planning integration.
Implementation updates `docs/api/model-checkpoint-api.md`, running/declarative guides, Javadoc
and targeted glossary; status reconciliation changes no accepted semantics or future decisions.

## Result

Seven code/test files deliver ten ModelCheckpoint overloads, sole public InputArtifactFingerprint,
package-private codec/composition and unit/integration coverage. Existing values-only APIs/wire unchanged.
Worker `:extensions:checkpoint:test :extensions:checkpoint:javadoc --max-workers=2 --console=plain`
PASS: 57 tests (24 new), no failures/errors/skips; 3 tasks executed, 31 up-to-date. Tests not rerun by docs.
Coordinator architecture command above PASS: 21 up-to-date; valid prior 10-test evidence reused,
no dependency/architecture-test input change. Focused integration command above actually ran PASS:
3 tests, no failures/errors/skips; XML 2026-10-15T05:14:25Z; 2 tasks executed, 35 up-to-date.
Independent final code APPROVE covers all seven frozen files; no unresolved production finding.
Independent docs review finalized ModelCheckpoint/InputArtifactFingerprint comments only, API,
running/declarative guides, glossary, accepted-contract/explanation/ADR and current plan statuses.
Final `./gradlew :extensions:checkpoint:javadoc --max-workers=2 --console=plain` PASS after comments:
2 tasks executed, 15 up-to-date; no behavior change (comment-stripped hashes unchanged).
Docs links/anchors (524), whitespace including untracked files and brief size PASS; running-guide
Java 26 example executed: `0.0, 8.0`. No unresolved CP0005B follow-up; no commit/stage/push.
NN0027/CP0005C/Training0006/adapter remain Draft/Proposed; no recurrent wire or API expansion.
Status: Complete
