# Task 0005B: Whole-[model](../../../../glossary.md#term-model) save and automatic [inference](../../../../glossary.md#term-inference) load

> **Editorial reading context — not a status update.** Unlike values-only restore into an
> application-built target, whole-model load reads a supported layer definition as well as state.
> NN reconstructs the supported structure; the checkpoint layer validates and installs values
> into a private candidate before returning it. “Usable” does not mean already prepared for a
> backend: numerical execution still needs Engine, and optimizer history is absent.
> An input fingerprint is an identity the application has verified, not proof Checkpoint
> inspected the external tokenizer or dataset. Compare today's
> [whole-model workflow](../../../../reference/checkpoint/models.md#whole-model-save-and-automatic-load) and
> [expected input identity](../../../../reference/checkpoint/models.md#expected-input-identity).

## Status

Complete — 2026-10-15. Frozen implementation on accepted packet base `7d65f660051becc9bc6adc6f08af155b08096b59`,
independent code/documentation APPROVE and recorded [module](../../../../glossary.md#term-module)/[Javadoc](../../../../glossary.md#term-javadoc)/coordinator gates. No successor authorized.

## Change class

Class C — new persistent kind and automatic semantic reconstruction/[publication](../../../../glossary.md#term-publication) boundary.

## Goal

Save an ordinary supported Sequential or DeclarativeModel and load a usable [heap](../../../../glossary.md#term-heap)-backed [evaluation](../../../../glossary.md#term-evaluation)
DeclarativeModel without manual reconstruction, callbacks, digests or [Arena](../../../../glossary.md#term-arena) plumbing. Deliver [codec](../../../../glossary.md#term-codec),
existing-[facade](../../../../glossary.md#term-facade) file/envelope APIs, tests, [Javadoc](../../../../glossary.md#term-javadoc) and focused docs as one [Checkpoint](../../../../glossary.md#term-checkpoint) capability.

## [Scope](../../../../glossary.md#term-scope) and frozen [API](../../../../glossary.md#term-application-programming-interface-api)

The [owning whole-model contract](../../../../architecture/contracts/extensions-training.md#whole-model-definition-and-composition)<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>
is the sole detailed authority for API, failures, wire fields, limits, [preflight](../../../../glossary.md#term-preflight) and coherence.
Add these static methods to existing ModelCheckpoint; file calls throw IOException, envelope calls
ArtifactFormatException. [Module](../../../../glossary.md#term-module) roots are recognized ONLY as Sequential or DeclarativeModel.

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
equals/[hashCode](../../../../glossary.md#hash-codes-and-content-digests), exactly 32 [SHA-256](../../../../glossary.md#term-sha-256) bytes. Explicit overloads require non-null [owner](../../../../glossary.md#term-owner)-verified
[identity](../../../../glossary.md#term-identity-meanings). Others save absence and reject stored presence. Existing values-only APIs stay unchanged.
Nulls: NullPointerException; invalid source/limits: IllegalArgumentException; unbound/[Engine](../../../../glossary.md#term-engine) contract:
IllegalStateException; [capture](../../../../glossary.md#term-capture-meanings) arithmetic [overflow](../../../../glossary.md#term-overflow): ArithmeticException. Stored malformed/unsupported/
incoherent/over-limit data and [fingerprint](../../../../glossary.md#term-fingerprint) mismatch: ArtifactFormatException, owner failures wrapped
with cause; file I/O/[publication](../../../../glossary.md#term-publication): IOException. [Tensor](../../../../glossary.md#tensor)-ID exhaustion stays IllegalStateException.

## Non-goals

No Training recipe/state/[run](../../../../glossary.md#term-run), recurrence, branches/custom [callback](../../../../glossary.md#term-callback) or record I/O, [registry](../../../../glossary.md#term-registry),
reflection/Java [serialization](../../../../glossary.md#term-serialization), [backend](../../../../glossary.md#backend) discovery/[compile](../../../../glossary.md#compile)/[device](../../../../glossary.md#term-device) format, new state installer,
public image wrapper/validator/credential, zero-copy [publication](../../../../glossary.md#term-publication) or generic [transaction](../../../../glossary.md#term-transaction).
No [topology](../../../../glossary.md#term-topology) guessing from [model](../../../../glossary.md#term-model).v1; no changes to ArtifactEnvelope or old wire/methods.
No backend [rank](../../../../glossary.md#term-rank)-16 limit, optimized RecurrentScan/backend [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt)/Data prerequisite or performance campaign.

## Contracts

- Root exact headings: [Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence),
  [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [normative glossary](../../../../glossary.md#term-normative))</span>,
  [Module-ownership routing](../../../../../ARCHITECTURE.md#module-ownership-routing)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>,
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>,
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- Scoped exact headings: [extensions/nn](../../../../architecture/contracts/extensions-training.md#extensionsnn),
  [Closed declarative models](../../../../architecture/contracts/extensions-training.md#closed-declarative-models),
  [extensions/checkpoint](../../../../architecture/contracts/extensions-training.md#extensionscheckpoint)<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>,
  [Whole-model definition and composition](../../../../architecture/contracts/extensions-training.md#whole-model-definition-and-composition)<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model))</span>,
  [Optimizer/training lifecycle](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)<span data-glossary-navigation="true"> ([Optimizer glossary](../../../../glossary.md#term-optimizer); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
  (quiescence/no installation into open Training session only).<span data-glossary-navigation="true"> ([quiescence glossary](../../../../glossary.md#term-quiescence); [Training session glossary](../../../../glossary.md#training-session--trainingsession))</span>
- [ADR 0038 accepted CP scope](../../../../design/decisions/0038-declarative-model-and-training-resume.md#accepted-checkpoint-decision).

Stop on real conflict or ambiguity; plans and [ADR](../../../../glossary.md#term-architecture-decision-record-adr) do not override these contracts.

## Dependencies and integration

- Depends on: NN0026 and Checkpoint0002/0004/0004A/0005A Complete, including required review/docs/
  gates; accepted whole-model contract with final independent approval and coordinator verification.
- Conflicts with: NN/Checkpoint semantic/wire/API and shared authoritative/explanatory documents;
  NN0027, CP0005C, Training0006 and adapter0006 serialized behind this capability.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Parallel group: None; separate clean implementation and independent review contexts.
- Common base revision: source `f892cd3fbde4f8b7722b7eb5b14f4b0a5f3b2669` plus accepted contract
  packet; coordinator records the final contract commit in the launcher after integration.
- Integration order: NN0026 → CP0005B → NN0027 semantic recurrent reconstruction
  → CP0005C versioned persistence → Training0006 → optional adapter CP0006; CP0005A also feeds adapter.<span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence))</span>
- Integration validation: architecture + WholeModelCheckpointEngineIntegrationTest after review; one combined
  source-matched test/Javadoc checkpoint at the program join after adapter integration.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training/[Checkpoint](../../../../glossary.md#term-checkpoint) coordinator.
- [Authorized frontier](../../../../glossary.md#term-authorized-frontier): None after CP0005B completion; successors remain Draft pending separate gates.

Verified 2026-10-14 by coordinator: stable source HEAD/base f892cd3fbde4f8b7722b7eb5b14f4b0a5f3b2669;
NN0026 Complete with 326 NN/2 [CPU](../../../../glossary.md#term-cpu)/10 architecture tests and final reviews; CP0002/0004/0004A/0005A
Complete with review/docs/gates. Exact root/scoped headings above and ADR0038 CP [scope](../../../../glossary.md#term-scope) are accepted;
final independent contract review APPROVE on all 13 packet docs, no P1/P2, 276 links validated.
User-approved serial order and sole Checkpoint frontier verified; no conflicting code writers;
NN/[Engine](../../../../glossary.md#term-engine)/[Model](../../../../glossary.md#term-model) are still exactly the three direct public project edges. Execution base is source
plus accepted packet; final contract [commit](../../../../glossary.md#term-commit) belongs in the post-integration launcher, not guessed here.

## Files and symbols

All production names below are in `io.github.pho001.synaptik.checkpoint.model`:

- Existing `extensions/checkpoint/src/main/java/.../checkpoint/model/ModelCheckpoint.java`: overloads.
- New package-private final `WholeModelDefinitionCodec`: [dry sizing](../../../../glossary.md#term-dry-sizing), fixed wire IDs, encode/decode.
- New package-private final `WholeModelComposition`: outer [preflight](../../../../glossary.md#term-preflight)/projection and [owner](../../../../glossary.md#term-owner) coherence.
- New public final `InputArtifactFingerprint`: the only new public production type.
- Existing public `checkpoint.artifact.{ArtifactFile,ArtifactEnvelope,ArtifactLimits}` reused unchanged;
  public ModelCheckpoint.capture/inspect/canonicalStateBytes/load are the component bridge.<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- Read [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) `nn.module.{ModelSpecification,DeclarativeModel,Sequential}` and their focused tests;
  read Model ScalarValue and Conv1d/2d/3dAttrs only for exact codec semantics.<span data-glossary-navigation="true"> ([ScalarValue glossary](../../../../glossary.md#typed-scalar-value--scalarvalue))</span>
- Extend [Checkpoint](../../../../glossary.md#term-checkpoint) [model](../../../../glossary.md#term-model)/[artifact](../../../../glossary.md#term-artifact)/[API](../../../../glossary.md#term-application-programming-interface-api) unit tests for [codec](../../../../glossary.md#term-codec)/composition/limits/corruption.
- Add `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/WholeModelCheckpointEngineIntegrationTest.java`
  in existing integration-test package for the new end-to-end user workflow. Existing NN/CPU/
  Checkpoint test edges suffice; no build/whitelist change or NN production change.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>

## Acceptance criteria

- Reject custom/unbound source before Engine/filesystem work. Describe Sequential normally, or
  DeclarativeModel's current bindings, under whole-call caller quiescence. Description + public
  values capture may read state twice; promise neither one combined snapshot nor a transaction.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [bindings glossary](../../../../glossary.md#term-binding); [quiescence glossary](../../../../glossary.md#term-quiescence); [capture glossary](../../../../glossary.md#term-capture-meanings); [snapshot glossary](../../../../glossary.md#term-snapshot); [transaction glossary](../../../../glossary.md#term-transaction))</span>
- Implement the exact owning canonical definition: SYNDEF01/i32 v1, prefix Sequence tree with
  eight closed kind IDs/complete attributes/guard/scalar raw bits, then canonical alias groups.
  Numeric traversal paths, not lexical order; no node IDs, edge tables or cycle fields.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [guard glossary](../../../../glossary.md#term-guard); [scalar glossary](../../../../glossary.md#term-scalar); [raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [alias glossary](../../../../glossary.md#aliasing-and-arena); [node glossary](../../../../glossary.md#node))</span>
- Keep DataType IDs 1 F64, 2 F32, 3 BF16, 4 I32, 5 I64, 6 BOOL, 7 F16; only floating parameters.
  Unknown recurrent/type/enum/version IDs fail closed; future adoption explicitly versions format.<span data-glossary-navigation="true"> ([DataType glossary](../../../../glossary.md#data-type--datatype); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [BF16 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [BOOL glossary](../../../../glossary.md#data-type--datatype); [F16 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [parameters glossary](../../../../glossary.md#term-parameter); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
- Outer whole-model.v1 = definition/schema + model/ prefixed UNCHANGED model.v1 entries.
  Configuration SHA-256 is exact definition bytes, automatically computed, not authentication.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [schema glossary](../../../../glossary.md#term-schema); [SHA-256 glossary](../../../../glossary.md#term-sha-256); [authentication glossary](../../../../glossary.md#term-authentication))</span>
- Default ArtifactLimits: (268435456L,4194304,16384,64,67108864,268435456L); fixed definition
  ceiling 1048576 bytes under advanced limits too. NN occurrence/depth 4096/64 remain fixed.
  Limits bound encoded file/aggregate/manifest/entries/keys/single payload, not peak heap.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [occurrence glossary](../../../../glossary.md#term-occurrence); [manifest glossary](../../../../glossary.md#term-manifest); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [heap glossary](../../../../glossary.md#term-heap))</span>
- For n inner entries (states + schema), delta manifest = 67+6n; file delta = D+67+6n;
  aggregate delta D; entries delta 1; projected key limit outer minus 6. Check outer kind/definition
  keys and D separately; maxPayload stays unchanged. Checked residual file ≥48, manifest ≥8 and
  valid other minima before constructing projected limits. Public capture's preflight must ensure
  outer bounds BEFORE Engine/bulk value copies without duplicating private schema/ordinal parsing.
  Order: dry-size/check D and residuals → encode bounded ≤1 MiB metadata → SHA → public capture
  (actual component preflight) → recheck complete outer/coherence → prefixed payload copies.
  Test each outer-only boundary where the unprefixed component fits; rejection must precede
  Engine admission/bulk values (including residuals below constructor minima).<span data-glossary-navigation="true"> ([residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand); [preflight glossary](../../../../glossary.md#term-preflight); [metadata glossary](../../../../glossary.md#term-metadata); [admission glossary](../../../../glossary.md#term-admission))</span>
- Envelope-load outer preflight precedes copies/projection. Before decoded allocation, bound
  counts, paths and LayerNorm rank by remaining definition bytes (including scalar suffix), not
  backend rank 16. Reject strict-UTF8 errors, unknown fields, noncanonical flags/order, duplicates,
  extra/missing keys, trailing bytes, overflow, depth/count and every encoded-bound violation.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [backend glossary](../../../../glossary.md#backend); [overflow glossary](../../../../glossary.md#term-overflow))</span>
- Independently use public inspect/canonicalStateBytes to compare complete ordered descriptors,
  kinds/types/Shapes/byte lengths, config/input digests and every alias group's exact raw bytes.
  Reconstruct private NN descriptor skeleton, public target-load, return only after complete success.
  No private component parser, temp files, trusted inspection token or weakened defensive copy.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [Shapes glossary](../../../../glossary.md#shape); [target glossary](../../../../glossary.md#term-target); [token glossary](../../../../glossary.md#term-token); [defensive copy glossary](../../../../glossary.md#term-defensive-copy))</span>
- Return independent heap-backed eval model; parameters retain gradient eligibility. Load does
  no Engine/autograd/compile/native ownership/RNG/throwaway initialization. Test after Engine/source
  Arena/file closure, alias joins/splits on re-save, all layer attrs/floating types/scalar bits,
  variable leading dimensions, Embedding integral inputs and direct CPU inference/re-save.
  Named integration fixture: ordinary Sequential save → close source Arena/Engine → automatic
  load → CPU compute with a fresh Engine; cover exact aliases, variable batches and loaded re-save.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [autograd glossary](../../../../glossary.md#autograd); [compile glossary](../../../../glossary.md#compile); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [Arena glossary](../../../../glossary.md#term-arena); [dimensions glossary](../../../../glossary.md#dimension); [Embedding glossary](../../../../glossary.md#embedding); [CPU glossary](../../../../glossary.md#term-cpu); [inference glossary](../../../../glossary.md#term-inference); [fixture glossary](../../../../glossary.md#term-fixture); [batches glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- Empty/activation-only/identity semantic trees are allowed; empty state makes no Engine call.
  Do not claim generic zero-node admission. Old values-only load gives target-load diagnostic.<span data-glossary-navigation="true"> ([activation glossary](../../../../glossary.md#term-activation); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Reuse ArtifactFile force/revalidate/one atomic move; failures return no partial model and do not
  mutate published models. Preserve existing file failure/cleanup semantics and wire compatibility.<span data-glossary-navigation="true"> ([atomic move glossary](../../../../glossary.md#term-atomic-move))</span>

## [Validation](../../../../glossary.md#term-validation)

Worker: focused new [codec](../../../../glossary.md#term-codec)/composition/public [API](../../../../glossary.md#term-application-programming-interface-api) tests, then once (completed):

```bash
./gradlew :extensions:checkpoint:test :extensions:checkpoint:javadoc --max-workers=2 --console=plain
```

Coordinator ran these distinct gates once after independent CP review (no duplicate [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/[module](../../../../glossary.md#term-module)
tests or performance campaign):

```bash
./gradlew :testing:architecture-tests:test --max-workers=2 --console=plain
./gradlew :testing:integration-tests:test --tests io.github.pho001.synaptik.testing.integration.WholeModelCheckpointEngineIntegrationTest --max-workers=2 --console=plain
```

Reuse worker evidence unless integration changed code. At program join:
`./gradlew test javadoc --max-workers=2 --console=plain`.
No [dependency](../../../../glossary.md#term-dependency) change: no new architecture edge test/build inclusion needed for CP0005B.

## Documentation and review impact

Independent Class C contract/[API](../../../../glossary.md#term-application-programming-interface-api)/wire/[preflight](../../../../glossary.md#term-preflight) and implementation/documentation reviews APPROVE.
Coordinator owns root/scoped contract, architecture explanations, [ADR](../../../../glossary.md#term-architecture-decision-record-adr) and [planning](../../../../glossary.md#term-planning) integration.
Implementation updates `docs/api/model-checkpoint-api.md`, running/declarative guides, [Javadoc](../../../../glossary.md#term-javadoc)
and targeted glossary; status reconciliation changes no accepted semantics or future decisions.

## Result

Seven code/test files deliver ten ModelCheckpoint overloads, sole public InputArtifactFingerprint,
package-private [codec](../../../../glossary.md#term-codec)/composition and unit/integration coverage. Existing values-only APIs/wire unchanged.
Worker `:extensions:checkpoint:test :extensions:checkpoint:javadoc --max-workers=2 --console=plain`
PASS: 57 tests (24 new), no failures/errors/skips; 3 tasks executed, 31 up-to-date. Tests not rerun by docs.
Coordinator architecture command above PASS: 21 up-to-date; valid prior 10-test evidence reused,
no [dependency](../../../../glossary.md#term-dependency)/architecture-test input change. Focused integration command above actually ran PASS:
3 tests, no failures/errors/skips; XML 2026-10-15T05:14:25Z; 2 tasks executed, 35 up-to-date.
Independent final code APPROVE covers all seven frozen files; no unresolved production finding.
Independent docs review finalized ModelCheckpoint/InputArtifactFingerprint comments only, [API](../../../../glossary.md#term-application-programming-interface-api),
running/declarative guides, glossary, accepted-contract/explanation/[ADR](../../../../glossary.md#term-architecture-decision-record-adr) and current plan statuses.
Final `./gradlew :extensions:checkpoint:javadoc --max-workers=2 --console=plain` PASS after comments:
2 tasks executed, 15 up-to-date; no behavior change (comment-stripped [hashes](../../../../glossary.md#term-hash) unchanged).
Docs links/anchors (524), whitespace including untracked files and brief size PASS; running-guide
Java 26 example executed: `0.0, 8.0`. No unresolved CP0005B follow-up; no [commit](../../../../glossary.md#term-commit)/stage/push.
NN0027/CP0005C/Training0006/adapter remain Draft/Proposed; no recurrent wire or API expansion.
Status: Complete
