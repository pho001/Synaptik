# Task 0006: Stored Model + TrainingState adapter

## Status

Draft — concrete proposal for approval, not implementation authorization.
Single remaining decision: approve the API/wire below and optional adapter build gate together.
No executor may launch or mark this Ready before approval and the authoritative contract decision.

## Change class

Class C implementation — new downstream module edge, durable wire, and cross-owner restore workflow.
This planning-only change alters no normative contract, build, Java, or test.

## Goal

Persist ONE file containing Model + TrainingState and restore its stored state through existing
owners. A detached snapshot means immutable bytes, not an automatically resumable live session.

## Scope and non-goals

Propose optional `extensions/training-checkpoint`, with exactly two direct public project edges:
`api(project(":extensions:training"))` and `api(project(":extensions:checkpoint"))`.
NN/Model/Engine types are transitive; no reverse/lower/backend edge. Reuse public Artifact/Model/
Training APIs without changing `model.v1` or envelope v1 or accessing private codecs.
Exclude progress/cursor/epoch, RNG, modes, effective ForwardContexts, objective construction,
scheduler/scaler, compiled/prepared/device state: all remain application-owned.
No automatic exact continuation, generic participants/transactions/serializers, new pure Training
validator, general options/limits class, cross-device bitwise claim, or broad performance campaign.

## Contracts

- [Root authority](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence),
  [scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules), and
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- Owning scoped headings: [extensions/checkpoint](../../../../architecture/contracts/extensions-training.md#extensionscheckpoint),
  [extensions/training](../../../../architecture/contracts/extensions-training.md#extensionstraining),
  [Optimizer/training lifecycle](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle).
- [Proposed ADR 0037](../../../../design/decisions/0037-stored-training-checkpoint-adapter.md).

The current contract reserves but forbids inclusion/implementation before the separate adapter gate.
This proposal does not override it. Stop on any missing or ambiguous applicable contract.

## Dependencies and integration

- Depends on: completed Checkpoint 0002, 0004, 0004A, 0005, 0005A; completed Training 0004.
- Dependency verification: satisfied against current public source/tests/contracts at the base below.
- Conflicts with: adapter wire/API, owning contracts, build inclusion, shared Training/Checkpoint docs.
- Parallel group: None; serialize contract cutover and implementation.
- Common base revision: `6bbd17321e246c9c7ded90f6931ffcfc7d344711`.
- Integration order: approval and independently Class C-reviewed authoritative update before Ready;
  clean implementation delivers project/build/tests/docs in the same overall change, followed by
  independent Class C review and acceptance; no coordinator code work or new prerequisite task.
- Integration validation: named worker/coordinator commands below, with source-matched device library.
- Shared-document integration owner: Training/Checkpoint coordinator.
- Authorized frontier: None while Draft; satisfied predecessors do not authorize launch.

## Proposed production types and facade

In optional package `io.github.pho001.synaptik.training.checkpoint`: public final `TrainingCheckpoint`
and its public final nested `Snapshot`; package-private
`TrainingArtifactCodec` (metadata/slots), `TrainingArtifactComposition` (namespace/size preflight),
and `TrainingParameterMatch` (complete path/descriptor/logical-byte comparison); no extra public type.

Proposed signatures (not current APIs):

```text
TrainingCheckpoint.save(Path, ArtifactEnvelope model, TrainingState, ArtifactLimits): void throws IOException
TrainingCheckpoint.read(Path, ArtifactLimits): Snapshot throws IOException
Snapshot.modelInspection(): ModelCheckpointInspection
Snapshot.trainingState(): TrainingState
Snapshot.loadModel(Module privateTarget, byte[] configDigest, byte[] inputDigest, ArtifactLimits): void throws ArtifactFormatException
Snapshot.restoreTraining(TrainingSession privateSession): void
```

Snapshot is privately constructed after complete checks, owning detached envelope/inspection and
TrainingState; accessors expose immutable values, never borrowed Module/Engine/session/storage.
Read/accessors prove neither whole-model installation nor a restorable-session credential.
`loadModel` independently delegates ModelCheckpoint.load with supplied limits and digests; existing
bound-target/alias policy and one strict NN install remain unchanged. Null inputDigest means absent.
`restoreTraining` obtains privateSession.state(), compares the complete parameter path set, types,
shapes and exact logical bytes with saved state, then calls privateSession.restore(saved).
These are TWO separately admitted operations: caller grants exclusive candidate/session use across
both; no new lock or atomic admission is claimed. Preserve Training lifecycle errors and its
validate-before-install guarantee. Mismatch is IllegalArgumentException, not partial installation.
This checks only session parameters: the caller must loadModel onto the SAME private tree before
session open and keep its loaded buffers unchanged through restore/publication; no identity token.

## Proposed canonical wire

Outer kind `training-checkpoint.v1`; framing remains ArtifactFile v1. Prefix every supplied
`model.v1` entry key with `model/`, preserving payload bytes exactly. Save validates the supplied
kind/schema through public ModelCheckpoint APIs. Add schema/slots below; reconstruct `model.v1`
by stripping exactly `model/`; use public ModelCheckpoint.inspect/canonicalStateBytes/load,
never parse private model ordinal metadata.
`training/schema` is BIG_ENDIAN, no padding/trailing bytes, in this order:

1. Eight ASCII bytes `SYNTRN01`, i32 version `1`, default optimizer record.
2. i32 group count (>=0); each group: optimizer record, i32 path count (>0), ordered paths.
3. i64 executions, i64 optimizerSteps, i64 accumulatedGradientRuns.
4. i32 parameter count (>0, <=100000000); each ordered parameter: path, u8 dtype ID,
   i32 static rank (>=0), rank i64 nonnegative extents, u8 second-slot presence (0/1),
   i64 logical length and i64 optimizer-width length.

Paths: i32 positive UTF-8 length + exact bytes; reject malformed UTF-8/surrogates; no normalization/sorting.
Optimizer record: u8 stable ID `1` SGD, `2` Adam, `3` AdamW. SGD has four raw IEEE binary64
coefficients in accessor order learningRate, momentum, dampening, weightDecay, then u8 nesterov
(0/1). Adam/AdamW have five: learningRate, beta1, beta2, epsilon, weightDecay.
Encode raw long bits, decode with longBitsToDouble and public constructors; preserve signed zero.
No enum IDs, NaN normalization or F32 narrowing; dtype IDs: `1` FLOAT64, `2` FLOAT32,
`3` BFLOAT16, `7` FLOAT16.

For parameter ordinal N, eight-digit ASCII decimal `00000000`..`99999999`, required keys are
`training/parameter/N/logical`, `/master`, `/first`, `/pending`, and `/second` iff presence=1
(each suffix shares the full `training/parameter/N` prefix). Accept exactly schema-derived ordinal/
presence keys, not arbitrary keys matching a prefix. Logical bytes have declared width;
all other slots use F64 for F64, F32 otherwise. Each entry length must equal its schema length,
which must equal checked Shape element count times width. Preserve every raw byte, including low
logical bytes distinct from masters, signed zeros, NaN payloads, and pending sums at zero count.

## Validation and allocation responsibilities

Save/read must compare the COMPLETE model parameter path set with TrainingState: exact paths,
PARAMETER kind, type, structural Shape and logical bytes, not a subset/schema-only comparison.
Buffers retain all seven types/raw bits. Preserve Training order; reject duplicate/missing/extra
paths, buffer substitution, mismatched type/shape/bytes, and unknown/extra/
missing/duplicate artifact keys or IDs, unsupported versions, truncation or trailing data.

Structural checks match constructors: nonempty parameter list, static shapes (zero-element allowed),
four floating types, exact widths, nonblank paths; 0<=steps,pending<=executions. ParameterGroup
enforces nonempty/distinct/nonblank LOCAL paths.
Global membership/cross-group overlap, algorithm-specific second-slot presence, slot finiteness,
nonnegative second moments, positive session element counts, coefficient narrowing, captured order,
and session compatibility remain Training.restore authority. NN path grammar for parameter entries
comes from complete model comparison; do not silently strengthen group-path constructor rules.
Read reconstructs through public grouped TrainingState and defensive ParameterState constructors.
Before payload-sized schema allocation or ANY Entry/ParameterState clone, preflight complete
projected outer AND reconstructed model framing, names, counts, schema/payload/aggregate/file sizes.
ArtifactLimits applies unchanged: conservative array ceiling Integer.MAX_VALUE-8, stricter than
ParameterState's ceiling. For each projection use checked long arithmetic:
manifest=8+kindASCIIlength+sum(44+keyASCIIlength); file=48+manifest+aggregate.
Names include prefixed outer keys and shorter reconstructed model keys/kind; independently enforce
all six limits on both projections. Do not expose private ArtifactWire to obtain sizing.
Bound EVERY nested count, rank and path length by remaining metadata/minimum record widths before
list/string/extent allocation, plus byte-array ceilings; group count is not bounded by maxEntries
alone. Do a noncopying metadata/descriptor scan and complete expected-key/length plan before payload
copies. ArtifactFile.read itself bounds/checks outer bytes before returning its immutable envelope;
adapter scan precedes additional component/Training copies. Existing immutable values may already
copy; encoded canonical-byte limits are NOT a peak-heap budget or large-model support promise.

Malformed/unsupported/over-limit wire and cross-component save/read mismatch use checked
ArtifactFormatException (IOException subtype); wrap encoded constructor/arithmetic failures there.
Null caller arguments use NullPointerException; filesystem/force/move failures remain IOException.
Model load semantic/limit decode failures remain checked ArtifactFormatException; target/digest
mismatches and Training compatibility/lifecycle failures retain their documented unchecked types.
No partial Snapshot or trusted inspection. ArtifactFile.save retains temporary-file validation,
force and one atomic move; no fallback, target deletion, authentication or universal power-loss promise.

## Caller workflow and acceptance criteria

Under application exclusion spanning BOTH session.state() and ModelCheckpoint.capture, capture one
coherent pair including buffers, slots and counters; equality cannot prove capture quiescence.
Detached save performs no live captures. For recovery, read; create a distinct fully bound private
tree; loadModel on that SAME tree BEFORE session open; copy to supported native storage if needed;
apply app modes/contexts and build objective from final Tensors; open private session; restoreTraining
under exclusion; validate app handoff; publish replacement only after success. Never load NN bindings
while that tree's session is open. Failure closes/discards candidate and preserves published tree.
Example acceptance: CPU save/read/reopen followed by the same caller-selected next input matches the
uninterrupted admitted route; mismatched candidate logical bytes reject before Training mutation.
This proves stored-state recovery, not absent external-state or cross-device exact continuation.

Tests cover all four Training dtypes, SGD/Adam/AdamW, ordered/mixed groups, raw coefficients/slots,
counters/pending sums/low masters; immutable snapshots after source closure; all seven buffer types;
complete path/type/Shape/raw-byte mismatches; structurally valid but session-invalid round trips;
owner rejection; every outer/projected limit and nested allocation bound; corruption/extra keys;
pre-move failure preserves target; failed private candidate preserves published tree;
CPU next-input recovery and actual Metal smoke with source-matched library (explicit skip if absent).

## Architecture cutover, documentation and validation

After approval, independently Class C-review root/owning-contract decision before Ready; update root
Dependency rules/ownership/Testing requirements, extensions/checkpoint + lifecycle; accept ADR 0037.
Clean implementation delivers the following build/tests/docs in that same overall change, including
`docs/architecture/{module-boundaries,dependency-rules,lifecycle}.md` and API/example/glossary as needed.
Include optional project in settings with exactly the two api edges. Update existing
`testing/architecture-tests/.../CheckpointDependencyContractTest.java`: replace absent-adapter
assertion with exact inclusion/two-edge checks and forbid reverse edges from NN/Training/Checkpoint,
modules/backends/tuning; retain Checkpoint's exact three-edge checks. Add no backend behavior.
Add adapter test dependency and durable private-candidate fixture to testing/integration-tests.
Clean implementation and independent targeted Class C review/docs are mandatory; Javadocs cover
every parameter/result/failure/exclusion responsibility. Reuse successful executable evidence.

Future worker: `./gradlew :extensions:training-checkpoint:test :extensions:training-checkpoint:javadoc`.
Coordinator: `SYNAPTIK_METAL_TEST_LIBRARY=<source-matched-library> ./gradlew test javadoc --max-workers=2 --console=plain`
once after integration: architecture, durable integration and actual Metal/device evidence; use a
verified source-matched library, reuse worker evidence and report device skips honestly.
Validate local documentation links/anchors, brief <=200 lines/15KB, and `git diff --check`.
Planning-only validation: no Java tests; existing Javadocs and glossary terms remain unchanged.

## Result
