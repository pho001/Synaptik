# Task Data0002: Valid sequence lengths and right padded batch metadata

## Status

Complete — accepted 2026-10-20 after independent approval and coordinator checkpoint.
User approved fluent API/implementation; main verified Ready, sole frontier, dependencies, base,
enacted authority and inactive conflicts before launch. Data0001 remains accepted and uncommitted.
All Data/NN/CPU/Text/Vision frontiers: None; Data0003/0004, NN0022A and Text/Vision remain held Draft.

## Change class

Class C: public Data API, durable borrowing boundary and bounded root/scoped permission update;
no new storage owner/lifetime or Model contract. Fresh implementation and independent review required.

## Goal

Deliver three top-level types plus nested Builder, tests, Javadoc and API explanation: an existing
Tensor paired with immutable host prefixes per original batch row, not numeric sentinels.

### Approved API

All three public types are directly in `io.github.pho001.synaptik.data`:

- Final `ValidSequenceLengths`: private constructor; `static ValidSequenceLengths of(long... lengths)`,
  `int batchSize()`, `long length(int row)`, `long[] toLongArray()`, ordered `equals`/`hashCode`
  and metadata-only `toString`; no stored T (upper bounds come from Shape at batch construction).
- Enum `SequenceAxisOrder`: exactly `TIME_MAJOR` and `BATCH_MAJOR`, no custom methods. Its name
  distinguishes logical sequence axes from Model's physical `LayoutDescriptor`.
- Record `RightPaddedSequenceBatch(Tensor values, ValidSequenceLengths validLengths,
  SequenceAxisOrder axisOrder)`: validated public canonical constructor and the three component
  accessors, plus only the builder factory below. It retains exact component references. Record equality
  uses Tensor object identity, structural lengths and enum identity, never Tensor contents.

### Fluent construction

- `static Builder builder(Tensor values)` borrows a non-null values reference fixed for the builder;
  null throws `NullPointerException` with message `values`. Creation reads no Shape.
- One `public static final` nested `Builder`, private constructor; exactly `timeMajor()`,
  `batchMajor()`, `validLengths(long... lengths)` and `build()`. No overloads, generic axis setter,
  facade, staged interface or new pipeline/lifetime. Direct constructor/standalone lengths remain public.
- Both axis order and lengths require explicit selection; no defaults. Empty varargs selects
  empty lengths for B=0, distinct from missing lengths. Setters return the same Builder;
  last successful setter wins, including changing axes or replacing lengths.
- `validLengths` calls `ValidSequenceLengths.of` immediately before assignment, copying/checking
  negativity once. Failed setters preserve prior state. Caller coordinates source-array mutation.
- `build()` requires both selections (`IllegalStateException` before geometry), then delegates to
  the record constructor's single metadata algorithm, without recopying immutable lengths.
  Each successful build returns a fresh batch; later setters/failures leave previous batches unchanged.
  Builder is reusable, caller-coordinated and not thread-safe; no resources are closed.

### Metadata validation and borrowing

- `of` checks non-null `lengths` (`NullPointerException`, message `lengths`), clones the array once
  and validates that owned copy in row order: `length >= 0`. Negative lengths fail with
  `IllegalArgumentException` identifying the row. Callers coordinate array mutation during construction;
  subsequent mutation cannot affect metadata. Every `toLongArray()` returns a fresh array, including empty.
- `length(row)` accepts only `0 <= row < batchSize()`; negative indices do not wrap and invalid
  indices throw `IndexOutOfBoundsException`.
- The batch constructor checks null `values`, `validLengths`, `axisOrder` in that order, with
  `NullPointerException` and the parameter name. Then validate rank, numeric axes, batch count,
  and length upper bounds in row order. Geometry failures throw `IllegalArgumentException`
  identifying the mismatch or offending row and T.
- Rank is at least two. `TIME_MAJOR` means `[T, B, ...]`, `BATCH_MAJOR` means `[B, T, ...]`.
  Both leading axes must be `StaticDimension`; B equals widened `validLengths.batchSize()`;
  every length <= T from Shape. Never narrow to `int`.
  B above `Integer.MAX_VALUE` fails; T/lengths may reach `Long.MAX_VALUE`. No element products/byte spans.
- Trailing dimensions are opaque Model dimensions, including zero or dynamic dimensions; scalar
  steps need no trailing axis. Only time/batch must be known because only those establish prefix
  geometry. No dtype restriction (including BOOL), gradient-flag restriction, provenance/leaf
  restriction, resolved-layout requirement or storage requirement is justified by this wrapper.
- Rows retain order; axis order neither proves contiguity nor transposes values or changes physical layout.
- The batch reads only immutable descriptor/shape metadata. It never reads or captures
  `hostStorage()`, checks storage liveness, obtains a segment, leases a scope, copies Tensor values,
  attaches/clears/replaces storage, or closes anything. The wrapper confers no readiness,
  liveness, thread-safety or materialization guarantee; callers retain existing Model obligations.
- Immutability covers copied lengths and fixed component references, not Tensor contents or its
  mutable borrowed-storage association. Length creation and batch validation are O(B), independent of time extent
  and element count; only host lengths are scanned. No values scan, rewrite, hidden `contiguous()`,
  Tensor/expression creation or execution occurs. Padding lengths `T - length(row)` are derived;
  neither those lengths nor a separate validity mask are stored.

Example: `RightPaddedSequenceBatch.builder(values).batchMajor().validLengths(3, 2, 1).build()`.
See the [implemented API example](../../../../api/data-api.md#complete-fluent-example) for values and results.

## Non-goals

Data0003 owns padding/truncation algorithms, collection batchers and Tensor allocation. NN0022A
owns separately gated host-to-runtime INT64 materialization/binding and any NN→Data edge.
No extra type/framework, static specialization presented as runtime reuse, validity-mask storage,
DataLoader, dataset/file/token/image policy, Training, Engine, backend, device or persistence work.

## Contracts

Read the [root scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts), then:

- Root [Java namespace](../../../../../ARCHITECTURE.md#java-namespace) and [Core invariants](../../../../../ARCHITECTURE.md#core-invariants).
- Root [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules) and [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- Root [Repository layout](../../../../../ARCHITECTURE.md#repository-layout), [Module-ownership routing](../../../../../ARCHITECTURE.md#module-ownership-routing)
  and Data row in [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts).
- Owning scoped heading [`extensions/data`](../../../../architecture/contracts/extensions-training.md#extensionsdata).
- Model scoped heading [`modules/model`](../../../../architecture/contracts/foundational-modules.md#modulesmodel)
  for immutable descriptors/provenance and existing mutable borrowed storage.

Root/owning Data permission and module-boundaries explanation preceded code; no dependency/Model
lifetime/ADR0042 change. Stop on ambiguity; plans are not authority.

## Dependencies and integration

- Depends on: Data0001 (Complete, accepted uncommitted foundation); current public Model APIs.
- Conflicts with: other Data API/namespace, Tensor-ownership or shared-authority writers;
  Data0003/0004 and NN0022A until this capability is accepted. No parallel writes.
- Parallel group: None.
- Common base revision: `ef35f2ec001b0c161d1abe825ff3be725b9be49e` (HEAD=main=origin/main,
  verified 2026-10-20) plus accepted uncommitted Data0001 foundation (17 files), not a merged
  predecessor, plus the approved Data0002 contract/planning activation.
- Integration order: authority/explanation → fresh implementation → independent review/docs →
  coordinator checkpoint → acceptance; no successor launch.
- Integration validation: Data0002 metadata checkpoint below; reuse Data0001 evidence, no root rerun.
- Shared-document integration owner: main/coordinator; serial writers for authority, docs and plans.

## Files and symbols

- Main types: `extensions/data/src/main/java/io/github/pho001/synaptik/data/{ValidSequenceLengths,SequenceAxisOrder,RightPaddedSequenceBatch}.java`.
- Matching tests: `extensions/data/src/test/java/io/github/pho001/synaptik/data/{ValidSequenceLengthsTest,SequenceAxisOrderTest,RightPaddedSequenceBatchTest}.java`.
- Read Model `tensor/{Tensor,TensorDescriptor,TensorFactory}`, `shape/{Shape,StaticDimension}`,
  `layout/LayoutDescriptor`, `storage/HostTensorStorage` under `modules/model/src/main/java/io/github/pho001/synaptik/model/`;
  tests: `TensorTest`, `TensorDescriptorTest`, `ShapeTest`, `LayoutDescriptorTest`, `HostTensorStorageTest` in matching packages. No Model edits.
- `testing/architecture-tests/src/dataApiConsumer/java/io/github/pho001/synaptik/testing/architecture/DataApiConsumer.java`:
  direct/fluent/Builder/Model export on the existing Data-only classpath.
- `testing/architecture-tests/src/test/java/io/github/pho001/synaptik/testing/architecture/DataDependencyContractTest.java`:
  stale comments/name only; retain seven tests/fixtures.
- Independent guide/glossary/navigation pass; closure: this brief, Data/NN masters, roadmap only.
  No build/migration/Data0001/ADR0042/NN/Text/Vision API edits.

## Acceptance criteria

- Test the exact Builder API, both fluent orders and setter order, missing selections before invalid
  geometry, explicit empty vs missing lengths, null values/varargs, immediate defensive copy/negative
  checks, same-builder chaining, last-successful-wins, failure recovery and fresh repeat builds.
  Earlier batches and exact immutable lengths references survive later setters/build failures.
- Test both non-square axis orders, rank 0/1 and unknown time/batch failures, opaque suffixes,
  upper-bound/count mismatches, negative lengths, all nulls, invalid indices and defensive copies.
  Equality/hash/diagnostics survive caller array mutations. B=0 accepts empty lengths at any T;
  T=0 requires all lengths zero. Zero feature extents do not force lengths zero.
- Descriptor-only tests cover `T = Long.MAX_VALUE`, large lengths and `B > Integer.MAX_VALUE`
  rejection without giant arrays, geometry products, Tensor data allocation or narrowing. All dtypes and
  Model-valid gradient flags are accepted; layout and exact Tensor/component identity survive.
- Tests retain exact values/storage bytes and associations through success/failure; cover
  absent/read-only/dead-after-attachment storage before/after wrapping, replacement/clearing,
  strided/offset/broadcast/unresolved geometry, real zeros and arbitrary padded suffixes. No content validation,
  persistent storage snapshot, scope ownership, expression producer or execution is introduced.

## Validation

Worker (sandbox stopped before compilation; approved cache-access retry completed once, not duplicate validation):

```bash
JAVA_HOME=/Users/phujka/Library/Java/JavaVirtualMachines/openjdk-26.0.1/Contents/Home ./gradlew :extensions:data:test
```

Accepted final coordinator command after independent review (Javadoc and bounded metadata checkpoint):

```bash
JAVA_HOME=/Users/phujka/Library/Java/JavaVirtualMachines/openjdk-26.0.1/Contents/Home ./gradlew :extensions:data:javadoc :testing:architecture-tests:test --tests 'io.github.pho001.synaptik.testing.architecture.DataDependencyContractTest' :testing:architecture-tests:compileDataApiConsumerJava
git diff --check
```

Closure: links/anchors/limits/whitespace/diff only, no Java reruns. No root/native/conformance/performance
run: dependency/build/Model lifetime/backend contracts unchanged.

## Documentation and review impact

Independent API/test/Javadoc/example/borrowing/contract/link review; guide/glossary distinguish host
prefixes from runtime INT64 and logical axes from physical layout. No Model Javadoc change needed.

## Result

- Accepted 2026-10-20: three public metadata types and nested reusable Builder; descriptor-only borrowing,
  immediate length copying and one canonical geometry validator; no Tensor content access/copy/lifetime.
- Scope: 20 unique paths, grouped below (module-boundaries belongs to both authority and explanation).
- Authority: `ARCHITECTURE.md`, `docs/architecture/{contracts/extensions-training,module-boundaries}.md`.
- Code/tests: three main types and three matching tests under `extensions/data/src/{main,test}/java/`;
  `DataApiConsumer.java` and `DataDependencyContractTest.java` at the exact paths above (eight files).
- Documentation: `docs/api/{data-api,public-api}.md`, `docs/{index,glossary}.md`,
  `docs/developer-guide/repository-layout.md`, and module-boundaries above (six files).
- Planning: this brief, `../master-plan.md`, `../../nn/master-plan.md`, `../../../roadmap.md`.
- Worker command above PASS: 38 new tests / 3 suites (batch 28, lengths 9, enum 1), zero failures/errors/skips.
- Independent Pascal binding APPROVE, zero issues; Javadoc unchanged and accurate. Guide examples reuse
  passing `realZerosAndArbitrarySuffixesAreNeitherValidatedNorChanged` and `snapshotsLengthsImmediatelyAndChecksNegativesBeforeBuild`.
- Review links: 24 PASS; pre-code links: 281 PASS. Meaningful new-API Javadoc fully reviewed;
  host-metadata/runtime-INT64 and logical-axis/physical-layout glossary distinctions added; no package-info intended.
- Final coordinator command above PASS, exit 0 in 1s: 26 actionable tasks, 4 executed, 22 UP-TO-DATE.
- Fresh architecture XML: 7 tests / 1 suite, zero failures/errors/skips; final Data Javadoc generated without warnings.
- External Data-only consumer freshly compiled direct/fluent/public nested Builder access and Model export.
- Main confirmed all eight worker source hashes unchanged through review/checkpoint, ten protected prior-dirty
  hashes unchanged, and `git diff --check` PASS; no source behavior fixes, obsolete code or API removals.
- Limitations: host declarations only; no numeric batching, runtime binding, execution or future capability claim.
- Issues/follow-up: none for this capability; all frontiers None, successors held Draft. Status: Complete.
