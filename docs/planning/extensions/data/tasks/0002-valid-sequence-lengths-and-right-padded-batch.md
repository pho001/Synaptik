# Task Data0002: Valid sequence lengths and right padded [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) [metadata](../../../../glossary.md#term-metadata)

> **Editorial reading context — not a status update.** The application supplies a Tensor
> plus one host count per batch row. The historical wrapper retains that exact Tensor and
> independently owns copied counts; it checks descriptor geometry, not stored padding values
> or storage liveness. For time extent four, count two declares the first two positions valid
> even when either contains zero. Choosing time-major metadata does not transpose data.
> Today's [axis geometry](../../../../reference/data/sequences.md#axis-order-and-geometry) and
> [borrowing contract](../../../../reference/data/sequences.md#validation-failures-and-borrowing)
> explain the same distinctions; its later replacement type does not rename the historical API below.

## Status

Complete — accepted 2026-10-20 after independent approval and coordinator [checkpoint](../../../../glossary.md#term-checkpoint).
User approved fluent [API](../../../../glossary.md#term-application-programming-interface-api)/implementation; main verified [Ready](../../../../glossary.md#term-ready), sole frontier, dependencies, base,
enacted authority and inactive conflicts before launch. Data0001 remains accepted and uncommitted.
All Data/[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/[CPU](../../../../glossary.md#term-cpu)/Text/Vision frontiers: None; Data0003/0004, NN0022A and Text/Vision remain held Draft.

## Change class

Class C: public Data [API](../../../../glossary.md#term-application-programming-interface-api), durable [borrowing](../../../../glossary.md#term-borrowing) boundary and bounded root/scoped permission update;
no new storage [owner](../../../../glossary.md#term-owner)/[lifetime](../../../../glossary.md#term-lifetime) or [Model](../../../../glossary.md#term-model) contract. Fresh implementation and independent review required.

## Goal

Deliver three top-level types plus nested Builder, tests, [Javadoc](../../../../glossary.md#term-javadoc) and [API](../../../../glossary.md#term-application-programming-interface-api) explanation: an existing
[Tensor](../../../../glossary.md#tensor) paired with [immutable](../../../../glossary.md#term-immutable) [host](../../../../glossary.md#term-host) prefixes per original [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) row, not numeric sentinels.

### Approved [API](../../../../glossary.md#term-application-programming-interface-api)

All three public types are directly in `io.github.pho001.synaptik.data`:

- Final `ValidSequenceLengths`: private constructor; `static ValidSequenceLengths of(long... lengths)`,
  `int batchSize()`, `long length(int row)`, `long[] toLongArray()`, ordered `equals`/[`hashCode`](../../../../glossary.md#hash-codes-and-content-digests)
  and metadata-only <span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>`toString`; no stored T (upper bounds come from [Shape](../../../../glossary.md#shape) at [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) construction).
- Enum `SequenceAxisOrder`: exactly `TIME_MAJOR` and `BATCH_MAJOR`, no custom methods. Its name
  distinguishes logical sequence axes from Model's physical <span data-glossary-navigation="true"> ([axes glossary](../../../../glossary.md#term-axis); [Model glossary](../../../../glossary.md#term-model))</span>`LayoutDescriptor`.
- Record `RightPaddedSequenceBatch(Tensor values, ValidSequenceLengths validLengths,
  SequenceAxisOrder axisOrder)`: validated public canonical constructor and the three component
  accessors, plus only the builder factory below. It retains exact component references. Record equality
  uses Tensor object identity, structural lengths and enum identity, never Tensor contents.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [Tensor glossary](../../../../glossary.md#tensor))</span>

### Fluent construction

- `static Builder builder(Tensor values)` borrows a non-null values reference fixed for the builder;
  null throws `NullPointerException` with message `values`. Creation reads no [Shape](../../../../glossary.md#shape).
- One `public static final` nested `Builder`, private constructor; exactly `timeMajor()`,
  `batchMajor()`, `validLengths(long... lengths)` and `build()`. No overloads, generic axis setter,
  facade, staged interface or new pipeline/lifetime. Direct constructor/standalone lengths remain public.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [facade glossary](../../../../glossary.md#term-facade); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- Both axis order and lengths require explicit selection; no defaults. Empty varargs selects
  empty lengths for B=0, distinct from missing lengths. Setters return the same Builder;
  last successful setter wins, including changing axes or replacing lengths.
- `validLengths` calls `ValidSequenceLengths.of` immediately before assignment, copying/checking
  negativity once. Failed setters preserve prior state. Caller coordinates source-array mutation.
- `build()` requires both selections (`IllegalStateException` before geometry), then delegates to
  the record constructor's single metadata algorithm, without recopying immutable lengths.
  Each successful build returns a fresh batch; later setters/failures leave previous batches unchanged.
  Builder is reusable, caller-coordinated and not thread-safe; no resources are closed.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [immutable glossary](../../../../glossary.md#term-immutable); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [thread-safe glossary](../../../../glossary.md#term-thread-safe))</span>

### [Metadata](../../../../glossary.md#term-metadata) [validation](../../../../glossary.md#term-validation) and [borrowing](../../../../glossary.md#term-borrowing)

- `of` checks non-null `lengths` (`NullPointerException`, message `lengths`), clones the array once
  and validates that owned copy in row order: `length >= 0`. Negative lengths fail with
  `IllegalArgumentException` identifying the row. Callers coordinate array mutation during construction;
  subsequent mutation cannot affect metadata. Every `toLongArray()` returns a fresh array, including empty.
- `length(row)` accepts only `0 <= row < batchSize()`; negative indices do not wrap and invalid
  indices throw `IndexOutOfBoundsException`.
- The [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) constructor checks null `values`, `validLengths`, `axisOrder` in that order, with
  `NullPointerException` and the parameter name. Then validate rank, numeric axes, batch count,
  and length upper bounds in row order. Geometry failures throw <span data-glossary-navigation="true"> ([parameter glossary](../../../../glossary.md#term-parameter))</span>`IllegalArgumentException`
  identifying the mismatch or offending row and T.
- [Rank](../../../../glossary.md#term-rank) is at least two. `TIME_MAJOR` means `[T, B, ...]`, `BATCH_MAJOR` means `[B, T, ...]`.
  Both leading axes must be `StaticDimension`; B equals widened `validLengths.batchSize()`;
  every length <= T from Shape. Never narrow to `int`.
  B above `Integer.MAX_VALUE` fails; T/lengths may reach `Long.MAX_VALUE`. No [element](../../../../glossary.md#term-element) products/byte spans.
- Trailing dimensions are opaque Model dimensions, including zero or dynamic dimensions; scalar
  steps need no trailing axis. Only time/batch must be known because only those establish prefix
  geometry. No dtype restriction (including BOOL), gradient-flag restriction, provenance/leaf
  restriction, resolved-layout requirement or storage requirement is justified by this wrapper.<span data-glossary-navigation="true"> ([dimensions glossary](../../../../glossary.md#dimension); [opaque glossary](../../../../glossary.md#term-opaque); [Model glossary](../../../../glossary.md#term-model); [scalar glossary](../../../../glossary.md#term-scalar); [dtype glossary](../../../../glossary.md#data-type--datatype); [BOOL glossary](../../../../glossary.md#data-type--datatype); [gradient glossary](../../../../glossary.md#term-gradient); [provenance glossary](../../../../glossary.md#provenance); [leaf glossary](../../../../glossary.md#term-leaf))</span>
- Rows retain order; [axis](../../../../glossary.md#term-axis) order neither proves [contiguity](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) nor transposes values or changes physical [layout](../../../../glossary.md#layout).
- The batch reads only [immutable](../../../../glossary.md#term-immutable) [descriptor](../../../../glossary.md#term-descriptor)/[shape](../../../../glossary.md#shape) metadata. It never reads or [captures](../../../../glossary.md#term-capture-meanings)
  `hostStorage()`, checks storage liveness, obtains a segment, leases a scope, copies Tensor values,
  attaches/clears/replaces storage, or closes anything. The wrapper confers no readiness,
  liveness, thread-safety or materialization guarantee; callers retain existing Model obligations.<span data-glossary-navigation="true"> ([liveness glossary](../../../../glossary.md#term-liveness); [leases glossary](../../../../glossary.md#term-lease); [scope glossary](../../../../glossary.md#term-scope); [Tensor glossary](../../../../glossary.md#tensor); [materialization glossary](../../../../glossary.md#materialization))</span>
- Immutability covers copied lengths and fixed component references, not Tensor contents or its
  mutable borrowed-storage association. Length creation and batch validation are O(B), independent of time extent
  and element count; only host lengths are scanned. No values scan, rewrite, hidden <span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [O(B) glossary](../../../../glossary.md#memory-management-atomic-counters-and-complexity); [extent glossary](../../../../glossary.md#term-extent); [host glossary](../../../../glossary.md#term-host))</span>`contiguous()`,
  Tensor/expression creation or execution occurs. Padding lengths <span data-glossary-navigation="true"> ([Padding glossary](../../../../glossary.md#padding))</span>`T - length(row)` are derived;
  neither those lengths nor a separate validity mask are stored.<span data-glossary-navigation="true"> ([mask glossary](../../../../glossary.md#term-boolean-mask))</span>

Example: `RightPaddedSequenceBatch.builder(values).batchMajor().validLengths(3, 2, 1).build()`.
See the [implemented API example](../../../../api/data-api.md#complete-fluent-example)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span> for values and results.

## Non-goals

Data0003 owns [padding](../../../../glossary.md#padding)/truncation algorithms, collection batchers and [Tensor](../../../../glossary.md#tensor) allocation. NN0022A
owns separately gated [host](../../../../glossary.md#term-host)-to-[runtime](../../../../glossary.md#term-runtime) [INT64](../../../../glossary.md#data-type--datatype) [materialization](../../../../glossary.md#materialization)/[binding](../../../../glossary.md#term-binding) and any [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)→Data edge.
No extra type/framework, static [specialization](../../../../glossary.md#term-specialization) presented as runtime reuse, validity-[mask](../../../../glossary.md#term-boolean-mask) storage,
DataLoader, [dataset](../../../../glossary.md#term-dataset)/file/[token](../../../../glossary.md#term-token)/image policy, Training, [Engine](../../../../glossary.md#term-engine), [backend](../../../../glossary.md#backend), [device](../../../../glossary.md#term-device) or [persistence](../../../../glossary.md#term-persistence) work.

## Contracts

Read the [root scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>, then:

- Root [Java namespace](../../../../../ARCHITECTURE.md#java-namespace) and [Core invariants](../../../../../ARCHITECTURE.md#core-invariants).
- Root [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules) and [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- Root [Repository layout](../../../../../ARCHITECTURE.md#repository-layout), [Module-ownership routing](../../../../../ARCHITECTURE.md#module-ownership-routing)<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
  and Data row in [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([normative glossary](../../../../glossary.md#term-normative))</span>.
- Owning scoped heading [`extensions/data`](../../../../architecture/contracts/extensions-training.md#extensionsdata).
- [Model](../../../../glossary.md#term-model) scoped heading [`modules/model`](../../../../architecture/contracts/foundational-modules.md#modulesmodel)
  for immutable descriptors/provenance and existing mutable borrowed storage.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [descriptors glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [mutable glossary](../../../../glossary.md#term-mutable); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>

Root/owning Data permission and [module](../../../../glossary.md#term-module)-boundaries explanation preceded code; no [dependency](../../../../glossary.md#term-dependency)/Model
[lifetime](../../../../glossary.md#term-lifetime)/ADR0042 change. Stop on ambiguity; plans are not authority.

## Dependencies and integration

- Depends on: Data0001 (Complete, accepted uncommitted foundation); current public [Model](../../../../glossary.md#term-model) APIs.
- Conflicts with: other Data API/namespace, Tensor-ownership or shared-authority writers;
  Data0003/0004 and NN0022A until this capability is accepted. No parallel writes.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Tensor glossary](../../../../glossary.md#tensor); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Parallel group: None.
- Common base revision: `ef35f2ec001b0c161d1abe825ff3be725b9be49e` (HEAD=main=origin/main,
  verified 2026-10-20) plus accepted uncommitted Data0001 foundation (17 files), not a merged
  predecessor, plus the approved Data0002 contract/planning activation.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [activation glossary](../../../../glossary.md#term-activation))</span>
- Integration order: authority/explanation → fresh implementation → independent review/docs →
  coordinator checkpoint → acceptance; no successor launch.
- Integration [validation](../../../../glossary.md#term-validation): Data0002 [metadata](../../../../glossary.md#term-metadata) [checkpoint](../../../../glossary.md#term-checkpoint) below; reuse Data0001 evidence, no root rerun.
- Shared-document integration [owner](../../../../glossary.md#term-owner): main/coordinator; serial writers for authority, docs and plans.

## Files and symbols

- Main types: `extensions/data/src/main/java/io/github/pho001/synaptik/data/{ValidSequenceLengths,SequenceAxisOrder,RightPaddedSequenceBatch}.java`.
- Matching tests: `extensions/data/src/test/java/io/github/pho001/synaptik/data/{ValidSequenceLengthsTest,SequenceAxisOrderTest,RightPaddedSequenceBatchTest}.java`.
- Read [Model](../../../../glossary.md#term-model) `tensor/{Tensor,TensorDescriptor,TensorFactory}`, `shape/{Shape,StaticDimension}`,
  `layout/LayoutDescriptor`, `storage/HostTensorStorage` under `modules/model/src/main/java/io/github/pho001/synaptik/model/`;
  tests: `TensorTest`, `TensorDescriptorTest`, `ShapeTest`, `LayoutDescriptorTest`, `HostTensorStorageTest` in matching packages. No Model edits.
- `testing/architecture-tests/src/dataApiConsumer/java/io/github/pho001/synaptik/testing/architecture/DataApiConsumer.java`:
  direct/fluent/Builder/Model export on the existing Data-only classpath.<span data-glossary-navigation="true"> ([classpath glossary](../../../../glossary.md#term-classpath))</span>
- `testing/architecture-tests/src/test/java/io/github/pho001/synaptik/testing/architecture/DataDependencyContractTest.java`:
  stale comments/name only; retain seven tests/fixtures.
- Independent guide/glossary/navigation pass; closure: this brief, Data/NN masters, roadmap only.
  No build/migration/Data0001/ADR0042/NN/Text/Vision API edits.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Acceptance criteria

- Test the exact Builder API, both fluent orders and setter order, missing selections before invalid
  geometry, explicit empty vs missing lengths, null values/varargs, immediate defensive copy/negative
  checks, same-builder chaining, last-successful-wins, failure recovery and fresh repeat builds.
  Earlier batches and exact immutable lengths references survive later setters/build failures.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [defensive copy glossary](../../../../glossary.md#term-defensive-copy); [batches glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- Test both non-square axis orders, rank 0/1 and unknown time/batch failures, opaque suffixes,
  upper-bound/count mismatches, negative lengths, all nulls, invalid indices and defensive copies.
  Equality/hash/diagnostics survive caller array mutations. B=0 accepts empty lengths at any T;
  T=0 requires all lengths zero. Zero feature extents do not force lengths zero.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [rank glossary](../../../../glossary.md#term-rank); [opaque glossary](../../../../glossary.md#term-opaque); [hash glossary](../../../../glossary.md#term-hash); [feature glossary](../../../../glossary.md#term-feature))</span>
- [Descriptor](../../../../glossary.md#term-descriptor)-only tests cover `T = Long.MAX_VALUE`, large lengths and `B > Integer.MAX_VALUE`
  rejection without giant arrays, geometry products, Tensor data allocation or narrowing. All dtypes and
  Model-valid gradient flags are accepted; layout and exact Tensor/component identity survive.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [narrowing glossary](../../../../glossary.md#term-narrowing); [dtypes glossary](../../../../glossary.md#data-type--datatype); [Model glossary](../../../../glossary.md#term-model); [gradient glossary](../../../../glossary.md#term-gradient); [layout glossary](../../../../glossary.md#layout); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Tests retain exact values/storage bytes and associations through success/failure; cover
  absent/read-only/dead-after-attachment storage before/after wrapping, replacement/clearing,
  strided/offset/broadcast/unresolved geometry, real zeros and arbitrary padded suffixes. No content validation,
  persistent storage snapshot, scope ownership, expression producer or execution is introduced.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [snapshot glossary](../../../../glossary.md#term-snapshot); [scope glossary](../../../../glossary.md#term-scope); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [producer glossary](../../../../glossary.md#term-producer))</span>

## [Validation](../../../../glossary.md#term-validation)

Worker (sandbox stopped before [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names); approved [cache](../../../../glossary.md#term-cache)-access retry completed once, not duplicate validation):

```bash
JAVA_HOME=/Users/phujka/Library/Java/JavaVirtualMachines/openjdk-26.0.1/Contents/Home ./gradlew :extensions:data:test
```

Accepted final coordinator command after independent review ([Javadoc](../../../../glossary.md#term-javadoc) and bounded [metadata](../../../../glossary.md#term-metadata) [checkpoint](../../../../glossary.md#term-checkpoint)):

```bash
JAVA_HOME=/Users/phujka/Library/Java/JavaVirtualMachines/openjdk-26.0.1/Contents/Home ./gradlew :extensions:data:javadoc :testing:architecture-tests:test --tests 'io.github.pho001.synaptik.testing.architecture.DataDependencyContractTest' :testing:architecture-tests:compileDataApiConsumerJava
git diff --check
```

Closure: links/anchors/limits/whitespace/diff only, no Java reruns. No root/native/[conformance](../../../../glossary.md#term-conformance)/performance
[run](../../../../glossary.md#term-run): [dependency](../../../../glossary.md#term-dependency)/build/[Model](../../../../glossary.md#term-model) [lifetime](../../../../glossary.md#term-lifetime)/[backend](../../../../glossary.md#backend) contracts unchanged.

## Documentation and review impact

Independent [API](../../../../glossary.md#term-application-programming-interface-api)/test/[Javadoc](../../../../glossary.md#term-javadoc)/example/[borrowing](../../../../glossary.md#term-borrowing)/contract/link review; guide/glossary distinguish [host](../../../../glossary.md#term-host)
prefixes from [runtime](../../../../glossary.md#term-runtime) [INT64](../../../../glossary.md#data-type--datatype) and logical [axes](../../../../glossary.md#term-axis) from physical [layout](../../../../glossary.md#layout). No [Model](../../../../glossary.md#term-model) Javadoc change needed.

## Result

- Accepted 2026-10-20: three public metadata types and nested reusable Builder; descriptor-only borrowing,
  immediate length copying and one canonical geometry validator; no Tensor content access/copy/lifetime.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [descriptor glossary](../../../../glossary.md#term-descriptor); [borrowing glossary](../../../../glossary.md#term-borrowing); [canonical glossary](../../../../glossary.md#term-canonical-form); [Tensor glossary](../../../../glossary.md#tensor); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- [Scope](../../../../glossary.md#term-scope): 20 unique paths, grouped below ([module](../../../../glossary.md#term-module)-boundaries belongs to both authority and explanation).
- Authority: `ARCHITECTURE.md`, `docs/architecture/{contracts/extensions-training,module-boundaries}.md`.
- Code/tests: three main types and three matching tests under `extensions/data/src/{main,test}/java/`;
  `DataApiConsumer.java` and `DataDependencyContractTest.java` at the exact paths above (eight files).
- Documentation: `docs/api/{data-api,public-api}.md`, `docs/{index,glossary}.md`,
  `docs/developer-guide/repository-layout.md`, and module-boundaries above (six files).
- [Planning](../../../../glossary.md#term-planning): this brief, `../master-plan.md`, `../../nn/master-plan.md`, `../../../roadmap.md`.
- Worker command above PASS: 38 new tests / 3 suites ([batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) 28, lengths 9, enum 1), zero failures/errors/skips.
- Independent Pascal binding APPROVE, zero issues; Javadoc unchanged and accurate. Guide examples reuse
  passing `realZerosAndArbitrarySuffixesAreNeitherValidatedNorChanged` and `snapshotsLengthsImmediatelyAndChecksNegativesBeforeBuild`.
- Review links: 24 PASS; pre-code links: 281 PASS. Meaningful new-API Javadoc fully reviewed;
  host-metadata/runtime-INT64 and logical-axis/physical-layout glossary distinctions added; no package-info intended.<span data-glossary-navigation="true"> ([INT64 glossary](../../../../glossary.md#data-type--datatype); [axis glossary](../../../../glossary.md#term-axis); [layout glossary](../../../../glossary.md#layout))</span>
- Final coordinator command above PASS, exit 0 in 1s: 26 actionable tasks, 4 executed, 22 UP-TO-DATE.
- Fresh architecture XML: 7 tests / 1 suite, zero failures/errors/skips; final Data [Javadoc](../../../../glossary.md#term-javadoc) generated without warnings.
- External Data-only consumer freshly compiled direct/fluent/public nested Builder access and [Model](../../../../glossary.md#term-model) export.
- Main confirmed all eight worker source hashes unchanged through review/checkpoint, ten protected prior-dirty
  hashes unchanged, and <span data-glossary-navigation="true"> ([hashes glossary](../../../../glossary.md#term-hash))</span><span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>`git diff --check` PASS; no source behavior fixes, obsolete code or [API](../../../../glossary.md#term-application-programming-interface-api) removals.
- Limitations: [host](../../../../glossary.md#term-host) declarations only; no numeric batching, [runtime](../../../../glossary.md#term-runtime) [binding](../../../../glossary.md#term-binding), execution or future capability claim.
- Issues/follow-up: none for this capability; all frontiers None, successors held Draft. Status: Complete.
