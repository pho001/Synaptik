# Task 0003: Durable Generated-[Kernel](../../../../glossary.md#kernel) [Artifact](../../../../glossary.md#term-artifact) Store and Cold Loading

## Reader context

Read the heading and recorded decisions rather than inferring the design from this retained
filename: this checkpoint replaced a bounded in-memory-cache proposal with a durable
generated-class store. CPU finalization supplied the already-selected specialization and
matching family emitter after shared Prepare assigned storage positions. The explicit root
belonged to the caller or composition owner; the store held executable class bytes and
compatibility metadata, not application inputs or numerical outputs.

Loading defined a hidden class in the current JVM. An immutable prepared executable could keep
that loaded artifact alive, while weak interning alone did not guarantee its lifetime or
preserve JIT machine code. The trusted-root and atomic-publication rules below are historical
requirements; the focused current cache explanation distinguishes the later optional-persistence
policy.

Current reading alongside this historical record:

- [CPU artifact compatibility and lifetime](../../../../explanation/packages/cpu.md#iogithubpho001synaptikbackendcpuinternalcache)
  — distinguishes reusable code bytes and loaded artifacts from numerical values and JIT state.
- [CPU analysis and finalization](../../../../explanation/packages/cpu.md#iogithubpho001synaptikbackendcpuinternalprepare)
  — explains declaration before assignment and realization afterward.

## Status

Superseded

## Goal

Add one [CPU](../../../../glossary.md#term-cpu)-private filesystem [artifact](../../../../glossary.md#term-artifact) store that durably reuses deterministic generated class
bytes across models, independently constructed store instances, and later [Java Virtual Machine](../../../../glossary.md#term-java-virtual-machine-jvm)
([JVM](../../../../glossary.md#java-platform-and-build-tools)) processes. A caller or later composition [owner](../../../../glossary.md#term-owner) supplies the [cache](../../../../glossary.md#term-cache) root explicitly. CPU
finalization will ask the store for one already-selected complete [specialization](../../../../glossary.md#term-specialization) after shared
[Prepare](../../../../glossary.md#prepare) assigns slots; the store either validates and loads exact compatible bytes or emits,
publishes, validates, defines, and resolves them once.

The durable reuse boundary is class bytes plus exact compatibility [metadata](../../../../glossary.md#term-metadata), not a strongly held
in-process [hidden class](../../../../glossary.md#term-hidden-class):

```text
explicit trusted local root + complete specialization + matching family emitter
  -> exact compatibility metadata -> deterministic content-addressed artifact path
  -> validate stored envelope and class shape, or deterministically emit verified bytes
  -> atomically publish one immutable complete envelope
  -> define hidden class and resolve exact static MethodHandle during CPU finalization
  -> later PreparedExecution ownership strongly retains the loaded artifact
run
  -> invoke the already-resolved exact typed handle only
```

Compatible age alone never invalidates an entry. Modification time, access time, hit rate, and
time-to-live are neither compatibility facts nor correctness inputs.

## [Scope](../../../../glossary.md#term-scope)

- Replace the obsolete bounded strong in-JVM least-recently-used cache implementation and tests
  with a package-private <span data-glossary-navigation="true"> ([JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [cache glossary](../../../../glossary.md#term-cache))</span>`CpuGeneratedKernelArtifactStore` rooted at an explicit caller-supplied
  `Path`.
- Canonicalize the supplied root once to an absolute normalized store identity. Different store
  objects using the same canonical root and compatibility key address the same artifact. The
  store owns only its fixed namespace below that root; the caller owns root selection,
  permissions, isolation, backup, capacity, and lifecycle.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [compatibility key glossary](../../../../glossary.md#cache-and-memoization); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Address each entry from a SHA-256 digest over a domain separator and the exact canonical
  compatibility metadata. Use the fixed relative form<span data-glossary-navigation="true"> ([SHA-256 glossary](../../../../glossary.md#term-sha-256); [digest glossary](../../../../glossary.md#term-digest); [metadata glossary](../../../../glossary.md#term-metadata))</span>
  `generated-kernels/v1/sha256/<first-two-hex>/<remaining-62-hex>.cpuclass` beneath the supplied
  root. No model, Tensor, graph, node, value, slot, storage object, address, run, emitter object,
  handle, class-loader, or store-instance identity participates.<span data-glossary-navigation="true"> ([model glossary](../../../../glossary.md#term-model); [Tensor glossary](../../../../glossary.md#tensor); [graph glossary](../../../../glossary.md#graph); [node glossary](../../../../glossary.md#node); [run glossary](../../../../glossary.md#term-run))</span>
- Persist one self-contained, length-delimited binary envelope containing exact compatibility
  metadata, the deterministic generated class bytes, and their SHA-256 checksum. Use no Java
  object serialization, JSON, executable object graph, or sidecar whose partial presence could be
  mistaken for a complete entry.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [checksum glossary](../../../../glossary.md#term-checksum); [serialization glossary](../../../../glossary.md#term-serialization); [JSON glossary](../../../../glossary.md#term-json))</span>
- Compare the full canonical structural specialization encoding and every fixed generator/
  class-entry compatibility field before accepting stored bytes. A path digest, specialization
  fingerprint, Java hash code, or checksum alone never establishes compatibility.<span data-glossary-navigation="true"> ([hash code glossary](../../../../glossary.md#term-hash-code))</span><span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [fingerprint glossary](../../../../glossary.md#term-fingerprint))</span>
- Validate bounded lengths, complete parsing, absence of trailing bytes, exact metadata, class
  checksum, Java class-file verification, and exact generated-class shape before definition.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape))</span>
- Treat a missing, structurally incompatible, truncated, corrupt, or invalid envelope as a safe
  miss: never define or invoke its bytes. Regenerate verified bytes and atomically replace the
  unusable entry. Ordinary root, directory, permission, read, write, synchronization, or atomic-
  move failures fail CPU finalization rather than silently bypassing the explicit store.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>
- Publish through a uniquely named temporary regular file in the final entry directory. Write the
  complete envelope, force its contents, and atomically move it to the deterministic final path.
  Never fall back to a non-atomic move. Clean up only the current attempt's unpublished temporary
  file after failure. Readers accept only the final regular file and therefore observe either an
  earlier complete envelope or a later complete envelope, never a partial publication.<span data-glossary-navigation="true"> ([atomic move glossary](../../../../glossary.md#term-atomic-move); [publication glossary](../../../../glossary.md#term-publication))</span>
- Allow redundant generation by separate JVM processes. Before publication, re-read an artifact
  that appeared after the original miss and reuse it when it validates. Concurrent writers for
  the same key may still publish equivalent complete envelopes; atomic replacement and mandatory
  post-publication re-read/validation determine the bytes that are defined.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Coordinate equal requests process-locally with one single-flight attempt keyed by canonical
  root plus full compatibility metadata, including across distinct store instances. Remove the
  attempt after its shared success or exact unchecked failure; failures are not negative-cached.<span data-glossary-navigation="true"> ([Coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [single-flight glossary](../../../../glossary.md#single-flight-and-least-recently-used-caching))</span>
- Weakly intern loaded `CpuGeneratedKernel` values process-locally by the same full key. The
  interner holds only `WeakReference` values and removes their exact keys through a
  `ReferenceQueue` when stale. It has no strong completed table, capacity, LRU order, expiry,
  background thread, or unloading promise.<span data-glossary-navigation="true"> ([LRU glossary](../../../../glossary.md#single-flight-and-least-recently-used-caching))</span>
- Return one exact weakly interned loaded artifact when it is still live. A caller and, later, a
  CPU [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable) reachable from [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) strongly retain the returned
  artifact, including its hidden class, lookup, exact handle, method type, specialization, and
  bytes. Collection is permitted only after every strong owner disappears.<span data-glossary-navigation="true"> ([hidden class glossary](../../../../glossary.md#term-hidden-class); [lookup glossary](../../../../glossary.md#term-method-handle-lookup); [owner glossary](../../../../glossary.md#term-owner))</span>
- Narrowly split `CpuClassFileKernelGenerator` so it remains the sole bytecode-emission,
  verification, hidden-class-definition, and exact-entry-resolution owner: one operation emits
  deterministic verified bytes; a second accepts stored bytes, re-verifies their exact class-file
  compatibility/shape, defines the hidden class, and resolves the existing exact static<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [operation glossary](../../../../glossary.md#operation))</span>
  [`MethodHandle`](../../../../glossary.md#native-numerical-library-and-instruction-architecture). The current convenience generation path may delegate to those two operations.
- Expose a defensive [canonical](../../../../glossary.md#term-canonical-form) [artifact](../../../../glossary.md#term-artifact)-compatibility encoding from
  `CpuKernelSpecialization`; it must encode every existing structural equality component in fixed
  order and must not expose mutable retained state.<span data-glossary-navigation="true"> ([structural equality glossary](../../../../glossary.md#term-structural-equality); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- Preserve [fail-closed](../../../../glossary.md#term-fail-closed) [CPU](../../../../glossary.md#term-cpu) capability reporting and the existing synthetic generator probes.
- Finalize affected Javadocs, package documentation, CPU guide, glossary, and planning evidence
  in the required separate clean documentation-focused pass after implementation.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- a bounded strong completed cache, least-recently-used policy, hit-rate policy, time-to-live,
  compatible-age invalidation, access-time correctness, automatic disk eviction, disk quota,
  pruning, background cleanup, watcher, daemon, executor, or service;<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache))</span>
- attacker-controlled bytecode safety, signature authentication, sandboxing, bytecode
  instrumentation, permission hardening, remote/shared-host cache safety, or a claim that SHA-256
  makes untrusted executable bytes safe;<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [signature glossary](../../../../glossary.md#term-signature); [authentication glossary](../../../../glossary.md#term-authentication); [host glossary](../../../../glossary.md#term-host); [SHA-256 glossary](../../../../glossary.md#term-sha-256))</span>
- a public or generic cache/store API, global service locator, public filesystem format, Config
  setting, default cache location, environment-variable lookup, home-directory convention, or
  Engine/composition implementation;<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [service locator glossary](../../../../glossary.md#term-service-locator); [lookup glossary](../../../../glossary.md#term-lookup); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Model operation semantics, family lowering implementations, Tensor evaluation, capability
  advertisement, a production CPU analyzer/finalizer/preparer, route selection, candidate
  generation, slot assignment, <span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [lowering glossary](../../../../glossary.md#lowering); [Tensor glossary](../../../../glossary.md#tensor); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>[`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable) implementation, generated
  [`BoundInvocation`](../../../../glossary.md#bound-invocation--boundinvocation), or any [CPU](../../../../glossary.md#term-cpu) 0004 behavior;
- replacing the existing exact static [`MethodHandle`](../../../../glossary.md#native-numerical-library-and-instruction-architecture) with reflection,
  `Method.invoke`, `invokeWithArguments`, a generated bound invocation, or another invocation
  abstraction;<span data-glossary-navigation="true"> ([bound invocation glossary](../../../../glossary.md#bound-invocation--boundinvocation))</span>
- Runtime disk access, cache access, hashing, checksum validation, class-file parsing, class
  definition, method lookup, route selection, argument discovery, or storage processing;<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [checksum glossary](../../../../glossary.md#term-checksum); [validation glossary](../../../../glossary.md#term-validation); [argument glossary](../../../../glossary.md#term-argument))</span>
- workload-tuning cache, model-plan cache, tuning evidence, measurements, benchmarks, policy,
  statistics, or mutation of tuning artifacts;<span data-glossary-navigation="true"> ([artifacts glossary](../../../../glossary.md#term-artifact))</span>
- source persistence, native code, ahead-of-time compilation, named-class loading, class-loader
  registries, module-layer changes, or serialization of a loaded class, lookup, or method handle;<span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence); [native code glossary](../../../../glossary.md#native-numerical-library-and-instruction-architecture); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [module glossary](../../../../glossary.md#term-module); [serialization glossary](../../../../glossary.md#term-serialization); [method handle glossary](../../../../glossary.md#term-method-handle))</span>
- dependency, Gradle, Java toolchain, architecture, ADR, architecture-test,
  backend-conformance, integration-test, shared-module, other-backend, or provider changes;<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [toolchain glossary](../../../../glossary.md#term-toolchain); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- CPU tasks 0004–0016 specifications or behavior; and
- [Model](../../../../glossary.md#term-model) task 0026 or any [FLOAT16](../../../../glossary.md#term-float16) or mixed-[precision](../../../../glossary.md#term-precision) semantic/capability claim.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
  - Core invariants
  - `modules/runtime`
  - `modules/prepare`
  - [Concrete backend](../../../../glossary.md#concrete-backend) [modules](../../../../glossary.md#term-module)
  - [CPU](../../../../glossary.md#term-cpu) [backend](../../../../glossary.md#backend) routes
  - [Prepare](../../../../glossary.md#prepare) [lifecycle](../../../../glossary.md#lifecycle)
  - [Run](../../../../glossary.md#term-run) lifecycle
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>
- [ADR 0002: Backend-owned lowering](../../../../design/decisions/0002-backend-owned-lowering.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Backend-owned lowering glossary](../../../../glossary.md#backend-owned-lowering))</span>
- [ADR 0010: Staged backend preparation](../../../../design/decisions/0010-staged-backend-preparation.md)
- [CPU backend guide](../../../../backend-guide/cpu-backend.md)
- [CPU kernel strategy](../../../../design/notes/cpu-kernel-strategy.md)<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel))</span>
- [Task 0001 CPU capability, representation, binding, and parallel foundation](0001-cpu-capability-representation-binding-and-parallel-foundation.md)<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- [Task 0002 Portable Class-File API generator foundation](0002-portable-class-file-api-generator-foundation.md)<span data-glossary-navigation="true"> ([Class-File API glossary](../../../../glossary.md#term-class-file-api))</span>

## Architecture constraints

- Generated JVM bytecode and its compatible artifact store remain CPU-internal. No shared
  lowering module, separate generated-code backend, or Runtime cache owner is introduced.<span data-glossary-navigation="true"> ([JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [bytecode glossary](../../../../glossary.md#term-bytecode); [artifact glossary](../../../../glossary.md#term-artifact); [CPU glossary](../../../../glossary.md#term-cpu); [lowering glossary](../../../../glossary.md#lowering); [module glossary](../../../../glossary.md#term-module); [backend glossary](../../../../glossary.md#backend); [Runtime glossary](../../../../glossary.md#term-runtime); [cache glossary](../../../../glossary.md#term-cache); [owner glossary](../../../../glossary.md#term-owner))</span>
- CPU analysis later selects the route, lowering, and complete specialization before finalization.
  This task consumes those already-selected facts and changes none of them.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization))</span>
- Filesystem lookup, validation, emission, publication, class definition, and exact handle
  resolution occur only in CPU finalization after shared slot assignment. Runtime receives an
  immutable recipe retaining the result and never repeats those actions.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [validation glossary](../../../../glossary.md#term-validation); [publication glossary](../../../../glossary.md#term-publication); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- The exact [`MethodHandle`](../../../../glossary.md#native-numerical-library-and-instruction-architecture) entry remains the cold-resolved typed invocation mechanism. Generated
  hot code and Runtime hot calls perform no lookup, reflection, cache work, hashing, validation,
  selection, or argument processing.<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument))</span>
- Persistent class bytes are immutable prepared inputs, not run-owned buffer/workspace resources
  and not tuning evidence. Loading them must not allocate physical tensor storage or acquire a
  closeable prepared resource.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [buffer glossary](../../../../glossary.md#term-buffer); [tensor glossary](../../../../glossary.md#tensor); [prepared resource glossary](../../../../glossary.md#prepared-resource--preparedresource))</span>
- Strong lifetime follows active ownership: a returned artifact and a later prepared executable
  may retain it. Process-global completed retention is forbidden; weak interning creates no
  lifetime guarantee.<span data-glossary-navigation="true"> ([lifetime glossary](../../../../glossary.md#term-lifetime); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable); [weak interning glossary](../../../../glossary.md#weak-reference-and-weak-interning))</span>
- Capability remains fail-closed because durable synthetic class bytes do not establish truthful
  operation coverage.<span data-glossary-navigation="true"> ([fail-closed glossary](../../../../glossary.md#term-fail-closed); [operation glossary](../../../../glossary.md#operation))</span>
- If full specialization compatibility does not determine bytecode, or a new target/generator
  fact is required, add that fact to the canonical compatibility contract and schema before reuse.
  Do not add model or runtime identity as a substitute.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [canonical glossary](../../../../glossary.md#term-canonical-form); [schema glossary](../../../../glossary.md#term-schema); [model glossary](../../../../glossary.md#term-model); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- If implementation requires a public type, Config value, production finalizer, shared contract,
  new dependency, non-atomic publication fallback, or architecture change, stop and report the
  conflict.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [atomic publication glossary](../../../../glossary.md#term-atomic-publication))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu` — unchanged public [fail-closed](../../../../glossary.md#term-fail-closed) capability surface.
- `io.github.pho001.synaptik.backend.cpu.execution` — owns all package-private specialization,
  generation, durable artifact, weak interning, cold definition, and direct invocation machinery.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [artifact glossary](../../../../glossary.md#term-artifact); [weak interning glossary](../../../../glossary.md#weak-reference-and-weak-interning))</span>

Packages added or changed:

- No package is added. The existing `execution` package remains the narrow visibility boundary.

Type placement and exact package-private surface:

- `io.github.pho001.synaptik.backend.cpu.execution.CpuGeneratedKernelArtifactStore` — new
  package-private final concrete store. Its only constructor is
  `CpuGeneratedKernelArtifactStore(Path root)`. Its only non-private [operation](../../../../glossary.md#operation) is
  `CpuGeneratedKernel loadOrGenerate(CpuKernelSpecialization specialization,
  CpuFamilyKernelEmitter familyEmitter)`.
- `CpuGeneratedKernelArtifactStore` owns private nested immutable key/envelope records, bounded
  binary parsing, filesystem I/O, atomic publication, the process-local single-flight table, and
  the weak loaded-artifact interner/reference queue. No second top-level cache abstraction is
  added.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [atomic publication glossary](../../../../glossary.md#term-atomic-publication); [single-flight glossary](../../../../glossary.md#single-flight-and-least-recently-used-caching); [reference queue glossary](../../../../glossary.md#weak-reference-and-weak-interning); [cache glossary](../../../../glossary.md#term-cache))</span>
- `io.github.pho001.synaptik.backend.cpu.execution.CpuClassFileKernelGenerator` — remains
  package-private and the sole generator/definition owner. Add package-private<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
  `byte[] generateClassBytes(CpuKernelSpecialization, CpuFamilyKernelEmitter)` and
  `CpuGeneratedKernel defineClassBytes(CpuKernelSpecialization, byte[])`; retain
  `generate(...)` only as a delegating compatibility seam if current tests still need it.
- `io.github.pho001.synaptik.backend.cpu.execution.CpuKernelSpecialization` — add only
  `byte[] artifactCompatibilityBytes()`, returning a defensive copy of the exact canonical
  structural encoding already used for its specialization fingerprint.<span data-glossary-navigation="true"> ([defensive copy glossary](../../../../glossary.md#term-defensive-copy); [canonical glossary](../../../../glossary.md#term-canonical-form); [fingerprint glossary](../../../../glossary.md#term-fingerprint))</span>
- `io.github.pho001.synaptik.backend.cpu.execution.CpuGeneratorSchema` — add only the fixed
  artifact-format/domain constants and generated class/entry naming constants needed by both
  deterministic metadata construction and generator shape validation.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [metadata glossary](../../../../glossary.md#term-metadata); [shape glossary](../../../../glossary.md#shape); [validation glossary](../../../../glossary.md#term-validation))</span>
- Delete obsolete `CpuGeneratedKernelCache`; there is no renamed strong cache or compatibility
  adapter.

## Filesystem format and [ownership](../../../../glossary.md#term-ownership-meanings)

The envelope uses [big-endian](../../../../glossary.md#text-encoding-and-file-boundaries) fixed-width integers and exact byte sequences in this order:

1. eight [ASCII](../../../../glossary.md#term-ascii) [magic bytes](../../../../glossary.md#term-magic-bytes) `SYNCPUK1`;
2. unsigned 32-bit [artifact](../../../../glossary.md#term-artifact) format version `1`;
3. unsigned 32-bit [metadata](../../../../glossary.md#term-metadata) byte length;
4. unsigned 32-bit class byte length;
5. 32-byte [SHA-256](../../../../glossary.md#term-sha-256) [checksum](../../../../glossary.md#term-checksum) of the class bytes;
6. the exact metadata bytes;
7. the exact class bytes; and
8. [end of file](../../../../glossary.md#text-encoding-and-file-boundaries), with trailing bytes rejected.

The implementation sets explicit conservative maximum metadata, class, and envelope lengths
before allocation and rejects integer [overflow](../../../../glossary.md#term-overflow), negative Java representations of unsigned lengths,
[truncation](../../../../glossary.md#term-truncation-toward-zero), non-regular final entries, and post-size-check races safely.

[Canonical](../../../../glossary.md#term-canonical-form) metadata contains, in fixed order and length-delimited form:

- artifact format and key-domain version;
- generator [schema](../../../../glossary.md#term-schema)/behavior version;
- exact generated [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) class name derived from the complete [specialization](../../../../glossary.md#term-specialization) [fingerprint](../../../../glossary.md#term-fingerprint);
- exact entry name `invoke` and exact `MethodType.descriptorString()`;
- class-file major version from the specialization and required minor version zero;
- exact Java [feature](../../../../glossary.md#term-software-feature)/[toolchain](../../../../glossary.md#term-toolchain) compatibility value `26` for the current Class-File and incubating
  Vector API contract; and<span data-glossary-navigation="true"> ([Vector API glossary](../../../../glossary.md#term-vector-api))</span>
- the complete `artifactCompatibilityBytes()` value, not merely either SHA-256 fingerprint.

The path [digest](../../../../glossary.md#term-digest) is SHA-256 over ASCII
`synaptik.cpu.generated-kernel.artifact-key.v1`, one zero byte, and the exact metadata bytes. The
stored metadata must compare byte-for-byte with the request even when the path digest matches.
The class checksum detects accidental damage to the stored [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication); it is not [authentication](../../../../glossary.md#term-authentication).

The root is a trusted local executable-code [cache](../../../../glossary.md#term-cache) boundary. The caller must ensure that
untrusted users and processes cannot write the root or replace its ancestors. The implementation
does not execute path fragments from the envelope, follows no envelope-supplied path, and rejects
a final entry that is not a regular file, but these checks do not make attacker-controlled
[bytecode](../../../../glossary.md#term-bytecode) safe. An attacker able to modify the root can replace both bytes and checksum and is
outside this task's safety claim.

The store creates only its fixed namespace, deterministic shard directory, final envelope, and
unique temporary files beneath the supplied root. It performs no valid-entry deletion, expiry,
eviction, directory sweep, quota enforcement, or background maintenance. Replacement is limited
to the exact final path after that entry has failed [validation](../../../../glossary.md#term-validation) for the current exact key.

## [Validation](../../../../glossary.md#term-validation) and failure order

For construction:

1. reject `null` root with `NullPointerException("root")`;
2. convert it to an absolute normalized path without consulting a default or environment value;
3. create/validate only the fixed namespace lazily on first use; and
4. fail with one contextual unchecked artifact-store exception whose cause retains the exact
   filesystem failure if directory or atomic-publication requirements are unavailable.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [publication glossary](../../../../glossary.md#term-publication))</span>

For `loadOrGenerate(...)`:

1. reject null [specialization](../../../../glossary.md#term-specialization), then null family emitter, using the existing [argument](../../../../glossary.md#term-argument) names;
2. obtain and compare the family emitter lowering fingerprint before weak-intern, in-flight, or
   disk lookup; retain the existing mismatch message;<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup))</span><span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [fingerprint glossary](../../../../glossary.md#term-fingerprint))</span>
3. build full [canonical](../../../../glossary.md#term-canonical-form) [metadata](../../../../glossary.md#term-metadata) and the full structural process key;
4. drain stale weak references, then return an exact live weakly interned [artifact](../../../../glossary.md#term-artifact) if present;
5. join or establish the one process-local attempt for the canonical root and full metadata;
6. as attempt owner, read and validate a final stored envelope in format order: path/type and
   bounded size, envelope framing, exact metadata, checksum, then generator-owned class-file
   verification and exact shape;<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [framing glossary](../../../../glossary.md#term-framing))</span>
7. on a valid hit, define the validated bytes and resolve the exact handle;
8. on a missing/corrupt/incompatible entry, generate deterministic verified bytes, construct the
   complete envelope, re-check a concurrently appeared valid final entry, otherwise force and
   atomically publish the temporary envelope;
9. re-read and fully validate the final path after publication, then define only those final
   stored bytes and resolve the exact handle;
10. weakly intern the loaded artifact, publish the exact shared success, and remove the in-flight
    attempt; or publish the exact unchecked failure and remove the attempt without retaining a
    completed failure.

[Checksum](../../../../glossary.md#term-checksum) mismatch never reaches class parsing. Metadata mismatch never reaches checksum or class
definition. Class verification/[shape](../../../../glossary.md#shape) failure never reaches hidden-class definition. Definition
failure never returns an artifact. No validation result depends on file age or access history.

Waiting is uninterruptible with respect to the shared attempt: interruption neither cancels the
attempt nor starts another one. A waiter observes the shared exact success or unchecked failure
and restores its interrupt status before returning or throwing. Different keys never generate or
perform filesystem I/O while holding the process coordination lock.

Exact class shape means: required class-file major/minor version; exact generated internal class
name and `java.lang.Object` superclass; final class; no interface or field; exactly one static
method named `invoke` with the expected [descriptor](../../../../glossary.md#term-descriptor) and code; and no second callable entry. Java
26 `ClassFile.verify` must report no error before definition. Definition still uses a hidden
[nestmate](../../../../glossary.md#term-nestmate) from the existing full-privilege package [lookup](../../../../glossary.md#term-method-handle-lookup) and resolves the same exact static
handle.

## Affected files

Expected [CPU](../../../../glossary.md#term-cpu) production paths:

- delete `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuGeneratedKernelCache.java`
- add `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuGeneratedKernelArtifactStore.java`
- update `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuClassFileKernelGenerator.java`
- update `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuKernelSpecialization.java`
- update `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuGeneratorSchema.java`
- update `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/package-info.java`

Expected CPU test paths:

- delete `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuGeneratedKernelCacheTest.java`
- add `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuGeneratedKernelArtifactStoreTest.java`
- update `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuClassFileKernelGeneratorTest.java`
- update `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuKernelSpecializationTest.java`

Expected explanatory documentation:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`

Expected [planning](../../../../glossary.md#term-planning) paths:

- this task
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

Review only, with a reasoned no-change conclusion unless a concrete contradiction is found:

- remaining CPU production/tests and `CpuGeneratedKernel` [Javadoc](../../../../glossary.md#term-javadoc);
- Config, Model, Planning, Prepare, Runtime, Backend Contract, Trace, Compiler, Engine,
  OpenBLAS provider, other backends, extensions, tuning, and benchmark contracts;<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Backend glossary](../../../../glossary.md#backend); [Trace glossary](../../../../glossary.md#trace); [Compiler glossary](../../../../glossary.md#term-compiler); [Engine glossary](../../../../glossary.md#term-engine); [OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>
- CPU/root Gradle, architecture/focused architecture/ADRs, design notes, architecture tests,
  backend conformance, integration tests, and completed CPU tasks 0001–0002.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

At most 15 paths: six [CPU](../../../../glossary.md#term-cpu) production paths, four CPU test paths, two explanatory documentation
paths, and three [planning](../../../../glossary.md#term-planning) paths listed above. Deletion and replacement of each obsolete [cache](../../../../glossary.md#term-cache) path
count separately in this ceiling.

No other top-level type, helper package, public [API](../../../../glossary.md#term-application-programming-interface-api), finalizer, Config value, [dependency](../../../../glossary.md#term-dependency)/build
file, shared [module](../../../../glossary.md#term-module), architecture/[ADR](../../../../glossary.md#term-architecture-decision-record-adr)/design-note, [conformance](../../../../glossary.md#term-conformance), integration, other [backend](../../../../glossary.md#backend), or
later task specification may change. If an additional path or broader boundary is required, stop
and update planning rather than hiding the expansion.

## Acceptance criteria

- [x] The obsolete bounded strong LRU cache and its tests are removed; no strong global completed
      map, capacity, eviction, expiry, hit-rate, or age-based correctness remains.<span data-glossary-navigation="true"> ([LRU glossary](../../../../glossary.md#single-flight-and-least-recently-used-caching); [cache glossary](../../../../glossary.md#term-cache))</span>
- [x] The only new production type is package-private final
      `CpuGeneratedKernelArtifactStore` with exactly the constructor and operation specified
      above; there is no public/generic abstraction or implicit root.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- [x] Two store instances using the same root and a later JVM process reuse the exact stored class
      bytes without family emission on a valid hit.<span data-glossary-navigation="true"> ([JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm))</span>
- [x] Canonical path/key metadata excludes every model/Tensor/graph/value/slot/storage/run/object
      identity and includes the complete canonical specialization plus every generator,
      class-file, Java/Vector, class-name, entry-name, and entry-descriptor compatibility fact.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-reference-identity))</span><span data-glossary-navigation="true"> ([Canonical glossary](../../../../glossary.md#term-canonical-form); [metadata glossary](../../../../glossary.md#term-metadata); [model glossary](../../../../glossary.md#term-model); [Tensor glossary](../../../../glossary.md#tensor); [graph glossary](../../../../glossary.md#graph); [run glossary](../../../../glossary.md#term-run); [specialization glossary](../../../../glossary.md#term-specialization); [Vector glossary](../../../../glossary.md#term-vector); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- [x] A matching digest/path never bypasses byte-for-byte structural metadata comparison. Tests
      prove unequal structural specializations cannot alias when a derived fingerprint or digest
      is forced to collide.<span data-glossary-navigation="true"> ([digest glossary](../../../../glossary.md#term-digest); [alias glossary](../../../../glossary.md#aliasing-and-arena); [fingerprint glossary](../../../../glossary.md#term-fingerprint))</span>
- [x] Envelope parsing is bounded and exact. Metadata, checksum, truncation, trailing-byte,
      class-file version/name/supertype/flags/interface/field/method/descriptor/code, and verifier
      corruption cases fail before definition in the specified order.<span data-glossary-navigation="true"> ([checksum glossary](../../../../glossary.md#term-checksum); [truncation glossary](../../../../glossary.md#term-truncation-toward-zero))</span>
- [x] Corrupt or incompatible entries are never defined. They regenerate and are replaced only by
      a forced complete temporary envelope atomically moved to the exact final path, followed by
      mandatory final re-read/validation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- [x] The implementation never falls back to non-atomic publication. Concurrent reader/writer and
      multi-process writer tests prove no partial final artifact is observed; redundant writers
      converge on a fully valid deterministic final envelope.<span data-glossary-navigation="true"> ([atomic publication glossary](../../../../glossary.md#term-atomic-publication); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- [x] The security documentation clearly states that checksum and structural verification detect
      accidental corruption but do not authenticate attacker-controlled bytecode; the root must
      be trusted and locally write-isolated by its caller.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode))</span>
- [x] Equal process-local requests, including requests through distinct store instances, share one
      attempt and exact outcome. Different keys progress independently. Failure is fanned out
      exactly, removed, and retriable.
- [x] Loaded interning is weak only. Reference-queue cleanup removes stale keys without a
      background thread, and no test depends on garbage-collection or hidden-class-unloading
      timing for correctness.
- [x] A caller-held artifact remains directly invocable regardless of weak-interner cleanup.
      Later prepared ownership, not the store, is documented as the strong lifetime boundary.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- [x] `CpuClassFileKernelGenerator` remains the sole emission, class verification, hidden
      definition, and exact handle-resolution owner. Stored-byte loading cannot define through a
      second mechanism.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- [x] Existing exact static [`MethodHandle`](../../../../glossary.md#native-numerical-library-and-instruction-architecture) type and `invokeExact` behavior remain unchanged; no
      reflection or generated [`BoundInvocation`](../../../../glossary.md#bound-invocation--boundinvocation) is introduced.
- [x] Source/bytecode tests prove Runtime hot calls contain no disk, cache, hash, checksum,
      parsing, lookup, validation, route selection, or argument processing.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [hash glossary](../../../../glossary.md#term-hash); [lookup glossary](../../../../glossary.md#term-lookup); [argument glossary](../../../../glossary.md#term-argument))</span>
- [x] No valid entry is invalidated for age, and tests may change timestamps without changing hit
      behavior.
- [x] `CpuCapabilityProvider` remains unchanged and fail-closed; no Model operation, production
      CPU finalizer, task-0004 behavior, Config/tuning/benchmark policy, or later specification is
      added.<span data-glossary-navigation="true"> ([fail-closed glossary](../../../../glossary.md#term-fail-closed); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- [x] Focused store/generator tests and one final CPU module suite pass after executable Java
      stabilizes.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>
- [x] Changed production declarations have meaningful complete Javadoc for format, validation,
      failures, concurrency, security, ownership, lifetime, inputs, results, and nullability.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [concurrency glossary](../../../../glossary.md#term-concurrency); [nullability glossary](../../../../glossary.md#term-nullability))</span>
- [x] A separate clean documentation-focused pass finalizes package Javadoc, CPU guidance,
      glossary impact, planning evidence, links, and status without repeating successful Java
      tests unless executable behavior changes or a concrete stale-evidence risk is recorded.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- [x] CPU Javadoc, generated pages, documentation checks, exact 15-path scope, package/surface/
      mechanism/status/later-specification checks, and <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` pass.

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation development may [run](../../../../glossary.md#term-run):

```bash
./gradlew :backends:cpu:test --tests '*CpuGeneratedKernelArtifactStoreTest'
./gradlew :backends:cpu:test --tests '*CpuClassFileKernelGeneratorTest' --tests '*CpuKernelSpecializationTest'
```

The focused [matrix](../../../../glossary.md#term-matrix-meanings) must cover:

- constructor/request nulls, emitter mismatch, validation order, and stable messages;
- exact path and envelope golden bytes, deterministic metadata/key construction, defensive
  specialization encoding, bounded lengths, integer-overflow rejection, and no trailing bytes;<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [specialization glossary](../../../../glossary.md#term-specialization); [overflow glossary](../../../../glossary.md#term-overflow))</span>
- same-instance, different-instance/same-root, and child-[JVM](../../../../glossary.md#term-java-virtual-machine-jvm) valid-hit reuse without emission;
- full structural differences, forced fingerprint/path-digest collision handling, and exclusion of
  prohibited identities;<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([fingerprint glossary](../../../../glossary.md#term-fingerprint); [digest glossary](../../../../glossary.md#term-digest))</span>
- missing artifact generation; metadata/checksum/truncation/trailing-byte corruption; wrong class
  version/name/superclass/flags/interface/field/method/descriptor/code; verifier failure; and proof
  that rejected bytes are never defined;<span data-glossary-navigation="true"> ([checksum glossary](../../../../glossary.md#term-checksum); [truncation glossary](../../../../glossary.md#term-truncation-toward-zero); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- timestamp changes without invalidation;
- failed root/directory/read/write/force/[atomic](../../../../glossary.md#resource-transaction-and-rollback)-move behavior without a non-atomic fallback;
- same-process equal-key single-flight across store objects, exact success/failure fan-out,
  interruption restoration, retry, and unrelated-key progress;<span data-glossary-navigation="true"> ([single-flight glossary](../../../../glossary.md#single-flight-and-least-recently-used-caching); [fan-out glossary](../../../../glossary.md#term-fan-out))</span>
- controlled concurrent readers and writers plus coordinated child-JVM writers sharing one root,
  permitting redundant emission while proving the final path is always a complete valid envelope;
- valid concurrently appeared [artifact](../../../../glossary.md#term-artifact) preference and mandatory post-[publication](../../../../glossary.md#term-publication) re-read;
- weak loaded reuse while live, reference-queue stale-key removal through deterministic test
  control rather than a garbage-collection timing assertion, and continued invocation through a
  caller-held strong artifact;
- generator emit/verify versus stored-byte verify/define split, identical bytes, exact hidden
  class/lookup/handle/type, and unchanged <span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-method-handle-lookup))</span><span data-glossary-navigation="true"> ([hidden class glossary](../../../../glossary.md#term-hidden-class))</span>`invokeExact` probes; and
- package-private/final surface, absence of strong completed/LRU/expiry/background/reflection/
  tuning/Runtime-hot-path mechanisms, and unchanged fail-closed public CPU surface.<span data-glossary-navigation="true"> ([LRU glossary](../../../../glossary.md#single-flight-and-least-recently-used-caching); [Runtime glossary](../../../../glossary.md#term-runtime); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>

After executable Java stabilizes, run exactly once:

```bash
./gradlew :backends:cpu:test
```

Implementation pass also runs:

```bash
git diff --check
```

Documentation-focused pass, after Javadocs and explanatory text:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

Validate repository-local Markdown links and heading anchors for this task, [CPU](../../../../glossary.md#term-cpu) master plan,
roadmap, CPU guide, and glossary; balanced fences, final newlines, and trailing whitespace; exact
15-path implementation [scope](../../../../glossary.md#term-scope); CPU 0001–0002 `Complete`; CPU 0003 synchronized to its
implementation status; CPU 0004–0016 `Draft` without detailed specifications; preserved [Model](../../../../glossary.md#term-model)
0026 [dependency](../../../../glossary.md#term-dependency); and unchanged build/architecture/[ADR](../../../../glossary.md#term-architecture-decision-record-adr)/design-note/[conformance](../../../../glossary.md#term-conformance)/integration paths.

Repository-wide Java, architecture, [backend](../../../../glossary.md#backend)-conformance, and integration validation remain
deferred to CPU 0009 or [continuous integration](../../../../glossary.md#term-continuous-integration-ci) because this task changes one backend-private
mechanism without a dependency or architecture boundary change. Stop if implementation evidence
requires a repository-wide contract change.

## Dependencies

- Complete CPU 0001 representation, cold binding, exact typed invocation, and worker foundation,
  preserved unchanged.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- Complete CPU 0002 generator schema, immutable structural specialization, family-emitter
  fingerprint contract, deterministic bytes, hidden artifact, and exact static handle.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [immutable glossary](../../../../glossary.md#term-immutable); [specialization glossary](../../../../glossary.md#term-specialization); [fingerprint glossary](../../../../glossary.md#term-fingerprint); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Current staged Prepare contract: generated artifact lookup/loading occurs in CPU finalization
  only after shared slot assignment.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Current Java 26 [CPU](../../../../glossary.md#term-cpu)-local Class-File and [Vector API](../../../../glossary.md#term-vector-api) build, unchanged.
- An explicit trusted local cache root supplied by a future caller/composition owner. Task 0003
  defines the private constructor seam but no production owner or default.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [owner glossary](../../../../glossary.md#term-owner))</span>
- Model task 0026 remains future Draft work and is required only before FLOAT16 semantics or
  backend capability. It does not block this identity-free artifact mechanism for current types.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [FLOAT16 glossary](../../../../glossary.md#term-float16); [backend capability glossary](../../../../glossary.md#backend-capability); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>

## Follow-up tasks

- CPU 0004 remains Draft and will own typed portable analysis, final specialization selection, the
  production CPU finalizer, explicit artifact-store construction/root supply by composition, and
  strong retention of the returned artifact in the prepared executable.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [artifact glossary](../../../../glossary.md#term-artifact); [prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable))</span>
- CPU 0005–0009 remain Draft for portable semantic coverage and the capability/conformance
  checkpoint.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- [CPU](../../../../glossary.md#term-cpu) 0010–0015 remain Draft for optional native routes.
- CPU 0016 and shared tuning work remain Draft and own compatible workload-tuning evidence. They
  neither replace nor select the generated-class artifact store.
- Automatic disk eviction, cache administration, remote sharing, hostile-root authentication,
  and prepared-executable serialization have no task and are not implied follow-up requirements.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [authentication glossary](../../../../glossary.md#term-authentication); [serialization glossary](../../../../glossary.md#term-serialization))</span>

## Architecture impact

Expected impact: None.

This task implements the existing [CPU](../../../../glossary.md#term-cpu)-owned compatible generated-[artifact](../../../../glossary.md#term-artifact) reuse permission at the
existing post-slot-assignment finalization stage. [Durability](../../../../glossary.md#term-durability), [validation](../../../../glossary.md#term-validation), and explicit root
[ownership](../../../../glossary.md#term-ownership-meanings) stay inside the CPU [module](../../../../glossary.md#term-module) and add no public/shared contract, [dependency](../../../../glossary.md#term-dependency), [lifecycle](../../../../glossary.md#lifecycle)
stage, or [Runtime](../../../../glossary.md#term-runtime) behavior. Stop if implementation reveals otherwise.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/planning/backends/cpu/master-plan.md, and
docs/planning/backends/cpu/tasks/0003-bounded-generated-artifact-cache-and-cold-finalization.md.
Read completed CPU tasks 0001–0002 and every directly referenced specialization, generator,
artifact, package, test, CPU guide/glossary, staged Prepare/Runtime, security, filesystem, and
Java 26 build contract needed by the task.

Implement CPU task 0003 exactly as specified. Replace the obsolete current uncommitted strong
LRU implementation; do not treat it as accepted work or completion evidence. Stop and report any
architecture, compatibility-key, atomic-filesystem, package-surface, or maximum-scope conflict
instead of weakening validation or expanding the task.

After implementation and the one final CPU test run, hand the diff and exact evidence to a
separate clean documentation-focused context. It must follow the documentation rules, finalize
affected Javadocs, CPU guidance, glossary impact, planning evidence, and documentation validation
in the same overall change, and reuse Java evidence unless executable behavior changes or a
concrete stale-evidence risk is recorded.

Update this task with implementation-local decisions, known limitations, validation evidence,
implementation notes, the canonical completion summary, and synchronized final status. Do not
mark Complete before every acceptance criterion and the documentation pass succeed.
```

## Local decisions

- The durable source of reuse is one self-contained `.cpuclass` envelope under an explicit root,
  not a loaded hidden class and not tuning evidence.<span data-glossary-navigation="true"> ([hidden class glossary](../../../../glossary.md#term-hidden-class))</span>
- Full canonical metadata is both stored and compared. SHA-256 selects a bounded deterministic
  path and checks class integrity but never substitutes for structural compatibility or trust.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [metadata glossary](../../../../glossary.md#term-metadata); [SHA-256 glossary](../../../../glossary.md#term-sha-256); [integrity glossary](../../../../glossary.md#term-integrity))</span>
- A single final file plus force-and-atomic-move publication gives readers one visibility point.
  Multi-file metadata/class/marker layouts were rejected because they create partial-entry states.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [publication glossary](../../../../glossary.md#term-publication); [layouts glossary](../../../../glossary.md#layout))</span>
- Atomic move support at the supplied root is a requirement. Silent non-atomic fallback was
  rejected because correctness under concurrent processes would become filesystem-dependent.<span data-glossary-navigation="true"> ([Atomic move glossary](../../../../glossary.md#term-atomic-move))</span>
- Corrupt/incompatible exact-path entries are safe misses and may be atomically replaced only
  after verified regeneration. Compatible entries are immutable and never rewritten for age,
  access, recency, or statistics.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- Process coordination uses canonical root plus full metadata equality, never only the path
  digest. This prevents digest collision from aliasing unequal structures.<span data-glossary-navigation="true"> ([digest glossary](../../../../glossary.md#term-digest); [aliasing glossary](../../../../glossary.md#term-aliasing))</span>
- Loaded values are weakly interned only to avoid duplicate definitions while live. The reference
  queue removes stale key/value pairs during ordinary finalization calls; there is no maintenance
  thread and no strong completed retention.<span data-glossary-navigation="true"> ([reference queue glossary](../../../../glossary.md#weak-reference-and-weak-interning))</span>
- Process-local single-flight is shared across store instances. Cross-process locking is omitted:
  redundant deterministic generation is allowed, while atomic complete-file publication and
  final revalidation provide safety.<span data-glossary-navigation="true"> ([single-flight glossary](../../../../glossary.md#single-flight-and-least-recently-used-caching))</span>
- Stored bytes are always verified again before definition, even when generated by the current
  process. The generator remains the sole class-file verification/definition owner.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- The trusted-local-root boundary is explicit. Cryptographic hashes detect accidental mismatch;
  they do not authenticate executable content from an attacker with write access.<span data-glossary-navigation="true"> ([hashes glossary](../../../../glossary.md#term-cryptographic-hash))</span>
- The existing filename is retained for this planning pass so the user-authorized scope remains
  exactly the current CPU 0003 task path, CPU master plan, and roadmap. Its heading and linked task
  name are authoritative for the revised design.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [scope glossary](../../../../glossary.md#term-scope); [CPU glossary](../../../../glossary.md#term-cpu))</span>

## Known limitations

- No production CPU finalizer constructs or calls the store yet; CPU 0004 owns integration and
  prepared strong retention.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- CPU still advertises and executes no Model operation. Stored artifacts remain synthetic until
  later family tasks add truthful coverage.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [operation glossary](../../../../glossary.md#operation); [artifacts glossary](../../../../glossary.md#term-artifact))</span>
- The caller must supply a trusted local filesystem root that supports required regular-file,
  force, and atomic-move semantics. No default, discovery, portability fallback, capacity
  management, or administrative API is provided.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- A malicious writer with access to the root is outside the security model and can replace
  executable content together with its checksum.<span data-glossary-navigation="true"> ([checksum glossary](../../../../glossary.md#term-checksum))</span>
- Weak interning does not promise artifact identity or JIT-profile retention after all strong
  owners disappear, and no unloading or collection time is promised.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([Weak interning glossary](../../../../glossary.md#weak-reference-and-weak-interning); [identity glossary](../../../../glossary.md#term-identity-meanings); [JIT glossary](../../../../glossary.md#term-just-in-time-compilation-jit))</span>
- Distinct JVM processes may redundantly emit the same artifact. This is accepted work, not a
  correctness or persistence failure.<span data-glossary-navigation="true"> ([JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- Hash collisions fail closed through exact metadata comparison. The task does not attempt a
  second alternate address for a cryptographic path collision.<span data-glossary-navigation="true"> ([Hash glossary](../../../../glossary.md#term-hash); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- No automatic deletion occurs, so durable valid entries accumulate until the caller or external
  administration manages the explicitly supplied root outside active use.
- Model task 0026 remains future work. This store authorizes no FLOAT16 or mixed-precision
  semantics, specialization selection, capability, or execution.<span data-glossary-navigation="true"> ([FLOAT16 glossary](../../../../glossary.md#term-float16); [precision glossary](../../../../glossary.md#term-precision); [specialization glossary](../../../../glossary.md#term-specialization))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning)-only evidence from clean planning context `/root`:

- the corrected repository-local checker passed all 369 local links and heading anchors across
  this task, the CPU master plan, and the roadmap; its first invocation had a checker-only Ruby
  quoting syntax error and validated nothing;
- balanced Markdown fences, final newlines, and trailing-whitespace checks passed for all three
  planning paths;
- required task sections, task/master/roadmap [`Ready`](../../../../glossary.md#term-ready) [synchronization](../../../../glossary.md#term-synchronization), [CPU](../../../../glossary.md#term-cpu) 0001–0002 `Complete`,
  CPU 0004–0016 `Draft`, absence of CPU 0004–0009 task files, and preservation of future Model
  0026 passed;<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- `git status --short docs/planning` reported exactly the three permitted planning paths; and
- `git diff --check` passed for the complete current worktree diff.

No Java, test, [Javadoc](../../../../glossary.md#term-javadoc), [Gradle](../../../../glossary.md#term-gradle), generator, [artifact](../../../../glossary.md#term-artifact), implementation, architecture, [ADR](../../../../glossary.md#term-architecture-decision-record-adr),
[conformance](../../../../glossary.md#term-conformance), integration, [commit](../../../../glossary.md#term-commit), or push command ran in this planning pass. The current
uncommitted [cache](../../../../glossary.md#term-cache) Java/tests are obsolete design evidence only and must be replaced by the clean
implementation context. This is planning validation, not completed implementation evidence.

Final implementation and documentation evidence:

- Implementation context `019fc96e-494b-74f2-b6e9-5b55d649cd6c` recorded the focused command
  passing 30 tests across `CpuGeneratedKernelArtifactStoreTest`,
  `CpuClassFileKernelGeneratorTest`, and `CpuKernelSpecializationTest`, then exactly one final
  `./gradlew :backends:cpu:test` passing 10 suites and 48 tests. Both runs had zero failures,
  errors, or skips. That context also recorded `git diff --check` passing and confirmed the
  obsolete `CpuGeneratedKernelCache` production and test paths were absent. The documentation
  pass changed no executable Java or tests, so it reused this evidence and did not rerun them.
- Mandatory separate clean documentation-focused context `/root` applied the General,
  API/Javadoc, Backend Guide, Planning, and Example profiles. It independently reviewed the
  final production and test implementation, generated pages, CPU guide, glossary, task/master/
  roadmap records, completed CPU tasks 0001–0002, and the focused architecture, staged Prepare,
  Runtime ownership/hot-path, trusted-root security, and atomic-filesystem contracts. No
  executable correctness or architecture conflict was found.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Backend glossary](../../../../glossary.md#backend); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>
- `./gradlew :backends:cpu:javadoc` — passed with 11 actionable tasks, two executed and nine
  up-to-date. The expected Java Vector API incubator warnings and 26 pre-existing task-0001
  documentation warnings remained non-failing. Generated pages for<span data-glossary-navigation="true"> ([Vector API glossary](../../../../glossary.md#term-vector-api))</span>
  `CpuGeneratedKernelArtifactStore`, `CpuClassFileKernelGenerator`, `CpuGeneratorSchema`,
  `CpuKernelSpecialization`, and the execution package were inspected and contained the final
  format, validation, trust, ownership, lifetime, synthetic-only, and CPU-0004 boundaries.<span data-glossary-navigation="true"> ([lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- `ruby /tmp/check_cpu_0002_markdown.rb` over this task, CPU master plan, roadmap, CPU guide, and
  glossary — passed five files, 672 local links, 287 anchors, balanced fences, final newlines,
  and trailing-whitespace checks. The generic checker name is historical; its input was the five
  CPU-0003 documentation paths.
- Exact-scope validation passed with 13 present changed paths plus the two verified-absent
  obsolete cache paths, accounting for all 15 authorized paths and no excluded path.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- Package/surface, atomic-only publication, weak-retention, no strong/LRU/background/
  serialization/reflection mechanism, trusted-root security, no production-finalizer call, and
  no Runtime reference checks passed. The first combined mechanism scan used overly broad text
  patterns that also matched the required explanatory words and atomic move itself; a second
  invocation then stopped on <span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [LRU glossary](../../../../glossary.md#single-flight-and-least-recently-used-caching); [serialization glossary](../../../../glossary.md#term-serialization); [scan glossary](../../../../glossary.md#term-cumulative-scan); [atomic move glossary](../../../../glossary.md#term-atomic-move))</span>`rg`'s ordinary no-match status. The final explicit empty-result and
  exact-move checker passed and is the recorded result.
- CPU 0003 `Complete` synchronization, CPU 0004–0016 `Draft` rows, absence of their detailed task
  specs, and preserved future Model 0026 Draft dependency passed. Two preliminary status-checker
  invocations validated nothing because one used a non-portable multiline <span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>`sed` expression and
  the next attempted zsh's read-only `status` variable; the corrected `awk`/`task_state` checker
  passed.
- The first aggregate newline checker accidentally retained only the untracked-command result and
  therefore checked three paths; the corrected concatenation passed all 13 changed paths. The
  first final combined scope/status invocation also reused zsh's special `path` variable as a loop
  name and consequently lost command lookup before completing; the corrected <span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup))</span>`scope_item`
  invocation passed every final assertion.
- Final changed-file newline/trailing-whitespace, unchecked-acceptance-item, generated-page,
  surface/mechanism/security/hot-path/status/later-specification, exact-scope, Markdown, and
  `git diff --check` checks passed after the completion record was finalized.

Reasoned no-change conclusions:

- `CpuGeneratedKernel` Javadoc remains accurate: the object is still the immutable strong
  lifetime identity for one loaded hidden class, lookup, exact handle, specialization, and byte
  snapshot. The new weak interner changes neither that object's ownership nor its no-close
  contract.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-method-handle-lookup))</span><span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [identity glossary](../../../../glossary.md#term-identity-meanings); [hidden class glossary](../../../../glossary.md#term-hidden-class); [specialization glossary](../../../../glossary.md#term-specialization); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Prepare and Runtime production APIs and Javadocs remain accurate. The store is only a
  package-private CPU seam; CPU 0004 still owns a production finalizer and prepared-executable
  strong retention, so no shared finalization, [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution), [`PreparedExecutable`](../../../../glossary.md#prepared-executable--preparedexecutable),
  allocation, binding, or hot-path contract changed.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- Config, Model, Planning, Backend Contract, Trace, Compiler, Engine, OpenBLAS provider, other
  backends, extensions, tuning, and benchmarking require no source or documentation change. The
  task adds no public/config surface, operation semantics, capability, route selection, tuning
  evidence, measurement, provider behavior, or cross-module dependency.<span data-glossary-navigation="true"> ([Trace glossary](../../../../glossary.md#trace); [Compiler glossary](../../../../glossary.md#term-compiler); [Engine glossary](../../../../glossary.md#term-engine); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [benchmarking glossary](../../../../glossary.md#term-benchmarking); [operation glossary](../../../../glossary.md#operation); [module glossary](../../../../glossary.md#term-module))</span>
- `ARCHITECTURE.md`, focused architecture pages, ADRs, and the CPU strategy note already permit
  CPU-owned compatible generated-artifact reuse after slot assignment and remain accurate. No
  architecture decision, dependency edge, or module boundary changed, so architecture tests need
  no update.<span data-glossary-navigation="true"> ([dependency edge glossary](../../../../glossary.md#term-dependency-edge))</span>
- Backend-conformance and integration tests remain unchanged because CPU still advertises and
  executes no Model operation. The new behavior is correctly covered by CPU-private synthetic
  store/generator tests. CPU and root Gradle configuration remains unchanged because task 0002
  already supplied the Java 26 Class-File/Vector setup required by this implementation.<span data-glossary-navigation="true"> ([Vector glossary](../../../../glossary.md#term-vector))</span>

## Implementation notes

- Added package-private final `CpuGeneratedKernelArtifactStore`, which canonicalizes its explicit
  root with `toAbsolutePath().normalize()` and lazily owns only the fixed sharded namespace. It
  bounds metadata at 1 MiB, class bytes at 16 MiB, metadata fields at 64 KiB, and the complete
  envelope before allocation.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata))</span>
- The store reads final entries through a no-follow regular-file check and `FileChannel`, checks
  stable bounded size and complete framing, then validates exact metadata and checksum before
  delegating class verification/shape checking and hidden definition to the generator. Invalid
  entries return a safe miss; I/O and atomic-publication failures become contextual<span data-glossary-navigation="true"> ([framing glossary](../../../../glossary.md#term-framing); [checksum glossary](../../../../glossary.md#term-checksum); [shape glossary](../../../../glossary.md#shape); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback))</span>
  `IllegalStateException` subclasses retaining the cause.
- [Publication](../../../../glossary.md#term-publication) uses a unique `.cpuclass-*.tmp` regular file in the final directory, writes and
  forces the whole envelope, and calls `Files.move(..., ATOMIC_MOVE, REPLACE_EXISTING)`. It never
  falls back. Failure cleanup attempts only the current unpublished temporary file, and every
  successful publication is re-read from the deterministic final path before definition.
- Process-wide maps are keyed by canonical root plus defensively copied full metadata. One map
  contains only active attempts; the other contains weak artifact references associated with a<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [artifact glossary](../../../../glossary.md#term-artifact))</span>
  `ReferenceQueue`. Attempts are removed after success or exact unchecked failure, waiters restore
  interruption, unrelated keys do no I/O under the coordination lock, and stale weak keys are
  drained during ordinary requests.
- Split `CpuClassFileKernelGenerator` into deterministic verified-byte emission and stored-byte
  verification/definition while keeping the existing `generate` convenience seam. Exact class
  version, binary name, superclass, flags, interfaces, fields, sole static <span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>`invoke` method,
  descriptor, and code presence are checked before hidden definition and exact handle resolution.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- `CpuKernelSpecialization.artifactCompatibilityBytes()` returns a fresh canonical structural
  encoding. `CpuGeneratorSchema` now owns the fixed envelope/domain, Java/class-file, generated
  binary-name, and entry-name constants used consistently by storage and generator validation.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [validation glossary](../../../../glossary.md#term-validation))</span>

## Completion summary

- Completed changes: replaced the obsolete bounded in-memory cache design with the explicit-root
  durable <span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache))</span>`.cpuclass` store, complete compatibility/envelope/checksum/class-shape validation,
  forced atomic publication, cross-JVM reuse, process-local single-flight, weak loaded interning,
  generator emit/define separation, and defensive specialization encoding.<span data-glossary-navigation="true"> ([checksum glossary](../../../../glossary.md#term-checksum); [shape glossary](../../../../glossary.md#shape); [atomic publication glossary](../../../../glossary.md#term-atomic-publication); [JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [single-flight glossary](../../../../glossary.md#single-flight-and-least-recently-used-caching); [specialization glossary](../../../../glossary.md#term-specialization))</span>
- Files changed or created: 13 present changed paths within the exact 15-path authorization; the
  other two authorized obsolete cache production/test paths are verified absent. The present set
  comprises five CPU production paths, three CPU tests, the CPU guide, glossary, this task, CPU
  master plan, and roadmap.
- Tests and [validation](../../../../glossary.md#term-validation): implementation context `019fc96e-494b-74f2-b6e9-5b55d649cd6c` recorded
  the focused 3-suite/30-test pass and sole final CPU 10-suite/48-test pass with no failures,
  errors, or skips. The clean documentation context reused those results and passed CPU Javadoc,
  generated-page inspection, Markdown, exact-scope, surface/mechanism/security/hot-path/status/
  later-specification, newline/whitespace, and final diff checks.
- Documentation-agent review: mandatory clean context `/root` finalized affected production and
  package Javadocs, CPU guidance, glossary terminology, and synchronized planning evidence without
  changing executable behavior.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: the CPU guide now explains the explicit root, deterministic path,
  envelope validation, atomic publication, concurrency/lifetime behavior, trust boundary, tuning
  distinction, and CPU-0004 integration boundary.<span data-glossary-navigation="true"> ([concurrency glossary](../../../../glossary.md#term-concurrency); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: all changed production declarations and `CpuGeneratedKernel` were reviewed;
  the store and generator contracts were finalized, while schema, specialization, package, and
  loaded-artifact contracts were confirmed accurate.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Glossary impact: the loaded CPU generated-kernel artifact entry now reflects durable byte reuse,
  and a distinct generated-kernel artifact-store entry defines persistence, weak lifetime,
  security, tuning, Runtime, and production-integration boundaries.<span data-glossary-navigation="true"> ([CPU generated-kernel artifact glossary](../../../../glossary.md#cpu-generated-kernel-artifact); [kernel glossary](../../../../glossary.md#kernel); [persistence glossary](../../../../glossary.md#term-persistence); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Architecture, dependency, build, ADR, architecture-test, backend-conformance, and integration
  impact: none; reasoned no-change conclusions are recorded above.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Unresolved issues: None within [CPU](../../../../glossary.md#term-cpu) 0003 [scope](../../../../glossary.md#term-scope).
- Follow-up required: None for CPU 0003. CPU 0004 remains Draft and owns production finalizer
  integration, explicit root supply through composition, and prepared-executable strong retention.

Status: Complete

## Final [planning](../../../../glossary.md#term-planning) status

Status: Complete
