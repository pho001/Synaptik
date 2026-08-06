# Task 0003: Durable Generated-Kernel Artifact Store and Cold Loading

## Status

Superseded

## Goal

Add one CPU-private filesystem artifact store that durably reuses deterministic generated class
bytes across models, independently constructed store instances, and later Java Virtual Machine
(JVM) processes. A caller or later composition owner supplies the cache root explicitly. CPU
finalization will ask the store for one already-selected complete specialization after shared
Prepare assigns slots; the store either validates and loads exact compatible bytes or emits,
publishes, validates, defines, and resolves them once.

The durable reuse boundary is class bytes plus exact compatibility metadata, not a strongly held
in-process hidden class:

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

## Scope

- Replace the obsolete bounded strong in-JVM least-recently-used cache implementation and tests
  with a package-private `CpuGeneratedKernelArtifactStore` rooted at an explicit caller-supplied
  `Path`.
- Canonicalize the supplied root once to an absolute normalized store identity. Different store
  objects using the same canonical root and compatibility key address the same artifact. The
  store owns only its fixed namespace below that root; the caller owns root selection,
  permissions, isolation, backup, capacity, and lifecycle.
- Address each entry from a SHA-256 digest over a domain separator and the exact canonical
  compatibility metadata. Use the fixed relative form
  `generated-kernels/v1/sha256/<first-two-hex>/<remaining-62-hex>.cpuclass` beneath the supplied
  root. No model, Tensor, graph, node, value, slot, storage object, address, run, emitter object,
  handle, class-loader, or store-instance identity participates.
- Persist one self-contained, length-delimited binary envelope containing exact compatibility
  metadata, the deterministic generated class bytes, and their SHA-256 checksum. Use no Java
  object serialization, JSON, executable object graph, or sidecar whose partial presence could be
  mistaken for a complete entry.
- Compare the full canonical structural specialization encoding and every fixed generator/
  class-entry compatibility field before accepting stored bytes. A path digest, specialization
  fingerprint, Java hash code, or checksum alone never establishes compatibility.
- Validate bounded lengths, complete parsing, absence of trailing bytes, exact metadata, class
  checksum, Java class-file verification, and exact generated-class shape before definition.
- Treat a missing, structurally incompatible, truncated, corrupt, or invalid envelope as a safe
  miss: never define or invoke its bytes. Regenerate verified bytes and atomically replace the
  unusable entry. Ordinary root, directory, permission, read, write, synchronization, or atomic-
  move failures fail CPU finalization rather than silently bypassing the explicit store.
- Publish through a uniquely named temporary regular file in the final entry directory. Write the
  complete envelope, force its contents, and atomically move it to the deterministic final path.
  Never fall back to a non-atomic move. Clean up only the current attempt's unpublished temporary
  file after failure. Readers accept only the final regular file and therefore observe either an
  earlier complete envelope or a later complete envelope, never a partial publication.
- Allow redundant generation by separate JVM processes. Before publication, re-read an artifact
  that appeared after the original miss and reuse it when it validates. Concurrent writers for
  the same key may still publish equivalent complete envelopes; atomic replacement and mandatory
  post-publication re-read/validation determine the bytes that are defined.
- Coordinate equal requests process-locally with one single-flight attempt keyed by canonical
  root plus full compatibility metadata, including across distinct store instances. Remove the
  attempt after its shared success or exact unchecked failure; failures are not negative-cached.
- Weakly intern loaded `CpuGeneratedKernel` values process-locally by the same full key. The
  interner holds only `WeakReference` values and removes their exact keys through a
  `ReferenceQueue` when stale. It has no strong completed table, capacity, LRU order, expiry,
  background thread, or unloading promise.
- Return one exact weakly interned loaded artifact when it is still live. A caller and, later, a
  CPU `PreparedExecutable` reachable from `PreparedExecution` strongly retain the returned
  artifact, including its hidden class, lookup, exact handle, method type, specialization, and
  bytes. Collection is permitted only after every strong owner disappears.
- Narrowly split `CpuClassFileKernelGenerator` so it remains the sole bytecode-emission,
  verification, hidden-class-definition, and exact-entry-resolution owner: one operation emits
  deterministic verified bytes; a second accepts stored bytes, re-verifies their exact class-file
  compatibility/shape, defines the hidden class, and resolves the existing exact static
  `MethodHandle`. The current convenience generation path may delegate to those two operations.
- Expose a defensive canonical artifact-compatibility encoding from
  `CpuKernelSpecialization`; it must encode every existing structural equality component in fixed
  order and must not expose mutable retained state.
- Preserve fail-closed CPU capability reporting and the existing synthetic generator probes.
- Finalize affected Javadocs, package documentation, CPU guide, glossary, and planning evidence
  in the required separate clean documentation-focused pass after implementation.

## Out of scope

- a bounded strong completed cache, least-recently-used policy, hit-rate policy, time-to-live,
  compatible-age invalidation, access-time correctness, automatic disk eviction, disk quota,
  pruning, background cleanup, watcher, daemon, executor, or service;
- attacker-controlled bytecode safety, signature authentication, sandboxing, bytecode
  instrumentation, permission hardening, remote/shared-host cache safety, or a claim that SHA-256
  makes untrusted executable bytes safe;
- a public or generic cache/store API, global service locator, public filesystem format, Config
  setting, default cache location, environment-variable lookup, home-directory convention, or
  Engine/composition implementation;
- Model operation semantics, family lowering implementations, Tensor evaluation, capability
  advertisement, a production CPU analyzer/finalizer/preparer, route selection, candidate
  generation, slot assignment, `PreparedExecutable` implementation, generated
  `BoundInvocation`, or any CPU 0004 behavior;
- replacing the existing exact static `MethodHandle` with reflection,
  `Method.invoke`, `invokeWithArguments`, a generated bound invocation, or another invocation
  abstraction;
- Runtime disk access, cache access, hashing, checksum validation, class-file parsing, class
  definition, method lookup, route selection, argument discovery, or storage processing;
- workload-tuning cache, model-plan cache, tuning evidence, measurements, benchmarks, policy,
  statistics, or mutation of tuning artifacts;
- source persistence, native code, ahead-of-time compilation, named-class loading, class-loader
  registries, module-layer changes, or serialization of a loaded class, lookup, or method handle;
- dependency, Gradle, Java toolchain, architecture, ADR, architecture-test,
  backend-conformance, integration-test, shared-module, other-backend, or provider changes;
- CPU tasks 0004–0016 specifications or behavior; and
- Model task 0026 or any FLOAT16 or mixed-precision semantic/capability claim.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
  - Core invariants
  - `modules/runtime`
  - `modules/prepare`
  - Concrete backend modules
  - CPU backend routes
  - Prepare lifecycle
  - Run lifecycle
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [ADR 0002: Backend-owned lowering](../../../../design/decisions/0002-backend-owned-lowering.md)
- [ADR 0010: Staged backend preparation](../../../../design/decisions/0010-staged-backend-preparation.md)
- [CPU backend guide](../../../../backend-guide/cpu-backend.md)
- [CPU kernel strategy](../../../../design/notes/cpu-kernel-strategy.md)
- [Task 0001 CPU capability, representation, binding, and parallel foundation](0001-cpu-capability-representation-binding-and-parallel-foundation.md)
- [Task 0002 Portable Class-File API generator foundation](0002-portable-class-file-api-generator-foundation.md)

## Architecture constraints

- Generated JVM bytecode and its compatible artifact store remain CPU-internal. No shared
  lowering module, separate generated-code backend, or Runtime cache owner is introduced.
- CPU analysis later selects the route, lowering, and complete specialization before finalization.
  This task consumes those already-selected facts and changes none of them.
- Filesystem lookup, validation, emission, publication, class definition, and exact handle
  resolution occur only in CPU finalization after shared slot assignment. Runtime receives an
  immutable recipe retaining the result and never repeats those actions.
- The exact `MethodHandle` entry remains the cold-resolved typed invocation mechanism. Generated
  hot code and Runtime hot calls perform no lookup, reflection, cache work, hashing, validation,
  selection, or argument processing.
- Persistent class bytes are immutable prepared inputs, not run-owned buffer/workspace resources
  and not tuning evidence. Loading them must not allocate physical tensor storage or acquire a
  closeable prepared resource.
- Strong lifetime follows active ownership: a returned artifact and a later prepared executable
  may retain it. Process-global completed retention is forbidden; weak interning creates no
  lifetime guarantee.
- Capability remains fail-closed because durable synthetic class bytes do not establish truthful
  operation coverage.
- If full specialization compatibility does not determine bytecode, or a new target/generator
  fact is required, add that fact to the canonical compatibility contract and schema before reuse.
  Do not add model or runtime identity as a substitute.
- If implementation requires a public type, Config value, production finalizer, shared contract,
  new dependency, non-atomic publication fallback, or architecture change, stop and report the
  conflict.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu` — unchanged public fail-closed capability surface.
- `io.github.pho001.synaptik.backend.cpu.execution` — owns all package-private specialization,
  generation, durable artifact, weak interning, cold definition, and direct invocation machinery.

Packages added or changed:

- No package is added. The existing `execution` package remains the narrow visibility boundary.

Type placement and exact package-private surface:

- `io.github.pho001.synaptik.backend.cpu.execution.CpuGeneratedKernelArtifactStore` — new
  package-private final concrete store. Its only constructor is
  `CpuGeneratedKernelArtifactStore(Path root)`. Its only non-private operation is
  `CpuGeneratedKernel loadOrGenerate(CpuKernelSpecialization specialization,
  CpuFamilyKernelEmitter familyEmitter)`.
- `CpuGeneratedKernelArtifactStore` owns private nested immutable key/envelope records, bounded
  binary parsing, filesystem I/O, atomic publication, the process-local single-flight table, and
  the weak loaded-artifact interner/reference queue. No second top-level cache abstraction is
  added.
- `io.github.pho001.synaptik.backend.cpu.execution.CpuClassFileKernelGenerator` — remains
  package-private and the sole generator/definition owner. Add package-private
  `byte[] generateClassBytes(CpuKernelSpecialization, CpuFamilyKernelEmitter)` and
  `CpuGeneratedKernel defineClassBytes(CpuKernelSpecialization, byte[])`; retain
  `generate(...)` only as a delegating compatibility seam if current tests still need it.
- `io.github.pho001.synaptik.backend.cpu.execution.CpuKernelSpecialization` — add only
  `byte[] artifactCompatibilityBytes()`, returning a defensive copy of the exact canonical
  structural encoding already used for its specialization fingerprint.
- `io.github.pho001.synaptik.backend.cpu.execution.CpuGeneratorSchema` — add only the fixed
  artifact-format/domain constants and generated class/entry naming constants needed by both
  deterministic metadata construction and generator shape validation.
- Delete obsolete `CpuGeneratedKernelCache`; there is no renamed strong cache or compatibility
  adapter.

## Filesystem format and ownership

The envelope uses big-endian fixed-width integers and exact byte sequences in this order:

1. eight ASCII magic bytes `SYNCPUK1`;
2. unsigned 32-bit artifact format version `1`;
3. unsigned 32-bit metadata byte length;
4. unsigned 32-bit class byte length;
5. 32-byte SHA-256 checksum of the class bytes;
6. the exact metadata bytes;
7. the exact class bytes; and
8. end of file, with trailing bytes rejected.

The implementation sets explicit conservative maximum metadata, class, and envelope lengths
before allocation and rejects integer overflow, negative Java representations of unsigned lengths,
truncation, non-regular final entries, and post-size-check races safely.

Canonical metadata contains, in fixed order and length-delimited form:

- artifact format and key-domain version;
- generator schema/behavior version;
- exact generated binary class name derived from the complete specialization fingerprint;
- exact entry name `invoke` and exact `MethodType.descriptorString()`;
- class-file major version from the specialization and required minor version zero;
- exact Java feature/toolchain compatibility value `26` for the current Class-File and incubating
  Vector API contract; and
- the complete `artifactCompatibilityBytes()` value, not merely either SHA-256 fingerprint.

The path digest is SHA-256 over ASCII
`synaptik.cpu.generated-kernel.artifact-key.v1`, one zero byte, and the exact metadata bytes. The
stored metadata must compare byte-for-byte with the request even when the path digest matches.
The class checksum detects accidental damage to the stored payload; it is not authentication.

The root is a trusted local executable-code cache boundary. The caller must ensure that
untrusted users and processes cannot write the root or replace its ancestors. The implementation
does not execute path fragments from the envelope, follows no envelope-supplied path, and rejects
a final entry that is not a regular file, but these checks do not make attacker-controlled
bytecode safe. An attacker able to modify the root can replace both bytes and checksum and is
outside this task's safety claim.

The store creates only its fixed namespace, deterministic shard directory, final envelope, and
unique temporary files beneath the supplied root. It performs no valid-entry deletion, expiry,
eviction, directory sweep, quota enforcement, or background maintenance. Replacement is limited
to the exact final path after that entry has failed validation for the current exact key.

## Validation and failure order

For construction:

1. reject `null` root with `NullPointerException("root")`;
2. convert it to an absolute normalized path without consulting a default or environment value;
3. create/validate only the fixed namespace lazily on first use; and
4. fail with one contextual unchecked artifact-store exception whose cause retains the exact
   filesystem failure if directory or atomic-publication requirements are unavailable.

For `loadOrGenerate(...)`:

1. reject null specialization, then null family emitter, using the existing argument names;
2. obtain and compare the family emitter lowering fingerprint before weak-intern, in-flight, or
   disk lookup; retain the existing mismatch message;
3. build full canonical metadata and the full structural process key;
4. drain stale weak references, then return an exact live weakly interned artifact if present;
5. join or establish the one process-local attempt for the canonical root and full metadata;
6. as attempt owner, read and validate a final stored envelope in format order: path/type and
   bounded size, envelope framing, exact metadata, checksum, then generator-owned class-file
   verification and exact shape;
7. on a valid hit, define the validated bytes and resolve the exact handle;
8. on a missing/corrupt/incompatible entry, generate deterministic verified bytes, construct the
   complete envelope, re-check a concurrently appeared valid final entry, otherwise force and
   atomically publish the temporary envelope;
9. re-read and fully validate the final path after publication, then define only those final
   stored bytes and resolve the exact handle;
10. weakly intern the loaded artifact, publish the exact shared success, and remove the in-flight
    attempt; or publish the exact unchecked failure and remove the attempt without retaining a
    completed failure.

Checksum mismatch never reaches class parsing. Metadata mismatch never reaches checksum or class
definition. Class verification/shape failure never reaches hidden-class definition. Definition
failure never returns an artifact. No validation result depends on file age or access history.

Waiting is uninterruptible with respect to the shared attempt: interruption neither cancels the
attempt nor starts another one. A waiter observes the shared exact success or unchecked failure
and restores its interrupt status before returning or throwing. Different keys never generate or
perform filesystem I/O while holding the process coordination lock.

Exact class shape means: required class-file major/minor version; exact generated internal class
name and `java.lang.Object` superclass; final class; no interface or field; exactly one static
method named `invoke` with the expected descriptor and code; and no second callable entry. Java
26 `ClassFile.verify` must report no error before definition. Definition still uses a hidden
nestmate from the existing full-privilege package lookup and resolves the same exact static
handle.

## Affected files

Expected CPU production paths:

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

Expected planning paths:

- this task
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

Review only, with a reasoned no-change conclusion unless a concrete contradiction is found:

- remaining CPU production/tests and `CpuGeneratedKernel` Javadoc;
- Config, Model, Planning, Prepare, Runtime, Backend Contract, Trace, Compiler, Engine,
  OpenBLAS provider, other backends, extensions, tuning, and benchmark contracts;
- CPU/root Gradle, architecture/focused architecture/ADRs, design notes, architecture tests,
  backend conformance, integration tests, and completed CPU tasks 0001–0002.

## Maximum scope

At most 15 paths: six CPU production paths, four CPU test paths, two explanatory documentation
paths, and three planning paths listed above. Deletion and replacement of each obsolete cache path
count separately in this ceiling.

No other top-level type, helper package, public API, finalizer, Config value, dependency/build
file, shared module, architecture/ADR/design-note, conformance, integration, other backend, or
later task specification may change. If an additional path or broader boundary is required, stop
and update planning rather than hiding the expansion.

## Acceptance criteria

- [x] The obsolete bounded strong LRU cache and its tests are removed; no strong global completed
      map, capacity, eviction, expiry, hit-rate, or age-based correctness remains.
- [x] The only new production type is package-private final
      `CpuGeneratedKernelArtifactStore` with exactly the constructor and operation specified
      above; there is no public/generic abstraction or implicit root.
- [x] Two store instances using the same root and a later JVM process reuse the exact stored class
      bytes without family emission on a valid hit.
- [x] Canonical path/key metadata excludes every model/Tensor/graph/value/slot/storage/run/object
      identity and includes the complete canonical specialization plus every generator,
      class-file, Java/Vector, class-name, entry-name, and entry-descriptor compatibility fact.
- [x] A matching digest/path never bypasses byte-for-byte structural metadata comparison. Tests
      prove unequal structural specializations cannot alias when a derived fingerprint or digest
      is forced to collide.
- [x] Envelope parsing is bounded and exact. Metadata, checksum, truncation, trailing-byte,
      class-file version/name/supertype/flags/interface/field/method/descriptor/code, and verifier
      corruption cases fail before definition in the specified order.
- [x] Corrupt or incompatible entries are never defined. They regenerate and are replaced only by
      a forced complete temporary envelope atomically moved to the exact final path, followed by
      mandatory final re-read/validation.
- [x] The implementation never falls back to non-atomic publication. Concurrent reader/writer and
      multi-process writer tests prove no partial final artifact is observed; redundant writers
      converge on a fully valid deterministic final envelope.
- [x] The security documentation clearly states that checksum and structural verification detect
      accidental corruption but do not authenticate attacker-controlled bytecode; the root must
      be trusted and locally write-isolated by its caller.
- [x] Equal process-local requests, including requests through distinct store instances, share one
      attempt and exact outcome. Different keys progress independently. Failure is fanned out
      exactly, removed, and retriable.
- [x] Loaded interning is weak only. Reference-queue cleanup removes stale keys without a
      background thread, and no test depends on garbage-collection or hidden-class-unloading
      timing for correctness.
- [x] A caller-held artifact remains directly invocable regardless of weak-interner cleanup.
      Later prepared ownership, not the store, is documented as the strong lifetime boundary.
- [x] `CpuClassFileKernelGenerator` remains the sole emission, class verification, hidden
      definition, and exact handle-resolution owner. Stored-byte loading cannot define through a
      second mechanism.
- [x] Existing exact static `MethodHandle` type and `invokeExact` behavior remain unchanged; no
      reflection or generated `BoundInvocation` is introduced.
- [x] Source/bytecode tests prove Runtime hot calls contain no disk, cache, hash, checksum,
      parsing, lookup, validation, route selection, or argument processing.
- [x] No valid entry is invalidated for age, and tests may change timestamps without changing hit
      behavior.
- [x] `CpuCapabilityProvider` remains unchanged and fail-closed; no Model operation, production
      CPU finalizer, task-0004 behavior, Config/tuning/benchmark policy, or later specification is
      added.
- [x] Focused store/generator tests and one final CPU module suite pass after executable Java
      stabilizes.
- [x] Changed production declarations have meaningful complete Javadoc for format, validation,
      failures, concurrency, security, ownership, lifetime, inputs, results, and nullability.
- [x] A separate clean documentation-focused pass finalizes package Javadoc, CPU guidance,
      glossary impact, planning evidence, links, and status without repeating successful Java
      tests unless executable behavior changes or a concrete stale-evidence risk is recorded.
- [x] CPU Javadoc, generated pages, documentation checks, exact 15-path scope, package/surface/
      mechanism/status/later-specification checks, and `git diff --check` pass.

## Tests / validation

Implementation development may run:

```bash
./gradlew :backends:cpu:test --tests '*CpuGeneratedKernelArtifactStoreTest'
./gradlew :backends:cpu:test --tests '*CpuClassFileKernelGeneratorTest' --tests '*CpuKernelSpecializationTest'
```

The focused matrix must cover:

- constructor/request nulls, emitter mismatch, validation order, and stable messages;
- exact path and envelope golden bytes, deterministic metadata/key construction, defensive
  specialization encoding, bounded lengths, integer-overflow rejection, and no trailing bytes;
- same-instance, different-instance/same-root, and child-JVM valid-hit reuse without emission;
- full structural differences, forced fingerprint/path-digest collision handling, and exclusion of
  prohibited identities;
- missing artifact generation; metadata/checksum/truncation/trailing-byte corruption; wrong class
  version/name/superclass/flags/interface/field/method/descriptor/code; verifier failure; and proof
  that rejected bytes are never defined;
- timestamp changes without invalidation;
- failed root/directory/read/write/force/atomic-move behavior without a non-atomic fallback;
- same-process equal-key single-flight across store objects, exact success/failure fan-out,
  interruption restoration, retry, and unrelated-key progress;
- controlled concurrent readers and writers plus coordinated child-JVM writers sharing one root,
  permitting redundant emission while proving the final path is always a complete valid envelope;
- valid concurrently appeared artifact preference and mandatory post-publication re-read;
- weak loaded reuse while live, reference-queue stale-key removal through deterministic test
  control rather than a garbage-collection timing assertion, and continued invocation through a
  caller-held strong artifact;
- generator emit/verify versus stored-byte verify/define split, identical bytes, exact hidden
  class/lookup/handle/type, and unchanged `invokeExact` probes; and
- package-private/final surface, absence of strong completed/LRU/expiry/background/reflection/
  tuning/Runtime-hot-path mechanisms, and unchanged fail-closed public CPU surface.

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

Validate repository-local Markdown links and heading anchors for this task, CPU master plan,
roadmap, CPU guide, and glossary; balanced fences, final newlines, and trailing whitespace; exact
15-path implementation scope; CPU 0001–0002 `Complete`; CPU 0003 synchronized to its
implementation status; CPU 0004–0016 `Draft` without detailed specifications; preserved Model
0026 dependency; and unchanged build/architecture/ADR/design-note/conformance/integration paths.

Repository-wide Java, architecture, backend-conformance, and integration validation remain
deferred to CPU 0009 or continuous integration because this task changes one backend-private
mechanism without a dependency or architecture boundary change. Stop if implementation evidence
requires a repository-wide contract change.

## Dependencies

- Complete CPU 0001 representation, cold binding, exact typed invocation, and worker foundation,
  preserved unchanged.
- Complete CPU 0002 generator schema, immutable structural specialization, family-emitter
  fingerprint contract, deterministic bytes, hidden artifact, and exact static handle.
- Current staged Prepare contract: generated artifact lookup/loading occurs in CPU finalization
  only after shared slot assignment.
- Current Java 26 CPU-local Class-File and Vector API build, unchanged.
- An explicit trusted local cache root supplied by a future caller/composition owner. Task 0003
  defines the private constructor seam but no production owner or default.
- Model task 0026 remains future Draft work and is required only before FLOAT16 semantics or
  backend capability. It does not block this identity-free artifact mechanism for current types.

## Follow-up tasks

- CPU 0004 remains Draft and will own typed portable analysis, final specialization selection, the
  production CPU finalizer, explicit artifact-store construction/root supply by composition, and
  strong retention of the returned artifact in the prepared executable.
- CPU 0005–0009 remain Draft for portable semantic coverage and the capability/conformance
  checkpoint.
- CPU 0010–0015 remain Draft for optional native routes.
- CPU 0016 and shared tuning work remain Draft and own compatible workload-tuning evidence. They
  neither replace nor select the generated-class artifact store.
- Automatic disk eviction, cache administration, remote sharing, hostile-root authentication,
  and prepared-executable serialization have no task and are not implied follow-up requirements.

## Architecture impact

Expected impact: None.

This task implements the existing CPU-owned compatible generated-artifact reuse permission at the
existing post-slot-assignment finalization stage. Durability, validation, and explicit root
ownership stay inside the CPU module and add no public/shared contract, dependency, lifecycle
stage, or Runtime behavior. Stop if implementation reveals otherwise.

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
  not a loaded hidden class and not tuning evidence.
- Full canonical metadata is both stored and compared. SHA-256 selects a bounded deterministic
  path and checks class integrity but never substitutes for structural compatibility or trust.
- A single final file plus force-and-atomic-move publication gives readers one visibility point.
  Multi-file metadata/class/marker layouts were rejected because they create partial-entry states.
- Atomic move support at the supplied root is a requirement. Silent non-atomic fallback was
  rejected because correctness under concurrent processes would become filesystem-dependent.
- Corrupt/incompatible exact-path entries are safe misses and may be atomically replaced only
  after verified regeneration. Compatible entries are immutable and never rewritten for age,
  access, recency, or statistics.
- Process coordination uses canonical root plus full metadata equality, never only the path
  digest. This prevents digest collision from aliasing unequal structures.
- Loaded values are weakly interned only to avoid duplicate definitions while live. The reference
  queue removes stale key/value pairs during ordinary finalization calls; there is no maintenance
  thread and no strong completed retention.
- Process-local single-flight is shared across store instances. Cross-process locking is omitted:
  redundant deterministic generation is allowed, while atomic complete-file publication and
  final revalidation provide safety.
- Stored bytes are always verified again before definition, even when generated by the current
  process. The generator remains the sole class-file verification/definition owner.
- The trusted-local-root boundary is explicit. Cryptographic hashes detect accidental mismatch;
  they do not authenticate executable content from an attacker with write access.
- The existing filename is retained for this planning pass so the user-authorized scope remains
  exactly the current CPU 0003 task path, CPU master plan, and roadmap. Its heading and linked task
  name are authoritative for the revised design.

## Known limitations

- No production CPU finalizer constructs or calls the store yet; CPU 0004 owns integration and
  prepared strong retention.
- CPU still advertises and executes no Model operation. Stored artifacts remain synthetic until
  later family tasks add truthful coverage.
- The caller must supply a trusted local filesystem root that supports required regular-file,
  force, and atomic-move semantics. No default, discovery, portability fallback, capacity
  management, or administrative API is provided.
- A malicious writer with access to the root is outside the security model and can replace
  executable content together with its checksum.
- Weak interning does not promise artifact identity or JIT-profile retention after all strong
  owners disappear, and no unloading or collection time is promised.
- Distinct JVM processes may redundantly emit the same artifact. This is accepted work, not a
  correctness or persistence failure.
- Hash collisions fail closed through exact metadata comparison. The task does not attempt a
  second alternate address for a cryptographic path collision.
- No automatic deletion occurs, so durable valid entries accumulate until the caller or external
  administration manages the explicitly supplied root outside active use.
- Model task 0026 remains future work. This store authorizes no FLOAT16 or mixed-precision
  semantics, specialization selection, capability, or execution.

## Validation evidence

Planning-only evidence from clean planning context `/root`:

- the corrected repository-local checker passed all 369 local links and heading anchors across
  this task, the CPU master plan, and the roadmap; its first invocation had a checker-only Ruby
  quoting syntax error and validated nothing;
- balanced Markdown fences, final newlines, and trailing-whitespace checks passed for all three
  planning paths;
- required task sections, task/master/roadmap `Ready` synchronization, CPU 0001–0002 `Complete`,
  CPU 0004–0016 `Draft`, absence of CPU 0004–0009 task files, and preservation of future Model
  0026 passed;
- `git status --short docs/planning` reported exactly the three permitted planning paths; and
- `git diff --check` passed for the complete current worktree diff.

No Java, test, Javadoc, Gradle, generator, artifact, implementation, architecture, ADR,
conformance, integration, commit, or push command ran in this planning pass. The current
uncommitted cache Java/tests are obsolete design evidence only and must be replaced by the clean
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
  executable correctness or architecture conflict was found.
- `./gradlew :backends:cpu:javadoc` — passed with 11 actionable tasks, two executed and nine
  up-to-date. The expected Java Vector API incubator warnings and 26 pre-existing task-0001
  documentation warnings remained non-failing. Generated pages for
  `CpuGeneratedKernelArtifactStore`, `CpuClassFileKernelGenerator`, `CpuGeneratorSchema`,
  `CpuKernelSpecialization`, and the execution package were inspected and contained the final
  format, validation, trust, ownership, lifetime, synthetic-only, and CPU-0004 boundaries.
- `ruby /tmp/check_cpu_0002_markdown.rb` over this task, CPU master plan, roadmap, CPU guide, and
  glossary — passed five files, 672 local links, 287 anchors, balanced fences, final newlines,
  and trailing-whitespace checks. The generic checker name is historical; its input was the five
  CPU-0003 documentation paths.
- Exact-scope validation passed with 13 present changed paths plus the two verified-absent
  obsolete cache paths, accounting for all 15 authorized paths and no excluded path.
- Package/surface, atomic-only publication, weak-retention, no strong/LRU/background/
  serialization/reflection mechanism, trusted-root security, no production-finalizer call, and
  no Runtime reference checks passed. The first combined mechanism scan used overly broad text
  patterns that also matched the required explanatory words and atomic move itself; a second
  invocation then stopped on `rg`'s ordinary no-match status. The final explicit empty-result and
  exact-move checker passed and is the recorded result.
- CPU 0003 `Complete` synchronization, CPU 0004–0016 `Draft` rows, absence of their detailed task
  specs, and preserved future Model 0026 Draft dependency passed. Two preliminary status-checker
  invocations validated nothing because one used a non-portable multiline `sed` expression and
  the next attempted zsh's read-only `status` variable; the corrected `awk`/`task_state` checker
  passed.
- The first aggregate newline checker accidentally retained only the untracked-command result and
  therefore checked three paths; the corrected concatenation passed all 13 changed paths. The
  first final combined scope/status invocation also reused zsh's special `path` variable as a loop
  name and consequently lost command lookup before completing; the corrected `scope_item`
  invocation passed every final assertion.
- Final changed-file newline/trailing-whitespace, unchecked-acceptance-item, generated-page,
  surface/mechanism/security/hot-path/status/later-specification, exact-scope, Markdown, and
  `git diff --check` checks passed after the completion record was finalized.

Reasoned no-change conclusions:

- `CpuGeneratedKernel` Javadoc remains accurate: the object is still the immutable strong
  lifetime identity for one loaded hidden class, lookup, exact handle, specialization, and byte
  snapshot. The new weak interner changes neither that object's ownership nor its no-close
  contract.
- Prepare and Runtime production APIs and Javadocs remain accurate. The store is only a
  package-private CPU seam; CPU 0004 still owns a production finalizer and prepared-executable
  strong retention, so no shared finalization, `PreparedExecution`, `PreparedExecutable`,
  allocation, binding, or hot-path contract changed.
- Config, Model, Planning, Backend Contract, Trace, Compiler, Engine, OpenBLAS provider, other
  backends, extensions, tuning, and benchmarking require no source or documentation change. The
  task adds no public/config surface, operation semantics, capability, route selection, tuning
  evidence, measurement, provider behavior, or cross-module dependency.
- `ARCHITECTURE.md`, focused architecture pages, ADRs, and the CPU strategy note already permit
  CPU-owned compatible generated-artifact reuse after slot assignment and remain accurate. No
  architecture decision, dependency edge, or module boundary changed, so architecture tests need
  no update.
- Backend-conformance and integration tests remain unchanged because CPU still advertises and
  executes no Model operation. The new behavior is correctly covered by CPU-private synthetic
  store/generator tests. CPU and root Gradle configuration remains unchanged because task 0002
  already supplied the Java 26 Class-File/Vector setup required by this implementation.

## Implementation notes

- Added package-private final `CpuGeneratedKernelArtifactStore`, which canonicalizes its explicit
  root with `toAbsolutePath().normalize()` and lazily owns only the fixed sharded namespace. It
  bounds metadata at 1 MiB, class bytes at 16 MiB, metadata fields at 64 KiB, and the complete
  envelope before allocation.
- The store reads final entries through a no-follow regular-file check and `FileChannel`, checks
  stable bounded size and complete framing, then validates exact metadata and checksum before
  delegating class verification/shape checking and hidden definition to the generator. Invalid
  entries return a safe miss; I/O and atomic-publication failures become contextual
  `IllegalStateException` subclasses retaining the cause.
- Publication uses a unique `.cpuclass-*.tmp` regular file in the final directory, writes and
  forces the whole envelope, and calls `Files.move(..., ATOMIC_MOVE, REPLACE_EXISTING)`. It never
  falls back. Failure cleanup attempts only the current unpublished temporary file, and every
  successful publication is re-read from the deterministic final path before definition.
- Process-wide maps are keyed by canonical root plus defensively copied full metadata. One map
  contains only active attempts; the other contains weak artifact references associated with a
  `ReferenceQueue`. Attempts are removed after success or exact unchecked failure, waiters restore
  interruption, unrelated keys do no I/O under the coordination lock, and stale weak keys are
  drained during ordinary requests.
- Split `CpuClassFileKernelGenerator` into deterministic verified-byte emission and stored-byte
  verification/definition while keeping the existing `generate` convenience seam. Exact class
  version, binary name, superclass, flags, interfaces, fields, sole static `invoke` method,
  descriptor, and code presence are checked before hidden definition and exact handle resolution.
- `CpuKernelSpecialization.artifactCompatibilityBytes()` returns a fresh canonical structural
  encoding. `CpuGeneratorSchema` now owns the fixed envelope/domain, Java/class-file, generated
  binary-name, and entry-name constants used consistently by storage and generator validation.

## Completion summary

- Completed changes: replaced the obsolete bounded in-memory cache design with the explicit-root
  durable `.cpuclass` store, complete compatibility/envelope/checksum/class-shape validation,
  forced atomic publication, cross-JVM reuse, process-local single-flight, weak loaded interning,
  generator emit/define separation, and defensive specialization encoding.
- Files changed or created: 13 present changed paths within the exact 15-path authorization; the
  other two authorized obsolete cache production/test paths are verified absent. The present set
  comprises five CPU production paths, three CPU tests, the CPU guide, glossary, this task, CPU
  master plan, and roadmap.
- Tests and validation: implementation context `019fc96e-494b-74f2-b6e9-5b55d649cd6c` recorded
  the focused 3-suite/30-test pass and sole final CPU 10-suite/48-test pass with no failures,
  errors, or skips. The clean documentation context reused those results and passed CPU Javadoc,
  generated-page inspection, Markdown, exact-scope, surface/mechanism/security/hot-path/status/
  later-specification, newline/whitespace, and final diff checks.
- Documentation-agent review: mandatory clean context `/root` finalized affected production and
  package Javadocs, CPU guidance, glossary terminology, and synchronized planning evidence without
  changing executable behavior.
- Documentation impact: the CPU guide now explains the explicit root, deterministic path,
  envelope validation, atomic publication, concurrency/lifetime behavior, trust boundary, tuning
  distinction, and CPU-0004 integration boundary.
- Javadoc review: all changed production declarations and `CpuGeneratedKernel` were reviewed;
  the store and generator contracts were finalized, while schema, specialization, package, and
  loaded-artifact contracts were confirmed accurate.
- Glossary impact: the loaded CPU generated-kernel artifact entry now reflects durable byte reuse,
  and a distinct generated-kernel artifact-store entry defines persistence, weak lifetime,
  security, tuning, Runtime, and production-integration boundaries.
- Architecture, dependency, build, ADR, architecture-test, backend-conformance, and integration
  impact: none; reasoned no-change conclusions are recorded above.
- Unresolved issues: None within CPU 0003 scope.
- Follow-up required: None for CPU 0003. CPU 0004 remains Draft and owns production finalizer
  integration, explicit root supply through composition, and prepared-executable strong retention.

Status: Complete

## Final planning status

Status: Complete
