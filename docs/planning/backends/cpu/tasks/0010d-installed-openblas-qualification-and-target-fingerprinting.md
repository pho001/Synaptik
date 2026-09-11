# Task 0010D: Installed OpenBLAS Qualification and Target Fingerprinting

## Status

Complete

## Goal

Turn one already loaded, explicitly owned OpenBLAS discovery session into an immutable CPU-private
qualification credential only after cold target, application binary interface (ABI), required-
symbol, thread-control, and bounded numerical checks succeed. Bind that credential to the exact
discovery/coordinator lifetime so a different loaded provider cannot reuse it accidentally.

For an absolute-path load, also derive one stable, versioned, content-based binary and target
fingerprint suitable for later persistent compatibility decisions. A name-loaded library may pass
the same runtime qualification checks, but the JDK lookup exposes no resolved file. It therefore
remains session-only and must never authorize persistent reuse. Loading or binding alone remains
insufficient for either scope.

The mental model is:

```text
bounded discovery load
  -> transfer the exact provider lifetime to the 0010C coordinator
  -> inspect the cold host target and, when path-loaded, the exact binary bytes/header
  -> exclusively install/verify count one and exercise all four required symbols
  -> run bounded SGEMM and DGEMM numerical cases
  -> issue one session-bound qualification credential
     - path load: persistent-compatible binary/target fingerprint
     - name load: session-only identity, never persistent-compatible
  -> CPU analysis may construct OpenBLAS candidates from that credential
```

## Scope

- Add one immutable CPU-private qualification value containing a schema version, exact target
  fingerprint, exact required-symbol inventory, ordinary 32-bit-`blasint` evidence, numerical-
  case version, qualification scope, optional stable binary identity, and an opaque session key.
- Add one bounded cold binary inspector for exact absolute-path selections. It resolves the real
  path for diagnostics, reads a regular file through a bounded streaming digest, parses only the
  supported 64-bit executable headers, and rejects a file whose target disagrees with the host.
- Add one field-free qualifier that consumes immutable discovery metadata and the exact 0010C
  coordinator that owns the transferred provider. It performs all provider work through that
  coordinator, never through another load or uncoordinated invocation.
- Extend the discovery owned-resource transfer with one opaque per-load session key. The
  qualification retains only that key, not an `OpenBlasLibrary`, invocation, native segment,
  coordinator, close action, address, or other mutable resource.
- Extend the coordinator with one narrow cold qualification operation that excludes configuration
  writers and admitted calls, installs and verifies thread count one, executes the qualifier's
  provider callback, and retains the existing original-count restoration ownership. No native
  query or setter is added to ordinary prepared invocation.
- Replace manually asserted `Availability.QUALIFIED` analysis input with an exact successful
  `CpuOpenBlasQualification`. Disabled configuration remains the default. The route plan retains
  the same immutable qualification value, and finalization rejects a coordinator whose session key
  differs before changing provider state or constructing a native recipe.
- Preserve all current 0010B representation masks, 0010C thread candidates and shared permit
  budget, portable alternatives, cost equations, resource declarations, binding, execution, and
  no-late-fallback behavior.
- Add deterministic provider-free tests using fake discovery resources, fake binary-header files,
  and fake OpenBLAS invocation. Ordinary tests perform no installed-library lookup.
- Extend the existing opt-in CPU native checkpoint to run qualification against a caller-supplied
  exact OpenBLAS 0.3.34 path when that installation is available, print the stable target/binary
  fingerprint, exercise later route construction from the credential, and restore the original
  provider count through the coordinator.
- Finalize affected Javadoc, the OpenBLAS package description, CPU backend guide, glossary impact,
  this task, CPU master plan, and roadmap in the same overall change through the mandatory separate
  clean documentation-focused context.

## Out of scope

- CPU 0010E typed tuning-candidate schemas, workload signatures, selected-decision consumption,
  measurement, benchmarking, persistence, cache loading or mutation, objective/budget policy, or
  performance claims.
- Config 0006A, Prepare 0004, Tuning 0001, a persistent artifact format, or any implementation of
  the later cache consumer. This task produces only the stable input identity those owners need.
- Public Config, Engine, CPU construction, Tensor, Compile, Runtime, Prepare, Training, or provider
  API changes; another `BackendId`; Planning ownership or scoring changes; or supported public
  composition.
- Any OpenBLAS-provider source, test, build, symbol, descriptor, or documentation change. The
  provider remains the exact four-symbol JDK Foreign Function and Memory (FFM) leaf completed by
  provider tasks 0001-0003.
- Requiring or binding `openblas_get_config`, `openblas_get_corename`, or another optional symbol.
  Current provider ownership and its exact public surface remain unchanged. A later provider task
  may expose optional metadata for diagnostics, but config/core-name text must never replace the
  content identity defined here or turn loading into qualification.
- Deriving identity from a short loader name, OpenBLAS version/config/core-name text, symbol
  address, real path alone, file timestamp, file key, thread count, Java object identity, or a
  discovery-attempt string.
- Downloading, installing, copying, rewriting, locking, or packaging a native library; filesystem
  search or directory enumeration; environment mutation; package-manager or subprocess use; or a
  hidden registry, singleton, watcher, refresh task, or retry loop.
- Supporting 32-bit processes, big-endian targets, ILP64/OpenBLAS `USE64BITINT`, 32-bit executable
  formats, universal/fat Mach-O containers, architectures other than x86-64 and AArch64, or an
  unrecognized operating system. Such a target remains portable-only until a later task adds exact
  evidence.
- Broad numerical, determinism, cross-build, security, or performance certification. Qualification
  is limited to the exact current four symbols, ABI contract, target, and bounded cases.
- MATMUL semantic or representation expansion: no batch, broadcast, epilogue, transpose flag,
  packing, persistent weights, empty geometry, new data type, or new provider call.
- Provider calls, target probing, file I/O, digesting, qualification, route selection, cache work,
  or mutation during deterministic backend analysis, Runtime cold binding, or hot execution.
- Generated-code, emitter, Class-File schema, Java-oracle, decompilation, or generated-performance
  work. This task changes no generated implementation; if generated code becomes necessary, stop
  and replan with the repository's clean-Java oracle discipline.
- Architecture contract, architecture explanation, ADR, module dependency, Gradle, integration-
  test, or other-module changes.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially concrete-backend ownership,
  staged preparation, CPU routes, OpenBLAS provider direction, Runtime prohibitions, and tuning
  artifact compatibility
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Runtime, Prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Performance evidence and model autotuning](../../../../architecture/performance-evidence-and-tuning.md)
- [CPU backend guide](../../../../backend-guide/cpu-backend.md)
- [CPU master plan](../master-plan.md)
- [CPU 0010 narrow OpenBLAS route](0010-narrow-openblas-blas-compatible-native-route.md)
- [CPU 0010A discovery and internal composition](0010a-automatic-openblas-discovery-and-internal-composition-foundation.md)
- [CPU 0010B bounded representations](0010b-bounded-openblas-matmul-representation-expansion.md)
- [CPU 0010C coordinated threads and shared budget](0010c-coordinated-openblas-thread-candidates-and-shared-cpu-thread-budget.md)
- [OpenBLAS provider master plan](../../openblas-provider/master-plan.md)
- [Provider 0001 loading and binding](../../openblas-provider/tasks/0001-library-loading-and-required-symbol-binding.md)
- [Provider 0002 row-major GEMM](../../openblas-provider/tasks/0002-float32-float64-row-major-gemm-invocation.md)
- [Provider 0003 thread control and checkpoint](../../openblas-provider/tasks/0003-thread-control-and-native-provider-checkpoint.md)
- [Config master plan](../../../modules/config/master-plan.md)
- [Prepare master plan](../../../modules/prepare/master-plan.md)
- [Runtime master plan](../../../modules/runtime/master-plan.md)
- [Engine master plan](../../../modules/engine/master-plan.md)
- [Tuning master plan](../../../tools/tuning/master-plan.md)

## Architecture constraints

- Planning continues to select only `BackendId("cpu")`. CPU owns installed-provider
  qualification, target/binary identity, native-route eligibility, representation/thread
  candidates, finalization, and provider coordination.
- `backends/cpu -> backends/openblas-provider` remains the only direction. CPU consumes the
  provider's current exact loader, SGEMM/DGEMM, and thread query/set surface. The provider gains no
  discovery, qualification, fingerprint, CPU policy, fallback, or persistence knowledge.
- Discovery, qualification, and selection remain distinct. `LOADED` proves only complete binding;
  a successful credential proves the bounded checks; CPU analysis still decides route eligibility
  and profitability from immutable inputs.
- Qualification is cold and explicitly invoked before deterministic backend analysis. Analysis
  receives only immutable qualification and cost/configuration facts and performs no provider
  call, host property read, file access, digest, probe, benchmark, cache work, or mutation.
- The 0010C coordinator remains the sole owner after transfer. Qualification must use its writer/
  call exclusion and restoration lifecycle. A qualification cannot outlive or authorize a
  different provider session merely because binary fingerprints compare equal.
- Finalization validates the selected credential against the exact live coordinator before any
  configuration transition. Runtime receives only the already prepared native recipe and never
  sees target, binary, symbol, ABI, or qualification vocabulary.
- A stable persistent identity is a compatibility input, not authentication. The caller remains
  responsible for a trusted installed-library path and for excluding adversarial replacement.
  Ordinary before/after file stability checks detect changes during inspection but do not create a
  code-signing or hostile-filesystem guarantee.
- Any need for a provider API expansion, shared Prepare/Runtime type, Engine dependency, new module
  edge, public API, architecture rule, or persistent-cache contract is a stop condition.

## Current-code evidence

- `FfmOpenBlasNativeAccess` binds exactly `cblas_sgemm`, `cblas_dgemm`,
  `openblas_set_num_threads`, and `openblas_get_num_threads`. Its FFM descriptors use `JAVA_INT`
  for every C integer, establishing the provider side of the ordinary 32-bit-`blasint` contract.
- `OpenBlasAbiContractTest` locks the exact symbol order and SGEMM, DGEMM, setter, and getter
  descriptors. No optional config/core-name symbol exists in the provider seam.
- `CpuOpenBlasDiscoveryResult` retains the exact selected `LibraryName` or
  `AbsoluteLibraryPath`, but intentionally stores no resolved path, binary bytes, qualification,
  provider, or live identity. This is sufficient to distinguish persistent-safe inspection from
  name-loaded session-only qualification.
- `CpuOpenBlasDiscoverySession` already separates immutable discovery metadata from one
  atomically transferable `OwnedResource`. Adding an opaque per-resource key preserves that
  ownership shape without exposing the invocation to analysis.
- `CpuOpenBlasCoordinator` exclusively owns a transferred provider, serializes configuration
  writers against admitted calls, captures the original positive count, and restores it before
  provider close. It is the existing owner for cold qualification calls and later prepared use.
- `CpuOpenBlasInvocation` already exposes exactly the four operations needed to exercise the four
  required symbols through provider-free fakes. No provider surface or alternate loader is needed.
- `OpenBlasRouteConfig` currently accepts a manually asserted `Availability.QUALIFIED` value.
  Replacing that assertion with a successful session-bound qualification is the smallest way to
  prevent loading or copied diagnostic strings from authorizing native candidates.
- `CpuOpenBlasRoutePlan` already carries the selected thread candidate, fixed count/demand,
  representation, resources, and complete costs. It can retain the immutable qualification so
  finalization can verify the exact coordinator association without Runtime involvement.
- `CpuOpenBlasNativeCheckpoint` already covers both precisions, all eight representation masks,
  thread counts one/two, call overlap, writer exclusion, and restoration against OpenBLAS 0.3.34.
  This task extends it with qualification and fingerprint evidence rather than repeating
  representation or performance proof.

## Exact design and invariants

### Qualification value and session binding

Add technically public but unsupported-internal `CpuOpenBlasQualification` under
`internal.route.nativeblas.openblas`, because `internal.prepare` must consume it. Its construction
remains package-private and only `CpuOpenBlasQualifier` can issue a successful value.

The value contains:

- qualification schema version `1`;
- `Scope.PERSISTENT_BINARY` or `Scope.SESSION_ONLY`;
- one immutable `TargetFingerprint`;
- the exact ordered required-symbol list
  `[cblas_sgemm, cblas_dgemm, openblas_set_num_threads, openblas_get_num_threads]`;
- `BlasIntAbi.C_INT_32`, backed by the provider descriptor contract plus successful dimension-
  sensitive native cases;
- numerical-case version `SYNAPTIK_OPENBLAS_GEMM_QUALIFICATION_V1`;
- optional `BinaryIdentity`, present exactly for `PERSISTENT_BINARY`; and
- one opaque non-serializable session key used only for exact live-owner comparison.

All collections and byte sequences are defensively copied. Equality and hash code for persistent
compatibility facts exclude the opaque key through a separate immutable
`PersistentIdentity` projection. The qualification object itself must not implement a general
serialization interface. `persistentIdentity()` returns empty for `SESSION_ONLY` and the complete
versioned projection for `PERSISTENT_BINARY`.

The discovery owned resource creates one fresh opaque key per successful load. Transfer carries
the same key into the coordinator. The qualifier copies only the key into the credential.
Finalization calls a narrow qualification/coordinator association check before `configure`; a
credential from another load is rejected even if its persistent fingerprint is equal. Tests must
prove cross-session rejection and same-session reuse across repeated preparations.

### Stable target fingerprint

Production target probing occurs exactly once per explicit qualification attempt and records:

- target schema version `1`;
- operating-system family `MACOS`, `LINUX`, or `WINDOWS`;
- canonical architecture `AARCH64` or `X86_64`;
- native address width `64` from `ValueLayout.ADDRESS.byteSize()`; and
- native byte order `LITTLE_ENDIAN`.

Normalize `os.name` and `os.arch` with the same trim/lower-case/`Locale.ROOT` rules already used by
discovery. Recognize `aarch64` and `arm64` as `AARCH64`, and `amd64` and `x86_64` as `X86_64`.
Reject absent or unrecognized values, an `OTHER` operating system, non-64-bit address width, or
non-little-endian execution. Tests inject target facts and do not mutate system properties.

Do not include operating-system release, JVM vendor/version, hostname, user path, processor count,
or volatile runtime state. These facts are either irrelevant to the current C ABI or belong to
0010E's later hardware/workload compatibility, not this stable target identity.

### Exact path-loaded binary identity

Add package-private `CpuOpenBlasBinaryInspector`. For an `AbsoluteLibraryPath` selection it:

1. resolves `toRealPath()` once and requires a regular file without following another link after
   resolution;
2. opens that exact real path read-only and captures basic size, last-modified time, and file key
   before inspection;
3. requires a positive size no greater than `1 GiB` and streams the complete bytes through
   SHA-256 with a fixed bounded buffer while retaining only the bounded header prefix;
4. parses exactly one supported 64-bit header and requires its format/machine to agree with the
   target fingerprint; and
5. rereads basic attributes and rejects a changed size, last-modified time, or unequal available
   file key before returning.

Supported headers are deliberately closed:

| Target | Required executable evidence |
|---|---|
| macOS AArch64 | thin 64-bit Mach-O, `MH_MAGIC_64`/swapped equivalent, `CPU_TYPE_ARM64` |
| macOS x86-64 | thin 64-bit Mach-O, `MH_MAGIC_64`/swapped equivalent, `CPU_TYPE_X86_64` |
| Linux AArch64 | ELF magic, class `ELFCLASS64`, native little-endian data, `EM_AARCH64` |
| Linux x86-64 | ELF magic, class `ELFCLASS64`, native little-endian data, `EM_X86_64` |
| Windows AArch64 | DOS header plus bounded PE offset, PE signature, `IMAGE_FILE_MACHINE_ARM64`, PE32+ optional header |
| Windows x86-64 | DOS header plus bounded PE offset, PE signature, `IMAGE_FILE_MACHINE_AMD64`, PE32+ optional header |

Reject fat/universal Mach-O, 32-bit Mach-O/ELF/PE, truncated/oversized offsets, unknown machines,
format/OS mismatch, target mismatch, zero/oversized files, and file changes. Header parsing uses
explicit unsigned reads and checked bounds; it never maps or executes the file.

`BinaryIdentity` contains binary schema version `1`, digest algorithm `SHA-256`, the lowercase
64-hex digest, exact byte length, executable format, and canonical machine. The real path, original
selection, file key, and timestamps are diagnostic inspection facts only and are excluded from
persistent equality so an identical binary may move without becoming incompatible. Any content,
length, format, machine, target, or schema change invalidates compatibility.

Because the provider was already loaded before inspection, require the file-stability checks and
document the trusted-installation/no-concurrent-replacement precondition. Do not claim that JDK
`libraryLookup(Path, ...)` exposes a retained file descriptor or cryptographically associates a
later pathname read with already mapped pages.

For `LibraryName`, perform no filesystem guess, path lookup, symbol-address inspection, or digest.
The binary identity is absent and scope is exactly `SESSION_ONLY`.

### Required-symbol and ordinary-ABI evidence

Discovery status must be `LOADED`; its completed provider contract means all four required symbols
were resolved together. Qualification records that exact list rather than accepting a caller list.
It then uses the coordinator to:

1. quiesce admitted calls and writers;
2. query a positive original count, set count one, and query one back through the existing 32-bit
   C-int getter/setter descriptors;
3. execute the exact SGEMM and DGEMM cases below through the current int-dimension provider methods;
   and
4. leave original-count ownership with the coordinator, which may later install a selected 0010C
   count and must restore the captured original before provider close.

The existing provider descriptor test remains the structural proof that Synaptik requests the
ordinary ABI. The installed binary supplies behavioral evidence by accepting the count round trip
and distinct positive `m`, `n`, and `k` cases with correct row-major results. If future optional
config text is available and contains the OpenBLAS `USE64BITINT` marker, qualification must reject
it; absence of optional metadata neither qualifies nor disqualifies and never changes identity.
This task binds no optional symbol, so current results contain no config/core-name field.

The bounded checks are not a safe general detector for every arbitrarily mislabeled ABI. The task
therefore qualifies only trusted installed OpenBLAS builds under the provider's exact ordinary-ABI
contract and never treats a successful load of an unknown hostile binary as safe.

### Numerical qualification cases

Allocate fresh confined native segments for every case. Validate all expected results before
issuing a credential and close every arena on success or failure. Use exactly `alpha = 1` and
`beta = 0`, the same row-major non-transposed route contract as CPU 0010.

For both SGEMM and DGEMM run:

- one finite non-square `2 x 3` by `3 x 2` case with mixed signs and a nonzero sentinel-filled
  output, proving distinct dimension positions, full contraction, row-major mapping, and beta-zero
  overwrite;
- one `1 x 3` by `3 x 1` case with exactly one contributing NaN product, requiring NaN;
- separate `1 x 2` by `2 x 1` sole-positive-infinity and sole-negative-infinity cases, requiring
  the matching infinity; and
- one `1 x 2` by `2 x 1` all-positive-zero-times-positive case, requiring exact positive zero.

For finite outputs reuse CPU 0010's independent higher-precision sum-of-products oracle. Let `u`
be `2^-24` for FLOAT32 or `2^-53` for FLOAT64, require `2 * k * u < 1`, set
`gamma = (2 * k * u) / (1 - 2 * k * u)`, and accept absolute error at most:

```text
max(gamma * sumAbs, 4 * ulp(target-rounded higher-precision result))
```

Do not compare NaN payloads, impose sequential accumulation, or claim broader accuracy. Any
allocation, provider, thread, result-class, tolerance, cleanup, target, header, file, or invariant
failure produces no credential. Fatal JVM failures propagate; ordinary failures use one stable
CPU-private qualification exception with the original cause retained and no partially usable
result.

### Coordinator-owned cold qualification

The coordinator adds one package-private `qualify` callback boundary rather than composing its
ordinary `configure` and `execute` methods. It acquires one shared-budget permit, claims the
existing writer protocol, waits for admitted calls to quiesce, captures the original count once,
installs and verifies count one, and invokes the callback while the writer flag still excludes new
calls. The Java lock is released before every provider query, setter, GEMM, and callback. On
success it publishes the configured count-one state; on failure it follows 0010C's immediate
restore-and-verify rule and publishes `OPEN_UNCONFIGURED` or terminal `FAILED`. Every path clears
the writer flag and releases the permit. This operation is cold qualification only and is never
retained by or called from a prepared recipe.

### Analysis, finalization, and invalidation

`CpuPartitionAnalysisInputs.OpenBlasRouteConfig` replaces the manually constructible
`Availability` component with `Optional<CpuOpenBlasQualification>`. `DISABLED` has an empty
qualification. Every qualified factory requires a successful credential; there is no factory or
constructor path that turns `LOADED`, a name/version string, or binary digest alone into
qualification.

The selector requires the credential but otherwise preserves the current complete semantic,
representation, thread-capacity, threshold, and cost filters. The selected route plan retains the
credential. Finalization requires the exact coordinator session association and target equality
before installing the selected thread count. It does not reread the binary, rerun qualification,
or compare persistent cache data.

Closing the coordinator invalidates live use through the current lifecycle checks. Replacing a
path-loaded binary changes its digest/length or header identity and therefore rejects later
persistent compatibility. A session-only qualification may be reused only with its exact live
coordinator in the current process and exposes no persistent identity.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.prepare`
- `io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas`
- `io.github.pho001.synaptik.backend.provider.openblas` through its unchanged public leaf API

Packages added or changed:

- `io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas` — owns target/binary
  inspection, cold qualification, the immutable credential, and session association.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — consumes the credential as an immutable
  OpenBLAS eligibility fact and carries it into the selected plan/finalization boundary.

Type placement:

- `CpuOpenBlasQualification` — technically public unsupported-internal immutable credential and
  nested target, binary, evidence, scope, and persistent-identity values used across CPU internal
  packages.
- `CpuOpenBlasBinaryInspector` — package-private bounded file/header/digest operation beside the
  provider selection it inspects.
- `CpuOpenBlasQualifier` — package-private field-free cold orchestration that alone issues a
  successful credential.

No package is added. `CpuCapabilityProvider` remains the sole supported public CPU API.

## Affected files

Expected production paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionAnalysisInputs.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasBinaryInspector.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasCoordinator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasDiscoverySession.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasQualification.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasQualifier.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasRoutePlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasRouteSelector.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/package-info.java`

Expected test and checkpoint paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasBinaryInspectorTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasCoordinatorTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasDiscoveryTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasNativeCheckpoint.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasQualifierTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuOpenBlasRouteSelectorTest.java`
- `testing/backend-conformance/src/test/java/io/github/pho001/synaptik/testing/conformance/CpuOpenBlasRouteConformanceTest.java`

Expected documentation and planning paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- `docs/planning/backends/cpu/tasks/0010d-installed-openblas-qualification-and-target-fingerprinting.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum scope

This task may create or modify only the 24 exact paths above: ten CPU production paths, eight CPU
test/checkpoint paths, one existing backend-conformance test, and five documentation/planning
paths. It adds exactly three production types and two test types, with no new package.

If implementation needs another path, provider/shared/public API, module edge, Gradle change,
architecture document, optional native symbol, generated-code change, persistent artifact, or
0010E behavior, stop and return the task to planning rather than expanding it silently.

## Acceptance criteria

1. Only a successful cold `CpuOpenBlasQualifier` attempt can construct a qualification. A loaded
   discovery result, manually supplied enum/string/digest, or copied diagnostics cannot authorize
   OpenBLAS analysis.
2. Each loaded owned resource has one opaque session key transferred unchanged to its coordinator.
   The credential retains no live resource. Finalization accepts the same-session coordinator and
   rejects another load before provider mutation or recipe construction, even when persistent
   fingerprints are equal.
3. Production target probing snapshots and normalizes the exact supported OS/architecture facts
   once. Deterministic tests cover aliases, absence, unsupported targets, address width, byte
   order, and every target/header agreement and disagreement.
4. Absolute-path inspection streams the complete regular file through SHA-256 within the 1-GiB
   ceiling, performs before/after stability validation, parses the exact closed Mach-O/ELF/PE32+
   header matrix with checked bounds, and produces the exact versioned content identity. Tests use
   generated minimal header fixtures and cover truncation, offsets, formats, machines, sizes,
   mutation, and path diagnostics without loading native code.
5. Persistent identity includes every specified schema, target, binary digest/length/format/
   machine, required-symbol, ABI, and numerical-case fact. Changing any member changes equality;
   path/timestamp/file-key changes alone do not. Defensive copying and lowercase fixed-width digest
   validation are proved.
6. Name-loaded qualification performs no path inference or filesystem inspection, returns
   `SESSION_ONLY`, and has no persistent identity. Exact-name and automatic-name credentials are
   rejected by every persistent-compatibility accessor regardless of version/config text.
7. Qualification requires discovery `LOADED`, the exact four-symbol inventory, the provider's
   locked C-int descriptors, one verified count-one setter/getter transition, and every SGEMM/
   DGEMM finite and exceptional case. Missing, mismatched, failed, or non-finite evidence yields no
   credential.
8. Finite numerical cases use the documented higher-precision gamma/ULP oracle. NaN, signed
   infinity, and unambiguous positive-zero cases use exact result-class assertions. Ordinary tests
   prove all branches with provider-free fakes and never weaken or silently skip an assertion.
9. Coordinator qualification is cold, exclusive, and failure-safe. No provider call occurs under
   its Java lock; writers/admitted calls cannot overlap qualification; interruption and callback
   failure release state; original-count ownership/restoration remains exactly 0010C's; and close
   still restores before provider close.
10. `OpenBlasRouteConfig`, selector, route plan, and finalizer require and retain the exact
    credential while preserving all current 0010B masks, 0010C candidates/costs/capacity,
    declarations, portable fallback, and no-late-fallback behavior. Analysis remains provider-,
    probe-, file-, benchmark-, and cache-free.
11. Backend conformance remains provider-free, native-free, Engine-free, deterministic, and
    exercises a fake-issued qualification through analysis, assignment, same-session coordinated
    finalization, cold binding, and execution. It also proves a foreign-session credential fails.
12. The opt-in native checkpoint, when an exact installed OpenBLAS 0.3.34 path is available, runs
    both precisions and the complete qualification path, emits the target and SHA-256 binary
    identity, constructs current route candidates from that credential, verifies session binding,
    and restores the captured positive thread count in `finally`. Absence is reported as
    `NOT RUN: compatible OpenBLAS 0.3.34 path unavailable`, not a skipped passing JUnit test and
    not a task-completion blocker.
13. Source/API inventory confirms no optional provider symbol, alternate loader, short-name
    fingerprint, provider change, public supported CPU type, mutable static state, registry,
    singleton, service loader, subprocess, file write, cache, Runtime qualification, hot query/set,
    or late fallback.
14. Generated Java/Class-File implementation and schema are unchanged. No generated structural or
    performance evidence is required; this explicit no-change boundary is verified by source and
    changed-path inspection.
15. A distinct clean documentation-focused context independently inspects the implementation and
    tests, finalizes every affected Javadoc/package description, CPU guide, glossary impact, and
    planning evidence under the General, API/Javadoc, Backend Guide, and Planning profiles, and
    records reasoned no-change conclusions for provider docs/API, public APIs, architecture/ADRs,
    shared modules, Gradle/dependencies, generated code, integration, tuning/config/prepare
    follow-ups, and other modules.
16. Focused and final affected-project tests, optional native-checkpoint recording, CPU Javadoc and
    rendered-page inspection, Markdown links/anchors/fences/terminology, exact path/status/order,
    and whitespace validation pass before 0010D becomes `Complete`.

## Tests / validation

Run focused classes while developing. After executable Java and tests stabilize, run once:

```bash
./gradlew :backends:cpu:test :testing:backend-conformance:test
```

Ordinary tests must be deterministic and provider-free. They cover target normalization, binary
headers/digests/stability, qualification success and every failure class, session binding,
coordinator exclusion/restoration, immutable config/plan propagation, finalization, portable
fallback, and backend conformance. Use latches/barriers and fakes for concurrency; do not rely only
on sleeps or wall-clock ordering.

When the implementation environment provides an exact compatible OpenBLAS 0.3.34 path, compile and
run the opt-in checkpoint:

```bash
./gradlew :backends:cpu:testClasses
java --enable-native-access=ALL-UNNAMED --add-modules jdk.incubator.vector \
  -cp backends/cpu/build/classes/java/test:backends/cpu/build/classes/java/main:backends/openblas-provider/build/classes/java/main:modules/model/build/classes/java/main:modules/config/build/classes/java/main:modules/planning/build/classes/java/main:modules/runtime/build/classes/java/main:modules/prepare/build/classes/java/main:modules/backend-contract/build/classes/java/main:modules/trace/build/classes/java/main \
  io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas.CpuOpenBlasNativeCheckpoint \
  <ABSOLUTE_OPENBLAS_0_3_34_LIBRARY>
```

Record the exact path, resolved diagnostic path, target fingerprint, SHA-256 identity, qualified
scope, exercised cases, and restored count. If the library is absent, record the exact `NOT RUN`
reason; do not discover a substitute, mutate the installation, or treat absence as a passing
native check. The deterministic fake suite remains the task's required qualification correctness
gate.

Repository-wide validation is deferred to the 0010E OpenBLAS candidate/tuning-handoff capability
checkpoint or CI. This task changes one production module and one already-established conformance
test, with no module edge, shared contract, build file, or architecture rule. Run `./gradlew test`
only if implementation expands into a repository-wide risk already authorized by this spec; any
scope-expanding cause must otherwise stop planning.

The documentation-focused context receives the exact diff and successful test/native evidence. It
does not repeat a successful Java suite unless it changes executable behavior or records a concrete
stale-evidence risk. After final Javadoc/documentation edits it runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git status --short -uall
```

It renders and inspects every changed generated Javadoc page; validates local Markdown links and
anchors, unique headings, balanced fences, LF/final newlines, terminology, exact path scope, type
placement, public/internal inventory, and no optional-provider-symbol/provider-source change; and
confirms 0010D is `Complete` only after all required gates, 0010E remains a master-plan-only
`Draft` with no detailed task, and CPU 0011 remains `Blocked`.

## Dependencies

- Complete [CPU 0010C](0010c-coordinated-openblas-thread-candidates-and-shared-cpu-thread-budget.md),
  including transferable discovery ownership, the explicit provider coordinator, positive thread
  candidates, fixed permit demand, and restoration through the still-open owner.
- Complete CPU 0010, 0010A, and 0010B for the current four-operation provider-free invocation,
  exact discovery selections, all eight representations, route plan/finalization, numerical
  oracle, and OpenBLAS 0.3.34 checkpoint path.
- Complete OpenBLAS provider tasks 0001-0003 for exact four-symbol binding, locked ordinary C-int
  FFM descriptors, SGEMM/DGEMM calls, thread control, and caller-owned lifetime.
- Stable current CPU analysis/finalization, backend-conformance, package inventory, and
  documentation contracts.

All dependencies are complete. Current code supplies the required ownership and invocation seams;
no architecture, provider, shared-contract, Config, Prepare, Runtime, Engine, or tuning dependency
is needed.

## Follow-up tasks

- CPU 0010E remains the next master-plan-only `Draft` row. It owns typed/versioned portable-versus-
  OpenBLAS candidates, full workload and target/hardware compatibility, selected-evidence
  consumption, and the Prepare/Tuning handoff. It must consume this task's persistent identity
  without weakening session binding or duplicating binary qualification.
- Config 0006A, Prepare 0004, and Tuning 0001 remain separate Draft prerequisites of 0010E and own
  their declarative request, opaque handoff, measurement, and persistent cache responsibilities.
- A later provider task may add optional OpenBLAS config/core-name metadata only after a concrete
  diagnostic consumer justifies the provider API. Such text is never a binary identity or a
  qualification substitute.
- CPU 0011 remains `Blocked` after 0010E on its independent Intel use-case and oneMKL ABI evidence.
- Broader architectures, executable containers, ILP64, OpenBLAS shapes/types, packing, batching,
  epilogues, and relaxed math are not implied follow-ups.

## Architecture impact

Expected architecture impact: None. This task implements CPU-owned qualification and target
compatibility under the existing concrete-backend boundary, consumes the unchanged OpenBLAS leaf,
preserves 0010C ownership/coordination, and keeps all probing outside deterministic analysis and
Runtime.

If implementation requires a provider API or symbol, shared/public lifecycle type, new dependency,
Engine access to CPU internals, Runtime selection, persistent cache contract, or architecture
change, stop and report the exact conflict before editing outside this specification.

## Implementation prompt

Use this prompt in a distinct clean implementation task/thread:

```text
You are the clean implementation agent for Synaptik CPU task 0010D. Work in
/Users/phujka/IdeaProjects/Synaptik. Do not use GSD, commit, or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, the CPU master plan, and
docs/planning/backends/cpu/tasks/0010d-installed-openblas-qualification-and-target-fingerprinting.md
in full, plus every architecture, provider task, prerequisite task, source, test, and
documentation contract directly referenced by that specification. Implement it exactly within
its 24-path ceiling. Stop for any architecture, provider, optional-symbol, shared-contract,
generated-code, public-API, persistent-cache, or scope conflict instead of inventing a boundary.

After executable code and required Java plus available native validation stabilize, hand the exact
diff, affected behavior, architecture constraints, drafted documentation, and recorded evidence to
a distinct clean documentation-focused agent in the same overall change. That agent must follow
docs/developer-guide/documentation-rules.md, independently inspect the implementation/tests,
finalize affected Javadocs, package documentation, CPU guide, glossary impact, task/master/roadmap
evidence, and documentation validation, and must not repeat successful Java suites unless
executable behavior changes or a concrete stale-evidence risk is recorded. Do not mark 0010D
Complete until that pass and every required gate succeed. Keep 0010E master-plan-only Draft.
```

## Local decisions

- Use SHA-256 over complete exact binary bytes plus explicit format/machine/length and target/schema
  fields. A path, timestamp, file key, symbol address, or OpenBLAS version string is not stable
  content identity.
- Permit identical content at a different trusted absolute path to compare compatible. Keep the
  real path and file attributes diagnostic-only, while before/after attributes detect ordinary
  replacement during inspection.
- Reject fat Mach-O initially. Selecting and identifying one loader-chosen slice would require a
  separate exact slice-selection contract; guessing from host architecture would weaken identity.
- Keep name-loaded binaries usable only in the current exact session after full qualification.
  This preserves automatic loader-name usefulness without inventing a resolved path or allowing
  persistent reuse.
- Tie qualification to an opaque per-load key rather than a provider address, invocation object,
  or coordinator reference. Analysis receives immutable identity without retaining or owning live
  native state.
- Run qualification through the 0010C coordinator at count one. This exercises both thread symbols,
  preserves writer/call exclusion and restoration ownership, and avoids a second coordination
  model.
- Reuse the existing bounded gamma/ULP and exceptional-class numerical contract. Qualification
  checks correctness and ABI plausibility, not speed, universal determinism, or every OpenBLAS
  algorithm.
- Do not add optional config/core-name binding in a CPU task. The provider currently owns an exact
  four-symbol surface; optional metadata requires its own provider justification and remains
  diagnostic-only even if later added.
- Make the real 0.3.34 checkpoint opt-in and non-blocking when unavailable, as requested for this
  qualification/identity task. Deterministic provider-free tests remain mandatory and exhaustive
  over the qualification state machine.

## Known limitations

- Persistent identity is available only for absolute-path discovery selections and only for the
  closed supported 64-bit target/header matrix. Name-based JDK lookup exposes no resolved path.
- The file fingerprint describes bytes read after loading. Stable attributes plus a trusted,
  quiescent installation are required; this is compatibility evidence, not hostile-filesystem or
  code-signing security.
- The ordinary 32-bit-`blasint` conclusion combines Synaptik's locked FFM C-int descriptors with
  bounded installed-binary behavior. It is not a safe probe for executing arbitrary untrusted or
  deliberately ABI-mismatched native code.
- Optional OpenBLAS build/config/core-name metadata is not collected because the completed provider
  exposes only the four required symbols. Its absence does not weaken the exact content identity.
- Qualification covers only the current positive row-major non-transposed SGEMM/DGEMM route and
  exact result classes. It does not certify all shapes, thread counts, builds, CPUs, numerical
  orders, or performance.
- There is still no supported public Config or Engine composition API. The credential and
  qualification operation remain unsupported CPU internals pending their later owners.

## Validation evidence

- Implementation context `01a0906a-6f0a-7890-9f94-09b4b70bd3f9` ran
  `./gradlew :backends:cpu:test :testing:backend-conformance:test`: passed. The CPU XML reports
  982 tests, 28 conditional skips, zero failures, and zero errors; backend conformance reports
  9 tests with no skips, failures, or errors. Executable production and test behavior did not
  change afterward, so documentation context `01a09081-2fcc-7593-950e-afac039d7e8a` reused this
  evidence without repeating the Java suites.
- The implementation context compiled and ran the specified native checkpoint against exact path
  `/opt/homebrew/Cellar/openblas/0.3.34/lib/libopenblasp-r0.3.34.dylib`. It passed with target
  macOS/AARCH64/64-bit/little-endian, thin Mach-O64, 16,085,376 bytes, SHA-256
  `aeb5f40d3b5cc0fca84e05e90b8e7da6921cea2c33ca701062b0ccd7b4caf117`,
  `PERSISTENT_BINARY` scope, both floating precisions and bounded result classes, credential-based
  route construction, same-session association, and restoration of original count 16.
- The implementation context passed `git diff --check`, exact 19-path executable scope, source
  inventory, public/internal surface checks, and `javap` checks. Those checks found no optional
  provider symbol, alternate loader, supported public CPU API, mutable static registry, process,
  file-write, cache, Runtime qualification, late fallback, or generated-code/schema change.
- Documentation context `01a09081-2fcc-7593-950e-afac039d7e8a` applied the General,
  API/Javadoc, Backend Guide, Planning, and Example profiles. It independently inspected all
  changed production/test source and focused assertions, then finalized affected Javadocs, the
  package description, CPU guide, glossary, and three planning records without changing
  executable behavior.
- The documentation context ran `./gradlew :backends:cpu:javadoc`: passed with the two expected
  incubating-module warnings and 92 pre-existing missing-`@param` warnings in unchanged
  `CpuPartitionLowering.LoweredPartition` and `CpuPartitionPreparationPlan` records. It inspected
  the rendered qualification, coordinator, route plan, selector, analysis-input, finalizer, and
  package pages for session ownership, target/binary identity, failures, nullability, finalization
  ordering, and non-authentication boundaries.
- The documentation context's Markdown validator passed for the CPU guide, glossary, this task,
  CPU master plan, and roadmap, covering local files/anchors, unique headings, balanced fences,
  terminology, LF/final newlines, and trailing whitespace. Final source/inventory, public-surface,
  exact 24-path, 0010D-Complete/0010E-master-plan-only-Draft ordering, no-detailed-0010E, status,
  and `git diff --check` checks passed.
- No-change conclusions: provider source, API, tests, and documentation remain accurate because
  CPU consumes the unchanged exact four-symbol leaf; public Config, Engine, Tensor, Compile, and
  Training APIs expose no qualification or route activation; architecture, ADRs, and architecture
  tests need no update because ownership, lifecycle boundaries, dependencies, and supported public
  contracts are unchanged; shared Prepare and Runtime gain no type or behavior; generated code,
  schema, Java oracle, and performance evidence are unaffected; Gradle and dependency declarations
  are unchanged; integration tests are not required because no public end-to-end composition
  changed; Config 0006A, Prepare 0004, CPU 0010E, and Tuning 0001 remain separate Draft work; and
  other backends/modules, native packaging, benchmarks, and environment configuration are outside
  this CPU-private change.

## Implementation notes

- Added a technically public but unsupported-internal immutable qualification carrier so
  `internal.prepare` can consume it while keeping construction package-private. The credential
  snapshots schema-one target, exact four-symbol order, ordinary C-int ABI, numerical-case
  version, optional binary identity, and one identity-only per-load key; only its persistent
  projection excludes that live key.
- Added field-free qualification and a bounded inspector. Absolute-path inspection resolves one
  real regular file, streams complete bytes through SHA-256 under the 1-GiB ceiling, retains only
  a bounded header prefix, accepts the closed thin-Mach-O64/ELF64/PE32+ and AArch64/x86-64 matrix,
  and checks ordinary before/after stability facts. Name selections perform no path inference and
  produce session-only credentials.
- Extended discovery transfer and the 0010C coordinator with one fresh per-load key and an
  exclusive count-one qualification callback. Success retains the qualified target; failure
  attempts immediate restore and leaves no credential. Provider calls remain outside the Java
  lock, and the original count remains coordinator-owned through close.
- Replaced manually asserted route availability with an optional successful qualification. The
  selected plan retains that exact credential, and finalization now requires and validates its
  exact coordinator session and target before installing a selected count or realizing a recipe.
  The former borrowed-invocation compatibility constructor cannot authorize qualified native
  finalization.
- Added provider-free deterministic header, target, identity, numerical, failure, concurrency,
  route, finalization, conformance, and inventory coverage, and extended the opt-in checkpoint with
  exact target/binary output and qualification-based route construction.
- Qualification remains bounded compatibility evidence. It does not authenticate the binary,
  safely probe arbitrary untrusted ABI mismatches, certify all OpenBLAS shapes or numerical modes,
  or establish determinism or performance.

## Completion summary

- Completed changes: implemented immutable session-bound OpenBLAS qualification, supported-target
  and binary fingerprinting, coordinator-exclusive cold checks, credential-based analysis and
  finalization, deterministic tests/conformance, native checkpoint evidence, and final affected
  documentation.
- Files changed or created: exactly the 24 authorized paths—ten CPU production paths, eight CPU
  tests/checkpoint paths, one backend-conformance test, and the five documentation/planning paths.
- Tests and validation: reused the implementation context's passing 982-test CPU and 9-test
  conformance run, passing exact native checkpoint, source/inventory/`javap`/scope/whitespace
  checks; the documentation context passed CPU Javadoc, rendered-page inspection, Markdown,
  terminology, exact-scope/status/frontier, and final whitespace checks.
- Documentation-agent review: clean documentation context
  `01a09081-2fcc-7593-950e-afac039d7e8a` independently reviewed implementation and tests,
  finalized all affected Javadocs/package prose and explanatory/planning documentation, and made
  no executable Java or test change.
- Documentation impact: the CPU guide now explains the load-to-qualification lifecycle, supported
  target/header matrix, session-only versus persistent scope, finalization ordering, native
  evidence, and explicit security/numerical/performance limits.
- Javadoc review: affected constructors, accessors, qualification values, coordinator lifecycle,
  route configuration/plan, selector, finalizer, and package contracts document inputs, returns,
  failures, ownership, session binding, and bounded claims.
- Glossary impact: added OpenBLAS qualification and binary identity and synchronized discovery and
  coordinator ownership terms.
- Unresolved issues: None.
- Follow-up required: None for 0010D. CPU 0010E remains the next master-plan-only `Draft`; no
  detailed task was created. The requested universal no-conversion BF16 provider-plus-CPU planning
  change remains separate and is not created or planned here.

Status: Complete
