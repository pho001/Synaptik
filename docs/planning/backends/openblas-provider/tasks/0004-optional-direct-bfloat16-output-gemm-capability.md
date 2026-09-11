# Task 0004: Optional Direct BFLOAT16-Output GEMM Capability

## Status

Ready

## Goal

Prove and, only when that proof succeeds, add an optional typed OpenBLAS capability for one
direct `cblas_bgemm`-style call over BFLOAT16 inputs and BFLOAT16 output. The capability must
represent the exact loaded binary rather than an operating-system, processor-architecture, or
OpenBLAS-version guess.

The provider must keep its existing FLOAT32/FLOAT64 GEMM and thread-control symbols mandatory.
A binary that does not export a compatible direct BFLOAT16-output symbol must still load and
retain every current operation. The new operation must not allocate or convert BFLOAT16 inputs,
stage a FLOAT32 output, convert a FLOAT32 output to BFLOAT16, or route through `cblas_sbgemm`.

The mental model is:

```text
existing mandatory four-symbol load
  + optional cblas_bgemm lookup and exact ABI binding
  -> ordinary OpenBlasLibrary always remains usable when the four symbols bind
  -> Optional<direct BFLOAT16 GEMM capability>
       absent: current SGEMM/DGEMM/thread control are unchanged
       present: exact symbol and descriptor are bound, but numerical qualification is separate
  -> explicit real-native invocation checkpoint
       records symbol presence separately from passing direct numerical evidence
```

This task is both a bounded application binary interface (ABI) proof and an implementation task.
If the upstream and executable evidence cannot prove the exact ABI or its compatibility with the
Model MATMUL contract, stop before publishing or invoking the capability and report the task as
incomplete. Do not substitute a conversion pipeline.

## Scope

- Establish a pinned primary-source evidence record for the exact OpenBLAS extension, including
  its first supported release, build-time optionality, C declaration, BFLOAT16 representation,
  scalar passing convention, output type, and documented operation meaning.
- Prove the JDK Foreign Function and Memory (FFM) descriptor on every currently supported CPU
  target ABI: macOS, Linux, and Windows on AArch64 and x86-64. The proof must cover 32-bit CBLAS
  enum and ordinary `blasint` parameters, 16-bit BFLOAT16 scalars passed by value, pointer
  parameters, and `void` return. Any unsupported or unproved ABI remains capability-absent.
- Probe exactly `cblas_bgemm` as an optional symbol after the existing mandatory four-symbol set
  binds. Missing or unbindable optional state must not fail `OpenBlasLibrary.open(...)`, remove a
  current binding, or alter SGEMM, DGEMM, and thread-control behavior.
- Expose one minimal provider-owned typed capability tied to the existing caller-owned
  `OpenBlasLibrary` lifetime. The API must distinguish exact versioned ABI identity from proof of
  numerical usability and expose no `MethodHandle`, address, lookup, arena, provider policy, CPU
  route, or Tensor type.
- Add one validated dense row-major, non-transposed BFLOAT16-input/BFLOAT16-output call over
  caller-owned native `MemorySegment` values, beginning at byte offset zero.
- Use exact raw 16-bit `alpha` and `beta` ABI values. Validate library lifetime; `m`, `n`, and `k`;
  segment nullity, native provenance, lifetime, current-thread accessibility, output writability,
  two-byte alignment, checked required spans, and output/input non-overlap before invocation.
- Fix `CblasRowMajor = 101`, `CblasNoTrans = 111`, `lda = max(1, k)`,
  `ldb = max(1, n)`, and `ldc = max(1, n)`, matching the existing provider GEMM geometry.
- Preserve the operation `C <- alpha * (A x B) + beta * C` with direct BFLOAT16 result storage.
  The provider forwards values and owns no higher-level numerical policy. CPU will consume only
  BFLOAT16 one (`0x3f80`) and positive zero (`0x0000`) for bare MATMUL.
- Add deterministic native-free tests for optional lookup/binding, exact FFM shape, lifetime,
  validation order, raw-bit forwarding, failure translation, and the unchanged mandatory
  capability when the optional symbol is missing or rejected.
- Extend the explicit provider checkpoint so a caller-supplied binary records `ABSENT`,
  `PRESENT_UNQUALIFIED`, or `NUMERICALLY_PROVED` for the versioned capability identity. Absence is
  a valid checkpoint outcome for direct BFLOAT16 while SGEMM/DGEMM/thread checks must still pass.
- Finalize affected Javadocs, package documentation, CPU backend guide, glossary impact, and
  planning evidence in the required separate clean documentation-focused context.

## Out of scope

- `cblas_sbgemm`, FLOAT32 output, an intermediate FLOAT32 matrix, BFLOAT16-to-FLOAT32 input
  conversion, FLOAT32-to-BFLOAT16 result conversion, or any provider-owned conversion helper
- making `cblas_bgemm` mandatory for ordinary library loading
- assuming capability from OpenBLAS version text, header presence, `BUILD_BFLOAT16`, processor
  architecture, instruction-set labels, operating system, package manager, filename, or
  `cblas_sbgemm` presence
- accepting an ABI because it works only on the current Apple AArch64 installation
- transpose flags, column-major, caller offsets or leading dimensions, batching, broadcasting,
  packing, persistent weights, provider-owned loops, layout normalization, or allocation
- CPU capability reporting, qualification, discovery policy, route selection, fallback,
  materialization, fingerprinting, tuning, preparation, finalization, execution, or tracing
- Tensor, Model, Compiler, Planning, Prepare, Runtime, Engine, Config, Training, or another
  concrete backend change
- FLOAT16, integral, complex, sparse, quantized, or mixed-type GEMM
- performance claims, benchmarks, relaxed math, universal determinism, or broad numerical
  certification
- bundled or downloaded binaries, native packaging, default installation lookup, platform-name
  probing, environment or system-property selection, or filesystem search
- changing the mandatory four-symbol ABI, adding optional OpenBLAS config/core-name metadata, or
  supporting ILP64
- architecture-contract, ADR, dependency, Gradle, architecture-test, backend-conformance, or
  integration-test changes
- a detailed CPU 0010D1 specification or CPU 0010E implementation

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially OpenBLAS provider, CPU routes,
  concrete backend ownership, dependency rules, and performance-evidence boundaries
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [CPU backend guide](../../../../backend-guide/cpu-backend.md)
- [OpenBLAS provider master plan](../master-plan.md)
- [Provider 0001 loading and required symbols](0001-library-loading-and-required-symbol-binding.md)
- [Provider 0002 FLOAT32/FLOAT64 GEMM](0002-float32-float64-row-major-gemm-invocation.md)
- [Provider 0003 thread control and checkpoint](0003-thread-control-and-native-provider-checkpoint.md)
- [CPU master plan](../../cpu/master-plan.md)
- [CPU 0010D qualification and fingerprinting](../../cpu/tasks/0010d-installed-openblas-qualification-and-target-fingerprinting.md)
- [Model MATMUL API contract](../../../../api/tensor-api.md#matrix-multiplication-expressions)

Primary upstream evidence to pin during implementation:

- OpenBLAS 0.3.31 release notes, which first advertise the BFLOAT16-output BGEMM extension.
- The matching tagged OpenBLAS `cblas.h`, installed `openblas_config.h`, BFLOAT16 extension
  documentation, build configuration, exported-symbol rules, CBLAS wrapper, and implementation
  sources. Development-branch files may guide discovery but cannot replace one pinned tag.
- JDK 26 `Linker`, `FunctionDescriptor`, `MemorySegment`, and native-layout contracts.

Current local evidence is diagnostic only: the installed OpenBLAS 0.3.34 header declares
`cblas_bgemm` with BFLOAT16 inputs, scalars, and output, but the corresponding binary exports
`cblas_sbgemm` and not `cblas_bgemm`. This proves that header/version/related-symbol inference is
unsafe; it is not a platform rule or a reason to require the symbol.

## Architecture constraints

- `backends/openblas-provider` remains a JDK-only leaf. Dependency direction remains
  `backends/cpu -> backends/openblas-provider`, never the reverse.
- The existing SGEMM, DGEMM, set-thread-count, and get-thread-count symbols remain the complete
  mandatory baseline. Optional BFLOAT16 failure is isolated from that baseline.
- The provider owns only loading, binding, low-level GEMM invocation, thread control, capability
  identity, and its caller-owned lookup lifetime. It does not qualify a CPU route or choose
  fallback.
- The typed capability must be created only from an optional exact bound handle and must check the
  owning library lifetime on every call. It must not become a generic native capability registry.
- CPU and higher layers may treat symbol presence only as structural evidence. Successful direct
  numerical invocation is a separate prerequisite for CPU qualification.
- Caller segments remain borrowed. The provider allocates, copies, converts, retains, closes, or
  replaces no matrix storage and mutates only the logical `C` region through the direct call.
- Ordinary tests remain native-free. The real checkpoint accepts only an explicit caller-supplied
  compatible absolute path and does not discover or install a binary.
- If exact scalar/enum/integer ABI, direct-output semantics, FLOAT32 accumulation with one final
  BFLOAT16 narrowing, or supported-target portability cannot be proved, implementation stops
  before the public capability is exposed.

## ABI and semantic proof gates

All gates are mandatory before production exposure:

1. **Extension identity.** Pinned OpenBLAS sources must show that `cblas_bgemm` is an
   OpenBLAS-specific, build-optional extension, not standardized Netlib CBLAS. The provider
   identity must therefore include an explicit Synaptik schema and the pinned OpenBLAS ABI family;
   no generic “CBLAS BF16” claim is allowed.
2. **C representation.** The installed/tagged configuration must define OpenBLAS `bfloat16` as an
   exact unsigned 16-bit object representation. The FFM call passes raw bits with a 16-bit native
   scalar layout and 16-bit-addressed matrix storage; Java signedness must not reinterpret bits.
3. **Integer and enum carriers.** Supported target evidence must prove 32-bit C `enum` values for
   `CBLAS_ORDER` and `CBLAS_TRANSPOSE` and 32-bit ordinary `blasint`. ILP64 and any ABI with an
   unproved enum representation are rejected.
4. **Exact descriptor.** The downcall is `void` with fourteen ordered arguments:
   three 32-bit enums, three 32-bit dimensions, one 16-bit BFLOAT16 alpha, address A, 32-bit lda,
   address B, 32-bit ldb, one 16-bit BFLOAT16 beta, address C, and 32-bit ldc.
5. **Direct result.** Primary documentation and implementation sources must prove BFLOAT16 A, B,
   and C with no FLOAT32 result parameter. Synaptik must not call a conversion entry point or
   allocate a conversion/staging segment.
6. **MATMUL compatibility.** Primary implementation evidence must establish FLOAT32 product/sum
   accumulation followed by one direct BFLOAT16 result narrowing, with reassociation and fused
   multiply-add permitted, so it can satisfy the Model contract. If implementation variants can
   accumulate in BFLOAT16 or narrow after each term, the capability is incompatible.
7. **Executable proof.** A supplied binary must both export the symbol and pass exact invocation
   cases. At minimum, test `alpha = 1`, `beta = 0`, rectangular geometry and output overwrite;
   signed zero, infinity, and NaN result classes; and the contraction
   `[1, 0.0625, 0.0625] @ [1, 0.0625, 0.0625]` whose once-narrowed result is BFLOAT16 raw
   `0x3f81`. The evidence record must state which claims come from source and which from execution.

The numerical fixture alone is not a universal proof of accumulation order. It complements,
rather than replaces, pinned implementation-source evidence. A binary can be reported as
symbol-present but not numerically proved; that state must never become a CPU route credential.

## Package impact

Existing package changed:

- `io.github.pho001.synaptik.backend.provider.openblas` — adds one optional capability and its
  package-private exact lookup/invocation mechanics beside the existing lifetime owner.

No package is added.

Type placement:

- `io.github.pho001.synaptik.backend.provider.openblas.OpenBlasLibrary` — exposes an optional
  capability lookup because the capability is bound to this exact library lifetime.
- `io.github.pho001.synaptik.backend.provider.openblas.OpenBlasBFloat16Gemm` — public final typed
  capability with package-private construction, a nested closed ABI identity enum, and the one
  direct call. It is not a CPU availability or qualification value.
- `io.github.pho001.synaptik.backend.provider.openblas.OpenBlasBFloat16GemmInvocation` —
  package-private final field-free validator and exact downcall helper.
- `io.github.pho001.synaptik.backend.provider.openblas.OpenBlasNativeBindings` — retains the
  optional exact handle without weakening its mandatory lifetime or four required handles.

Tests remain in the production package to use the existing package-private native-access seam and
lock exact visibility and binding behavior.

## Exact public API intent

Add to `OpenBlasLibrary`:

```java
public Optional<OpenBlasBFloat16Gemm> directBFloat16Gemm();
```

The returned `Optional` is non-null. Presence means only that the exact loaded lookup exported and
successfully bound the versioned ABI; it does not mean CPU qualification or successful native
numerical invocation. Repeated calls return the same immutable capability instance. A capability
obtained before `OpenBlasLibrary.close()` remains an object but every later call fails through the
owner's existing closed-lifetime check.

Add exactly one public final capability type:

```java
public final class OpenBlasBFloat16Gemm {
    public enum Abi {
        OPENBLAS_CBLAS_BGEMM_BFLOAT16_V1
    }

    public Abi abi();

    public void gemm(
            int m,
            int n,
            int k,
            short alpha,
            MemorySegment a,
            MemorySegment b,
            short beta,
            MemorySegment c);
}
```

The class has no public constructor, subclassing point, availability boolean, raw handle/address,
generic operation selector, offsets, strides, transpose flags, leading dimensions, result object,
or qualification claim. `short` values are raw BFLOAT16 bits, not Java numeric conversions.

## Invocation and validation contract

The logical operation is:

```text
C[m,n] <- alpha_bf16 * (A_bf16[m,k] x B_bf16[k,n]) + beta_bf16 * C_bf16[m,n]
```

Validation follows the current SGEMM/DGEMM order and stable failure categories:

1. check the owning `OpenBlasLibrary` is open;
2. validate non-negative `m`, `n`, and `k`, in order;
3. null-check `a`, `b`, and `c`, in order;
4. require native segments, live scopes, and current-thread accessibility, in role order;
5. require writable `c`;
6. require each base address aligned to two bytes;
7. compute exact checked row-major spans using `long` and the existing formulas;
8. require complete A, B, and C byte coverage; and
9. reject C-with-A then C-with-B overlap over the required regions.

After complete validation, `m == 0 || n == 0` is a no-op. Positive-output `k == 0` invokes the
direct symbol so `beta` can apply to C. The exact fourteen arguments use constants and leading
dimensions stated in Scope. The helper uses typed `invokeExact`, rethrows `Error`, and wraps other
invocation failures with a stable operation-specific `IllegalStateException` retaining the cause.

The provider does not canonicalize raw BFLOAT16 scalar bits or promise NaN payloads, exact
accumulation order, bitwise cross-build identity, determinism, or performance.

## Affected files

Expected production paths:

- `backends/openblas-provider/src/main/java/io/github/pho001/synaptik/backend/provider/openblas/FfmOpenBlasNativeAccess.java`
- `backends/openblas-provider/src/main/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasNativeBindings.java`
- `backends/openblas-provider/src/main/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasLibrary.java`
- add `backends/openblas-provider/src/main/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasBFloat16Gemm.java`
- add `backends/openblas-provider/src/main/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasBFloat16GemmInvocation.java`
- `backends/openblas-provider/src/main/java/io/github/pho001/synaptik/backend/provider/openblas/package-info.java`

Expected test and checkpoint paths:

- `backends/openblas-provider/src/test/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasAbiContractTest.java`
- `backends/openblas-provider/src/test/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasLibraryPublicShapeTest.java`
- `backends/openblas-provider/src/test/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasLibraryTest.java`
- add `backends/openblas-provider/src/test/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasBFloat16GemmInvocationTest.java`
- `backends/openblas-provider/src/test/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasNativeCheckpoint.java`

Expected documentation and planning paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- this task
- `docs/planning/backends/openblas-provider/master-plan.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum scope

This task may create or modify at most the 17 exact paths above: six provider production paths,
five provider test/checkpoint paths, two explanatory-documentation paths, and four planning paths.
It adds exactly two production source files and one test source file, with no new package.

If the ABI proof requires another production type, another symbol, native build integration, a
project dependency, CPU implementation, Gradle change, or architecture change, stop and return the
task to planning. Primary evidence notes may be recorded in this task without adding a separate
repository artifact.

## Acceptance criteria

1. The seven ABI and semantic proof gates are recorded against pinned primary sources and pass
   before any public capability is exposed. A failed or ambiguous gate leaves production without
   the new public operation and the task `Incomplete`.
2. Existing mandatory loading remains exactly four-symbol fail-closed loading. Absence or binding
   failure of `cblas_bgemm` returns a usable library with an empty optional capability.
3. Optional lookup probes exactly `cblas_bgemm`; `cblas_sbgemm` and conversion symbols are neither
   probed nor bound by this task.
4. The public surface is exactly the optional library accessor plus the final capability and
   nested ABI enum specified above. It exposes no raw FFM or CPU-owned state.
5. The FFM descriptor and typed `invokeExact` call match all fourteen proven C ABI arguments.
   Tests lock descriptor order, width, return, and raw 16-bit scalar forwarding.
6. Validation order, checked spans, zero behavior, alignment, ownership, overlap, failure
   translation, concurrency, and close-race boundaries match the specified direct call.
7. Production allocates no matrix or conversion storage and contains no BFLOAT16/FLOAT32
   conversion call or `cblas_sbgemm` route.
8. Native-free tests distinguish mandatory-load success, optional-symbol absence, optional-symbol
   binding failure, optional-symbol presence, invocation success/failure, and owner closure.
9. The real-native checkpoint always re-proves existing SGEMM, DGEMM, and thread behavior. It may
   report direct BFLOAT16 `ABSENT` and pass that optional sub-check; if present, it must invoke and
   pass all direct numerical cases before reporting `NUMERICALLY_PROVED`.
10. Capability evidence records the exact ABI identity and pinned source version independently of
    symbol presence and numerical result. No platform or architecture allowlist determines
    presence.
11. A binary with the local 0.3.34 evidence shape—SBGEMM export but no BGEMM export—loads and
    retains existing functionality with the direct BFLOAT16 capability absent.
12. Ordinary tests need no installed OpenBLAS and perform no native lookup or silent skip.
13. The provider remains JDK-only with zero project dependencies; no architecture, Gradle,
    conformance, integration, CPU code, or other module source changes occur.
14. A separate clean documentation-focused context finalizes Javadocs, package documentation,
    CPU guide, glossary impact, task/master/roadmap evidence, and reasoned no-change conclusions.
15. Provider tests, available real-native evidence, provider Javadoc/rendered pages, Markdown
    links/anchors/fences, exact path/public surface/status/frontier, and whitespace checks pass
    before this task becomes `Complete`.

## Tests / validation

After executable Java stabilizes, run exactly one final ordinary provider command:

```bash
./gradlew :backends:openblas-provider:test \
  :backends:openblas-provider:testClasses
```

Run the explicit checkpoint against any caller-supplied compatible binary available for the
implementation environment. Direct BFLOAT16 absence is an expected optional outcome; failure of
existing SGEMM/DGEMM/thread checks is not:

```bash
java --enable-native-access=ALL-UNNAMED \
  --illegal-native-access=deny \
  -cp backends/openblas-provider/build/classes/java/main:backends/openblas-provider/build/classes/java/test \
  io.github.pho001.synaptik.backend.provider.openblas.OpenBlasNativeCheckpoint \
  <ABSOLUTE_OPENBLAS_LIBRARY>
```

The implementation must record the supplied path, exact exported-capability state, versioned ABI
identity when present, numerical outcome when invoked, existing F32/F64/thread outcome, and
restored thread count. When no supplied library is available, record `NOT RUN` with the exact
reason; native-free proof remains mandatory. A later CPU task owns multi-target real-native
qualification. This provider task must nevertheless record primary ABI evidence for all six
supported OS/architecture combinations and must not infer portability from the local checkpoint.

The clean documentation-focused context reuses successful Java evidence unless executable Java
changes and runs:

```bash
./gradlew :backends:openblas-provider:javadoc
git diff --check
git status --short -uall
```

It must inspect rendered Javadocs; validate local Markdown links, anchors, unique headings,
balanced fences, final newlines, and terminology; verify the exact path ceiling and public/package-
private surface; and confirm provider 0004 is the only detailed unfinished provider task, CPU
0010D1 remains master-plan-only `Draft`, and CPU 0010E remains after 0010D1.

Repository-wide validation is deferred to the dependent CPU 0010D1 capability checkpoint or CI.
This task changes one JDK-only leaf without a module edge or architecture rule.

## Dependencies

- Complete provider tasks 0001–0003 for caller-directed lifetime, mandatory bindings,
  FLOAT32/FLOAT64 GEMM, thread control, and checkpoint infrastructure.
- JDK 26 FFM API and the repository Java 26 toolchain.
- Pinned OpenBLAS 0.3.31-or-later primary source for the exact optional extension. A header alone
  or unpinned development branch is insufficient.

No installed binary is required for native-free implementation tests. Public exposure remains
blocked inside this task until the ABI and semantic proof gates pass. Complete CPU 0010D is
downstream compatibility/fingerprint context only, not a provider dependency; CPU code remains
review-only here, and dependent CPU 0010D1 cannot start until this provider task completes.

## Follow-up tasks

- CPU 0010D1 remains a master-plan-only `Draft`. It consumes only a provider capability that is
  both present and CPU-qualified, extends qualification/fingerprints with versioned direct-BF16
  evidence, and adds the exact BFLOAT16 MATMUL route while retaining portable fallback.
- CPU 0010E remains after 0010D1 and must include the optional direct-BF16 capability schema and
  evidence in tuning compatibility.
- No follow-up may replace failed direct-BF16 proof with SBGEMM or conversion staging.

## Architecture impact

Expected impact: None.

This task implements optional low-level GEMM binding and invocation already allowed for the
OpenBLAS provider. It adds no dependency direction or owner. If implementation requires CPU
policy, shared lifecycle state, native packaging, or another architecture responsibility, stop
and report the conflict rather than editing the architecture contract.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are the clean implementation agent for Synaptik OpenBLAS provider task 0004. Work in
/Users/phujka/IdeaProjects/Synaptik. Do not use GSD, commit, or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, the OpenBLAS provider master
plan, and docs/planning/backends/openblas-provider/tasks/0004-optional-direct-bfloat16-output-gemm-capability.md
in full, plus every primary source, prerequisite task, provider source/test, Model MATMUL contract,
CPU boundary, and documentation contract directly referenced by the specification.

Execute the ABI and semantic proof gates first. Implement the optional typed direct-BFLOAT16-
output capability exactly within the seventeen-path ceiling only if every gate passes. Stop before
public exposure and report the task Incomplete if any exact ABI, accumulation, narrowing, or
supported-target fact remains ambiguous. Do not implement cblas_sbgemm, conversions, staging,
downloads, platform/architecture inference, CPU code, or another fallback.

After executable code and validation stabilize, hand the exact diff and evidence to a separate
clean documentation-focused agent in the same overall change. That agent must follow the
documentation rules, independently inspect source/tests/evidence, finalize Javadocs, package
documentation, CPU guide, glossary impact, and planning records, and run the specified
documentation checks without repeating successful Java suites unless executable behavior changes
or a concrete stale-evidence risk is recorded. Do not mark 0004 Complete until every gate passes.
```

## Local decisions

- `cblas_bgemm` is not treated as standardized CBLAS. It is a versioned OpenBLAS-specific
  extension first advertised in OpenBLAS 0.3.31 and available only in builds that supply it.
- The upstream declaration and extension documentation are sufficiently concrete to justify a
  bounded proof task, but current evidence is not sufficient to expose execution before the task
  pins implementation-source accumulation/narrowing behavior and supported-target ABI carriers.
- The loaded symbol, not an operating-system, architecture, version, build flag, installed header,
  or related SBGEMM symbol, decides structural capability presence.
- One optional typed capability object is smaller and safer than an availability boolean plus a
  callable method on every library. It keeps ABI identity beside the operation and cannot be
  mistaken for CPU numerical qualification.
- Direct means that Synaptik passes BFLOAT16 A, B, and C to one BFLOAT16-output ABI. Internal
  OpenBLAS implementation details may use wider arithmetic, as the Model requires, but Synaptik
  introduces no conversion or FLOAT32 staging resource.
- The once-narrowing fixture complements source review because reassociation makes no finite test
  a universal accumulation proof by itself.

## Known limitations

- The local OpenBLAS 0.3.34 binary cannot exercise direct BFLOAT16 output because it lacks the
  `cblas_bgemm` export despite a declaring installed header. That is a valid absent-capability
  result.
- `cblas_bgemm` is an OpenBLAS extension, not a portable Netlib BLAS/CBLAS contract. Each binary
  requires exact optional binding and later CPU numerical qualification.
- The provider checkpoint is bounded and does not certify every shape, value, thread count,
  numerical order, OpenBLAS build, or target.
- ILP64 and any supported-host ABI whose enum or 16-bit scalar calling convention cannot be
  proved remain unsupported for this capability.
- The provider operation alone does not make CPU BFLOAT16 MATMUL eligible. CPU 0010D1 remains
  responsible for qualification, Model compatibility, route resources, fallback, and execution.

## Validation evidence

Empty until implemented.

## Implementation notes

Empty until implemented.

## Completion summary

Empty until implemented.
