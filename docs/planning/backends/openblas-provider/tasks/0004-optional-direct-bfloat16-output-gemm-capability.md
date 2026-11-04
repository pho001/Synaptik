# Task 0004: Optional Direct [BFLOAT16](../../../../glossary.md#term-bfloat16)-Output [GEMM](../../../../glossary.md#general-matrix-multiplication--gemm) Capability

## Reader context

This is a blocked proposal, not an implemented `OpenBlasBFloat16Gemm` API. Required FLOAT32/FLOAT64/thread functionality remained usable. A header and related `cblas_sbgemm` export did not establish a direct BFLOAT16-output ABI; the source audit also found intermediate contraction-panel narrowing, a separate semantic failure.

The proposed provider would borrow BFLOAT16 A/B/C segments only after both gates passed; CPU would still qualify the binary and select the route. Missing prerequisites are upstream/exported-binary evidence satisfying both gates—not conversion staging, smaller fixtures or a claim that portable BFLOAT16 is unsupported.

- [Current absent low OpenBLAS route and exact requirements](../../../../backend-guide/cpu-backend.md#current-narrow-openblas-matmul-route) — Separates the optional side branch from portable low types.
- [Current mandatory provider surface](../../../../backend-guide/cpu-backend.md#current-low-level-openblas-foundation) — Distinguishes proposed optional symbols from the working baseline.

## Status

Blocked

## Goal

Prove and, only when that proof succeeds, add an optional typed [OpenBLAS](../../../../glossary.md#term-openblas) capability for one
direct `cblas_bgemm`-style call over [BFLOAT16](../../../../glossary.md#term-bfloat16) inputs and BFLOAT16 output. The capability must
represent the exact loaded [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) rather than an operating-system, processor-architecture, or
OpenBLAS-version guess.

The provider must keep its existing [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) [GEMM](../../../../glossary.md#general-matrix-multiplication--gemm) and thread-control symbols mandatory.
A binary that does not export a compatible direct BFLOAT16-output [symbol](../../../../glossary.md#term-symbol) must still load and
retain every current [operation](../../../../glossary.md#operation). The new operation must not allocate or convert BFLOAT16 inputs,
stage a FLOAT32 output, convert a FLOAT32 output to BFLOAT16, or route through `cblas_sbgemm`.

The [mental model](../../../../glossary.md#term-mental-model) is:

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

This task is both a bounded [application binary interface](../../../../glossary.md#term-application-binary-interface-abi) ([ABI](../../../../glossary.md#application-binary-interface-and-native-bridge)) proof and an implementation task.
If the upstream and executable evidence cannot prove the exact ABI or its compatibility with the
[Model](../../../../glossary.md#term-model) [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) contract, stop before publishing or invoking the capability and report the task as
incomplete. Do not substitute a conversion pipeline.

## [Scope](../../../../glossary.md#term-scope)

- Establish a pinned primary-source evidence record for the exact OpenBLAS extension, including
  its first supported release, build-time optionality, C declaration, BFLOAT16 representation,
  scalar passing convention, output type, and documented operation meaning.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [release glossary](../../../../glossary.md#term-release); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Prove the JDK Foreign Function and Memory (FFM) descriptor on every currently supported CPU
  target ABI: macOS, Linux, and Windows on AArch64 and x86-64. The proof must cover 32-bit CBLAS
  enum and ordinary <span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [Foreign Function and Memory glossary](../../../../glossary.md#application-binary-interface-and-native-bridge); [FFM glossary](../../../../glossary.md#term-ffm); [descriptor glossary](../../../../glossary.md#term-descriptor); [CPU glossary](../../../../glossary.md#term-cpu); [target glossary](../../../../glossary.md#term-target); [AArch64 glossary](../../../../glossary.md#term-aarch64); [x86-64 glossary](../../../../glossary.md#term-x86-64); [CBLAS glossary](../../../../glossary.md#term-cblas))</span>`blasint` parameters, 16-bit BFLOAT16 scalars passed by value, pointer
  parameters, and <span data-glossary-navigation="true"> ([parameters glossary](../../../../glossary.md#term-parameter))</span>`void` return. Any unsupported or unproved [ABI](../../../../glossary.md#term-application-binary-interface-abi) remains capability-absent.
- Probe exactly `cblas_bgemm` as an optional symbol after the existing mandatory four-symbol set
  binds. Missing or unbindable optional state must not fail <span data-glossary-navigation="true"> ([symbol glossary](../../../../glossary.md#term-symbol))</span>`OpenBlasLibrary.open(...)`, remove a
  current binding, or alter SGEMM, DGEMM, and thread-control behavior.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [SGEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [DGEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm))</span>
- Expose one minimal provider-owned typed capability tied to the existing caller-owned
  [`OpenBlasLibrary`](../../../../glossary.md#openblas-library-handle--openblaslibrary) lifetime. The API must distinguish exact versioned ABI identity from proof of
  numerical usability and expose no <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>[`MethodHandle`](../../../../glossary.md#native-numerical-library-and-instruction-architecture), address, lookup, arena, provider policy, CPU
  route, or Tensor type.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [arena glossary](../../../../glossary.md#term-arena); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Add one validated dense row-major, non-transposed BFLOAT16-input/BFLOAT16-output call over
  caller-owned native <span data-glossary-navigation="true"> ([row-major glossary](../../../../glossary.md#term-row-major))</span>[`MemorySegment`](../../../../glossary.md#term-memorysegment) values, beginning at byte offset zero.
- Use exact raw 16-bit `alpha` and `beta` ABI values. Validate library [lifetime](../../../../glossary.md#term-lifetime); `m`, `n`, and `k`;
  segment nullity, native provenance, lifetime, current-thread accessibility, output writability,
  two-byte alignment, checked required spans, and output/input non-overlap before invocation.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance); [byte alignment glossary](../../../../glossary.md#term-byte-alignment))</span>
- Fix `CblasRowMajor = 101`, `CblasNoTrans = 111`, `lda = max(1, k)`,
  `ldb = max(1, n)`, and `ldc = max(1, n)`, matching the existing provider [GEMM](../../../../glossary.md#general-matrix-multiplication--gemm) geometry.
- Preserve the [operation](../../../../glossary.md#operation) `C <- alpha * (A x B) + beta * C` with direct BFLOAT16 result storage.
  The provider forwards values and owns no higher-level numerical policy. CPU will consume only
  BFLOAT16 one (`0x3f80`) and [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) (`0x0000`) for bare [MATMUL](../../../../glossary.md#matrix-multiplication--matmul).
- Add deterministic native-free tests for optional lookup/binding, exact FFM shape, lifetime,
  validation order, raw-bit forwarding, failure translation, and the unchanged mandatory
  capability when the optional symbol is missing or rejected.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [validation glossary](../../../../glossary.md#term-validation))</span>
- Extend the explicit provider [checkpoint](../../../../glossary.md#term-checkpoint) so a caller-supplied [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) records `ABSENT`,
  `PRESENT_UNQUALIFIED`, or `NUMERICALLY_PROVED` for the versioned capability identity. Absence is
  a valid checkpoint outcome for direct BFLOAT16 while SGEMM/DGEMM/thread checks must still pass.
- Finalize affected Javadocs, package documentation, CPU backend guide, glossary impact, and
  planning evidence in the required separate clean documentation-focused context.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- `cblas_sbgemm`, FLOAT32 output, an intermediate FLOAT32 matrix, BFLOAT16-to-FLOAT32 input
  conversion, FLOAT32-to-BFLOAT16 result conversion, or any provider-owned conversion helper<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>
- making `cblas_bgemm` mandatory for ordinary library loading
- assuming capability from [OpenBLAS](../../../../glossary.md#term-openblas) version text, header presence, `BUILD_BFLOAT16`, processor
  architecture, instruction-set labels, operating system, package manager, filename, or
  `cblas_sbgemm` presence
- accepting an [ABI](../../../../glossary.md#term-application-binary-interface-abi) because it works only on the current Apple [AArch64](../../../../glossary.md#term-aarch64) installation
- transpose flags, column-major, caller offsets or leading dimensions, batching, broadcasting,
  packing, persistent weights, provider-owned loops, layout normalization, or allocation<span data-glossary-navigation="true"> ([transpose glossary](../../../../glossary.md#term-transpose); [column-major glossary](../../../../glossary.md#term-column-major); [dimensions glossary](../../../../glossary.md#dimension); [broadcasting glossary](../../../../glossary.md#broadcasting); [weights glossary](../../../../glossary.md#term-weight); [layout glossary](../../../../glossary.md#layout); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- CPU capability reporting, qualification, discovery policy, route selection, fallback,
  materialization, fingerprinting, tuning, preparation, finalization, execution, or tracing<span data-glossary-navigation="true"> ([qualification glossary](../../../../glossary.md#term-qualification); [materialization glossary](../../../../glossary.md#materialization))</span>
- Tensor, Model, Compiler, Planning, Prepare, Runtime, Engine, Config, Training, or another
  concrete backend change<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Engine glossary](../../../../glossary.md#term-engine); [concrete backend glossary](../../../../glossary.md#concrete-backend))</span>
- [FLOAT16](../../../../glossary.md#term-float16), integral, complex, sparse, quantized, or mixed-type [GEMM](../../../../glossary.md#general-matrix-multiplication--gemm)
- performance claims, benchmarks, relaxed math, universal determinism, or broad numerical
  certification
- bundled or downloaded binaries, native packaging, default installation lookup, platform-name
  probing, environment or system-property selection, or filesystem search<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup))</span>
- changing the mandatory four-symbol ABI, adding optional OpenBLAS config/core-name metadata, or
  supporting ILP64<span data-glossary-navigation="true"> ([symbol glossary](../../../../glossary.md#term-symbol); [metadata glossary](../../../../glossary.md#term-metadata); [ILP64 glossary](../../../../glossary.md#term-ilp64))</span>
- architecture-contract, ADR, dependency, Gradle, architecture-test, backend-conformance, or
  integration-test changes<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- a detailed [CPU](../../../../glossary.md#term-cpu) 0010D1 specification or any [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) CPU tuning implementation

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially OpenBLAS provider, CPU routes,
  concrete backend ownership, dependency rules, and performance-evidence boundaries<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [concrete backend glossary](../../../../glossary.md#concrete-backend); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [CPU backend guide](../../../../backend-guide/cpu-backend.md)<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- [OpenBLAS provider master plan](../master-plan.md)
- [Provider 0001 loading and required symbols](0001-library-loading-and-required-symbol-binding.md)
- [Provider 0002 FLOAT32/FLOAT64 GEMM](0002-float32-float64-row-major-gemm-invocation.md)<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm))</span>
- [Provider 0003 thread control and checkpoint](0003-thread-control-and-native-provider-checkpoint.md)<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- [CPU master plan](../../cpu/master-plan.md)
- [CPU 0010D qualification and fingerprinting](../../cpu/tasks/0010d-installed-openblas-qualification-and-target-fingerprinting.md)<span data-glossary-navigation="true"> ([qualification glossary](../../../../glossary.md#term-qualification))</span>
- [Model MATMUL API contract](../../../../api/tensor-api.md#matrix-multiplication-expressions)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

Primary upstream evidence to pin during implementation:

- [OpenBLAS](../../../../glossary.md#term-openblas) 0.3.31 [release](../../../../glossary.md#term-release) notes, which first advertise the [BFLOAT16](../../../../glossary.md#term-bfloat16)-output BGEMM extension.
- The matching tagged OpenBLAS `cblas.h`, installed `openblas_config.h`, BFLOAT16 extension
  documentation, build configuration, exported-symbol rules, CBLAS wrapper, and implementation
  sources. Development-branch files may guide discovery but cannot replace one pinned tag.<span data-glossary-navigation="true"> ([CBLAS glossary](../../../../glossary.md#term-cblas))</span>
- [JDK](../../../../glossary.md#term-java-development-kit-jdk) 26 `Linker`, `FunctionDescriptor`, [`MemorySegment`](../../../../glossary.md#term-memorysegment), and native-[layout](../../../../glossary.md#layout) contracts.

Current local evidence is diagnostic only: the installed OpenBLAS 0.3.34 header declares
`cblas_bgemm` with BFLOAT16 inputs, [scalars](../../../../glossary.md#term-scalar), and output, but the corresponding [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) exports
`cblas_sbgemm` and not `cblas_bgemm`. This proves that header/version/related-[symbol](../../../../glossary.md#term-symbol) [inference](../../../../glossary.md#term-inference) is
unsafe; it is not a platform rule or a reason to require the symbol.

## Architecture constraints

- `backends/openblas-provider` remains a [JDK](../../../../glossary.md#term-java-development-kit-jdk)-only [leaf](../../../../glossary.md#term-leaf). [Dependency direction](../../../../glossary.md#java-platform-and-build-tools) remains
  `backends/cpu -> backends/openblas-provider`, never the reverse.
- The existing SGEMM, DGEMM, set-thread-count, and get-thread-count symbols remain the complete
  mandatory baseline. Optional BFLOAT16 failure is isolated from that baseline.<span data-glossary-navigation="true"> ([SGEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [DGEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>
- The provider owns only loading, binding, low-level GEMM invocation, thread control, capability
  identity, and its caller-owned lookup lifetime. It does not qualify a CPU route or choose
  fallback.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [identity glossary](../../../../glossary.md#term-identity-meanings); [lookup glossary](../../../../glossary.md#term-lookup); [lifetime glossary](../../../../glossary.md#term-lifetime); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- The typed capability must be created only from an optional exact bound handle and must check the
  owning library lifetime on every call. It must not become a generic native capability registry.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry))</span>
- CPU and higher layers may treat symbol presence only as structural evidence. Successful direct
  numerical invocation is a separate prerequisite for CPU qualification.<span data-glossary-navigation="true"> ([symbol glossary](../../../../glossary.md#term-symbol); [qualification glossary](../../../../glossary.md#term-qualification))</span>
- Caller segments remain borrowed. The provider allocates, copies, converts, retains, closes, or
  replaces no matrix storage and mutates only the logical <span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>`C` region through the direct call.
- Ordinary tests remain native-free. The real checkpoint accepts only an explicit caller-supplied
  compatible absolute path and does not discover or install a binary.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>
- If exact scalar/enum/integer ABI, direct-output semantics, FLOAT32 accumulation with one final
  BFLOAT16 narrowing, or supported-target portability cannot be proved, implementation stops
  before the public capability is exposed.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [FLOAT32 glossary](../../../../glossary.md#term-float32); [narrowing glossary](../../../../glossary.md#term-narrowing); [target glossary](../../../../glossary.md#term-target))</span>

## [ABI](../../../../glossary.md#term-application-binary-interface-abi) and semantic proof gates

All gates are mandatory before production exposure:

1. **Extension [identity](../../../../glossary.md#term-identity-meanings).** Pinned [OpenBLAS](../../../../glossary.md#term-openblas) sources must show that `cblas_bgemm` is an
   OpenBLAS-specific, build-optional extension, not standardized Netlib CBLAS. The provider
   identity must therefore include an explicit Synaptik schema and the pinned OpenBLAS ABI family;
   no generic “CBLAS BF16” claim is allowed.<span data-glossary-navigation="true"> ([CBLAS glossary](../../../../glossary.md#term-cblas); [schema glossary](../../../../glossary.md#term-schema); [BF16 glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>
2. **C representation.** The installed/tagged configuration must define OpenBLAS `bfloat16` as an
   exact unsigned 16-bit object representation. The FFM call passes raw bits with a 16-bit native
   scalar layout and 16-bit-addressed matrix storage; Java signedness must not reinterpret bits.<span data-glossary-navigation="true"> ([FFM glossary](../../../../glossary.md#term-ffm); [raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [scalar glossary](../../../../glossary.md#term-scalar); [layout glossary](../../../../glossary.md#layout); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
3. **Integer and enum carriers.** Supported [target](../../../../glossary.md#term-target) evidence must prove 32-bit C `enum` values for
   `CBLAS_ORDER` and `CBLAS_TRANSPOSE` and 32-bit ordinary `blasint`. ILP64 and any ABI with an
   unproved enum representation are rejected.<span data-glossary-navigation="true"> ([ILP64 glossary](../../../../glossary.md#term-ilp64))</span>
4. **Exact [descriptor](../../../../glossary.md#term-descriptor).** The [downcall](../../../../glossary.md#term-downcall) is `void` with fourteen ordered arguments:
   three 32-bit enums, three 32-bit dimensions, one 16-bit BFLOAT16 alpha, address A, 32-bit lda,
   address B, 32-bit ldb, one 16-bit BFLOAT16 beta, address C, and 32-bit ldc.<span data-glossary-navigation="true"> ([arguments glossary](../../../../glossary.md#term-argument); [dimensions glossary](../../../../glossary.md#dimension))</span>
5. **Direct result.** Primary documentation and implementation sources must prove BFLOAT16 A, B,
   and C with no FLOAT32 result parameter. Synaptik must not call a conversion entry point or
   allocate a conversion/staging segment.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [parameter glossary](../../../../glossary.md#term-parameter))</span>
6. **[MATMUL](../../../../glossary.md#matrix-multiplication--matmul) compatibility.** Primary implementation evidence must establish FLOAT32 product/sum
   accumulation followed by one direct BFLOAT16 result narrowing, with reassociation and fused
   multiply-add permitted, so it can satisfy the Model contract. If implementation variants can
   accumulate in BFLOAT16 or narrow after each term, the capability is incompatible.<span data-glossary-navigation="true"> ([narrowing glossary](../../../../glossary.md#term-narrowing); [reassociation and fused multiply-add glossary](../../../../glossary.md#rounding-sites-fma-and-aggregate-floor); [Model glossary](../../../../glossary.md#term-model))</span>
7. **Executable proof.** A supplied binary must both export the symbol and pass exact invocation
   cases. At minimum, test `alpha = 1`, `beta = 0`, rectangular geometry and output overwrite;
   signed zero, infinity, and NaN result classes; and the contraction<span data-glossary-navigation="true"> ([signed zero glossary](../../../../glossary.md#term-signed-zero); [infinity glossary](../../../../glossary.md#term-infinity); [NaN glossary](../../../../glossary.md#term-nan); [contraction glossary](../../../../glossary.md#contraction-dimension))</span>
   `[1, 0.0625, 0.0625] @ [1, 0.0625, 0.0625]` whose once-narrowed result is [BFLOAT16](../../../../glossary.md#term-bfloat16) raw
   `0x3f81`. The evidence record must state which claims come from source and which from execution.

The numerical [fixture](../../../../glossary.md#term-fixture) alone is not a universal proof of accumulation order. It complements,
rather than replaces, pinned implementation-source evidence. A [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) can be reported as
[symbol](../../../../glossary.md#term-symbol)-present but not numerically proved; that state must never become a [CPU](../../../../glossary.md#term-cpu) route credential.

## Package impact

Existing package changed:

- `io.github.pho001.synaptik.backend.provider.openblas` — adds one optional capability and its
  package-private exact lookup/invocation mechanics beside the existing lifetime owner.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup); [lifetime glossary](../../../../glossary.md#term-lifetime); [owner glossary](../../../../glossary.md#term-owner))</span>

No package is added.

Type placement:

- `io.github.pho001.synaptik.backend.provider.openblas.OpenBlasLibrary` — exposes an optional
  capability lookup because the capability is bound to this exact library lifetime.
- `io.github.pho001.synaptik.backend.provider.openblas.OpenBlasBFloat16Gemm` — public final typed
  capability with package-private construction, a nested closed ABI identity enum, and the one
  direct call. It is not a CPU availability or qualification value.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [identity glossary](../../../../glossary.md#term-identity-meanings); [CPU glossary](../../../../glossary.md#term-cpu); [qualification glossary](../../../../glossary.md#term-qualification))</span>
- `io.github.pho001.synaptik.backend.provider.openblas.OpenBlasBFloat16GemmInvocation` —
  package-private final field-free validator and exact downcall helper.<span data-glossary-navigation="true"> ([downcall glossary](../../../../glossary.md#term-downcall))</span>
- `io.github.pho001.synaptik.backend.provider.openblas.OpenBlasNativeBindings` — retains the
  optional exact handle without weakening its mandatory lifetime or four required handles.

Tests remain in the production package to use the existing package-private native-access seam and
lock exact visibility and [binding](../../../../glossary.md#term-binding) behavior.

## Exact public [API](../../../../glossary.md#term-application-programming-interface-api) intent

Add to [`OpenBlasLibrary`](../../../../glossary.md#openblas-library-handle--openblaslibrary):

```java
public Optional<OpenBlasBFloat16Gemm> directBFloat16Gemm();
```

The returned `Optional` is non-null. Presence means only that the exact loaded [lookup](../../../../glossary.md#term-lookup) exported and
successfully bound the versioned [ABI](../../../../glossary.md#term-application-binary-interface-abi); it does not mean [CPU](../../../../glossary.md#term-cpu) [qualification](../../../../glossary.md#term-qualification) or successful native
numerical invocation. Repeated calls return the same [immutable](../../../../glossary.md#term-immutable) capability instance. A capability
obtained before `OpenBlasLibrary.close()` remains an object but every later call fails through the
[owner](../../../../glossary.md#term-owner)'s existing closed-[lifetime](../../../../glossary.md#term-lifetime) check.

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

The class has no public constructor, subclassing point, availability [boolean](../../../../glossary.md#data-type--datatype), raw handle/address,
generic [operation](../../../../glossary.md#operation) selector, offsets, [strides](../../../../glossary.md#term-stride), [transpose](../../../../glossary.md#term-transpose) flags, leading [dimensions](../../../../glossary.md#dimension), result object,
or qualification claim. `short` values are raw [BFLOAT16](../../../../glossary.md#term-bfloat16) bits, not Java numeric conversions.

## Invocation and [validation](../../../../glossary.md#term-validation) contract

The logical [operation](../../../../glossary.md#operation) is:

```text
C[m,n] <- alpha_bf16 * (A_bf16[m,k] x B_bf16[k,n]) + beta_bf16 * C_bf16[m,n]
```

Validation follows the current [SGEMM](../../../../glossary.md#general-matrix-multiplication--gemm)/[DGEMM](../../../../glossary.md#general-matrix-multiplication--gemm) order and stable failure categories:

1. check the owning [`OpenBlasLibrary`](../../../../glossary.md#openblas-library-handle--openblaslibrary) is open;
2. validate non-negative `m`, `n`, and `k`, in order;
3. null-check `a`, `b`, and `c`, in order;
4. require native segments, live scopes, and current-thread accessibility, in role order;
5. require writable `c`;
6. require each base address aligned to two bytes;
7. compute exact checked [row-major](../../../../glossary.md#term-row-major) spans using `long` and the existing formulas;
8. require complete A, B, and C byte coverage; and
9. reject C-with-A then C-with-B overlap over the required regions.

After complete validation, `m == 0 || n == 0` is a no-op. Positive-output `k == 0` invokes the
direct [symbol](../../../../glossary.md#term-symbol) so `beta` can apply to C. The exact fourteen [arguments](../../../../glossary.md#term-argument) use [constants](../../../../glossary.md#term-constant) and leading
[dimensions](../../../../glossary.md#dimension) stated in [Scope](../../../../glossary.md#term-scope). The helper uses typed `invokeExact`, rethrows `Error`, and wraps other
invocation failures with a stable operation-specific `IllegalStateException` retaining the cause.

The provider does not canonicalize raw [BFLOAT16](../../../../glossary.md#term-bfloat16) [scalar](../../../../glossary.md#term-scalar) bits or promise [NaN](../../../../glossary.md#term-nan) payloads, exact
accumulation order, bitwise cross-build [identity](../../../../glossary.md#term-identity-meanings), determinism, or performance.

## Affected files

Expected production paths:

- `backends/openblas-provider/src/main/java/io/github/pho001/synaptik/backend/provider/openblas/FfmOpenBlasNativeAccess.java`
- `backends/openblas-provider/src/main/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasNativeBindings.java`
- `backends/openblas-provider/src/main/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasLibrary.java`
- add `backends/openblas-provider/src/main/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasBFloat16Gemm.java`
- add `backends/openblas-provider/src/main/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasBFloat16GemmInvocation.java`
- `backends/openblas-provider/src/main/java/io/github/pho001/synaptik/backend/provider/openblas/package-info.java`

Expected test and [checkpoint](../../../../glossary.md#term-checkpoint) paths:

- `backends/openblas-provider/src/test/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasAbiContractTest.java`
- `backends/openblas-provider/src/test/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasLibraryPublicShapeTest.java`
- `backends/openblas-provider/src/test/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasLibraryTest.java`
- add `backends/openblas-provider/src/test/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasBFloat16GemmInvocationTest.java`
- `backends/openblas-provider/src/test/java/io/github/pho001/synaptik/backend/provider/openblas/OpenBlasNativeCheckpoint.java`

Expected documentation and [planning](../../../../glossary.md#term-planning) paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- this task
- `docs/planning/backends/openblas-provider/master-plan.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most the 17 exact paths above: six provider production paths,
five provider test/[checkpoint](../../../../glossary.md#term-checkpoint) paths, two explanatory-documentation paths, and four [planning](../../../../glossary.md#term-planning) paths.
It adds exactly two production source files and one test source file, with no new package.

If the [ABI](../../../../glossary.md#term-application-binary-interface-abi) proof requires another production type, another [symbol](../../../../glossary.md#term-symbol), native build integration, a
project [dependency](../../../../glossary.md#term-dependency), [CPU](../../../../glossary.md#term-cpu) implementation, [Gradle](../../../../glossary.md#term-gradle) change, or architecture change, stop and return the
task to planning. Primary evidence notes may be recorded in this task without adding a separate
repository [artifact](../../../../glossary.md#term-artifact).

## Acceptance criteria

1. The seven ABI and semantic proof gates are recorded against pinned primary sources and pass
   before any public capability is exposed. A failed or ambiguous gate leaves production without
   the new public operation and the task <span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [operation glossary](../../../../glossary.md#operation))</span>`Incomplete`.
2. Existing mandatory loading remains exactly four-symbol fail-closed loading. Absence or binding
   failure of <span data-glossary-navigation="true"> ([symbol glossary](../../../../glossary.md#term-symbol); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [binding glossary](../../../../glossary.md#term-binding))</span>`cblas_bgemm` returns a usable library with an empty optional capability.
3. Optional [lookup](../../../../glossary.md#term-lookup) probes exactly `cblas_bgemm`; `cblas_sbgemm` and conversion symbols are neither
   probed nor bound by this task.
4. The public surface is exactly the optional library accessor plus the final capability and
   nested ABI enum specified above. It exposes no raw FFM or CPU-owned state.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
5. The [FFM](../../../../glossary.md#term-ffm) [descriptor](../../../../glossary.md#term-descriptor) and typed `invokeExact` call match all fourteen proven C ABI arguments.
   Tests lock descriptor order, width, return, and raw 16-bit scalar forwarding.<span data-glossary-navigation="true"> ([arguments glossary](../../../../glossary.md#term-argument); [scalar glossary](../../../../glossary.md#term-scalar))</span>
6. Validation order, checked spans, zero behavior, alignment, ownership, overlap, failure
   translation, concurrency, and close-race boundaries match the specified direct call.<span data-glossary-navigation="true"> ([Validation glossary](../../../../glossary.md#term-validation); [alignment glossary](../../../../glossary.md#term-alignment); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [concurrency glossary](../../../../glossary.md#term-concurrency); [race glossary](../../../../glossary.md#term-race))</span>
7. Production allocates no matrix or conversion storage and contains no BFLOAT16/FLOAT32
   conversion call or <span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT32 glossary](../../../../glossary.md#term-float32))</span>`cblas_sbgemm` route.
8. Native-free tests distinguish mandatory-load success, optional-symbol absence, optional-symbol
   binding failure, optional-symbol presence, invocation success/failure, and owner closure.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
9. The real-native checkpoint always re-proves existing SGEMM, DGEMM, and thread behavior. It may
   report direct BFLOAT16 <span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [SGEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [DGEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm))</span>`ABSENT` and pass that optional sub-check; if present, it must invoke and
   pass all direct numerical cases before reporting `NUMERICALLY_PROVED`.
10. Capability evidence records the exact ABI identity and pinned source version independently of
    symbol presence and numerical result. No platform or architecture allowlist determines
    presence.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
11. A binary with the local 0.3.34 evidence shape—SBGEMM export but no BGEMM export—loads and
    retains existing functionality with the direct BFLOAT16 capability absent.<span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [shape glossary](../../../../glossary.md#shape))</span>
12. Ordinary tests need no installed [OpenBLAS](../../../../glossary.md#term-openblas) and perform no native lookup or silent skip.
13. The provider remains JDK-only with zero project dependencies; no architecture, Gradle,
    conformance, integration, CPU code, or other module source changes occur.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [module glossary](../../../../glossary.md#term-module))</span>
14. A separate clean documentation-focused context finalizes Javadocs, package documentation,
    CPU guide, glossary impact, task/master/roadmap evidence, and reasoned no-change conclusions.
15. Provider tests, available real-native evidence, provider Javadoc/rendered pages, Markdown
    links/anchors/fences, exact path/public surface/status/frontier, and whitespace checks pass
    before this task becomes <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`Complete`.

## Tests / [validation](../../../../glossary.md#term-validation)

After executable Java stabilizes, [run](../../../../glossary.md#term-run) exactly one final ordinary provider command:

```bash
./gradlew :backends:openblas-provider:test \
  :backends:openblas-provider:testClasses
```

Run the explicit [checkpoint](../../../../glossary.md#term-checkpoint) against any caller-supplied compatible [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) available for the
implementation environment. Direct [BFLOAT16](../../../../glossary.md#term-bfloat16) absence is an expected optional outcome; failure of
existing [SGEMM](../../../../glossary.md#general-matrix-multiplication--gemm)/[DGEMM](../../../../glossary.md#general-matrix-multiplication--gemm)/thread checks is not:

```bash
java --enable-native-access=ALL-UNNAMED \
  --illegal-native-access=deny \
  -cp backends/openblas-provider/build/classes/java/main:backends/openblas-provider/build/classes/java/test \
  io.github.pho001.synaptik.backend.provider.openblas.OpenBlasNativeCheckpoint \
  <ABSOLUTE_OPENBLAS_LIBRARY>
```

The implementation must record the supplied path, exact exported-capability state, versioned [ABI](../../../../glossary.md#term-application-binary-interface-abi)
[identity](../../../../glossary.md#term-identity-meanings) when present, numerical outcome when invoked, existing [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding)/thread outcome, and
restored thread count. When no supplied library is available, record `NOT RUN` with the exact
reason; native-free proof remains mandatory. A later [CPU](../../../../glossary.md#term-cpu) task owns multi-[target](../../../../glossary.md#term-target) real-native
[qualification](../../../../glossary.md#term-qualification). This provider task must nevertheless record primary ABI evidence for all six
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
private surface; and confirm provider 0004 remains the only detailed unfinished provider task and
CPU 0010D1 remains a blocked/deferred optional side branch.

Repository-wide validation is deferred to a future resumed CPU 0010D1 capability checkpoint or [CI](../../../../glossary.md#term-continuous-integration-ci).
This task changes one [JDK](../../../../glossary.md#term-java-development-kit-jdk)-only [leaf](../../../../glossary.md#term-leaf) without a [module](../../../../glossary.md#term-module) edge or architecture rule.

## Dependencies

- Complete provider tasks 0001–0003 for caller-directed lifetime, mandatory bindings,
  FLOAT32/FLOAT64 GEMM, thread control, and checkpoint infrastructure.<span data-glossary-navigation="true"> ([lifetime glossary](../../../../glossary.md#term-lifetime); [bindings glossary](../../../../glossary.md#term-binding); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- [JDK](../../../../glossary.md#term-java-development-kit-jdk) 26 [FFM](../../../../glossary.md#term-ffm) [API](../../../../glossary.md#term-application-programming-interface-api) and the repository Java 26 [toolchain](../../../../glossary.md#term-toolchain).
- Pinned OpenBLAS 0.3.31-or-later primary source for the exact optional extension. A header alone
  or unpinned development branch is insufficient.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>

No installed [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) is required for native-free implementation tests. Public exposure remains
blocked inside this task until the [ABI](../../../../glossary.md#term-application-binary-interface-abi) and semantic proof gates pass. Complete [CPU](../../../../glossary.md#term-cpu) 0010D is
downstream compatibility/[fingerprint](../../../../glossary.md#term-fingerprint) context only, not a provider [dependency](../../../../glossary.md#term-dependency); CPU code remains
review-only here, and dependent CPU 0010D1 cannot start until this provider task completes.

## Follow-up tasks

- CPU 0010D1 remains a master-plan-only blocked/deferred optional side branch. It consumes only a provider capability that is
  both present and CPU-qualified, extends qualification/fingerprints with versioned direct-BF16
  evidence, and adds the exact BFLOAT16 MATMUL route while retaining portable fallback.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [qualification glossary](../../../../glossary.md#term-qualification); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
- CPU 0010E proceeds independently for the completed FLOAT32/FLOAT64 route. A later resumed
  BFLOAT16 capability must extend compatibility explicitly rather than being inferred from that
  FLOAT32/FLOAT64 schema.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [schema glossary](../../../../glossary.md#term-schema))</span>
- No follow-up may replace failed direct-[BF16](../../../../glossary.md#floating-point-precision-and-rounding) proof with SBGEMM or conversion staging.

## Architecture impact

Expected impact: None.

This task implements optional low-level [GEMM](../../../../glossary.md#general-matrix-multiplication--gemm) [binding](../../../../glossary.md#term-binding) and invocation already allowed for the
[OpenBLAS](../../../../glossary.md#term-openblas) provider. It adds no [dependency direction](../../../../glossary.md#java-platform-and-build-tools) or [owner](../../../../glossary.md#term-owner). If implementation requires [CPU](../../../../glossary.md#term-cpu)
policy, shared [lifecycle](../../../../glossary.md#lifecycle) state, native packaging, or another architecture responsibility, stop
and report the conflict rather than editing the [architecture contract](../../../../glossary.md#architecture-contract).

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
  extension first advertised in OpenBLAS 0.3.31 and available only in builds that supply it.<span data-glossary-navigation="true"> ([CBLAS glossary](../../../../glossary.md#term-cblas); [OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>
- The upstream declaration and extension documentation are sufficiently concrete to justify a
  bounded proof task, but current evidence is not sufficient to expose execution before the task
  pins implementation-source accumulation/narrowing behavior and supported-target ABI carriers.<span data-glossary-navigation="true"> ([narrowing glossary](../../../../glossary.md#term-narrowing); [target glossary](../../../../glossary.md#term-target); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
- The loaded symbol, not an operating-system, architecture, version, build flag, installed header,
  or related SBGEMM symbol, decides structural capability presence.<span data-glossary-navigation="true"> ([symbol glossary](../../../../glossary.md#term-symbol))</span>
- One optional typed capability object is smaller and safer than an availability boolean plus a
  callable method on every library. It keeps ABI identity beside the operation and cannot be
  mistaken for CPU numerical qualification.<span data-glossary-navigation="true"> ([boolean glossary](../../../../glossary.md#data-type--datatype); [identity glossary](../../../../glossary.md#term-identity-meanings); [operation glossary](../../../../glossary.md#operation); [CPU glossary](../../../../glossary.md#term-cpu); [qualification glossary](../../../../glossary.md#term-qualification))</span>
- Direct means that Synaptik passes BFLOAT16 A, B, and C to one BFLOAT16-output ABI. Internal
  OpenBLAS implementation details may use wider arithmetic, as the Model requires, but Synaptik
  introduces no conversion or FLOAT32 staging resource.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [Model glossary](../../../../glossary.md#term-model); [FLOAT32 glossary](../../../../glossary.md#term-float32))</span>
- The once-narrowing fixture complements source review because reassociation makes no finite test
  a universal accumulation proof by itself.<span data-glossary-navigation="true"> ([fixture glossary](../../../../glossary.md#term-fixture); [reassociation glossary](../../../../glossary.md#term-reassociation); [finite glossary](../../../../glossary.md#term-finite))</span>

## Known limitations

- The local [OpenBLAS](../../../../glossary.md#term-openblas) 0.3.34 [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) cannot exercise direct [BFLOAT16](../../../../glossary.md#term-bfloat16) output because it lacks the
  `cblas_bgemm` export despite a declaring installed header. That is a valid absent-capability
  result.
- `cblas_bgemm` is an OpenBLAS extension, not a portable Netlib BLAS/CBLAS contract. Each binary
  requires exact optional binding and later CPU numerical qualification.<span data-glossary-navigation="true"> ([BLAS glossary](../../../../glossary.md#term-blas); [CBLAS glossary](../../../../glossary.md#term-cblas); [binding glossary](../../../../glossary.md#term-binding); [CPU glossary](../../../../glossary.md#term-cpu); [qualification glossary](../../../../glossary.md#term-qualification))</span>
- The provider checkpoint is bounded and does not certify every shape, value, thread count,
  numerical order, OpenBLAS build, or target.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [shape glossary](../../../../glossary.md#shape); [target glossary](../../../../glossary.md#term-target))</span>
- ILP64 and any supported-host ABI whose enum or 16-bit scalar calling convention cannot be
  proved remain unsupported for this capability.<span data-glossary-navigation="true"> ([ILP64 glossary](../../../../glossary.md#term-ilp64); [host glossary](../../../../glossary.md#term-host); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [scalar glossary](../../../../glossary.md#term-scalar); [calling convention glossary](../../../../glossary.md#term-calling-convention))</span>
- The provider operation alone does not make CPU BFLOAT16 MATMUL eligible. CPU 0010D1 remains
  responsible for qualification, Model compatibility, route resources, fallback, and execution.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [Model glossary](../../../../glossary.md#term-model))</span>
- This blocked task does not change supported Model or portable CPU BFLOAT16 behavior and is not a
  prerequisite for existing FLOAT32/FLOAT64 OpenBLAS candidates, preparation, or tuning.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- Clean implementation context `01a090a7-0a5a-7b82-aced-9dc7368cdbdd` pinned upstream [OpenBLAS](../../../../glossary.md#term-openblas) tag `v0.3.31` at [commit](../../../../glossary.md#term-commit)
  `76f1be470c9b9f80dc6e27407e13b975df436489` and inspected its release, header, configuration,
  build, export, wrapper, driver, fallback-kernel, and optimized AArch64-kernel sources before
  changing production. The seven proof gates resolved as follows:<span data-glossary-navigation="true"> ([release glossary](../../../../glossary.md#term-release); [AArch64 glossary](../../../../glossary.md#term-aarch64))</span>

  | Gate | Result | Pinned evidence |
  |---|---|---|
  | Extension [identity](../../../../glossary.md#term-identity-meanings) | **Not proved** | `Changelog.txt:1-10` calls BGEMM/BGEMV new bfloat16 extensions, `CMakeLists.txt:152-161` and `docs/build_system.md:101` make `BUILD_BFLOAT16` optional, and `interface/Makefile:327-333,2049-2052` plus `interface/CMakeLists.txt:151-165` build `cblas_bgemm` only for that option. However, `docs/extensions.md:18-29` documents only `cblas_sbgemm`, not `cblas_bgemm`, and both official shared-library [symbol](../../../../glossary.md#term-symbol) generators omit `cblas_bgemm` from `bfcblasobjs` at `exports/gensymbol:131` and `exports/gensymbol.pl:128` even though they include the Fortran-style `bgemm`. The tag therefore does not establish the required exported C [ABI](../../../../glossary.md#term-application-binary-interface-abi) family. |
  | C representation | **Proved for the source [shape](../../../../glossary.md#shape)** | `openblas_config_template.h:35-40` and `common.h:264-268` define `bfloat16` as `uint16_t`; `cblas.h:481-482` passes both [scalars](../../../../glossary.md#term-scalar) by value and addresses A, B, and C as `bfloat16 *`. [JDK](../../../../glossary.md#term-java-development-kit-jdk) 26 `ValueLayout.JAVA_SHORT` is a two-byte, two-byte-aligned native-order [layout](../../../../glossary.md#layout), and unsigned C integers use the corresponding signed [carrier](../../../../glossary.md#term-storage-carrier) without changing [raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity). |
  | Integer and enum carriers | **Proved for the selected ordinary build shape** | `openblas_config_template.h:49-53` makes ordinary `blasint` C `int` and selects a wider carrier only under `OPENBLAS_USE64BITINT`; `cblas.h:73-79` defines the [CBLAS](../../../../glossary.md#term-cblas) enums and their fixed values. No `-fshort-enums` use exists in the tag. Apple Clang 21 cross-[target](../../../../glossary.md#term-target) LLVM [lowering](../../../../glossary.md#lowering) of the exact declaration produced `i32` for all three enums and six [dimension](../../../../glossary.md#dimension)/leading-dimension values, `i16` for alpha/beta, pointers for A/B/C, and `void` return for `arm64-apple-macos15`, `x86_64-apple-macos15`, `aarch64-linux-gnu`, `x86_64-linux-gnu`, `aarch64-pc-windows-msvc`, and `x86_64-pc-windows-msvc`. [ILP64](../../../../glossary.md#term-ilp64) remains rejected. |
  | Exact [descriptor](../../../../glossary.md#term-descriptor) | **Proved for the selected ordinary build shape** | `cblas.h:481-482` and `interface/gemm.c:371-378` give exactly fourteen ordered [arguments](../../../../glossary.md#term-argument): three enums, M/N/K, [BFLOAT16](../../../../glossary.md#term-bfloat16) alpha, A, lda, B, ldb, BFLOAT16 beta, C, ldc, with `void` return. The official JDK 26 `Linker` contract maps C `short`, C `int`, and pointers to [canonical](../../../../glossary.md#term-canonical-form) value/address layouts and `FunctionDescriptor.ofVoid` models argument layouts with no return layout. The resulting descriptor would be `JAVA_INT` six times, `JAVA_SHORT`, `ADDRESS`, `JAVA_INT`, `ADDRESS`, `JAVA_INT`, `JAVA_SHORT`, `ADDRESS`, `JAVA_INT`. |
  | Direct result | **Proved for the source shape** | `cblas.h:481-482`, the `BGEMM` [specialization](../../../../glossary.md#term-specialization) in `common.h:317-323`, and `interface/gemm.c:371-378` use BFLOAT16 A, B, scalar, and C carriers directly; there is no [FLOAT32](../../../../glossary.md#term-float32) C [parameter](../../../../glossary.md#term-parameter) in this entry point. |
  | [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) compatibility | **Failed** | `param.h:85-87` sets `BGEMM_DEFAULT_Q` to 256. `driver/level3/level3.c:305-322` partitions the [contraction dimension](../../../../glossary.md#contraction-dimension) into repeated `GEMM_Q`-bounded `ls` blocks, and `driver/level3/level3.c:381-387` invokes the output [kernel](../../../../glossary.md#kernel) for every block. The fallback BGEMM kernel accumulates each block in FLOAT32 at `kernel/generic/gemmkernel_2x2.c:42-46,56-109`, then reads BFLOAT16 C and writes `TO_OUTPUT(...)` at lines 110-117. `kernel/generic/conversion_macros.h:34-51` defines that BGEMM output conversion as FLOAT32-to-BFLOAT16. The optimized NEOVERSEV1 BGEMM kernel likewise converts and stores C in each `UPDATE_C` at `kernel/arm64/bgemm_kernel_2vlx4_neoversev1_impl.c:34-69,203-206`. Consequently a sufficiently large [contraction](../../../../glossary.md#contraction-dimension) is narrowed to BFLOAT16 between K blocks, then widened and accumulated again. This is not FLOAT32 accumulation over the complete contraction followed by one final BFLOAT16 conversion, so it cannot satisfy the task's [Model](../../../../glossary.md#term-model) gate merely by allowing [reassociation](../../../../glossary.md#term-reassociation) or [fused multiply-add](../../../../glossary.md#term-fused-multiply-add-fma). |
  | Executable proof | **Absent, as required after the failed semantic gate** | `/opt/homebrew/Cellar/openblas/0.3.34/lib/libopenblasp-r0.3.34.dylib` is a real [arm64](../../../../glossary.md#native-packaging-and-platform-vocabulary) [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) whose `nm -gU` output contains the four mandatory symbols and `cblas_sbgemm` but no `cblas_bgemm`. No direct BFLOAT16 invocation was attempted. The unchanged [checkpoint](../../../../glossary.md#term-checkpoint) passed its mandatory [SGEMM](../../../../glossary.md#general-matrix-multiplication--gemm), [DGEMM](../../../../glossary.md#general-matrix-multiplication--gemm), shared-thread observation, and restoration checks and restored thread count 16. |

- Official JDK 26 contracts used for the [FFM](../../../../glossary.md#term-ffm) review were
  [`Linker`](https://docs.oracle.com/en/java/javase/26/docs/api/java.base/java/lang/foreign/Linker.html),
  [`FunctionDescriptor`](https://docs.oracle.com/en/java/javase/26/docs/api/java.base/java/lang/foreign/FunctionDescriptor.html),
  [`MemorySegment`](https://docs.oracle.com/en/java/javase/26/docs/api/java.base/java/lang/foreign/MemorySegment.html),
  and [`ValueLayout`](https://docs.oracle.com/en/java/javase/26/docs/api/java.base/java/lang/foreign/ValueLayout.html).
  The downloaded pages identify themselves as generated JDK 26 API documentation. <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>`Linker`
  requires a platform-dependent C signature, guarantees canonical layouts for C <span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature))</span>`short`, `int`,
  and `void *`, maps unsigned integers through the corresponding signed carrier, and derives the
  exact downcall method type from the descriptor carriers. <span data-glossary-navigation="true"> ([downcall glossary](../../../../glossary.md#term-downcall))</span>`FunctionDescriptor` defines ordered
  argument layouts and zero or one return layout. [`MemorySegment`](../../../../glossary.md#term-memorysegment) supplies the native, address,
  size, scope, accessibility, read-only, slice, and backing-overlap contracts required by the
  planned validator. <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [slice glossary](../../../../glossary.md#slice))</span>`ValueLayout.JAVA_SHORT` has Java-short size, two-byte alignment, and native
  byte order.<span data-glossary-navigation="true"> ([byte alignment glossary](../../../../glossary.md#term-byte-alignment); [byte order glossary](../../../../glossary.md#text-encoding-and-file-boundaries))</span>
- The exact final ordinary command
  `./gradlew :backends:openblas-provider:test :backends:openblas-provider:testClasses` passed with
  `BUILD SUCCESSFUL`; all three actionable tasks were up-to-date. Existing XML results report five
  suites and 50 tests with zero failures, errors, or skips. Executable Java and tests were not
  changed before or after this run.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- The unchanged real-native command used Java `26.0.1` with
  `--enable-native-access=ALL-UNNAMED --illegal-native-access=deny` against the exact 0.3.34 path
  above. It exited zero with `OpenBLAS native checkpoint passed; restored thread count 16`.
- `javap -public` confirmed that production still exposes only existing library loading,
  SGEMM/DGEMM, thread control, lifecycle, and load-failure API. Source inspection found no<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
  `cblas_bgemm`, `cblas_sbgemm`, `OpenBlasBFloat16Gemm`, or `JAVA_SHORT` in provider production or
  tests. `git diff --check` passed before this evidence record was written.
- Clean documentation context `/root/openblas_0004_doc_review` independently re-read the
  architecture, planning, documentation-profile, prerequisite-provider, CPU-ordering, and Model
  MATMUL contracts and inspected the exact implementation diff. It verified tag <span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [CPU glossary](../../../../glossary.md#term-cpu))</span>`v0.3.31` at
  commit `76f1be470c9b9f80dc6e27407e13b975df436489`, the omitted `cblas_bgemm` shared-library export,
  the `GEMM_Q`-bounded K-panel loop, and per-panel BFLOAT16 conversion in both the fallback and
  selected optimized kernel source. It also reproduced the six-target Clang descriptor lowering
  and confirmed that the local arm64 0.3.34 binary exports the mandatory four symbols and
  `cblas_sbgemm`, but not `cblas_bgemm`.
- At that checkpoint, the documentation context changed only this task, the provider master
  plan, and the roadmap. Its four-file Markdown validation passed local links, anchors, unique
  headings, balanced fences, final newlines, and trailing-whitespace checks. Exact-path,
  stale-[`Ready`](../../../../glossary.md#term-ready), status, [dependency](../../../../glossary.md#term-dependency)/order, task-file-absence, unchanged-`ARCHITECTURE.md`, and
  `git diff --check` checks passed. It did not rerun Java tests or Javadoc because no executable
  Java, tests, public API, behavior, or Javadoc changed after the reusable implementation evidence.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

## Implementation notes

- The proof phase stopped before every production/test/Javadoc edit because the pinned first
  release does not prove a supported exported <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [release glossary](../../../../glossary.md#term-release))</span>`cblas_bgemm` C ABI family and its implementation
  positively contradicts the required one-final-narrowing Model compatibility gate.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [narrowing glossary](../../../../glossary.md#term-narrowing); [Model glossary](../../../../glossary.md#term-model))</span>
- No optional lookup, descriptor, method handle, public capability, invocation helper, checkpoint
  extension, conversion, staging, <span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-lookup))</span><span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [method handle glossary](../../../../glossary.md#term-method-handle); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>`cblas_sbgemm` route, platform/version inference, or CPU change
  was added. The existing mandatory four-symbol load and all current provider behavior remain
  byte-for-byte unchanged.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [CPU glossary](../../../../glossary.md#term-cpu); [symbol glossary](../../../../glossary.md#term-symbol))</span>
- A later retry requires pinned upstream OpenBLAS source and exported-binary evidence for a
  versioned direct-BGEMM ABI whose complete contraction remains FLOAT32 until one final BFLOAT16
  result conversion. A small numerical fixture cannot replace that source proof.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [contraction glossary](../../../../glossary.md#contraction-dimension); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [fixture glossary](../../../../glossary.md#term-fixture))</span>

## Completion summary

- Completed changes: recorded and independently verified the failed proof result, marked provider
  0004 `Blocked`, and synchronized the provider, CPU, and global frontier without exposing or
  invoking an executable capability.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- Files changed or created: exactly four planning paths—this task, the provider master plan, the
  CPU master plan, and the roadmap.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and [validation](../../../../glossary.md#term-validation): reused implementation context
  `01a090a7-0a5a-7b82-aced-9dc7368cdbdd`'s passing five-suite/50-test provider result and local
  0.3.34 mandatory native checkpoint that restored thread count 16. Documentation context<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
  `/root/openblas_0004_doc_review` independently reproduced the six-target Clang ABI lowering,
  inspected the pinned sources and local exports, and passed four-file Markdown, exact-path,
  stale-status/frontier, dependency/order, task-file-absence, unchanged-architecture, and
  whitespace validation.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [lowering glossary](../../../../glossary.md#lowering); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Documentation-agent review: clean context `/root/openblas_0004_doc_review` completed the
  targeted planning-document review and status/frontier synchronization without changing or
  rerunning executable Java, tests, Javadoc, or generated documentation.<span data-glossary-navigation="true"> ([synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- Documentation impact: the task evidence must remain as the durable failed-gate record. Provider
  Javadocs, package documentation, CPU guide, and glossary remain unchanged because no capability,
  API, behavior, configuration, workflow, example, or reusable terminology was added. Tensor,
  Compile, and Training APIs; Model capabilities; architecture documents/tests; backend
  conformance and integration tests; Gradle; native packaging; and other modules remain accurate
  for the same reason.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [Model glossary](../../../../glossary.md#term-model); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: all provider Java remains unchanged, so its existing Javadoc remains accurate.
- Glossary impact: none; no new usable capability or project term was introduced.
- Unresolved issues: upstream v0.3.31 does not export `cblas_bgemm` through its official C symbol
  lists and its BGEMM driver/kernel path performs intermediate BFLOAT16 narrowing across K blocks.<span data-glossary-navigation="true"> ([symbol glossary](../../../../glossary.md#term-symbol); [kernel glossary](../../../../glossary.md#kernel); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16))</span>
- Follow-up required for the optional side branch: obtain and pin an upstream direct-BGEMM ABI/implementation that both exports
  the C symbol and proves complete FLOAT32 contraction accumulation with one final BFLOAT16
  conversion. Until then CPU 0010D1 is non-executable. The approved FLOAT32/FLOAT64 CPU 0010E ->
  Prepare 0004 -> tools/tuning 0001 mainline proceeds independently. No semantic relaxation or
  substitute implementation is authorized here.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [contraction glossary](../../../../glossary.md#contraction-dimension); [FLOAT64 glossary](../../../../glossary.md#term-float64); [Prepare glossary](../../../../glossary.md#prepare))</span>

Status: Incomplete
Follow-up required: resolve the failed exported-[ABI](../../../../glossary.md#term-application-binary-interface-abi) and one-final-[narrowing](../../../../glossary.md#term-narrowing) proof gates before any production capability is added.
