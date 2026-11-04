# Task 0080: Shared generation-time [kernel](../../../../glossary.md#kernel) method emission

## Reader context

`CpuClassFileKernelGenerator.emitMethods` is called while **building a Java class**, not while
evaluating each tensor element. Its internal caller supplies a `ClassBuilder`, the class's
declaring identity, a deterministic name prefix and checked specialization/IR facts. The method
adds static methods to that builder and returns descriptions of the declarations it actually
emitted. The caller still owns completing and verifying the class; the returned inventory is
not tensor storage, a Runtime executor or a list interpreted for every step.

The prefix prevents helpers for different ordinary units from colliding in one generated
class. Direct calls in those methods must name the same declaring class and prefix, including
loss helpers; merely renaming the entry would leave a broken internal call boundary.
[Current Scan composition](../../../../backend-guide/cpu-backend.md#current-general-scan-composition)
explains how the later composition uses ordinary helpers under one time loop. This task's
historical entry predates controlled composition; its recorded nested-Scan rejection, retained
bytes and consumer failures remain unchanged rather than being rewritten as current API promises.

## Status

Complete — MAIN integrated the two exact reviewed source [owners](../../../../glossary.md#term-owner) after the [CPU](../../../../glossary.md#term-cpu)/[Javadoc](../../../../glossary.md#term-javadoc) gate,
actual prefixed numeric dispatches, ordinary actual-[VJP](../../../../glossary.md#term-vector-jacobian-product-vjp) replay and retained-byte comparison.
Final independent source/evidence/documentation APPROVE0; no wider [Scan](../../../../glossary.md#term-cumulative-scan)/control capability follows.

## Goal and authority

Class C. Factor the existing CpuClassFileKernelGenerator method-emission [dispatch](../../../../glossary.md#term-dispatch) so the same
ordinary emitters can populate deterministically named static helpers in a supplied ClassBuilder.
Use the existing [backend](../../../../glossary.md#backend) generation/[carrier](../../../../glossary.md#term-storage-carrier)/low-[admission](../../../../glossary.md#term-admission) [owners](../../../../glossary.md#term-owner) and ADR0061 direction. No new
execution path, numerical emitter, [Runtime](../../../../glossary.md#term-runtime) [owner](../../../../glossary.md#term-owner), scheduler, [cache](../../../../glossary.md#term-cache) format or production capability.

## Generation contract

In the existing generator, expose one package-private generation-time entry:

```java
static List<GeneratedMethod> emitMethods(ClassBuilder builder, ClassDesc owner, String prefix,
        CpuKernelSpecialization specialization, CpuKernelIr ir);
record GeneratedMethod(String name, MethodTypeDesc type, int accessFlags) {}
```

- generateClassBytes remains the production class owner, calls this same dispatch with an empty
  prefix, then performs existing Class-File verification. Preserve class version/flags, method
  emission order, names/descriptors/access, instruction/local/constant-pool order and exact old bytes.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [dispatch glossary](../../../../glossary.md#term-dispatch); [descriptors glossary](../../../../glossary.md#term-descriptor); [constant glossary](../../../../glossary.md#term-constant))</span>
- A nonempty deterministic caller prefix namespaces the entry and every subsidiary method and
  direct call. CpuLossEmitter's entry, lossContiguousInt and lossGenericAffine must use it together.
  Preserve the exact existing ordinary carrier/scratch/geometry/start/end descriptor and local slots.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Keep one existing family dispatch and every current specialized emitter, including common
  CpuCarrierEmitter layout setup before loads. No copied opcode/math switch, method-handle unit
  interpreter, per-operation runtime lookup or extra generated wrapper call. The returned immutable
  inventory describes methods actually emitted, not guessed method counts or a second registry.<span data-glossary-navigation="true"> ([lookup glossary](../../../../glossary.md#term-method-handle-lookup))</span><span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [opcode glossary](../../../../glossary.md#term-opcode); [operation glossary](../../../../glossary.md#operation); [runtime glossary](../../../../glossary.md#term-runtime); [immutable glossary](../../../../glossary.md#term-immutable); [registry glossary](../../../../glossary.md#term-registry))</span>
- Preserve existing multi-phase ordinary row helpers and their names/descriptors for the empty
  prefix. Namespace their actual emitted inventory when used as helpers; do not silently narrow
  supported ordinary generation to one-method cases. Existing scanRowRange/Scan paths retain exact
  empty-prefix behavior; reject a prefixed nested Scan rather than introduce nested composition.<span data-glossary-navigation="true"> ([Scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- Existing numerical low-type admission/validation applies independently at this entry before
  emission; no F32 envelope may hide locked low arithmetic. Retain raw/exact/cast/INT/BOOL behavior.<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission); [validation glossary](../../../../glossary.md#term-validation); [F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- The whole-class verifier may consume the actual immutable inventory where necessary without
  weakening descriptor/access/code-budget checks. No global schema/family/key bump, new profile or
  empty serialized field. Unchanged retained recipes keep both identity streams and full class bytes.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Do not add an unused activity argument, fake control implementation or placeholder. Controlled
  helper ABI/activity/contributor hooks belong to the separately qualified full composition task.
  Current production generation must genuinely use this completed factoring.<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>

## [Scope](../../../../glossary.md#term-scope) and integration

- Exact initial source owners: CpuClassFileKernelGenerator.java and CpuLossEmitter.java under
  modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit.
  Existing direct callers/focused tests only if signature migration requires them; remove obsolete
  overloads instead of shims. Source lookup confirms one current loss-emitter call in the generator.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module); [backend glossary](../../../../glossary.md#backend); [signature glossary](../../../../glossary.md#term-signature); [lookup glossary](../../../../glossary.md#term-lookup); [loss glossary](../../../../glossary.md#loss))</span>
- Depends on Complete CPU0078/0079 and current retained generation contracts, verified by MAIN.
  MAIN records the exact accepted dirty-source manifest and clean context before source launch.<span data-glossary-navigation="true"> ([manifest glossary](../../../../glossary.md#term-manifest))</span>
- Conflicts with those generation owners; disjoint from Model0048 source/Compiler row semantics,
  Compiler0017B ordinary rule construction and Engine0024 build files. MAIN owns shared docs/join.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>
- Parallel group RELEASE-SCAN-COORDINATE-METHODS. Workers run no commands/builds/tests/javac/lint/
  formatters/native loads. Use absolute paths in the assigned clean context; report exact source
  preimage/final SHA and actual inventory/API, no verified-runtime claim.<span data-glossary-navigation="true"> ([RELEASE glossary](../../../../glossary.md#term-release); [COORDINATE glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [run glossary](../../../../glossary.md#term-run); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Order: independent contract/Ready -> exact clean context -> full factoring/Javadoc -> independent
  source review -> MAIN CPU/Javadoc gate and actual hidden-class execution -> final evidence/docs.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready); [Javadoc glossary](../../../../glossary.md#term-javadoc); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- MAIN compares retained 14-family/28-kernel/85-file capture unchanged and ordinary scalar-oracle
  forward/actual backward consumers. A throwaway same-class smoke emits multiple prefixed ordinary
  units including loss helpers, verifies their real descriptors and executes numeric results.
  This is actual behavior, not a permanent wiring/member/count test. Preserve first failures.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [capture glossary](../../../../glossary.md#term-capture-meanings); [scalar glossary](../../../../glossary.md#term-scalar); [oracle glossary](../../../../glossary.md#term-test-oracle); [smoke glossary](../../../../glossary.md#term-smoke-test); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>

No all-[PURE](../../../../glossary.md#term-pure) [admission](../../../../glossary.md#term-admission) or production-completion claim follows. The later composed [Scan](../../../../glossary.md#term-cumulative-scan) still must
reuse these methods through fixed emitted calls with complete activity, validators, typed scratch,
geometry resets, native/[heap](../../../../glossary.md#term-heap) [ownership](../../../../glossary.md#term-ownership-meanings) and selective [identity](../../../../glossary.md#term-identity-meanings) [qualification](../../../../glossary.md#term-qualification).

## Result

The existing production generator genuinely calls one package-private `emitMethods` [dispatch](../../../../glossary.md#term-dispatch) with
the empty prefix; a nested [immutable](../../../../glossary.md#term-immutable) `GeneratedMethod` inventory records actual declarations.
Namespaced ordinary entries, selected ordinary row phases and all three [loss](../../../../glossary.md#loss) methods use that
same dispatch. Both direct loss calls use the caller's declaring [owner](../../../../glossary.md#term-owner)/prefix. Prefixed [general
Scan](../../../../glossary.md#general-differentiable-scan) rejects before declarations; standalone [Scan](../../../../glossary.md#term-cumulative-scan) keeps its existing range-helper order.
The complete existing low-type and [specialization](../../../../glossary.md#term-specialization)/[IR](../../../../glossary.md#term-intermediate-representation-ir) checks [run](../../../../glossary.md#term-run) before emission at this seam.
No numerical body, [carrier](../../../../glossary.md#term-storage-carrier) [ABI](../../../../glossary.md#term-application-binary-interface-abi), local [layout](../../../../glossary.md#layout), wrapper invocation, [identity](../../../../glossary.md#term-identity-meanings)/[schema](../../../../glossary.md#term-schema) or [runtime](../../../../glossary.md#term-runtime)
[lookup](../../../../glossary.md#term-lookup)/interpreter was added or changed.

Independent source/[Javadoc](../../../../glossary.md#term-javadoc) review approved both exact source owners with no P1/P2 findings.
The actual [CPU](../../../../glossary.md#term-cpu) gate passed 1,474 tests with 60 existing opt-in skips; Javadocs passed with 95
recorded existing warnings. The unchanged [capture](../../../../glossary.md#term-capture-meanings) generated fourteen source families and
twenty-eight forward/reverse [kernels](../../../../glossary.md#kernel): all 85 identity/compatibility/class/index files match the
retained preimage byte-for-byte.

A throwaway consumer emitted and verified one real class containing [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding) ordinary [TANH](../../../../glossary.md#term-tanh) plus
separately prefixed indexed cross entropy and [MSE](../../../../glossary.md#mean-squared-error--mse). All eight methods coexist. The first [contiguous](../../../../glossary.md#term-contiguous)
MSE [fixture](../../../../glossary.md#term-fixture) did not exercise generic-affine dispatch despite the initial receipt's claim; review
identified and retained that gap. An admitted [stride](../../../../glossary.md#term-stride)-two MSE [prediction](../../../../glossary.md#term-prediction) with physical gap
sentinels now actually selects the generic [target](../../../../glossary.md#term-target) and independently yields [mean](../../../../glossary.md#term-mean) loss 3.25.
Together with the contiguous indexed-loss case, both namespaced dispatch targets are exercised.
The unchanged 84 ordinary forward/actual-[Compiler](../../../../glossary.md#term-compiler)-[VJP](../../../../glossary.md#term-vector-jacobian-product-vjp) cases also passed. Their first launch
failed because the [smoke](../../../../glossary.md#term-smoke-test) [classpath](../../../../glossary.md#term-classpath) omitted the already compiled Tuning runtime classes; that
harness failure is retained, not suppressed or repaired by removing the production [dependency](../../../../glossary.md#term-dependency).
The complete source-matched compiled product closure passed without changing the [oracle](../../../../glossary.md#term-test-oracle).

Exact source and evidence: `cpu0080-main-integration.json`, `cpu0080-affected-gate-evidence.json`,
`cpu0080-retained-artifact-evidence.json` and `cpu0080-prefixed-noncontiguous-evidence.json` in the
production-run directory; the latter explicitly corrects the original receipt. No timing, wider
[PURE](../../../../glossary.md#term-pure) Scan, controlled ABI, new [cache](../../../../glossary.md#term-cache) format, [backend](../../../../glossary.md#backend) grant or complete [release](../../../../glossary.md#term-release) is claimed.
