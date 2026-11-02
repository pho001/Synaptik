# Task 0080: Shared generation-time kernel method emission

## Status

Complete — MAIN integrated the two exact reviewed source owners after the CPU/Javadoc gate,
actual prefixed numeric dispatches, ordinary actual-VJP replay and retained-byte comparison.
Final independent source/evidence/documentation APPROVE0; no wider Scan/control capability follows.

## Goal and authority

Class C. Factor the existing CpuClassFileKernelGenerator method-emission dispatch so the same
ordinary emitters can populate deterministically named static helpers in a supplied ClassBuilder.
Use the existing backend generation/carrier/low-admission owners and ADR0061 direction. No new
execution path, numerical emitter, Runtime owner, scheduler, cache format or production capability.

## Generation contract

In the existing generator, expose one package-private generation-time entry:

```java
static List<GeneratedMethod> emitMethods(ClassBuilder builder, ClassDesc owner, String prefix,
        CpuKernelSpecialization specialization, CpuKernelIr ir);
record GeneratedMethod(String name, MethodTypeDesc type, int accessFlags) {}
```

- generateClassBytes remains the production class owner, calls this same dispatch with an empty
  prefix, then performs existing Class-File verification. Preserve class version/flags, method
  emission order, names/descriptors/access, instruction/local/constant-pool order and exact old bytes.
- A nonempty deterministic caller prefix namespaces the entry and every subsidiary method and
  direct call. CpuLossEmitter's entry, lossContiguousInt and lossGenericAffine must use it together.
  Preserve the exact existing ordinary carrier/scratch/geometry/start/end descriptor and local slots.
- Keep one existing family dispatch and every current specialized emitter, including common
  CpuCarrierEmitter layout setup before loads. No copied opcode/math switch, method-handle unit
  interpreter, per-operation runtime lookup or extra generated wrapper call. The returned immutable
  inventory describes methods actually emitted, not guessed method counts or a second registry.
- Preserve existing multi-phase ordinary row helpers and their names/descriptors for the empty
  prefix. Namespace their actual emitted inventory when used as helpers; do not silently narrow
  supported ordinary generation to one-method cases. Existing scanRowRange/Scan paths retain exact
  empty-prefix behavior; reject a prefixed nested Scan rather than introduce nested composition.
- Existing numerical low-type admission/validation applies independently at this entry before
  emission; no F32 envelope may hide locked low arithmetic. Retain raw/exact/cast/INT/BOOL behavior.
- The whole-class verifier may consume the actual immutable inventory where necessary without
  weakening descriptor/access/code-budget checks. No global schema/family/key bump, new profile or
  empty serialized field. Unchanged retained recipes keep both identity streams and full class bytes.
- Do not add an unused activity argument, fake control implementation or placeholder. Controlled
  helper ABI/activity/contributor hooks belong to the separately qualified full composition task.
  Current production generation must genuinely use this completed factoring.

## Scope and integration

- Exact initial source owners: CpuClassFileKernelGenerator.java and CpuLossEmitter.java under
  modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit.
  Existing direct callers/focused tests only if signature migration requires them; remove obsolete
  overloads instead of shims. Source lookup confirms one current loss-emitter call in the generator.
- Depends on Complete CPU0078/0079 and current retained generation contracts, verified by MAIN.
  MAIN records the exact accepted dirty-source manifest and clean context before source launch.
- Conflicts with those generation owners; disjoint from Model0048 source/Compiler row semantics,
  Compiler0017B ordinary rule construction and Engine0024 build files. MAIN owns shared docs/join.
- Parallel group RELEASE-SCAN-COORDINATE-METHODS. Workers run no commands/builds/tests/javac/lint/
  formatters/native loads. Use absolute paths in the assigned clean context; report exact source
  preimage/final SHA and actual inventory/API, no verified-runtime claim.
- Order: independent contract/Ready -> exact clean context -> full factoring/Javadoc -> independent
  source review -> MAIN CPU/Javadoc gate and actual hidden-class execution -> final evidence/docs.
- MAIN compares retained 14-family/28-kernel/85-file capture unchanged and ordinary scalar-oracle
  forward/actual backward consumers. A throwaway same-class smoke emits multiple prefixed ordinary
  units including loss helpers, verifies their real descriptors and executes numeric results.
  This is actual behavior, not a permanent wiring/member/count test. Preserve first failures.

No all-PURE admission or production-completion claim follows. The later composed Scan still must
reuse these methods through fixed emitted calls with complete activity, validators, typed scratch,
geometry resets, native/heap ownership and selective identity qualification.

## Result

The existing production generator genuinely calls one package-private `emitMethods` dispatch with
the empty prefix; a nested immutable `GeneratedMethod` inventory records actual declarations.
Namespaced ordinary entries, selected ordinary row phases and all three loss methods use that
same dispatch. Both direct loss calls use the caller's declaring owner/prefix. Prefixed general
Scan rejects before declarations; standalone Scan keeps its existing range-helper order.
The complete existing low-type and specialization/IR checks run before emission at this seam.
No numerical body, carrier ABI, local layout, wrapper invocation, identity/schema or runtime
lookup/interpreter was added or changed.

Independent source/Javadoc review approved both exact source owners with no P1/P2 findings.
The actual CPU gate passed 1,474 tests with 60 existing opt-in skips; Javadocs passed with 95
recorded existing warnings. The unchanged capture generated fourteen source families and
twenty-eight forward/reverse kernels: all 85 identity/compatibility/class/index files match the
retained preimage byte-for-byte.

A throwaway consumer emitted and verified one real class containing F32/F64 ordinary TANH plus
separately prefixed indexed cross entropy and MSE. All eight methods coexist. The first contiguous
MSE fixture did not exercise generic-affine dispatch despite the initial receipt's claim; review
identified and retained that gap. An admitted stride-two MSE prediction with physical gap
sentinels now actually selects the generic target and independently yields mean loss 3.25.
Together with the contiguous indexed-loss case, both namespaced dispatch targets are exercised.
The unchanged 84 ordinary forward/actual-Compiler-VJP cases also passed. Their first launch
failed because the smoke classpath omitted the already compiled Tuning runtime classes; that
harness failure is retained, not suppressed or repaired by removing the production dependency.
The complete source-matched compiled product closure passed without changing the oracle.

Exact source and evidence: `cpu0080-main-integration.json`, `cpu0080-affected-gate-evidence.json`,
`cpu0080-retained-artifact-evidence.json` and `cpu0080-prefixed-noncontiguous-evidence.json` in the
production-run directory; the latter explicitly corrects the original receipt. No timing, wider
PURE Scan, controlled ABI, new cache format, backend grant or complete release is claimed.
