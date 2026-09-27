# Task 0068: Release-Grade Native Integration Verification

## Status

Verification gates `V0..V13` passed from Task-0067 native/source base
`cb830587c0922e63be7a8e2d47822fe18e2b2bd6` with registered-matrix commit
`6820dd631cbd28f855640a0edc5d125ae6a905a6`. No source defect, environmental failure, relevant
skip, or uncovered runtime surface was found. Independent final code/security/evidence review is
the only remaining gate.

## Change class

Verification-first Class C. This task rebuilds, signs, packages, archives, extracts, and exercises the
current native Metal implementation end to end. It adds no capability, route, schema, ABI, export,
identity, fallback, retry, host repair, timing decision, benchmark, or autotuning behavior. A
demonstrated source defect is in scope for direct remediation followed by the affected and complete
matrix; an environmental blocker must be proven and recorded separately rather than treated as a
source pass.

## Goal

Prove from current native source that the fixed local package and optional Gradle ZIP preserve the
exact ABI-5/schema-15/identity-22 artifact, and that the permission-preservingly extracted packaged
dylib passes every current Metal native/backend, conformance, public Engine, Compiler/autograd,
architecture, Javadoc, repository-test, and full-build surface.

The proof must cover the exact Task-0067 closure: 115 unique operation wires; capability `83/32`;
structural execution `101/14`; MPSGraph `75/35/5`; custom `70/45/0`; production routes
`68 custom / 13 MPSGraph / 1 NEG dual / 1 MATMUL by-domain / 32 blocked`; and Compiler
`38 families / 111 kinds / 133 fingerprints + four deferred signatures = 40 / 115 / 137`.

## Dependencies and integration

- Depends on: Task 0067 Complete and independently approved at
  `cb830587c0922e63be7a8e2d47822fe18e2b2bd6`; Tasks 0045–0046 package/archive contract; current
  ABI-5/schema-15/identity-22 native and Java boundary.
- Conflicts with: every concurrent Metal, Compiler, native, packaging, Gradle, test, planning, or
  shared-document edit during verification.
- Parallel group: None.
- Integration order: matrix registration commit, serial artifact construction, serial verification,
  evidence finalization, independent cumulative code/security/evidence review.
- Shared-document integration owner: Task 0068 finalizer.

## Fixed boundaries

- The native compiler invocation must retain `-Wall -Wextra -Werror`; warnings are failures.
- Signing is local ad hoc with fixed identifier
  `io.github.pho001.synaptik.metal.foundation`, no timestamp, and no Team ID.
- Every runtime test must use the absolute dylib path extracted from the freshly generated Gradle
  ZIP, not the build output or canonical package member.
- No standalone benchmark, timing, route-race, cache-choice, or autotuning command may run, and no
  elapsed value or locally selected winner may become verification evidence or a release/source
  decision. The mandatory complete JVM/public Engine suites retain their ordinary behavioral tests
  of measured tuning, warmup, fallback, and cache paths; those results are asserted only as API
  behavior and are never inspected, compared, or promoted into this verification decision. No
  production fallback, retry, or host repair is permitted. Gradle may compile ordinary
  benchmark-module sources as part of a full build, but no benchmark entry point is invoked.
- A throwaway actual Engine smoke is allowed only after the permanent runtime inventory proves one
  requested surface has no existing behavioral coverage. It must exercise only that uncovered
  surface and be removed before final validation and commit. No smoke is created merely to add a
  test count.
- Ignored local build/package/ZIP outputs are verification artifacts, not source changes. The fresh
  extraction directory is removed after evidence capture.
- Existing production-false rows and Task-0053 proof blockers remain fail-closed; verification is
  not authorization to narrow or special-case them.

## Registered verification matrix

| Gate | Exact command or action | Required evidence |
|---|---|---|
| `V0` | Commit this registered matrix from clean `cb830587`; no production edit. | Task/master/roadmap identify the exact base, serial owner, commands, pass criteria, and blocker policy. |
| `V1` | `./native/metal-macos-arm64/build.sh` | Fresh private-directory compile from current Objective-C source; generated Task-0053 header check; arm64/macOS-26 dylib; native `-Wall -Wextra -Werror` passes. |
| `V2` | `/usr/bin/codesign --force --sign - --timestamp=none --identifier io.github.pho001.synaptik.metal.foundation native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib` | Strict ad-hoc signature with exact identifier and no Team ID. |
| `V3` | `package-local.sh` on the signed dylib, then `verify-package.sh` on `build/package-v1/macos-arm64`. | Exactly three canonical package members; checksums, modes, arm64 Mach-O, macOS 26.0, install name, no rpath, normalized system dependencies, ABI 5, schema 15, and exact thirteen exports. |
| `V4` | `./gradlew :backends:metal:verifyMetalNativePackage :backends:metal:metalNativeLocalZip --rerun-tasks -PsynaptikMetalNativePackage=<absolute canonical package>` | Gradle ingests only the explicit verified package and produces the reproducible local ZIP. |
| `V5` | Extract the ZIP with `/usr/bin/ditto -x -k` into one fresh `mktemp -d` directory; rerun `verify-package.sh` on the extracted `macos-arm64`. | Permission-preserved extracted package independently passes the same verifier and every member is byte-identical to the canonical package. Record dylib/package metadata/ZIP sizes and SHA-256 values plus signature CDHash. |
| `V6` | With `SYNAPTIK_METAL_TEST_LIBRARY=<absolute extracted dylib>`, run `./gradlew :backends:metal:test --rerun-tasks`. | Complete Metal JVM/native suite passes against the packaged dylib with no relevant skip. It includes the 115-row real-provider audit, 83 positive and 32 false representatives, structural/catalog/route partitions, MPSGraph/custom/NEG/MATMUL-by-domain execution, raw ABI/schema/preflight, alias and error-before-mutation, rank-zero, affine, saved-state, multi-output, reuse, sessions, concurrency, and resource lifecycle coverage. |
| `V7` | Run `./gradlew :modules:compiler:test --rerun-tasks`. | Complete Compiler inference, autograd preflight, direct/generated backward, saved-role, logical-layout, ordering, window, convolution/pooling, dropout, loss, deferred Conv3d/recurrent, and inventory closure suites pass. |
| `V8` | Run `./gradlew :testing:backend-conformance:test --rerun-tasks --tests '*Metal*'` and `./gradlew :testing:integration-tests:test --rerun-tasks --tests '*Metal*'`. | All Metal conformance and public Engine surfaces pass against the extracted dylib. Tests exercise both numerical profiles where admitted and cover transfer, rank-zero, affine, reuse/session/concurrency, saved state, multi-output, generated backward, route tuning without benchmark authority, and failure-before-mutation. |
| `V9` | Inventory the permanent runtime evidence after `V6..V8`; create a throwaway Engine smoke only for a named uncovered requested surface, execute it, and remove it. | Either an exact permanent test citation for every requested surface or one removed throwaway smoke result; no retained padding test. |
| `V10` | Run `./gradlew :testing:architecture-tests:test --rerun-tasks` and `./gradlew javadoc --rerun-tasks`. | Architecture and documentation generation pass; warnings are reported accurately and not hidden. |
| `V11` | Run `./gradlew test --rerun-tasks` with the extracted dylib. | Every JVM test task reruns; parse all JUnit XML for exact suite/test/failure/error/skip totals and separately prove relevant Metal/conformance/integration suites have no skip or failure. |
| `V12` | Run `./gradlew build --rerun-tasks` with the extracted dylib. | Complete repository build succeeds from the same source and packaged artifact. This invokes no benchmark entry point. |
| `V13` | Run documentation/diff/status checks, remove the extraction directory and every throwaway source, record exact commands/results/artifact identities, and commit completion records or source fixes. | Clean tree; current Task/master/roadmap status is truthful; no source defect or environmental blocker is omitted. |
| `V14` | Independent cumulative code, security, and evidence review of the final committed state. | Three independent `APPROVE` results with zero remaining P0/P1/P2 before final report. |

## Failure and rerun policy

Any failed assertion, crash, native status, package mismatch, relevant skip, or build failure is a
source/test/package defect unless the environment itself is demonstrated unavailable or corrupt.
Fix the source of a defect; do not suppress the symptom, weaken an oracle, special-case an input,
retry a route, or substitute CPU/host behavior. After a fix, rerun the smallest affected gate for
diagnosis and then rerun `V1..V14` from a newly rebuilt artifact. Record the original failure, root
cause, remediation, and both rerun results.

## Completion criteria

Task 0068 is Complete only when all gates pass against one freshly rebuilt, fixed-identifier signed,
canonically packaged, Gradle-ingested, ZIP-extracted dylib; artifact identity and JUnit totals are
recorded; no throwaway remains; the tree is clean; all active planning status is updated; and
independent final code/security/evidence review reports zero remaining P0/P1/P2. Package integrity
is local verification, not Developer ID authentication, notarization, publication, or provenance.

## Executed artifact chain

The warnings-as-errors build completed from current Objective-C source with no compiler output.
The resulting dylib was signed with the fixed ad-hoc identifier, packaged, independently verified,
ingested by Gradle, archived, extracted permission-preservingly into fresh directory
`/tmp/synaptik-metal-task0068.XXx4yf`, and independently verified again. Recursive comparison found
the canonical and extracted three-member packages byte-identical. Every runtime command used only:

```text
/tmp/synaptik-metal-task0068.XXx4yf/macos-arm64/libsynaptik_metal_foundation.dylib
```

The extraction directory was removed after all runtime evidence and final JUnit results were
captured.

Canonical package identities:

- dylib: `468416` bytes, mode `0755`, SHA-256
  `32660991dd284af45c5b550a04bc8c12233f4b64014b783c1c06bbed111306d6`;
- `manifest.json`: `512` bytes, mode `0644`, SHA-256
  `edeb697628d0618020c339205fbb6a8059dbadc191b4e1fb835c37d9b7dad405`;
- `SHA256SUMS`: `181` bytes, mode `0644`, SHA-256
  `47d9753ffe220428848654e7ba375d173a5050ffcf81c34825152b111f637ad6`;
- ad-hoc signature identifier `io.github.pho001.synaptik.metal.foundation`, no Team ID, CDHash
  `0b9bc598e347def859990d24496bfd1c014a25f4`.

Gradle local ZIP identity:

- `backends/metal/build/distributions/synaptik-metal-macos-arm64-local.zip`;
- `139951` bytes, SHA-256
  `e3bbac2e47be15694b22d518f7ec0cc8f50d5fe4450ede1ab677b2085046aa1b`;
- four ordered entries with fixed `1980-02-01 00:00:00` timestamps: directory mode `0755`, dylib
  mode `0755`, and both metadata files mode `0644`;
- `469109` uncompressed and `139415` compressed member bytes.

Both verifiers proved arm64 Mach-O 64-bit `DYLIB`, macOS minimum `26.0`, install name
`@rpath/libsynaptik_metal_foundation.dylib`, no `LC_RPATH` or legacy minimum-version command,
normalized system dependencies with Foundation/Metal/MetalPerformanceShadersGraph exactly once,
ABI 5, schema 15, and the exact thirteen exports.

## Command results

| Gate | Result |
|---|---|
| `V1` | `./native/metal-macos-arm64/build.sh` passed with native `-Wall -Wextra -Werror` and no warning. |
| `V2..V3` | Fixed-identifier signing, `package-local.sh`, and canonical `verify-package.sh` passed. |
| `V4` | Gradle package verification and local ZIP creation passed; `2/2` tasks executed. |
| `V5` | Fresh `ditto` extraction, extracted-package verification, and recursive byte comparison passed. |
| `V6` | Complete `:backends:metal:test --rerun-tasks` passed; `19/19` tasks executed. |
| `V7` | Complete `:modules:compiler:test --rerun-tasks` passed; `13/13` tasks executed. |
| `V8` | Metal conformance plus every `*Metal*` public integration surface passed; `38/38` tasks executed. |
| `V10` | Architecture tests plus Javadoc passed; `47/47` tasks executed. Javadoc emitted the existing 100 missing-`@param` warnings in CPU-internal constructors; they were visible and are outside native warnings-as-errors scope. |
| `V11` | `./gradlew test --rerun-tasks` passed against the extracted dylib; `76/76` tasks executed. |
| `V12` | `./gradlew build --rerun-tasks` passed against the same extracted dylib; `87/87` tasks executed. Ordinary benchmark-module sources were compiled, but no benchmark entry point or standalone timing/autotuning command ran, and no local elapsed value or selected winner informed verification or source. Mandatory full-suite tuning tests exercised only their permanent public API assertions. |

The JVM emitted its existing restricted-native-access and incubator-vector warnings, and one
existing CPU test compilation note about unchecked operations. They did not hide a failed task,
native compiler warning, relevant skip, or Metal assertion.

## Runtime coverage inventory

No throwaway Engine smoke was created because every requested surface has permanent behavioral
coverage:

- `MetalOperationCompletenessAuditTest` executes the real-provider 115-row audit and proves
  `83/32`, `101/14`, MPSGraph `75/35/5`, custom `70/45/0`, and route `68/13/1/1/32`;
- `MetalMpsGraphAbsNativeTest`, `MetalMpsGraphAffineNativeTest`,
  `MetalMpsGraphMatmulNativeTest`, `MetalNegPreparedExecutionTest`, and
  `MetalRandomDropoutNativeTest` cover direct/custom route execution, repeated reuse, independent
  and concurrent contexts/runs, lifecycle, affine storage, rank-zero transfer, and state isolation;
- `MetalMpsGraphIndexingNativeTest`, `MetalMpsGraphRawAbiNativeTest`, and the raw alias cases cover
  complete validation and bounds/duplicate/topology/alias rejection before mutation;
- `DtypeLayoutGradientMetalIntegrationTest`,
  `EngineExplicitCompositionMetalIntegrationTest`,
  `EngineConvolutionPoolingMetalIntegrationTest`, `EngineOrderingMetalIntegrationTest`, and
  `RandomDropoutMetalIntegrationTest` cover both profiles where permitted, all-carrier transfer,
  rank zero/one/sixteen, affine reuse, saved roles/state, multi-output, route-specific MATMUL/NEG,
  and direct/generated backward through public Engine;
- the complete Compiler suite covers autograd preflight, saved-role liveness, direct/generated
  formulas, static-result layout closure, and the exact supported/deferred inventory.

## Final JUnit evidence

All XML reports from the final full build were parsed:

- repository: `558` suites, `3712` tests, `0` failures, `0` errors, `29` skips;
- Metal-related: `306` tests, `0` failures, `0` errors, `0` skips;
- Metal backend: `34` suites, `247` tests, `0` failures, `0` errors, `0` skips;
- Compiler: `40` suites, `282` tests, `0` failures, `0` errors, `0` skips;
- backend conformance: `3` suites, `22` tests, `0` failures, `0` errors, `0` skips;
- integration: `13` suites, `69` tests, `0` failures, `0` errors, `1` skip;
- architecture: `7` suites, `9` tests, `0` failures, `0` errors, `0` skips.

The one integration skip is CPU-only
`EngineModelAutotuningIntegrationTest.eligibleCpuAlternativesCompleteBothPublicTuningPhases`; the
other 28 skips are CPU backend opt-in evidence/performance cases. No Metal-related test skipped.

## Verification result before final review

Gates `V0..V13` passed without a source change or diagnostic rerun. The package, archive, extracted
dylib, native/backend behavior, public Engine behavior, Compiler backward closure, and repository
build all match the Task-0067 approved boundary. No environmental blocker or Task-0068 blocker
remains. Developer ID signing, notarization, authenticated publication, and the separate Task-0053
constructive-real proof bridge remain outside this local verification.
