# Task 0068: [Release](../../../../glossary.md#term-release)-Grade Native Integration Verification

## Status

Complete — gates `V0..V14` passed from Task-0067 native/source base
`cb830587c0922e63be7a8e2d47822fe18e2b2bd6` with registered-[matrix](../../../../glossary.md#term-matrix-meanings) [commit](../../../../glossary.md#term-commit)
`6820dd631cbd28f855640a0edc5d125ae6a905a6`. No production source defect, environmental failure,
relevant skip, uncovered [runtime](../../../../glossary.md#term-runtime) surface, or remaining P0/P1/P2 was found.

## Change class

Verification-first Class C. This task rebuilds, signs, packages, archives, extracts, and exercises the
current native [Metal](../../../../glossary.md#term-metal) implementation end to end. It adds no capability, route, [schema](../../../../glossary.md#term-schema), [ABI](../../../../glossary.md#term-application-binary-interface-abi), export,
[identity](../../../../glossary.md#term-identity-meanings), fallback, retry, [host](../../../../glossary.md#term-host) repair, timing decision, benchmark, or [autotuning](../../../../glossary.md#autotuning--model-autotuning) behavior. A
demonstrated source defect is in [scope](../../../../glossary.md#term-scope) for direct remediation followed by the affected and complete
[matrix](../../../../glossary.md#term-matrix-meanings); an environmental blocker must be proven and recorded separately rather than treated as a
source pass.

## Goal

Prove from current native source that the fixed local package and optional [Gradle](../../../../glossary.md#term-gradle) ZIP preserve the
exact [ABI](../../../../glossary.md#term-application-binary-interface-abi)-5/[schema](../../../../glossary.md#term-schema)-15/[identity](../../../../glossary.md#term-identity-meanings)-22 [artifact](../../../../glossary.md#term-artifact), and that the permission-preservingly extracted packaged
[dylib](../../../../glossary.md#term-dylib) passes every current [Metal](../../../../glossary.md#term-metal) native/[backend](../../../../glossary.md#backend), [conformance](../../../../glossary.md#term-conformance), public [Engine](../../../../glossary.md#term-engine), [Compiler](../../../../glossary.md#term-compiler)/[autograd](../../../../glossary.md#autograd),
architecture, [Javadoc](../../../../glossary.md#term-javadoc), repository-test, and full-build surface.

The proof must cover the exact Task-0067 closure: 115 unique [operation](../../../../glossary.md#operation) wires; capability `83/32`;
structural execution `101/14`; [MPSGraph](../../../../glossary.md#term-mpsgraph) `75/35/5`; custom `70/45/0`; production routes
`68 custom / 13 MPSGraph / 1 NEG dual / 1 MATMUL by-domain / 32 blocked`; and Compiler
`38 families / 111 kinds / 133 fingerprints + four deferred signatures = 40 / 115 / 137`.

## Dependencies and integration

- Depends on: Task 0067 Complete and independently approved at
  `cb830587c0922e63be7a8e2d47822fe18e2b2bd6`; Tasks 0045–0046 package/archive contract; current
  ABI-5/schema-15/identity-22 native and Java boundary.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [schema glossary](../../../../glossary.md#term-schema); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Conflicts with: every concurrent Metal, Compiler, native, packaging, Gradle, test, planning, or
  shared-document edit during verification.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [Compiler glossary](../../../../glossary.md#term-compiler); [Gradle glossary](../../../../glossary.md#term-gradle); [planning glossary](../../../../glossary.md#term-planning))</span>
- Parallel group: None.
- Integration order: matrix registration commit, serial artifact construction, serial verification,
  evidence finalization, independent cumulative code/security/evidence review.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [commit glossary](../../../../glossary.md#term-commit); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): Task 0068 finalizer.

## Fixed boundaries

- The native [compiler](../../../../glossary.md#term-compiler) invocation must retain `-Wall -Wextra -Werror`; warnings are failures.
- Signing is local ad hoc with fixed identifier
  `io.github.pho001.synaptik.metal.foundation`, no timestamp, and no Team ID.
- Every runtime test must use the absolute dylib path extracted from the freshly generated Gradle
  ZIP, not the build output or canonical package member.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [dylib glossary](../../../../glossary.md#term-dylib); [Gradle glossary](../../../../glossary.md#term-gradle); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- No standalone benchmark, timing, route-race, cache-choice, or autotuning command may run, and no
  elapsed value or locally selected winner may become verification evidence or a release/source
  decision. The mandatory complete JVM/public Engine suites retain their ordinary behavioral tests
  of measured tuning, warmup, fallback, and cache paths; those results are asserted only as API
  behavior and are never inspected, compared, or promoted into this verification decision. No
  production fallback, retry, or host repair is permitted. Gradle may compile ordinary
  benchmark-module sources as part of a full build, but no benchmark entry point is invoked.<span data-glossary-navigation="true"> ([race glossary](../../../../glossary.md#term-race); [cache glossary](../../../../glossary.md#term-cache); [autotuning glossary](../../../../glossary.md#autotuning--model-autotuning); [run glossary](../../../../glossary.md#term-run); [release glossary](../../../../glossary.md#term-release); [JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [Engine glossary](../../../../glossary.md#term-engine); [warmup glossary](../../../../glossary.md#term-warmup); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [host glossary](../../../../glossary.md#term-host); [compile glossary](../../../../glossary.md#compile); [module glossary](../../../../glossary.md#term-module))</span>
- A throwaway actual Engine smoke is allowed only after the permanent runtime inventory proves one
  requested surface has no existing behavioral coverage. It must exercise only that uncovered
  surface and be removed before final validation and commit. No smoke is created merely to add a
  test count.<span data-glossary-navigation="true"> ([smoke glossary](../../../../glossary.md#term-smoke-test); [validation glossary](../../../../glossary.md#term-validation); [commit glossary](../../../../glossary.md#term-commit))</span>
- Ignored local build/package/ZIP outputs are verification artifacts, not source changes. The fresh
  extraction directory is removed after evidence capture.<span data-glossary-navigation="true"> ([artifacts glossary](../../../../glossary.md#term-artifact); [capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- Existing production-false rows and Task-0053 proof blockers remain fail-closed; verification is
  not authorization to narrow or special-case them.<span data-glossary-navigation="true"> ([fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>

## Registered verification [matrix](../../../../glossary.md#term-matrix-meanings)

| Gate | Exact command or action | Required evidence |
|---|---|---|
| `V0` | [Commit](../../../../glossary.md#term-commit) this registered matrix from clean `cb830587`; no production edit. | Task/master/roadmap identify the exact base, serial [owner](../../../../glossary.md#term-owner), commands, pass criteria, and blocker policy. |
| `V1` | `./native/metal-macos-arm64/build.sh` | Fresh private-directory [compile](../../../../glossary.md#compile) from current [Objective-C](../../../../glossary.md#native-packaging-and-platform-vocabulary) source; generated Task-0053 header check; [arm64](../../../../glossary.md#native-packaging-and-platform-vocabulary)/macOS-26 [dylib](../../../../glossary.md#term-dylib); native `-Wall -Wextra -Werror` passes. |
| `V2` | `/usr/bin/codesign --force --sign - --timestamp=none --identifier io.github.pho001.synaptik.metal.foundation native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib` | Strict ad-hoc [signature](../../../../glossary.md#term-signature) with exact identifier and no Team ID. |
| `V3` | `package-local.sh` on the signed dylib, then `verify-package.sh` on `build/package-v1/macos-arm64`. | Exactly three [canonical](../../../../glossary.md#term-canonical-form) package members; checksums, modes, arm64 [Mach-O](../../../../glossary.md#term-mach-o), macOS 26.0, [install name](../../../../glossary.md#term-install-name), no [rpath](../../../../glossary.md#term-rpath), normalized system dependencies, [ABI](../../../../glossary.md#term-application-binary-interface-abi) 5, [schema](../../../../glossary.md#term-schema) 15, and exact thirteen exports. |
| `V4` | `./gradlew :backends:metal:verifyMetalNativePackage :backends:metal:metalNativeLocalZip --rerun-tasks -PsynaptikMetalNativePackage=<absolute canonical package>` | [Gradle](../../../../glossary.md#term-gradle) ingests only the explicit verified package and produces the reproducible local ZIP. |
| `V5` | Extract the ZIP with `/usr/bin/ditto -x -k` into one fresh `mktemp -d` directory; rerun `verify-package.sh` on the extracted `macos-arm64`. | Permission-preserved extracted package independently passes the same verifier and every member is byte-identical to the canonical package. Record dylib/package [metadata](../../../../glossary.md#term-metadata)/ZIP sizes and [SHA-256](../../../../glossary.md#term-sha-256) values plus signature CDHash. |
| `V6` | With `SYNAPTIK_METAL_TEST_LIBRARY=<absolute extracted dylib>`, [run](../../../../glossary.md#term-run) `./gradlew :backends:metal:test --rerun-tasks`. | Complete [Metal](../../../../glossary.md#term-metal) [JVM](../../../../glossary.md#term-java-virtual-machine-jvm)/native suite passes against the packaged dylib with no relevant skip. It includes the 115-row real-provider audit, 83 positive and 32 false representatives, structural/catalog/route partitions, [MPSGraph](../../../../glossary.md#term-mpsgraph)/custom/NEG/[MATMUL](../../../../glossary.md#matrix-multiplication--matmul)-by-domain execution, raw ABI/schema/[preflight](../../../../glossary.md#term-preflight), [alias](../../../../glossary.md#aliasing-and-arena) and error-before-mutation, [rank-zero](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), affine, saved-state, multi-output, reuse, sessions, [concurrency](../../../../glossary.md#term-concurrency), and resource [lifecycle](../../../../glossary.md#lifecycle) coverage. |
| `V7` | Run `./gradlew :modules:compiler:test --rerun-tasks`. | Complete [Compiler](../../../../glossary.md#term-compiler) [inference](../../../../glossary.md#term-inference), [autograd](../../../../glossary.md#autograd) preflight, direct/generated backward, saved-role, logical-[layout](../../../../glossary.md#layout), ordering, window, [convolution](../../../../glossary.md#term-channels-first-convolution-layers)/[pooling](../../../../glossary.md#term-pooling), [dropout](../../../../glossary.md#dropout--inverted-dropout), [loss](../../../../glossary.md#loss), deferred [Conv3d](../../../../glossary.md#conv3d--ncdhw)/recurrent, and inventory closure suites pass. |
| `V8` | Run `./gradlew :testing:backend-conformance:test --rerun-tasks --tests '*Metal*'` and `./gradlew :testing:integration-tests:test --rerun-tasks --tests '*Metal*'`. | All Metal [conformance](../../../../glossary.md#term-conformance) and public [Engine](../../../../glossary.md#term-engine) surfaces pass against the extracted dylib. Tests exercise both numerical profiles where admitted and cover transfer, rank-zero, affine, reuse/session/concurrency, saved state, multi-output, generated backward, route tuning without benchmark authority, and failure-before-mutation. |
| `V9` | Inventory the permanent [runtime](../../../../glossary.md#term-runtime) evidence after `V6..V8`; create a throwaway Engine [smoke](../../../../glossary.md#term-smoke-test) only for a named uncovered requested surface, execute it, and remove it. | Either an exact permanent test citation for every requested surface or one removed throwaway smoke result; no retained [padding](../../../../glossary.md#padding) test. |
| `V10` | Run `./gradlew :testing:architecture-tests:test --rerun-tasks` and `./gradlew javadoc --rerun-tasks`. | Architecture and documentation generation pass; warnings are reported accurately and not hidden. |
| `V11` | Run `./gradlew test --rerun-tasks` with the extracted dylib. | Every JVM test task reruns; parse all [JUnit](../../../../glossary.md#term-junit) XML for exact suite/test/failure/error/skip totals and separately prove relevant Metal/conformance/integration suites have no skip or failure. |
| `V12` | Run `./gradlew build --rerun-tasks` with the extracted dylib. | Complete repository build succeeds from the same source and packaged [artifact](../../../../glossary.md#term-artifact). This invokes no benchmark entry point. |
| `V13` | Run documentation/diff/status checks, remove the extraction directory and every throwaway source, record exact commands/results/artifact [identities](../../../../glossary.md#term-identity-meanings), and commit completion records or source fixes. | Clean tree; current Task/master/roadmap status is truthful; no source defect or environmental blocker is omitted. |
| `V14` | Independent cumulative code, security, and evidence review of the final committed state. | Three independent `APPROVE` results with zero remaining P0/P1/P2 before final report. |

## Failure and rerun policy

Any failed assertion, crash, native status, package mismatch, relevant skip, or build failure is a
source/test/package defect unless the environment itself is demonstrated unavailable or corrupt.
Fix the source of a defect; do not suppress the symptom, weaken an [oracle](../../../../glossary.md#term-test-oracle), special-case an input,
retry a route, or substitute [CPU](../../../../glossary.md#term-cpu)/[host](../../../../glossary.md#term-host) behavior. After a fix, rerun the smallest affected gate for
diagnosis and then rerun `V1..V14` from a newly rebuilt [artifact](../../../../glossary.md#term-artifact). Record the original failure, root
cause, remediation, and both rerun results.

## Completion criteria

Task 0068 is Complete only when all gates pass against one freshly rebuilt, fixed-identifier signed,
canonically packaged, [Gradle](../../../../glossary.md#term-gradle)-ingested, ZIP-extracted [dylib](../../../../glossary.md#term-dylib); [artifact](../../../../glossary.md#term-artifact) [identity](../../../../glossary.md#term-identity-meanings) and [JUnit](../../../../glossary.md#term-junit) totals are
recorded; no throwaway remains; the tree is clean; all active [planning](../../../../glossary.md#term-planning) status is updated; and
independent final code/security/evidence review reports zero remaining P0/P1/P2. Package [integrity](../../../../glossary.md#term-integrity)
is local verification, not Developer ID [authentication](../../../../glossary.md#term-authentication), [notarization](../../../../glossary.md#term-notarization), [publication](../../../../glossary.md#term-publication), or [provenance](../../../../glossary.md#provenance).

## Executed [artifact](../../../../glossary.md#term-artifact) chain

The warnings-as-errors build completed from current [Objective-C](../../../../glossary.md#native-packaging-and-platform-vocabulary) source with no [compiler](../../../../glossary.md#term-compiler) output.
The resulting [dylib](../../../../glossary.md#term-dylib) was signed with the fixed ad-hoc identifier, packaged, independently verified,
ingested by [Gradle](../../../../glossary.md#term-gradle), archived, extracted permission-preservingly into fresh directory
`/tmp/synaptik-metal-task0068.XXx4yf`, and independently verified again. Recursive comparison found
the [canonical](../../../../glossary.md#term-canonical-form) and extracted three-member packages byte-identical. Every [runtime](../../../../glossary.md#term-runtime) command used only:

```text
/tmp/synaptik-metal-task0068.XXx4yf/macos-arm64/libsynaptik_metal_foundation.dylib
```

The extraction directory was removed after all runtime evidence and final [JUnit](../../../../glossary.md#term-junit) results were
captured.

Canonical package [identities](../../../../glossary.md#term-identity-meanings):

- dylib: `468416` bytes, mode `0755`, [SHA-256](../../../../glossary.md#term-sha-256)
  `32660991dd284af45c5b550a04bc8c12233f4b64014b783c1c06bbed111306d6`;
- `manifest.json`: `512` bytes, mode `0644`, SHA-256
  `edeb697628d0618020c339205fbb6a8059dbadc191b4e1fb835c37d9b7dad405`;
- `SHA256SUMS`: `181` bytes, mode `0644`, SHA-256
  `47d9753ffe220428848654e7ba375d173a5050ffcf81c34825152b111f637ad6`;
- ad-hoc [signature](../../../../glossary.md#term-signature) identifier `io.github.pho001.synaptik.metal.foundation`, no Team ID, CDHash
  `0b9bc598e347def859990d24496bfd1c014a25f4`.

Gradle local ZIP [identity](../../../../glossary.md#term-identity-meanings):

- `backends/metal/build/distributions/synaptik-metal-macos-arm64-local.zip`;
- `139951` bytes, SHA-256
  `e3bbac2e47be15694b22d518f7ec0cc8f50d5fe4450ede1ab677b2085046aa1b`;
- four ordered entries with fixed `1980-02-01 00:00:00` timestamps: directory mode `0755`, dylib
  mode `0755`, and both [metadata](../../../../glossary.md#term-metadata) files mode `0644`;
- `469109` uncompressed and `139415` compressed member bytes.

Both verifiers proved [arm64](../../../../glossary.md#native-packaging-and-platform-vocabulary) [Mach-O](../../../../glossary.md#term-mach-o) 64-bit `DYLIB`, macOS minimum `26.0`, [install name](../../../../glossary.md#term-install-name)
`@rpath/libsynaptik_metal_foundation.dylib`, no `LC_RPATH` or legacy minimum-version command,
normalized system dependencies with Foundation/[Metal](../../../../glossary.md#term-metal)/MetalPerformanceShadersGraph exactly once,
[ABI](../../../../glossary.md#term-application-binary-interface-abi) 5, [schema](../../../../glossary.md#term-schema) 15, and the exact thirteen exports.

## Command results

| Gate | Result |
|---|---|
| `V1` | `./native/metal-macos-arm64/build.sh` passed with native `-Wall -Wextra -Werror` and no warning. |
| `V2..V3` | Fixed-identifier signing, `package-local.sh`, and [canonical](../../../../glossary.md#term-canonical-form) `verify-package.sh` passed. |
| `V4` | [Gradle](../../../../glossary.md#term-gradle) package verification and local ZIP creation passed; `2/2` tasks executed. |
| `V5` | Fresh `ditto` extraction, extracted-package verification, and recursive byte comparison passed. |
| `V6` | Complete `:backends:metal:test --rerun-tasks` passed; `19/19` tasks executed. |
| `V7` | Complete `:modules:compiler:test --rerun-tasks` passed; `13/13` tasks executed. |
| `V8` | [Metal](../../../../glossary.md#term-metal) [conformance](../../../../glossary.md#term-conformance) plus every `*Metal*` public integration surface passed; `38/38` tasks executed. |
| `V10` | Architecture tests plus [Javadoc](../../../../glossary.md#term-javadoc) passed; `47/47` tasks executed. Javadoc emitted the existing 100 missing-`@param` warnings in [CPU](../../../../glossary.md#term-cpu)-internal constructors; they were visible and are outside native warnings-as-errors [scope](../../../../glossary.md#term-scope). |
| `V11` | `./gradlew test --rerun-tasks` passed against the extracted [dylib](../../../../glossary.md#term-dylib); `76/76` tasks executed. |
| `V12` | `./gradlew build --rerun-tasks` passed against the same extracted dylib; `87/87` tasks executed. Ordinary benchmark-[module](../../../../glossary.md#term-module) sources were compiled, but no benchmark entry point or standalone timing/[autotuning](../../../../glossary.md#autotuning--model-autotuning) command ran, and no local elapsed value or selected winner informed verification or source. Mandatory full-suite tuning tests exercised only their permanent public [API](../../../../glossary.md#term-application-programming-interface-api) assertions. |

The [JVM](../../../../glossary.md#term-java-virtual-machine-jvm) emitted its existing restricted-native-access and incubator-[vector](../../../../glossary.md#term-vector) warnings, and one
existing CPU test [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) note about unchecked operations. They did not hide a failed task,
native [compiler](../../../../glossary.md#term-compiler) warning, relevant skip, or Metal assertion.

## [Runtime](../../../../glossary.md#term-runtime) coverage inventory

No throwaway [Engine](../../../../glossary.md#term-engine) [smoke](../../../../glossary.md#term-smoke-test) was created because every requested surface has permanent behavioral
coverage:

- `MetalOperationCompletenessAuditTest` executes the real-provider 115-row audit and proves
  `83/32`, `101/14`, [MPSGraph](../../../../glossary.md#term-mpsgraph) `75/35/5`, custom `70/45/0`, and route `68/13/1/1/32`;
- `MetalMpsGraphAbsNativeTest`, `MetalMpsGraphAffineNativeTest`,
  `MetalMpsGraphMatmulNativeTest`, `MetalNegPreparedExecutionTest`, and
  `MetalRandomDropoutNativeTest` cover direct/custom route execution, repeated reuse, independent
  and concurrent contexts/runs, lifecycle, affine storage, rank-zero transfer, and state isolation;<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- `MetalMpsGraphIndexingNativeTest`, `MetalMpsGraphRawAbiNativeTest`, and the raw alias cases cover
  complete validation and bounds/duplicate/topology/alias rejection before mutation;<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [validation glossary](../../../../glossary.md#term-validation); [topology glossary](../../../../glossary.md#term-topology))</span>
- `DtypeLayoutGradientMetalIntegrationTest` (`8` tests),
  `EngineExplicitCompositionMetalIntegrationTest` (`26`),
  `EngineConvolutionPoolingMetalIntegrationTest` (`6`),
  `EngineOrderingMetalIntegrationTest` (`4`), and
  `RandomDropoutMetalIntegrationTest` (`3`) form the five explicitly named Metal integration
  classes (<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal))</span>`8 + 26 + 6 + 4 + 3 = 47` tests). They cover both profiles where permitted, all-carrier
  transfer, rank zero/one/sixteen, affine reuse, saved roles/state, multi-output, route-specific
  MATMUL/NEG, and direct/generated backward through public Engine;<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [rank glossary](../../../../glossary.md#term-rank); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
- the complete Compiler suite covers autograd preflight, saved-role liveness, direct/generated
  formulas, static-result layout closure, and the exact supported/deferred inventory.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [autograd glossary](../../../../glossary.md#autograd); [preflight glossary](../../../../glossary.md#term-preflight); [liveness glossary](../../../../glossary.md#term-liveness); [layout glossary](../../../../glossary.md#layout))</span>

## Final [JUnit](../../../../glossary.md#term-junit) evidence

All XML reports from the final full build were parsed:

- repository: `558` suites, `3712` tests, `0` failures, `0` errors, `29` skips;
- precisely scoped [Metal](../../../../glossary.md#term-metal) [runtime](../../../../glossary.md#term-runtime) subset: `304` tests, `0` failures, `0` errors, `0` skips =
  `247` Metal [backend](../../../../glossary.md#backend) + `10` `MetalNegCapabilityPartitionConformanceTest` + `47` across the five
  explicitly named Metal integration classes above;
- Metal backend: `34` suites, `247` tests, `0` failures, `0` errors, `0` skips;
- [Compiler](../../../../glossary.md#term-compiler): `40` suites, `282` tests, `0` failures, `0` errors, `0` skips;
- backend [conformance](../../../../glossary.md#term-conformance): `3` suites, `22` tests, `0` failures, `0` errors, `0` skips;
- integration: `13` suites, `69` tests, `0` failures, `0` errors, `1` skip;
- architecture: `7` suites, `9` tests, `0` failures, `0` errors, `0` skips.

The five-class `47` subtotal intentionally excludes `TrainingSessionCpuIntegrationTest`: that
[CPU](../../../../glossary.md#term-cpu)-named class has three tests in the full `69`-test integration result, although one method name
contains `Metal`. The one integration skip is CPU-only
`EngineModelAutotuningIntegrationTest.eligibleCpuAlternativesCompleteBothPublicTuningPhases`; the
other 28 skips are CPU backend opt-in evidence/performance cases. No test in the precisely scoped
`304`-test Metal subset skipped.

## Final verification result

Gates `V0..V13` passed without a production source change or diagnostic rerun. The package, archive,
extracted [dylib](../../../../glossary.md#term-dylib), native/[backend](../../../../glossary.md#backend) behavior, public [Engine](../../../../glossary.md#term-engine) behavior, [Compiler](../../../../glossary.md#term-compiler) backward closure, and
repository build all match the Task-0067 approved boundary. No environmental blocker or
Task-0068 blocker remains. Developer ID signing, [notarization](../../../../glossary.md#term-notarization), authenticated [publication](../../../../glossary.md#term-publication), and the
separate Task-0053 constructive-real proof bridge remain outside this local verification.

The initial code/evidence reviews at `372d0261` found one P1 evidence contradiction: the prose said
no tuning or fallback path ran even though the mandatory complete public Engine/full [JVM](../../../../glossary.md#term-java-virtual-machine-jvm) suites
exercise measured tuning and safe-[heuristic](../../../../glossary.md#term-heuristic) fallback as permanent [API](../../../../glossary.md#term-application-programming-interface-api) behavior. Revision
`1775081ab1de43789006b0e704a222195e94edde` corrected Task/master/roadmap consistently: no
standalone benchmark/timing/[autotuning](../../../../glossary.md#autotuning--model-autotuning) command ran, no production fallback/retry/[host](../../../../glossary.md#term-host) repair
occurred, and no elapsed value or locally selected winner informed verification or source. The
full-suite behavior remains included rather than silently excluded. No [runtime](../../../../glossary.md#term-runtime) rerun was needed
because neither source, command, [artifact](../../../../glossary.md#term-artifact), test selection, nor result changed; the affected
documentation was independently re-reviewed.

Independent cumulative code, security, and evidence reviewers each returned `APPROVE` with zero
remaining P0/P1/P2 at exact clean revision
`1775081ab1de43789006b0e704a222195e94edde`. The code reviewer confirmed the docs-only delta,
packaged-dylib chain, permanent coverage, and no local timing authority; the security reviewer
confirmed the signing/package/archive/loading/[fail-closed](../../../../glossary.md#term-fail-closed) boundary and local ad-hoc limitations;
the evidence reviewer reconciled all arithmetic, [identities](../../../../glossary.md#term-identity-meanings), counts, skips, cleanup, and active
[planning](../../../../glossary.md#term-planning) status.
