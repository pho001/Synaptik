# Task 0047: Apple Silicon [Metal](../../../../glossary.md#term-metal) [CI](../../../../glossary.md#term-continuous-integration-ci)

## Reader context

The executable-looking workflow below was a gated proposal, not an installed continuous
integration (CI) system. A macOS job compiling successfully could not prove real GPU execution,
and skipped device tests could not satisfy the proposed functional job. Runner labels, action
pins, prices and account eligibility were recorded planning facts, not present-day provider
guarantees.

An authorized repository owner needed to resolve provider access and obtain actual device
evidence before implementation could be authorized. The proposed ephemeral job would own its
generated packages, extraction and cleanup; it was not a distribution service or a timing-based
performance gate.
The [current Metal qualification and hosted-runner boundary](../../../../backend-guide/production-support.md#metal-production-target)
keeps local source-build acceptance separate from hosted GPU CI and trusted public prebuilts.
The original Blocked status and proposed commands remain unchanged.

## Status

Blocked

The current public repository is owned by a personal account. GitHub documents the standard
`macos-26` [arm64](../../../../glossary.md#native-packaging-and-platform-vocabulary) runner as M1 hardware but does not document [Metal](../../../../glossary.md#term-metal) [GPU](../../../../glossary.md#term-gpu) acceleration there, and the
standard M1 runner history records unavailable [MPS](../../../../glossary.md#term-mps) hardware. The required `macos-26-xlarge` runner
is the documented arm64 M2 option with eight-GPU hardware acceleration, but GitHub makes larger
runners available only to eligible organizations and enterprises with billing and a positive
spending limit. No workflow may be added until a real GPU-capable [hosted runner](../../../../glossary.md#continuous-integration-and-hosted-runners) is enabled for this
repository. A [compile](../../../../glossary.md#compile)-only macOS [lane](../../../../glossary.md#term-lane), skipped Metal tests, or conditional [device](../../../../glossary.md#term-device) fallback does not
satisfy this task. No [CI](../../../../glossary.md#term-continuous-integration-ci) implementation is [Ready](../../../../glossary.md#term-ready) under Task 0047; independent Task 0048 is
Complete at independently approved implementation `89f9fbb9`, and no Metal task is Ready.

## Change class

Class C — a future workflow will execute pull-request code, download pinned toolchains, and build,
sign, package, load, and execute native bytes on a hosted [GPU](../../../../glossary.md#term-gpu). After the provider gate is resolved,
use one clean implementation context and an independent security- and [lifecycle](../../../../glossary.md#lifecycle)-focused review
before completion.

## Goal

Add the smallest useful continuous-integration boundary: one ordinary portable Java [lane](../../../../glossary.md#term-lane) and one
Apple-silicon macOS functional lane. The functional lane must build and ad-hoc sign the current
[native library](../../../../glossary.md#term-native-library), create and verify the reviewed local package and archive, load the extracted [dylib](../../../../glossary.md#term-dylib)
through the explicit absolute path, [run](../../../../glossary.md#term-run) the actual [Metal](../../../../glossary.md#term-metal) and [Engine](../../../../glossary.md#term-engine) integration coverage without
skips, and run one fixed report-only benchmark [smoke](../../../../glossary.md#term-smoke-test). It must not create a performance, context,
executable, profile, or [Shape](../../../../glossary.md#shape) [matrix](../../../../glossary.md#term-matrix-meanings).

## Provider gate and evidence

- GitHub's [hosted-runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
  lists standard `macos-26` as arm64 M1 with three CPU cores, 7 GB RAM, and 14 GB storage for public
  and private repositories. It does not promise GPU acceleration.<span data-glossary-navigation="true"> ([arm64 glossary](../../../../glossary.md#native-packaging-and-platform-vocabulary); [CPU glossary](../../../../glossary.md#term-cpu); [RAM glossary](../../../../glossary.md#term-random-access-memory-ram); [GPU glossary](../../../../glossary.md#term-gpu))</span>
- The standard-M1 [runner announcement](https://github.com/actions/runner-images/issues/9254)
  records that MPS hardware was unavailable, and the transferred<span data-glossary-navigation="true"> ([MPS glossary](../../../../glossary.md#term-mps))</span>
  [Metal support discussion](https://github.com/orgs/community/discussions/160669) remains
  unanswered. Standard `macos-26` may compile and inspect Mach-O bytes, but it is not evidence for
  functional Metal execution.<span data-glossary-navigation="true"> ([Mach-O glossary](../../../../glossary.md#term-mach-o))</span>
- GitHub's [larger-runner reference](https://docs.github.com/en/actions/reference/runners/larger-runners)
  lists `macos-26-xlarge` as arm64 M2 with five CPU cores and eight-GPU hardware acceleration.
  GitHub's [hosted-runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners#larger-runners)
  restricts larger runners to organizations and enterprises on Team or Enterprise Cloud plans.
  [Actions pricing](https://docs.github.com/en/billing/reference/actions-runner-pricing) additionally
  requires billing; included minutes do not apply, and the current macOS M2 rate is USD 0.102 per
  minute.
- The repository is currently public under personal [owner](../../../../glossary.md#term-owner) `pho001`, so the documented capable
  runner cannot be scheduled. Unblocking requires an eligible organization/enterprise owner,
  billing and positive Actions spending, repository access to `macos-26-xlarge`, and one real
  Synaptik provider run proving a usable Metal device and MPSGraph execution.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph))</span>
- A persistent self-hosted Mac is not an implicit fallback. GitHub's
  [secure-use reference](https://docs.github.com/en/actions/reference/security/secure-use#hardening-for-self-hosted-runners)
  says self-hosted runners should almost never execute public-repository pull-request code. Any
  ephemeral self-hosted or different-provider design requires a separate isolation, network,
  token, cleanup, and availability plan.<span data-glossary-navigation="true"> ([token glossary](../../../../glossary.md#term-token))</span>

If the provider cannot expose a usable [Metal](../../../../glossary.md#term-metal) [device](../../../../glossary.md#term-device), keep this task Blocked. Do not merge a green
portable plus [compile](../../../../glossary.md#compile)-only macOS workflow and call it Apple Silicon functional [CI](../../../../glossary.md#term-continuous-integration-ci).

## Proposed workflow contract after unblocking

Create one `.github/workflows/ci.yml` with no `strategy.matrix` and exactly two named jobs.

### Events, [concurrency](../../../../glossary.md#term-concurrency), and permissions

- Trigger `pull_request` targeting `main`, `push` to `main`, and input-free `workflow_dispatch`.
  Do not use `pull_request_target`, `workflow_run`, schedules, path filters, or tag/[release](../../../../glossary.md#term-release) events.
- Use workflow concurrency group
  `ci-${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}` with
  `cancel-in-progress: true` so stale billed [GPU](../../../../glossary.md#term-gpu) work is cancelled.
- Set workflow permissions to `contents: read`; every unspecified permission remains none.
- Checkout uses `persist-credentials: false`, `fetch-depth: 1`, no submodules, and no [LFS](../../../../glossary.md#continuous-integration-and-hosted-runners).
- Use no repository/environment secrets, OIDC, write token, package permission, deployment,
  attestation, release, or publication credential.<span data-glossary-navigation="true"> ([OIDC glossary](../../../../glossary.md#continuous-integration-and-hosted-runners); [token glossary](../../../../glossary.md#term-token); [publication glossary](../../../../glossary.md#term-publication))</span>

### [Immutable](../../../../glossary.md#term-immutable) tooling

Pin every action by full [commit](../../../../glossary.md#term-commit) SHA, retaining the reviewed [release](../../../../glossary.md#term-release) in a comment:

```yaml
- uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7
- uses: gradle/actions/wrapper-validation@9c971963bec38e04b3d30dcc455b5382be2fdbfb # v6.3.0
- uses: actions/setup-java@de7274f081f381c8f8158605e0321c36c376e2e6 # v6
```

Both jobs install exact Eclipse Temurin `26.0.2.1+1`; use architecture `x64` on Ubuntu and
[`aarch64`](../../../../glossary.md#term-aarch64) on macOS, plus `force-download: true`, `verify-signature: true`, and
`cache-jdk: false`. Do not set a setup-java [dependency](../../../../glossary.md#term-dependency) [cache](../../../../glossary.md#term-cache). Add the official [Gradle](../../../../glossary.md#term-gradle) 9.6.1 [binary](../../../../glossary.md#term-binary-descriptor-and-metadata)
[distribution](../../../../glossary.md#term-distribution) [checksum](../../../../glossary.md#term-checksum) to `gradle/wrapper/gradle-wrapper.properties` without changing the wrapper
version:

```properties
distributionSha256Sum=9c0f7faeeb306cb14e4279a3e084ca6b596894089a0638e68a07c945a32c9e14
```

Use only `./gradlew`; ignore the runner's preinstalled Gradle. Pin the functional job's
`DEVELOPER_DIR` to `/Applications/Xcode_26.6.app/Contents/Developer`. The hosted image itself cannot
be content-addressed, so record `ImageOS`, `ImageVersion`, `sw_vers`, `uname -m`, `xcodebuild
-version`, `xcrun --sdk macosx --show-sdk-version`, `java --version`, and `./gradlew --version`, then
fail unless the architecture is [arm64](../../../../glossary.md#native-packaging-and-platform-vocabulary), macOS major is 26, and the selected [SDK](../../../../glossary.md#term-sdk) is 26.x.

### Portable job

[Run](../../../../glossary.md#term-run) on `ubuntu-24.04` and perform, in order:

1. pinned checkout;
2. pinned [Gradle](../../../../glossary.md#term-gradle) wrapper [JAR](../../../../glossary.md#term-jar) [validation](../../../../glossary.md#term-validation);
3. exact signed Temurin 26 installation and version recording;
4. `./gradlew build`.

Do not set `SYNAPTIK_METAL_TEST_LIBRARY`, select either native-package Gradle task, invoke a native
script, or install platform tools. This [lane](../../../../glossary.md#term-lane) proves the ordinary portable [lifecycle](../../../../glossary.md#lifecycle) stays isolated.

### [Metal](../../../../glossary.md#term-metal) functional job

[Run](../../../../glossary.md#term-run) on `macos-26-xlarge`, depend on the portable job to avoid billed [GPU](../../../../glossary.md#term-gpu) work after an ordinary
failure, and use a [finite](../../../../glossary.md#term-finite) operational timeout only as a cost/safety ceiling, never as benchmark
evidence. Set fixed workspace/temporary absolute paths without repository input interpolation.
Then perform exactly:

```bash
./native/metal-macos-arm64/build.sh
/usr/bin/codesign --force --sign - --timestamp=none \
  --identifier io.github.pho001.synaptik.metal.foundation \
  native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./native/metal-macos-arm64/package-local.sh \
  native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./native/metal-macos-arm64/verify-package.sh \
  native/metal-macos-arm64/build/package-v1/macos-arm64
PACKAGE="$GITHUB_WORKSPACE/native/metal-macos-arm64/build/package-v1/macos-arm64"
./gradlew :backends:metal:metalNativeLocalZip --rerun-tasks \
  -PsynaptikMetalNativePackage="$PACKAGE"
mkdir -p "$RUNNER_TEMP/synaptik-metal-local"
/usr/bin/ditto -x -k \
  backends/metal/build/distributions/synaptik-metal-macos-arm64-local.zip \
  "$RUNNER_TEMP/synaptik-metal-local"
EXTRACTED="$RUNNER_TEMP/synaptik-metal-local/macos-arm64"
./native/metal-macos-arm64/verify-package.sh "$EXTRACTED"
LIBRARY="$EXTRACTED/libsynaptik_metal_foundation.dylib"
test -f "$LIBRARY" && test ! -L "$LIBRARY"
SYNAPTIK_METAL_TEST_LIBRARY="$LIBRARY" \
  ./gradlew :backends:metal:test --tests '*Metal*' --rerun-tasks
SYNAPTIK_METAL_TEST_LIBRARY="$LIBRARY" \
  ./gradlew :testing:integration-tests:test \
    --tests '*EngineExplicitCompositionMetalIntegrationTest' --rerun-tasks
SYNAPTIK_METAL_TEST_LIBRARY="$LIBRARY" \
  ./gradlew :tools:benchmarks:metalBenchmark -Pprofile=smoke --rerun-tasks
```

After each test command, require at least one matching [Gradle](../../../../glossary.md#term-gradle) [JUnit](../../../../glossary.md#term-junit) XML report and fail if any suite
reports a positive `skipped`, `failures`, or `errors` count. The existing environment assumptions
are allowed only to keep ordinary portable tests portable; once this [lane](../../../../glossary.md#term-lane) supplies the verified
regular [dylib](../../../../glossary.md#term-dylib), a missing Metal [device](../../../../glossary.md#term-device), context, [MPSGraph](../../../../glossary.md#term-mpsgraph) route, or execution must fail the job.
Never use `continue-on-error`, a device-dependent `if`, an assumption whitelist, or a success path
that omits either real test command.

The [smoke](../../../../glossary.md#term-smoke-test) benchmark is functional evidence only. It must complete its current fixed [singleton](../../../../glossary.md#term-singleton)-NEG
workload, attest both current routes, and preserve exact output facts. Do not compare timings,
select a winner, set a [regression](../../../../glossary.md#term-regression) threshold, repeat [forks](../../../../glossary.md#term-fork-meanings), retain a baseline, or infer production
policy from the hosted machine.

### [Cache](../../../../glossary.md#term-cache), [artifact](../../../../glossary.md#term-artifact), and [release](../../../../glossary.md#term-release) boundary

Do not configure Actions, [Gradle](../../../../glossary.md#term-gradle), wrapper, [dependency](../../../../glossary.md#term-dependency), [JDK](../../../../glossary.md#term-java-development-kit-jdk), native-output, or benchmark caches.
`org.gradle.caching=true` may reuse only the current ephemeral job's local state. This avoids
cross-[run](../../../../glossary.md#term-run) writes from untrusted pull-request code and keeps generated native bytes out of shared
storage.

Do not use `upload-artifact`. The [dylib](../../../../glossary.md#term-dylib), package, local ZIP, test reports, and benchmark output must
remain inside the ephemeral runner and be discarded after the job. GitHub retains ordinary [logs](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings)
under the repository's Actions retention setting; logs may contain public tool/image versions,
temporary paths, checksums, and descriptive [smoke](../../../../glossary.md#term-smoke-test) timings, but no credentials or release artifact.
The current package remains ad-hoc signed, unauthenticated local transport with no redistribution,
[provenance](../../../../glossary.md#provenance), [notarization](../../../../glossary.md#term-notarization), or release claim.

## Non-goals

- a strategy, performance, context, executable, profile, route, Shape, runner, JDK, Xcode, or OS
  matrix<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- a compile-only Apple-silicon success claim, device skip, conditional fallback, CPU substitute, or
  soft failure<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [device glossary](../../../../glossary.md#term-device); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- benchmark baseline/evidence profiles, timing gates, winner/default/cache decisions, retained
  reports, or hosted-machine performance claims<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache))</span>
- Developer ID, Team ID, keychain, secret, notarization, provenance, artifact attestation, upload,
  publication, coordinates, install, runtime discovery, or public release<span data-glossary-navigation="true"> ([notarization glossary](../../../../glossary.md#term-notarization); [provenance glossary](../../../../glossary.md#provenance); [artifact glossary](../../../../glossary.md#term-artifact); [publication glossary](../../../../glossary.md#term-publication); [runtime glossary](../../../../glossary.md#term-runtime); [release glossary](../../../../glossary.md#term-release))</span>
- a shared cache, dependency update, Gradle upgrade, wrapper regeneration, dependency locking, or
  dependency-verification project<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- a self-hosted runner, external CI provider, branch-protection setting, or organization/billing
  automation<span data-glossary-navigation="true"> ([hosted runner glossary](../../../../glossary.md#continuous-integration-and-hosted-runners); [CI glossary](../../../../glossary.md#term-continuous-integration-ci))</span>

## Contracts

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md) headings `Core invariants` and `Testing
  requirements` — [CI](../../../../glossary.md#term-continuous-integration-ci) validates current [ownership](../../../../glossary.md#term-ownership-meanings) and behavior without changing architecture.
- [`docs/architecture/contracts/backend-execution.md`](../../../../architecture/contracts/backend-execution.md)
  heading `Metal backend` — Metal remains explicitly configured by one absolute native-library
  path; no discovery or installation is added.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal))</span>
- [`docs/developer-guide/release-process.md`](../../../../developer-guide/release-process.md) — CI
  output remains verification, not publication or release.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication))</span>
- [Task 0045](0045-verified-local-native-package.md) — the functional lane builds, ad-hoc signs,
  packages, and verifies the exact reviewed package contract.<span data-glossary-navigation="true"> ([lane glossary](../../../../glossary.md#term-lane))</span>
- [Task 0046](0046-explicit-verified-native-local-archive.md) — the functional lane selects the
  explicit verifier-backed archive task and preserves manual extraction/reverification.
- [Task 0043](0043-reproducible-metal-route-benchmark.md) and
  [Task 0044](0044-custom-singleton-neg-benchmark-evaluation.md) — hosted smoke remains
  report-only, ineligible for a performance or production decision.<span data-glossary-navigation="true"> ([smoke glossary](../../../../glossary.md#term-smoke-test))</span>

If implementation needs a different provider trust model, secret, uploaded [artifact](../../../../glossary.md#term-artifact), [release](../../../../glossary.md#term-release)
[identity](../../../../glossary.md#term-identity-meanings), [cache](../../../../glossary.md#term-cache), performance threshold, test skip, or broader [matrix](../../../../glossary.md#term-matrix-meanings), stop and return this task to
[planning](../../../../glossary.md#term-planning).

## Dependencies and integration

- Depends on: [Metal](../../../../glossary.md#term-metal) 0046 Complete at independently approved implementation `4aad1ab6`; an enabled
  real GPU-capable hosted runner satisfying the provider gate above<span data-glossary-navigation="true"> ([GPU glossary](../../../../glossary.md#term-gpu); [hosted runner glossary](../../../../glossary.md#continuous-integration-and-hosted-runners))</span>
- Conflicts with: every concurrent CI/workflow, Gradle wrapper/toolchain, Metal native package or
  archive, benchmark protocol, native README, release-process, master, or roadmap edit<span data-glossary-navigation="true"> ([CI glossary](../../../../glossary.md#term-continuous-integration-ci); [Gradle glossary](../../../../glossary.md#term-gradle); [toolchain glossary](../../../../glossary.md#term-toolchain); [release glossary](../../../../glossary.md#term-release))</span>
- Parallel group: None
- Common base revision: `4aad1ab6ced318107e65bb9beef0013f8a7ff6e5`
- Integration order: no implementation is authorized while Blocked; after the external gate, sole
  CI/native-distribution frontier after 0046<span data-glossary-navigation="true"> ([distribution glossary](../../../../glossary.md#term-distribution))</span>
- Integration validation: actual hosted two-lane functional checkpoint followed by independent
  Class C review<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [lane glossary](../../../../glossary.md#term-lane))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): None while Blocked

## Files and responsibilities after unblocking

- `.github/workflows/ci.yml` — exact two-lane workflow, immutable actions, least privilege,
  concurrency, commands, and no-skip enforcement<span data-glossary-navigation="true"> ([lane glossary](../../../../glossary.md#term-lane); [immutable glossary](../../../../glossary.md#term-immutable); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- `gradle/wrapper/gradle-wrapper.properties` — official checksum for the unchanged Gradle 9.6.1
  binary distribution<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [distribution glossary](../../../../glossary.md#term-distribution))</span>
- `native/metal-macos-arm64/README.md` — hosted functional workflow and local/non-[release](../../../../glossary.md#term-release) boundary
- `docs/developer-guide/release-process.md` — [CI](../../../../glossary.md#term-continuous-integration-ci) verification versus [artifact](../../../../glossary.md#term-artifact) [publication](../../../../glossary.md#term-publication)
- this task, `../master-plan.md`, and `../../../roadmap.md` — authorization, blocker resolution,
  evidence, and status

Exactly these seven paths may change after the provider gate. This [planning](../../../../glossary.md#term-planning)-only blocker record
adds only this task and synchronizes Task 0046, the [Metal](../../../../glossary.md#term-metal) master plan, and the repository roadmap;
it adds no `.github` path, wrapper [checksum](../../../../glossary.md#term-checksum), executable, configuration, or generated artifact.

## Acceptance criteria after unblocking

1. The workflow contains exactly the portable and [Metal](../../../../glossary.md#term-metal) functional jobs, no `strategy.matrix`, and
   only the three full-SHA action pins and exact toolchain/checksum identities above.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([toolchain glossary](../../../../glossary.md#term-toolchain); [checksum glossary](../../../../glossary.md#term-checksum))</span>
2. Pull-request, main-push, and manual runs have read-only token permissions, no secrets, no
   privileged trigger, no shared cache, no uploaded artifact, and stale-run cancellation.<span data-glossary-navigation="true"> ([token glossary](../../../../glossary.md#term-token); [cache glossary](../../../../glossary.md#term-cache); [artifact glossary](../../../../glossary.md#term-artifact); [run glossary](../../../../glossary.md#term-run); [cancellation glossary](../../../../glossary.md#term-cancellation))</span>
3. The portable job passes `./gradlew build` without evaluating a native-package property or
   executing a native/archive task.
4. A real `macos-26-xlarge` run proves arm64 macOS 26, selected Xcode 26.6/SDK 26.x, exact Temurin
   26, Gradle 9.6.1, and a usable Metal device; provider or device absence fails.<span data-glossary-navigation="true"> ([arm64 glossary](../../../../glossary.md#native-packaging-and-platform-vocabulary); [SDK glossary](../../../../glossary.md#term-sdk); [Gradle glossary](../../../../glossary.md#term-gradle); [device glossary](../../../../glossary.md#term-device))</span>
5. Native build, fixed ad-hoc signing, package verification, Gradle ZIP creation, permission-
   preserving extraction, and extracted-package verification pass in the declared order.<span data-glossary-navigation="true"> ([ad-hoc signing glossary](../../../../glossary.md#term-ad-hoc-signing))</span>
6. The actual Metal and Engine integration commands execute against the extracted absolute dylib;
   JUnit XML exists and reports zero skipped, failures, and errors.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [dylib glossary](../../../../glossary.md#term-dylib); [JUnit glossary](../../../../glossary.md#term-junit))</span>
7. Exactly one smoke benchmark invocation succeeds for both current routes and exact correctness;
   no timing assertion, winner, baseline, matrix, or retained evidence is created.<span data-glossary-navigation="true"> ([smoke glossary](../../../../glossary.md#term-smoke-test); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
8. Documentation preserves the explicit-path, unauthenticated-local, no-release boundary; no
   generated file or path outside the seven-path scope remains.<span data-glossary-navigation="true"> ([release glossary](../../../../glossary.md#term-release); [scope glossary](../../../../glossary.md#term-scope))</span>
9. Local syntax/Markdown/scope checks and the actual hosted PR/main runs pass, then an independent
   Class C review approves provider security, permissions, pins, failure behavior, native lifecycle,
   cleanup, and release boundary with zero unresolved findings.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>

## [Validation](../../../../glossary.md#term-validation) while Blocked

Validate only links, Markdown headings/fences/final newline, status/frontier consistency,
[planning](../../../../glossary.md#term-planning)-only path [scope](../../../../glossary.md#term-scope), and `git diff --check`. Do not add or [run](../../../../glossary.md#term-run) a workflow, install a provider
runner, [compile](../../../../glossary.md#compile) [native code](../../../../glossary.md#native-numerical-library-and-instruction-architecture), invoke [Gradle](../../../../glossary.md#term-gradle), perform a [device](../../../../glossary.md#term-device) probe, upload an [artifact](../../../../glossary.md#term-artifact), or claim
local validation substitutes for GitHub scheduling and [GPU](../../../../glossary.md#term-gpu) execution.

## Blocker resolution

An authorized [owner](../../../../glossary.md#term-owner) must first make `macos-26-xlarge` available to this repository through an
eligible organization/enterprise plan, valid billing, a positive Actions spending limit, and
repository runner access. A one-[run](../../../../glossary.md#term-run) real Synaptik check must then prove that the assigned runner
returns a usable [Metal](../../../../glossary.md#term-metal) [device](../../../../glossary.md#term-device) and executes the current [MPSGraph](../../../../glossary.md#term-mpsgraph) path. Record the account/runner
availability without credentials, replace this status with [Ready](../../../../glossary.md#term-ready) in one [planning](../../../../glossary.md#term-planning)-only revision,
reconfirm the [immutable](../../../../glossary.md#term-immutable) action/[JDK](../../../../glossary.md#term-java-development-kit-jdk)/Xcode/[checksum](../../../../glossary.md#term-checksum) pins against authoritative sources, and only then
start implementation.

## Result

[Planning](../../../../glossary.md#term-planning)-only blocked record from clean revision
`4aad1ab6ced318107e65bb9beef0013f8a7ff6e5`. The smallest useful workflow is fully specified, but
current provider/account facts cannot execute its mandatory functional [lane](../../../../glossary.md#term-lane). No workflow,
configuration, [checksum](../../../../glossary.md#term-checksum), [cache](../../../../glossary.md#term-cache), [artifact](../../../../glossary.md#term-artifact), executable, test, or production file changed.
