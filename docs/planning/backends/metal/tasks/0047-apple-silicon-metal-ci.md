# Task 0047: Apple Silicon Metal CI

## Status

Blocked

The current public repository is owned by a personal account. GitHub documents the standard
`macos-26` arm64 runner as M1 hardware but does not document Metal GPU acceleration there, and the
standard M1 runner history records unavailable MPS hardware. The required `macos-26-xlarge` runner
is the documented arm64 M2 option with eight-GPU hardware acceleration, but GitHub makes larger
runners available only to eligible organizations and enterprises with billing and a positive
spending limit. No workflow may be added until a real GPU-capable hosted runner is enabled for this
repository. A compile-only macOS lane, skipped Metal tests, or conditional device fallback does not
satisfy this task. No Metal task is Ready.

## Change class

Class C — a future workflow will execute pull-request code, download pinned toolchains, and build,
sign, package, load, and execute native bytes on a hosted GPU. After the provider gate is resolved,
use one clean implementation context and an independent security- and lifecycle-focused review
before completion.

## Goal

Add the smallest useful continuous-integration boundary: one ordinary portable Java lane and one
Apple-silicon macOS functional lane. The functional lane must build and ad-hoc sign the current
native library, create and verify the reviewed local package and archive, load the extracted dylib
through the explicit absolute path, run the actual Metal and Engine integration coverage without
skips, and run one fixed report-only benchmark smoke. It must not create a performance, context,
executable, profile, or Shape matrix.

## Provider gate and evidence

- GitHub's [hosted-runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
  lists standard `macos-26` as arm64 M1 with three CPU cores, 7 GB RAM, and 14 GB storage for public
  and private repositories. It does not promise GPU acceleration.
- The standard-M1 [runner announcement](https://github.com/actions/runner-images/issues/9254)
  records that MPS hardware was unavailable, and the transferred
  [Metal support discussion](https://github.com/orgs/community/discussions/160669) remains
  unanswered. Standard `macos-26` may compile and inspect Mach-O bytes, but it is not evidence for
  functional Metal execution.
- GitHub's [larger-runner reference](https://docs.github.com/en/actions/reference/runners/larger-runners)
  lists `macos-26-xlarge` as arm64 M2 with five CPU cores and eight-GPU hardware acceleration.
  GitHub's [hosted-runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners#larger-runners)
  restricts larger runners to organizations and enterprises on Team or Enterprise Cloud plans.
  [Actions pricing](https://docs.github.com/en/billing/reference/actions-runner-pricing) additionally
  requires billing; included minutes do not apply, and the current macOS M2 rate is USD 0.102 per
  minute.
- The repository is currently public under personal owner `pho001`, so the documented capable
  runner cannot be scheduled. Unblocking requires an eligible organization/enterprise owner,
  billing and positive Actions spending, repository access to `macos-26-xlarge`, and one real
  Synaptik provider run proving a usable Metal device and MPSGraph execution.
- A persistent self-hosted Mac is not an implicit fallback. GitHub's
  [secure-use reference](https://docs.github.com/en/actions/reference/security/secure-use#hardening-for-self-hosted-runners)
  says self-hosted runners should almost never execute public-repository pull-request code. Any
  ephemeral self-hosted or different-provider design requires a separate isolation, network,
  token, cleanup, and availability plan.

If the provider cannot expose a usable Metal device, keep this task Blocked. Do not merge a green
portable plus compile-only macOS workflow and call it Apple Silicon functional CI.

## Proposed workflow contract after unblocking

Create one `.github/workflows/ci.yml` with no `strategy.matrix` and exactly two named jobs.

### Events, concurrency, and permissions

- Trigger `pull_request` targeting `main`, `push` to `main`, and input-free `workflow_dispatch`.
  Do not use `pull_request_target`, `workflow_run`, schedules, path filters, or tag/release events.
- Use workflow concurrency group
  `ci-${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}` with
  `cancel-in-progress: true` so stale billed GPU work is cancelled.
- Set workflow permissions to `contents: read`; every unspecified permission remains none.
- Checkout uses `persist-credentials: false`, `fetch-depth: 1`, no submodules, and no LFS.
- Use no repository/environment secrets, OIDC, write token, package permission, deployment,
  attestation, release, or publication credential.

### Immutable tooling

Pin every action by full commit SHA, retaining the reviewed release in a comment:

```yaml
- uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7
- uses: gradle/actions/wrapper-validation@9c971963bec38e04b3d30dcc455b5382be2fdbfb # v6.3.0
- uses: actions/setup-java@de7274f081f381c8f8158605e0321c36c376e2e6 # v6
```

Both jobs install exact Eclipse Temurin `26.0.2.1+1`; use architecture `x64` on Ubuntu and
`aarch64` on macOS, plus `force-download: true`, `verify-signature: true`, and
`cache-jdk: false`. Do not set a setup-java dependency cache. Add the official Gradle 9.6.1 binary
distribution checksum to `gradle/wrapper/gradle-wrapper.properties` without changing the wrapper
version:

```properties
distributionSha256Sum=9c0f7faeeb306cb14e4279a3e084ca6b596894089a0638e68a07c945a32c9e14
```

Use only `./gradlew`; ignore the runner's preinstalled Gradle. Pin the functional job's
`DEVELOPER_DIR` to `/Applications/Xcode_26.6.app/Contents/Developer`. The hosted image itself cannot
be content-addressed, so record `ImageOS`, `ImageVersion`, `sw_vers`, `uname -m`, `xcodebuild
-version`, `xcrun --sdk macosx --show-sdk-version`, `java --version`, and `./gradlew --version`, then
fail unless the architecture is arm64, macOS major is 26, and the selected SDK is 26.x.

### Portable job

Run on `ubuntu-24.04` and perform, in order:

1. pinned checkout;
2. pinned Gradle wrapper JAR validation;
3. exact signed Temurin 26 installation and version recording;
4. `./gradlew build`.

Do not set `SYNAPTIK_METAL_TEST_LIBRARY`, select either native-package Gradle task, invoke a native
script, or install platform tools. This lane proves the ordinary portable lifecycle stays isolated.

### Metal functional job

Run on `macos-26-xlarge`, depend on the portable job to avoid billed GPU work after an ordinary
failure, and use a finite operational timeout only as a cost/safety ceiling, never as benchmark
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

After each test command, require at least one matching Gradle JUnit XML report and fail if any suite
reports a positive `skipped`, `failures`, or `errors` count. The existing environment assumptions
are allowed only to keep ordinary portable tests portable; once this lane supplies the verified
regular dylib, a missing Metal device, context, MPSGraph route, or execution must fail the job.
Never use `continue-on-error`, a device-dependent `if`, an assumption whitelist, or a success path
that omits either real test command.

The smoke benchmark is functional evidence only. It must complete its current fixed singleton-NEG
workload, attest both current routes, and preserve exact output facts. Do not compare timings,
select a winner, set a regression threshold, repeat forks, retain a baseline, or infer production
policy from the hosted machine.

### Cache, artifact, and release boundary

Do not configure Actions, Gradle, wrapper, dependency, JDK, native-output, or benchmark caches.
`org.gradle.caching=true` may reuse only the current ephemeral job's local state. This avoids
cross-run writes from untrusted pull-request code and keeps generated native bytes out of shared
storage.

Do not use `upload-artifact`. The dylib, package, local ZIP, test reports, and benchmark output must
remain inside the ephemeral runner and be discarded after the job. GitHub retains ordinary logs
under the repository's Actions retention setting; logs may contain public tool/image versions,
temporary paths, checksums, and descriptive smoke timings, but no credentials or release artifact.
The current package remains ad-hoc signed, unauthenticated local transport with no redistribution,
provenance, notarization, or release claim.

## Non-goals

- a strategy, performance, context, executable, profile, route, Shape, runner, JDK, Xcode, or OS
  matrix
- a compile-only Apple-silicon success claim, device skip, conditional fallback, CPU substitute, or
  soft failure
- benchmark baseline/evidence profiles, timing gates, winner/default/cache decisions, retained
  reports, or hosted-machine performance claims
- Developer ID, Team ID, keychain, secret, notarization, provenance, artifact attestation, upload,
  publication, coordinates, install, runtime discovery, or public release
- a shared cache, dependency update, Gradle upgrade, wrapper regeneration, dependency locking, or
  dependency-verification project
- a self-hosted runner, external CI provider, branch-protection setting, or organization/billing
  automation

## Contracts

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md) headings `Core invariants` and `Testing
  requirements` — CI validates current ownership and behavior without changing architecture.
- [`docs/architecture/contracts/backend-execution.md`](../../../../architecture/contracts/backend-execution.md)
  heading `Metal backend` — Metal remains explicitly configured by one absolute native-library
  path; no discovery or installation is added.
- [`docs/developer-guide/release-process.md`](../../../../developer-guide/release-process.md) — CI
  output remains verification, not publication or release.
- [Task 0045](0045-verified-local-native-package.md) — the functional lane builds, ad-hoc signs,
  packages, and verifies the exact reviewed package contract.
- [Task 0046](0046-explicit-verified-native-local-archive.md) — the functional lane selects the
  explicit verifier-backed archive task and preserves manual extraction/reverification.
- [Task 0043](0043-reproducible-metal-route-benchmark.md) and
  [Task 0044](0044-custom-singleton-neg-benchmark-evaluation.md) — hosted smoke remains
  report-only, ineligible for a performance or production decision.

If implementation needs a different provider trust model, secret, uploaded artifact, release
identity, cache, performance threshold, test skip, or broader matrix, stop and return this task to
planning.

## Dependencies and integration

- Depends on: Metal 0046 Complete at independently approved implementation `4aad1ab6`; an enabled
  real GPU-capable hosted runner satisfying the provider gate above
- Conflicts with: every concurrent CI/workflow, Gradle wrapper/toolchain, Metal native package or
  archive, benchmark protocol, native README, release-process, master, or roadmap edit
- Parallel group: None
- Common base revision: `4aad1ab6ced318107e65bb9beef0013f8a7ff6e5`
- Integration order: no implementation is authorized while Blocked; after the external gate, sole
  CI/native-distribution frontier after 0046
- Integration validation: actual hosted two-lane functional checkpoint followed by independent
  Class C review
- Shared-document integration owner: None while Blocked

## Files and responsibilities after unblocking

- `.github/workflows/ci.yml` — exact two-lane workflow, immutable actions, least privilege,
  concurrency, commands, and no-skip enforcement
- `gradle/wrapper/gradle-wrapper.properties` — official checksum for the unchanged Gradle 9.6.1
  binary distribution
- `native/metal-macos-arm64/README.md` — hosted functional workflow and local/non-release boundary
- `docs/developer-guide/release-process.md` — CI verification versus artifact publication
- this task, `../master-plan.md`, and `../../../roadmap.md` — authorization, blocker resolution,
  evidence, and status

Exactly these seven paths may change after the provider gate. This planning-only blocker record
adds only this task and synchronizes Task 0046, the Metal master plan, and the repository roadmap;
it adds no `.github` path, wrapper checksum, executable, configuration, or generated artifact.

## Acceptance criteria after unblocking

1. The workflow contains exactly the portable and Metal functional jobs, no `strategy.matrix`, and
   only the three full-SHA action pins and exact toolchain/checksum identities above.
2. Pull-request, main-push, and manual runs have read-only token permissions, no secrets, no
   privileged trigger, no shared cache, no uploaded artifact, and stale-run cancellation.
3. The portable job passes `./gradlew build` without evaluating a native-package property or
   executing a native/archive task.
4. A real `macos-26-xlarge` run proves arm64 macOS 26, selected Xcode 26.6/SDK 26.x, exact Temurin
   26, Gradle 9.6.1, and a usable Metal device; provider or device absence fails.
5. Native build, fixed ad-hoc signing, package verification, Gradle ZIP creation, permission-
   preserving extraction, and extracted-package verification pass in the declared order.
6. The actual Metal and Engine integration commands execute against the extracted absolute dylib;
   JUnit XML exists and reports zero skipped, failures, and errors.
7. Exactly one smoke benchmark invocation succeeds for both current routes and exact correctness;
   no timing assertion, winner, baseline, matrix, or retained evidence is created.
8. Documentation preserves the explicit-path, unauthenticated-local, no-release boundary; no
   generated file or path outside the seven-path scope remains.
9. Local syntax/Markdown/scope checks and the actual hosted PR/main runs pass, then an independent
   Class C review approves provider security, permissions, pins, failure behavior, native lifecycle,
   cleanup, and release boundary with zero unresolved findings.

## Validation while Blocked

Validate only links, Markdown headings/fences/final newline, status/frontier consistency,
planning-only path scope, and `git diff --check`. Do not add or run a workflow, install a provider
runner, compile native code, invoke Gradle, perform a device probe, upload an artifact, or claim
local validation substitutes for GitHub scheduling and GPU execution.

## Blocker resolution

An authorized owner must first make `macos-26-xlarge` available to this repository through an
eligible organization/enterprise plan, valid billing, a positive Actions spending limit, and
repository runner access. A one-run real Synaptik check must then prove that the assigned runner
returns a usable Metal device and executes the current MPSGraph path. Record the account/runner
availability without credentials, replace this status with Ready in one planning-only revision,
reconfirm the immutable action/JDK/Xcode/checksum pins against authoritative sources, and only then
start implementation.

## Result

Planning-only blocked record from clean revision
`4aad1ab6ced318107e65bb9beef0013f8a7ff6e5`. The smallest useful workflow is fully specified, but
current provider/account facts cannot execute its mandatory functional lane. No workflow,
configuration, checksum, cache, artifact, executable, test, or production file changed.
