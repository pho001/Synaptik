# Task 0046: Explicit Verified [Metal](../../../../glossary.md#term-metal) Native Local Archive

## Status

Complete

## Change class

Class C — the [runtime](../../../../glossary.md#term-runtime) [API](../../../../glossary.md#term-application-programming-interface-api) and native [ABI](../../../../glossary.md#term-application-binary-interface-abi) remain unchanged, but this task creates a durable [Gradle](../../../../glossary.md#term-gradle)
[handoff](../../../../glossary.md#term-handoff) for security-sensitive native executable bytes. Use one clean implementation context and
an independent targeted review before completion.

## Goal

Given one explicitly supplied absolute path to the final reviewed Task 0045 macOS-[arm64](../../../../glossary.md#native-packaging-and-platform-vocabulary) package,
rerun its [fail-closed](../../../../glossary.md#term-fail-closed) verifier and create one reproducible, unversioned local ZIP. Preserve the
caller-owned extraction [lifecycle](../../../../glossary.md#lifecycle) and the existing explicit absolute [dylib](../../../../glossary.md#term-dylib) path passed to
`MetalBackendConfiguration`.

## [Scope](../../../../glossary.md#term-scope)

- Add two lazily registered, opt-in tasks to `backends/metal`: `verifyMetalNativePackage` and
  `metalNativeLocalZip`.
- Require `-PsynaptikMetalNativePackage=/absolute/path` only when either task is selected.
- Reuse the Task 0045 verifier unchanged and archive only its exact three verified members.
- Document manual extraction, reverification, explicit loading, and the local unauthenticated
  boundary.
- Preserve ordinary [Gradle](../../../../glossary.md#term-gradle) [lifecycle](../../../../glossary.md#lifecycle) behavior and every [runtime](../../../../glossary.md#term-runtime)/native contract.

## Non-goals

- native build, signing, repair, normalization, install, extraction, discovery, download, cache, or
  runtime cleanup<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization); [cache glossary](../../../../glossary.md#term-cache); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Gradle distribution plugin, consumable configuration, outgoing artifact, publication,
  repository, coordinates, version, classifier contract, or lifecycle relationship<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [distribution glossary](../../../../glossary.md#term-distribution); [plugin glossary](../../../../glossary.md#term-plugin); [artifact glossary](../../../../glossary.md#term-artifact); [publication glossary](../../../../glossary.md#term-publication); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Developer ID, Team ID, credentials, keychain/CI secrets, notarization, provenance, CI, or public
  release<span data-glossary-navigation="true"> ([CI glossary](../../../../glossary.md#term-continuous-integration-ci); [notarization glossary](../../../../glossary.md#term-notarization); [provenance glossary](../../../../glossary.md#provenance); [release glossary](../../../../glossary.md#term-release))</span>
- Java, Config, loader, native source, ABI, schema, capability, dependency, test-source, settings,
  root-build, or architecture-contract changes<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [schema glossary](../../../../glossary.md#term-schema); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Intel macOS, another platform selector, or a reproducible cross-[toolchain](../../../../glossary.md#term-toolchain) [dylib](../../../../glossary.md#term-dylib) claim

## Contracts

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md) headings `Authority, incorporation, and
  precedence`, `Repository layout`, `Core invariants`, and `Testing requirements` — repository
  placement and proportionate verification remain unchanged.
- [`docs/architecture/contracts/backend-execution.md`](../../../../architecture/contracts/backend-execution.md)
  headings `Concrete backend modules` and `Metal backend` — Metal owns its bridge while callers
  continue to select one explicit absolute library path; Engine performs no discovery.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [Engine glossary](../../../../glossary.md#term-engine))</span>
- [`docs/developer-guide/release-process.md`](../../../../developer-guide/release-process.md) — the
  local archive must not invent release versioning, coordinates, publisher identity, provenance,
  redistribution rights, or publication.<span data-glossary-navigation="true"> ([coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [identity glossary](../../../../glossary.md#term-identity-meanings); [provenance glossary](../../../../glossary.md#provenance); [publication glossary](../../../../glossary.md#term-publication))</span>
- [Task 0045](0045-verified-local-native-package.md) headings `Package contract`, `Verification
  contract`, and `Follow-up` — this task consumes only the final reviewed exact package boundary
  and reuses its verifier.

If implementation needs another platform, package member, public workflow, install/[cache](../../../../glossary.md#term-cache) [owner](../../../../glossary.md#term-owner),
[release](../../../../glossary.md#term-release) mode, or [architecture contract](../../../../glossary.md#architecture-contract), stop and return this task to [planning](../../../../glossary.md#term-planning).

## Dependencies and integration

- Depends on: [Metal](../../../../glossary.md#term-metal) 0045 Complete at same-reviewer-approved remediation `26c6c911`
- Conflicts with: every concurrent Metal Gradle/native distribution, native README,
  release-process, master, or roadmap edit; native discovery, release-signing, notarization, or
  publication work<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [distribution glossary](../../../../glossary.md#term-distribution); [release glossary](../../../../glossary.md#term-release); [notarization glossary](../../../../glossary.md#term-notarization); [publication glossary](../../../../glossary.md#term-publication))</span>
- Parallel group: None
- Common base revision: `26c6c911`
- Integration order: sole Metal frontier after 0045
- Integration [validation](../../../../glossary.md#term-validation): Metal verified local archive [checkpoint](../../../../glossary.md#term-checkpoint)
- Shared-document integration [owner](../../../../glossary.md#term-owner): Main planner

## Files and responsibilities

- `backends/metal/build.gradle.kts` — [lazy](../../../../glossary.md#term-lazy-evaluation) explicit input, verifier task, and exact reproducible ZIP
- `native/metal-macos-arm64/README.md` — opt-in archive, manual extraction/reverification, and
  absolute loader workflow
- `docs/developer-guide/release-process.md` — local archive versus public [release](../../../../glossary.md#term-release) boundary
- this task, `../master-plan.md`, and `../../../roadmap.md` — authorization and status

Exactly these six repository paths may change. Generated archives and test reports remain in
ignored build directories; disposable negative fixtures remain outside the repository and are
removed.

## [Gradle](../../../../glossary.md#term-gradle) input and isolation contract

The sole input is `-PsynaptikMetalNativePackage=/absolute/path/to/macos-arm64`. It has no default,
environment fallback, repository [lookup](../../../../glossary.md#term-lookup), working-directory resolution, or platform discovery.
Blank and relative values fail. The exact supplied string reaches the existing verifier: Gradle
must not trim, normalize, call `toRealPath()`, canonicalize, or otherwise hide a trailing separator,
terminal dot, repeated separator, or symlink package root.

`verifyMetalNativePackage` is an `Exec` task with no outputs. It directly runs
`native/metal-macos-arm64/verify-package.sh` on every selection and inherits its exact [schema](../../../../glossary.md#term-schema),
[Mach-O](../../../../glossary.md#term-mach-o), [dependency](../../../../glossary.md#term-dependency), export, [signature](../../../../glossary.md#term-signature), mode, [checksum](../../../../glossary.md#term-checksum), path, and member checks. It does not call
`build.sh`, `package-local.sh`, or signing commands.

`metalNativeLocalZip` is one dedicated `Zip` task depending on `verifyMetalNativePackage`. Neither
task is related to `build`, `assemble`, `check`, `jar`, `test`, `clean`, [publication](../../../../glossary.md#term-publication), or another
[lifecycle](../../../../glossary.md#lifecycle) task. Configuring or running ordinary Gradle work without the property must not evaluate
it, inspect a package, or invoke native tools.

## Archive contract

The only output is
`backends/metal/build/distributions/synaptik-metal-macos-arm64-local.zip`. The fixed name is
unversioned and local; it must not inherit the root `0.1.0-SNAPSHOT` version.

The archive contains exactly:

```text
macos-arm64/
  libsynaptik_metal_foundation.dylib  0755
  manifest.json                       0644
  SHA256SUMS                          0644
```

Directory mode is `0755`. File order is reproducible and filesystem timestamps are not preserved.
No source path, [host](../../../../glossary.md#term-host), time, revision, product version, [SDK](../../../../glossary.md#term-sdk) version, Team ID, [provenance](../../../../glossary.md#provenance), legal
placeholder, or extra file is added. Platform selection is fixed to `macos-arm64`; the reused
verifier, not a new [Gradle](../../../../glossary.md#term-gradle) selector, authenticates that package [shape](../../../../glossary.md#shape).

## Extraction, loader, and legal boundary

[Gradle](../../../../glossary.md#term-gradle) installs or extracts nothing. The caller chooses a directory, uses a permission-preserving
extractor, reruns the same verifier on the extracted `macos-arm64` directory, retains/removes that
directory, and passes the extracted [dylib](../../../../glossary.md#term-dylib)'s absolute path to `MetalBackendConfiguration`.
`SYNAPTIK_METAL_TEST_LIBRARY` remains test-only.

The ZIP is a verified local development transport, not an authenticated or public [release](../../../../glossary.md#term-release).
Self-contained [hashes](../../../../glossary.md#term-hash) do not authenticate hostile replacement, and the ad-hoc [signature](../../../../glossary.md#term-signature) supplies
no publisher [identity](../../../../glossary.md#term-identity-meanings) or Apple trust. The repository has no project-wide license file;
`THIRD_PARTY_NOTICES.md` applies only to [CPU](../../../../glossary.md#term-cpu) [ERF](../../../../glossary.md#term-erf) material absent from this archive and must not be
copied into it. Do not invent redistribution rights or legal [metadata](../../../../glossary.md#term-metadata).

## Acceptance criteria

- Ordinary `:backends:metal:build` succeeds without the property and neither new task runs.
- Selecting either task without the property fails clearly; blank and relative inputs fail without
  creating a new archive.
- A symlink-root package and the Task 0045 redundant terminal path forms fail through the existing
  verifier; no path is canonicalized before that check.
- A valid absolute Task 0045 package is verified before archiving, and the task graph has no native
  build/sign or ordinary lifecycle edge.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Two forced ZIP executions over identical package bytes are byte-identical.
- The ZIP has exactly the three paths, fixed modes/order/timestamps, fixed local filename, and no
  source path or inferred version.
- Permission-preserving extraction followed by the unchanged verifier passes; a focused existing
  real-device Metal test loads the extracted dylib through an absolute path.<span data-glossary-navigation="true"> ([device glossary](../../../../glossary.md#term-device); [Metal glossary](../../../../glossary.md#term-metal); [dylib glossary](../../../../glossary.md#term-dylib))</span>
- Documentation preserves manual caller ownership and distinguishes integrity from authentication,
  notarization, publication, and release.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [integrity glossary](../../../../glossary.md#term-integrity); [authentication glossary](../../../../glossary.md#term-authentication); [notarization glossary](../../../../glossary.md#term-notarization); [publication glossary](../../../../glossary.md#term-publication); [release glossary](../../../../glossary.md#term-release))</span>
- No generated file or repository path outside the six-path [scope](../../../../glossary.md#term-scope) remains.

## [Validation](../../../../glossary.md#term-validation)

Worker validation:

```bash
./gradlew :backends:metal:build
./gradlew :backends:metal:verifyMetalNativePackage \
  -PsynaptikMetalNativePackage=/absolute/path/to/macos-arm64
./gradlew :backends:metal:metalNativeLocalZip --rerun-tasks \
  -PsynaptikMetalNativePackage=/absolute/path/to/macos-arm64
```

Also [run](../../../../glossary.md#term-run) missing, blank, relative, symlink-root, trailing-`/`, terminal-`/.`, and repeated-`/`
negative invocations; inspect the task [graph](../../../../glossary.md#graph) and exact ZIP entries/modes/timestamps; compare two
fresh archive [hashes](../../../../glossary.md#term-hash)/bytes; extract with `/usr/bin/ditto -x -k`; rerun `verify-package.sh`; and run
one focused existing real-[device](../../../../glossary.md#term-device) [Metal](../../../../glossary.md#term-metal) test with `SYNAPTIK_METAL_TEST_LIBRARY` set to the extracted
absolute [dylib](../../../../glossary.md#term-dylib) path. Run shell/Markdown/six-path/diff checks. Do not run a full repository build.

No permanent [Gradle](../../../../glossary.md#term-gradle) TestKit test is required: the repository has no such convention, and command
behavior plus the existing permanent package verifier directly exercise laziness, isolation, and
[artifact](../../../../glossary.md#term-artifact) semantics without a source-text wiring assertion.

## Follow-up

Developer ID signing, secure keychain/[CI](../../../../glossary.md#term-continuous-integration-ci) secret handling, notary credentials, an accepted
notarizable container/process, project license and notice policy, product versioning, [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix),
repository, [provenance](../../../../glossary.md#provenance), CI promotion, [rollback](../../../../glossary.md#term-rollback), support, and public [release](../../../../glossary.md#term-release) remain separately
blocked.

## Documentation and review impact

The native README and [release](../../../../glossary.md#term-release) guide changed a durable developer workflow and security terminology.
Independent Class C review inspected Provider laziness, exact raw path forwarding, task [graph](../../../../glossary.md#graph)
isolation, verifier ordering, archive members/modes/[reproducibility](../../../../glossary.md#artifact-manifest-and-source-provenance), absence of path/version/legal
leaks, manual extraction/loading, and absence of [authentication](../../../../glossary.md#term-authentication) or release claims.

## Result

Implemented at `4aad1ab6ced318107e65bb9beef0013f8a7ff6e5` from [planning](../../../../glossary.md#term-planning) [commit](../../../../glossary.md#term-commit) `35b222be`
and clean reviewed Task 0045 base `26c6c911`.
`backends/metal` now lazily registers only `verifyMetalNativePackage` and
`metalNativeLocalZip`. The exact raw absolute property reaches the existing verifier; the verifier
has no outputs and executes on every selection. The ZIP task depends only on it, copies the exact
three members under `macos-arm64`, and has no ordinary [lifecycle](../../../../glossary.md#lifecycle), build, signing, install,
discovery, configuration, [publication](../../../../glossary.md#term-publication), [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix), version, Java, or native-source effect.

Task listing and ordinary `:backends:metal:build` passed without the property. Missing-property
selection of both tasks and blank, relative, symlink-root, trailing-`/`, terminal-`/.`, and
repeated-`/` inputs failed while the prior archive stayed unchanged. The explicit ZIP dry-[run](../../../../glossary.md#term-run)
contained only the verifier and ZIP tasks. Valid verifier executions ran under both stored and
reused configuration [cache](../../../../glossary.md#term-cache) entries.

Two forced archives were byte-identical at [SHA-256](../../../../glossary.md#term-sha-256)
`82e569f96772addeaf661b6d9d6fb2651926d727d7dd16ab3ae299d9f6dd5655`.
The 37,426-byte ZIP has exactly the `macos-arm64/` directory plus the three package members, fixed
`0755` directory/[dylib](../../../../glossary.md#term-dylib) and `0644` [metadata](../../../../glossary.md#term-metadata) modes, one `1980-02-01 00:00:00` timestamp, and no
source path or `0.1.0-SNAPSHOT` text. Permission-preserving `ditto` extraction followed by the
unchanged verifier passed. The extracted absolute dylib path passed the existing real-[device](../../../../glossary.md#term-device)
`MetalMpsGraphMatmulNativeTest`: one test, zero skips, failures, or errors.

The native README and [release](../../../../glossary.md#term-release) guide document manual caller-owned extraction, reverification,
retention, and explicit absolute-path loading while retaining the unauthenticated ad-hoc local
boundary. Six-path [scope](../../../../glossary.md#term-scope), Markdown, and diff checks passed; no full repository build ran. The
independent Class C review returned `APPROVE` with zero findings. Status: Complete.
