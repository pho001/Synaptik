# Task Engine0020: Mechanical [backend](../../../../glossary.md#backend) and native [layout](../../../../glossary.md#layout) cutover

## Status

Complete — integrated and validated on 2026-10-16 after independent review with no P1/P2.
User edits are preserved; combined tests/[Javadoc](../../../../glossary.md#term-javadoc) passed. No later [API](../../../../glossary.md#term-application-programming-interface-api) phase is authorized.

## Change class

Class C — coordinated repository-[layout](../../../../glossary.md#layout) authority and multi-project/native path consumers.
No numerical, algorithm, [ABI](../../../../glossary.md#term-application-binary-interface-abi), [dependency](../../../../glossary.md#term-dependency)-direction or resource-[lifecycle](../../../../glossary.md#lifecycle) change.

## Goal

Relocate [concrete backend](../../../../glossary.md#concrete-backend) projects and [Metal](../../../../glossary.md#term-metal) native sources under [modules](../../../../glossary.md#term-module)/[backend](../../../../glossary.md#backend) while preserving
all [Gradle](../../../../glossary.md#term-gradle) project [identities](../../../../glossary.md#term-identity-meanings), Java packages, [kernel](../../../../glossary.md#kernel) bytes, [ABI](../../../../glossary.md#term-application-binary-interface-abi) and current behavior.

## [Scope](../../../../glossary.md#term-scope) and non-goals

First update the root and owning scoped [layout](../../../../glossary.md#layout)/path descriptions, then move files in the same
reviewed change and repair path consumers. Do not apply the new layout without its coordinated
authority. Keep historical plans in place and completed outcomes unchanged.

| Current | [Target](../../../../glossary.md#term-target) |
|---|---|
| [backends](../../../../glossary.md#backend)/cpu | [modules](../../../../glossary.md#term-module)/backend/cpu |
| backends/[metal](../../../../glossary.md#term-metal) | modules/backend/metal |
| backends/cuda | modules/backend/cuda |
| backends/openblas-provider | modules/backend/openblas-provider |
| native/metal-macos-[arm64](../../../../glossary.md#native-packaging-and-platform-vocabulary) | modules/backend/metal/macos-arm64 (no native/ nesting) |
| native/cuda/README.md | Merge scaffold text into modules/backend/cuda/README.md; retire the old README. |

The table maps physical directories only. Preserve :backends:cpu, :backends:metal, :backends:cuda
and :backends:openblas-provider using explicit settings projectDir mappings. Also map the implicit
parent :backends to modules/backend: [Gradle](../../../../glossary.md#term-gradle) 9.6.1 rejects its nonexistent default backends directory
after a clean relocation. Do not leave an old directory to [mask](../../../../glossary.md#term-boolean-mask) a missing mapping. Keep
backend-contract and [Engine](../../../../glossary.md#term-engine) where they are. No empty [CUDA](../../../../glossary.md#term-cuda) platform tree, executable CUDA, Training/[Checkpoint](../../../../glossary.md#term-checkpoint) move,
[dependency](../../../../glossary.md#term-dependency) inversion, [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [API](../../../../glossary.md#term-application-programming-interface-api), [compiler](../../../../glossary.md#term-compiler)/[runtime](../../../../glossary.md#term-runtime) change, renaming packages or numerical/[schema](../../../../glossary.md#term-schema) bump.
No broad performance/certification campaign, old report regeneration or unrelated dirty-edit cleanup.

## Contracts

- [Root](../../../../../ARCHITECTURE.md): `Authority, incorporation, and precedence`, `Java namespace`,
  `Repository layout`, `Core lifecycle`, `Module-ownership routing`, `Dependency rules`,
  `Explicit non-goals`, `Testing requirements`. Update physical references, not semantic rules.
- [Backend execution](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend))</span>:
  `Concrete backend modules`, `Metal backend`, `OpenBLAS provider`; path-only coordinated update.
- [Engine](../../../../architecture/contracts/runtime-prepare-engine.md#public-explicit-composition)<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>:
  `modules/engine`, `Public explicit composition`, `Compile ownership and mixed-owner Prepare routing`;
  update backend path references only, retain public composition/dependency visibility.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Foundational](../../../../architecture/contracts/foundational-modules.md#modulesbackend-contract):
  `modules/backend-contract`; unchanged semantic boundary, not the new physical parent directory.

If a path repair would alter [ABI](../../../../glossary.md#term-application-binary-interface-abi), [kernel](../../../../glossary.md#kernel) bytes, [lifecycle](../../../../glossary.md#lifecycle) or [dependency direction](../../../../glossary.md#java-platform-and-build-tools), stop and report
the exact conflict instead of [widening](../../../../glossary.md#term-widening) this mechanical cutover.

## Dependencies and integration

- Depends on: CP0005D Complete (bounded planning/review only), not acceptance of all ADR0039
  API choices; user-approved backend/native physical layout.
  Existing backend/Engine baseline is Complete; no NN API or generic-module decision dependency.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [layout glossary](../../../../glossary.md#layout); [Engine glossary](../../../../glossary.md#term-engine); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Conflicts with: every writer under backends/native, settings/build path inputs, architecture
  contracts/tests and shared current documentation; serialize and preserve pre-existing dirty edits.
- Parallel group: None.
- Common base revision: `9e7e8aa15e0833a1bc91263a7add99a80d88cd92`; reverify exact launch revision
  and integrated planning packet at launch, never assume another worktree's cleanliness.
- Integration order: integrate reviewed packet → launch-state revalidation → coordinated path
  authority and moves/consumers → independent implementation review → one combined checkpoint.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Integration [validation](../../../../glossary.md#term-validation): commands below; native package [smoke](../../../../glossary.md#term-smoke-test) when available with exact skips.
- Shared-document integration owner: model-execution coordinator with Engine/backend owners;
  one executor receives exclusive authority/path-consumer write scope for this cutover.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [model glossary](../../../../glossary.md#term-model); [scope glossary](../../../../glossary.md#term-scope))</span>

Launch verified by coordinator on 2026-10-16 10:31 UTC: [Ready](../../../../glossary.md#term-ready) at the sole [authorized frontier](../../../../glossary.md#term-authorized-frontier),
CP0005D Complete, exact common base above, no conflicting writer. The implementation worktree
`/private/tmp/synaptik-checkpoint-leaf.oCW897` was clean before the nine reviewed [planning](../../../../glossary.md#term-planning) documents
were seeded. Main's notices/[oracle](../../../../glossary.md#term-test-oracle) edits remained integration-owned; no later [API](../../../../glossary.md#term-application-programming-interface-api) choice was authorized.

## Files and symbols

- settings.[gradle](../../../../glossary.md#term-gradle).kts: retain include [identities](../../../../glossary.md#term-identity-meanings); explicitly map the parent and all four children:
  `project(":backends").projectDir = file("modules/backend")`;
  `project(":backends:cpu").projectDir = file("modules/backend/cpu")`;
  `project(":backends:metal").projectDir = file("modules/backend/metal")`;
  `project(":backends:cuda").projectDir = file("modules/backend/cuda")`;
  `project(":backends:openblas-provider").projectDir = file("modules/backend/openblas-provider")`.
- Existing [backend](../../../../glossary.md#backend) trees, native/[metal](../../../../glossary.md#term-metal)-macos-[arm64](../../../../glossary.md#native-packaging-and-platform-vocabulary) and [CUDA](../../../../glossary.md#term-cuda) README: moves/README merge only.
- backends/metal/build.gradle.kts: root-relative verifier path; root/shared build scripts and
  testing/architecture-tests/build.gradle.kts: source-input globs and API-consumer path assumptions.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- EngineCompositionContractTest, CpuDependencyContractTest and other actual path-consuming
  architecture tests: semantic role stays backend after move; do not grant arbitrary modules an
  Engine dependency or enable later Training/Checkpoint edges in this task.<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module); [Engine glossary](../../../../glossary.md#term-engine); [dependency glossary](../../../../glossary.md#term-dependency); [Checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Native build.sh/package-local.sh/verify-package.sh, evidence tools with parents[n], relative
  include/library paths and source manifests: repair paths only where they are executable consumers.
  Preserve evidence contents/results, and do not replay historical proof campaigns.
- Root/scoped headings above; current architecture/module/dependency/native-build explanations,
  affected README/API links, ADR0039 and owner master/roadmap path references. Historical task
  prose is not rewritten en masse; document old→new mapping in current navigation.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>

No new/moved Java type package: every production type retains its existing fully qualified name.
Before moving, inventory tracked/untracked files and any user edits; preserve their bytes in the
new location. The worker must not import or overwrite unrelated main-checkout user edits into its
clean baseline. Coordinator integration must intentionally preserve those edits at their retained
or moved destinations, including notices and the [CPU](../../../../glossary.md#term-cpu) [oracle](../../../../glossary.md#term-test-oracle). Integrating the approved [planning](../../../../glossary.md#term-planning)
packet is allowed. Never reset/clean the worktree or treat a dirty file as disposable.

## Acceptance criteria

- All four backend directories and Metal native tree are at the exact targets, with no redundant
  native nesting. CUDA README content is retained in the backend README; no new CUDA capability.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [Metal glossary](../../../../glossary.md#term-metal); [targets glossary](../../../../glossary.md#term-target); [CUDA glossary](../../../../glossary.md#term-cuda))</span>
- Gradle identities, dependency configurations, Java packages, native source/header bytes,
  library/symbol names, generated schema/numerical identity and algorithm/lifecycle remain unchanged.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [symbol glossary](../../../../glossary.md#term-symbol); [schema glossary](../../../../glossary.md#term-schema); [identity glossary](../../../../glossary.md#term-identity-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Settings/project discovery, architecture path checks, API-consumer inputs, Metal verifier,
  active shell/Python evidence root calculations and current links use valid paths.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Authority, focused explanations and ADR record path-only cutover in the same change. Preserve
  completed history; distinguish historical paths from live commands. No broadened dependency rules.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- Focused/module and combined validation pass or name precise environmental limitations; any
  unavailable Metal device run is a skip, not successful device evidence.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [validation glossary](../../../../glossary.md#term-validation); [device glossary](../../../../glossary.md#term-device); [run glossary](../../../../glossary.md#term-run))</span>

## [Validation](../../../../glossary.md#term-validation)

Executor inventories relocation pairs and compares tracked source/native [hashes](../../../../glossary.md#term-hash) before/after;
allowed differences are reviewed path consumers/docs only, never [kernels](../../../../glossary.md#kernel) or Java behavior.
[Run](../../../../glossary.md#term-run) projects from the clean relocated checkout after confirming no old [backends](../../../../glossary.md#backend) directory remains;
a stale directory must not conceal an unmapped implicit parent. Record that absence with the
project-discovery result; do not delete unrelated files merely to satisfy this check.

```bash
git diff --check
./gradlew projects --console=plain
./gradlew :testing:architecture-tests:test --max-workers=2 --console=plain
./gradlew :backends:cpu:test :backends:metal:test :backends:cuda:test :backends:openblas-provider:test --max-workers=2 --console=plain
```

Validate current Markdown links and each changed executable path consumer. On supported macOS,
run the relocated build/package/verification workflow using the existing README's [arguments](../../../../glossary.md#term-argument) and
one bounded source-matched [Metal](../../../../glossary.md#term-metal) load/execution [smoke](../../../../glossary.md#term-smoke-test); preserve [ABI](../../../../glossary.md#term-application-binary-interface-abi) and do not regenerate numerical
certificates. Record exact commands/options and unavailable native prerequisites in the result.

Integration [owner](../../../../glossary.md#term-owner) runs `./gradlew test javadoc --max-workers=2 --console=plain` once after final
integration because shared paths/build/architecture change. Reuse successful worker evidence and
[Gradle](../../../../glossary.md#term-gradle) up-to-date work; independent review does not repeat successful suites. Java 26 is required;
coordinator reports 26.0.1 available but executor verifies its actual environment at launch.

Executed environment: Java 26.0.1, macOS 26.7.1 [arm64](../../../../glossary.md#native-packaging-and-platform-vocabulary); native commands used
`DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer` (MacOSX27.0.sdk), not default CLT.
Successful Gradle commands added `--no-daemon --offline` and used worktree-local
`GRADLE_USER_HOME="$PWD/build/engine0020/gradle-user-home"` and `TMPDIR="$PWD/build/engine0020/tmp"`
after shared-[cache](../../../../glossary.md#term-cache)/sandbox-socket restrictions; a local-socket exception enabled validation.
Native build/sign/package/verify used the relocated README's commands, with fixed signing arguments
`--force --sign - --timestamp=none --identifier io.github.pho001.synaptik.metal.foundation`.
The bounded invocation combined `:backends:metal:verifyMetalNativePackage
:backends:metal:metalNativeLocalZip :backends:metal:test`, `--max-workers=2 --console=plain`,
`-PsynaptikMetalNativePackage="$PWD/modules/backend/metal/macos-arm64/build/package-v1/macos-arm64"`,
and `SYNAPTIK_METAL_TEST_LIBRARY="$PWD/modules/backend/metal/macos-arm64/build/package-v1/macos-arm64/libsynaptik_metal_foundation.dylib"`.
Exact expanded native commands and six `--tests` filters remain in the frozen local
`build/engine0020/in-place-path-consumers.patch`; [logs](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings)/XML and inventories are alongside it.

## Documentation and review impact

Required independent targeted Class C review: exact move diff, path consumers, unchanged semantic
contracts, current docs/links and glossary path references. No new domain term. Java [Javadoc](../../../../glossary.md#term-javadoc) remains
accurate unless it contains moved file paths; review those, do not add behavioral prose for a move.
Use general/[planning](../../../../glossary.md#term-planning)/decision and relevant architecture/[backend](../../../../glossary.md#backend)/developer profiles when editing.

## Result

- Implemented exclusively in the designated worktree; 682 tracked relocations: 652 byte-identical, 30 path/doc/tool deltas; all modes preserved.
- Pre-move inventory: 2,822 ignored, zero untracked; immediate move preserved all 3,504 files. Exact inventories/deltas: `build/engine0020/{before.json,pure-moves.tsv,moved-path-consumers.tsv,changed.json,in-place-changes.json,*path-consumers.patch}`.
- Parent/four child mappings, consumers, architecture tests, coordinated authority/docs and [CUDA](../../../../glossary.md#term-cuda) README merge completed; old roots absent. All 495 Java packages, 16 native sources/headers, [ABI](../../../../glossary.md#term-application-binary-interface-abi)/[schema](../../../../glossary.md#term-schema)/numerics/lifetimes/dependencies preserved; 12 source differences are reviewed path consumers/[Javadoc](../../../../glossary.md#term-javadoc) only.
- Source-[manifest](../../../../glossary.md#term-manifest) [SHA-256](../../../../glossary.md#term-sha-256) before: `422321299d305303c6edf8b4e7f0b5eaf6d6b7572417338cecd94244dccafd9e`.
- Source-manifest SHA-256 after: `37358f899f8492a58c818de5e3ffd53c1a2c6d57005f515b710616b06ae8dbe7` (original-path-normalized order/modes).
- Unchanged native-source aggregate SHA-256: `6bcc9408c68a36c12fb52ffd00442e4615c757f0eb75bf5970ea6c89129016f3`; 112 archived non-tool evidence files unchanged.
- Fresh Oct16 evidence: `projects` and architecture 11/11 passed; [CPU](../../../../glossary.md#term-cpu) 1,075 tests, zero failures, 30 opt-in evidence/performance skips; [OpenBLAS](../../../../glossary.md#term-openblas) 50/50; CUDA `NO-SOURCE`.
- Initial four-[backend](../../../../glossary.md#backend) command above stopped at [Metal](../../../../glossary.md#term-metal): 356 tests, four unset-library `Path.of(null)` failures, 134 missing-library skips; the other [targets](../../../../glossary.md#term-target) then passed via `./gradlew :backends:cpu:test :backends:cuda:test :backends:openblas-provider:test --max-workers=2 --console=plain` plus common options.
- One source-matched [device](../../../../glossary.md#term-device) invocation above: six tests passed, zero skips, including all four failed-unconfigured [EXP](../../../../glossary.md#term-expx)/[SIGMOID](../../../../glossary.md#term-sigmoid) cases; full Metal suite was not repeated. Exact counts/skips/timestamps: `build/engine0020/fresh-test-summary.json`.
- Native build/sign/package/verify, [Gradle](../../../../glossary.md#term-gradle) package/ZIP, permission-preserving `ditto -x -k` extraction and extracted-package verification passed: ABI7/schema20, thirteen exports, minimum macOS26.0.
- [Dylib](../../../../glossary.md#term-dylib) SHA-256: `03cba3ebfa0bbb78c1bff369ff3451c591846fd975c0034fa7f67fdd2209985a`.
- ZIP SHA-256: `2405a9bbd453ea9fb570ff81e8f66930f48456928485d79b4c4215831e0f5915`.
- `git diff --check`, 15 script path/syntax checks, 1,000 current links, relocated `generate-task0053-header.py --check`, and `xcrun clang -std=c17 -Wall -Wextra -Werror -fsyntax-only modules/backend/metal/evidence/0053/model/task0053_model.c` passed.
- Task0053 `tools/write_manifest.py --check` is historical, not renewed; no proof/[certificate](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions) regeneration, all-word checker, five-[fork](../../../../glossary.md#term-fork-meanings)/performance campaign, or worker full-repository suite.
- Documentation/Javadoc: current paths/navigation updated; semantics remain accurate, no new glossary term. Independent read-only review: no P1/P2; 1,082 links/38 docs and preservation verified.
- Integrated via main's original-directory moves and reviewed deltas: [oracle](../../../../glossary.md#term-test-oracle) SHA-256 `57ce4dc701a83d2ed01b96c725197a72f5aa752fe5e200f3a354f9d273f8f842` preserved; notices differ only by three path-prefix repairs, with all user text retained.
- Coordinator `test javadoc --max-workers=2 --console=plain --no-daemon --offline` passed in main, 5m32s: 106 tasks (33 executed/8 cached/65 up-to-date); Metal 356 tests, zero failures, 10 skips; Javadoc warnings concern unchanged APIs.
- `GRADLE_USER_HOME`/`TMPDIR` used the designated worktree's `build/engine0020/{gradle-user-home,tmp}`; `SYNAPTIK_METAL_TEST_LIBRARY` used its source-matched packaged dylib. Final [metadata](../../../../glossary.md#term-metadata) review APPROVE: 245 links passed; executable evidence reused.

Status: Complete
