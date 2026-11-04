# Release candidate qualification and publication gates

## Scope and ownership

This guide explains how to build and qualify an exact local release candidate, and why that
qualification is separate from permission to publish. The product goal is now first-public-release
readiness under the [framework-wide audit](../planning/public-release-audit-2026-11-04.md).
The owner explicitly retains **no public redistribution grant for now**: public publication is
**Blocked** pending a licence/publisher decision. Technical stabilization and local qualification
continue; this guide is not a public-publish procedure.

The standard local packaging mechanism is implemented and independently qualified by
[Engine0024](../planning/modules/engine/tasks/0024-versioned-local-distribution.md#result).
Its component acceptance and earlier private/internal candidate receipts do not qualify changed
audit repairs or permit uploads. No public project licence, registry, official publisher signing
identity or notarization credential has been selected.

Use the exercised [installation commands](../getting-started.md#install-a-versioned-local-candidate)
and [support matrix](../backend-guide/production-support.md), rather than inventing another
packager or dependency list. Ordinary development defaults to `0.1.0-SNAPSHOT`; the earlier
explicit candidate is `0.1.0`. The current audit qualifies `0.1.0-public-audit.20261104` separately,
without overwriting those receipts. These coordinates do not establish a future compatibility
or semantic-versioning guarantee.

The sequence is repaired source → independent review/joined checks → matched archives and
provenance → clean installed consumers → recorded local acceptance. Provenance here means the
packaging manifest of selected source paths and hashes, not proof of publisher identity.
The coordinator owns final audit verification and receipts; repair workers do not manufacture
acceptance by reusing an older package.

## Candidate outputs

The root `releaseDistribution` task uses ordinary Gradle Java/Maven publications for eighteen
implemented products. Each has binary, source and Javadoc artifacts in the versioned local
Maven repository. Grouping, test, benchmark, CUDA, ONNX and CLI marker projects are not supported
publications. Their retained source is not silently promoted into the release API.

The task creates source, repository and binary ZIPs below `build/release/<version>/`, plus
`source-provenance.json` and `SHA256SUMS`. The binary ZIP contains the same Java repository and
the backend-owned, independently verified Metal local ZIP. There is no shading, fat JAR,
remote upload, credential discovery, automatic native installation or runtime library discovery.

`releaseBaseRevision` and the absolute `synaptikMetalNativePackage` directory are explicit
inputs. Build, ad-hoc sign and verify that native package using the
[Metal-owned process](../../modules/backend/metal/macos-arm64/README.md#build-prerequisites-and-local-package).
The release task reuses that verifier; it does not build or repair native bytes. Preserve
permissions during extraction, verify again, retain the installed directory for the Engine
lifetime and pass the dylib's absolute path to `MetalBackendConfiguration`.

## Exact source and notices

`gradle/release-source-paths.txt` is the reviewed, sorted exact-path inventory shared by source
archiving and provenance. It selects current bytes, including accepted dirty changes, without
requiring Git during the build. Update it when accepted source adds, removes or renames a file.
Do not substitute a HEAD-only archive or an unchecked recursive checkout copy.

Provenance records the base revision, product version and every selected relative path/SHA256.
The base revision alone does not identify a dirty build. Archive order/timestamps are fixed and
executable modes retained. `SHA256SUMS` is outside the artifacts it hashes. These accompanying
hashes detect corruption; they do not authenticate hostile replacement or establish publisher
identity. The local Metal process uses ad-hoc signing and package verification, not an official
trusted publisher signature or notarization. Native signatures/rebuilds are not promised
bit-identical; never bypass verification or relabel local signing as official trust.

Preserve the complete root `THIRD_PARTY_NOTICES.md` in applicable CPU artifacts and outer
distributions, including the Cephes conditions and disclaimer. Preserve the Gradle wrapper's
embedded Apache licence. Those notices grant no rights to Synaptik's own code.

## Final qualification

Freeze a new candidate only after the
[Ready audit repairs](../planning/public-release-audit-2026-11-04.md#confirmed-finding-repair-frontiers)
are integrated and independently reviewed. Historical Pool3d, typed-Metal, oneDNN and CPU low-parity
receipts remain limited to their recorded source and artifacts. The approved CPU low-parity
cutover supersedes CPU0078's lock; oneDNN still excludes native F64 and Metal keeps its separate
bounded domain. Neither capability admission nor a prior candidate's success proves repaired
numerical, lifecycle, import/restore or diagnostic behavior.

Record exact repaired source and native/artifact identities, actual commands, failures and
recovery runs. The source inventory, provenance and authenticated native package must identify
the bytes exercised by the final checks and installed consumers. A stale archive or an older
native library does not qualify current source; later executable/native changes require affected
requalification and regenerated matched artifacts. Then:

1. Run the affected and final repository/architecture checks and public Javadocs with the declared
   JDK/native inputs. Record every skip, observer limitation and unsupported domain honestly.
2. Rebuild the versioned candidate from the qualified source and matched native package. Verify
   every archive/checksum/provenance entry and published dependency closure; bind the acceptance
   receipt to these exact artifacts, not merely the reused `0.1.0` coordinate or base revision.
3. Extract source outside any checkout and compile with the build cache disabled and tasks rerun;
   do not mistake checkout classes or cached outputs for a clean installation.
4. Resolve only the extracted repository from fresh external Maven and Gradle consumers. Compile
   the documented public examples and execute the required CPU/Metal compute, gradient, repeated
   training, save/close/load and newly configured training journeys.
5. Record relevant performance observations and independent code/evidence review before marking
   the candidate accepted. Numerical correctness does not imply universal speed or zero allocation.

The [production ledger](../backend-guide/production-support.md) and
[audit receipts](../planning/public-release-audit-2026-11-04.md#results) record acceptance only after
these checks actually run. A successful packaging task, local archive, ad-hoc native signature or
earlier component gate is not a complete public production release. Original failed commands,
successor failures and archived successful candidates keep their original status and scope.

Public publication requires the owner's separate licence/publisher decision; no public upload is
authorized now. Official signing/trusted prebuilt distribution, hosted CI promotion and rollback
policy remain separate decisions, not infrastructure or credentials supplied by this process.
They are not hidden prerequisites for building and technically qualifying the local candidate.

## Related documentation

- [Production target](../planning/production-release-goal.md)
- [Public API status](../api/public-api.md)
- [Documentation rules](documentation-rules.md)
