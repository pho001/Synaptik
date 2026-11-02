# Private release process

## Scope and ownership

The first-production distribution is **private/internal**. The standard local packaging
mechanism is implemented and independently qualified by
[Engine0024](../planning/modules/engine/tasks/0024-versioned-local-distribution.md#result).
That component acceptance is not a successful final release gate or permission to upload.
No public project licence, redistribution grant, registry, publisher signing identity or
notarization credential has been selected.

Use the exercised [installation commands](../getting-started.md#install-a-versioned-local-candidate)
and [support matrix](../backend-guide/production-support.md), rather than inventing another
packager or dependency list. Ordinary development defaults to `0.1.0-SNAPSHOT`; the explicit
first candidate version is `0.1.0`. These coordinates do not establish a future compatibility
or semantic-versioning guarantee.

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
identity. Native signatures/rebuilds are not promised bit-identical.

Preserve the complete root `THIRD_PARTY_NOTICES.md` in applicable CPU artifacts and outer
distributions, including the Cephes conditions and disclaimer. Preserve the Gradle wrapper's
embedded Apache licence. Those notices grant no rights to Synaptik's own code.

## Final qualification

Freeze a candidate only after the required CPU/Metal changes are integrated. Record exact source
and artifact identities, actual commands, failures and recovery runs. Then:

1. Run the affected and final repository/architecture checks and public Javadocs with the declared
   JDK/native inputs. Record every skip, observer limitation and unsupported domain honestly.
2. Verify all archive/checksum/provenance entries and published dependency closures.
3. Extract source outside any checkout and compile with the build cache disabled and tasks rerun;
   do not mistake checkout classes or cached outputs for a clean installation.
4. Resolve only the extracted repository from fresh external Maven and Gradle consumers. Compile
   the documented public examples and execute the required CPU/Metal compute, gradient, repeated
   training, save/close/load and newly configured training journeys.
5. Record relevant performance observations and independent code/evidence review before marking
   the candidate accepted. Numerical correctness does not imply universal speed or zero allocation.

The production matrix owns the current acceptance status. A successful packaging task, local
archive, ad-hoc native signature or earlier component gate must never be relabeled as a complete
production release. Public distribution, trusted signing, CI promotion and rollback policy need
separate owner decisions; they are not hidden prerequisites for this private local candidate.

## Related documentation

- [Production target](../planning/production-release-goal.md)
- [Public API status](../api/public-api.md)
- [Documentation rules](documentation-rules.md)
