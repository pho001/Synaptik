import groovy.json.JsonOutput
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.LinkOption
import java.nio.file.Path
import java.security.MessageDigest
import java.util.HexFormat
import org.gradle.api.DefaultTask
import org.gradle.api.file.ConfigurableFileCollection
import org.gradle.api.file.DirectoryProperty
import org.gradle.api.file.RegularFileProperty
import org.gradle.api.plugins.JavaPluginExtension
import org.gradle.api.provider.ListProperty
import org.gradle.api.provider.Property
import org.gradle.api.publish.PublishingExtension
import org.gradle.api.publish.maven.MavenPublication
import org.gradle.api.tasks.Input
import org.gradle.api.tasks.InputFile
import org.gradle.api.tasks.InputFiles
import org.gradle.api.tasks.Internal
import org.gradle.api.tasks.Nested
import org.gradle.api.tasks.OutputFile
import org.gradle.api.tasks.PathSensitive
import org.gradle.api.tasks.PathSensitivity
import org.gradle.api.tasks.TaskAction
import org.gradle.api.tasks.bundling.AbstractArchiveTask
import org.gradle.api.tasks.bundling.Jar
import org.gradle.api.tasks.bundling.Zip
import org.gradle.api.tasks.javadoc.Javadoc
import org.gradle.external.javadoc.StandardJavadocDocletOptions

/** Reuses hashing state within one task action, never across parallel tasks. */
class ReleaseHasher {
    private val digest = MessageDigest.getInstance("SHA-256")
    private val buffer = ByteArray(8192)
    private val hex = HexFormat.of()

    fun sha256(path: Path): String {
        digest.reset()
        Files.newInputStream(path).use { input ->
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return hex.formatHex(digest.digest())
    }
}

/** Couples each literal archive path to its own file input, including Ant-default-excluded names. */
data class ReleaseSource(
    @get:Input val relativePath: String,
    @get:InputFile @get:PathSensitive(PathSensitivity.NONE) val file: java.io.File)

/** Hashes the reviewed relative paths and current bytes without requiring a Git checkout. */
abstract class SourceProvenance : DefaultTask() {
    @get:Nested abstract val sources: ListProperty<ReleaseSource>
    @get:Input abstract val baseRevision: Property<String>
    @get:Input abstract val productVersion: Property<String>
    @get:Internal abstract val sourceRoot: DirectoryProperty
    @get:OutputFile abstract val manifestFile: RegularFileProperty

    @TaskAction
    fun writeManifest() {
        val selected = sources.get()
        val paths = selected.map { it.relativePath }
        require(paths == paths.distinct().sorted()) {
            "release-source-paths.txt must contain unique sorted paths"
        }
        val root = sourceRoot.get().asFile.toPath()
        val hasher = ReleaseHasher()
        val forbiddenDirectories = setOf(".git", ".gradle", ".idea", ".vscode", "build", "out",
            "dist", "tmp", "temp", "evidence", "__pycache__")
        val unsafeSuffix = Regex("(?i).*(\\.(pem|key|p12|pfx|jks|keystore|iml)|~)$")
        val entries = selected.map { source ->
            val relative = source.relativePath
            val path = Path.of(relative)
            require(!path.isAbsolute && !relative.contains('\\')
                    && path.normalize().toString() == relative && !path.startsWith("..")
                    && !unsafeSuffix.matches(relative)) {
                "Unsafe release source path: $relative"
            }
            var current = root
            for (segment in path) {
                val name = segment.toString()
                require(name !in forbiddenDirectories && !name.startsWith(".env")) {
                    "Unsafe release source path: $relative"
                }
                current = current.resolve(segment)
                require(!Files.isSymbolicLink(current)) {
                    "Release source must not traverse a symlink: $relative"
                }
            }
            require(current == source.file.toPath()) { "Mismatched release source: $relative" }
            require(Files.isRegularFile(current, LinkOption.NOFOLLOW_LINKS)) {
                "Missing release source: $relative"
            }
            mapOf("path" to relative, "sha256" to hasher.sha256(current))
        }
        val manifest = mapOf("schemaVersion" to 1, "baseRevision" to baseRevision.get(),
            "productVersion" to productVersion.get(), "sourceState" to "selected-working-tree-bytes",
            "files" to entries)
        val output = manifestFile.get().asFile.toPath()
        Files.createDirectories(output.parent)
        Files.writeString(output, JsonOutput.prettyPrint(JsonOutput.toJson(manifest)) + "\n", StandardCharsets.UTF_8)
    }
}

/** Records artifact hashes outside the hashed archives, provenance and repository files. */
abstract class DistributionChecksums : DefaultTask() {
    @get:InputFiles @get:PathSensitive(PathSensitivity.RELATIVE)
    abstract val artifacts: ConfigurableFileCollection
    @get:Internal abstract val artifactRoot: DirectoryProperty
    @get:OutputFile abstract val checksumsFile: RegularFileProperty

    @TaskAction
    fun writeChecksums() {
        val root = artifactRoot.get().asFile.toPath()
        val hasher = ReleaseHasher()
        val files = artifacts.files.filter { it.isFile }.sortedBy { it.path }
        val sums = files.joinToString("") { file ->
            val relative = root.relativize(file.toPath()).toString().replace('\\', '/')
            "${hasher.sha256(file.toPath())}  $relative\n"
        }
        val output = checksumsFile.get().asFile.toPath()
        Files.createDirectories(output.parent)
        Files.writeString(output, sums, StandardCharsets.UTF_8)
    }
}

// Grouping projects, CUDA, ONNX, CLI, benchmarks and tests are not supported publications.
val productProjects = listOf(
    ":modules:model", ":modules:backend-contract", ":modules:trace", ":modules:config",
    ":modules:planning", ":modules:compiler", ":modules:runtime", ":modules:prepare",
    ":backends:cpu", ":backends:metal", ":backends:openblas-provider", ":tools:tuning",
    ":extensions:training", ":extensions:checkpoint", ":modules:engine",
    ":extensions:nn", ":extensions:data", ":extensions:text")
val candidateVersion = version.toString()
val releaseDirectory = layout.buildDirectory.dir("release/$candidateVersion")
val repositoryDirectory = releaseDirectory.map { it.dir("repository") }
val publishProducts = tasks.register("publishReleaseRepository") {
    group = "distribution"
    description = "Publishes the 18 implemented Java products to the versioned local Maven repository."
}
val binaryJars = objects.fileCollection()
for (projectPath in productProjects) {
    val product = project(projectPath)
    product.pluginManager.apply("maven-publish")
    product.extensions.configure<JavaPluginExtension> {
        withSourcesJar()
        withJavadocJar()
    }
    product.tasks.withType<AbstractArchiveTask>().configureEach {
        isPreserveFileTimestamps = false
        isReproducibleFileOrder = true
    }
    product.tasks.withType<Javadoc>().configureEach {
        (options as StandardJavadocDocletOptions).addBooleanOption("notimestamp", true)
    }
    product.extensions.configure<PublishingExtension> {
        publications.create<MavenPublication>("java") {
            from(product.components["java"])
            artifactId = product.name
        }
        repositories.maven {
            name = "releaseLocal"
            url = repositoryDirectory.get().asFile.toURI()
        }
    }
    publishProducts.configure {
        dependsOn("$projectPath:publishJavaPublicationToReleaseLocalRepository")
    }
    binaryJars.from(product.tasks.named<Jar>("jar").flatMap { it.archiveFile })
}

// Keep the complete unmodified upstream notice with independently usable CPU artifacts.
val thirdPartyNotices = layout.projectDirectory.file("THIRD_PARTY_NOTICES.md")
project(":backends:cpu").tasks.withType<Jar>().configureEach {
    from(thirdPartyNotices) { into("META-INF") }
}

// A single exact-path inventory supplies both relative input identity and archived current bytes.
val sourcePaths = providers.fileContents(layout.projectDirectory.file("gradle/release-source-paths.txt"))
    .asText.map { text -> text.lineSequence().filter { it.isNotBlank() && !it.startsWith("#") }.toList() }
val selectedSources = sourcePaths.get().map { relative ->
    ReleaseSource(relative, layout.projectDirectory.file(relative).asFile)
}
val sourceManifest = tasks.register<SourceProvenance>("sourceManifest") {
    group = "distribution"
    description = "Hashes exactly the reviewed working-tree source selection without consulting Git."
    sources.set(selectedSources)
    sourceRoot.set(layout.projectDirectory)
    baseRevision.set(providers.gradleProperty("releaseBaseRevision"))
    productVersion.set(candidateVersion)
    manifestFile.set(releaseDirectory.map { it.file("source-provenance.json") })
}
val sourceSnapshot = tasks.register<Zip>("sourceSnapshot") {
    group = "distribution"
    description = "Archives the reviewed dirty source and its relative-path SHA256 provenance."
    dependsOn(sourceManifest)
    archiveFileName.set("synaptik-$candidateVersion-sources.zip")
    destinationDirectory.set(releaseDirectory)
    isPreserveFileTimestamps = false
    isReproducibleFileOrder = true
    includeEmptyDirs = false
    duplicatesStrategy = DuplicatesStrategy.FAIL
    useFileSystemPermissions()
    dirPermissions { unix("rwxr-xr-x") }
    into("synaptik-$candidateVersion") {
        for (source in selectedSources) {
            val directory = source.relativePath.substringBeforeLast('/', "")
            from(source.file) {
                if (directory.isNotEmpty()) into(directory)
            }
        }
        from(sourceManifest.flatMap { it.manifestFile })
    }
}
val repositoryZip = tasks.register<Zip>("releaseRepositoryZip") {
    group = "distribution"
    description = "Archives the versioned standard Maven repository and third-party notices."
    dependsOn(publishProducts)
    archiveFileName.set("synaptik-$candidateVersion-repository.zip")
    destinationDirectory.set(releaseDirectory)
    isPreserveFileTimestamps = false
    isReproducibleFileOrder = true
    includeEmptyDirs = false
    duplicatesStrategy = DuplicatesStrategy.FAIL
    from(repositoryDirectory) { into("repository") }
    from(thirdPartyNotices)
}

// Reuse the backend-owned verifier/ZIP after its build script registers the existing task.
evaluationDependsOn(":backends:metal")
val metalZip = project(":backends:metal").tasks.named<Zip>("metalNativeLocalZip")
val binaryZip = tasks.register<Zip>("releaseBinaryZip") {
    group = "distribution"
    description = "Packages Java binaries, Maven repository and the existing explicitly verified Metal ZIP."
    dependsOn(publishProducts, metalZip)
    archiveFileName.set("synaptik-$candidateVersion-bin.zip")
    destinationDirectory.set(releaseDirectory)
    isPreserveFileTimestamps = false
    isReproducibleFileOrder = true
    includeEmptyDirs = false
    duplicatesStrategy = DuplicatesStrategy.FAIL
    from(binaryJars) { into("lib") }
    from(repositoryDirectory) { into("repository") }
    from(metalZip.flatMap { it.archiveFile }) { into("native") }
    from(thirdPartyNotices)
}
val checksums = tasks.register<DistributionChecksums>("releaseChecksums") {
    group = "distribution"
    description = "Records distribution/provenance/repository SHA256 values outside the hashed artifacts."
    dependsOn(sourceSnapshot, repositoryZip, binaryZip)
    artifactRoot.set(releaseDirectory)
    artifacts.from(sourceSnapshot.flatMap { it.archiveFile }, repositoryZip.flatMap { it.archiveFile },
        binaryZip.flatMap { it.archiveFile }, sourceManifest.flatMap { it.manifestFile },
        repositoryDirectory.map { it.asFileTree })
    checksumsFile.set(releaseDirectory.map { it.file("SHA256SUMS") })
}
tasks.register("releaseDistribution") {
    group = "distribution"
    description = "Builds an internal local candidate; no public upload, signing or licence grant."
    dependsOn(checksums)
}
