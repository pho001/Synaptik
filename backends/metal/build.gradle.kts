import java.nio.file.InvalidPathException
import java.nio.file.Path
import org.gradle.api.file.DuplicatesStrategy
import org.gradle.api.provider.Property
import org.gradle.api.tasks.Exec
import org.gradle.api.tasks.Input
import org.gradle.api.tasks.bundling.Zip
import org.gradle.process.CommandLineArgumentProvider

abstract class MetalNativePackageArgumentProvider : CommandLineArgumentProvider {
    @get:Input
    abstract val packagePath: Property<String>

    override fun asArguments(): Iterable<String> = listOf(packagePath.get())
}

dependencies {
    implementation(project(":modules:model"))
    implementation(project(":modules:config"))
    implementation(project(":modules:planning"))
    implementation(project(":modules:runtime"))
    implementation(project(":modules:prepare"))
    implementation(project(":modules:backend-contract"))
    implementation(project(":modules:trace"))
    testImplementation(project(":modules:compiler"))
}

val metalNativePackageProperty = providers.gradleProperty("synaptikMetalNativePackage")
    .orElse("")
val metalNativePackagePath = metalNativePackageProperty.map { configured ->
    if (configured.isEmpty()) {
        throw GradleException(
            "verifyMetalNativePackage and metalNativeLocalZip require "
                + "-PsynaptikMetalNativePackage=/absolute/path"
        )
    }
    if (configured.isBlank()) {
        throw GradleException("synaptikMetalNativePackage must not be blank")
    }
    val path = try {
        Path.of(configured)
    } catch (exception: InvalidPathException) {
        throw GradleException("synaptikMetalNativePackage is not a valid path", exception)
    }
    if (!path.isAbsolute) {
        throw GradleException("synaptikMetalNativePackage must be absolute")
    }
    configured
}
val metalNativePackageDirectory = layout.dir(
    metalNativePackagePath.map { configured -> Path.of(configured).toFile() }
)
val metalNativePackageVerifier = rootProject.layout.projectDirectory.file(
    "native/metal-macos-arm64/verify-package.sh"
)

val verifyMetalNativePackage = tasks.register<Exec>("verifyMetalNativePackage") {
    group = "verification"
    description = "Verifies one explicitly supplied Task 0045 macOS-arm64 package."
    executable(metalNativePackageVerifier.asFile.absolutePath)
    val packageArgumentProvider =
        objects.newInstance(MetalNativePackageArgumentProvider::class.java)
    packageArgumentProvider.packagePath.set(metalNativePackagePath)
    argumentProviders.add(packageArgumentProvider)
}
tasks.register<Zip>("metalNativeLocalZip") {
    group = "distribution"
    description = "Creates an unversioned local ZIP from one verified Metal native package."
    dependsOn(verifyMetalNativePackage)

    archiveFileName.set("synaptik-metal-macos-arm64-local.zip")
    destinationDirectory.set(layout.buildDirectory.dir("distributions"))
    isPreserveFileTimestamps = false
    isReproducibleFileOrder = true
    includeEmptyDirs = false
    duplicatesStrategy = DuplicatesStrategy.FAIL
    dirPermissions {
        unix("rwxr-xr-x")
    }

    into("macos-arm64") {
        from(metalNativePackageDirectory.map {
            it.file("libsynaptik_metal_foundation.dylib")
        }) {
            filePermissions {
                unix("rwxr-xr-x")
            }
        }
        from(metalNativePackageDirectory.map { it.file("manifest.json") }) {
            filePermissions {
                unix("rw-r--r--")
            }
        }
        from(metalNativePackageDirectory.map { it.file("SHA256SUMS") }) {
            filePermissions {
                unix("rw-r--r--")
            }
        }
    }
}
