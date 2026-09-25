import java.util.Locale
import org.gradle.api.tasks.javadoc.Javadoc
import org.gradle.external.javadoc.StandardJavadocDocletOptions

dependencies {
    implementation(project(":modules:engine"))
    implementation(project(":modules:model"))
    implementation(project(":backends:cpu"))
    runtimeOnly(project(":backends:openblas-provider"))
    implementation(project(":modules:backend-contract"))
    implementation(project(":modules:compiler"))
    implementation(project(":modules:config"))
    implementation(project(":modules:planning"))
    implementation(project(":modules:prepare"))
    implementation(project(":modules:runtime"))
    implementation(project(":modules:trace"))
    implementation(project(":backends:metal"))
}

val benchmarkProfile = providers.gradleProperty("profile").orElse("smoke")
    .map { it.lowercase(Locale.ROOT) }

tasks.register<JavaExec>("benchmark") {
    group = "verification"
    val profile = benchmarkProfile.get()
    jvmArgs("--enable-native-access=ALL-UNNAMED")
    jvmArgs("--add-modules=jdk.incubator.vector")
    if (profile == "evidence") {
        jvmArgs("-Xms1g", "-Xmx1g", "-XX:-TieredCompilation", "-Xbatch")
        mapOf(
            "benchmarkBaseRevision" to "baseRevision",
            "benchmarkSourceIdentity" to "sourceIdentity",
            "benchmarkGitTreeObjectSha1" to "gitTreeObjectSha1",
            "benchmarkHarnessSourceSha256" to "harnessSourceSha256",
            "benchmarkProductionSourceSha256" to "productionSourceSha256",
            "benchmarkConfiguration" to "configuration",
            "benchmarkFork" to "fork",
            "benchmarkOrderSeed" to "orderSeed",
            "benchmarkOrderIndex" to "orderIndex",
            "benchmarkPlannedOrder" to "plannedOrder",
            "benchmarkCpuIdentity" to "cpuIdentity",
            "benchmarkCpuFeatures" to "cpuFeatures",
            "benchmarkComputePreference" to "computePreference",
            "benchmarkConfiguredMaximumParallelism" to "configuredMaximumParallelism",
            "benchmarkAvailableParallelism" to "availableParallelism",
            "benchmarkMinimumElementsPerWorker" to "minimumElementsPerWorker",
            "benchmarkWorkerCount" to "workerCount",
            "benchmarkMaterializationPolicy" to "materializationPolicy",
            "benchmarkOpenBlasPolicy" to "openBlasPolicy"
        ).forEach { (gradleName, systemName) ->
            val value = providers.gradleProperty(gradleName).orNull
                ?: throw GradleException("evidence profile requires -P$gradleName=<value>")
            systemProperty("synaptik.benchmark.$systemName", value)
        }
    }
    classpath = files(sourceSets["main"].output, configurations.runtimeClasspath)
    mainClass.set("io.github.pho001.synaptik.tools.benchmarks.CpuLifecycleBenchmark")
    args(profile)
    dependsOn(tasks.named("classes"))
}

tasks.register<JavaExec>("metalBenchmark") {
    group = "verification"
    description = "Runs the fixed report-only Metal singleton-NEG route benchmark."
    val profile = benchmarkProfile.get()
    jvmArgs("--enable-native-access=ALL-UNNAMED")
    if (profile == "baseline") {
        jvmArgs("-Xms1g", "-Xmx1g", "-XX:-TieredCompilation", "-Xbatch")
        mapOf(
            "benchmarkBaseRevision" to "baseRevision",
            "benchmarkSourceIdentity" to "sourceIdentity",
            "benchmarkGitTreeObjectSha1" to "gitTreeObjectSha1",
            "benchmarkHarnessSourceSha256" to "harnessSourceSha256",
            "benchmarkHostIdentity" to "hostIdentity",
            "benchmarkMetalDeviceIdentity" to "metalDeviceIdentity",
            "benchmarkOsBuild" to "osBuild",
            "benchmarkNativeBuildIdentity" to "nativeBuildIdentity",
            "benchmarkPowerState" to "powerState",
            "benchmarkThermalState" to "thermalState",
            "benchmarkFork" to "fork"
        ).forEach { (gradleName, systemName) ->
            val value = providers.gradleProperty(gradleName).orNull
                ?: throw GradleException("baseline profile requires -P$gradleName=<value>")
            systemProperty("synaptik.benchmark.metal.$systemName", value)
        }
    }
    classpath = files(sourceSets["main"].output, configurations.runtimeClasspath)
    mainClass.set("io.github.pho001.synaptik.tools.benchmarks.MetalRouteBenchmark")
    args(profile)
    dependsOn(tasks.named("classes"))
}

tasks.named<Javadoc>("javadoc") {
    (options as StandardJavadocDocletOptions)
        .addStringOption("-add-modules", "jdk.incubator.vector")
}
