dependencies {
    implementation(project(":modules:engine"))
    implementation(project(":modules:model"))
    implementation(project(":backends:cpu"))
    runtimeOnly(project(":backends:openblas-provider"))
}
tasks.register<JavaExec>("benchmark") {
    group = "verification"
    jvmArgs("--enable-native-access=ALL-UNNAMED")
    jvmArgs("--add-modules=jdk.incubator.vector")
    classpath = files(sourceSets["main"].output, configurations.runtimeClasspath)
    mainClass.set("io.github.pho001.synaptik.tools.benchmarks.CpuLifecycleBenchmark")
    args = project.findProperty("profile")?.toString()?.let { listOf(it) } ?: listOf("smoke")
    dependsOn(tasks.named("classes"))
}
