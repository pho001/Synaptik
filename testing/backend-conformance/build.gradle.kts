import org.gradle.api.tasks.compile.JavaCompile
import org.gradle.api.tasks.testing.Test

dependencies {
    implementation(project(":modules:backend-contract"))
    testImplementation(project(":modules:model"))
    testImplementation(project(":modules:planning"))
    testImplementation(project(":modules:runtime"))
    testImplementation(project(":modules:prepare"))
    testImplementation(project(":backends:cpu"))
}

tasks.withType<JavaCompile>().configureEach {
    options.compilerArgs.addAll(listOf("--add-modules", "jdk.incubator.vector"))
}

tasks.withType<Test>().configureEach {
    jvmArgs("--add-modules", "jdk.incubator.vector")
}
