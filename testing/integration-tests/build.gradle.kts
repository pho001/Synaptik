import org.gradle.api.tasks.testing.Test

dependencies {
    implementation(project(":modules:engine"))
    testImplementation(project(":modules:compiler"))
    testImplementation(project(":modules:runtime"))
    testImplementation(project(":modules:config"))
    testImplementation(project(":modules:model"))
    testImplementation(project(":backends:cpu"))
}

tasks.withType<Test>().configureEach {
    jvmArgs("--add-modules", "jdk.incubator.vector")
}
