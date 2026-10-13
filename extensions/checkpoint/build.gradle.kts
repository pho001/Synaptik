dependencies {
    implementation(project(":extensions:nn"))
    implementation(project(":modules:engine"))
}

tasks.withType<Test>().configureEach {
    jvmArgs("--add-modules", "jdk.incubator.vector")
}
