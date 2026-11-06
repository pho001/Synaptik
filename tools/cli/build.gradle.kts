plugins {
    application
}

dependencies {
    implementation(project(":modules:engine"))
    implementation(project(":extensions:nn"))
    implementation("org.jline:jline-terminal:4.5.0")
    runtimeOnly("org.jline:jline-terminal-ffm:4.5.0")
}

application {
    mainClass.set("io.github.pho001.synaptik.tools.cli.SynaptikCli")
    applicationName = "synaptik"
    applicationDefaultJvmArgs = listOf("--enable-native-access=ALL-UNNAMED")
}

distributions {
    main {
        contents {
            from(rootProject.file("THIRD_PARTY_NOTICES.md"))
            from("licenses") { into("licenses") }
        }
    }
}

tasks.withType<Test>().configureEach {
    jvmArgs("--enable-native-access=ALL-UNNAMED")
}
