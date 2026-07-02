dependencies {
    implementation(project(":modules:compiler"))
    implementation(project(":modules:runtime"))
    implementation(project(":modules:prepare"))
    implementation(project(":modules:config"))
    implementation(project(":modules:trace"))
    implementation(project(":backends:cpu"))
    implementation(project(":backends:metal"))
    implementation(project(":backends:cuda"))
}
