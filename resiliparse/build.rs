// Copyright 2023 Janek Bevendorff
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::env::consts;
use std::fs;
use std::path::PathBuf;

extern crate bindgen;

fn main() {
    let arch = consts::ARCH.replace("x86_64", "x64").replace("aarch64", "arm64");
    let os = consts::OS.replace("macos", "osx");
    let mut vcpkg_dir = PathBuf::from(format!("../vcpkg_installed/{}-{}", arch, os));
    vcpkg_dir = fs::canonicalize(vcpkg_dir.clone()).unwrap_or(vcpkg_dir);

    println!("cargo:rustc-link-search=native={}/lib", vcpkg_dir.to_str().unwrap());
    println!("cargo:rustc-link-lib=lexbor");
    println!("cargo:rerun-if-changed=src/third_party/lexbor.h");

    bindgen::Builder::default()
        .header("src/third_party/lexbor.h")
        .clang_arg(format!("-I{}/include", vcpkg_dir.to_str().unwrap()))
        .allowlist_function("(lexbor|lxb)_.*")
        .allowlist_type("(LEXBOR|lexbor|lxb)_.*")
        .allowlist_var("(LEXBOR|LXB)_.*")
        .default_enum_style(bindgen::EnumVariation::ModuleConsts)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Error generating Lexbor binding")
        .write_to_file(PathBuf::from("src").join("third_party").join("lexbor.rs"))
        .unwrap();
}
