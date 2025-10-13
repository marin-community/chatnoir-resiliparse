vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO phoerious/lexbor
    REF 647a0b5
    SHA512 a8d7c71a197edab5446bf757bfda4653cb839d69c59e0e6dc80ddf5512c94103e3c54c6a52ffc4de882dea3c7901b1103e7b5c835b80403feee48c0c9ef48d83
)

string(COMPARE EQUAL "${VCPKG_LIBRARY_LINKAGE}" "static" BUILD_STATIC)
string(COMPARE EQUAL "${VCPKG_LIBRARY_LINKAGE}" "dynamic" BUILD_SHARED)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
    -DLEXBOR_BUILD_SHARED=${BUILD_SHARED}
    -DLEXBOR_BUILD_STATIC=${BUILD_STATIC}
    -DLEXBOR_OPTIMIZATION_LEVEL=-O3
)
vcpkg_cmake_install()
vcpkg_copy_pdbs()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include"
    "${CURRENT_PACKAGES_DIR}/include/lexbor/html/tree/insertion_mode"
    "${CURRENT_PACKAGES_DIR}/debug/include/lexbor/html/tree/insertion_mode"
)

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
