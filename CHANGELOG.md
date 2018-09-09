Changelog for the Metabolize fork
=================================

## 1.1.8.dev1 (Sep 9, 2018)

- Increased support for quad faces
    1. Reading a quad mesh's faces, normals, and texture coordinates from an OBJ. f4, fn4, and ft4 are set in addition to f, fn, and ft which hold the triangulated faces as before.
    2. Writing a mesh with quad faces to OBJ. `write_obj` prioritizes f4, fn4, and ft4 over the triangulated faces.
    3. `keep_vertices` operates correctly on either f and f4, though you need to `del` the one you don't want updated.

This is an experimental release, with API and behavior subject to change in
future `.post` releases. Refer to the API in
[#9](https://github.com/metabolize/lace/pull/9). Track the development in
[#2](https://github.com/metabolize/lace/pull/2).


## 1.1.8.post1 (Sep 9, 2018)

Substantively identical to upstream 1.1.8.
