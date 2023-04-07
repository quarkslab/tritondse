# TritonDSE

TritonDSE is a Python library providing exploration capabilities to Triton
and some refinement easing its usage. This library is primarily designed
to perform pure emulation symbolic execution even though it can also be
applied under different settings. It works by performing an elementary
loading of the program and starts exploring from the entrypoint. The whole
exploration can be instrumented using a hook mechanism enabling to obtain
a handle on various events.

At the moment solely ELF and Linux are supported. But further development
can lead to more platform. Furthermore it provides facilities on the C
runtime and it has not been tested on other types of programs.

[Documentation](https://quarkslab.github.io/tritondse)


<p align="center">
  <a href="https://github.com/quarkslab/tritondse/releases">
    <img src="https://img.shields.io/github/v/release/quarkslab/tritondse?logo=github">
  </a>
  <img src="https://img.shields.io/github/license/quarkslab/tritondse"/>
  <a href="https://github.com/quarkslab/tritondse/releases">
    <img src="https://img.shields.io/github/actions/workflow/status/quarkslab/tritondse/doc.yml">
  </a>
  <a href="https://github.com/quarkslab/pastis/releases">
    <img src="https://img.shields.io/github/actions/workflow/status/quarkslab/tritondse/release.yml">
  </a>
  <img src="https://img.shields.io/github/downloads/quarkslab/tritondse/total"/>
  <img src="https://img.shields.io/pypi/dm/tritondse"/>
</p>

---

TritonDSE goal is to provide higher-level primitives than [Triton](https://triton-library.github.io/).
Triton is a low-level framework where one have to provide manually all instructions to be executed
symbolically. As such, TritonDSE provides the following features:

* Loader mechanism (based on [LIEF](https://lief-project.github.io/), [cle](https://github.com/angr/cle) or custom ones)
* Memory segmentation
* Coverage strategies (block, edge, path)
* Pointer coverage
* Automatic input injection on stdin, argv
* Input replay with QBDI
* input scheduling *(customizable)*
* sanitizer mechanism
* basic heap allocator
* some libc symbolic stubs

---

# Quick start

* [Installation](#installation)
* [Documentation](#documentation)


## Installation

```bash
pip install tritondse
```

The pip package will install all dependencies.


## Documentation

A complete documentation on how to use TritonDSE is available on
[Github pages](https://quarkslab.github.io/qsynthesis).


---


## External Contributors

* Jonathan Salwan
* Richard Abou Chaaya

[*All contributions**](https://github.com/quarkslab/tritondse/graphs/contributors)

