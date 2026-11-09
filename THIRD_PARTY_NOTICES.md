# Third-party notices

This extension contains code adapted from the projects below. Their license
texts are in [licenses/](licenses/) and are installed with the extension.

| Project | Used for | License | Revision | Source |
| --- | --- | --- | --- | --- |
| llama.cpp / ggml | Quantized matrix kernels (`kernels/rocm/mmq/`) and the IQ2_XXS grid (`iq2xxs_grid.hpp`) | `MIT` | `5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a` | [llama.cpp](https://github.com/ggml-org/llama.cpp) |
| Gufo | HIP adaptation of the quantized matrix kernels, from its DeepSeek V4 Flash copy | `MIT` | `5f9612c3b19ab6676b842095bc88bb9b32da5cf0` | [gufo-org/gufo](https://github.com/gufo-org/gufo) |
| DS4 | The quantized matrix kernels' raw-pointer wrapper and ggml stub layer (`nb_mmq*`, `nb_ggml_stubs*`), through Gufo's copy | `MIT` | `84cc882352757baf628a1776badf7cc54d584e28` | [antirez/ds4](https://github.com/antirez/ds4) |
| LFM3 Gufo extension | Continuation hooks, layout and adapter (`continuation_*`) and the adapter test | `MIT` | `cbc13fed6deffabe39c2677abf915643137c248a` | [gufo-org/lfm3-gufo-extension](https://github.com/gufo-org/lfm3-gufo-extension) |

The quantized matrix kernels came to this extension through Gufo's DeepSeek V4
Flash kernels; [kernels/rocm/mmq/VENDOR.md](kernels/rocm/mmq/VENDOR.md) lists
what was kept, removed and added. The parts Gufo's copy took from the DS4
GB10/GX10 fork (xangel82/DS4-GB10-GX10-DSpark-CUDA, `910501e`: the fused and
paired MoE dispatch) are among the removed ones.

| License text | Projects |
| --- | --- |
| [licenses/llama.cpp.txt](licenses/llama.cpp.txt) | llama.cpp / ggml |
| [licenses/gufo.txt](licenses/gufo.txt) | Gufo |
| [licenses/ds4.txt](licenses/ds4.txt) | DS4 |
| [licenses/lfm3-gufo-extension.txt](licenses/lfm3-gufo-extension.txt) | LFM3 Gufo extension |
