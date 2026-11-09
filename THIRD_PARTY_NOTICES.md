# Third-party notices

This extension contains code adapted from the projects below. Their license
texts are in [licenses/](licenses/) and are installed with the extension.

| Project | Used for | License | Revision | Source |
| --- | --- | --- | --- | --- |
| llama.cpp / ggml | Quantized matrix kernels (`kernels/rocm/mmq/`) and the IQ2_XXS grid (`iq2xxs_grid.hpp`) | `MIT` | `5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a` | [llama.cpp](https://github.com/ggml-org/llama.cpp) |
| DS4 | HIP adaptation of the quantized matrix kernels | `MIT` | `84cc882352757baf628a1776badf7cc54d584e28` | [antirez/ds4](https://github.com/antirez/ds4) |
| DS4 GB10/GX10 fork | HIP adaptation of the quantized matrix kernels | `MIT` | `910501e` | [xangel82/DS4](https://github.com/xangel82/DS4-GB10-GX10-DSpark-CUDA) |
| LFM3 Gufo extension | Continuation hooks, layout and adapter (`continuation_*`) and the adapter test | `MIT` | `cbc13fed6deffabe39c2677abf915643137c248a` | [gufo-org/lfm3-gufo-extension](https://github.com/gufo-org/lfm3-gufo-extension) |

The quantized matrix kernels came to this extension through Gufo's DeepSeek V4
Flash kernels; [kernels/rocm/mmq/VENDOR.md](kernels/rocm/mmq/VENDOR.md) lists
what was kept, removed and added.

| License text | Projects |
| --- | --- |
| [licenses/llama.cpp.txt](licenses/llama.cpp.txt) | llama.cpp / ggml |
| [licenses/ds4.txt](licenses/ds4.txt) | DS4 |
| [licenses/DS4-GB10.txt](licenses/DS4-GB10.txt) | DS4 GB10/GX10 fork |
| [licenses/lfm3-gufo-extension.txt](licenses/lfm3-gufo-extension.txt) | LFM3 Gufo extension |
