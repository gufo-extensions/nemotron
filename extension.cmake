# NVIDIA Nemotron-H family external model extension.

gufo_extension_target(
  NAME gufo_nemotron
  SOURCES
    config.cpp
    nvfp4.cpp
    weights.cpp
    tokenizer.cpp
    chat_template.cpp
    logit_head.cpp
    continuation_layout.cpp
    vision_image.cpp
    vision_prompt.cpp
    probe.cpp
    extension.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
)

if(TARGET gufo_nemotron)
  target_include_directories(gufo_nemotron PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>"
  )
  if(ENGINE_ENABLE_HIP)
    target_compile_definitions(gufo_nemotron PRIVATE ENGINE_ENABLE_HIP=1)
  endif()
endif()

# llama.cpp-derived quantized matrix kernels (kernels/rocm/mmq/VENDOR.md).
# They keep their own C++17 dialect, GGML feature macros, fast math and
# upstream style, so their warnings stay off; one
# MMQ translation unit per weight format keeps the template-heavy build
# parallel. The vendored headers stay private to this library, so they can't
# shadow another extension's ggml.h; consumers include nb_mmq.h.
gufo_extension_target(
  NAME gufo_nemotron_mmq
  LINK_LIBRARIES
    hip::host
    roc::hipblas
)
if(TARGET gufo_nemotron_mmq)
  gufo_extension_hip_sources(gufo_nemotron_mmq
    SOURCES
      kernels/rocm/mmq/nb_ggml_stubs.hip.cpp
      kernels/rocm/mmq/nb_mmq.hip.cpp
      kernels/rocm/mmq/mmq_instance_iq2_xxs.hip.cpp
      kernels/rocm/mmq/mmq_instance_nvfp4.hip.cpp
      kernels/rocm/mmq/mmq_instance_q8_0.hip.cpp
      kernels/rocm/mmq/mmvq.hip.cpp
      kernels/rocm/mmq/quantize.hip.cpp
    OPTIONS
      -O3 -ffast-math -fno-finite-math-only -w
    DEFINITIONS
      GGML_USE_HIP
      GGML_HIP_NO_VMM
  )
  set_target_properties(gufo_nemotron_mmq PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
  target_include_directories(gufo_nemotron_mmq PRIVATE
    "${GUFO_EXTENSION_BUILD_ROOT}/kernels/rocm/mmq")
endif()

# Device side: the trunk and drafter kernels, the weight upload, the runtime
# (sessions, snapshots, batched decoding) and draft-and-verify rounds.
gufo_extension_target(
  NAME gufo_nemotron_hip
  SOURCES
    kernels/rocm/device_model.cpp
    kernels/rocm/vision_gemm.cpp
    engine.cpp
    drafter.cpp
    speculative.cpp
    vision_encoder.cpp
  LINK_LIBRARIES
    gufo_nemotron
    gufo_nemotron_mmq
    gufo_ext_runtime
    gufo_ext_core
    gufo_ext_hip
    hip::host
    roc::hipblas
    roc::hipblaslt
)
if(TARGET gufo_nemotron_hip)
  gufo_extension_hip_sources(gufo_nemotron_hip
    SOURCES
      kernels/rocm/drafter_kernels.hip
      kernels/rocm/gemm.hip
      kernels/rocm/gemv.hip
      kernels/rocm/kernels.hip
      kernels/rocm/sample.hip
      kernels/rocm/vision.hip
    OPTIONS
      -O3 -Wno-unused-command-line-argument
  )
  target_include_directories(gufo_nemotron_hip PUBLIC
    "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>"
  )
endif()

# Text provider: the serving runner with its continuation adapter, the
# prompt, chat and bench drivers and the provider registration.
gufo_extension_target(
  NAME gufo_nemotron_text
  SOURCES
    continuation_adapter.cpp
    runner.cpp
    drivers.cpp
    provider.cpp
  LINK_LIBRARIES
    gufo_nemotron_hip
    gufo_nemotron
    gufo_ext_cache
    gufo_ext_runtime
    gufo_ext_hip
    hip::host
)

# Scalar CPU reference of the trunk, for tests and probes only.
# It decodes every weight on the fly; without OpenMP it is far too slow.
set(_gufo_nemotron_openmp)
if(TARGET OpenMP::OpenMP_CXX)
  set(_gufo_nemotron_openmp OpenMP::OpenMP_CXX)
endif()
gufo_extension_target(
  NAME gufo_nemotron_reference
  SOURCES
    cpu_ops.cpp
    reference.cpp
  LINK_LIBRARIES
    gufo_nemotron
    gufo_ext_runtime
    ${_gufo_nemotron_openmp}
)
if(TARGET gufo_nemotron_reference)
  target_compile_options(gufo_nemotron_reference PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>)
endif()

# gufo_ext_runtime only declares the public services; these link their
# implementations into executables that stand apart from `gufo`.
set(_gufo_nemotron_test_libs gufo_ext_core gufo_ext_cache)
if(ENGINE_ENABLE_HIP)
  list(APPEND _gufo_nemotron_test_libs gufo_ext_hip)
endif()

gufo_extension_target(
  NAME nemotron_config_test
  SOURCES
    tests/config_test.cpp
  LINK_LIBRARIES
    gufo_nemotron
    gufo_ext_runtime
    ${_gufo_nemotron_test_libs}
)
if(TARGET nemotron_config_test)
  add_test(NAME nemotron.config COMMAND nemotron_config_test)
  set_tests_properties(nemotron.config PROPERTIES LABELS "nemotron;cpu")
endif()

gufo_extension_target(
  NAME nemotron_probe_test
  SOURCES
    tests/probe_test.cpp
  LINK_LIBRARIES
    gufo_nemotron
    gufo_ext_runtime
    ${_gufo_nemotron_test_libs}
)
if(TARGET nemotron_probe_test)
  add_test(NAME nemotron.probe COMMAND nemotron_probe_test)
  set_tests_properties(nemotron.probe PROPERTIES
    LABELS "nemotron;cpu;external-model"
    SKIP_RETURN_CODE 77)
endif()

gufo_extension_target(
  NAME nemotron_nvfp4_test
  SOURCES
    tests/nvfp4_test.cpp
  LINK_LIBRARIES
    gufo_nemotron
    gufo_ext_runtime
    ${_gufo_nemotron_test_libs}
)
if(TARGET nemotron_nvfp4_test)
  add_test(NAME nemotron.nvfp4 COMMAND nemotron_nvfp4_test)
  set_tests_properties(nemotron.nvfp4 PROPERTIES LABELS "nemotron;cpu")
endif()

set(_gufo_nemotron_fixtures "${GUFO_EXTENSION_BUILD_ROOT}/tests/fixtures")

gufo_extension_target(
  NAME nemotron_tokenizer_test
  SOURCES
    tests/tokenizer_test.cpp
  LINK_LIBRARIES
    gufo_nemotron
    gufo_ext_runtime
    ${_gufo_nemotron_test_libs}
)
if(TARGET nemotron_tokenizer_test)
  target_compile_definitions(nemotron_tokenizer_test PRIVATE
    GUFO_NEMOTRON_BOLT_TOKENIZER_GOLDEN="${_gufo_nemotron_fixtures}/tokenizer_golden.json")
  add_test(NAME nemotron.pretokenizer
    COMMAND nemotron_tokenizer_test pretokenizer)
  set_tests_properties(nemotron.pretokenizer PROPERTIES
    LABELS "nemotron;cpu;tokenizer")
  add_test(NAME nemotron.tokenizer COMMAND nemotron_tokenizer_test encode)
  set_tests_properties(nemotron.tokenizer PROPERTIES
    LABELS "nemotron;cpu;tokenizer;external-model"
    SKIP_RETURN_CODE 77)
endif()

gufo_extension_target(
  NAME nemotron_chat_template_test
  SOURCES
    tests/chat_template_test.cpp
  LINK_LIBRARIES
    gufo_nemotron
    gufo_ext_runtime
    ${_gufo_nemotron_test_libs}
)
if(TARGET nemotron_chat_template_test)
  target_compile_definitions(nemotron_chat_template_test PRIVATE
    GUFO_NEMOTRON_BOLT_CHAT_GOLDEN="${_gufo_nemotron_fixtures}/chat_template_golden.json")
  add_test(NAME nemotron.chat_template
    COMMAND nemotron_chat_template_test render)
  set_tests_properties(nemotron.chat_template PROPERTIES
    LABELS "nemotron;cpu;chat_template")
  add_test(NAME nemotron.chat_template_tokens
    COMMAND nemotron_chat_template_test encode)
  set_tests_properties(nemotron.chat_template_tokens PROPERTIES
    LABELS "nemotron;cpu;chat_template;external-model"
    SKIP_RETURN_CODE 77)
endif()

gufo_extension_target(
  NAME nemotron_vision_test
  SOURCES
    tests/vision_test.cpp
  LINK_LIBRARIES
    gufo_nemotron
    gufo_ext_runtime
    ${_gufo_nemotron_test_libs}
)
if(TARGET nemotron_vision_test)
  target_compile_definitions(nemotron_vision_test PRIVATE
    GUFO_NEMOTRON_BOLT_VISION_GOLDEN="${_gufo_nemotron_fixtures}/vision_image_golden.json")
  add_test(NAME nemotron.vision_image COMMAND nemotron_vision_test image)
  set_tests_properties(nemotron.vision_image PROPERTIES
    LABELS "nemotron;cpu;vision")
  add_test(NAME nemotron.vision_prompt COMMAND nemotron_vision_test prompt)
  set_tests_properties(nemotron.vision_prompt PROPERTIES
    LABELS "nemotron;cpu;vision")
endif()

# Full-model CPU reference run, built only on request.
gufo_extension_target(
  NAME nemotron_reference_probe
  SOURCES
    tests/reference_probe.cpp
  LINK_LIBRARIES
    gufo_nemotron_reference
    gufo_nemotron
    gufo_ext_runtime
    ${_gufo_nemotron_test_libs}
)
if(TARGET nemotron_reference_probe)
  set_target_properties(nemotron_reference_probe PROPERTIES EXCLUDE_FROM_ALL ON)
endif()

# GPU operator checks against CPU evaluations of the same formulas.
foreach(_gufo_nemotron_op kernel_ops gemv_ops drafter_ops vision_ops)
  gufo_extension_target(
    NAME nemotron_${_gufo_nemotron_op}_test
    SOURCES
      tests/${_gufo_nemotron_op}_test.cpp
    LINK_LIBRARIES
      gufo_nemotron_hip
      gufo_nemotron
      gufo_ext_runtime
      hip::host
      ${_gufo_nemotron_test_libs}
  )
  if(TARGET nemotron_${_gufo_nemotron_op}_test)
    add_test(NAME nemotron.${_gufo_nemotron_op}
      COMMAND nemotron_${_gufo_nemotron_op}_test)
    set_tests_properties(nemotron.${_gufo_nemotron_op} PROPERTIES
      LABELS "nemotron;hip;gpu;gfx1151")
  endif()
endforeach()

# The vendored quantized products against Gufo's ggml dequantizers.
gufo_extension_target(
  NAME nemotron_mmq_ops_test
  SOURCES
    tests/mmq_ops_test.cpp
  LINK_LIBRARIES
    gufo_nemotron_mmq
    gufo_nemotron
    gufo_ext_runtime
    hip::host
    ${_gufo_nemotron_test_libs}
)
if(TARGET nemotron_mmq_ops_test)
  add_test(NAME nemotron.mmq_ops COMMAND nemotron_mmq_ops_test)
  set_tests_properties(nemotron.mmq_ops PROPERTIES
    LABELS "nemotron;hip;gpu;gfx1151")
endif()

# Sessions on the released files: decode against prefill, snapshots,
# batches and, with the drafter beside the trunk, draft-and-verify rounds.
gufo_extension_target(
  NAME nemotron_runtime_test
  SOURCES
    tests/runtime_test.cpp
  LINK_LIBRARIES
    gufo_nemotron_hip
    gufo_nemotron
    gufo_ext_runtime
    hip::host
    ${_gufo_nemotron_test_libs}
)
if(TARGET nemotron_runtime_test)
  add_test(NAME nemotron.runtime COMMAND nemotron_runtime_test)
  set_tests_properties(nemotron.runtime PROPERTIES
    LABELS "nemotron;hip;gpu;gfx1151;external-model;slow"
    SKIP_RETURN_CODE 77
    TIMEOUT 600)
endif()

# The continuation adapter on the released files: restores reproduce the
# source session byte for byte, and broken restores are refused.
gufo_extension_target(
  NAME nemotron_continuation_test
  SOURCES
    tests/continuation_test.cpp
  LINK_LIBRARIES
    gufo_nemotron_text
    gufo_nemotron_hip
    gufo_nemotron
    gufo_ext_runtime
    hip::host
    ${_gufo_nemotron_test_libs}
)
if(TARGET nemotron_continuation_test)
  add_test(NAME nemotron.continuation COMMAND nemotron_continuation_test)
  set_tests_properties(nemotron.continuation PROPERTIES
    LABELS "nemotron;hip;gpu;gfx1151;external-model;slow"
    SKIP_RETURN_CODE 77
    TIMEOUT 600)
endif()

# The vision encoder on NVIDIA's projector against the reference model.
gufo_extension_target(
  NAME nemotron_vision_encoder_test
  SOURCES
    tests/vision_encoder_test.cpp
  LINK_LIBRARIES
    gufo_nemotron_hip
    gufo_nemotron
    gufo_ext_runtime
    hip::host
    ${_gufo_nemotron_test_libs}
)
if(TARGET nemotron_vision_encoder_test)
  target_compile_definitions(nemotron_vision_encoder_test PRIVATE
    GUFO_NEMOTRON_BOLT_VISION_ENCODER_GOLDEN="${_gufo_nemotron_fixtures}/vision_encoder_golden.json")
  add_test(NAME nemotron.vision_encoder COMMAND nemotron_vision_encoder_test)
  set_tests_properties(nemotron.vision_encoder PROPERTIES
    LABELS "nemotron;hip;gpu;gfx1151;external-model;vision"
    SKIP_RETURN_CODE 77
    TIMEOUT 600)
endif()

# Runtime, drafter and vision probes for parity dumps, rates and timings,
# built on request.
foreach(_gufo_nemotron_probe runtime drafter vision)
  gufo_extension_target(
    NAME nemotron_${_gufo_nemotron_probe}_probe
    SOURCES
      tests/${_gufo_nemotron_probe}_probe.cpp
    LINK_LIBRARIES
      gufo_nemotron_hip
      gufo_nemotron
      gufo_ext_runtime
      hip::host
      ${_gufo_nemotron_test_libs}
  )
  if(TARGET nemotron_${_gufo_nemotron_probe}_probe)
    set_target_properties(nemotron_${_gufo_nemotron_probe}_probe PROPERTIES
      EXCLUDE_FROM_ALL ON)
  endif()
endforeach()

install(FILES
  "${GUFO_EXTENSION_BUILD_ROOT}/LICENSE"
  "${GUFO_EXTENSION_BUILD_ROOT}/THIRD_PARTY_NOTICES.md"
  DESTINATION share/licenses/gufo/extensions/nemotron)
install(DIRECTORY "${GUFO_EXTENSION_BUILD_ROOT}/licenses/"
  DESTINATION share/licenses/gufo/extensions/nemotron/licenses)
