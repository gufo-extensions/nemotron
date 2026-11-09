// NVFP4 decoding against hand-computed values: halved UE4M3 scales, the NaN
// scale code, a subnormal scale, and the nibble order within a sub-block.

#include "nvfp4.hpp"

#include <array>
#include <cstddef>
#include <iostream>

namespace nemotron = gufo::models::nemotron;

int main() {
  nemotron::BlockNvfp4 block{};
  // UE4M3 scales, halved like ggml: 0x38 = 1.0 -> 0.5, 0x40 = 2.0 -> 1.0,
  // 0x7F is the NaN code and decodes to zero, 0x01 is the subnormal 2^-9.
  block.d[0] = 0x38;
  block.d[1] = 0x40;
  block.d[2] = 0x7F;
  block.d[3] = 0x01;
  block.qs[0] = 0xF2;   // element 0: code 2 (1.0), element 8: code 15 (-6.0)
  block.qs[1] = 0x97;   // element 1: code 7 (6.0), element 9: code 9 (-0.5)
  block.qs[8] = 0x5A;   // element 16: code 10 (-1.0), element 24: code 5 (3.0)
  block.qs[16] = 0x77;  // zero scale
  block.qs[24] = 0x07;  // element 48: code 7 (6.0)

  std::array<float, 64> actual{};
  nemotron::DequantizeNvfp4(&block, actual.data(), actual.size());
  std::array<float, 64> expected{};
  expected[0] = 1.0F;
  expected[8] = -6.0F;
  expected[1] = 6.0F;
  expected[9] = -0.5F;
  expected[16] = -2.0F;
  expected[24] = 6.0F;
  expected[48] = 12.0F * 0.0009765625F;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] != expected[i]) {
      std::cerr << "FAIL: element " << i << " is " << actual[i] << ", expected "
                << expected[i] << '\n';
      return 1;
    }
  }
  std::cout << "PASS\n";
  return 0;
}
