/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "velox/type/windows/Int128.h"

#include <boost/multiprecision/cpp_int.hpp>
#include <fmt/format.h>
#include <gtest/gtest.h>

#include <array>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <random>
#include <type_traits>

using namespace facebook::velox;

namespace {

using boost::multiprecision::cpp_int;

cpp_int unsignedValue(uint64_t high, uint64_t low) {
  return (cpp_int(high) << 64) | low;
}

cpp_int signedValue(const Int128& value) {
  return (cpp_int(value.high()) << 64) + value.low();
}

template <typename T>
void expectBits(const T& actual, const cpp_int& expected) {
  const cpp_int mask = (cpp_int(1) << 64) - 1;
  EXPECT_EQ(actual.low(), (expected & mask).convert_to<uint64_t>());
  EXPECT_EQ(
      static_cast<uint64_t>(actual.high()),
      ((expected >> 64) & mask).convert_to<uint64_t>());
}

TEST(Int128Test, signedLeftShift) {
  constexpr auto shifted = Int128(0, uint64_t{1} << 63) << 1;
  static_assert(shifted.high() == 1 && shifted.low() == 0);
  constexpr std::array<int64_t, 4> highValues{0, 1, -1, INT64_MIN};
  constexpr std::array<uint64_t, 5> lowValues{
      0, 1, uint64_t{1} << 63, (uint64_t{1} << 63) + 1, UINT64_MAX};
  for (const auto high : highValues) {
    for (const auto low : lowValues) {
      for (int shift = 0; shift < 128; ++shift) {
        SCOPED_TRACE(fmt::format("high={} low={} shift={}", high, low, shift));
        const Int128 value(high, low);
        expectBits(value << shift, signedValue(value) << shift);
      }
    }
  }
}

TEST(Int128Test, unsignedConstruction) {
  const auto checkSigned = []<typename T>(T value) {
    expectBits(UInt128(value), cpp_int(value));
  };
  checkSigned(int8_t{-1});
  checkSigned(int16_t{-17});
  checkSigned(int32_t{-1});
  checkSigned(int64_t{-1});
  checkSigned(long{-1});
  checkSigned(INT64_MIN);
  expectBits(UInt128(UINT64_MAX), cpp_int(UINT64_MAX));
  expectBits(UInt128(uint32_t{UINT32_MAX}), cpp_int(UINT32_MAX));
  expectBits(UInt128(true), cpp_int(1));
  constexpr UInt128 negative(-1);
  static_assert(negative.high() == UINT64_MAX && negative.low() == UINT64_MAX);
  expectBits(Int128(long{-1}), cpp_int(-1));
  expectBits(Int128(static_cast<unsigned long>(ULONG_MAX)), cpp_int(ULONG_MAX));
}

TEST(Int128Test, mixedIntegralOperands) {
  const Int128 one(1);
  const Int128 two(2);
  const uint64_t large = UINT64_MAX;

  EXPECT_TRUE(one < large);
  EXPECT_TRUE(large > one);
  EXPECT_FALSE(one == large);
  EXPECT_FALSE(Int128(-1) == large);
  EXPECT_FALSE(large == Int128(-1));
  expectBits(two * large, cpp_int(2) * large);
  expectBits(large * two, cpp_int(large) * 2);
  expectBits(large + two, cpp_int(large) + 2);
  expectBits(large - two, cpp_int(large) - 2);
  expectBits(Int128(1, 0) / large, cpp_int(1));
  auto product = two;
  product *= large;
  expectBits(product, cpp_int(2) * large);
  expectBits(UInt128(2) * int64_t{-1}, cpp_int(-2));
  expectBits(int64_t{-1} * UInt128(2), cpp_int(-2));

  const UInt128 unsignedHighBit(uint64_t{1} << 63, 1);
  static_assert(std::is_same_v<decltype(1 + unsignedHighBit), UInt128>);
  static_assert(std::is_same_v<decltype(1 - unsignedHighBit), UInt128>);
  static_assert(
      std::is_same_v<decltype(int64_t{-1} * unsignedHighBit), UInt128>);
  static_assert(
      std::is_same_v<decltype(unsignedHighBit * int64_t{-1}), UInt128>);
  const auto exact =
      unsignedValue(unsignedHighBit.high(), unsignedHighBit.low());
  expectBits(1 + unsignedHighBit, 1 + exact);
  expectBits((1 + unsignedHighBit) / 2, (1 + exact) / 2);
  expectBits(1 - unsignedHighBit, 1 - exact);
  expectBits(int64_t{-1} + UInt128(2), cpp_int(1));
  expectBits(int64_t{-1} - UInt128(2), cpp_int(-3));
}

TEST(Int128Test, mixedIntegralComparisons) {
  const std::array<UInt128, 6> unsignedValues{
      UInt128(0),
      UInt128(1),
      UInt128(UINT64_MAX),
      UInt128(1, 0),
      UInt128(uint64_t{1} << 63, 1),
      std::numeric_limits<UInt128>::max()};
  const std::array<Int128, 6> signedValues{
      Int128(0),
      Int128(1),
      Int128(-1),
      Int128(0, UINT64_MAX),
      std::numeric_limits<Int128>::min(),
      std::numeric_limits<Int128>::max()};
  const auto checkValues =
      []<typename Value>(const std::array<Value, 6>& values) {
        const auto check = [&]<typename T>(T left) {
          cpp_int expectedLeft(left);
          if constexpr (std::is_same_v<Value, UInt128>) {
            expectedLeft &= (cpp_int(1) << 128) - 1;
          }
          for (const auto& right : values) {
            SCOPED_TRACE(
                fmt::format(
                    "left={} high={} low={}", left, right.high(), right.low()));
            const cpp_int expectedRight =
                (cpp_int(right.high()) << 64) + right.low();
            EXPECT_EQ(left < right, expectedLeft < expectedRight);
            EXPECT_EQ(left > right, expectedLeft > expectedRight);
            EXPECT_EQ(left <= right, expectedLeft <= expectedRight);
            EXPECT_EQ(left >= right, expectedLeft >= expectedRight);
            EXPECT_EQ(left == right, expectedLeft == expectedRight);
            EXPECT_EQ(left != right, expectedLeft != expectedRight);
            EXPECT_EQ(right < left, expectedRight < expectedLeft);
            EXPECT_EQ(right > left, expectedRight > expectedLeft);
            EXPECT_EQ(right <= left, expectedRight <= expectedLeft);
            EXPECT_EQ(right >= left, expectedRight >= expectedLeft);
            EXPECT_EQ(right == left, expectedRight == expectedLeft);
            EXPECT_EQ(right != left, expectedRight != expectedLeft);
          }
        };
        check(0);
        check(1);
        check(int8_t{-1});
        check(int64_t{-1});
        check(INT64_MIN);
        check(UINT64_MAX);
        check(uint32_t{UINT32_MAX});
        check(true);
      };
  checkValues(unsignedValues);
  checkValues(signedValues);
}

TEST(Int128Test, negativeFloatingPointConversion) {
  for (const int64_t value : {-1, -17, -1'024, -65'537}) {
    EXPECT_EQ(static_cast<double>(Int128(value)), static_cast<double>(value));
    EXPECT_EQ(static_cast<float>(Int128(value)), static_cast<float>(value));
  }
  const auto minimum = std::numeric_limits<Int128>::min();
  EXPECT_EQ(static_cast<double>(minimum), -std::ldexp(1.0, 127));
}

TEST(Int128Test, floatingPointRounding) {
  const auto infinity = std::numeric_limits<double>::infinity();
  const UInt128 unsignedTie((uint64_t{1} << 63) + 1'024, 0);
  const UInt128 unsignedAboveTie((uint64_t{1} << 63) + 1'024, 1);
  EXPECT_EQ(static_cast<double>(unsignedTie), std::ldexp(1.0, 127));
  EXPECT_EQ(
      static_cast<double>(unsignedAboveTie),
      std::nextafter(std::ldexp(1.0, 127), infinity));

  const Int128 signedAboveTie((int64_t{1} << 62) + 512, 1);
  const double rounded = std::nextafter(std::ldexp(1.0, 126), infinity);
  EXPECT_EQ(static_cast<double>(signedAboveTie), rounded);
  EXPECT_EQ(static_cast<double>(-signedAboveTie), -rounded);

  // Converting through double would erase the low bit and round a second time
  // in the wrong direction when producing float.
  const uint64_t aboveFloatTie = (uint64_t{1} << 54) + (uint64_t{1} << 30) + 1;
  const float roundedFloat = std::nextafter(
      std::ldexp(1.0f, 54), std::numeric_limits<float>::infinity());
  EXPECT_EQ(static_cast<float>(UInt128(aboveFloatTie)), roundedFloat);
  EXPECT_EQ(static_cast<float>(Int128(aboveFloatTie)), roundedFloat);
  EXPECT_EQ(static_cast<float>(-Int128(aboveFloatTie)), -roundedFloat);
  EXPECT_EQ(
      static_cast<float>(std::numeric_limits<UInt128>::max()),
      std::numeric_limits<float>::infinity());
}

TEST(Int128Test, floatingPointOracle) {
  std::mt19937_64 random(0xF128);
  const auto check = []<typename T>(const T& value, const cpp_int& exact) {
    // Decimal conversion is independent of the implementation's limb rounding.
    const auto decimal = exact.convert_to<std::string>();
    EXPECT_EQ(
        static_cast<double>(value), std::strtod(decimal.c_str(), nullptr));
    EXPECT_EQ(static_cast<float>(value), std::strtof(decimal.c_str(), nullptr));
  };
  for (int i = 0; i < 2'000; ++i) {
    const uint64_t high = random();
    const uint64_t low = random();
    SCOPED_TRACE(i);
    const Int128 signedInput(static_cast<int64_t>(high), low);
    check(signedInput, signedValue(signedInput));
    check(UInt128(high, low), unsignedValue(high, low));
  }
}

TEST(Int128Test, floatingPointConstruction) {
  const std::array<double, 9> values{
      -0.5,
      0,
      0.5,
      17.75,
      std::ldexp(1.0, 63),
      std::ldexp(1.0, 64),
      std::ldexp(1.0, 100),
      std::ldexp(1.0, 100) + std::ldexp(1.0, 50),
      std::nextafter(std::ldexp(1.0, 128), 0.0)};
  for (const auto value : values) {
    expectBits(UInt128(value), cpp_int(value));
    expectBits(UInt128(static_cast<long double>(value)), cpp_int(value));
  }
  expectBits(
      UInt128(std::numeric_limits<float>::max()),
      cpp_int(std::numeric_limits<float>::max()));
  expectBits(UInt128(17.75f), cpp_int(17));
  EXPECT_THROW(UInt128(-1.0), std::out_of_range);
  EXPECT_THROW(UInt128(std::ldexp(1.0, 128)), std::out_of_range);
  EXPECT_THROW(
      static_cast<void>(UInt128(std::numeric_limits<double>::infinity())),
      std::out_of_range);
  EXPECT_THROW(
      static_cast<void>(UInt128(std::numeric_limits<double>::quiet_NaN())),
      std::out_of_range);
}

TEST(Int128Test, portableDivision) {
  constexpr auto result =
      detail::divideUnsigned128By64(uint64_t{1} << 63, 0, UINT64_MAX);
  static_assert(result.first == uint64_t{1} << 63);
  static_assert(result.second == uint64_t{1} << 63);

  std::mt19937_64 random(0x64);
  for (int i = 0; i < 2'000; ++i) {
    const uint64_t divisor = random() | 1;
    const uint64_t high = random() % divisor;
    const uint64_t low = random();
    const auto [quotient, remainder] =
        detail::divideUnsigned128By64(high, low, divisor);
    const auto numerator = unsignedValue(high, low);
    SCOPED_TRACE(i);
    EXPECT_EQ(quotient, (numerator / divisor).convert_to<uint64_t>());
    EXPECT_EQ(remainder, (numerator % divisor).convert_to<uint64_t>());
  }
  EXPECT_THROW(detail::divideUnsigned128By64(0, 1, 0), std::invalid_argument);
  EXPECT_THROW(detail::divideUnsigned128By64(1, 0, 1), std::invalid_argument);
}

TEST(Int128Test, unsignedArithmetic) {
  std::mt19937_64 random(0x128);
  for (int i = 0; i < 2'000; ++i) {
    const UInt128 left(random(), random());
    const UInt128 right(i % 2 == 0 ? 0 : random(), random() | 1);
    const auto expectedLeft = unsignedValue(left.high(), left.low());
    const auto expectedRight = unsignedValue(right.high(), right.low());
    SCOPED_TRACE(i);
    expectBits(left + right, expectedLeft + expectedRight);
    expectBits(left - right, expectedLeft - expectedRight);
    expectBits(left * right, expectedLeft * expectedRight);
    expectBits(left / right, expectedLeft / expectedRight);
    expectBits(left % right, expectedLeft % expectedRight);
  }
}

TEST(Int128Test, signedDivision) {
  const std::array<Int128, 6> values{
      Int128(0),
      Int128(1),
      Int128(-1),
      Int128(0, UINT64_MAX),
      std::numeric_limits<Int128>::min(),
      std::numeric_limits<Int128>::max()};
  for (const auto& left : values) {
    for (const auto& right : values) {
      if (right == Int128(0) ||
          (left == std::numeric_limits<Int128>::min() && right == Int128(-1))) {
        continue;
      }
      expectBits(left / right, signedValue(left) / signedValue(right));
      expectBits(left % right, signedValue(left) % signedValue(right));
    }
  }
}

} // namespace
