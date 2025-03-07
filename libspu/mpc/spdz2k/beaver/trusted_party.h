// Copyright 2021 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <mutex>
#include <optional>

#include "absl/types/span.h"

#include "libspu/mpc/common/prg_tensor.h"

namespace spu::mpc::spdz2k {

class TrustedParty {
 private:
  std::vector<std::optional<PrgSeed>> seeds_;
  mutable std::mutex seeds_mutex_;

 public:
  void setSeed(size_t rank, size_t world_size, const PrgSeed& seed);

  std::vector<PrgSeed> getSeeds() const;

  NdArrayRef adjustSpdzKey(const PrgArrayDesc& descs) const;

  std::vector<NdArrayRef> adjustAuthCoinTossing(const PrgArrayDesc& desc,
                                                const PrgArrayDesc& mac_desc,
                                                uint128_t global_key, size_t k,
                                                size_t s) const;

  std::vector<NdArrayRef> adjustAuthRandBit(const PrgArrayDesc& desc,
                                            const PrgArrayDesc& mac_desc,
                                            uint128_t global_key,
                                            size_t s) const;

  std::vector<NdArrayRef> adjustAuthMul(
      absl::Span<const PrgArrayDesc> descs,
      absl::Span<const PrgArrayDesc> mac_descs, uint128_t global_key) const;

  std::vector<NdArrayRef> adjustAuthDot(
      absl::Span<const PrgArrayDesc> descs,
      absl::Span<const PrgArrayDesc> mac_descs, int64_t m, int64_t n, int64_t k,
      uint128_t global_key) const;

  std::vector<NdArrayRef> adjustAuthHadam(
      absl::Span<const PrgArrayDesc> descs,
      absl::Span<const PrgArrayDesc> mac_descs, int64_t m, int64_t n, 
      uint128_t global_key) const;

  std::vector<NdArrayRef> adjustAuthAnd(
      absl::Span<const PrgArrayDesc> descs,
      absl::Span<const PrgArrayDesc> mac_descs, uint128_t global_key) const;

  std::vector<NdArrayRef> adjustAuthTrunc(
      absl::Span<const PrgArrayDesc> descs,
      absl::Span<const PrgArrayDesc> mac_descs, size_t bits,
      uint128_t global_key, size_t k, size_t s) const;
};

}  // namespace spu::mpc::spdz2k
