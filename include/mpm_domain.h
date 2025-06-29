#pragma once
#include <limits>
#include <utility>
#include "base.h"

namespace mpm
{

template <int... kOffset>
struct MPMDomainOffset
{
	constexpr static uint32_t kNDim_ = sizeof...(kOffset);
	constexpr static int kOffset_[sizeof...(kOffset)] = {kOffset...};
};

template <int... kDim>
struct MPMDomainRange
{
	constexpr static uint32_t kNDim_ = sizeof...(kDim);
	constexpr static int kDim_[sizeof...(kDim)] = {kDim...};
	constexpr static uint32_t kGridBlockCount_ = (kDim * ...);

	constexpr static auto IsInfiniteDomainRange() -> bool
	{
		bool finite = true;
		for (uint32_t i = 0; i < kNDim_; ++i)
			finite &= (kDim_[i] != std::numeric_limits<int>::max());
		return !finite;
	}
};

namespace internal
{
struct MPMDomainTag
{
};
template <int... kDummyRange>
constexpr auto MPMInfiniteDomainRangeImplHelper(std::integer_sequence<int, kDummyRange...>)
	-> MPMDomainRange<(std::numeric_limits<int>::max() + 0 * kDummyRange)...>
{
	return {};
}

template <uint32_t kNDim>
requires(1 <= kNDim <= 3) using MPMInfiniteDomainRangeImpl =
	decltype(MPMInfiniteDomainRangeImplHelper(std::make_index_sequence<kNDim>{}));

}  // namespace internal

template <uint32_t kNDim>
using MPMInfiniteDomainRange = internal::MPMInfiniteDomainRangeImpl<kNDim>;

template <typename DomainRange, typename DomainOffset = MPMDomainOffset<0, 0, 0>, int kBoundarySize = -1>
requires(DomainRange::kNDim_ == DomainOffset::kNDim_) struct MPMDomain : internal::MPMDomainTag
{
	typedef DomainRange DomainRange_;
	typedef DomainOffset DomainOffset_;
	constexpr static uint32_t kNDim_ = DomainRange::kNDim_;
	constexpr static int kBoundarySize_ = kBoundarySize;
	constexpr static uint32_t kGridBlockCount_ = DomainRange::kGridBlockCount_;
};

template <typename Domain>
concept MPMDomainType = std::is_base_of_v<internal::MPMDomainTag, Domain>;

template <typename Domain>
concept MPMFiniteDomainType = !Domain::DomainRange_::IsInfiniteDomainRange() && MPMDomainType<Domain>;

template <typename Domain>
concept MPMInfiniteDomainType = Domain::DomainRange_::IsInfiniteDomainRange() && MPMDomainType<Domain>;

}  // namespace mpm
