#pragma once
#include "base.h"
#include "mpm_meta.h"

#include <cuda.h>
#include <stdio.h>
#include <algorithm>
#include <array>
#include <type_traits>

namespace mpm
{

template <typename T>
MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto sgn(T val) -> bool
{
	return (T(0) < val) - (val < T(0));
}

template <int kFactor, typename Scalar>
requires std::is_integral_v<Scalar> MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto NextNearestMultipleOf(Scalar x) -> Scalar
{
	return ((x + kFactor - 1) / kFactor) * kFactor;
}

namespace internal
{
template <int kN>
constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto ConstexprMultiplyLastN() -> int
{
	return 1;
}

template <int kN, int kX, int... kY>
requires(1 <= kN) constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto ConstexprMultiplyLastN() -> int
{
	if constexpr (kN >= 1 + sizeof...(kY))
		return kX * ConstexprMultiplyLastN<kN, kY...>();
	return ConstexprMultiplyLastN<kN, kY...>();
}

template <int kN, int... kX>
requires(0 <= kN && kN <= sizeof...(kX)) struct FlattenMultiplier
{
	constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto static GetMultiplier() -> int
	{
		if constexpr (kN == sizeof...(kX))
			return 1;
		else
			{
				return ConstexprMultiplyLastN<sizeof...(kX) - kN, kX...>();
			}
	}
};

}  // namespace internal

template <int... kSizes, typename... Coordinate>
requires(sizeof...(kSizes) + 1 == sizeof...(Coordinate) &&
		 (std::is_same_v<Coordinate, int> && ...)) constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC
	auto GetFlattenedIndex(Coordinate... x) -> int
{
	const int x_[] = {x...};
	int flattenedIndex = 0;
	meta::ConstexprLoop<0, sizeof...(Coordinate), 1>(
		[&flattenedIndex, &x_](auto indexWrapper) -> void
		{
			constexpr int index = decltype(indexWrapper)::kIndex_;
			flattenedIndex += internal::FlattenMultiplier<index, kSizes...>::GetMultiplier() * x_[index];
		});
	return flattenedIndex;
}

}  // namespace mpm
