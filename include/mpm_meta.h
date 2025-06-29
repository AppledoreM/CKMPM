#pragma once
#include <type_traits>
#include "base.h"

namespace mpm::meta
{
struct Empty
{
	Empty() = default;
};

namespace internal
{
template <typename T, T Index>
struct ConstexprIndex
{
	static constexpr T kIndex_ = Index;
};

template <int kBegin, int kStep, int... Indices>
MPM_HOST_DEV_FUNC MPM_FORCE_INLINE auto ConstexprLoopImpl(auto func, std::integer_sequence<int, Indices...>) -> void
{
	(func(internal::ConstexprIndex<int, Indices * kStep + kBegin>{}), ...);
}
}  // namespace internal

template <typename T>
constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto ConstexprLoopIndex(T x) -> int
{
	return T::kIndex_;
}

template <int kBegin, int kEnd, int kStep = 1>
MPM_HOST_DEV_FUNC MPM_FORCE_INLINE auto ConstexprLoop(auto func) -> void
{
	internal::ConstexprLoopImpl<kBegin, kStep>(
		func, std::make_integer_sequence<int, int(double(kEnd - kBegin) / kStep + 0.5)>{});
}

template <typename... Attribute>
struct AttributeWrapper
{
   private:
	template <size_t I, size_t N, typename... Ts>
	requires(I <= N) struct AttributeOffsetWrapper
	{
	   private:
		consteval inline static auto ComputeOffset() -> size_t
		{
			if constexpr (I == 0)
				return 0;
			else
				return AttributeOffsetWrapper<I - 1, N, Ts...>::kOffset_ +
					   sizeof(std::tuple_element_t<I - 1, std::tuple<Ts...>>);
		}

	   public:
		constexpr static size_t kOffset_ = ComputeOffset();
	};

   public:
	template <size_t AttributeIndex>
	struct AttributeWrapperAccessor
	{
		typedef std::tuple_element_t<AttributeIndex, std::tuple<Attribute...>> Type;
		constexpr static size_t kOffset_ =
			AttributeOffsetWrapper<AttributeIndex, AttributeIndex, Attribute...>::kOffset_;
		constexpr static size_t kSize_ = sizeof(Type);
	};

	constexpr static size_t kAttributeCount_ = sizeof...(Attribute);
	constexpr static size_t kSize_ = (sizeof(Attribute) + ...);
};
}  // namespace mpm::meta
