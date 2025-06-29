#pragma once
#include "data_type.cuh"

#include <cuda.h>
#include <array>
#include <cstdio>
#include <numeric>

namespace mpm
{
namespace detail
{
constexpr static auto kAutoAlignment = std::numeric_limits<size_t>::max();
};

template <size_t... Dimensions>
struct GridDimension
{
	constexpr static size_t kDimensions_[] = {Dimensions...};
	constexpr static size_t kNDim_ = sizeof...(Dimensions);

	MPM_HOST_DEV_FUNC constexpr static auto GetSize() -> size_t
	{
		size_t size = 1;
		for (size_t i = 0; i < kNDim_; ++i)
			size *= kDimensions_[i];
		return size;
	}
	MPM_HOST_DEV_FUNC constexpr static auto GetDimension(size_t index) -> size_t { return kDimensions_[index]; }
};

namespace detail
{
template <typename T>
struct ScalarTypeHelper
{
	typedef typename T::ScalarType_ ScalarType_;
};

template <typename T>
requires std::is_scalar_v<T> struct ScalarTypeHelper<T>
{
	typedef T ScalarType_;
};

template <typename T>
struct DataTypeSizeHelper
{
	constexpr static auto kSize_ = T::kSize_;
};

template <typename T>
requires std::is_scalar_v<T> struct DataTypeSizeHelper<T>
{
	constexpr static auto kSize_ = 1;
};
}  // namespace detail

template <typename DataType, typename Dimensions, size_t kAlignment = detail::kAutoAlignment>
class DenseGrid
{
   public:
	typedef detail::ScalarTypeHelper<DataType>::ScalarType_ ScalarType_;
	typedef DataType DataType_;
	typedef Dimensions Dimensions_;

	constexpr static auto kSize_ = detail::DataTypeSizeHelper<DataType>::kSize_;
	constexpr static auto kAlignment_ = kAlignment;

	MPM_HOST_DEV_FUNC DenseGrid()
	{
		cudaMalloc((void**)&data_, Dimensions_::GetSize() * sizeof(DataType_));
		cudaDeviceSynchronize();
	}

	MPM_HOST_DEV_FUNC ~DenseGrid() = default;
	MPM_HOST_DEV_FUNC auto Destroy() -> void { cudaFree(data_); }

	template <typename T, size_t kN>
	requires std::is_integral_v<T> MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto operator[](const Vector<T, kN>& index)
		-> DataType_&
	{
		size_t index_ = 0;
		if constexpr (Dimensions_::kNDim_ == 2)
			{
				index_ = index[0] * Dimensions_::kDimensions_[1] + index[1];
			}
		else if constexpr (Dimensions_::kNDim_ == 3)
			{
				index_ = index[0] * Dimensions_::kDimensions_[1] * Dimensions_::kDimensions_[2] +
						 index[1] * Dimensions_::kDimensions_[2] + index[2];
			}
		return *reinterpret_cast<DataType_*>(&data_[index_ * kSize_]);
	}

	template <typename T, size_t kN>
	requires std::is_integral_v<T> MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto operator[](const Vector<T, kN>& index) const
		-> const DataType_&
	{
		size_t index_ = 0;
		if constexpr (Dimensions_::kNDim_ == 2)
			{
				index_ = index[0] * Dimensions_::kDimensions_[1] + index[1];
			}
		else if constexpr (Dimensions_::kNDim_ == 3)
			{
				index_ = index[0] * Dimensions_::kDimensions_[1] * Dimensions_::kDimensions_[2] +
						 index[1] * Dimensions_::kDimensions_[2] + index[2];
			}
		return *reinterpret_cast<DataType_*>(&data_[index_ * kSize_]);
	}

   public:
	ScalarType_* data_ = nullptr;
};

}  // namespace mpm
