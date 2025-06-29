#pragma once
#include "data_type.cuh"
#include "mpm_math.cuh"
#include "mpm_meta.h"

#include <cmath>
#include <limits>
#include <numbers>
#include <type_traits>

namespace mpm
{

namespace detail
{

template <typename T>
MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto sgn(T val) -> T
{
	return copysign(static_cast<T>(1.0f), val);
}
struct MPMKernelTag
{
};
template <typename T>
concept KernelType = std::is_base_of_v<MPMKernelTag, T>;

}  // namespace detail

template <typename ScalarType, int Sign, size_t NDim = 3>
struct SmoothLinear : detail::MPMKernelTag
{
	typedef ScalarType ScalarType_;
	constexpr static int kSign_ = Sign;
	constexpr static size_t kNDim_ = NDim;
	constexpr static ScalarType_ kOffset_ = static_cast<ScalarType>(0.25) * kSign_;
	constexpr static size_t kKernelRadius_ = 2;

	MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto static Weight1D(ScalarType_ w) -> ScalarType_
	{
		constexpr ScalarType_ inv_two_pi = static_cast<ScalarType>(1.0) / (2 * std::numbers::pi_v<ScalarType_>);

		return (ScalarType_(1) - abs(w) + inv_two_pi * sinpif(static_cast<ScalarType>(2.0) * abs(w)));
		/* * (abs(w) <= static_cast<ScalarType>(1)); */
	}

	MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto static Weight(const mpm::Vector<ScalarType_, NDim>& dW) -> ScalarType_
	{
		ScalarType_ res = 1.0;
#pragma unroll NDim
		for (size_t i = 0; i < NDim; ++i)
			res *= Weight1D(dW[i]);

		return res;
	}

	MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto static WeightStencil(const mpm::Vector<ScalarType_, NDim>& dW)
		-> mpm::Matrix<ScalarType_, kKernelRadius_, NDim>
	{
		auto stencil = mpm::Matrix<ScalarType_, kKernelRadius_, NDim>{};

		meta::ConstexprLoop<0, 3>(
			[&dW, &stencil](auto indexWrapper) -> void
			{
				constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
				stencil[index * 2] = Weight1D(dW[index] - kOffset_);
				stencil[index * 2 + 1] = Weight1D(dW[index] - 1 - kOffset_);
			});
		return stencil;
	}

	MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto static Gradient1D(ScalarType_ w) -> ScalarType_
	{
		return detail::sgn(w) * (cospif(2.f * abs(w)) - 1.f);
		/* * (abs(w) <= static_cast<ScalarType>(1)); */
	}

	MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto static Gradient(const mpm::Vector<ScalarType_, NDim>& dW)
		-> mpm::Vector<ScalarType_, NDim>
	{
		const auto gx = Gradient1D(dW[0] - kOffset_);
		const auto gy = Gradient1D(dW[1] - kOffset_);
		const auto gz = Gradient1D(dW[2] - kOffset_);
		const auto kx = Weight1D(dW[0] - kOffset_);
		const auto ky = Weight1D(dW[1] - kOffset_);
		const auto kz = Weight1D(dW[2] - kOffset_);

		return mpm::Vector<ScalarType_, NDim>{gx * ky * kz, gy * kx * kz, gz * kx * ky};
	}

	MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto static OptimizedWeightComputation(
		const mpm::Vector<ScalarType_, NDim>& dW, mpm::Vector<ScalarType_, NDim>& weightStencil) -> void
	{

		meta::ConstexprLoop<0, kNDim_>(
			[&](auto indexWrapper) -> void
			{
				constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
                // For unit-test 
                {
                    // constexpr double inv2Pi = static_cast<double>(1.0) / (2.0 * std::numbers::pi_v<double>);
                    // double offset = dW[index] - kOffset_;
                    // weightStencil[index] = 1 - offset - inv2Pi * sinpi(2 * offset - 1);
                }

                {
                    constexpr float inv2Pi = static_cast<float>(1.0) / (2.0 * std::numbers::pi_v<double>);
                    constexpr float twoPi = 2.0 * std::numbers::pi_v<double>;
                    ScalarType_ offset = dW[index] - kOffset_;
                    weightStencil[index] = 1.f - offset + inv2Pi * __sinf(twoPi * offset);
                }

			});
	}
};

template <detail::KernelType T>
class MPMKernel
{
   public:
	typedef T KernelType_;

	MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto static Weight1D(typename T::ScalarType_ w) -> typename T::ScalarType_
	{
        return KernelType_::Weight1D(w);
	}

	MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto static Gradient1D(typename T::ScalarType_ w) -> typename T::ScalarType_
	{
        return KernelType_::Gradient1D(w);
	}

	MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto static Weight(const Vector<typename T::ScalarType_, T::kNDim_>& dW) ->
		typename T::ScalarType_
	{
		return KernelType_::Weight(dW);
	}

	MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto static WeightStencil(const Vector<typename T::ScalarType_, T::kNDim_>& dW)
		-> Matrix<typename T::ScalarType_, T::kKernelRadius_, T::kNDim_>
	{
		return KernelType_::WeightStencil(dW);
	}

	MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto static Gradient(const Vector<typename T::ScalarType_, T::kNDim_>& dW)
		-> Vector<typename T::ScalarType_, T::kNDim_>
	{
		return KernelType_::Gradient(dW);
	}

	MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto static OptimizedWeightComputation(
		const mpm::Vector<typename T::ScalarType_, T::kNDim_>& dW,
		mpm::Vector<typename T::ScalarType_, T::kNDim_>& weightStencil) -> void
	{
		return KernelType_::OptimizedWeightComputation(dW, weightStencil);
	}
};

}  // namespace mpm
