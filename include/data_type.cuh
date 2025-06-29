#pragma once

#include "base.h"
#include "mpm_meta.h"

#include <cuda.h>
#include <cstdio>
#include <type_traits>

namespace mpm
{

namespace detail
{

template <typename T, typename U>
using BinaryScalarOpPromoteType = decltype(T(0) + U(0));
}  // namespace detail

/*
 *
 *   Matrix Implementation
 *
 */
template <typename Scalar, size_t kN, size_t kM>
class Matrix;

template <typename Scalar, size_t kN>
using Vector = Matrix<Scalar, kN, 1>;

template <typename Scalar, size_t kN, size_t kM>
class Matrix
{
   public:
	typedef Scalar Scalar_;
	constexpr static auto kN_ = kN;
	constexpr static auto kM_ = kM;
	constexpr static auto kSize_ = kN * kM;

	constexpr Matrix() = default;

	template <typename... Scalars>
		requires(sizeof...(Scalars) == kN * kM) && (std::is_scalar_v<Scalars> && ...) &&
		(std::is_convertible_v<Scalars, Scalar> && ...) constexpr MPM_HOST_DEV_FUNC explicit Matrix(Scalars... data)
		: data_{static_cast<Scalar>(data)...}
	{
	}

	constexpr Matrix(const Matrix& rhs) = default;

	template <typename T>
		requires std::is_convertible_v<T, Scalar_> && (!std::is_same_v<T, Scalar_>)constexpr MPM_HOST_DEV_FUNC
		Matrix(const Matrix<T, kN_, kM_>& rhs)
	{
		meta::ConstexprLoop<0, kSize_>(
			[&](auto indexWrapper) -> void
			{
				constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
				data_[index] = static_cast<Scalar_>(rhs[index]);
			});
	}

	constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto operator[](size_t index) -> Scalar_& { return data_[index]; }
	constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto operator[](size_t index) const -> const Scalar_
	{
		return data_[index];
	}

	template <typename T>
	requires std::is_scalar_v<T>&& std::is_convertible_v<T, Scalar_> constexpr MPM_HOST_DEV_FUNC auto operator=(T rhs)
		-> Matrix&
	{
		meta::ConstexprLoop<0, kSize_>(
			[&](auto indexWrapper) -> void
			{
				constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
				data_[index] = rhs;
			});

		return *this;
	}

	template <typename T>
	requires std::is_convertible_v<T, Scalar_> MPM_HOST_DEV_FUNC auto operator=(const Matrix<T, kN_, kM_>& rhs)
		-> Matrix&
	{
		meta::ConstexprLoop<0, kSize_>(
			[&](auto indexWrapper) -> void
			{
				constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
				data_[index] = rhs[index];
			});
		return *this;
	}

	constexpr MPM_HOST_DEV_FUNC auto operator-() const -> Matrix
	{
		Matrix res{};
		meta::ConstexprLoop<0, kSize_>(
			[&](auto indexWrapper) -> void
			{
				constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
				res[index] = -data_[index];
			});

		return res;
	}

	template <typename T>
	MPM_HOST_DEV_FUNC auto operator+=(const Matrix<T, kN_, kM_>& rhs) -> void
	{
		meta::ConstexprLoop<0, kSize_>(
			[&](auto indexWrapper) -> void
			{
				constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
				data_[index] += rhs[index];
			});
	}

	template <typename T>
	MPM_HOST_DEV_FUNC auto operator-=(const Matrix<T, kN_, kM_>& rhs) -> void
	{
		meta::ConstexprLoop<0, kSize_>(
			[&](auto indexWrapper) -> void
			{
				constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
				data_[index] -= rhs[index];
			});
	}

	template <typename T>
	MPM_HOST_DEV_FUNC auto operator*=(const Matrix<T, kN_, kM_>& rhs) -> void
	{
		meta::ConstexprLoop<0, kSize_>(
			[&](auto indexWrapper) -> void
			{
				constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
				data_[index] *= rhs[index];
			});
	}

	template <typename T>
	MPM_HOST_DEV_FUNC auto operator/=(const Matrix<T, kN_, kM_>& rhs) -> void
	{
		meta::ConstexprLoop<0, kSize_>(
			[&](auto indexWrapper) -> void
			{
				constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
				data_[index] /= rhs[index];
			});
	}

	template <typename T>
	requires std::is_scalar_v<T> MPM_HOST_DEV_FUNC auto operator+=(T rhs) -> void
	{
		meta::ConstexprLoop<0, kSize_>(
			[&](auto indexWrapper) -> void
			{
				constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
				data_[index] += rhs;
			});
	}

	template <typename T>
	requires std::is_scalar_v<T> MPM_HOST_DEV_FUNC auto operator-=(T rhs) -> void
	{
		meta::ConstexprLoop<0, kSize_>(
			[&](auto indexWrapper) -> void
			{
				constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
				data_[index] -= rhs;
			});
	}

	template <typename T>
	requires std::is_scalar_v<T> MPM_HOST_DEV_FUNC auto operator*=(T rhs) -> void
	{
		meta::ConstexprLoop<0, kSize_>(
			[&](auto indexWrapper) -> void
			{
				constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
				data_[index] *= rhs;
			});
	}

	template <typename T>
	requires std::is_scalar_v<T> MPM_HOST_DEV_FUNC auto operator/=(T rhs) -> void
	{
		meta::ConstexprLoop<0, kSize_>(
			[&](auto indexWrapper) -> void
			{
				constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
				data_[index] /= rhs;
			});
	}

	template <typename T>
	MPM_HOST_DEV_FUNC auto operator+(const Matrix<T, kN_, kM_>& rhs) const
		-> Matrix<detail::BinaryScalarOpPromoteType<Scalar_, T>, kN_, kM_>
	{
		Matrix<detail::BinaryScalarOpPromoteType<Scalar_, T>, kN_, kM_> res = {};
		meta::ConstexprLoop<0, kSize_>(
			[&](auto indexWrapper) -> void
			{
				constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
				res[index] = data_[index] + rhs[index];
			});
		return res;
	}

	template <typename T>
	MPM_HOST_DEV_FUNC auto operator-(const Matrix<T, kN_, kM_>& rhs) const
		-> Matrix<detail::BinaryScalarOpPromoteType<Scalar_, T>, kN_, kM_>
	{
		Matrix<detail::BinaryScalarOpPromoteType<Scalar_, T>, kN_, kM_> res = {};
		meta::ConstexprLoop<0, kSize_>(
			[&](auto indexWrapper) -> void
			{
				constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
				res[index] = data_[index] - rhs[index];
			});
		return res;
	}

	template <typename T>
	MPM_HOST_DEV_FUNC auto operator*(const Matrix<T, kN_, kM_>& rhs) const
		-> Matrix<detail::BinaryScalarOpPromoteType<Scalar_, T>, kN_, kM_>
	{
		Matrix<detail::BinaryScalarOpPromoteType<Scalar_, T>, kN_, kM_> res = {};
		meta::ConstexprLoop<0, kSize_>(
			[&](auto indexWrapper) -> void
			{
				constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
				res[index] = data_[index] * rhs[index];
			});
		return res;
	}

	template <typename T>
	MPM_HOST_DEV_FUNC auto operator/(const Matrix<T, kN_, kM_>& rhs) const
		-> Matrix<detail::BinaryScalarOpPromoteType<Scalar_, T>, kN_, kM_>
	{
		Matrix<detail::BinaryScalarOpPromoteType<Scalar_, T>, kN_, kM_> res = {};
		meta::ConstexprLoop<0, kSize_>(
			[&](auto indexWrapper) -> void
			{
				constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
				res[index] = data_[index] / rhs[index];
			});
		return res;
	}

	/*
     *  Atomic Operations
     */
	template <typename T>
	MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto AtomicAddition(const Matrix<T, kN_, kM_>& rhs) -> void
	{
		meta::ConstexprLoop<0, kSize_>(
			[&](auto indexWrapper) -> void
			{
				constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
				atomicAdd(&data_[index], rhs[index]);
			});
	}

	/*
     * Matrix special operations
     */

	template <typename T, size_t kP>
	requires(kN_ != 1 || kP != 1) MPM_FORCE_INLINE MPM_HOST_DEV_FUNC
		auto MatrixMultiplication(const Matrix<T, kM_, kP>& rhs) const
		-> Matrix<detail::BinaryScalarOpPromoteType<Scalar_, T>, kN_, kP>
	{
		Matrix<detail::BinaryScalarOpPromoteType<Scalar_, T>, kN_, kP> res = {};

		meta::ConstexprLoop<0, kN_>(
			[&](auto indexWrapper0) -> void
			{
				meta::ConstexprLoop<0, kM_>(
					[&](auto indexWrapper1) -> void
					{
						meta::ConstexprLoop<0, kP>(
							[&](auto indexWrapper2) -> void
							{
								constexpr int i = meta::ConstexprLoopIndex(indexWrapper0);
								constexpr int j = meta::ConstexprLoopIndex(indexWrapper1);
								constexpr int k = meta::ConstexprLoopIndex(indexWrapper2);
								res[i * kP + k] += data_[i * kM_ + j] * rhs[j * kP + k];
							});
					});
			});
		return res;
	}

	constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC static auto Identity() -> Matrix<Scalar_, kN_, kN_> requires(kN_ ==
																											  kM_)
	{
		Matrix<Scalar_, kN_, kN_> Id;
		meta::ConstexprLoop<0, kN_>(
			[&](auto indexWrapper) -> void
			{
				constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
				Id[index * kN_ + index] = static_cast<Scalar_>(1);
			});
		return Id;
	}

	template <typename T, size_t kP, size_t kQ>
		requires(kN_ == 1 || kM_ == 1) && (kP == 1 || kQ == 1) &&
		(kN_ * kM_ == kP * kQ) constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC
		auto Dot(const Matrix<T, kP, kQ>& rhs) const -> detail::BinaryScalarOpPromoteType<Scalar_, T>
	{
		detail::BinaryScalarOpPromoteType<Scalar_, T> res = 0;

		meta::ConstexprLoop<0, kN_ * kM_>(
			[&](auto indexWrapper) -> void
			{
				constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
				res += data_[index] * rhs[index];
			});
		return res;
	}

	template <typename T, size_t kP, size_t kQ>
		requires(kN_ == 1 || kM_ == 1) && (kP == 1 || kQ == 1) &&
		(kN_ * kM_ == kP * kQ) constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC
		auto OuterProduct(const Matrix<T, kP, kQ>& rhs) const
		-> Matrix<detail::BinaryScalarOpPromoteType<Scalar_, T>, kN_ * kM_, kP * kQ>
	{
		Matrix<detail::BinaryScalarOpPromoteType<Scalar_, T>, kN_ * kM_, kP * kQ> res;

		meta::ConstexprLoop<0, kN_ * kM_ * kP * kQ>(
			[&](auto indexWrapper) -> void
			{
				constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
				constexpr int lhsIndex = index / (kP * kQ);
				constexpr int rhsIndex = index % (kP * kQ);
				res[index] = data_[lhsIndex] * rhs[rhsIndex];
			});
		return res;
	}

	constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto Norm() const -> Scalar_
	{
		auto res = Scalar_(0);
		meta::ConstexprLoop<0, kN_ * kM_>(
			[&](auto indexWrapper) -> void
			{
				constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
				res += data_[index] * data_[index];
			});
		return sqrt(res);
	}

	constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto Trace() const -> Scalar_ requires(kN_ == kM_)
	{
		auto res = Scalar_(0);
		meta::ConstexprLoop<0, kN_>(
			[&](auto indexWrapper) -> void
			{
				constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
				res += data_[index * kN_ + index];
			});
		return res;
	}

	constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto Inverse() const -> Matrix<Scalar_, kN_, kM_> requires(kN_ ==
																												kM_ &&
																											kM_ == 3)
	{
		Matrix<Scalar_, kN_, kM_> res;
		const auto inverseDeterminant = static_cast<Scalar_>(1.0f) / Determinant();

		res[0] = inverseDeterminant * (data_[4] * data_[8] - data_[5] * data_[7]);
		res[1] = inverseDeterminant * (data_[2] * data_[7] - data_[1] * data_[8]);
		res[2] = inverseDeterminant * (data_[1] * data_[5] - data_[2] * data_[4]);
		res[3] = inverseDeterminant * (data_[5] * data_[6] - data_[3] * data_[8]);
		res[4] = inverseDeterminant * (data_[0] * data_[8] - data_[2] * data_[6]);
		res[5] = inverseDeterminant * (data_[2] * data_[3] - data_[0] * data_[5]);
		res[6] = inverseDeterminant * (data_[3] * data_[7] - data_[4] * data_[6]);
		res[7] = inverseDeterminant * (data_[1] * data_[6] - data_[0] * data_[7]);
		res[8] = inverseDeterminant * (data_[0] * data_[4] - data_[1] * data_[3]);

		return res;
	}

	constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto Transpose() const -> Matrix<Scalar_, kM_, kN_>
	{
		Matrix<Scalar_, kM_, kN_> res;
		meta::ConstexprLoop<0, kN_ * kM_>(
			[&](auto indexWrapper) -> void
			{
				constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
				constexpr int transposedIndex = (index % kM_) * kN_ + (index / kM_);
				res[index] = data_[transposedIndex];
			});
		return res;
	}

	constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto Determinant() const -> Scalar_
		requires((kN_ == kM_) && (kN_ == 2))
	{
		return data_[0] * data_[3] - data_[1] * data_[2];
	}

	constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto Determinant() const -> Scalar_
		requires((kN_ == kM_) && (kN_ == 3))
	{
		auto res = data_[0] * data_[4] * data_[8] + data_[1] * data_[5] * data_[6] + data_[2] * data_[3] * data_[7];
		res -= data_[2] * data_[4] * data_[6] + data_[1] * data_[3] * data_[8] + data_[0] * data_[5] * data_[7];
		return res;
	}

	MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto Data() -> Scalar_* { return data_; }

	MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto Data() const -> const Scalar_*
	{
		return static_cast<const Scalar_*>(data_);
	}

   private:
	Scalar_ data_[kN * kM] = {};
};

/*
     * Matrix & Scalar Operations
     */

template <typename T, size_t kN, size_t kM, typename U>
requires std::is_scalar_v<U> MPM_HOST_DEV_FUNC auto operator+(const Matrix<T, kN, kM>& lhs, const U rhs)
	-> Matrix<detail::BinaryScalarOpPromoteType<T, U>, kN, kM>
{
	Matrix<detail::BinaryScalarOpPromoteType<T, U>, kN, kM> res(lhs);
	res += rhs;
	return res;
}
template <typename T, size_t kN, size_t kM, typename U>
requires std::is_scalar_v<U> MPM_HOST_DEV_FUNC auto operator+(const U lhs, const Matrix<T, kN, kM>& rhs)
	-> Matrix<detail::BinaryScalarOpPromoteType<T, U>, kN, kM>
{
	Matrix<detail::BinaryScalarOpPromoteType<T, U>, kN, kM> res(rhs);
	res += lhs;
	return res;
}

template <typename T, size_t kN, size_t kM, typename U>
requires std::is_scalar_v<U> MPM_HOST_DEV_FUNC auto operator-(const Matrix<T, kN, kM>& lhs, const U rhs)
	-> Matrix<detail::BinaryScalarOpPromoteType<T, U>, kN, kM>
{
	Matrix<detail::BinaryScalarOpPromoteType<T, U>, kN, kM> res(lhs);
	res -= rhs;
	return res;
}
template <typename T, size_t kN, size_t kM, typename U>
requires std::is_scalar_v<U> MPM_HOST_DEV_FUNC auto operator-(const U lhs, const Matrix<T, kN, kM>& rhs)
	-> Matrix<detail::BinaryScalarOpPromoteType<T, U>, kN, kM>
{
	Matrix<detail::BinaryScalarOpPromoteType<T, U>, kN, kM> res;
	meta::ConstexprLoop<0, kN * kM>(
		[&](auto indexWrapper) -> void
		{
			constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
			res[index] = lhs - rhs[index];
		});
	return res;
}

template <typename T, size_t kN, size_t kM, typename U>
requires std::is_scalar_v<U> MPM_HOST_DEV_FUNC auto operator*(const Matrix<T, kN, kM>& lhs, const U rhs)
	-> Matrix<detail::BinaryScalarOpPromoteType<T, U>, kN, kM>
{
	Matrix<detail::BinaryScalarOpPromoteType<T, U>, kN, kM> res(lhs);
	res *= rhs;
	return res;
}
template <typename T, size_t kN, size_t kM, typename U>
requires std::is_scalar_v<U> MPM_HOST_DEV_FUNC auto operator*(const U lhs, const Matrix<T, kN, kM>& rhs)
	-> Matrix<detail::BinaryScalarOpPromoteType<T, U>, kN, kM>
{
	Matrix<detail::BinaryScalarOpPromoteType<T, U>, kN, kM> res(rhs);
	res *= lhs;
	return res;
}

template <typename T, size_t kN, size_t kM, typename U>
requires std::is_scalar_v<U> MPM_HOST_DEV_FUNC auto operator/(const Matrix<T, kN, kM>& lhs, const U rhs)
	-> Matrix<detail::BinaryScalarOpPromoteType<T, U>, kN, kM>
{
	Matrix<detail::BinaryScalarOpPromoteType<T, U>, kN, kM> res(lhs);
	res /= rhs;
	return res;
}

template <typename T, size_t kN, size_t kM, typename U>
requires std::is_scalar_v<U> MPM_HOST_DEV_FUNC auto operator/(const U lhs, const Matrix<T, kN, kM>& rhs)
	-> Matrix<detail::BinaryScalarOpPromoteType<T, U>, kN, kM>
{
	Matrix<detail::BinaryScalarOpPromoteType<T, U>, kN, kM> res{};
	meta::ConstexprLoop<0, kN * kM>(
		[&](auto indexWrapper) -> void
		{
			constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
			res[index] = lhs / rhs[index];
		});
	return res;
}

};	// namespace mpm
