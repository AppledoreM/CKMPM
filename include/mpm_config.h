#pragma once
#include "base.h"

#include <cstdint>
#include <string>
#include <type_traits>

namespace mpm
{

template <typename DerivedConfig>
class MPMConfigBase
{
   public:
	constexpr MPMConfigBase() = default;

	constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetDx() const -> float
	{
		return (static_cast<const DerivedConfig*>(this))->GetDxImpl();
	}

	constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetInvDx() const -> float
	{
		return 1.0 / (static_cast<const DerivedConfig*>(this))->GetDxImpl();
	}

	constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetBlockVolume() const -> uint32_t
	{
		return (static_cast<const DerivedConfig*>(this))->GetBlockVolumeImpl();
	}

	constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetBlockSize() const -> uint32_t
	{
		return (static_cast<const DerivedConfig*>(this))->GetBlockSizeImpl();
	}

	constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxParticleCountPerBlock() const -> uint32_t
	{
		return (static_cast<const DerivedConfig*>(this))->GetMaxParticleCountPerBlockImpl();
	}

	constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxParticleCountPerCell() const -> uint32_t
	{
		return (static_cast<const DerivedConfig*>(this))->GetMaxParticleCountPerCellImpl();
	}

	constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetParticleBucketSize() const -> uint32_t
	{
		return (static_cast<const DerivedConfig*>(this))->GetParticleBucketSizeImpl();
	}

	constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxParticleCountPerBucket() const -> uint32_t
	{
		return (static_cast<const DerivedConfig*>(this))->GetMaxParticleCountPerBucketImpl();
	}

	constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxActiveBlockCount() const -> uint32_t
	{
		return (static_cast<const DerivedConfig*>(this))->GetMaxActiveBlockCountImpl();
	}

	constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetFps() const -> uint32_t
	{
		return (static_cast<const DerivedConfig*>(this))->GetFpsImpl();
	}

	constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetDt() const -> float
	{
		return (static_cast<const DerivedConfig*>(this))->GetDtImpl();
	}

	constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetCfl() const -> float
	{
		return (static_cast<const DerivedConfig*>(this))->GetCflImpl();
	}

	constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetTotalSimulatedFrameCount() const -> uint32_t
	{
		return (static_cast<const DerivedConfig*>(this))->GetTotalSimulatedFrameCountImpl();
	}

	constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetGravity() const -> float
	{
		return (static_cast<const DerivedConfig*>(this))->GetGravityImpl();
	}

	constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMassClamp() const -> float
	{
		return (static_cast<const DerivedConfig*>(this))->GetMassClampImpl();
	}

	constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetExistRigidParticle() const -> bool
	{
		return (static_cast<const DerivedConfig*>(this))->GetExistRigidParticleImpl();
	}

	constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetRigidParticleCount() const -> uint32_t
	{
		return (static_cast<const DerivedConfig*>(this))->GetRigidParticleCountImpl();
	}

    template<typename Scalar, typename... Args>
	constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetRigidParticleVelocity(Args&&... args) const -> Vector<Scalar, 3>
	{
		return (static_cast<const DerivedConfig*>(this))->template GetRigidParticleVelocityImpl<Scalar>(std::forward<Args...>(args...));
	}

    constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetExistIrregularBoundary() const -> bool
    {
        return (static_cast<const DerivedConfig*>(this))->GetExistIrregularBoundaryImpl();
    }

    template<typename... Args>
    MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto ProcessGridCellVelocity(const Vector<int, 3>& cell, Vector<float, 3>& velocity, Args&&... args) const -> void
    {
		(static_cast<const DerivedConfig*>(this))->ProcessGridCellVelocityImpl(cell, velocity, std::forward<Args...>(args...));
    }

    // template<typename... Args>
    MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto UpdateConfig(float dt, int frame) -> void
    {
        (static_cast<DerivedConfig*>(this))->UpdateConfigImpl(dt, frame);
    }
};

struct MPMDx
{
	constexpr MPMDx(float kDx) : kDx_(kDx) {}

	constexpr auto GetDx() const -> float { return kDx_; }

	float kDx_;
};

template <typename T>
concept MPMConfigType = std::is_base_of_v<MPMConfigBase<T>, T>;

}  // namespace mpm
