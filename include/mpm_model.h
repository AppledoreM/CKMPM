#pragma once
#include "mpm_material.cuh"

#include <span>
#include <vector>

namespace mpm
{

template <typename MPMMaterial, size_t kNDim = 3>
class MPMModel
{
   public:
	constexpr static MPMConstitutiveModel kConstitutiveModel_ = MPMMaterial::kConstitutiveModel_;
	typedef typename MPMMaterial::Particle_ Particle_;
	typedef typename MPMMaterial::MaterialParameter_ MaterialParameter_;

	MPMModel() = default;
	MPMModel(float particleMass, float particleVolume, std::span<const Vector<float, 3>> particlePosition,
			 std::span<const Vector<float, 3>> particleVelocity, const MPMMaterial& material, double freezeTime = 0.0)
		: particleMass_(particleMass), particleVolume_(particleVolume), material_(material), freezeTime_(freezeTime)
	{
		particlePosition_ = std::vector(particlePosition.begin(), particlePosition.end());
		particleVelocity_ = std::vector(particleVelocity.begin(), particleVelocity.end());

		assert(particleVelocity_.size() <= particlePosition_.size());
		assert(particleVelocity_.size() == 1 || particleVelocity_.size() == particlePosition.size());

		//TODO
		if constexpr (kConstitutiveModel_ == MPMConstitutiveModel::kLinear)
			{
			}
	}

	inline auto GetParticleMass() const -> float { return particleMass_; }
	inline auto GetParticleVolume() const -> float { return particleVolume_; }

	inline auto GetParticleMaterial() const -> MPMMaterial { return material_; }

	inline auto GetParticleCount() const -> uint32_t { return static_cast<uint32_t>(particlePosition_.size()); }

	inline auto GetParticlePosition() -> std::vector<Vector<float, kNDim>>& { return particlePosition_; }

	inline auto GetParticlePosition() const -> const std::vector<Vector<float, kNDim>>& { return particlePosition_; }

	inline auto GetParticleVelocity() -> std::vector<Vector<float, kNDim>>& { return particleVelocity_; }

	inline auto GetParticleVelocity() const -> const std::vector<Vector<float, kNDim>>& { return particleVelocity_; }

    inline auto IsFreezed(double elapsedSeconds) const -> bool { return  elapsedSeconds < freezeTime_;}

   private:
	float particleMass_ = 0.0;
	float particleVolume_ = 0.0;
    double freezeTime_ = 0.0;
	std::vector<Vector<float, kNDim>> particlePosition_;
	std::vector<Vector<float, kNDim>> particleVelocity_;
	MPMMaterial material_;
};

typedef std::variant<MPMModel<MPMMaterial<MPMConstitutiveModel::kFixedCorotated>>, 
            MPMModel<MPMMaterial<MPMConstitutiveModel::kLinear>>,
            MPMModel<MPMMaterial<MPMConstitutiveModel::kDruckerPragerStvkhencky>>,
            MPMModel<MPMMaterial<MPMConstitutiveModel::kFluid>>,
            MPMModel<MPMMaterial<MPMConstitutiveModel::kVonMises>>,
            MPMModel<MPMMaterial<MPMConstitutiveModel::kNonAssociatedCamClay>>
            > MPMModelVariant;
                

}  // namespace mpm
