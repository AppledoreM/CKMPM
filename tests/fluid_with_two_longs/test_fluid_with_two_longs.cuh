#pragma once

#include "mpm_test_base.h"
#include "mpm_material.cuh"
#include "mpm_model.h"
#include "mpm_domain.h"
#include "mpm_engine.cuh"
#include "data_type.cuh"

#include <array>
#include <vector>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>


namespace mpm
{
namespace test{

    class MPMTestScene
    {
    public:
        constexpr static std::string_view kTestName_ = "fluid_with_two_longs";

        constexpr static float kDx_ = 1.f / 512.f;
        constexpr static auto kFluidConstitutiveModel_ = MPMConstitutiveModel::kFluid;
        constexpr static auto kLongConstitutiveModel_ = MPMConstitutiveModel::kFixedCorotated;

        constexpr static float kE_ = 1e6;
        constexpr static float kNu_ = 0.3f;
        constexpr static float kLambda_ = ComputeLameParameters<float>(kE_, kNu_)[0];
        constexpr static float kMu_ = ComputeLameParameters<float>(kE_, kNu_)[1];

        constexpr static float kBulk_ = 10.f;
        constexpr static float kGamma_ = 7.15f;
        constexpr static float kViscosity_ = 0.1f;

        constexpr static auto kFluidMaterial_ = MPMMaterial<kFluidConstitutiveModel_>{ MPMMaterial<kFluidConstitutiveModel_>::FluidMaterialParameter{kBulk_, kGamma_, kViscosity_} };
        constexpr static auto kLongMaterial_ = MPMMaterial<kLongConstitutiveModel_>{ MPMMaterial<kLongConstitutiveModel_>::FixedCorotatedMaterialParameter{kLambda_, kMu_} };

        constexpr static float kFluidRho_ = 1000.0f;
        constexpr static float kLongRho_ = 3000.0f;

        constexpr static float kParticlePerCell_ = 8.0f;
        constexpr static float kFluidParticleVolume_ = kDx_ * kDx_ * kDx_ / kParticlePerCell_;
        constexpr static float kFluidParticleMass_ = kFluidParticleVolume_ * kFluidRho_;

        constexpr static float kLongParticleVolume_ = kDx_ * kDx_ * kDx_ / kParticlePerCell_;
        constexpr static float kLongParticleMass_ = kLongParticleVolume_ * kLongRho_;

        constexpr static uint32_t kFps_ = 24;
        constexpr static float kCfl_ = 0.5f;
        constexpr static float kDtFactor_= 0.1f;
        constexpr static float kTotalSimulatedTime_ = 360.f / 24.f;

        typedef MPMDomainRange<256, 128, 64> DomainRange_;
        typedef MPMDomainOffset<0, 0, 0> DomainOffset_;
        typedef MPMDomain<DomainRange_, DomainOffset_> Domain_;
        typedef MPMGridConfig<Domain_> GridConfig_;

        class MPMTestFluidWithTwoLongsConfig : public MPMConfigBase<MPMTestFluidWithTwoLongsConfig>
        {
           public:
            friend class MPMConfigBase<MPMTestFluidWithTwoLongsConfig>;

            constexpr MPMTestFluidWithTwoLongsConfig() = default;

           protected:
            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetDxImpl() const -> float { return kDx_;}

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetBlockVolumeImpl() const -> uint32_t { return 64; }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetBlockSizeImpl() const -> uint32_t { return 4; }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxParticleCountPerCellImpl() const -> uint32_t { return 16; }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxParticleCountPerBlockImpl() const -> uint32_t
            {
                return GetBlockVolumeImpl() * GetMaxParticleCountPerCellImpl();
            }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxParticleCountPerBucketImpl() const -> uint32_t
            {
                return 32;
            }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxActiveBlockCountImpl() const -> uint32_t { return 700000; }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetFpsImpl() const -> uint32_t { return kFps_; }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetDtImpl() const -> float { return EvaluateTestTimestep(kDtFactor_, GetDxImpl(), kE_, kNu_, kLongRho_, kCfl_); }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetCflImpl() const -> float { return kCfl_; }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetTotalSimulatedFrameCountImpl() const -> uint32_t
            {
                return static_cast<uint32_t>(std::round(kTotalSimulatedTime_ * GetFpsImpl()));
            }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetGravityImpl() const -> float
            {
                return -2.f;
            }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMassClampImpl() const -> float
            {
                return kFluidParticleMass_ * 1e-8;
            }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetExistRigidParticleImpl() const -> bool
            {
                return false;
            }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetRigidParticleCountImpl() const -> uint32_t
            {
                return 0;
            }

            template<typename Scalar>
            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetRigidParticleVelocityImpl() const -> Vector<Scalar, 3>
            {
                return Vector<Scalar, 3>{0.f, 0.f, 0.f};
            }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetExistIrregularBoundaryImpl() const -> bool
            {
                return true;
            }

            MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto ProcessGridCellVelocityImpl(const Vector<int, 3>& cell, Vector<float, 3>& velocity, int frame) const -> void
            {

	        constexpr int kRangeDim[] = {DomainRange_::kDim_[0], DomainRange_::kDim_[1], DomainRange_::kDim_[2]};
		Vector<float, 3> normal;
		bool onBoundary = false;
		
		for(int i = 0; i < 3; ++i)
		{
		    if(cell[i] < 8)
		    {
			normal[i] = 1.f;
			onBoundary = true;
		    }
		    else if(cell[i] >= kRangeDim[i] * 4 - 8)
		    {
			normal[i] = -1.f;
			onBoundary = true;
		    }
		}
		normal = normal / normal.Norm();
		
		if(onBoundary)
		{
		    float magnitude = normal.Dot(velocity);
		    velocity -= magnitude * normal;
		}
                return;
            }

            MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto UpdateConfigImpl(float dt, int frame) -> void
            {
            }

        };

        typedef MPMTestFluidWithTwoLongsConfig TestConfig_;
    };

    
    auto SetupModel(uint32_t& particleCount) -> std::vector<MPMModelVariant>
    {
        constexpr float kDx = MPMTestScene::kDx_;

        std::vector<MPMModelVariant> modelList = {};

        // Initialize fluid particles
        {
            std::vector<Vector<float, 3>> position;
            std::vector<Vector<float, 3>> velocity;
            velocity.emplace_back(Vector<float, 3>{0, 0, 0});

            for(int i = 916; i < 1016; ++i)
            {
                for(int j = 8; j < 400; ++j)
                {
                    for(int k = 8; k < 248; ++k)
                    {
                        for(int w = 0; w < 8; ++w)
                        {
                            int dx = w & 1;
                            int dy = (w & 2) >> 1;
                            int dz = (w & 4) >> 2;

                            float particleX = (i + 0.25f + 0.5f * dx) * kDx;
                            float particleY = (j + 0.25f + 0.5f * dy) * kDx;
                            float particleZ = (k + 0.25f + 0.5f * dz) * kDx;

                            position.emplace_back(Vector<float, 3>{particleX, particleY, particleZ});
                            ++particleCount;
                        }
                    }
                }
            }
            modelList.emplace_back(MPMModel<std::decay_t<decltype(MPMTestScene::kFluidMaterial_)>>{MPMTestScene::kFluidParticleMass_, MPMTestScene::kFluidParticleVolume_, position, velocity, MPMTestScene::kFluidMaterial_});
        }

        // Initialize two longs
        {

            std::vector<Vector<float, 3>> position;
            std::vector<Vector<float, 3>> velocity;
            velocity.emplace_back(Vector<float, 3>{0, 0, 0});

            const auto testAssetDirectory = kTestAssetRootDirectory / std::filesystem::path{"particle_data"};
            const auto testAssetName = std::filesystem::path("two_longs_narrow.bin");
            std::ifstream testAsset((testAssetDirectory / testAssetName).c_str(), std::ios::in | std::ios::binary);

            position.resize(8219227);

            if(!testAsset.is_open())
            {
                std::cerr << "Failed to find asset at: " << (testAssetDirectory / testAssetName).c_str() << " for test " << MPMTestScene::kTestName_ << std::endl;
                throw std::runtime_error("");
            }

            testAsset.read(reinterpret_cast<char*>(position.data()), 3 * sizeof(float) * position.size());

            modelList.emplace_back(MPMModel<std::decay_t<decltype(MPMTestScene::kLongMaterial_)>>{MPMTestScene::kLongParticleMass_, MPMTestScene::kLongParticleVolume_, position, velocity, MPMTestScene::kLongMaterial_});
        }

        return modelList;
    }
    
}

}
