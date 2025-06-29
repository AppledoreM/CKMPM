#pragma once

#include "mpm_test_base.h"
#include "mpm_material.cuh"
#include "mpm_model.h"
#include "mpm_domain.h"
#include "mpm_engine.cuh"
#include "data_type.cuh"
#include "mpm_config.h"

#include <array>
#include <algorithm>
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
        constexpr static std::string_view kTestName_ = "castle_crasher";

        constexpr static float kDx_ = 1 / 512.f;
        constexpr static auto kCannonballConstitutiveModel_ = MPMConstitutiveModel::kFixedCorotated;
        constexpr static auto kCastleConstitutiveModel_ = MPMConstitutiveModel::kNonAssociatedCamClay;
        constexpr static float kCannonballE_ = 1e6;
        constexpr static float kCannonballNu_ = 0.2f;
        constexpr static float kCannonballLambda_ = ComputeLameParameters<float>(kCannonballE_, kCannonballNu_)[0];
        constexpr static float kCannonballMu_ = ComputeLameParameters<float>(kCannonballE_, kCannonballNu_)[1];

        constexpr static float kCastleE_ = 1e4;
        constexpr static float kCastleNu_ = 0.3f;
        constexpr static float kCastleLambda_ = ComputeLameParameters<float>(kCastleE_, kCastleNu_)[0];
        constexpr static float kCastleMu_ = ComputeLameParameters<float>(kCastleE_, kCastleNu_)[1];

        constexpr static float kAlpha0_ = -0.006; 
        constexpr static float kBeta_ = 0.3;
        constexpr static float kHardeningFactor_ = 0.5;

        constexpr static float kM_ = 1.85;

        constexpr static auto kCannonballMaterial_ = MPMMaterial<kCannonballConstitutiveModel_>{ MPMMaterial<kCannonballConstitutiveModel_>::FixedCorotatedMaterialParameter{kCastleLambda_, kCastleMu_} };
        constexpr static auto kCastleMaterial_ = MPMMaterial<kCastleConstitutiveModel_>{ MPMMaterial<kCastleConstitutiveModel_>::NonAssociatedCamClayMaterialParameter{kCastleLambda_, kCastleMu_, kAlpha0_, kBeta_, kHardeningFactor_, kM_, true} };

        constexpr static float kCannonballRho_ = 100;
        constexpr static float kCastleRho_ = 2;

        constexpr static float kParticlePerCell_ = 8.0f;
        constexpr static float kCannonballParticleVolume_ = kDx_ * kDx_ * kDx_ / kParticlePerCell_;
        constexpr static float kCannonballParticleMass_ = kCannonballParticleVolume_ * kCannonballRho_;

        constexpr static float kCastleParticleVolume_ = kDx_ * kDx_ * kDx_ / kParticlePerCell_;
        constexpr static float kCastleParticleMass_ = kCastleParticleVolume_ * kCastleRho_;

        //constexpr static uint32_t kFps_ = 240;
        constexpr static uint32_t kFps_ = 480;
        constexpr static float kCfl_ = 0.5f;
        //constexpr static float kDtFactor_= 0.5f;
        constexpr static float kDtFactor_= 1.0f;
        //constexpr static float kTotalSimulatedTime_ = 360.f / 240.f;
        constexpr static float kTotalSimulatedTime_ = 100.f / 480.f;

        typedef MPMDomainRange<512, 256, 256> DomainRange_;
        typedef MPMDomainOffset<0, 0, 0> DomainOffset_;
        typedef MPMDomain<DomainRange_, DomainOffset_> Domain_;
        typedef MPMGridConfig<Domain_> GridConfig_;


        class MPMTestCastleConfig : public MPMConfigBase<MPMTestCastleConfig>
        {
           public:
            friend class MPMConfigBase<MPMTestCastleConfig>;

            constexpr MPMTestCastleConfig() = default;

           protected:
            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetDxImpl() const -> float { return kDx_;}

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetBlockVolumeImpl() const -> uint32_t { return 64; }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetBlockSizeImpl() const -> uint32_t { return 4; }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxParticleCountPerCellImpl() const -> uint32_t { return 20; }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxParticleCountPerBlockImpl() const -> uint32_t
            {
                return GetBlockVolumeImpl() * GetMaxParticleCountPerCellImpl();
            }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxParticleCountPerBucketImpl() const -> uint32_t
            {
                return 32;
            }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxActiveBlockCountImpl() const -> uint32_t { return 800000; }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetFpsImpl() const -> uint32_t { return kFps_; }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetDtImpl() const -> float { return std::min(EvaluateTestTimestep(kDtFactor_, GetDxImpl(), kCannonballE_, kCannonballNu_, kCannonballRho_, kCfl_), EvaluateTestTimestep(kDtFactor_, GetDxImpl(), kCastleE_, kCastleNu_, kCastleRho_, kCfl_)); }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetCflImpl() const -> float { return kCfl_; }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetTotalSimulatedFrameCountImpl() const -> uint32_t
            {
                return static_cast<uint32_t>(std::round(kTotalSimulatedTime_ * GetFpsImpl()));
            }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetGravityImpl() const -> float
            {
                return -9.8f;
            }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMassClampImpl() const -> float
            {
                return 0.f;
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
                return false;
            }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto ProcessGridCellVelocityImpl(const Vector<int, 3>& cell, Vector<float, 3>& velocity) const -> void
            {
                return;
            }

            MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto UpdateConfigImpl(float dt, int frame) -> void
            {
            }
        };

        typedef MPMTestCastleConfig TestConfig_;
    };

    
    auto SetupModel(uint32_t& particleCount) -> std::vector<MPMModelVariant>
    {
        constexpr float kDx = 1 / 512.f;
        std::vector<MPMModelVariant> modelList = {};
        // Process castle model
        {

            constexpr int kParticleCount = 45925181;
            std::vector<Vector<float, 3>> position(kParticleCount);
            std::vector<Vector<float, 3>> velocity;
            velocity.emplace_back(Vector<float, 3>{0.f, 0, 0});

            const auto testAssetDirectory = kTestAssetRootDirectory / std::filesystem::path{"particle_data"};
            const auto testAssetName = std::filesystem::path("castle_jittered.bin");
            std::ifstream testAsset((testAssetDirectory / testAssetName).c_str(), std::ios::in | std::ios::binary);

            if(!testAsset.is_open())
            {
                std::cerr << "Failed to find asset at: " << (testAssetDirectory / testAssetName).c_str() << " for test " << MPMTestScene::kTestName_ << std::endl;
                throw std::runtime_error("");
            }

            testAsset.read(reinterpret_cast<char*>(position.data()), 3 * sizeof(float) * kParticleCount);

	    Vector<float, 3> minPos = Vector<float, 3>{1.f, 1.f, 1.f};
	    for(int i = 0; i < kParticleCount; ++i)
	    {
		    minPos[0] = std::min(minPos[0], position[i][0]);
		    minPos[1] = std::min(minPos[1], position[i][1]);
		    minPos[2] = std::min(minPos[2], position[i][2]);
	    }

	    for(int i = 0; i < kParticleCount; ++i)
	    {
		    position[i][0] -= 0.25;
		    position[i][1] = position[i][1] - minPos[1] + 8 * kDx - 0.035;
	    }

	    std::erase_if(position, [](auto const &v) { return v[1] <= 8 * kDx;});



            modelList.emplace_back(MPMModel<std::decay_t<decltype(MPMTestScene::kCastleMaterial_)>>{MPMTestScene::kCastleParticleMass_, MPMTestScene::kCastleParticleVolume_, position, velocity, MPMTestScene::kCastleMaterial_});
        }

        // Process cannonball 
        {

            std::vector<Vector<float, 3>> position;
            std::vector<Vector<float, 3>> velocity;
            velocity.emplace_back(Vector<float, 3>{10.f, 0, 0});

            Vector<float, 3> center = Vector<float, 3>{20.f, 64.f, 256.f} * kDx;
	    int intr = 10;
            float radius = intr * kDx;

            for(int i = -intr; i < intr; ++i)
            {
                for(int j = -intr; j < intr; ++j)
                {
                    for(int k = -intr; k < intr; ++k)
                    {
                        for(int w = 0; w < 8; ++w)
                        {

                            int dx = w & 1;
                            int dy = (w & 2) >> 1;
                            int dz = (w & 4) >> 2;

                            float particleX = (i + 0.25f + 0.5f * dx) * kDx;
                            float particleY = (j + 0.25f + 0.5f * dy) * kDx;
                            float particleZ = (k + 0.25f + 0.5f * dz) * kDx;

                            auto pos = Vector<float, 3>{particleX, particleY, particleZ} + center;

                            if((pos - center).Norm() < radius)
                            {
                                position.emplace_back(pos);
                            }

                        }
                    }
                }
            }
            modelList.emplace_back(MPMModel<std::decay_t<decltype(MPMTestScene::kCannonballMaterial_)>>{MPMTestScene::kCannonballParticleMass_, MPMTestScene::kCannonballParticleVolume_, position, velocity, MPMTestScene::kCannonballMaterial_});
        }

        return modelList;
    }
    
}

}
