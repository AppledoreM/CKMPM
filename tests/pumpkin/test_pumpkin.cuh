
#pragma once

#include "mpm_test_base.h"
#include "mpm_material.cuh"
#include "mpm_model.h"
#include "mpm_domain.h"
#include "mpm_engine.cuh"
#include "data_type.cuh"
#include "mpm_config.h"

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
        constexpr static std::string_view kTestName_ = "pumpkin";

        constexpr static float kDx_ = 1 / 128.f;
        constexpr static auto kConstitutiveModel_ = MPMConstitutiveModel::kNonAssociatedCamClay;
        constexpr static float kE_ = 2000;
        constexpr static float kNu_ = 0.39f;
        constexpr static float kLambda_ = ComputeLameParameters<float>(kE_, kNu_)[0];
        constexpr static float kMu_ = ComputeLameParameters<float>(kE_, kNu_)[1];


        constexpr static float kAlpha0_ = -0.04; 
        constexpr static float kBeta_ = 2;
        constexpr static float kHardeningFactor_ = 3;

        constexpr static float kM_ = 2.36;

        constexpr static auto kMaterial_ = MPMMaterial<kConstitutiveModel_>{ MPMMaterial<kConstitutiveModel_>::NonAssociatedCamClayMaterialParameter{kLambda_, kMu_, kAlpha0_, kBeta_, kHardeningFactor_, kM_, true} };

        constexpr static float kRho_ = 2;
        constexpr static float kParticlePerCell_ = 8.0f;
        constexpr static float kParticleVolume_ = kDx_ * kDx_ * kDx_ / kParticlePerCell_;
        constexpr static float kParticleMass_ = kParticleVolume_ * kRho_;

        constexpr static uint32_t kFps_ = 48;
        constexpr static float kCfl_ = 0.5f;
        constexpr static float kDtFactor_= 1.0f;
        constexpr static float kTotalSimulatedTime_ = 100.f / 48.f;

        typedef MPMDomainRange<64, 64, 64> DomainRange_;
        typedef MPMDomainOffset<0, 0, 0> DomainOffset_;
        typedef MPMDomain<DomainRange_, DomainOffset_> Domain_;
        typedef MPMGridConfig<Domain_> GridConfig_;


        class MPMTestDragonConfig : public MPMConfigBase<MPMTestDragonConfig>
        {
           public:
            friend class MPMConfigBase<MPMTestDragonConfig>;

            constexpr MPMTestDragonConfig() = default;

           protected:
            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetDxImpl() const -> float { return kDx_;}

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetBlockVolumeImpl() const -> uint32_t { return 64; }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetBlockSizeImpl() const -> uint32_t { return 4; }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxParticleCountPerCellImpl() const -> uint32_t { return 64; }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxParticleCountPerBlockImpl() const -> uint32_t
            {
                return GetBlockVolumeImpl() * GetMaxParticleCountPerCellImpl();
            }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxParticleCountPerBucketImpl() const -> uint32_t
            {
                return 32;
            }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxActiveBlockCountImpl() const -> uint32_t { return 20000; }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetFpsImpl() const -> uint32_t { return kFps_; }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetDtImpl() const -> float { return EvaluateTestTimestep(kDtFactor_, GetDxImpl(), kE_, kNu_, kRho_, kCfl_); }

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

        typedef MPMTestDragonConfig TestConfig_;
    };

    
    auto SetupModel(uint32_t& particleCount) -> std::vector<MPMModelVariant>
    {
        constexpr int kParticleCount = 585420 * 2;
        std::vector<Vector<float, 3>> position(kParticleCount);
        std::vector<Vector<float, 3>> velocity;
        velocity.emplace_back(Vector<float, 3>{0, 0, 0});

        const auto testAssetDirectory = kTestAssetRootDirectory / std::filesystem::path{"particle_data"};
        const auto testAssetName = std::filesystem::path("pumpkin.bin");
        std::ifstream testAsset((testAssetDirectory / testAssetName).c_str(), std::ios::in | std::ios::binary);

        if(!testAsset.is_open())
        {
            std::cerr << "Failed to find asset at: " << (testAssetDirectory / testAssetName).c_str() << " for test " << MPMTestScene::kTestName_ << std::endl;
            throw std::runtime_error("");
        }

        testAsset.read(reinterpret_cast<char*>(position.data()), 3 * sizeof(float) * kParticleCount / 2);
        // Process pumpkin position 
        {
            Vector<float, 3> minPos = Vector<float, 3>{1e8, 1e8, 1e8};
            for(int i = 0; i * 2 < kParticleCount; ++i)
            {
                minPos[0] = std::min(minPos[0], position[i][0]);
                minPos[1] = std::min(minPos[1], position[i][1]);
                minPos[2] = std::min(minPos[2], position[i][2]);
            }

            Vector<float, 3> fixedOffset = Vector<float, 3>{50.f, 8.f, 50.f} / 128.f;
            for(int i = 0; i * 2 < kParticleCount; ++i)
            {
                position[i] = position[i] - minPos + fixedOffset;
                position[i + kParticleCount / 2] = position[i] + Vector<float, 3>{0.f, 1 / 128.f, 0.f} * 100.f;

            }

        }

        particleCount = kParticleCount; 
        return { MPMModel<std::decay_t<decltype(MPMTestScene::kMaterial_)>>{MPMTestScene::kParticleMass_, MPMTestScene::kParticleVolume_, position, velocity, MPMTestScene::kMaterial_} };
    }
    
}

}
