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
        constexpr static std::string_view kTestName_ = "oreo";

        constexpr static float kDx_ = 1 / 256.f;
        constexpr static auto kConstitutiveModel_ = MPMConstitutiveModel::kNonAssociatedCamClay;

        constexpr static float kE_ = 20000;
        constexpr static float kNu_ = 0.4;
        constexpr static float kRho_ = 2;

        constexpr static float kCrustE_ = 2e4;
        constexpr static float kCrustNu_ = 0.4f;

        // constexpr static float kHeartE_ = 1000;
        // constexpr static float kHeartNu_ = 0.35f;

        constexpr static float kHeartE_ = 2e4;
        constexpr static float kHeartNu_ = 0.4f;

        constexpr static float kCrustLambda_ = ComputeLameParameters<float>(kCrustE_, kCrustNu_)[0];
        constexpr static float kCrustMu_ = ComputeLameParameters<float>(kCrustE_, kCrustNu_)[1];

        constexpr static float kHeartLambda_ = ComputeLameParameters<float>(kHeartE_, kHeartNu_)[0];
        constexpr static float kHeartMu_ = ComputeLameParameters<float>(kHeartE_, kHeartNu_)[1];


        constexpr static float kCrustRho_ = 2;
        // constexpr static float kHeartRho_ = 2 * 0.2f;
        constexpr static float kHeartRho_ = 2;

        constexpr static float kCrustAlpha0_ = -0.01; 
        // constexpr static float kHeartAlpha0_ = -0.03; 
        constexpr static float kHeartAlpha0_ = -0.01; 

        constexpr static float kCrustBeta_ = 0.5;
        // constexpr static float kHeartBeta_ = 1;
        constexpr static float kHeartBeta_ = 0.5;

        constexpr static float kCrustHardeningFactor_ = 0.8;
        // constexpr static float kHeartHardeningFactor_ = 1;
        constexpr static float kHeartHardeningFactor_ = 0.8;

        constexpr static float kM_ = 2.36;

        constexpr static auto kMaterial_ = MPMMaterial<kConstitutiveModel_>{ MPMMaterial<kConstitutiveModel_>::NonAssociatedCamClayMaterialParameter{kCrustLambda_, kCrustMu_, kCrustAlpha0_, kCrustBeta_, kCrustHardeningFactor_, kM_, true} };
        constexpr static auto kCrustMaterial_ = MPMMaterial<kConstitutiveModel_>{ MPMMaterial<kConstitutiveModel_>::NonAssociatedCamClayMaterialParameter{kCrustLambda_, kCrustMu_, kCrustAlpha0_, kCrustBeta_, kCrustHardeningFactor_, kM_, true} };
        constexpr static auto kHeartMaterial_ = MPMMaterial<kConstitutiveModel_>{ MPMMaterial<kConstitutiveModel_>::NonAssociatedCamClayMaterialParameter{kHeartLambda_, kHeartMu_, kHeartAlpha0_, kHeartBeta_, kHeartHardeningFactor_, kM_, true} };

        constexpr static float kParticlePerCell_ = 8.0f;
        constexpr static float kParticleVolume_ = kDx_ * kDx_ * kDx_ / kParticlePerCell_;
        constexpr static float kCrustParticleMass_ = kParticleVolume_ * kCrustRho_;
        constexpr static float kHeartParticleMass_ = kParticleVolume_ * kHeartRho_;

        constexpr static uint32_t kFps_ = 48;
        constexpr static float kCfl_ = 0.4f;
        constexpr static float kDtFactor_= 1.0f;
        constexpr static float kTotalSimulatedTime_ = 100.f / 48.f;

        typedef MPMDomainRange<64, 64, 64> DomainRange_;
        typedef MPMDomainOffset<0, 0, 0> DomainOffset_;
        typedef MPMDomain<DomainRange_, DomainOffset_> Domain_;
        typedef MPMGridConfig<Domain_> GridConfig_;


        class MPMTestOreoConfig : public MPMConfigBase<MPMTestOreoConfig>
        {
           public:
            friend class MPMConfigBase<MPMTestOreoConfig>;

            constexpr MPMTestOreoConfig() = default;

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

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetDtImpl() const -> float { return EvaluateTestTimestep(kDtFactor_, GetDxImpl(), kCrustE_, kCrustNu_, kCrustRho_, kCfl_); }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetCflImpl() const -> float { return kCfl_; }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetTotalSimulatedFrameCountImpl() const -> uint32_t
            {
                return static_cast<uint32_t>(std::round(kTotalSimulatedTime_ * GetFpsImpl()));
            }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetGravityImpl() const -> float
            {
                return -3.f;
            }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMassClampImpl() const -> float
            {
                return kHeartParticleMass_ * 1e-6;
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

        typedef MPMTestOreoConfig TestConfig_;
    };

    
    auto SetupModel(uint32_t& particleCount) -> std::vector<MPMModelVariant>
    {

        constexpr int kOreoCrustParticleCount = 1615062;
        constexpr int kOreoHeartParticleCount = 658581;

        std::vector<Vector<float, 3>> oreoCrustPosition(kOreoCrustParticleCount);
        std::vector<Vector<float, 3>> oreoCrustVelocity;

        std::vector<Vector<float, 3>> oreoHeartPosition(kOreoHeartParticleCount);
        std::vector<Vector<float, 3>> oreoHeartVelocity;
        oreoCrustVelocity.emplace_back(Vector<float, 3>{0, 0, 0});
        oreoHeartVelocity.emplace_back(Vector<float, 3>{0, 0, 0});

        // Process oreo crust
        {
            const auto testAssetDirectory = kTestAssetRootDirectory / std::filesystem::path{"particle_data"};
            const auto testAssetName = std::filesystem::path("oreo_crust.bin");
            std::ifstream testAsset((testAssetDirectory / testAssetName).c_str(), std::ios::in | std::ios::binary);

            if(!testAsset.is_open())
            {
                std::cerr << "Failed to find asset at: " << (testAssetDirectory / testAssetName).c_str() << " for test " << MPMTestScene::kTestName_ << std::endl;
                throw std::runtime_error("");
            }

            testAsset.read(reinterpret_cast<char*>(oreoCrustPosition.data()), 3 * sizeof(float) * kOreoCrustParticleCount);
        }

        // Process oreo heart        
        {
            const auto testAssetDirectory = kTestAssetRootDirectory / std::filesystem::path{"particle_data"};
            const auto testAssetName = std::filesystem::path("oreo_heart.bin");
            std::ifstream testAsset((testAssetDirectory / testAssetName).c_str(), std::ios::in | std::ios::binary);

            if(!testAsset.is_open())
            {
                std::cerr << "Failed to find asset at: " << (testAssetDirectory / testAssetName).c_str() << " for test " << MPMTestScene::kTestName_ << std::endl;
                throw std::runtime_error("");
            }

            testAsset.read(reinterpret_cast<char*>(oreoHeartPosition.data()), 3 * sizeof(float) * kOreoHeartParticleCount);
        }

        // Process oreo position
        {
            Vector<float, 3> minPos = Vector<float, 3>{1e8, 1e8, 1e8};

            for(int i = 0; i < kOreoCrustParticleCount; ++i)
            {
                minPos[0] = std::min(minPos[0], oreoCrustPosition[i][0]);
                minPos[1] = std::min(minPos[1], oreoCrustPosition[i][1]);
                minPos[2] = std::min(minPos[2], oreoCrustPosition[i][2]);
            }

            for(int i = 0; i < kOreoHeartParticleCount; ++i)
            {

                minPos[0] = std::min(minPos[0], oreoHeartPosition[i][0]);
                minPos[1] = std::min(minPos[1], oreoHeartPosition[i][1]);
                minPos[2] = std::min(minPos[2], oreoHeartPosition[i][2]);
            }

            Vector<float, 3> fixedOffset = Vector<float, 3>{50.f, 24.f, 50.f} / 256.f ;
            for(int i = 0; i < kOreoCrustParticleCount; ++i)
            {
                oreoCrustPosition[i] = oreoCrustPosition[i] - minPos + fixedOffset;
            }

            for(int i = 0; i < kOreoHeartParticleCount; ++i)
            {
                oreoHeartPosition[i] = oreoHeartPosition[i] - minPos + fixedOffset;
            }
        }

        return { MPMModel<std::decay_t<decltype(MPMTestScene::kMaterial_)>>{MPMTestScene::kCrustParticleMass_, MPMTestScene::kParticleVolume_, oreoCrustPosition, oreoCrustVelocity, MPMTestScene::kCrustMaterial_},
                MPMModel<std::decay_t<decltype(MPMTestScene::kMaterial_)>>{MPMTestScene::kHeartParticleMass_, MPMTestScene::kParticleVolume_, oreoHeartPosition, oreoHeartVelocity, MPMTestScene::kHeartMaterial_}
                };
    }
    
}

}
