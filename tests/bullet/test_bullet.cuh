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
        constexpr static std::string_view kTestName_ = "bullet";

        constexpr static float kDx_ = 1 / 1024.f;
        constexpr static auto kTungstenConstitutiveModel_ = MPMConstitutiveModel::kFixedCorotated;
        constexpr static auto kBulletConstitutiveModel_ = MPMConstitutiveModel::kNonAssociatedCamClay;
        // constexpr static auto kBulletConstitutiveModel_ = MPMConstitutiveModel::kFixedCorotated;

        constexpr static float kTungstenE_ = 4.5e11;
        constexpr static float kTungstenNu_ = 0.27f;
        constexpr static float kTungstenLambda_ = ComputeLameParameters<float>(kTungstenE_, kTungstenNu_)[0];
        constexpr static float kTungstenMu_ = ComputeLameParameters<float>(kTungstenE_, kTungstenNu_)[1];

        constexpr static float kBulletE_ = 1.5e10;
        constexpr static float kBulletNu_ = 0.435f;
        constexpr static float kBulletLambda_ = ComputeLameParameters<float>(kBulletE_, kBulletNu_)[0];
        constexpr static float kBulletMu_ = ComputeLameParameters<float>(kBulletE_, kBulletNu_)[1];

        constexpr static float kAlpha0_ = -0.03; 
        constexpr static float kBeta_ = 1.0;
        constexpr static float kHardeningFactor_ = 0.01;

        constexpr static float kM_ = 2.36;

        constexpr static auto kTungstenMaterial_ = MPMMaterial<kTungstenConstitutiveModel_>{ MPMMaterial<kTungstenConstitutiveModel_>::FixedCorotatedMaterialParameter{kBulletLambda_, kBulletMu_} };
        constexpr static auto kBulletMaterial_ = MPMMaterial<kBulletConstitutiveModel_>{ MPMMaterial<kBulletConstitutiveModel_>::NonAssociatedCamClayMaterialParameter{kBulletLambda_, kBulletMu_, kAlpha0_, kBeta_, kHardeningFactor_, kM_, true} };
        // constexpr static auto kBulletMaterial_ = MPMMaterial<kBulletConstitutiveModel_>{ MPMMaterial<kBulletConstitutiveModel_>::FixedCorotatedMaterialParameter{kBulletLambda_, kBulletMu_} };

        constexpr static float kTungstenRho_ = 19250;
        constexpr static float kBulletRho_ = 11348;

        constexpr static float kParticlePerCell_ = 8.0f;
        constexpr static float kTungstenParticleVolume_ = kDx_ * kDx_ * kDx_ / kParticlePerCell_;
        constexpr static float kTungstenParticleMass_ = kTungstenParticleVolume_ * kTungstenRho_;

        constexpr static float kBulletParticleVolume_ = (1 / 4096.f) * (1 / 4096.f) * (1 / 4096.f);
        constexpr static float kBulletParticleMass_ = kBulletParticleVolume_ * kBulletRho_;

        constexpr static uint32_t kFps_ = 50000;
        constexpr static float kCfl_ = 0.5f;
        constexpr static float kDtFactor_= 1.0f;
        constexpr static float kTotalSimulatedTime_ = 100.f / 50000.f;

        typedef MPMDomainRange<256, 256, 256> DomainRange_;
        typedef MPMDomainOffset<0, 0, 0> DomainOffset_;
        typedef MPMDomain<DomainRange_, DomainOffset_> Domain_;
        typedef MPMGridConfig<Domain_> GridConfig_;


        class MPMTestBulletConfig : public MPMConfigBase<MPMTestBulletConfig>
        {
           public:
            friend class MPMConfigBase<MPMTestBulletConfig>;

            constexpr MPMTestBulletConfig() = default;

           protected:
            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetDxImpl() const -> float { return kDx_;}

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetBlockVolumeImpl() const -> uint32_t { return 64; }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetBlockSizeImpl() const -> uint32_t { return 4; }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxParticleCountPerCellImpl() const -> uint32_t { return 128; }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxParticleCountPerBlockImpl() const -> uint32_t
            {
                return GetBlockVolumeImpl() * GetMaxParticleCountPerCellImpl();
            }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxParticleCountPerBucketImpl() const -> uint32_t
            {
                return 32;
            }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxActiveBlockCountImpl() const -> uint32_t { return 50000; }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetFpsImpl() const -> uint32_t { return kFps_; }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetDtImpl() const -> float { return std::min(EvaluateTestTimestep(kDtFactor_, GetDxImpl(), kTungstenE_, kTungstenNu_, kTungstenRho_, kCfl_), EvaluateTestTimestep(kDtFactor_, GetDxImpl(), kBulletE_, kBulletNu_, kBulletRho_, kCfl_)); }

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

        typedef MPMTestBulletConfig TestConfig_;
    };

    
    auto SetupModel(uint32_t& particleCount) -> std::vector<MPMModelVariant>
    {
        constexpr float kDx = 1 / 1024.f;
        std::vector<MPMModelVariant> modelList = {};
        // Process bullet model
        {

            constexpr int kParticleCount = 40337;
            std::vector<Vector<float, 3>> position(kParticleCount);
            std::vector<Vector<float, 3>> velocity;
            velocity.emplace_back(Vector<float, 3>{300.f, 0, 0});

            const auto testAssetDirectory = kTestAssetRootDirectory / std::filesystem::path{"particle_data"};
            const auto testAssetName = std::filesystem::path("bullet.bin");
            std::ifstream testAsset((testAssetDirectory / testAssetName).c_str(), std::ios::in | std::ios::binary);

            if(!testAsset.is_open())
            {
                std::cerr << "Failed to find asset at: " << (testAssetDirectory / testAssetName).c_str() << " for test " << MPMTestScene::kTestName_ << std::endl;
                throw std::runtime_error("");
            }

            testAsset.read(reinterpret_cast<char*>(position.data()), 3 * sizeof(float) * kParticleCount);

            modelList.emplace_back(MPMModel<std::decay_t<decltype(MPMTestScene::kBulletMaterial_)>>{MPMTestScene::kBulletParticleMass_, MPMTestScene::kBulletParticleVolume_, position, velocity, MPMTestScene::kBulletMaterial_});
        }

        // Process tungsten cube 
        {

            constexpr int kParticleCount = 8615125;
            std::vector<Vector<float, 3>> position(kParticleCount);
            std::vector<Vector<float, 3>> velocity;
            velocity.emplace_back(Vector<float, 3>{0.f, 0, 0});

            const auto testAssetDirectory = kTestAssetRootDirectory / std::filesystem::path{"particle_data"};
            const auto testAssetName = std::filesystem::path("box.bin");
            std::ifstream testAsset((testAssetDirectory / testAssetName).c_str(), std::ios::in | std::ios::binary);

            if(!testAsset.is_open())
            {
                std::cerr << "Failed to find asset at: " << (testAssetDirectory / testAssetName).c_str() << " for test " << MPMTestScene::kTestName_ << std::endl;
                throw std::runtime_error("");
            }

            testAsset.read(reinterpret_cast<char*>(position.data()), 3 * sizeof(float) * kParticleCount);

            modelList.emplace_back(MPMModel<std::decay_t<decltype(MPMTestScene::kTungstenMaterial_)>>{MPMTestScene::kTungstenParticleMass_, MPMTestScene::kTungstenParticleVolume_, position, velocity, MPMTestScene::kTungstenMaterial_});
        }

        return modelList;
    }
    
}

}
