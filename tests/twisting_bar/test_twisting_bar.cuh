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
        constexpr static std::string_view kTestName_ = "twisting_bar";

        constexpr static float kDx_ = 1.0f / 256.f;
        constexpr static auto kConstitutiveModel_ = MPMConstitutiveModel::kFixedCorotated;
        constexpr static float kE_ = 1e2;
        constexpr static float kNu_ = 0.4f;
        constexpr static float kLambda_ = ComputeLameParameters<float>(kE_, kNu_)[0];
        constexpr static float kMu_ = ComputeLameParameters<float>(kE_, kNu_)[1];

        constexpr static auto kMaterial_ = MPMMaterial<kConstitutiveModel_>{ MPMMaterial<kConstitutiveModel_>::FixedCorotatedMaterialParameter{kLambda_, kMu_} };

        constexpr static float kRho_ = 2;
        constexpr static float kParticlePerCell_ = 8.0f;
        constexpr static int kParticlePerDimenion_ = 3;
        constexpr static float kParticleVolume_ = 1.f / 256.f / 256.f / 256.f / 27;
        constexpr static float kParticleMass_ = kParticleVolume_ * kRho_;

        constexpr static uint32_t kFps_ = 48;
        constexpr static float kCfl_ = 0.5f;
        constexpr static float kDtFactor_= 0.2f;
        constexpr static float kTotalSimulatedTime_ = 240.f / 48.f;

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

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxActiveBlockCountImpl() const -> uint32_t { return 30000; }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetFpsImpl() const -> uint32_t { return kFps_; }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetDtImpl() const -> float { return EvaluateTestTimestep(kDtFactor_, GetDxImpl(), kE_, kNu_, kRho_, kCfl_); }

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
                return true;
            }

            MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto ProcessGridCellVelocityImpl(const Vector<int, 3>& cell, Vector<float, 3>& velocity, int frame) const -> void
            {
                bool inRange0 = (148 <= cell[2]) && (cell[2] < 168) && (cell[0] != 128) && (cell[1] != 128) && (cell[1] >= 100) && (100 <= cell[0]) && (cell[0] <= 150);
                bool inRange1 = (88 <= cell[2]) && (cell[2] < 108) && (cell[0] != 128) && (cell[1] != 128) && (cell[1] >= 100) && (100 <= cell[0]) && (cell[0] <= 150);

                // bool inRange0 = (93 <= cell[2]) && (cell[2] < 105) && (cell[0] != 80) && (cell[1] != 80);
                // bool inRange1 = (55 <= cell[2]) && (cell[2] < 68) && (cell[0] != 80) && (cell[1] != 80);
                // bool inRange0 = (74 <= cell[2]) && (cell[2] < 84) && (cell[0] != 64) && (cell[1] != 64);
                // bool inRange1 = (44 <= cell[2]) && (cell[2] < 54) && (cell[0] != 64) && (cell[1] != 64);

                bool isOutOfBound = (cell[0] < 8) || (cell[1] < 8) || (cell[2] < 8) || (cell[0] >= 248)  || (cell[1] >= 248) || (cell[2] >= 248);
                const auto center = Vector<float, 3>{128, 128, 128} * kDx_;
                // const auto center = Vector<float, 3>{80, 80, 80} * kDx_;
                // const auto center = Vector<float, 3>{64, 64, 64} * kDx_;
                const auto rotationAxis = Vector<float, 3>{0, 0, -1};
                const float velocityFactor = 1.f;
                // const float velocityFactor = 2.56 / 1.5;
                // const float velocityFactor = 1.28;

                if(inRange0)
                {
                    auto cellPos = cell * kDx_ - center;
                    cellPos[2] = 0.f;
                    float radius = cellPos.Norm();
                    cellPos = cellPos / cellPos.Norm();
                    velocity = Vector<float, 3>{cellPos[1]  * rotationAxis[2] - rotationAxis[1] * cellPos[2], cellPos[2] * rotationAxis[0] - rotationAxis[2] * cellPos[0], 0};
                    velocity = velocityFactor * radius * velocity / velocity.Norm();
                }

                if(inRange1)
                {
                    auto cellPos = cell * kDx_ - center;
                    cellPos[2] = 0.f;
                    float radius = cellPos.Norm();
                    cellPos = cellPos / cellPos.Norm();
                    velocity = Vector<float, 3>{-cellPos[1]  * rotationAxis[2] + rotationAxis[1] * cellPos[2], -cellPos[2] * rotationAxis[0] + rotationAxis[2] * cellPos[0], 0};
                    velocity = velocityFactor * radius * velocity / velocity.Norm();
                }

                velocity[0] = isOutOfBound ? 0.f : velocity[0];
                velocity[1] = isOutOfBound ? 0.f : velocity[1];
                velocity[2] = isOutOfBound ? 0.f : velocity[2];

                if(cell[1] <= 90 && velocity[1] > 0.f) velocity[1] = 0.f;

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
        std::vector<Vector<float, 3>> position;
        std::vector<Vector<float, 3>> velocity;
        velocity.emplace_back(Vector<float, 3>{0, 0, 0});

        // particleCount = 256000;
        // particleCount = 384000;
        particleCount = 236806; // ppc 8
         // particleCount = 470845; // ppc 16
       // particleCount = 792242; // ppc 16

        position.resize(particleCount);
        
        const auto testAssetDirectory = kTestAssetRootDirectory / std::filesystem::path{"particle_data"};
        // const auto testAssetName = std::filesystem::path("twisting_bar.bin");
        // const auto testAssetName = std::filesystem::path("twisting_bar_1.5ppc.bin");
        const auto testAssetName = std::filesystem::path("twisting_bar_ppc8.bin");
        // const auto testAssetName = std::filesystem::path("twisting_bar_ppc24.bin");
         // const auto testAssetName = std::filesystem::path("twisting_bar_ppc16.bin");
       // const auto testAssetName = std::filesystem::path("twisting_bar_ppc27.bin");
        std::ifstream testAsset((testAssetDirectory / testAssetName).c_str(), std::ios::in | std::ios::binary);

        if(!testAsset.is_open())
        {
            std::cerr << "Failed to find asset at: " << (testAssetDirectory / testAssetName).c_str() << " for test " << MPMTestScene::kTestName_ << std::endl;
            throw std::runtime_error("");
        }

        testAsset.read(reinterpret_cast<char*>(position.data()), 3 * sizeof(float) * particleCount);
        //
        // const int xLimit[2] = { 118, 138 };
        // const int yLimit[2] = { 118, 138 };
        // const int zLimit[2] = { 88, 168 };

        // int particleCountPerDimension = MPMTestScene::kParticlePerDimenion_;;
        // const float dx = 1.f / 256;

        // float particleRadius = dx / (2 * particleCountPerDimension);

        // for(int i = xLimit[0]; i < xLimit[1]; ++i)
        // {
        //     for(int j = yLimit[0]; j < yLimit[1]; ++j)
        //     {
        //         for(int k = zLimit[0]; k < zLimit[1]; ++k)
        //         {
        //             for(int w = 0; w < particleCountPerDimension * particleCountPerDimension * particleCountPerDimension; ++w)
        //             {
                        
        //                 int di = w / (particleCountPerDimension * particleCountPerDimension);
        //                 int dj = (w / particleCountPerDimension) % particleCountPerDimension;
        //                 int dk = w % particleCountPerDimension;

        //                 auto pos = Vector<float, 3>{particleRadius + di * 2 * particleRadius + i * dx, particleRadius + dj * 2 * particleRadius + j * dx, particleRadius + dk * 2 * particleRadius + k * dx};

        //                 position.emplace_back(pos);  
        //             }
        //         }
        //     }
        // }
        // printf("Particle Volume: %.10f\n", MPMTestScene::kParticleVolume_);


        return { MPMModel<std::decay_t<decltype(MPMTestScene::kMaterial_)>>{MPMTestScene::kParticleMass_, MPMTestScene::kParticleVolume_, position, velocity, MPMTestScene::kMaterial_} };
    }
    
}

}

