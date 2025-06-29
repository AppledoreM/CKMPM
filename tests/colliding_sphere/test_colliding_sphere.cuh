
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
        constexpr static std::string_view kTestName_ = "colliding_sphere";

        constexpr static float kDx_ = 1.0f / 256.f;
        constexpr static auto kConstitutiveModel_ = MPMConstitutiveModel::kFixedCorotated;
        constexpr static float kE_ = 1e6;
        constexpr static float kNu_ = 0.4f;
        constexpr static float kLambda_ = ComputeLameParameters<float>(kE_, kNu_)[0];
        constexpr static float kMu_ = ComputeLameParameters<float>(kE_, kNu_)[1];

        constexpr static auto kMaterial_ = MPMMaterial<kConstitutiveModel_>{ MPMMaterial<kConstitutiveModel_>::FixedCorotatedMaterialParameter{kLambda_, kMu_} };

        constexpr static float kRho_ = 1000.f;
        constexpr static float kParticlePerCell_ = 8.0f;
        constexpr static float kParticleVolume_ = kDx_ * kDx_ * kDx_ / kParticlePerCell_;
        constexpr static float kParticleMass_ = kParticleVolume_ * kRho_;

        constexpr static uint32_t kFps_ = 48;
        constexpr static float kCfl_ = 0.5f;
        constexpr static float kDtFactor_= 1.0f;
        constexpr static float kTotalSimulatedTime_ = 5.f;

        typedef MPMDomainRange<64, 64, 64> DomainRange_;
        typedef MPMDomainOffset<0, 0, 0> DomainOffset_;
        typedef MPMDomain<DomainRange_, DomainOffset_> Domain_;
        typedef MPMGridConfig<Domain_> GridConfig_;


        class MPMTestCollidingSphereConfig : public MPMConfigBase<MPMTestCollidingSphereConfig>
        {
           public:
            friend class MPMConfigBase<MPMTestCollidingSphereConfig>;

            constexpr MPMTestCollidingSphereConfig() = default;

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

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxActiveBlockCountImpl() const -> uint32_t { return 6000; }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetFpsImpl() const -> uint32_t { return kFps_; }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetDtImpl() const -> float { return EvaluateTestTimestep(kDtFactor_, GetDxImpl(), kE_, kNu_, kRho_, kCfl_); }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetCflImpl() const -> float { return kCfl_; }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetTotalSimulatedFrameCountImpl() const -> uint32_t
            {
                return static_cast<uint32_t>(std::round(kTotalSimulatedTime_ * GetFpsImpl()));
            }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetGravityImpl() const -> float
            {
                return 0.f;
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

        typedef MPMTestCollidingSphereConfig TestConfig_;
    };

    
    auto SetupModel(uint32_t& particleCount) -> std::vector<MPMModelVariant>
    {
        std::vector<Vector<float, 3>> position[2];
        std::vector<Vector<float, 3>> velocity[2];

        constexpr float kDx = MPMTestScene::kDx_;
        constexpr float radius = 10 * kDx;
        // Two spheres generated at (32, 32, 32) and (128, 128, 128), and move to each other at a speed of +0.1 / -0.1 in xyz-axis
        Vector<float, 3> initialVelocitySum;
        Vector<float, 3> initialMomentumOneSphere;
        {

            auto center0 = Vector<float, 3>{32, 32, 32} * kDx;
            auto center1 = Vector<float, 3>{128, 128, 128} * kDx;
            for(int i = -10; i <= 10; ++i)
            {
                for(int j = -10; j <= 10; ++j)
                {
                    for(int k = -10; k <= 10; ++k)
                    {
                        for(int w = 0; w < 8; ++w)
                        {
                            int di = (w & 4) >> 2;
                            int dj = (w & 2) >> 1;
                            int dk = w & 1;

                            auto particlePosition = center0 + Vector<float, 3>{i + 0.25f + di * 0.5f, j + 0.25f + dj * 0.5f, k + 0.25f + dk * 0.5f} * kDx;
                            if((particlePosition - center0).Norm() <= radius)
                            {
                                position[0].emplace_back(particlePosition);
                                velocity[0].emplace_back(Vector<float, 3>{0.05, 0.05, 0.05});
                                initialVelocitySum += velocity[0].back();
                                initialMomentumOneSphere += velocity[0].back();
                            }

                            particlePosition = center1 + Vector<float, 3>{i + 0.25f + di * 0.5f, j + 0.25f + dj * 0.5f, k + 0.25f + dk * 0.5f} * kDx;
                            if((particlePosition - center1).Norm() <= radius)
                            {
                                position[1].emplace_back(particlePosition);
                                velocity[1].emplace_back(Vector<float, 3>{-0.05, -0.05, -0.05});
                                initialVelocitySum += velocity[1].back();
                            }
                        }
                    }
                }
            }


        }

        printf("Initial velocity sum: (%f, %f, %f)\n", initialVelocitySum[0], initialVelocitySum[1], initialVelocitySum[2]);
        printf("Initial momentum one sphere: (%f, %f, %f)\n", initialMomentumOneSphere[0], initialMomentumOneSphere[1], initialMomentumOneSphere[2]);
        getchar();

        particleCount = position[0].size(); 
        std::vector<MPMModelVariant> modelList;
        modelList.push_back({ MPMModel<std::decay_t<decltype(MPMTestScene::kMaterial_)>>{MPMTestScene::kParticleMass_, MPMTestScene::kParticleVolume_, position[0], velocity[0], MPMTestScene::kMaterial_} });
        modelList.push_back({ MPMModel<std::decay_t<decltype(MPMTestScene::kMaterial_)>>{MPMTestScene::kParticleMass_, MPMTestScene::kParticleVolume_, position[1], velocity[1], MPMTestScene::kMaterial_} });
        return modelList;
    }
    
}
}

