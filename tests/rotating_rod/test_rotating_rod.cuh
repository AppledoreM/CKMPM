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
        constexpr static std::string_view kTestName_ = "rotating_rod";

        constexpr static float kDx_ = 1.0f / 256.f;
        constexpr static auto kConstitutiveModel_ = MPMConstitutiveModel::kFixedCorotated;
        constexpr static float kE_ = 1e6;
        constexpr static float kNu_ = 0.4f;
        constexpr static float kLambda_ = ComputeLameParameters<float>(kE_, kNu_)[0];
        constexpr static float kMu_ = ComputeLameParameters<float>(kE_, kNu_)[1];

        constexpr static auto kMaterial_ = MPMMaterial<kConstitutiveModel_>{ MPMMaterial<kConstitutiveModel_>::FixedCorotatedMaterialParameter{kLambda_, kMu_} };

        constexpr static float kRho_ = 2;
        constexpr static float kParticlePerCell_ = 8.0f;
        constexpr static float kParticleVolume_ = 1.f / 256.f / 256.f / 256.f / kParticlePerCell_;
        constexpr static float kParticleMass_ = kParticleVolume_ * kRho_;

        constexpr static uint32_t kFps_ = 48;
        constexpr static float kCfl_ = 0.5f;
        constexpr static float kDtFactor_= 1.0f;
        constexpr static float kTotalSimulatedTime_ = 5.f;

        typedef MPMDomainRange<64, 64, 64> DomainRange_;
        typedef MPMDomainOffset<0, 0, 0> DomainOffset_;
        typedef MPMDomain<DomainRange_, DomainOffset_> Domain_;
        typedef MPMGridConfig<Domain_> GridConfig_;


        class MPMTestRotatingRodConfig : public MPMConfigBase<MPMTestRotatingRodConfig>
        {
           public:
            friend class MPMConfigBase<MPMTestRotatingRodConfig>;

            constexpr MPMTestRotatingRodConfig() = default;

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

        typedef MPMTestRotatingRodConfig TestConfig_;
    };

    
    auto SetupModel(uint32_t& particleCount) -> std::vector<MPMModelVariant>
    {
        std::vector<Vector<float, 3>> position;
        std::vector<Vector<float, 3>> velocity;

        constexpr float kDx = MPMTestScene::kDx_;
        constexpr float radius = 5 * kDx;
        constexpr float height = 10 * kDx;
        // Generate a rotating rod
        {

            auto center = Vector<float, 3>{128, 128, 128} * kDx;
            for(int i = -5; i < 5; ++i)
            {
                for(int j = -5; j < 5; ++j)
                {
                    for(int k = -20; k < 20; ++k)
                    {

                        for(int w = 0; w < 8; ++w)
                        {
                            int di = (w & 4) >> 2;
                            int dj = (w & 2) >> 1;
                            int dk = w & 1;

                            auto particlePosition = center + Vector<float, 3>{i + 0.25f + di * 0.5f, j + 0.25f + dj * 0.5f, k + 0.25f + dk * 0.5f} * kDx;
                            auto cylinderCenter = center;
                            cylinderCenter[2] = particlePosition[2];
                            if((particlePosition - cylinderCenter).Norm() <= radius)
                            {
                                position.emplace_back(particlePosition);

                                float zDistanceFromCenter = cylinderCenter[2] - center[2];
                                velocity.emplace_back(Vector<float, 3>{1.0, 0.0, 0.0} * zDistanceFromCenter / (20.f / kDx));
                                // if(zDistanceFromCenter >= 19 * kDx)
                                // {
                                //     velocity.emplace_back(Vector<float, 3>{0.1, 0.0, 0.0} * zDistanceFromCenter / kDx);
                                // }
                                // else
                                // {
                                //     velocity.emplace_back(Vector<float, 3>{0.f, 0.f, 0.f});
                                // }
                            }
                        }

                    }
                }
            }
        }

        particleCount = position.size(); 
        return { MPMModel<std::decay_t<decltype(MPMTestScene::kMaterial_)>>{MPMTestScene::kParticleMass_, MPMTestScene::kParticleVolume_, position, velocity, MPMTestScene::kMaterial_} };
    }
    
}
}

