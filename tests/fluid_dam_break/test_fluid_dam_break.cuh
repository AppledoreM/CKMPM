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
        constexpr static std::string_view kTestName_ = "fluid_dam_break";

        constexpr static float kDx_ = 1.f / 256.f;
        constexpr static auto kConstitutiveModel_ = MPMConstitutiveModel::kFluid;
        constexpr static float kE_ = 5e3;
        constexpr static float kNu_ = 0.4f;
        constexpr static float kLambda_ = ComputeLameParameters<float>(kE_, kNu_)[0];
        constexpr static float kMu_ = ComputeLameParameters<float>(kE_, kNu_)[1];
        constexpr static float kBulk_ = 10.f;
        constexpr static float kGamma_ = 7.15f;
        constexpr static float kViscosity_ = 0.1f;

        constexpr static auto kMaterial_ = MPMMaterial<kConstitutiveModel_>{ MPMMaterial<kConstitutiveModel_>::FluidMaterialParameter{kBulk_, kGamma_, kViscosity_} };

        constexpr static float kRho_ = 1000.0f;
        constexpr static float kParticlePerCell_ = 8.0f;
        constexpr static float kParticleVolume_ = 1.f / 256.f / 256.f / 256.f / kParticlePerCell_;
        constexpr static float kParticleMass_ = kParticleVolume_ * kRho_;

        constexpr static uint32_t kFps_ = 48;
        constexpr static float kCfl_ = 0.5f;
        constexpr static float kDtFactor_= 1.f;
        constexpr static float kTotalSimulatedTime_ = 240.f / 48.f;
        // constexpr static float kTotalSimulatedTime_ = 100.f / 48.f;

        typedef MPMDomainRange<64, 64, 64> DomainRange_;
        typedef MPMDomainOffset<0, 0, 0> DomainOffset_;
        typedef MPMDomain<DomainRange_, DomainOffset_> Domain_;
        typedef MPMGridConfig<Domain_> GridConfig_;

        class MPMTestFluidDamBreakConfig : public MPMConfigBase<MPMTestFluidDamBreakConfig>
        {
           public:
            friend class MPMConfigBase<MPMTestFluidDamBreakConfig>;

            constexpr MPMTestFluidDamBreakConfig() = default;

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

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxActiveBlockCountImpl() const -> uint32_t { return 50000; }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetFpsImpl() const -> uint32_t { return kFps_; }

            // constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetDtImpl() const -> float { return EvaluateTestTimestep(kDtFactor_, GetDxImpl(), kE_, kNu_, kRho_, kCfl_); }
            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetDtImpl() const -> float { return 5e-5; }

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

        typedef MPMTestFluidDamBreakConfig TestConfig_;
    };

    
    auto SetupModel(uint32_t& particleCount) -> std::vector<MPMModelVariant>
    {
        constexpr float kDx = MPMTestScene::kDx_;

        std::vector<Vector<float, 3>> position;
        std::vector<Vector<float, 3>> velocity;
        velocity.emplace_back(Vector<float, 3>{0, 0, 0});

        for(int i = 8; i < 247; ++i)
        {
            for(int j = 8; j < 120; ++j)
            {
                for(int k = 8; k < 50; ++k)
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

        return { MPMModel<std::decay_t<decltype(MPMTestScene::kMaterial_)>>{MPMTestScene::kParticleMass_, MPMTestScene::kParticleVolume_, position, velocity, MPMTestScene::kMaterial_} };
    }
    
}

}
