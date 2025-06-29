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
        constexpr static std::string_view kTestName_ = "fire_hydrant";

        constexpr static float kDx_ = 1.f / 512.f;
        constexpr static float kParticlePerCell_ = 8.0f;
        constexpr static float kParticleVolume_ = 1 / 512.f / 512.f / 512.f / kParticlePerCell_;

        constexpr static float kFireHydrantE_ = 1e9;
        constexpr static float kFireHydrantNu_ = 0.4f;
        constexpr static float kFireHydrantLambda_ = ComputeLameParameters<float>(kFireHydrantE_, kFireHydrantNu_)[0];
        constexpr static float kFireHydrantMu_ = ComputeLameParameters<float>(kFireHydrantE_, kFireHydrantNu_)[1];
        constexpr static float kFireHydrantYieldStress_ = 3e6;
        constexpr static auto kFireHydrantMaterial_ = MPMMaterial<MPMConstitutiveModel::kVonMises>{ MPMMaterial<MPMConstitutiveModel::kVonMises>::VonMisesMaterialParameter{kFireHydrantLambda_, kFireHydrantMu_, kFireHydrantYieldStress_} };

        constexpr static float kFireHydrantRho_ = 1e5;
        constexpr static float kFireHydrantParticleMass_ = kParticleVolume_ * kFireHydrantRho_;


        constexpr static float kBallE_ = 1e8;
        constexpr static float kBallNu_ = 0.4f;
        constexpr static float kBallLambda_ = ComputeLameParameters<float>(kBallE_, kBallNu_)[0];
        constexpr static float kBallMu_ = ComputeLameParameters<float>(kBallE_, kBallNu_)[1];
        constexpr static auto kBallMaterial_ = MPMMaterial<MPMConstitutiveModel::kFixedCorotated>{ MPMMaterial<MPMConstitutiveModel::kFixedCorotated>::FixedCorotatedMaterialParameter{kBallLambda_, kBallMu_} };
        constexpr static float kBallRho_ = 1e4;
        constexpr static float kBallParticleMass_ = kParticleVolume_ * kBallRho_;

        constexpr static float kFluidE_ = 5e3;
        constexpr static float kFluidNu_ = 0.4f;
        constexpr static float kFluidLambda_ = ComputeLameParameters<float>(kFluidE_, kFluidNu_)[0];
        constexpr static float kFluidMu_ = ComputeLameParameters<float>(kFluidE_, kFluidNu_)[1];
        constexpr static float kFluidBulk_ = 10.f;
        constexpr static float kFluidGamma_ = 7.15f;
        constexpr static float kFluidViscosity_ = 0.1f;
        constexpr static auto kFluidMaterial_ = MPMMaterial<MPMConstitutiveModel::kFluid>{ MPMMaterial<MPMConstitutiveModel::kFluid>::FluidMaterialParameter{kFluidBulk_, kFluidGamma_, kFluidViscosity_, 0.2} };
        constexpr static float kFluidRho_ = 1000.0f;
        constexpr static float kFluidParticleMass_ = kParticleVolume_ * kFluidRho_;


        constexpr static uint32_t kFps_ = 2400;
        constexpr static float kCfl_ = 0.5f;
        constexpr static float kDtFactor_= 1.0f;
        //constexpr static float kTotalSimulatedTime_ = 240.f / 48.f;
        // constexpr static float kTotalSimulatedTime_ = 100.f / 240.f;
        constexpr static float kTotalSimulatedTime_ = 100.f / 2400.f;

        typedef MPMDomainRange<128, 192, 128> DomainRange_;
        typedef MPMDomainOffset<0, 0, 0> DomainOffset_;
        typedef MPMDomain<DomainRange_, DomainOffset_> Domain_;
        typedef MPMGridConfig<Domain_> GridConfig_;

        class MPMFireHydrantConfig : public MPMConfigBase<MPMFireHydrantConfig>
        {
           public:
            friend class MPMConfigBase<MPMFireHydrantConfig>;

            constexpr MPMFireHydrantConfig() = default;


           protected:
            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetDxImpl() const -> float { return kDx_;}

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetBlockVolumeImpl() const -> uint32_t { return 64; }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetBlockSizeImpl() const -> uint32_t { return 4; }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxParticleCountPerCellImpl() const -> uint32_t { return 12; }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxParticleCountPerBlockImpl() const -> uint32_t
            {
                return GetBlockVolumeImpl() * GetMaxParticleCountPerCellImpl();
            }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxParticleCountPerBucketImpl() const -> uint32_t
            {
                return 32;
            }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetMaxActiveBlockCountImpl() const -> uint32_t { return 900000; }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetFpsImpl() const -> uint32_t { return kFps_; }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetDtImpl() const -> float 
            { 
                auto fireHydrantTimestep = EvaluateTestTimestep(kDtFactor_, GetDxImpl(), kFireHydrantE_, kFireHydrantNu_, kFireHydrantRho_, kCfl_);
                auto fluidTimestep = EvaluateTestTimestep(kDtFactor_, GetDxImpl(), kFluidE_, kFluidNu_, kFluidRho_, kCfl_);
                auto ballTimestep = EvaluateTestTimestep(kDtFactor_, GetDxImpl(), kBallE_, kBallNu_, kBallRho_, kCfl_);
                return std::min(fireHydrantTimestep, std::min(fluidTimestep, ballTimestep)); 
            }

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
                return 0;
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
            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetRigidParticleVelocityImpl(int frame) const -> Vector<Scalar, 3>
            {
            }

            constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetExistIrregularBoundaryImpl() const -> bool
            {
                return true;
            }

            MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto ProcessGridCellVelocityImpl(const Vector<int, 3>& cell, Vector<float, 3>& velocity, int frame) const -> void
            {
                constexpr int kRangeDim[] = {DomainRange_::kDim_[0], DomainRange_::kDim_[1], DomainRange_::kDim_[2]};
                bool isWithinFixedBottom = (202 <= cell[0]) && (cell[0] <= 310);
                isWithinFixedBottom &= (cell[1] <= 50);
                isWithinFixedBottom &= (204 <= cell[2]) && (cell[2] <= 314);

                if(isWithinFixedBottom)
                {
                    velocity = 0.f;
                }
                else
                {
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
                }

            }

            MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto UpdateConfigImpl(float dt, int frame) -> void
            {
            }
       private:

        };

        typedef MPMFireHydrantConfig TestConfig_;
    };


    auto SetupModel(uint32_t& particleCount) -> std::vector<MPMModelVariant>
    {

        std::vector<MPMModelVariant> modelList; 

        // Load fire hydrant 
        {
            constexpr int kFireHydrantParticleCount = 3999705;
            std::vector<Vector<float, 3>> position(kFireHydrantParticleCount);
            std::vector<Vector<float, 3>> velocity;
            velocity.emplace_back(Vector<float, 3>{0.f, 0.f, 0.f});
            const auto testAssetDirectory = kTestAssetRootDirectory / std::filesystem::path{"particle_data"};;
            const auto testAssetName = std::filesystem::path("fire_hydrant.bin");
            std::ifstream testAsset((testAssetDirectory / testAssetName).c_str(), std::ios::in | std::ios::binary);

            if(!testAsset.is_open())
            {
                std::cerr << "Failed to find asset at: " << (testAssetDirectory / testAssetName).c_str() << " for test " << MPMTestScene::kTestName_ << std::endl;
                throw std::runtime_error("");
            }

            testAsset.read(reinterpret_cast<char*>(position.data()), 3 * sizeof(float) * kFireHydrantParticleCount);
            modelList.push_back({ MPMModel<std::decay_t<decltype(MPMTestScene::kFireHydrantMaterial_)>>{MPMTestScene::kFireHydrantParticleMass_, MPMTestScene::kParticleVolume_, position, velocity, MPMTestScene::kFireHydrantMaterial_} });
            
        }

        // Load fire hydrant's water
        {
            constexpr int kFireHydrantWaterParticleCount = 3172158;
            std::vector<Vector<float, 3>> position(kFireHydrantWaterParticleCount);
            std::vector<Vector<float, 3>> velocity;
            velocity.emplace_back(Vector<float, 3>{0.f, 0.f, 0.f});
            const auto testAssetDirectory = kTestAssetRootDirectory / std::filesystem::path{"particle_data"};;
            const auto testAssetName = std::filesystem::path("fire_hydrant_water.bin");
            std::ifstream testAsset((testAssetDirectory / testAssetName).c_str(), std::ios::in | std::ios::binary);

            if(!testAsset.is_open())
            {
                std::cerr << "Failed to find asset at: " << (testAssetDirectory / testAssetName).c_str() << " for test " << MPMTestScene::kTestName_ << std::endl;
                throw std::runtime_error("");
            }

            testAsset.read(reinterpret_cast<char*>(position.data()), 3 * sizeof(float) * kFireHydrantWaterParticleCount);
            modelList.push_back({ MPMModel<std::decay_t<decltype(MPMTestScene::kFluidMaterial_)>>{MPMTestScene::kFluidParticleMass_, MPMTestScene::kParticleVolume_, position, velocity, MPMTestScene::kFluidMaterial_} });

        }

        // Load ball
        {
            constexpr int kBallParticleCount = 161717;
 	    std::vector<Vector<float, 3>> position(kBallParticleCount);
            std::vector<Vector<float, 3>> velocity;
            velocity.emplace_back(Vector<float, 3>{100.f, 0.f, 0.f});
            const auto testAssetDirectory = kTestAssetRootDirectory / std::filesystem::path{"particle_data"};;
            const auto testAssetName = std::filesystem::path("fire_hydrant_ball.bin");
            std::ifstream testAsset((testAssetDirectory / testAssetName).c_str(), std::ios::in | std::ios::binary);

            if(!testAsset.is_open())
            {
                std::cerr << "Failed to find asset at: " << (testAssetDirectory / testAssetName).c_str() << " for test " << MPMTestScene::kTestName_ << std::endl;
                throw std::runtime_error("");
            }

            testAsset.read(reinterpret_cast<char*>(position.data()), 3 * sizeof(float) * kBallParticleCount);
            modelList.push_back({ MPMModel<std::decay_t<decltype(MPMTestScene::kBallMaterial_)>>{MPMTestScene::kBallParticleMass_, MPMTestScene::kParticleVolume_, position, velocity, MPMTestScene::kBallMaterial_} });
        }

        return modelList;
    }
    
}

}
