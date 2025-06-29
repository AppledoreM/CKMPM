#include "mpm_engine.cuh"
                                    /* if(val != 0.0) printf("Writing to grid: %f, Result: %f\n", val, block->template GetValue<attributeIndex>(dataIndex)); */
#include "mpm_model.h"
#include "data_type.cuh"
#include "mpm_material.cuh"
#include <vector>

std::vector<mpm::Vector<float, 3>> particlePosition;
std::vector<mpm::Vector<float, 3>> particleVelocity;

int main(int argc, char* argv[])
{



    particlePosition.emplace_back(mpm::Vector<float, 3>{4, 4, 4});
    /* particlePosition.emplace_back(mpm::Vector<float, 3>{2, 2, 2}); */
    /* particlePosition.emplace_back(mpm::Vector<float, 3>{3, 3, 3}); */
    particleVelocity.emplace_back(mpm::Vector<float, 3>{0, -0.1, 0});

    mpm::MPMModel<mpm::MPMMaterial<mpm::MPMConstitutiveModel::kLinear>> model(1.0f, particlePosition, particleVelocity);

    auto config = mpm::MPMDefaultStaticConfig<0.1f>{1e-4, 60, 0.5, 10.0};
    auto enginePtr = mpm::MPMEngine::GetInstance();
    enginePtr->Initialize(config);
    std::vector modelList{model};
    enginePtr->InitializeParticle(config, modelList);
    enginePtr->InitialSetup(config);
    enginePtr->Simulate(config);



	return 0;
}
