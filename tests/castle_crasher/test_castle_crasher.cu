#include "test_castle_crasher.cuh"
#include "mpm_engine.cuh"


int main()
{
    uint32_t particleCount = 0;
    auto modelList = mpm::test::SetupModel(particleCount);
    auto config = mpm::test::GetTestConfiguration<mpm::test::MPMTestScene>();

    std::filesystem::path resultDirectory = "result/" + mpm::test::GetTestName<mpm::test::MPMTestScene>() + "/";
    std::filesystem::create_directory(resultDirectory);

    auto simulator = mpm::test::GetSimulator<typename mpm::test::MPMTestScene>();
	simulator->Initialize(config);
	simulator->InitializeParticle(config, modelList);
	simulator->InitialSetup(config);
	simulator->Simulate(config, resultDirectory.c_str(), false, false, true);
}



