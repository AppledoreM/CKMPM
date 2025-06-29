#pragma once
#include "mpm_engine.cuh"

#include <filesystem>
#include <string>
#include <string_view>

namespace mpm
{
namespace test
{
const std::filesystem::path kTestAssetRootDirectory = "asset/";

template <typename T>
constexpr auto EvaluateSoundSpeed(const T E, const T nu, const T rho) -> T
{
	return std::sqrt(E * (1 - nu) / ((1 + nu) * (1 - 2 * nu) * rho));
}

template <typename T>
constexpr auto EvaluateTimestep(const T dx, const T E, const T nu, const T rho, const T cfl = 0.5f) -> T
{
	return cfl * dx / EvaluateSoundSpeed(E, nu, rho);
}

template <typename T>
constexpr auto EvaluateTestTimestep(const T dtFactor, const T dx, const T E, const T nu, const T rho,
									const T cfl = 0.5f) -> T
{
	return dtFactor * EvaluateTimestep<T>(dx, E, nu, rho, cfl);
}

template <typename TestScene>
auto GetTestConfiguration() -> typename TestScene::TestConfig_
{
    return typename TestScene::TestConfig_{};
}

template <typename TestScene>
auto GetTestName() -> std::string
{
	return std::string{TestScene::kTestName_.begin(), TestScene::kTestName_.end()};
}

template <typename TestScene>
auto GetSimulator() -> MPMEngine<typename TestScene::GridConfig_>*
{
	return MPMEngine<typename TestScene::GridConfig_>::GetInstance();
}

}  // namespace test
}  // namespace mpm
