#pragma once
#include "cuda_util.cuh"
#include "data_type.cuh"
#include "mpm_config.h"
#include "mpm_grid.cuh"
#include "mpm_kernel.cuh"
#include "mpm_material.cuh"
#include "mpm_meta.h"
#include "mpm_particle.cuh"
#include "mpm_partition.cuh"

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <cooperative_groups.h>
#include <array>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace mpm
{

template <typename Scalar>
auto ExclusiveScan(uint32_t scanCount, Scalar const* const input, Scalar* output,
				   cudaStream_t stream = cudaStreamDefault) -> void
{
	auto policy = thrust::cuda::par.on(stream);
	thrust::exclusive_scan(policy, input, input + scanCount, output);
}

template <typename Index>
__global__ auto InverseExclusiveScan(uint32_t scanCount, const Index* sum, Index* index) -> void
{
	const uint32_t flattenedIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (flattenedIndex >= scanCount)
		return;

	const auto s = sum[flattenedIndex];
	if (s != sum[flattenedIndex + 1])
		{
			index[s] = flattenedIndex;
		}
}

MPM_FORCE_INLINE __device__ auto atomicMax(float* address, float val) -> float
{
	int* address_as_i = (int*)address;
	int old = *address_as_i, assumed;
	do
		{
			assumed = old;
			old = ::atomicCAS(address_as_i, assumed, __float_as_int(::fmaxf(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}

template <int gridIndex = -1, typename Config, typename Scalar>
MPM_FORCE_INLINE __device__ auto GetCellIndex(const Config&,
											  const Vector<Scalar, 3>& particlePosition) -> Vector<int, 3>
{
	constexpr auto config = Config{};
	return Vector<int, 3>{static_cast<int>(particlePosition[0] * config.GetInvDx() - gridIndex * 0.25f),
						  static_cast<int>(particlePosition[1] * config.GetInvDx() - gridIndex * 0.25f),
						  static_cast<int>(particlePosition[2] * config.GetInvDx() - gridIndex * 0.25f)};
}

enum class AtomicAggregatedOperation
{
	Add
};

template <AtomicAggregatedOperation Operation, typename Scalar>
__device__ auto atomicAggregatedOperation(Scalar* x) -> Scalar
{
	auto g = cooperative_groups::coalesced_threads();
	if constexpr (Operation == AtomicAggregatedOperation::Add)
		{
			Scalar baseValue;
			if (g.thread_rank() == 0)
				baseValue = atomicAdd(x, g.size());
			return g.shfl(baseValue, 0) + g.thread_rank();
		}

	return {};
}

template <typename Scalar>
struct MPMParticleBucketIntermediateStorageBuffer
{
	Scalar position_[3];
	Scalar J_;
};

template <MPMConstitutiveModel ConstitutiveModel, typename Config, typename Particle, typename Scalar>
MPM_FORCE_INLINE MPM_DEV_FUNC auto UpdateForce(
	const Config& config, const MPMMaterial<ConstitutiveModel>& material,
	const MPMParticleBuffer<Particle>& particleBuffer, const MPMParticleBuffer<Particle>& nextParticleBuffer,
	int sourceBucketIndex, int sourceParticleIndexInBlock, int blockIndex, int particleIndexInBlock, double dt,
	const Matrix<float, 3, 3>& covariantVelocity, Matrix<float, 3, 3>& contribution,
	MPMParticleBucketIntermediateStorageBuffer<Scalar>& storageBuffer) -> void
{
}

template <MPMConstitutiveModel ConstitutiveModel, typename Config, typename Particle, typename Scalar>
	requires(ConstitutiveModel == MPMConstitutiveModel::kLinear) &&
	std::is_same_v<Particle, typename MPMMaterial<ConstitutiveModel>::Particle_> MPM_FORCE_INLINE MPM_DEV_FUNC
	auto UpdateForce(const Config&, const MPMMaterial<ConstitutiveModel>& material,
					 const MPMParticleBuffer<Particle>& particleBuffer,
					 const MPMParticleBuffer<Particle>& nextParticleBuffer, int sourceBucketIndex,
					 int sourceParticleIndexInBlock, int blockIndex, int particleIndexInBlock, float dt,
					 const Matrix<float, 3, 3>& covariantVelocity, Matrix<float, 3, 3>& contribution,
					 MPMParticleBucketIntermediateStorageBuffer<Scalar>& storageBuffer) -> void
{
	constexpr auto config = Config{};
    const float lambda = material.parameter_.lambda_;
    const float mu = material.parameter_.mu_;

	auto F = Matrix<float, 3, 3>{};

    {

        auto dW = Matrix<float, 3, 3>{};

        meta::ConstexprLoop<0, 9>(
            [&](auto indexWrapper) -> void
            {
                constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
                constexpr float kIdentityMatrixValue = (index & 0x3) ? 0.f : 1.f;
                dW[index] = covariantVelocity[index] * dt + kIdentityMatrixValue;
            });

        auto bucket =
            particleBuffer.template GetBucket<config.GetMaxParticleCountPerBucket()>(sourceBucketIndex);

            meta::ConstexprLoop<0, 9>(
                [&](auto indexWrapper) -> void
                {
                    constexpr auto static index = meta::ConstexprLoopIndex(indexWrapper);
                    F[index] = bucket.template GetAttribute<index + 3>(
                        sourceParticleIndexInBlock % config.GetMaxParticleCountPerBucket());
                });

            meta::ConstexprLoop<0, 9>(
                [&](auto indexWrapper) -> void
                {
                    constexpr auto static index = meta::ConstexprLoopIndex(indexWrapper);
                    contribution[index] = bucket.template GetAttribute<index + 12>(
                        sourceParticleIndexInBlock % config.GetMaxParticleCountPerBucket());
                });
        F = dW.MatrixMultiplication(F);
    }

    contribution +=  0.5 * 100 * (covariantVelocity + covariantVelocity.Transpose() - Matrix<float, 3, 3>::Identity()) * dt;
    contribution *= F.Determinant();
    // printf("F det: %f\n", F.Determinant());
    contribution[3] = contribution[4] = contribution[5] = 0;
    contribution[6] = contribution[7] = contribution[8] = 0;

	{
        auto bucket = nextParticleBuffer.template GetBucket<config.GetMaxParticleCountPerBucket()>(
            nextParticleBuffer.particleBinOffset_[blockIndex] +
            particleIndexInBlock / config.GetMaxParticleCountPerBucket());

        meta::ConstexprLoop<0, 21>(
            [&](auto indexWrapper) -> void
            {
                constexpr static auto index = meta::ConstexprLoopIndex(indexWrapper);
                if constexpr (index < 3)
                    bucket.template SetAttribute<index>(
                        particleIndexInBlock % config.GetMaxParticleCountPerBucket(),
                        storageBuffer.position_[index]);
                else if constexpr(index < 12)
                    {
                        bucket.template SetAttribute<index>(
                            particleIndexInBlock % config.GetMaxParticleCountPerBucket(), F[index - 3]);
                    }

                else
                {
                    bucket.template SetAttribute<index>(
                        particleIndexInBlock % config.GetMaxParticleCountPerBucket(), contribution[index - 12]);
                }
            });
	}



}

template <MPMConstitutiveModel ConstitutiveModel, typename Config, typename Particle, typename Scalar>
	requires(ConstitutiveModel == MPMConstitutiveModel::kFixedCorotated) &&
	std::is_same_v<Particle, typename MPMMaterial<ConstitutiveModel>::Particle_> MPM_FORCE_INLINE MPM_DEV_FUNC
	auto UpdateForce(const Config&, const MPMMaterial<ConstitutiveModel>& material,
					 const MPMParticleBuffer<Particle>& particleBuffer,
					 const MPMParticleBuffer<Particle>& nextParticleBuffer, int sourceBucketIndex,
					 int sourceParticleIndexInBlock, int blockIndex, int particleIndexInBlock, float dt,
					 const Matrix<float, 3, 3>& covariantVelocity, Matrix<float, 3, 3>& contribution,
					 MPMParticleBucketIntermediateStorageBuffer<Scalar>& storageBuffer) -> void
{
	constexpr auto config = Config{};
	auto F = Matrix<float, 3, 3>{};

    {
        auto dW = Matrix<float, 3, 3>{};

        meta::ConstexprLoop<0, 9>(
            [&](auto indexWrapper) -> void
            {
                constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
                constexpr float kIdentityMatrixValue = (index & 0x3) ? 0.f : 1.f;
                dW[index] = covariantVelocity[index] * dt + kIdentityMatrixValue;
            });

        {
            auto bucket =
                particleBuffer.template GetBucket<config.GetMaxParticleCountPerBucket()>(sourceBucketIndex);

            meta::ConstexprLoop<0, 9>(
                [&](auto indexWrapper) -> void
                {
                    constexpr auto static index = meta::ConstexprLoopIndex(indexWrapper);
                    contribution[index] = bucket.template GetAttribute<index + 3>(
                        sourceParticleIndexInBlock % config.GetMaxParticleCountPerBucket());
                });

            F = dW.MatrixMultiplication(contribution);
        }
    }

	{
        auto bucket = nextParticleBuffer.template GetBucket<config.GetMaxParticleCountPerBucket()>(
            nextParticleBuffer.particleBinOffset_[blockIndex] +
            particleIndexInBlock / config.GetMaxParticleCountPerBucket());

        meta::ConstexprLoop<0, 12>(
            [&](auto indexWrapper) -> void
            {
                constexpr static auto index = meta::ConstexprLoopIndex(indexWrapper);
                if constexpr (index < 3)
                    bucket.template SetAttribute<index>(
                        particleIndexInBlock % config.GetMaxParticleCountPerBucket(),
                        storageBuffer.position_[index]);
                else
                    {
                        bucket.template SetAttribute<index>(
                            particleIndexInBlock % config.GetMaxParticleCountPerBucket(), F[index - 3]);
                    }
            });
        ComputeStress<ConstitutiveModel>(particleBuffer.GetParticleVolume(), material.parameter_.mu_,
                                         material.parameter_.lambda_, F, contribution);
	}
}





template <MPMConstitutiveModel ConstitutiveModel, typename Config, typename Particle, typename Scalar>
	requires(ConstitutiveModel == MPMConstitutiveModel::kDruckerPragerStvkhencky) &&
	std::is_same_v<Particle, typename MPMMaterial<ConstitutiveModel>::Particle_> 
    MPM_FORCE_INLINE MPM_DEV_FUNC
	auto UpdateForce(const Config&, const MPMMaterial<ConstitutiveModel>& material,
					 const MPMParticleBuffer<Particle>& particleBuffer,
					 const MPMParticleBuffer<Particle>& nextParticleBuffer, int sourceBucketIndex,
					 int sourceParticleIndexInBlock, int blockIndex, int particleIndexInBlock, float dt,
					 const Matrix<float, 3, 3>& covariantVelocity, Matrix<float, 3, 3>& contribution,
					 MPMParticleBucketIntermediateStorageBuffer<Scalar>& storageBuffer) -> void
{
	constexpr auto config = Config{};
	auto F = Matrix<float, 3, 3>{};
    float logJp = 0.f;

    {
        auto dW = Matrix<float, 3, 3>{};

        meta::ConstexprLoop<0, 9>(
            [&](auto indexWrapper) -> void
            {
                constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
                constexpr float kIdentityMatrixValue = (index & 0x3) ? 0.f : 1.f;
                dW[index] = covariantVelocity[index] * dt + kIdentityMatrixValue;
            });

        {
            auto bucket =
                particleBuffer.template GetBucket<config.GetMaxParticleCountPerBucket()>(sourceBucketIndex);

            meta::ConstexprLoop<0, 9>(
                [&](auto indexWrapper) -> void
                {
                    constexpr auto static index = meta::ConstexprLoopIndex(indexWrapper);
                    contribution[index] = bucket.template GetAttribute<index + 3>(
                        sourceParticleIndexInBlock % config.GetMaxParticleCountPerBucket());
                });

            F = dW.MatrixMultiplication(contribution);

            logJp = bucket.template GetAttribute<12>(sourceParticleIndexInBlock % config.GetMaxParticleCountPerBucket());
        }
    }
    ComputeStress<ConstitutiveModel>(particleBuffer.GetParticleVolume(), material.parameter_.mu_,
                                     material.parameter_.lambda_, F, logJp, contribution);

	{
        auto bucket = nextParticleBuffer.template GetBucket<config.GetMaxParticleCountPerBucket()>(
            nextParticleBuffer.particleBinOffset_[blockIndex] +
            particleIndexInBlock / config.GetMaxParticleCountPerBucket());

        meta::ConstexprLoop<0, 12>(
            [&](auto indexWrapper) -> void
            {
                constexpr static auto index = meta::ConstexprLoopIndex(indexWrapper);
                if constexpr (index < 3)
                    bucket.template SetAttribute<index>(
                        particleIndexInBlock % config.GetMaxParticleCountPerBucket(),
                        storageBuffer.position_[index]);
                else
                    {
                        bucket.template SetAttribute<index>(
                            particleIndexInBlock % config.GetMaxParticleCountPerBucket(), F[index - 3]);
                    }
            });
        bucket.template SetAttribute<12>(particleIndexInBlock % config.GetMaxParticleCountPerBucket(), logJp);
	}
}

template <MPMConstitutiveModel ConstitutiveModel, typename Config, typename Particle, typename Scalar>
	requires(ConstitutiveModel == MPMConstitutiveModel::kNonAssociatedCamClay) &&
	std::is_same_v<Particle, typename MPMMaterial<ConstitutiveModel>::Particle_> 
    MPM_FORCE_INLINE MPM_DEV_FUNC
	auto UpdateForce(const Config&, const MPMMaterial<ConstitutiveModel>& material,
					 const MPMParticleBuffer<Particle>& particleBuffer,
					 const MPMParticleBuffer<Particle>& nextParticleBuffer, int sourceBucketIndex,
					 int sourceParticleIndexInBlock, int blockIndex, int particleIndexInBlock, float dt,
					 const Matrix<float, 3, 3>& covariantVelocity, Matrix<float, 3, 3>& contribution,
					 MPMParticleBucketIntermediateStorageBuffer<Scalar>& storageBuffer) -> void
{
	constexpr auto config = Config{};
	auto F = Matrix<float, 3, 3>{};
    float logJp = 0.f;

    {
        auto dW = Matrix<float, 3, 3>{};

        meta::ConstexprLoop<0, 9>(
            [&](auto indexWrapper) -> void
            {
                constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
                constexpr float kIdentityMatrixValue = (index & 0x3) ? 0.f : 1.f;
                dW[index] = covariantVelocity[index] * dt + kIdentityMatrixValue;
            });

        {
            auto bucket =
                particleBuffer.template GetBucket<config.GetMaxParticleCountPerBucket()>(sourceBucketIndex);

            logJp = bucket.template GetAttribute<3>(sourceParticleIndexInBlock % config.GetMaxParticleCountPerBucket());

            meta::ConstexprLoop<0, 9>(
                [&](auto indexWrapper) -> void
                {
                    constexpr auto static index = meta::ConstexprLoopIndex(indexWrapper);
                    contribution[index] = bucket.template GetAttribute<index + 4>(
                        sourceParticleIndexInBlock % config.GetMaxParticleCountPerBucket());
                });

            F = dW.MatrixMultiplication(contribution);
        }
    }

    ComputeStress<ConstitutiveModel>(material.parameter_, F, logJp, contribution);

	{
        auto bucket = nextParticleBuffer.template GetBucket<config.GetMaxParticleCountPerBucket()>(
            nextParticleBuffer.particleBinOffset_[blockIndex] +
            particleIndexInBlock / config.GetMaxParticleCountPerBucket());

        meta::ConstexprLoop<0, 13>(
            [&](auto indexWrapper) -> void
            {
                constexpr static auto index = meta::ConstexprLoopIndex(indexWrapper);
                if constexpr (index < 3)
                    bucket.template SetAttribute<index>(
                        particleIndexInBlock % config.GetMaxParticleCountPerBucket(),
                        storageBuffer.position_[index]);
                 if constexpr (index == 3)
                {
                    bucket.template SetAttribute<index>(
                        particleIndexInBlock % config.GetMaxParticleCountPerBucket(), logJp);
                }
                if constexpr (index > 3)
                {
                    bucket.template SetAttribute<index>(
                        particleIndexInBlock % config.GetMaxParticleCountPerBucket(), F[index - 4]);
                }
            });
	}
}


template <MPMConstitutiveModel ConstitutiveModel, typename Config, typename Particle, typename Scalar>
	requires(ConstitutiveModel == MPMConstitutiveModel::kVonMises) &&
	std::is_same_v<Particle, typename MPMMaterial<ConstitutiveModel>::Particle_> 
    MPM_FORCE_INLINE MPM_DEV_FUNC
	auto UpdateForce(const Config&, const MPMMaterial<ConstitutiveModel>& material,
					 const MPMParticleBuffer<Particle>& particleBuffer,
					 const MPMParticleBuffer<Particle>& nextParticleBuffer, int sourceBucketIndex,
					 int sourceParticleIndexInBlock, int blockIndex, int particleIndexInBlock, float dt,
					 const Matrix<float, 3, 3>& covariantVelocity, Matrix<float, 3, 3>& contribution,
					 MPMParticleBucketIntermediateStorageBuffer<Scalar>& storageBuffer) -> void
{
	constexpr auto config = Config{};
	auto elasticDeformationGradient = Matrix<float, 3, 3>{};
	auto F = Matrix<float, 3, 3>{};

    {
        auto dW = Matrix<float, 3, 3>{};

        meta::ConstexprLoop<0, 9>(
            [&](auto indexWrapper) -> void
            {
                constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
                constexpr float kIdentityMatrixValue = (index & 0x3) ? 0.f : 1.f;
                dW[index] = covariantVelocity[index] * dt + kIdentityMatrixValue;
            });

        {
            auto bucket =
                particleBuffer.template GetBucket<config.GetMaxParticleCountPerBucket()>(sourceBucketIndex);

            meta::ConstexprLoop<0, 9>(
                [&](auto indexWrapper) -> void
                {
                    constexpr auto static index = meta::ConstexprLoopIndex(indexWrapper);
                    elasticDeformationGradient[index] = bucket.template GetAttribute<index + 4>(
                        sourceParticleIndexInBlock % config.GetMaxParticleCountPerBucket());
                });

            elasticDeformationGradient = dW.MatrixMultiplication(elasticDeformationGradient);
        }
    }


    // Update Plasticity
	{
		Matrix<float, 3, 3> U, V;
		Vector<float, 3> S;

		svd(elasticDeformationGradient[0], elasticDeformationGradient[1], elasticDeformationGradient[2], elasticDeformationGradient[3],
			elasticDeformationGradient[4], elasticDeformationGradient[5], elasticDeformationGradient[6], elasticDeformationGradient[7], elasticDeformationGradient[8],
			U[0], U[1], U[2], U[3], U[4], U[5], U[6], U[7], U[8],
			S[0], S[1], S[2], V[0], V[1], V[2], V[3], V[4], V[5], V[6], V[7], V[8]);

		Vector<float, 3> epsilon;
		float epsilonTrace = 0.f;
		meta::ConstexprLoop<0, 3>([&](auto indexWrapper) -> void
								  {
									constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
									epsilon[index] = logf(S[index]);
									epsilonTrace += epsilon[index];
								  });
		Vector<float, 3> epsilonHat;
		meta::ConstexprLoop<0, 3>([&](auto indexWrapper) -> void
								  {
									constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
									epsilonHat[index] = epsilon[index] - (epsilonTrace / 3.f);
								  });

		float epsilonHatNorm = epsilonHat.Norm();

		float yieldFactor = epsilonHatNorm - material.parameter_.yieldStress_ / (2 * material.parameter_.mu_);

		if(yieldFactor > 0)
		{
			auto tmp = Matrix<float, 3, 3>{};
			meta::ConstexprLoop<0, 3>(
				[&](auto indexWrapper) -> void
				{
				  constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
				  float factor = expf(epsilon[index] - yieldFactor * epsilonHat[index] / epsilonHatNorm);
				  tmp[3 * index] = factor * V[index];
				  tmp[3 * index + 1] = factor * V[3 + index];
				  tmp[3 * index + 2] = factor * V[6 + index];
				}
			);
			elasticDeformationGradient = U.MatrixMultiplication(tmp);
		}
	}

	ComputeStress<ConstitutiveModel>(particleBuffer.GetParticleVolume(), material.parameter_.mu_,
									 material.parameter_.lambda_, material.parameter_.yieldStress_, elasticDeformationGradient, elasticDeformationGradient, contribution);


	{
		auto bucket = nextParticleBuffer.template GetBucket<config.GetMaxParticleCountPerBucket()>(
			nextParticleBuffer.particleBinOffset_[blockIndex] +
			particleIndexInBlock / config.GetMaxParticleCountPerBucket());

		meta::ConstexprLoop<0, 13>(
			[&](auto indexWrapper) -> void
			{
			  constexpr static auto index = meta::ConstexprLoopIndex(indexWrapper);
			  if constexpr (index < 3)
			  {
				  bucket.template SetAttribute<index>(
					  particleIndexInBlock % config.GetMaxParticleCountPerBucket(),
					  storageBuffer.position_[index]);
			  }

			  if constexpr(index == 3)
			  {
				  bucket.template SetAttribute<index>(
					  particleIndexInBlock % config.GetMaxParticleCountPerBucket(), storageBuffer.J_);
			  }

			  if constexpr(4 <= index && index < 13)
			  {
				  bucket.template SetAttribute<index>(
					  particleIndexInBlock % config.GetMaxParticleCountPerBucket(), elasticDeformationGradient[index - 4]);
			  }
			});
	}
}

template <MPMConstitutiveModel ConstitutiveModel, typename Config, typename Particle, typename Scalar>
requires(ConstitutiveModel == MPMConstitutiveModel::kFluid) &&
std::is_same_v<Particle, typename MPMMaterial<ConstitutiveModel>::Particle_>
MPM_FORCE_INLINE MPM_DEV_FUNC
auto UpdateForce(const Config&, const MPMMaterial<ConstitutiveModel>& material,
				 const MPMParticleBuffer<Particle>& particleBuffer,
				 const MPMParticleBuffer<Particle>& nextParticleBuffer, int sourceBucketIndex,
				 int sourceParticleIndexInBlock, int blockIndex, int particleIndexInBlock, float dt,
				 const Matrix<float, 3, 3>& covariantVelocity, Matrix<float, 3, 3>& contribution,
				 MPMParticleBucketIntermediateStorageBuffer<Scalar>& storageBuffer) -> void
{
	constexpr auto config = Config{};
	float J = storageBuffer.J_;;

	J += (covariantVelocity[0] + covariantVelocity[4] + covariantVelocity[8]) * dt * J;

	{
		float pressure = material.parameter_.bulk_ * (powf(J, -material.parameter_.gamma_) - 1.f);
		ComputeStress<ConstitutiveModel>(particleBuffer.GetParticleVolume(), pressure,
										 material.parameter_.viscosity_, covariantVelocity, J, contribution);
	}

	{
		auto bucket = nextParticleBuffer.template GetBucket<config.GetMaxParticleCountPerBucket()>(
			nextParticleBuffer.particleBinOffset_[blockIndex] +
			particleIndexInBlock / config.GetMaxParticleCountPerBucket());

		meta::ConstexprLoop<0, 4>(
			[&](auto indexWrapper) -> void
			{
			  constexpr static auto index = meta::ConstexprLoopIndex(indexWrapper);
			  if constexpr (index < 3)
				  bucket.template SetAttribute<index>(
					  particleIndexInBlock % config.GetMaxParticleCountPerBucket(),
					  storageBuffer.position_[index]);
			  else
			  {
				  bucket.template SetAttribute<index>(
					  particleIndexInBlock % config.GetMaxParticleCountPerBucket(), J);
			  }
			});
	}
}


template <MPMConfigType Config>
__global__ auto ComputeBinCount(const Config& config, uint32_t blockCount, uint32_t const* particleBucketSize,
								uint32_t* binCount) -> void
{
	const int blockIndex = blockDim.x * blockIdx.x + threadIdx.x;
	if (blockIndex >= blockCount)
		return;

	binCount[blockIndex] = (particleBucketSize[blockIndex] + config.GetMaxParticleCountPerBucket() - 1) /
						   config.GetMaxParticleCountPerBucket();
}


template <MPMConfigType Config, typename ParticleArray, typename Particle, typename Partition>
requires MPMFiniteDomainType<typename Partition::GridConfig_::Domain_> __global__ auto BuildParticleCellBucket(
	Config, uint32_t particleCount, ParticleArray particleArray, MPMParticleBuffer<Particle> buffer,
	Partition partition) -> void
{
	constexpr Config config;

	for (uint32_t particleIndex = blockDim.x * blockIdx.x + threadIdx.x; particleIndex < particleCount;
		 particleIndex += gridDim.x * blockDim.x)
	{
		Vector<int, 3> cellIndex = GetCellIndex(config, particleArray[particleIndex]) - 2;

		const MPMGridBlockCoordinate gridBlockIndex =
			MPMGridBlockCoordinate{cellIndex[0] / static_cast<int>(config.GetBlockSize()),
								   cellIndex[1] / static_cast<int>(config.GetBlockSize()),
								   cellIndex[2] / static_cast<int>(config.GetBlockSize())};
		/* printf("Particle (%f, %f, %f) is assigned to grid block (%d, %d, %d)\n", */
		/*        particleArray[particleIndex][0], particleArray[particleIndex][1], particleArray[particleIndex][2], */
		/*        gridBlockIndex[0], gridBlockIndex[1], gridBlockIndex[2]); */

		const uint32_t flattenedGridBlockIndex = partition.Find(gridBlockIndex);

		const auto flattenedCellIndex = GetFlattenedIndex<config.GetBlockSize(), config.GetBlockSize()>(
			cellIndex[0] % static_cast<int>(config.GetBlockSize()),
			cellIndex[1] % static_cast<int>(config.GetBlockSize()),
			cellIndex[2] % static_cast<int>(config.GetBlockSize()));

		auto particleIndexInCell = atomicAdd(
			buffer.cellParticleCount_ + flattenedGridBlockIndex * config.GetBlockVolume() + flattenedCellIndex, 1);
		if (particleIndexInCell >= config.GetMaxParticleCountPerCell())
		{
			atomicSub(buffer.cellParticleCount_ + flattenedGridBlockIndex * config.GetBlockVolume() +
					  flattenedCellIndex,
					  1);
			return;
		}

		buffer.cellBucket_[flattenedGridBlockIndex * config.GetMaxParticleCountPerBlock() +
						   flattenedCellIndex * config.GetMaxParticleCountPerCell() + particleIndexInCell] =
			particleIndex;
	}
}

template <MPMConfigType Config, typename Particle>
__global__ auto ParticleCellBucketToBlock(Config, MPMParticleBuffer<Particle> buffer) -> void
{
	constexpr Config config = {};
	const int cellIndexInBlock = static_cast<int>(threadIdx.x) % config.GetBlockVolume();
	const uint32_t particleCount = buffer.cellParticleCount_[blockIdx.x * config.GetBlockVolume() + cellIndexInBlock];

	for (uint32_t particleIndexInCell = 0; particleIndexInCell < config.GetMaxParticleCountPerCell();
		 ++particleIndexInCell)
	{
		if (particleIndexInCell < particleCount)
		{
			const auto particleIndexInBlock = atomicAggregatedOperation<AtomicAggregatedOperation::Add>(
				buffer.particleBucketSize_ + blockIdx.x);
			buffer.blockBucket_[blockIdx.x * config.GetMaxParticleCountPerBlock() + particleIndexInBlock] =
				buffer
					.cellBucket_[blockIdx.x * config.GetMaxParticleCountPerBlock() +
								 cellIndexInBlock * config.GetMaxParticleCountPerCell() + particleIndexInCell];
		}
		__syncthreads();
	}
}

enum class MPMActivateBlockPolicy
{
	Neighbor,
	Exterior
};

template <MPMActivateBlockPolicy Policy, MPMConfigType Config, typename Partition>
requires MPMFiniteDomainType<typename Partition::GridConfig_::Domain_> __global__ auto ActivateBlocks(
	Config, uint32_t blockCount, Partition partition) -> void
{

	typedef typename Partition::GridConfig_ GridConfig;
	typedef typename GridConfig::Domain_ Domain;
	typedef typename Domain::DomainRange_ DomainRange;
	typedef typename Domain::DomainOffset_ DomainOffset;
	constexpr int kRangeDim[] = {DomainRange::kDim_[0], DomainRange::kDim_[1], DomainRange::kDim_[2]};
	constexpr int kOffset[] = {DomainOffset::kOffset_[0], DomainOffset::kOffset_[1], DomainOffset::kOffset_[2]};

	const uint32_t blockIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (blockIndex >= blockCount)
		return;

	const auto blockCoord = partition.activeKeys_[blockIndex];
	constexpr bool activateNeighborBlocks = Policy == MPMActivateBlockPolicy::Neighbor;
	constexpr int activateBlockCountPerThread = activateNeighborBlocks ? 8 : 27;

	meta::ConstexprLoop<0, activateBlockCountPerThread>(
		[&](auto indexWrapper) -> void
		{
		  constexpr auto index = meta::ConstexprLoopIndex(indexWrapper);
		  constexpr int i = activateNeighborBlocks ? (index & 4) >> 2 : (index / 9) - 1;
		  constexpr int j = activateNeighborBlocks ? (index & 2) >> 1 : (index / 3) % 3 - 1;
		  constexpr int k = activateNeighborBlocks ? (index & 1) : (index % 3) - 1;
		  constexpr auto offset = Vector<int, 3>{i, j, k};

		  const auto blockIndex = offset + blockCoord;
		  //TODO: Optimize MPMGridBLockCoordiante Additions

		  if (blockIndex[0] >= 0 && blockIndex[1] >= 0 && blockIndex[2] >= 0 && blockIndex[0] < kRangeDim[0] &&
			  blockIndex[1] < kRangeDim[1] && blockIndex[2] < kRangeDim[2])
		  {
			  partition.Insert(MPMGridBlockCoordinate{blockIndex[0], blockIndex[1], blockIndex[2]});
		  }
		});
}

template <MPMConfigType Config, typename Partition>
requires MPMFiniteDomainType<typename Partition::GridConfig_::Domain_> __global__ auto SyncNeighborBlocks(
	Config, uint32_t blockCount, Partition destinationPartition, Partition sourcePartition) -> void
{
	const uint32_t blockIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (blockIndex >= blockCount)
		return;
	const auto key = sourcePartition.activeKeys_[blockIndex];
	if (destinationPartition.Find(key) == Partition::kValueSentinelValue_)
	{
		destinationPartition.Insert(key);
	}
}

template <MPMConfigType Config, typename Partition, typename ParticleArray>
requires MPMFiniteDomainType<typename Partition::GridConfig_::Domain_> __global__ auto ActivateBlocksWithParticles(
	Config, uint32_t particleCount, ParticleArray particle, Partition partition) -> void
{
	constexpr Config config;
	constexpr float invDx = config.GetInvDx();

	for (uint32_t particleIndex = blockIdx.x * blockDim.x + threadIdx.x; particleIndex < particleCount;
		 particleIndex += gridDim.x * blockDim.x)
	{
		meta::ConstexprLoop<-1, 2, 2>(
			[&](auto gridIndexWrapper) -> void
			{
			  constexpr int gridIndex = meta::ConstexprLoopIndex(gridIndexWrapper);
			  const Vector<float, 3>& p = particle[particleIndex];
			  const auto cellIndex = GetCellIndex<gridIndex>(config, p);
			  auto blockIndex =
				  MPMGridBlockCoordinate{(cellIndex[0] - 2) / static_cast<int>(config.GetBlockSize()),
										 (cellIndex[1] - 2) / static_cast<int>(config.GetBlockSize()),
										 (cellIndex[2] - 2) / static_cast<int>(config.GetBlockSize())};

			  partition.Insert(blockIndex);
			});
	}
}

template <bool kEnableGravity, typename Config, typename GridBlock, typename Partition>
__global__ auto UpdateGridVelocityAndQueryMax(const Config config, uint32_t blockCount, MPMGrid<GridBlock> grid,
											  const Partition partition, double dt, int frame, float* maxVelocityNorm) -> void
{
	typedef typename Partition::GridConfig_ GridConfig;
	typedef typename GridConfig::Domain_ Domain;
	typedef typename Domain::DomainRange_ DomainRange;
	typedef typename Domain::DomainOffset_ DomainOffset;
	constexpr int kRangeDim[] = {DomainRange::kDim_[0], DomainRange::kDim_[1], DomainRange::kDim_[2]};
	constexpr int kOffset[] = {DomainOffset::kOffset_[0], DomainOffset::kOffset_[1], DomainOffset::kOffset_[2]};
	constexpr int kWarpSize = 32;
	constexpr int kBlockCountPerCudaBlock = kWarpSize * 16;
	const int flattenedBlockIndex =
		static_cast<int>(blockIdx.x) * kBlockCountPerCudaBlock / kWarpSize + static_cast<int>(threadIdx.x) / kWarpSize;
    constexpr Config config_ = {};
	constexpr float kInfinity = std::numeric_limits<float>::infinity();

	__shared__ float maxVelocityNormInBlock[kBlockCountPerCudaBlock / kWarpSize];

	if (threadIdx.x < kBlockCountPerCudaBlock / kWarpSize)
		maxVelocityNormInBlock[threadIdx.x] = 0;
	__syncthreads();

	if (flattenedBlockIndex < blockCount)
	{
		auto blockIndex = partition.activeKeys_[flattenedBlockIndex];
		auto block = grid.GetBlock(flattenedBlockIndex);
		char isCellOutOfBound =
			((blockIndex[0] + kOffset[0] < 2 || blockIndex[0] + kOffset[0] + 2 >= kRangeDim[0]) << 2) |
			((blockIndex[1] + kOffset[1] < 2 || blockIndex[1] + kOffset[1] + 2 >= kRangeDim[1]) << 1) |
			(blockIndex[2] + kOffset[2] < 2 || blockIndex[2] + kOffset[2] + 2 >= kRangeDim[2]);
		for (int cellIndexInBlock = static_cast<int>(threadIdx.x) % kWarpSize; cellIndexInBlock < 2 * config_.GetBlockVolume();
			 cellIndexInBlock += kWarpSize)
		{

			float velocityNormSquared = 0.0f;
			const float mass = block->template GetValue<0>(cellIndexInBlock);
			Vector<float, 3> velocity = {};

			if (mass > config.GetMassClamp())
			{
				const float invMass = 1.0f / mass;

				meta::ConstexprLoop<0, 3>(
					[&](auto indexWrapper) -> void
					{
					  constexpr int index = meta::ConstexprLoopIndex(indexWrapper);

					  velocity[index] = block->template GetValue<1 + index>(cellIndexInBlock) * invMass;

					  if constexpr(index == 1 && kEnableGravity)
					  {
						  velocity[index] += config_.GetGravity() * dt;
					  }
                      }
                      );

                if constexpr(config_.GetExistIrregularBoundary())
                {
                    Vector<int, 3> cell = Vector<int, 3>{ (cellIndexInBlock >> 4) & 3, (cellIndexInBlock >> 2) & 3, cellIndexInBlock & 3} + blockIndex * config_.GetBlockSize();
                    config.ProcessGridCellVelocity(cell, velocity, frame);
                }
                else
                {
                    velocity[0] = isCellOutOfBound ? 0.f : velocity[0];
                    velocity[1] = isCellOutOfBound ? 0.f : velocity[1];
                    velocity[2] = isCellOutOfBound ? 0.f : velocity[2];
                }



				meta::ConstexprLoop<0, 3>(
					[&](auto indexWrapper) -> void
					{
					  constexpr int index = meta::ConstexprLoopIndex(indexWrapper);


					  block->template SetValue<1 + index>(cellIndexInBlock, velocity[index]);
					  velocityNormSquared += velocity[index] * velocity[index];
					});
			}
			else
			{
				block->template SetValue<1>(cellIndexInBlock, 0.f);
				block->template SetValue<2>(cellIndexInBlock, 0.f);
				block->template SetValue<3>(cellIndexInBlock, 0.f);
			}
			/* printf("Block Index: (%d, %d, %d), Flattened Block Index: %d, Grid Index: %d, Mass: %.15f, Velocity norm: %f, velocity component: (%f, %f, %f), CellIndexInBlock: %d\n", */
			/*        blockIndex[0], blockIndex[1], blockIndex[2], flattenedBlockIndex, */
			/*        cellIndexInBlock >= 2 * config.GetBlockVolume(), */
			/*        mass, sqrt(velocityNormSquared), velocity[0], velocity[1], velocity[2], cellIndexInBlock); */

			if (isnan(velocityNormSquared))
				velocityNormSquared = kInfinity;

			for (int iter = 1; iter % kWarpSize; iter <<= 1)
			{
				float tmp = __shfl_down_sync(0xffffffff, velocityNormSquared, iter, kWarpSize);
				if ((threadIdx.x % kWarpSize) + iter < kWarpSize)
				{
					velocityNormSquared = tmp > velocityNormSquared ? tmp : velocityNormSquared;
				}
			}

			if (velocityNormSquared > maxVelocityNormInBlock[threadIdx.x / kWarpSize] &&
				(threadIdx.x % kWarpSize) == 0)
			{
				maxVelocityNormInBlock[threadIdx.x / kWarpSize] = velocityNormSquared;
			}
		}
	}

	__syncthreads();
	for (int interval = (kBlockCountPerCudaBlock / kWarpSize) >> 1; interval > 0; interval >>= 1)
	{
		if (static_cast<int>(threadIdx.x) < interval)
		{
			if (maxVelocityNormInBlock[static_cast<int>(threadIdx.x) + interval] >
				maxVelocityNormInBlock[threadIdx.x])
			{
				maxVelocityNormInBlock[threadIdx.x] =
					maxVelocityNormInBlock[static_cast<int>(threadIdx.x) + interval];
			}
		}
		__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		atomicMax(maxVelocityNorm, maxVelocityNormInBlock[0]);
	}
}

template <typename Config, typename GridBlock, typename Partition, typename ParticleArray>
__global__ auto Rasterize(Config, uint32_t particleCount, const ParticleArray particleArray, MPMGrid<GridBlock> grid,
						  const Partition partition, float dt, float mass, uint32_t velocityCount,
						  ParticleArray particleVelocityArray) -> void
{
	constexpr auto config = Config{};
	const float dx = config.GetDx();
	const float invDx = 1.0f / dx;

	for (int particleIndex = threadIdx.x + blockIdx.x * blockDim.x; particleIndex < particleCount;
		 particleIndex += gridDim.x * blockDim.x)
	{
		const Vector<float, 3> particlePosition = particleArray[particleIndex];
		const auto particleVelocity =
			velocityCount == 1 ? particleVelocityArray[0] : particleVelocityArray[particleIndex];

		meta::ConstexprLoop<-1, 2, 2>(
			[&](auto gridIndexWrapper) -> void
			{
			  constexpr int gridIndex = meta::ConstexprLoopIndex(gridIndexWrapper);
			  constexpr int consecutiveGridIndex = gridIndex == -1 ? 0 : 1;

			  const auto particleCellIndex = GetCellIndex<gridIndex>(config, particlePosition);
			  const auto localParticlePosition = particlePosition * config.GetInvDx() - particleCellIndex;

			  typedef MPMKernel<SmoothLinear<float, gridIndex>> Kernel;
              Vector<float, 3> weightStencil;

              Kernel::OptimizedWeightComputation(localParticlePosition, weightStencil);

			  meta::ConstexprLoop<0, 8>(
				  [&](auto indexWrapper) -> void
				  {
					constexpr auto index = meta::ConstexprLoopIndex(indexWrapper);
					constexpr int di = (index & 4) >> 2;
					constexpr int dj = (index & 2) >> 1;
					constexpr int dk = index & 1;

					const auto sampleCellIndex = Vector<int, 3>{
						particleCellIndex[0] + di, particleCellIndex[1] + dj, particleCellIndex[2] + dk};

					const int flattenedBlockIndex = partition.Find(
						MPMGridBlockCoordinate{static_cast<int>(sampleCellIndex[0] / config.GetBlockSize()),
											   static_cast<int>(sampleCellIndex[1] / config.GetBlockSize()),
											   static_cast<int>(sampleCellIndex[2] / config.GetBlockSize())});

                    // const float weight = weightStencil[di] * weightStencil[2 + dj] * weightStencil[4 + dk];
                    float weight = 1.f;
                    if constexpr(di == 0) { weight *= weightStencil[0]; }
                    else weight *= (1 - weightStencil[0]);

                    if constexpr(dj == 0) { weight *= weightStencil[1]; }
                    else weight *= (1 - weightStencil[1]);

                    if constexpr(dk == 0) { weight *= weightStencil[2]; }
                    else weight *= (1 - weightStencil[2]);

					float wm = weight * mass;
					auto block = grid.GetBlock(flattenedBlockIndex);

					int flattenedCellIndexInBlock =
						GetFlattenedIndex<config.GetBlockSize(), config.GetBlockSize(), config.GetBlockSize()>(
							consecutiveGridIndex, static_cast<int>(sampleCellIndex[0] % config.GetBlockSize()),
							static_cast<int>(sampleCellIndex[1] % config.GetBlockSize()),
							static_cast<int>(sampleCellIndex[2] % config.GetBlockSize()));

					atomicAdd(&block->template GetValue<0>(flattenedCellIndexInBlock), wm);
					atomicAdd(&block->template GetValue<1>(flattenedCellIndexInBlock),
							  wm * particleVelocity[0]);
					atomicAdd(&block->template GetValue<2>(flattenedCellIndexInBlock),
							  wm * particleVelocity[1]);
					atomicAdd(&block->template GetValue<3>(flattenedCellIndexInBlock),
							  wm * particleVelocity[2]);
				  });
			});
	}
}

template <MPMConfigType Config>
__global__ auto InitializeAdvectionBucket(const Config& config, const uint32_t* particleBucketSize,
										  uint32_t* cellBucket) -> void
{

	const auto particleCount = particleBucketSize[blockIdx.x];
	uint32_t* bucket = cellBucket + static_cast<size_t>(blockIdx.x) * config.GetMaxParticleCountPerBlock();

	for (int particleIndexInBlock = static_cast<int>(threadIdx.x); particleIndexInBlock < particleCount;
		 particleIndexInBlock += static_cast<int>(blockDim.x))
	{

		bucket[particleIndexInBlock] =
			(GetFlattenedIndex<3, 3>(1, 1, 1) * config.GetMaxParticleCountPerBlock()) + particleIndexInBlock;
	}
}

/* template <MPMConfigType Config, MPMConstitutiveModel ConstitutiveModel, typename ParticleArray, typename Particle> */
/* __global__ auto CopyParticleArrayToParticleBuffer(const Config&, uint32_t particleCount, ParticleArray particleArray, */
/* 												  MPMParticleBuffer<Particle> buffer) -> void */
/* { */

/* } */

template <MPMConfigType Config, MPMConstitutiveModel ConstitutiveModel, typename ParticleArray, typename Particle>
__global__ auto CopyParticleArrayToParticleBuffer(const Config&, uint32_t particleCount, MPMMaterial<ConstitutiveModel> material, ParticleArray particleArray,
												  MPMParticleBuffer<Particle> buffer) -> void
{
	constexpr static auto config = Config{};
	const uint32_t particleCountInBlock = buffer.particleBucketSize_[blockIdx.x];
	const uint32_t* particleBucketInBlock = buffer.blockBucket_ + blockIdx.x * config.GetMaxParticleCountPerBlock();

	for (int particleIndexInBlock = static_cast<int>(threadIdx.x); particleIndexInBlock < particleCountInBlock;
		 particleIndexInBlock += blockDim.x)
	{
		const uint32_t particleIndex = particleBucketInBlock[particleIndexInBlock];
		auto particleBucket = buffer.GetBucket<config.GetMaxParticleCountPerBucket()>(
			buffer.particleBinOffset_[blockIdx.x] + particleIndexInBlock / config.GetMaxParticleCountPerBucket());
		const auto particlePos = particleArray[particleIndex];

		meta::ConstexprLoop<0, 3>(
			[&](auto indexWrapper) -> void
			{
			  constexpr auto index = meta::ConstexprLoopIndex(indexWrapper);
			  particleBucket.template SetAttribute<index>(
				  particleIndexInBlock % config.GetMaxParticleCountPerBucket(), particlePos[index]);
			});

		if constexpr (ConstitutiveModel == MPMConstitutiveModel::kLinear)
		{

			meta::ConstexprLoop<0, 9>(
				[&](auto indexWrapper) -> void
				{
				  constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
				  constexpr float kIdentityMatrixValue = (index & 0x3) ? 0.f : 1.f;
				  particleBucket.template SetAttribute<3 + index>(
					  particleIndexInBlock % config.GetMaxParticleCountPerBucket(), kIdentityMatrixValue);
				});

			meta::ConstexprLoop<0, 9>(
				[&](auto indexWrapper) -> void
				{
				  constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
				  particleBucket.template SetAttribute<12 + index>(
					  particleIndexInBlock % config.GetMaxParticleCountPerBucket(), 0);
				});
		}


		if constexpr (ConstitutiveModel == MPMConstitutiveModel::kFixedCorotated)
		{
			meta::ConstexprLoop<0, 9>(
				[&](auto indexWrapper) -> void
				{
				  constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
				  constexpr float kIdentityMatrixValue = (index & 0x3) ? 0.f : 1.f;
				  particleBucket.template SetAttribute<3 + index>(
					  particleIndexInBlock % config.GetMaxParticleCountPerBucket(), kIdentityMatrixValue);
				});
		}

		if constexpr (ConstitutiveModel == MPMConstitutiveModel::kDruckerPragerStvkhencky)
		{
			meta::ConstexprLoop<0, 9>(
				[&](auto indexWrapper) -> void
				{
				  constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
				  constexpr float kIdentityMatrixValue = (index & 0x3) ? 0.f : 1.f;
				  particleBucket.template SetAttribute<3 + index>(
					  particleIndexInBlock % config.GetMaxParticleCountPerBucket(), kIdentityMatrixValue);
				});
			particleBucket.template SetAttribute<12>(particleIndexInBlock % config.GetMaxParticleCountPerBucket(), MPMMaterial<ConstitutiveModel>::MaterialParameter_::kLogJp0);
		}

		if constexpr(ConstitutiveModel == MPMConstitutiveModel::kFluid)
		{
			particleBucket.template SetAttribute<3>(particleIndexInBlock % config.GetMaxParticleCountPerBucket(), material.GetJ0());
		}

		if constexpr (ConstitutiveModel == MPMConstitutiveModel::kNonAssociatedCamClay)
		{
			particleBucket.template SetAttribute<3>(
				particleIndexInBlock % config.GetMaxParticleCountPerBucket(), material.parameter_.alpha0_);

			// Fill both elastic / plastic deformation gradient as identity
			meta::ConstexprLoop<0, 9>(
				[&](auto indexWrapper) -> void
				{
				  constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
				  constexpr float kIdentityMatrixValue = (index & 0x3) ? 0.f : 1.f;
				  particleBucket.template SetAttribute<4 + index>(
					  particleIndexInBlock % config.GetMaxParticleCountPerBucket(), kIdentityMatrixValue);
				});
		}

		if constexpr (ConstitutiveModel == MPMConstitutiveModel::kVonMises)
		{
			particleBucket.template SetAttribute<3>(
				particleIndexInBlock % config.GetMaxParticleCountPerBucket(), particleIndex >= config.GetRigidParticleCount() ? 1.f : 0.f);

			// Fill both elastic / plastic deformation gradient as identity
			meta::ConstexprLoop<0, 9>(
				[&](auto indexWrapper) -> void
				{
				  constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
				  constexpr float kIdentityMatrixValue = (index & 0x3) ? 0.f : 1.f;
				  particleBucket.template SetAttribute<4 + index>(
					  particleIndexInBlock % config.GetMaxParticleCountPerBucket(), kIdentityMatrixValue);
				});
		}
	}
}
template <typename Config, typename ParticleBuffer,
	typename Partition, typename GridBlock>
requires MPMFiniteDomainType<typename Partition::GridConfig_::Domain_> 
__global__ auto CollectConservationMetric(
	const ParticleBuffer buffer, ParticleBuffer nextBuffer, const Partition prevPartition, Partition partition, const MPMGrid<GridBlock> grid, Vector<float, 3>* momentum) -> void
{
	constexpr auto config = Config{};

	typedef typename Partition::GridConfig_ GridConfig;
	typedef typename GridConfig::Domain_ Domain;
	typedef typename Domain::DomainRange_ DomainRange;
	typedef typename Domain::DomainOffset_ DomainOffset;
	constexpr int kDim[] = {DomainRange::kDim_[0], DomainRange::kDim_[1], DomainRange::kDim_[2]};
	constexpr int kOffset[] = {DomainOffset::kOffset_[0], DomainOffset::kOffset_[1], DomainOffset::kOffset_[2]};

    auto& eulerianLinearMomentum = momentum[0];
    auto& eulerianAngularMomentum = momentum[1];
    auto& lagrangianLinearMomentum = momentum[2];
    auto& lagrangianAngularMomentum = momentum[3];

	static constexpr uint64_t kG2PBlockBufferBlockSize = (config.GetBlockVolume() * 6) ;
	static constexpr uint64_t kG2PBlockBufferArenaSize = (8 * 8 * 8 + 7 * 7 * 7) * 3;

	static constexpr uint64_t kP2GBlockBufferBlockSize = (config.GetBlockVolume() << 3);
	static constexpr uint64_t kP2GBlockBufferArenaSize = (8 * 8 * 8 + 7 * 7 * 7) * 4;

	const uint32_t particleBucketSize = nextBuffer.particleBucketSize_[blockIdx.x];
	if (particleBucketSize == 0)
		return;

	__shared__ uint8_t sharedBuffer[(kP2GBlockBufferArenaSize + kG2PBlockBufferArenaSize) * sizeof(float)];
    __shared__ Vector<float, 3> blockMomentum[4];

    if(threadIdx.x == 0)
    {
        blockMomentum[0] = 0.f;
        blockMomentum[1] = 0.f;
        blockMomentum[2] = 0.f;
        blockMomentum[3] = 0.f;
    }
    __syncthreads();




	// Two background scratch pads for 2 different grids

	float * __restrict__ g2pBuffer[2], * __restrict__ p2gBuffer[2];
	g2pBuffer[0] = reinterpret_cast<float*>(sharedBuffer);
	g2pBuffer[1] = reinterpret_cast<float*>(sharedBuffer + 8 * 8 * 8 * 3 * sizeof(float));

	p2gBuffer[0] = reinterpret_cast<float*>(sharedBuffer + kG2PBlockBufferArenaSize * sizeof(float));
	p2gBuffer[1] = reinterpret_cast<float*>(sharedBuffer + kG2PBlockBufferArenaSize * sizeof(float) + 8 * 8 * 8 * 4 * sizeof(float));

	/*
         *  The steps will be:
         *  - Load the grid data to scratch pad. There are two grids data that needed to be stored as we are using a dual-grid scheme.
         *  - For each particle in the particle blocks, they are stored in the format of particle buckets.
         *    - There are two particle buffers. Each contain lots of little particle buckets.
         *      In each buckets, the particles are stored in form of (directions) | (particleId) in upper and lower bits.
         *    - The variable buffer stored the particle informations from last frame.
         *    - Using the particle Id and flattened block id, we can get the offset of the particle bins
         *    - With this, we can retrieve the position values (optionally also the Jacobian) from buffer
         *    - Then, we can perform the standard G2P. Note that we need to perform it on both grids and take average.
         *    - With the new particle positions, we perform P2G onto both scratch pads.
         *    - Finally, we store the scratch pads back to the grid.
         *
         * Special Modification:
         * - Particle buffer and GpuHashtables store the particle information in the original grid
         * - Scratch pads and grids store the information of two offset grids
		 */

	const Vector<int, 3> globalBlockIndex = partition.activeKeys_[blockIdx.x];
	const Vector<int, 3> globalCellIndex = globalBlockIndex * static_cast<int>(config.GetBlockSize());

    auto IsInSharedBufferFlattenedIndexRange = [&] __device__ (int consecutiveGridIndex, int i, int j, int k) -> bool
    {
        const int inRange[2] = {true, (i < 7) && (j < 7) && (k < 7)};
        return inRange[consecutiveGridIndex];

    };

    auto GetSharedBufferFlattenedIndex = [&] __device__ (int consecutiveGridIndex, int c, int i, int j, int k) -> int
    {

        const int flattenedIndex[2] = {GetFlattenedIndex<2 * config.GetBlockSize(), 2 * config.GetBlockSize(), 2 * config.GetBlockSize()>(c, i, j, k), GetFlattenedIndex<7, 7, 7>(c, i, j, k)};

        return flattenedIndex[consecutiveGridIndex];
    };

	// Load G2P Buffer & Clean G2P Buffer

	for (uint32_t index = threadIdx.x; index < 8 * kG2PBlockBufferBlockSize; index += blockDim.x)
	{
		const char localBlockIndex = index / kG2PBlockBufferBlockSize;
		const auto flattenedBlockIndex = partition.Find(MPMGridBlockCoordinate{
			globalBlockIndex[0] + ((localBlockIndex & 4) >> 2),
			globalBlockIndex[1] + ((localBlockIndex & 2) >> 1),
			globalBlockIndex[2] + (localBlockIndex & 1)});

		const auto gridBlock = grid.GetBlock(flattenedBlockIndex);

		int channelIndex = static_cast<int>(index % kG2PBlockBufferBlockSize);
		const char dataIndex = static_cast<char>(channelIndex % (2 * config.GetBlockVolume()));

		const char dataIndexZ = static_cast<char>(channelIndex % config.GetBlockSize());
		channelIndex /= config.GetBlockSize();

		const char dataIndexY = static_cast<char>(channelIndex % config.GetBlockSize());
		channelIndex /= config.GetBlockSize();

		const char dataIndexX = static_cast<char>(channelIndex % config.GetBlockSize());
		channelIndex /= config.GetBlockSize();

		const char consecutiveGridIndex = static_cast<char>(channelIndex & 1);
		channelIndex >>= 1;

		float val;
		if (channelIndex == 0)
			val = gridBlock->template GetValue<1>(dataIndex);
		else if (channelIndex == 1)
			val = gridBlock->template GetValue<2>(dataIndex);
		else
			val = gridBlock->template GetValue<3>(dataIndex);

        if(IsInSharedBufferFlattenedIndexRange(consecutiveGridIndex, static_cast<int>(dataIndexX + (localBlockIndex & 4 ? config.GetBlockSize() : 0)), static_cast<int>(dataIndexY + (localBlockIndex & 2 ? config.GetBlockSize() : 0)), static_cast<int>(dataIndexZ + (localBlockIndex & 1 ? config.GetBlockSize() : 0))))
        {
            g2pBuffer[consecutiveGridIndex][GetSharedBufferFlattenedIndex(consecutiveGridIndex, channelIndex, static_cast<int>(dataIndexX + (localBlockIndex & 4 ? config.GetBlockSize() : 0)), static_cast<int>(dataIndexY + (localBlockIndex & 2 ? config.GetBlockSize() : 0)), static_cast<int>(dataIndexZ + (localBlockIndex & 1 ? config.GetBlockSize() : 0)))] = val;
        }

	}
	__syncthreads();


	// For each particle in the particle block, process its update
	for (uint32_t particleIndexInBlock = threadIdx.x; particleIndexInBlock < particleBucketSize;
		 particleIndexInBlock += blockDim.x)
	{
		uint32_t advectionSourceParticleBinIndex;
		uint32_t sourceParticleIndex;
		{
			Vector<int, 3> sourceDirection = {};
			const uint32_t advectMasks =
				nextBuffer.blockBucket_[blockIdx.x * config.GetMaxParticleCountPerBlock() + particleIndexInBlock];
			int direction = advectMasks / config.GetMaxParticleCountPerBlock();
			sourceDirection[0] = (direction / 9) - 1;
			sourceDirection[1] = ((direction / 3) % 3) - 1;
			sourceDirection[2] = (direction % 3) - 1;

			sourceParticleIndex = advectMasks % config.GetMaxParticleCountPerBlock();
			sourceDirection += globalBlockIndex;

			MPMGridBlockCoordinate gridBlockCoord{sourceDirection[0], sourceDirection[1], sourceDirection[2]};

			const int advectionFlattenedGlobalSourceBlockIndex = prevPartition.Find(gridBlockCoord);
			advectionSourceParticleBinIndex = buffer.particleBinOffset_[advectionFlattenedGlobalSourceBlockIndex] +
											  sourceParticleIndex / config.GetMaxParticleCountPerBucket();
		}

		Vector<float, 3> particlePosition = {};
		Vector<float, 3> particleVelocity = {};
		Vector<int, 3> particleBlockIndexBeforeAdvection = {};

		/* 	//------------------------------------------------Fetch Data------------------------------------------------ */
		float J = 0.f;

		{
			auto bucket = buffer.template GetBucket<static_cast<size_t>(config.GetMaxParticleCountPerBucket())>(
				advectionSourceParticleBinIndex);
			particlePosition[0] =
				bucket.GetAttribute<0>(sourceParticleIndex % config.GetMaxParticleCountPerBucket());
			particlePosition[1] =
				bucket.GetAttribute<1>(sourceParticleIndex % config.GetMaxParticleCountPerBucket());
			particlePosition[2] =
				bucket.GetAttribute<2>(sourceParticleIndex % config.GetMaxParticleCountPerBucket());

			// Load velocity from previous unadvected timestep
			particleBlockIndexBeforeAdvection =
				(GetCellIndex(config, particlePosition) - 2) / static_cast<int>(config.GetBlockSize());
		}

		//------------------------------------------------Update Position------------------------------------------------
		// Formula for computing base cell idnex
		// floor(xp / dx + 0.25) for grid0
		// floor(xp / dx - 0.25) for grid1

		meta::ConstexprLoop<-1, 2, 2>([&](auto gridIndexWrapper) -> void
									  {
										constexpr int gridIndex = meta::ConstexprLoopIndex(gridIndexWrapper);
										constexpr auto consecutiveGridIndex = gridIndex == -1 ? 0 : 1;

										auto cellIndex = GetCellIndex<gridIndex>(config, particlePosition);
										Vector<float, 3> localParticlePosition = particlePosition * config.GetInvDx() - cellIndex;

										// Transform base block index to a "block local" index
										cellIndex -= globalCellIndex;

										// Start Computing Weights
										typedef MPMKernel<SmoothLinear<float, gridIndex>> Kernel;

										Vector<float, 3> weightStencil;

										Kernel::OptimizedWeightComputation(localParticlePosition, weightStencil);

										meta::ConstexprLoop<0, 8>(
											[&](auto spatialIndex) -> void
											{
											  constexpr auto offsetMask = meta::ConstexprLoopIndex(spatialIndex);
											  constexpr int dx = (offsetMask & 4) >> 2;
											  constexpr int dy = (offsetMask & 2) >> 1;
											  constexpr int dz = (offsetMask & 1);
                                              constexpr Vector<float, 3> xi = Vector<float, 3>{ (dx + gridIndex * 0.25f) * config.GetDx(), (dy + gridIndex * 0.25f) * config.GetDx(), (dz + gridIndex * 0.25f) * config.GetDx() };

                                              float weight = 1.f;
                                              if constexpr(dx == 0) { weight *= weightStencil[0]; }
                                              else weight *= (1 - weightStencil[0]);

                                              if constexpr(dy == 0) { weight *= weightStencil[1]; }
                                              else weight *= (1 - weightStencil[1]);

                                              if constexpr(dz == 0) { weight *= weightStencil[2]; }
                                              else weight *= (1 - weightStencil[2]);


											  auto bufferedVelocity = Vector<float, 3>{
												  g2pBuffer[consecutiveGridIndex][GetSharedBufferFlattenedIndex(consecutiveGridIndex, 0, cellIndex[0] + dx, cellIndex[1] + dy, cellIndex[2] + dz)],
												  g2pBuffer[consecutiveGridIndex][GetSharedBufferFlattenedIndex(consecutiveGridIndex, 1, cellIndex[0] + dx, cellIndex[1] + dy, cellIndex[2] + dz)],
												  g2pBuffer[consecutiveGridIndex][GetSharedBufferFlattenedIndex(consecutiveGridIndex, 2, cellIndex[0] + dx, cellIndex[1] + dy, cellIndex[2] + dz)]
											  };

											  particleVelocity += weight * bufferedVelocity;
											});
									  });

            if constexpr(config.GetExistRigidParticle())
            {
                if(J == 0.f)
                {
                    particleVelocity = 0.f;
                }
                else
                {
                    particleVelocity = 0.5f * particleVelocity;
                }
            }
            else
            {

                particleVelocity = 0.5f * particleVelocity;
            }


            auto particleLinearMomentum = particleVelocity;
            auto particleAngularMomentum = Vector<float, 3>{ 
                                    particlePosition[1] * particleVelocity[2] - particlePosition[2] * particleVelocity[1],
                                    particlePosition[2] * particleVelocity[0] - particlePosition[0] * particleVelocity[2],
                                    particlePosition[0] * particleVelocity[1] - particlePosition[1] * particleVelocity[0]
                                };
            atomicAdd(&blockMomentum[2][0], particleLinearMomentum[0]);
            atomicAdd(&blockMomentum[2][1], particleLinearMomentum[1]);
            atomicAdd(&blockMomentum[2][2], particleLinearMomentum[2]);

            atomicAdd(&blockMomentum[3][0], particleAngularMomentum[0]);
            atomicAdd(&blockMomentum[3][1], particleAngularMomentum[1]);
            atomicAdd(&blockMomentum[3][2], particleAngularMomentum[2]);

		}
    __syncthreads();

    if(threadIdx.x == 0)
    {
        atomicAdd(&lagrangianLinearMomentum[0], blockMomentum[2][0]);
        atomicAdd(&lagrangianLinearMomentum[1], blockMomentum[2][1]);
        atomicAdd(&lagrangianLinearMomentum[2], blockMomentum[2][2]);

        atomicAdd(&lagrangianAngularMomentum[0], blockMomentum[3][0]);
        atomicAdd(&lagrangianAngularMomentum[1], blockMomentum[3][1]);
        atomicAdd(&lagrangianAngularMomentum[2], blockMomentum[3][2]);
    }
}



template <typename Config, MPMConstitutiveModel ConstitutiveModel, typename ParticleBuffer,
	typename Partition, typename GridBlock>
requires MPMFiniteDomainType<typename Partition::GridConfig_::Domain_> __global__ auto G2P2G(
	int frame, float dt, float nextDt, bool isFreezed, const MPMMaterial<ConstitutiveModel> material,
	const ParticleBuffer buffer, ParticleBuffer nextBuffer, const Partition prevPartition, Partition partition,
	const MPMGrid<GridBlock> grid, MPMGrid<GridBlock> nextGrid) -> void
{
	constexpr auto config = Config{};

	typedef typename Partition::GridConfig_ GridConfig;
	typedef typename GridConfig::Domain_ Domain;
	typedef typename Domain::DomainRange_ DomainRange;
	typedef typename Domain::DomainOffset_ DomainOffset;
	constexpr int kDim[] = {DomainRange::kDim_[0], DomainRange::kDim_[1], DomainRange::kDim_[2]};
	constexpr int kOffset[] = {DomainOffset::kOffset_[0], DomainOffset::kOffset_[1], DomainOffset::kOffset_[2]};

	static constexpr uint64_t kG2PBlockBufferBlockSize = (config.GetBlockVolume() * 6) ;
	static constexpr uint64_t kG2PBlockBufferArenaSize = (8 * 8 * 8 + 7 * 7 * 7) * 3;

	static constexpr uint64_t kP2GBlockBufferBlockSize = (config.GetBlockVolume() << 3);
	static constexpr uint64_t kP2GBlockBufferArenaSize = (8 * 8 * 8 + 7 * 7 * 7) * 4;

	const uint32_t particleBucketSize = nextBuffer.particleBucketSize_[blockIdx.x];
	if (particleBucketSize == 0)
		return;

	__shared__ uint8_t sharedBuffer[(kP2GBlockBufferArenaSize + kG2PBlockBufferArenaSize) * sizeof(float)];


	// Two background scratch pads for 2 different grids

	float * __restrict__ g2pBuffer[2], * __restrict__ p2gBuffer[2];
	g2pBuffer[0] = reinterpret_cast<float*>(sharedBuffer);
	g2pBuffer[1] = reinterpret_cast<float*>(sharedBuffer + 8 * 8 * 8 * 3 * sizeof(float));

	p2gBuffer[0] = reinterpret_cast<float*>(sharedBuffer + kG2PBlockBufferArenaSize * sizeof(float));
	p2gBuffer[1] = reinterpret_cast<float*>(sharedBuffer + kG2PBlockBufferArenaSize * sizeof(float) + 8 * 8 * 8 * 4 * sizeof(float));

	/*
         *  The steps will be:
         *  - Load the grid data to scratch pad. There are two grids data that needed to be stored as we are using a dual-grid scheme.
         *  - For each particle in the particle blocks, they are stored in the format of particle buckets.
         *    - There are two particle buffers. Each contain lots of little particle buckets.
         *      In each buckets, the particles are stored in form of (directions) | (particleId) in upper and lower bits.
         *    - The variable buffer stored the particle informations from last frame.
         *    - Using the particle Id and flattened block id, we can get the offset of the particle bins
         *    - With this, we can retrieve the position values (optionally also the Jacobian) from buffer
         *    - Then, we can perform the standard G2P. Note that we need to perform it on both grids and take average.
         *    - With the new particle positions, we perform P2G onto both scratch pads.
         *    - Finally, we store the scratch pads back to the grid.
         *
         * Special Modification:
         * - Particle buffer and GpuHashtables store the particle information in the original grid
         * - Scratch pads and grids store the information of two offset grids
		 */

	const Vector<int, 3> globalBlockIndex = partition.activeKeys_[blockIdx.x];
	const Vector<int, 3> globalCellIndex = globalBlockIndex * static_cast<int>(config.GetBlockSize());

    auto IsInSharedBufferFlattenedIndexRange = [&] __device__ (int consecutiveGridIndex, int i, int j, int k) -> bool
    {
        const int inRange[2] = {true, (i < 7) && (j < 7) && (k < 7)};
        return inRange[consecutiveGridIndex];

    };

    auto GetSharedBufferFlattenedIndex = [&] __device__ (int consecutiveGridIndex, int c, int i, int j, int k) -> int
    {

        const int flattenedIndex[2] = {GetFlattenedIndex<2 * config.GetBlockSize(), 2 * config.GetBlockSize(), 2 * config.GetBlockSize()>(c, i, j, k), GetFlattenedIndex<7, 7, 7>(c, i, j, k)};

        return flattenedIndex[consecutiveGridIndex];
    };

	// Load G2P Buffer & Clean G2P Buffer

	for (uint32_t index = threadIdx.x; index < 8 * kG2PBlockBufferBlockSize; index += blockDim.x)
	{
		const char localBlockIndex = index / kG2PBlockBufferBlockSize;
		const auto flattenedBlockIndex = partition.Find(MPMGridBlockCoordinate{
			globalBlockIndex[0] + ((localBlockIndex & 4) >> 2),
			globalBlockIndex[1] + ((localBlockIndex & 2) >> 1),
			globalBlockIndex[2] + (localBlockIndex & 1)});

		const auto gridBlock = grid.GetBlock(flattenedBlockIndex);

		int channelIndex = static_cast<int>(index % kG2PBlockBufferBlockSize);
		const char dataIndex = static_cast<char>(channelIndex % (2 * config.GetBlockVolume()));

		const char dataIndexZ = static_cast<char>(channelIndex % config.GetBlockSize());
		channelIndex /= config.GetBlockSize();

		const char dataIndexY = static_cast<char>(channelIndex % config.GetBlockSize());
		channelIndex /= config.GetBlockSize();

		const char dataIndexX = static_cast<char>(channelIndex % config.GetBlockSize());
		channelIndex /= config.GetBlockSize();

		const char consecutiveGridIndex = static_cast<char>(channelIndex & 1);
		channelIndex >>= 1;

		float val;
		if (channelIndex == 0)
			val = gridBlock->template GetValue<1>(dataIndex);
		else if (channelIndex == 1)
			val = gridBlock->template GetValue<2>(dataIndex);
		else
			val = gridBlock->template GetValue<3>(dataIndex);

        if(IsInSharedBufferFlattenedIndexRange(consecutiveGridIndex, static_cast<int>(dataIndexX + (localBlockIndex & 4 ? config.GetBlockSize() : 0)), static_cast<int>(dataIndexY + (localBlockIndex & 2 ? config.GetBlockSize() : 0)), static_cast<int>(dataIndexZ + (localBlockIndex & 1 ? config.GetBlockSize() : 0))))
        {
            g2pBuffer[consecutiveGridIndex][GetSharedBufferFlattenedIndex(consecutiveGridIndex, channelIndex, static_cast<int>(dataIndexX + (localBlockIndex & 4 ? config.GetBlockSize() : 0)), static_cast<int>(dataIndexY + (localBlockIndex & 2 ? config.GetBlockSize() : 0)), static_cast<int>(dataIndexZ + (localBlockIndex & 1 ? config.GetBlockSize() : 0)))] = val;
        }

	}
	__syncthreads();

	//------------------------------------------------Clear P2G Buffer------------------------------------------------

	{
		float* __restrict__ p2gBufferRaw =
			reinterpret_cast<float*>(sharedBuffer + kG2PBlockBufferArenaSize * sizeof(float));
		for (uint32_t index = threadIdx.x; index < kP2GBlockBufferArenaSize; index += blockDim.x)
		{
			p2gBufferRaw[index] = 0.f;
		}
		__syncthreads();
	}


	// For each particle in the particle block, process its update
	for (uint32_t particleIndexInBlock = threadIdx.x; particleIndexInBlock < particleBucketSize;
		 particleIndexInBlock += blockDim.x)
	{
		uint32_t advectionSourceParticleBinIndex;
		uint32_t sourceParticleIndex;
		{
			Vector<int, 3> sourceDirection = {};
			const uint32_t advectMasks =
				nextBuffer.blockBucket_[blockIdx.x * config.GetMaxParticleCountPerBlock() + particleIndexInBlock];
			int direction = advectMasks / config.GetMaxParticleCountPerBlock();
			sourceDirection[0] = (direction / 9) - 1;
			sourceDirection[1] = ((direction / 3) % 3) - 1;
			sourceDirection[2] = (direction % 3) - 1;

			sourceParticleIndex = advectMasks % config.GetMaxParticleCountPerBlock();
			sourceDirection += globalBlockIndex;

			MPMGridBlockCoordinate gridBlockCoord{sourceDirection[0], sourceDirection[1], sourceDirection[2]};

			const int advectionFlattenedGlobalSourceBlockIndex = prevPartition.Find(gridBlockCoord);
			advectionSourceParticleBinIndex = buffer.particleBinOffset_[advectionFlattenedGlobalSourceBlockIndex] +
											  sourceParticleIndex / config.GetMaxParticleCountPerBucket();
		}

		Vector<float, 3> particlePosition = {};
		Vector<float, 3> particleVelocity = {};
		Vector<int, 3> particleBlockIndexBeforeAdvection = {};
		Matrix<float, 3, 3> contribution = {};
		Matrix<float, 3, 3> D = {};
		Matrix<float, 3, 3> covariantVelocity = {};

		/* 	//------------------------------------------------Fetch Data------------------------------------------------ */
		float J = 0.f;

		{
			auto bucket = buffer.template GetBucket<static_cast<size_t>(config.GetMaxParticleCountPerBucket())>(
				advectionSourceParticleBinIndex);
			particlePosition[0] =
				bucket.GetAttribute<0>(sourceParticleIndex % config.GetMaxParticleCountPerBucket());
			particlePosition[1] =
				bucket.GetAttribute<1>(sourceParticleIndex % config.GetMaxParticleCountPerBucket());
			particlePosition[2] =
				bucket.GetAttribute<2>(sourceParticleIndex % config.GetMaxParticleCountPerBucket());

			if constexpr(ConstitutiveModel == MPMConstitutiveModel::kFluid || config.GetExistRigidParticle())
			{
				J = bucket.GetAttribute<3>(sourceParticleIndex % config.GetMaxParticleCountPerBucket());
			}

			// Load velocity from previous unadvected timestep
			particleBlockIndexBeforeAdvection =
				(GetCellIndex(config, particlePosition) - 2) / static_cast<int>(config.GetBlockSize());
		}

		//------------------------------------------------Update Position------------------------------------------------
		// Formula for computing base cell idnex
		// floor(xp / dx + 0.25) for grid0
		// floor(xp / dx - 0.25) for grid1

		meta::ConstexprLoop<-1, 2, 2>([&](auto gridIndexWrapper) -> void
									  {
										constexpr int gridIndex = meta::ConstexprLoopIndex(gridIndexWrapper);
										constexpr auto consecutiveGridIndex = gridIndex == -1 ? 0 : 1;

										auto cellIndex = GetCellIndex<gridIndex>(config, particlePosition);
										Vector<float, 3> localParticlePosition = particlePosition * config.GetInvDx() - cellIndex;

										// Start Computing Weights
										typedef MPMKernel<SmoothLinear<float, gridIndex>> Kernel;

										// Matrix<float, 2, 3> weightStencil;
										Vector<float, 3> weightStencil;

										Kernel::OptimizedWeightComputation(localParticlePosition, weightStencil);
                                        localParticlePosition = particlePosition - cellIndex * config.GetDx();

										// Transform base block index to a "block local" index
										cellIndex -= globalCellIndex;

										meta::ConstexprLoop<0, 8>(
											[&](auto spatialIndex) -> void
											{
											  constexpr auto offsetMask = meta::ConstexprLoopIndex(spatialIndex);
											  constexpr int dx = (offsetMask & 4) >> 2;
											  constexpr int dy = (offsetMask & 2) >> 1;
											  constexpr int dz = (offsetMask & 1);
                                              constexpr Vector<float, 3> xi = Vector<float, 3>{ (dx + gridIndex * 0.25f) * config.GetDx(), (dy + gridIndex * 0.25f) * config.GetDx(), (dz + gridIndex * 0.25f) * config.GetDx() };

                                              // const float weight = weightStencil[dx] * weightStencil[2 + dy] * weightStencil[4 + dz];
                                              float weight = 1.f;
                                              if constexpr(dx == 0) { weight *= weightStencil[0]; }
                                              else weight *= (1 - weightStencil[0]);

                                              if constexpr(dy == 0) { weight *= weightStencil[1]; }
                                              else weight *= (1 - weightStencil[1]);

                                              if constexpr(dz == 0) { weight *= weightStencil[2]; }
                                              else weight *= (1 - weightStencil[2]);

											  auto bufferedVelocity = Vector<float, 3>{
												  g2pBuffer[consecutiveGridIndex][GetSharedBufferFlattenedIndex(consecutiveGridIndex, 0, cellIndex[0] + dx, cellIndex[1] + dy, cellIndex[2] + dz)],
												  g2pBuffer[consecutiveGridIndex][GetSharedBufferFlattenedIndex(consecutiveGridIndex, 1, cellIndex[0] + dx, cellIndex[1] + dy, cellIndex[2] + dz)],
												  g2pBuffer[consecutiveGridIndex][GetSharedBufferFlattenedIndex(consecutiveGridIndex, 2, cellIndex[0] + dx, cellIndex[1] + dy, cellIndex[2] + dz)]
											  };

											  particleVelocity += weight * bufferedVelocity;
                                              const Vector<float, 3> xixp = xi - localParticlePosition;
                                              covariantVelocity += bufferedVelocity.OuterProduct(weight * xixp);
                                              D += xixp.OuterProduct(weight * xixp);
											});
									  });


		if constexpr(config.GetExistRigidParticle())
		{
			if(J == 0.f)
			{
				covariantVelocity = 0.f;
                particlePosition += dt * config.template GetRigidParticleVelocity<float>(frame);
                particleVelocity = 0.f;
                D = 0.f;
			}
			else
			{
                covariantVelocity = covariantVelocity.MatrixMultiplication(D.Inverse());
				particleVelocity = 0.5f * particleVelocity;
                particlePosition += dt * particleVelocity;
            }
        }
        else
        {
            // if(isFreezed)
            // {
				// covariantVelocity = 0.f;
            //     particleVelocity = 0.f;
            //     D = 0.f;

            // }
            // else
            // {
            covariantVelocity = covariantVelocity.MatrixMultiplication(D.Inverse());
            particleVelocity = 0.5f * particleVelocity;
            particlePosition += dt * particleVelocity;
            // }
        }

			/* //------------------------------------------------Update Force & Store Position------------------------------------------------ */
			MPMParticleBucketIntermediateStorageBuffer<float> storageBuffer = {particlePosition[0], particlePosition[1],
																			   particlePosition[2], J};

            UpdateForce<ConstitutiveModel>(
                config, material, buffer, nextBuffer, advectionSourceParticleBinIndex, sourceParticleIndex,
                blockIdx.x, particleIndexInBlock, dt, covariantVelocity, contribution, storageBuffer);

			//------------------------------------------------P2G Process------------------------------------------------

            // Add advected particle index into the new particle buffer
            {

                auto updatedParticleCellIndex = GetCellIndex(config, particlePosition);

                {
                    Vector<int, 3> blockShiftDirection =
                        particleBlockIndexBeforeAdvection -
                        ((updatedParticleCellIndex - 2) / static_cast<int>(config.GetBlockSize()));

                    const int directionMask = GetFlattenedIndex<3, 3>(
                        blockShiftDirection[0] + 1, blockShiftDirection[1] + 1, blockShiftDirection[2] + 1);

                    nextBuffer.AddAdvection(config, partition, updatedParticleCellIndex, directionMask,
                                            particleIndexInBlock);
                }

                updatedParticleCellIndex -= globalCellIndex;

                /* // Detect if current particle is out of the particle block range */
                // We also need to detect case when = 0 because we assume that the particle would not reach the the bottom cells in grid 0
                if (updatedParticleCellIndex[0] <= 0 || updatedParticleCellIndex[1] <= 0 ||
                    updatedParticleCellIndex[2] <= 0 ||
                    updatedParticleCellIndex[0] >= 2 * config.GetBlockSize() - 1 ||
                    updatedParticleCellIndex[1] >= 2 * config.GetBlockSize() - 1 ||
                    updatedParticleCellIndex[2] >= 2 * config.GetBlockSize() - 1)
                    {
                        printf("Particle is out of particle block after update at (%d, %d, %d)\n",
                               updatedParticleCellIndex[0], updatedParticleCellIndex[1],
                               updatedParticleCellIndex[2]);
                        return;
                    }
            }

            // Matrix<float, 2, 3> savedWeightStencil[2];
            Vector<float, 3> savedWeightStencil[2];
            meta::ConstexprLoop<-1, 2, 2>([&](auto gridIndexWrapper) -> void
                                          {
                                            constexpr int gridIndex = meta::ConstexprLoopIndex(gridIndexWrapper);
                                            constexpr auto consecutiveGridIndex = gridIndex == -1 ? 0 : 1;

                                            auto cellIndex = GetCellIndex<gridIndex>(config, particlePosition);
                                            Vector<float, 3> localParticlePosition = particlePosition * config.GetInvDx() - cellIndex;

                                            // Start Computing Weights
                                            typedef MPMKernel<SmoothLinear<float, gridIndex>> Kernel;

                                            Kernel::OptimizedWeightComputation(localParticlePosition, savedWeightStencil[consecutiveGridIndex]);
                                            auto& weightStencil = savedWeightStencil[consecutiveGridIndex];

                                            localParticlePosition = particlePosition - cellIndex * config.GetDx();

                                            // Transform base block index to a "block local" index
                                            cellIndex -= globalCellIndex;

                                            meta::ConstexprLoop<0, 8>(
                                                [&](auto spatialIndex) -> void
                                                {
                                                  constexpr auto offsetMask = meta::ConstexprLoopIndex(spatialIndex);
                                                  constexpr int dx = (offsetMask & 4) >> 2;
                                                  constexpr int dy = (offsetMask & 2) >> 1;
                                                  constexpr int dz = (offsetMask & 1);
                                                  constexpr Vector<float, 3> xi = Vector<float, 3>{ (dx + gridIndex * 0.25f) * config.GetDx(), (dy + gridIndex * 0.25f) * config.GetDx(), (dz + gridIndex * 0.25f) * config.GetDx() };

                                                  // const float weight = weightStencil[dx] * weightStencil[2 + dy] * weightStencil[4 + dz];

                                                  float weight = 1.f;
                                                  if constexpr(dx == 0) { weight *= weightStencil[0]; }
                                                  else weight *= (1.f - weightStencil[0]);

                                                  if constexpr(dy == 0) { weight *= weightStencil[1]; }
                                                  else weight *= (1.f - weightStencil[1]);

                                                  if constexpr(dz == 0) { weight *= weightStencil[2]; }
                                                  else weight *= (1.f - weightStencil[2]);

                                                  const Vector<float, 3> xixp = xi - localParticlePosition;
                                                  D += xixp.OuterProduct(weight * xixp);
                                                });
                                          });
            D = (0.5f * D).Inverse();

			// Write back to the scratch pad
			// Only write to one part of the grid
            contribution = covariantVelocity * buffer.GetParticleMass() - (nextDt * buffer.GetParticleVolume()) * D.MatrixMultiplication(contribution);
            //contribution = -(nextDt * buffer.GetParticleVolume()) * D.MatrixMultiplication(contribution);

            meta::ConstexprLoop<-1, 2, 2>(
                [&](auto gridIndexWrapper) -> void
                {
                    constexpr int gridIndex = meta::ConstexprLoopIndex(gridIndexWrapper);
                    constexpr int consecutiveGridIndex = gridIndex == -1 ? 0 : 1;

                    auto cellIndex = GetCellIndex<gridIndex>(config, particlePosition);
                    Vector<float, 3> localParticlePosition = particlePosition - cellIndex * config.GetDx();
                    // Transform base block index to a "block local" index
                    cellIndex -= globalCellIndex;

                    typedef MPMKernel<SmoothLinear<float, gridIndex>> Kernel;

                    auto& weightStencil = savedWeightStencil[consecutiveGridIndex];
                    meta::ConstexprLoop<0, 8>(
                        [&](auto indexWrapper) -> void
                        {
                            constexpr auto offsetMask = meta::ConstexprLoopIndex(indexWrapper);
                            constexpr int dx = ((offsetMask & 4) >> 2);
                            constexpr int dy = ((offsetMask & 2) >> 1);
                            constexpr int dz = ((offsetMask & 1));
                            constexpr Vector<float, 3> xi = Vector<float, 3>{ (dx + gridIndex * 0.25f) * config.GetDx(), (dy + gridIndex * 0.25f) * config.GetDx(), (dz + gridIndex * 0.25f) * config.GetDx() };

                            // const float weight = weightStencil[dx] * weightStencil[2 + dy] * weightStencil[4 + dz];

                            float weight = 1.f;
                            if constexpr(dx == 0) { weight *= weightStencil[0]; }
                            else weight *= (1.f - weightStencil[0]);

                            if constexpr(dy == 0) { weight *= weightStencil[1]; }
                            else weight *= (1.f - weightStencil[1]);

                            if constexpr(dz == 0) { weight *= weightStencil[2]; }
                            else weight *= (1.f - weightStencil[2]);

                            const float weightedMass = weight * buffer.GetParticleMass();

                            Vector<float, 3> forceMomentum = weight * contribution.MatrixMultiplication(xi - localParticlePosition);


                            atomicAdd(&p2gBuffer[consecutiveGridIndex][GetSharedBufferFlattenedIndex(consecutiveGridIndex, 0, cellIndex[0] + dx, cellIndex[1] + dy, cellIndex[2] + dz)],
                                      weightedMass);

                            atomicAdd(&p2gBuffer[consecutiveGridIndex][GetSharedBufferFlattenedIndex(consecutiveGridIndex, 1, cellIndex[0] + dx, cellIndex[1] + dy, cellIndex[2] + dz)],
                                      weightedMass * particleVelocity[0] + forceMomentum[0]);

                            atomicAdd(&p2gBuffer[consecutiveGridIndex][GetSharedBufferFlattenedIndex(consecutiveGridIndex, 2, cellIndex[0] + dx, cellIndex[1] + dy, cellIndex[2] + dz)],
                                      weightedMass * particleVelocity[1] + forceMomentum[1]);

                            atomicAdd(&p2gBuffer[consecutiveGridIndex][GetSharedBufferFlattenedIndex(consecutiveGridIndex, 3, cellIndex[0] + dx, cellIndex[1] + dy, cellIndex[2] + dz)],
                                      weightedMass * particleVelocity[2] + forceMomentum[2]);

                        });
                });
		}

	__syncthreads();

	// Write back to grid

	for (int base = static_cast<int>(threadIdx.x); base < 8 * kP2GBlockBufferBlockSize;
		 base += static_cast<int>(blockDim.x))
		{

			const auto localBlockIndex = base / kP2GBlockBufferBlockSize;
			const auto flattenedBlockIndex = partition.Find(
				MPMGridBlockCoordinate{static_cast<int>(globalBlockIndex[0] + ((localBlockIndex & 4) >> 2)),
									   static_cast<int>(globalBlockIndex[1] + ((localBlockIndex & 2) >> 1)),
									   static_cast<int>(globalBlockIndex[2] + (localBlockIndex & 1))});

			int channelIndex = static_cast<int>(base % kP2GBlockBufferBlockSize);
			const char dataIndex = static_cast<char>(channelIndex % (2 * config.GetBlockVolume()));

			const char dataIndexZ = static_cast<char>(channelIndex % config.GetBlockSize());
			channelIndex /= config.GetBlockSize();

			const char dataIndexY = static_cast<char>(channelIndex % config.GetBlockSize());
			channelIndex /= config.GetBlockSize();

			const char dataIndexX = static_cast<char>(channelIndex % config.GetBlockSize());
			channelIndex /= config.GetBlockSize();

			const char consecutiveGridIndex = channelIndex & 1;
			channelIndex >>= 1;

            float val = 0.f;

            if(IsInSharedBufferFlattenedIndexRange(consecutiveGridIndex, static_cast<int>(dataIndexX + (localBlockIndex & 4 ? config.GetBlockSize() : 0)), static_cast<int>(dataIndexY + (localBlockIndex & 2 ? config.GetBlockSize() : 0)), static_cast<int>(dataIndexZ + (localBlockIndex & 1 ? config.GetBlockSize() : 0))))                                   
            {
                val = p2gBuffer[consecutiveGridIndex][GetSharedBufferFlattenedIndex(consecutiveGridIndex, channelIndex, 
                                            static_cast<int>(dataIndexX + (localBlockIndex & 4 ? config.GetBlockSize() : 0)), 
                                            static_cast<int>(dataIndexY + (localBlockIndex & 2 ? config.GetBlockSize() : 0)),
                                            static_cast<int>(dataIndexZ + (localBlockIndex & 1 ? config.GetBlockSize() : 0)))];
            }

			auto block = nextGrid.GetBlock(flattenedBlockIndex);

			if (channelIndex == 0)
				atomicAdd(&block->template GetValue<0>(dataIndex), val);
			else if (channelIndex == 1)
				atomicAdd(&block->template GetValue<1>(dataIndex), val);
			else if (channelIndex == 2)
				atomicAdd(&block->template GetValue<2>(dataIndex), val);
			else
				atomicAdd(&block->template GetValue<3>(dataIndex), val);
		}
}

template <typename Config, typename GridBlock>
__global__ auto MarkActiveGridBlocks(Config, uint32_t blockCount, const MPMGrid<GridBlock> grid, uint32_t* mark) -> void
{
	constexpr auto config = Config{};
	const auto flattenedIndex = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t flattenedBlockIndex = flattenedIndex / config.GetBlockVolume();
	const uint32_t flattenedCellIndex = flattenedIndex % config.GetBlockVolume();

	if (flattenedBlockIndex >= blockCount)
		return;

	const auto block = grid.GetBlock(flattenedBlockIndex);
	bool isActive = block->template GetValue<0>(flattenedCellIndex) > 0.f;
	isActive |= block->template GetValue<0>(config.GetBlockVolume() + flattenedCellIndex) > 0.f;
	if (isActive)
		{
			mark[flattenedBlockIndex] = 1;
		}
}

template <typename Config>
__global__ auto MarkActiveParticleBlocks(Config, uint32_t blockCount, const uint32_t* __restrict__ particleBucketSize,
										 uint32_t* mark) -> void
{
	const auto flattenedParticleBlockIndex = threadIdx.x + blockDim.x * blockIdx.x;
	if (flattenedParticleBlockIndex >= blockCount)
		return;

	if (particleBucketSize[flattenedParticleBlockIndex] > 0)
		{
			mark[flattenedParticleBlockIndex] = 1;
		}
}

template <typename Partition>
__global__ auto UpdatePartition(uint32_t blockCount, const uint32_t* __restrict__ sourceBlockIndex,
								const Partition partition, Partition nextPartition) -> void
{
	const uint32_t flattenedBlockIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (flattenedBlockIndex >= blockCount)
		return;

	const uint32_t flattenedSourceBlockIndex = sourceBlockIndex[flattenedBlockIndex];
	nextPartition.activeKeys_[flattenedBlockIndex] = partition.activeKeys_[flattenedSourceBlockIndex];
	nextPartition.Reinsert(flattenedBlockIndex);
}

template <typename Config, typename ParticleBuffer>
__global__ auto UpdateBucket(Config, uint32_t blockCount, const uint32_t* __restrict__ sourceBlockIndex,
							 const ParticleBuffer buffer, ParticleBuffer nextBuffer) -> void
{
	constexpr auto config = Config{};
	__shared__ uint32_t sourceIndex[1];
	const uint32_t flattenedBlockIndex = blockIdx.x;

	if (flattenedBlockIndex >= blockCount)
		return;
	if (threadIdx.x == 0)
		{
			sourceIndex[0] = sourceBlockIndex[flattenedBlockIndex];
			nextBuffer.particleBucketSize_[flattenedBlockIndex] = buffer.particleBucketSize_[sourceIndex[0]];
		}
	__syncthreads();

	const auto particleCount = nextBuffer.particleBucketSize_[flattenedBlockIndex];
	for (uint32_t particleIndexInBlock = threadIdx.x; particleIndexInBlock < particleCount;
		 particleIndexInBlock += blockDim.x)
		{
			nextBuffer.blockBucket_[flattenedBlockIndex * config.GetMaxParticleCountPerBlock() + particleIndexInBlock] =
				buffer.blockBucket_[sourceIndex[0] * config.GetMaxParticleCountPerBlock() + particleIndexInBlock];
		}
}

template <typename GridBlockIndex, typename Partition, typename Grid>
__global__ auto CopySelectedGridBlocks(const GridBlockIndex* __restrict__ previousGridBlockIndex,
									   const Partition partition, const uint32_t* __restrict__ mark, Grid previousGrid,
									   Grid grid) -> void
{
	typedef typename Grid::GridBlock_ GridBlock;
	const auto blockIndex = previousGridBlockIndex[blockIdx.x];

	if (mark[blockIdx.x])
		{
			const int flattenedBlockIndex = partition.Find(blockIndex);
			if (flattenedBlockIndex == -1)
				return;

			auto destinationBlock = grid.GetBlock(flattenedBlockIndex);
			auto sourceBlock = previousGrid.GetBlock(blockIdx.x);

			meta::ConstexprLoop<0, GridBlock::GridAttribute_::kAttributeCount_>(
				[&](auto attributeIndexWrapper) -> void
				{
					constexpr int attributeIndex = meta::ConstexprLoopIndex(attributeIndexWrapper);
					destinationBlock->template SetValue<attributeIndex>(
						threadIdx.x, sourceBlock->template GetValue<attributeIndex>(threadIdx.x));
				});
		}
}

template <typename Config, typename MPMParticleBuffer>
__global__ auto PrintParticleBucket(const Config, MPMParticleBuffer buffer) -> void
{
	constexpr auto config = Config{};
	for (uint32_t blockIndex = 0; blockIndex < buffer.activeBlockCount_; ++blockIndex)
		{
			const auto particleBucketSize = buffer.particleBucketSize_[blockIndex];
			const auto particleBucketCount = (particleBucketSize + config.GetMaxParticleCountPerBucket() - 1) /
											 config.GetMaxParticleCountPerBucket();

			for (int bucketIndex = buffer.particleBinOffset_[blockIndex];
				 bucketIndex < buffer.particleBinOffset_[blockIndex] + particleBucketCount; ++bucketIndex)
				{
					uint32_t particleCountInBucket =
						std::min(config.GetMaxParticleCountPerBucket(),
								 particleBucketSize - (bucketIndex - buffer.particleBinOffset_[blockIndex]) *
														  config.GetMaxParticleCountPerBucket());

					GpuDebugPrint("In particle bucket %u with particle count: %u:\n", bucketIndex,
								  particleCountInBucket);
					auto bucket = buffer.template GetBucket<config.GetMaxParticleCountPerBucket()>(bucketIndex);
					for (uint32_t pid = 0; pid < particleCountInBucket; ++pid)
						{
							GpuDebugPrint("\tParticle index %u has position: (%f, %f, %f)\n", pid,
										  bucket.GetAttribute<0>(pid), bucket.GetAttribute<1>(pid),
										  bucket.GetAttribute<2>(pid));
						}
				}
		}
}
    template <typename Config, typename Partition, typename ParticleBuffer, typename ParticleArray>
    __global__ auto RetrieveParticleBuffer(Config, Partition partition, Partition previousPartition,
                                           uint32_t* particleCount, ParticleBuffer buffer, ParticleBuffer nextBuffer,
                                           ParticleArray particleArray) -> void
    {
        constexpr auto config = Config{};
        const uint32_t particleCountInBlock = nextBuffer.particleBucketSize_[blockIdx.x];
        const auto blockIndex = partition.activeKeys_[blockIdx.x];
        const auto advectionBucket = nextBuffer.blockBucket_ + blockIdx.x * config.GetMaxParticleCountPerBlock();

        for (uint32_t particleIndexInBlock = threadIdx.x; particleIndexInBlock < particleCountInBlock;
             particleIndexInBlock += blockDim.x)
            {
                const uint32_t advectionMask = advectionBucket[particleIndexInBlock];

                const uint32_t sourceParticleIndexInBlock = advectionMask % config.GetMaxParticleCountPerBlock();

                MPMGridBlockCoordinate sourceBlockIndex;
                {
                    int direction = advectionMask / config.GetMaxParticleCountPerBlock();
                    sourceBlockIndex[0] = (direction / 9) - 1;
                    sourceBlockIndex[1] = ((direction / 3) % 3) - 1;
                    sourceBlockIndex[2] = (direction % 3) - 1;

                    sourceBlockIndex[0] += blockIndex[0];
                    sourceBlockIndex[1] += blockIndex[1];
                    sourceBlockIndex[2] += blockIndex[2];
                }

                const auto flattenedSourceBlockIndex = previousPartition.Find(sourceBlockIndex);
                const auto sourceBucket = buffer.template GetBucket<config.GetMaxParticleCountPerBucket()>(
                    buffer.particleBinOffset_[flattenedSourceBlockIndex] +
                    sourceParticleIndexInBlock / config.GetMaxParticleCountPerBucket());

                const auto particleIndex = atomicAdd(particleCount, 1);

                particleArray[particleIndex][0] =
                    sourceBucket.GetAttribute<0>(sourceParticleIndexInBlock % config.GetMaxParticleCountPerBucket());
                particleArray[particleIndex][1] =
                    sourceBucket.GetAttribute<1>(sourceParticleIndexInBlock % config.GetMaxParticleCountPerBucket());
                particleArray[particleIndex][2] =
                    sourceBucket.GetAttribute<2>(sourceParticleIndexInBlock % config.GetMaxParticleCountPerBucket());
            }
    }

    template <typename Config, typename Partition, typename Grid>
    __global__ auto PrintGridInformation(Config, uint32_t blockCount, Grid grid) -> void
    {
        for (int i = 0; i < blockCount; ++i)
            {
                auto gridBlock = grid.GetBlock(i);

                for (int j = 0; j < 128; ++j)
                    {
                        int gridIndex = j >= 64;
                        float mass = gridBlock->template GetValue<0>(j);
                        float vx = gridBlock->template GetValue<1>(j);
                        float vy = gridBlock->template GetValue<2>(j);
                        float vz = gridBlock->template GetValue<3>(j);

                        if (mass > 0 || vx != 0 || vy != 0 || vz != 0)
                            {
                                printf("gridIndeX: %d, mass: %.8f, velocity: (%f, %f, %f)\n", gridIndex, mass, vx, vy, vz);
                            }
                    }
            }
    }
}
