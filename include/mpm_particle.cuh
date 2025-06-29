#pragma once
#include "cuda_util.cuh"
#include "mpm_debug.cuh"
#include "mpm_grid.cuh"
#include "mpm_meta.h"
#include "mpm_partition.cuh"
#include "mpm_utility.h"

#include "spdlog/spdlog.h"

#include <assert.h>

namespace mpm
{

template <uint32_t kParicleMaterialTag, typename... ParticleAttribute>
struct MPMParticle
{
	using Attribute = meta::AttributeWrapper<ParticleAttribute...>;

	template <size_t kIndex>
	auto GetAttribute() const -> typename Attribute::template AttributeWrapperAccessor<kIndex>::Type
	{
		return *reinterpret_cast<typename Attribute::template AttributeWrapperAccessor<kIndex>::Type*>(
			data_ + Attribute::template AttributeWrapperAccessor<kIndex>::kOffset_);
	}

	template <size_t kIndex>
	auto GetAttribute() -> typename Attribute::template AttributeWrapperAccessor<kIndex>::Type&
	{
		return *reinterpret_cast<typename Attribute::template AttributeWrapperAccessor<kIndex>::Type*>(
			data_ + Attribute::template AttributeWrapperAccessor<kIndex>::kOffset_);
	}

	uint8_t data_[Attribute::kSize_];
};

template <typename Particle>
class MPMParticleBuffer
{
   public:
	typedef Particle Particle_;
	typedef uint32_t ParticleIndex;

	using Attribute = Particle::Attribute;

	template <size_t kBucketCapacity>
	struct MPMParticleBucket
	{
		MPM_FORCE_INLINE MPM_HOST_DEV_FUNC MPMParticleBucket(size_t bucketIndex, uint8_t* particleBuffer)
			: accessor_(particleBuffer + kBucketCapacity * Attribute::kSize_ * bucketIndex)
		{
		}

		template <size_t AttributeIndex>
		MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetAttribute(size_t particleIndexInBucket) const ->
			typename Attribute::template AttributeWrapperAccessor<AttributeIndex>::Type
		{
			assert(particleIndexInBucket < kBucketCapacity);
			constexpr size_t kAcessorAttributeOffset =
				kBucketCapacity * Attribute::template AttributeWrapperAccessor<AttributeIndex>::kOffset_;
			constexpr size_t kAttributeSize = Attribute::template AttributeWrapperAccessor<AttributeIndex>::kSize_;
			return *reinterpret_cast<typename Attribute::template AttributeWrapperAccessor<AttributeIndex>::Type*>(
				accessor_ + kAcessorAttributeOffset + particleIndexInBucket * kAttributeSize);
		}

		template <size_t AttributeIndex>
		MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetAttribute(size_t particleIndexInBucket) ->
			typename Attribute::template AttributeWrapperAccessor<AttributeIndex>::Type&
		{
			assert(particleIndexInBucket < kBucketCapacity);
			constexpr size_t kAcessorAttributeOffset =
				kBucketCapacity * Attribute::template AttributeWrapperAccessor<AttributeIndex>::kOffset_;
			constexpr size_t kAttributeSize = Attribute::template AttributeWrapperAccessor<AttributeIndex>::kSize_;
			return *reinterpret_cast<typename Attribute::template AttributeWrapperAccessor<AttributeIndex>::Type*>(
				accessor_ + kAcessorAttributeOffset + particleIndexInBucket * kAttributeSize);
		}

		template <size_t AttributeIndex>
		MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto SetAttribute(
			size_t particleIndexInBucket,
			typename Attribute::template AttributeWrapperAccessor<AttributeIndex>::Type value) -> void
		{
			assert(particleIndexInBucket < kBucketCapacity);
			constexpr size_t kAcessorAttributeOffset =
				kBucketCapacity * Attribute::template AttributeWrapperAccessor<AttributeIndex>::kOffset_;
			constexpr size_t kAttributeSize = Attribute::template AttributeWrapperAccessor<AttributeIndex>::kSize_;
			*reinterpret_cast<typename Attribute::template AttributeWrapperAccessor<AttributeIndex>::Type*>(
				accessor_ + kAcessorAttributeOffset + particleIndexInBucket * kAttributeSize) = value;
		}

		uint8_t* const accessor_ = nullptr;
	};

   protected:
	template <typename Config>
	inline auto PrintDebugInformationImpl(const Config&) const -> void
	{
		constexpr auto config = Config{};

		spdlog::debug("----------------------Particle Debug Information----------------------");
		spdlog::debug(
			"\tParticle Buffer Capacity: {}\n"
			"\tActive Block Count: {}\n"
			"\tParticle Mass: {}\n"
			"\tParticle Volume: {}\n",
			particleBufferCapacity_, activeBlockCount_, particleMass_, particleVolume_);

		std::vector<uint32_t> blockBucketCpu, cellBucketCpu, cellParticleCountCpu;
		std::vector<uint32_t> particleBinOffsetCpu, particleBucketSizeCpu;
		std::vector<uint8_t> particleBufferCpu;
		// Copy data to cpu
		{
			blockBucketCpu.resize(config.GetMaxParticleCountPerBlock() * activeBlockCount_);
			cellBucketCpu.resize(config.GetMaxParticleCountPerBlock() * activeBlockCount_);
			cellParticleCountCpu.resize(config.GetBlockVolume() * activeBlockCount_);
			particleBinOffsetCpu.resize(activeBlockCount_);
			particleBucketSizeCpu.resize(activeBlockCount_);
			particleBufferCpu.resize(Attribute::kSize_ * particleBufferCapacity_);

			cudaMemcpy(blockBucketCpu.data(), blockBucket_, blockBucketCpu.size() * sizeof(uint32_t),
					   cudaMemcpyDeviceToHost);
			cudaMemcpy(cellBucketCpu.data(), cellBucket_, cellBucketCpu.size() * sizeof(uint32_t),
					   cudaMemcpyDeviceToHost);
			cudaMemcpy(cellParticleCountCpu.data(), cellParticleCount_, cellParticleCountCpu.size() * sizeof(uint32_t),
					   cudaMemcpyDeviceToHost);
			cudaMemcpy(particleBinOffsetCpu.data(), particleBinOffset_, particleBinOffsetCpu.size() * sizeof(uint32_t),
					   cudaMemcpyDeviceToHost);
			cudaMemcpy(particleBucketSizeCpu.data(), particleBucketSize_,
					   particleBucketSizeCpu.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
			// type of uint8_t
			cudaMemcpy(particleBufferCpu.data(), particleBuffer_, particleBufferCpu.size(), cudaMemcpyDeviceToHost);

			cudaCheckError(cudaDeviceSynchronize());
		}

		// Print debug information
		for (int blockIndex = 0; blockIndex < activeBlockCount_; ++blockIndex)
			{
				const int bucketSize = particleBucketSizeCpu[blockIndex];
				if (bucketSize)
					{
						spdlog::debug("\tBlock with index {} contains {} particles.", blockIndex, bucketSize);
						const int bucketCount =
							NextNearestMultipleOf<config.GetMaxParticleCountPerBucket()>(bucketSize) /
							config.GetMaxParticleCountPerBucket();
						for (int i = 0; i < bucketCount; ++i)
							{

								const int bucketIndex = particleBinOffsetCpu[blockIndex] + i;
								const int bucketStart =
									bucketIndex * config.GetMaxParticleCountPerBucket() * Attribute::kSize_;
								const int particleCount =
									std::min(static_cast<int>(config.GetMaxParticleCountPerBucket()),
											 bucketSize - i * static_cast<int>(config.GetMaxParticleCountPerBucket()));
								spdlog::debug("\t\tBucket with index {} contains {} particles.", bucketIndex,
											  particleCount);

								for (int particleIndex = 0; particleIndex < particleCount; ++particleIndex)
									{
										// Read position
										float position[3] = {};
										position[0] = *reinterpret_cast<float*>(
											&particleBufferCpu[bucketStart + particleIndex * 4]);
										position[1] = *reinterpret_cast<float*>(
											&particleBufferCpu[bucketStart + particleIndex * 4 +
															   4 * config.GetMaxParticleCountPerBucket()]);
										position[2] = *reinterpret_cast<float*>(
											&particleBufferCpu[bucketStart + particleIndex * 4 +
															   8 * config.GetMaxParticleCountPerBucket()]);

										int block[3] = {};
										block[0] = (static_cast<int>(position[0] * config.GetInvDx() + 0.25f) - 2) / 4;
										block[1] = (static_cast<int>(position[1] * config.GetInvDx() + 0.25f) - 2) / 4;
										block[2] = (static_cast<int>(position[2] * config.GetInvDx() + 0.25f) - 2) / 4;

										spdlog::debug(
											"\t\t\tParticle position: ({}, {}, {}) belongs to block ({}, {}, {})",
											position[0], position[1], position[2], block[0], block[1], block[2]);
									}
							}
					}
				bool doneOnce = false;
				for (int i = 0; i < config.GetBlockVolume(); ++i)
					{
						const int cellParticleCount = cellParticleCountCpu[blockIndex * config.GetBlockVolume() + i];
						if (cellParticleCount)
							{
								if (!doneOnce)
									{
										doneOnce = true;
										spdlog::debug("\tIn block: {}", blockIndex);
									}
								spdlog::debug("\t\tCell {} contains {} particles.", i, cellParticleCount);
							}
					}
				if (doneOnce)
					spdlog::debug("--------");
			}
	}

   public:
	constexpr MPMParticleBuffer() = default;

	constexpr MPMParticleBuffer(const MPMParticleBuffer& rhs) = default;

	/* MPM_HOST_DEV_FUNC MPMParticleBuffer(const MPMParticleBuffer& rhs) */
	/* 	: particleBufferCapacity_(rhs.particleBufferCapacity_), */
	/* 	  particleBuffer_(rhs.particleBuffer_), */
	/* 	  activeBlockCount_(rhs.activeBlockCount_), */
	/* 	  cellParticleCount_(rhs.cellParticleCount_), */
	/* 	  cellBucket_(rhs.cellBucket_), */
	/* 	  blockBucket_(rhs.blockBucket_), */
	/* 	  particleBinOffset_(rhs.particleBinOffset_), */
	/* 	  particleBucketSize_(rhs.particleBucketSize_), */
	/* 	  particleMass_(rhs.particleMass_), */
	/* 	  particleVolume_(rhs.particleVolume_) */
	/* { */
	/* } */

	inline auto SetParticleMass(float mass) -> void { particleMass_ = mass; }
	MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetParticleMass() const -> float { return particleMass_; }

	inline auto SetParticleVolume(float volume) -> void
	{
		particleVolume_ = volume;
		;
	}
	MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetParticleVolume() const -> float { return particleVolume_; }

	template <typename Config>
	auto Copy(const Config&, const MPMParticleBuffer& rhs, uint32_t partitionBlockCount,
			  cudaStream_t stream = cudaStreamDefault) -> void
	{
		constexpr auto config = Config{};

		particleMass_ = rhs.particleMass_;
		particleVolume_ = rhs.particleVolume_;

		activeBlockCount_ = rhs.activeBlockCount_;
		particleBufferCapacity_ = rhs.particleBufferCapacity_;
		cudaMemcpyAsync(reinterpret_cast<void*>(blockBucket_), reinterpret_cast<void*>(rhs.blockBucket_),
						sizeof(ParticleIndex) * partitionBlockCount * config.GetMaxParticleCountPerBlock(),
						cudaMemcpyDefault, stream);
		cudaMemcpyAsync(
			reinterpret_cast<void*>(cellBucket_), reinterpret_cast<void*>(rhs.cellBucket_),
			sizeof(ParticleIndex) * partitionBlockCount * config.GetBlockVolume() * config.GetMaxParticleCountPerCell(),
			cudaMemcpyDefault, stream);
		cudaMemcpyAsync(reinterpret_cast<void*>(cellParticleCount_), reinterpret_cast<void*>(rhs.cellParticleCount_),
						sizeof(ParticleIndex) * partitionBlockCount * config.GetBlockVolume(), cudaMemcpyDefault,
						stream);
		cudaMemcpyAsync(reinterpret_cast<void*>(particleBinOffset_), reinterpret_cast<void*>(rhs.particleBinOffset_),
						sizeof(ParticleIndex) * partitionBlockCount, cudaMemcpyDefault, stream);
		cudaMemcpyAsync(reinterpret_cast<void*>(particleBucketSize_), reinterpret_cast<void*>(rhs.particleBucketSize_),
						sizeof(ParticleIndex) * partitionBlockCount, cudaMemcpyDefault, stream);
		cudaMemcpyAsync(reinterpret_cast<void*>(particleBuffer_), reinterpret_cast<void*>(rhs.particleBuffer_),
						Attribute::kSize_ * particleBufferCapacity_, cudaMemcpyDefault, stream);
	}

	template <size_t kBucketCapacity>
	MPM_HOST_DEV_FUNC MPM_FORCE_INLINE auto GetBucket(uint32_t bucketIndex) -> MPMParticleBucket<kBucketCapacity>
	{
		return {bucketIndex, particleBuffer_};
	}

	template <size_t kBucketCapacity>
	MPM_HOST_DEV_FUNC MPM_FORCE_INLINE auto GetBucket(uint32_t bucketIndex) const
		-> const MPMParticleBucket<kBucketCapacity>
	{
		return {bucketIndex, particleBuffer_};
	}

	template <typename Allocator>
	auto AllocateParticleBuffer(Allocator allocator, size_t particleBufferCapacity) -> void
	{
		particleBufferCapacity_ = particleBufferCapacity;
		allocator.Allocate(particleBuffer_, Attribute::kSize_ * particleBufferCapacity_);
	}

	template <typename Allocator>
	auto DeallocateParticleBuffer(Allocator allocator) -> void
	{
		allocator.Deallocate(particleBuffer_, Attribute::kSize_ * particleBufferCapacity_);
	}

	template <typename Allocator>
	auto ResizeParticleBuffer(Allocator allocator, size_t particleBufferCapacity) -> void
	{
		DeallocateParticleBuffer(allocator);
		AllocateParticleBuffer(allocator, particleBufferCapacity);
	}

	template <typename Allocator, typename Config>
	auto DeallocateBucket(Allocator allocator, const Config& config) -> void
	{
		if (activeBlockCount_)
			{
				allocator.Deallocate(particleBinOffset_, activeBlockCount_ * sizeof(ParticleIndex));
				allocator.Deallocate(cellBucket_,
									 activeBlockCount_ * sizeof(ParticleIndex) * config.GetMaxParticleCountPerBlock());
				allocator.Deallocate(cellParticleCount_,
									 activeBlockCount_ * sizeof(ParticleIndex) * config.GetBlockVolume());
				allocator.Deallocate(blockBucket_,
									 activeBlockCount_ * sizeof(ParticleIndex) * config.GetMaxParticleCountPerBlock());
				allocator.Deallocate(particleBucketSize_, activeBlockCount_ * sizeof(ParticleIndex));
				activeBlockCount_ = 0;
			}
	}

	template <typename Allocator, typename Config>
	auto ReserveBucket(Allocator allocator, const Config& config, size_t blockCount) -> void
	{

		activeBlockCount_ = blockCount;
		allocator.Allocate(particleBinOffset_, activeBlockCount_ * sizeof(ParticleIndex));
		allocator.Allocate(cellBucket_,
						   activeBlockCount_ * sizeof(ParticleIndex) * config.GetMaxParticleCountPerBlock());
		allocator.Allocate(cellParticleCount_, activeBlockCount_ * sizeof(ParticleIndex) * config.GetBlockVolume());
		allocator.Allocate(blockBucket_,
						   activeBlockCount_ * sizeof(ParticleIndex) * config.GetMaxParticleCountPerBlock());
		allocator.Allocate(particleBucketSize_, activeBlockCount_ * sizeof(ParticleIndex));
		ResetParticleCountPerCell(config);
	}

	template <typename Config>
	auto ResetParticleCountPerCell(const Config& config) -> void
	{
		cudaMemset(cellParticleCount_, 0, activeBlockCount_ * sizeof(ParticleIndex) * config.GetBlockVolume());
	}

	template <typename Config, typename Partition, typename CellIndex>
	requires MPMFiniteDomainType<typename Partition::GridConfig_::Domain_> MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto
	AddAdvection(const Config&, const Partition& partition, const CellIndex& cellIndex, int directionTag,
				 uint32_t particleIdInBlock) -> void
	{
		constexpr auto config = Config{};
		const MPMGridBlockCoordinate blockCoord =
			MPMGridBlockCoordinate{(cellIndex[0] - 2) / static_cast<int>(config.GetBlockSize()),
								   (cellIndex[1] - 2) / static_cast<int>(config.GetBlockSize()),
								   (cellIndex[2] - 2) / static_cast<int>(config.GetBlockSize())};
		const auto blockIndex = partition.Find(blockCoord);

		if (blockIndex == Partition::kValueSentinelValue_)
			{
				printf("When adding advection, the block has not existed.\n");

				return;
			}

		const int flattenedCellIndex = GetFlattenedIndex<config.GetBlockSize(), config.GetBlockSize()>(
			static_cast<int>(cellIndex[0] - 2) % static_cast<int>(config.GetBlockSize()),
			static_cast<int>(cellIndex[1] - 2) % static_cast<int>(config.GetBlockSize()),
			static_cast<int>(cellIndex[2] - 2) % static_cast<int>(config.GetBlockSize()));
		const int particleIndexInCell =
			atomicAdd(cellParticleCount_ + blockIndex * config.GetBlockVolume() + flattenedCellIndex, 1);

		if (particleIndexInCell >= config.GetMaxParticleCountPerCell())
			{
				atomicSub(cellParticleCount_ + blockIndex * config.GetBlockVolume() + flattenedCellIndex, 1);
				/*
				printf(
					"In AddAdvection, particle count per cell reached maximum. Reducing and "
					"exiting!\n");
					*/
				return;
			}

		cellBucket_[blockIndex * config.GetMaxParticleCountPerBlock() +
					flattenedCellIndex * config.GetMaxParticleCountPerCell() + particleIndexInCell] =
			(directionTag * config.GetMaxParticleCountPerBlock()) + particleIdInBlock;
	}

   public:
	// AOS-Data
	size_t particleBufferCapacity_ = 0;
	uint8_t* particleBuffer_ = nullptr;

	// Indexer
	size_t activeBlockCount_ = 0;
	uint32_t* cellBucket_ = nullptr;
	uint32_t* cellParticleCount_ = nullptr;
	uint32_t* blockBucket_ = nullptr;
	uint32_t* particleBucketSize_ = nullptr;
	uint32_t* particleBinOffset_ = nullptr;
	float particleMass_ = 0.0;
	float particleVolume_ = 0.0;
};

}  // namespace mpm
