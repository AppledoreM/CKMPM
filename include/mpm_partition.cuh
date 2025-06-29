#pragma once
#include <limits>
#include "base.h"
#include "mpm_debug.cuh"
#include "mpm_domain.h"
#include "mpm_grid.cuh"
#include "mpm_meta.h"
#include "mpm_utility.h"

#include "spdlog/spdlog.h"

namespace mpm
{

template <MPMDomainType Domain, typename Hashtable = meta::Empty>
struct MPMGridConfig
{
	typedef Domain Domain_;
	typedef Hashtable Hashtable_;
};

namespace internal
{
template <typename T>
concept MPMInfiniteGridConfigType =
	MPMInfiniteDomainType<typename T::Domain_> && !std::is_same_v<typename T::Hashtable_, meta::Empty>;

template <typename T>
concept MPMFiniteGridConfigType =
	MPMFiniteDomainType<typename T::Domain_> && std::is_same_v<typename T::Hashtable_, meta::Empty>;
}  // namespace internal

template <typename GridConfig>
class MPMPartition;

template <internal::MPMFiniteGridConfigType GridConfig>
class MPMPartition<GridConfig> : public IMPMDebugBase<MPMPartition<GridConfig>>
{

   public:
	friend class IMPMDebugBase<MPMPartition<GridConfig>>;
	typedef GridConfig GridConfig_;
	typedef MPMGridBlockCoordinate Key_;
	typedef uint32_t Value_;
	constexpr static uint32_t kValueSentinelValue_ = std::numeric_limits<uint32_t>::max();

	MPMPartition() = default;

	MPM_HOST_DEV_FUNC MPMPartition(const MPMPartition& rhs)
		: capacity_(rhs.capacity_), count_(rhs.count_), activeKeys_(rhs.activeKeys_), table_(rhs.table_)
	{
	}

	template <typename Allocator>
	inline auto Allocate(Allocator allocator, uint32_t capacity) -> void
	{
		capacity_ = capacity;
		allocator.Allocate(count_, sizeof(uint32_t));
		allocator.Allocate(activeKeys_, sizeof(Key_) * capacity_);
		allocator.Allocate(table_, sizeof(Value_) * GridConfig::Domain_::kGridBlockCount_);
	}

	template <typename Allocator>
	inline auto Resize(Allocator allocator, uint32_t capacity) -> void
	{
		capacity_ = capacity;
		allocator.Deallocate(activeKeys_);
		allocator.Allocate(activeKeys_, sizeof(Key_) * capacity);
	}

	template <typename Allocator>
	inline auto Deallocate(Allocator allocator) -> void
	{
		allocator.Deallocate(count_);
		allocator.Deallocate(activeKeys_);
		allocator.Deallocate(table_);
		capacity_ = 0;
		count_ = nullptr;
		activeKeys_ = nullptr;
		table_ = nullptr;
	}

	inline auto Copy(const MPMPartition& rhs, cudaStream_t stream = cudaStreamDefault) -> void
	{
		assert(capacity_ == rhs.capacity_);
		cudaMemcpyAsync(count_, rhs.count_, sizeof(uint32_t), cudaMemcpyDefault, stream);
		cudaMemcpyAsync(activeKeys_, rhs.activeKeys_, sizeof(Key_) * capacity_, cudaMemcpyDefault, stream);
		cudaMemcpyAsync(table_, rhs.table_, sizeof(Value_) * GridConfig::Domain_::kGridBlockCount_, cudaMemcpyDefault,
						stream);
	}

	inline auto Reset(cudaStream_t stream) -> void
	{
		cudaMemsetAsync(table_, 0xff, sizeof(Value_) * GridConfig::Domain_::kGridBlockCount_, stream);
		cudaMemsetAsync(count_, 0, sizeof(uint32_t), stream);
	}

	inline auto ResetPartitionTable(cudaStream_t stream) -> void
	{
		cudaMemsetAsync(table_, 0xff, sizeof(Value_) * GridConfig::Domain_::kGridBlockCount_, stream);
	}

	constexpr MPM_FORCE_INLINE MPM_DEV_FUNC auto Find(const Key_& key) const -> Value_ { return Index(key); }

	constexpr MPM_FORCE_INLINE MPM_DEV_FUNC auto Insert(const Key_& key) -> Value_
	{
		Value_ tag = atomicCAS(&Index(key), kValueSentinelValue_, 0);
		if (tag == kValueSentinelValue_)
			{
				auto newGridBlockIndex = atomicAdd(count_, 1);
				activeKeys_[newGridBlockIndex] = key;
				Index(key) = newGridBlockIndex;
				return newGridBlockIndex;
			}

		return kValueSentinelValue_;
	}

	constexpr MPM_FORCE_INLINE MPM_DEV_FUNC auto Reinsert(Value_ newGridBlockIndex) -> void
	{
		Index(activeKeys_[newGridBlockIndex]) = newGridBlockIndex;
	}

   protected:
	template <typename Config>
	auto PrintDebugInformationImpl(const Config&) const -> void
	{
		constexpr auto config = Config{};

		uint32_t blockCount = 0;
		{
			cudaMemcpy(&blockCount, count_, sizeof(uint32_t), cudaMemcpyDeviceToHost);
			cudaCheckError(cudaDeviceSynchronize());
		}

		std::vector<Key_> activeKeysCpu;

		{
			activeKeysCpu.resize(blockCount);
			cudaMemcpy(activeKeysCpu.data(), activeKeys_, sizeof(Key_) * blockCount, cudaMemcpyDeviceToHost);
			cudaCheckError(cudaDeviceSynchronize());
		}

		spdlog::debug("----------------------Partition Debug Information----------------------");
		spdlog::debug("\tTotal Used Block Count: {}", blockCount);
		for (uint32_t i = 0; i < blockCount; ++i)
			{
				auto& key = activeKeysCpu[i];
				spdlog::debug("\t\tBlock ({}, {}, {}) is flattened to {}.", key[0], key[1], key[2], i);
			}
	}

   private:
	constexpr MPM_FORCE_INLINE MPM_DEV_FUNC auto Index(const Key_& key) const -> Value_
	{
		using DomainRange = typename GridConfig_::Domain_::DomainRange_;
		size_t index = GetFlattenedIndex<DomainRange::kDim_[1], DomainRange::kDim_[2]>(key[0], key[1], key[2]);
		return table_[index];
	}

	constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto Index(const Key_& key) -> Value_&
	{
		using DomainRange = typename GridConfig_::Domain_::DomainRange_;
		size_t index = GetFlattenedIndex<DomainRange::kDim_[1], DomainRange::kDim_[2]>(key[0], key[1], key[2]);
		return table_[index];
	}

	uint32_t capacity_ = 0;
	Value_* table_ = nullptr;

   public:
	uint32_t* count_ = nullptr;
	Key_* activeKeys_ = nullptr;
};

template <internal::MPMInfiniteGridConfigType GridConfig>
class MPMPartition<GridConfig>
{
};

}  // namespace mpm
