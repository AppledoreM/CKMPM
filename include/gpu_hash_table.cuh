#pragma once
#include "base.h"
#include "cuda_util.cuh"
#include "mpm_utility.h"

#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <cuda/atomic>

#include <cooperative_groups.h>
#include <random>

namespace mpm
{
namespace hashtable
{
namespace internal
{
template <typename T>
struct XorshiftRngDevice
{
};

template <>
struct XorshiftRngDevice<uint32_t>
{
	MPM_HOST_DEV_FUNC XorshiftRngDevice() : y_(2463534242) {}

	MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto operator()() -> uint32_t
	{
		y_ ^= (y_ << 13);
		y_ = (y_ >> 17);
		return (y_ ^= (y_ << 5));
	}

	uint32_t y_;
};

template <typename Key>
struct MurmurHash3_32
{
	typedef Key Key_;
	typedef uint32_t Value_;
	using result_type = uint32_t;

	MPM_HOST_DEV_FUNC constexpr MurmurHash3_32() : seed_(0) {}
	MPM_HOST_DEV_FUNC constexpr MurmurHash3_32(uint32_t seed) : seed_(seed) {}

	MurmurHash3_32(const MurmurHash3_32&) = default;
	MurmurHash3_32(MurmurHash3_32&&) = default;
	MurmurHash3_32& operator=(MurmurHash3_32 const&) = default;
	MurmurHash3_32& operator=(MurmurHash3_32&&) = default;
	~MurmurHash3_32() = default;

	constexpr auto MPM_HOST_DEV_FUNC operator()(Key const& key) const noexcept -> uint32_t
	{
		constexpr int len = sizeof(Key);
		const uint8_t* const data = (const uint8_t*)&key;
		constexpr int nblocks = len / 4;

		uint32_t h1 = seed_;
		constexpr uint32_t c1 = 0xcc9e2d51;
		constexpr uint32_t c2 = 0x1b873593;
		//----------
		// body
		const uint32_t* const blocks = (const uint32_t*)(data + nblocks * 4);
		for (int i = -nblocks; i; i++)
			{
				uint32_t k1 = blocks[i];  // getblock32(blocks,i);
				k1 *= c1;
				k1 = Rotl32(k1, 15);
				k1 *= c2;
				h1 ^= k1;
				h1 = Rotl32(h1, 13);
				h1 = h1 * 5 + 0xe6546b64;
			}
		//----------
		// tail
		const uint8_t* tail = (const uint8_t*)(data + nblocks * 4);
		uint32_t k1 = 0;
		switch (len & 3)
			{
				case 3:
					k1 ^= tail[2] << 16;
				case 2:
					k1 ^= tail[1] << 8;
				case 1:
					k1 ^= tail[0];
					k1 *= c1;
					k1 = Rotl32(k1, 15);
					k1 *= c2;
					h1 ^= k1;
			};
		//----------
		// finalization
		h1 ^= len;
		h1 = Fmix32(h1);
		return h1;
	}

   private:
	constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto Rotl32(uint32_t x, int8_t r) const noexcept -> uint32_t
	{
		return (x << r) | (x >> (32 - r));
	}

	constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto Fmix32(uint32_t h) const noexcept -> uint32_t
	{
		h ^= h >> 16;
		h *= 0x85ebca6b;
		h ^= h >> 13;
		h *= 0xc2b2ae35;
		h ^= h >> 16;
		return h;
	}

	uint32_t seed_;
};

}  // namespace internal

namespace internal
{
struct GpuHashtableKeyEqualOperatorTag
{
};

struct GpuHashtableValueEqualOperatorTag
{
};

struct GpuHashtableDefaultKeyEqualOperator : GpuHashtableKeyEqualOperatorTag
{
	template <typename T>
	MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto operator()(const T& lhs, const T& rhs) const -> bool
	{
		return lhs == rhs;
	}
};

struct GpuHashtableDefaultValueEqualOperator : GpuHashtableValueEqualOperatorTag
{
	template <typename T>
	MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto operator()(const T& lhs, const T& rhs) const -> bool
	{
		return lhs == rhs;
	}
};

template <typename T, typename EqualOperator>
concept Equable = requires(T lhs, T rhs, EqualOperator op)
{
	{
		op(lhs, rhs)
	} -> std::same_as<bool>;
};

template <typename T>
concept GpuHashtableKeyEqualOperatorType = std::is_base_of_v<GpuHashtableKeyEqualOperatorTag, T>;

template <typename T>
concept GpuHashtableValueEqualOperatorType = std::is_base_of_v<GpuHashtableValueEqualOperatorTag, T>;

struct GpuHashtableKeySentinelTag
{
};

struct GpuHashtableValueSentinelTag
{
};

template <typename T>
struct GpuHashtableDefaultKeySentinel : GpuHashtableKeySentinelTag
{
	constexpr static T kKeySentinelValue_ = std::numeric_limits<T>::max();
};

template <typename T>
struct GpuHashtableDefaultValueSentinel : GpuHashtableValueSentinelTag
{
	constexpr static T kValueSentinelValue_ = std::numeric_limits<T>::max();
};

template <typename T>
concept GpuHashtableKeySentinelType = std::is_base_of_v<GpuHashtableKeySentinelTag, T>;

template <typename T>
concept GpuHashtableValueSentinelType = std::is_base_of_v<GpuHashtableValueSentinelTag, T>;

template <typename T, typename GpuHashtableKeySentinel>
concept GpuHashtableHasKeySentinelValue =
	std::is_same_v<std::remove_cv_t<decltype(GpuHashtableKeySentinel::kKeySentinelValue_)>, T>;

template <typename T, typename GpuHashtableValueSentinel>
concept GpuHashtableHasValueSentinelValue =
	std::is_same_v<std::remove_cv_t<decltype(GpuHashtableValueSentinel::kValueSentinelValue_)>, T>;
}  // namespace internal

template <typename Key, typename Value,
		  internal::GpuHashtableKeyEqualOperatorType GpuHashtableKeyEqualOperator =
			  internal::GpuHashtableDefaultKeyEqualOperator,
		  internal::GpuHashtableValueEqualOperatorType GpuHashtableValueEqualOperator =
			  internal::GpuHashtableDefaultValueEqualOperator,
		  internal::GpuHashtableKeySentinelType GpuHashtableKeySentinel = internal::GpuHashtableDefaultKeySentinel<Key>,
		  internal::GpuHashtableValueSentinelType GpuHashtableValueSentinel =
			  internal::GpuHashtableDefaultValueSentinel<Value>>
requires internal::GpuHashtableHasKeySentinelValue<Key, GpuHashtableKeySentinel>&&
	internal::GpuHashtableHasValueSentinelValue<Value, GpuHashtableValueSentinel>&&
		internal::Equable<Key, GpuHashtableKeyEqualOperator>&& internal::Equable<Value, GpuHashtableValueEqualOperator>
	// TODO: fix alignment
	struct alignas(sizeof(Key) + sizeof(Value)) GpuHashtablePair
{
	typedef Key Key_;
	typedef Value Value_;
	typedef GpuHashtableKeyEqualOperator GpuHashtableKeyEqualOperator_;
	typedef GpuHashtableValueEqualOperator GpuHashtableValueEqualOperator_;

	constexpr static auto kGpuHashtableSentinelKey_ = GpuHashtableKeySentinel::kKeySentinelValue_;
	constexpr static auto kGpuHashtableSentinelValue_ = GpuHashtableValueSentinel::kValueSentinelValue_;

	constexpr MPM_HOST_DEV_FUNC GpuHashtablePair()
		: key_(kGpuHashtableSentinelKey_), value_(kGpuHashtableSentinelValue_)
	{
	}

	constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC GpuHashtablePair(const Key_& key, const Value_& value)
		: key_(key), value_(value)
	{
	}

	constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto operator==(const GpuHashtablePair& rhs) const -> bool
	{
		GpuHashtableKeyEqualOperator_ opKey_{};
		GpuHashtableValueEqualOperator_ opValue_{};
		return opKey_(key_, rhs.key_) && opValue_(value_, rhs.value_);
	}

	Key key_;
	Value value_;
};

namespace internal
{
template <typename GpuHashtablePairType>
__device__ constexpr auto kGpuHashtableSentinelPair = GpuHashtablePairType();

template <typename Pair>
__global__ auto Reset(size_t capacity, auto* table, auto* activeKeys) -> void
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < capacity; i += gridDim.x * blockDim.x)
		{
			table[i] = internal::kGpuHashtableSentinelPair<Pair>;
			activeKeys[i] = Pair::kGpuHashtableSentinelKey_;
		}
}

__global__ auto CopyTable(size_t capacity, auto* table, const auto* rhsTable) -> void
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < capacity; i += gridDim.x * blockDim.x)
		{
			table[i] = rhsTable[i].load();
		}
}

__global__ auto LoadTableToBuffer(size_t capacity, auto* table, auto* buffer) -> void
{
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < capacity; i += gridDim.x * blockDim.x)
		{
			buffer[i] = table[i].load();
		}
}
}  // namespace internal

template <typename Pair, size_t BucketSize, typename GpuHashFunction>
class BucketedCuckooHashTable
{
   public:
	constexpr static size_t kGpuHashFunctionsCount_ = 3;
	constexpr static size_t kBucketSize_ = BucketSize;
	typedef Pair Pair_;
	typedef cuda::atomic<Pair_, cuda::thread_scope_device> AtomicPair_;
	typedef typename Pair_::Key_ Key_;
	typedef typename Pair_::Value_ Value_;
	typedef typename Pair_::GpuHashtableKeyEqualOperator_ GpuHashtableKeyEqualOperator_;
	typedef typename Pair_::GpuHashtableValueEqualOperator_ GpuHashtableValueEqualOperator_;

	Key_* activeKeys_ = nullptr;
	int* indices_ = nullptr;

   private:
	AtomicPair_* table_ = nullptr;
	size_t bucketCount_ = 0;
	size_t capacity_ = 0;
	uint32_t* activeCount_ = nullptr;

	GpuHashFunction hashFunction0_;
	GpuHashFunction hashFunction1_;
	GpuHashFunction hashFunction2_;

	uint32_t maxCuckooChainLength_ = 0;

   public:
	BucketedCuckooHashTable()
	{
		std::mt19937 rng(2);
		hashFunction0_ = GpuHashFunction{static_cast<uint32_t>(rng())};
		hashFunction1_ = GpuHashFunction{static_cast<uint32_t>(rng())};
		hashFunction2_ = GpuHashFunction{static_cast<uint32_t>(rng())};
	}

	MPM_FORCE_INLINE MPM_HOST_DEV_FUNC BucketedCuckooHashTable(const BucketedCuckooHashTable& rhs)
		: activeKeys_(rhs.activeKeys_),
		  capacity_(rhs.capacity_),
		  bucketCount_(rhs.bucketCount_),
		  maxCuckooChainLength_(rhs.maxCuckooChainLength_),
		  table_(rhs.table_),
		  hashFunction0_(rhs.hashFunction0_),
		  hashFunction1_(rhs.hashFunction1_),
		  hashFunction2_(rhs.hashFunction2_),
		  indices_(rhs.indices_),
		  activeCount_(rhs.activeCount_)
	{
	}

	~BucketedCuckooHashTable() = default;

	auto Copy(const BucketedCuckooHashTable& rhs) -> void
	{
		cudaMemcpy(activeKeys_, rhs.activeKeys_, sizeof(Key_) * capacity_, cudaMemcpyDefault);
		cudaMemcpy(indices_, rhs.indices_, sizeof(int) * capacity_, cudaMemcpyDefault);
		cudaMemcpy(activeCount_, rhs.activeCount_, sizeof(float), cudaMemcpyDefault);
		internal::CopyTable<<<2048, 256>>>(capacity_, table_, rhs.table_);
		cudaCheckError(cudaDeviceSynchronize());

		bucketCount_ = rhs.bucketCount_;
		capacity_ = rhs.capacity_;
		maxCuckooChainLength_ = rhs.maxCuckooChainLength_;
		hashFunction0_ = rhs.hashFunction0_;
		hashFunction1_ = rhs.hashFunction1_;
		hashFunction2_ = rhs.hashFunction2_;
	}

	MPM_FORCE_INLINE auto Allocate(size_t maxPairCount) -> void
	{
		bucketCount_ = (maxPairCount / BucketSize) + ((maxPairCount % BucketSize) != 0);
		capacity_ = bucketCount_ * BucketSize;
		maxCuckooChainLength_ = static_cast<uint32_t>(7 * (log(bucketCount_ * BucketSize) / log(2.0)));
		cudaMalloc(reinterpret_cast<void**>(&activeKeys_), sizeof(Key_) * capacity_);
		cudaMalloc(reinterpret_cast<void**>(&table_), sizeof(AtomicPair_) * capacity_);
		cudaMalloc(reinterpret_cast<void**>(&indices_), sizeof(uint32_t) * capacity_);
		cudaMalloc(&activeCount_, sizeof(float));
		internal::Reset<Pair_><<<4096, 256>>>(capacity_, table_, activeKeys_);
		cudaCheckError(cudaDeviceSynchronize());
	}

	MPM_FORCE_INLINE auto Deallocate() -> void
	{
		if (table_)
			{
				cudaFree(indices_);
				cudaFree(activeKeys_);
				cudaFree(table_);
				activeKeys_ = nullptr;
				table_ = nullptr;
				bucketCount_ = 0;
				capacity_ = 0;
				maxCuckooChainLength_ = 0;
				activeCount_ = nullptr;
			}
	}

	MPM_FORCE_INLINE auto Reallocate(size_t maxPairCount) -> void
	{
		Deallocate();
		Allocate(maxPairCount);
	}

	MPM_FORCE_INLINE MPM_DEV_FUNC auto CooperativeInsert(bool toInsert, Pair_ pair) -> bool
	{
		namespace cg = cooperative_groups;
		constexpr uint32_t kElectedLane = 0;
		bool tmpToInsert = toInsert;

		auto keyEqualOperator = GpuHashtableKeyEqualOperator_{};
		auto valueEqualOperator = GpuHashtableValueEqualOperator_{};

		auto tb = cg::this_thread_block();
		auto tile = cg::tiled_partition<BucketSize>(tb);
		auto threadRank = tile.thread_rank();

		bool success = true;
		auto k = pair.key_;

		while (uint32_t workBitmap = tile.ballot(toInsert))
			{
				auto curLane = __ffs(workBitmap) - 1;
				auto curPair = tile.shfl(pair, curLane);
				auto curResult = false;

				internal::XorshiftRngDevice<uint32_t> rng{};

				auto bucketId = hashFunction0_(curPair.key_) % bucketCount_;
				uint32_t cuckooCounter = 0;

				do
					{
						auto ptr = table_ + bucketId * BucketSize;
						auto lanePair = ptr[threadRank].load(cuda::memory_order_relaxed);
						auto curLoad = __popc(tile.ballot(
							!keyEqualOperator(lanePair.key_, internal::kGpuHashtableSentinelPair<Pair_>.key_)));

						if (curLoad < BucketSize)
							{
								if (tile.thread_rank() == kElectedLane)
									{
										Pair_ expected = internal::kGpuHashtableSentinelPair<Pair_>;
										Pair_ desired = curPair;
										curResult = ptr[curLoad].compare_exchange_strong(
											expected, desired, cuda::memory_order_relaxed, cuda::memory_order_relaxed);
									}
								curResult = tile.shfl(curResult, kElectedLane);
								if (curResult)
									break;
							}
						else
							{
								if (threadRank == kElectedLane)
									{
										auto randomIndex = rng() % BucketSize;
										auto oldPair = ptr[randomIndex].exchange(curPair, cuda::memory_order_relaxed);

										auto prevBucketId = bucketId;
										auto bucketId0 = hashFunction0_(oldPair.key_) % bucketCount_;
										auto bucketId1 = hashFunction1_(oldPair.key_) % bucketCount_;
										auto bucketId2 = hashFunction2_(oldPair.key_) % bucketCount_;

										bucketId = bucketId0;
										bucketId = prevBucketId == bucketId1 ? bucketId2 : bucketId;
										bucketId = prevBucketId == bucketId0 ? bucketId1 : bucketId;
										curPair = oldPair;
									}

								bucketId = tile.shfl(bucketId, kElectedLane);
								++cuckooCounter;
							}
				} while (cuckooCounter < maxCuckooChainLength_);

				if (threadRank == curLane)
					{
						toInsert = false;
						success = curResult;
					}
			}

		if (tmpToInsert && success)
			{
				activeKeys_[pair.value_] = pair.key_;
			}

		return success;
	}

	MPM_FORCE_INLINE MPM_DEV_FUNC auto CooperativeFind(bool toFind, const Key_& key) const -> Value_
	{

		namespace cg = cooperative_groups;
		auto tb = cg::this_thread_block();
		auto tile = cg::tiled_partition<BucketSize>(tb);
		auto result = internal::kGpuHashtableSentinelPair<Pair_>.value_;
		auto threadRank = tile.thread_rank();

		auto keyEqualOperator = GpuHashtableKeyEqualOperator_{};
		auto valueEqualOperator = GpuHashtableValueEqualOperator_{};

		while (uint32_t workBitmap = tile.ballot(toFind))
			{
				auto curLane = __ffs(workBitmap) - 1;
				auto curKey = tile.shfl(key, curLane);

				auto curResult = internal::kGpuHashtableSentinelPair<Pair_>.value_;

				for (uint32_t i = 0; i < kGpuHashFunctionsCount_; ++i)
					{
						uint32_t bucketId;
						if (i == 0)
							bucketId = hashFunction0_(curKey) % bucketCount_;
						else if (i == 1)
							bucketId = hashFunction1_(curKey) % bucketCount_;
						else
							bucketId = hashFunction2_(curKey) % bucketCount_;

						auto ptr = table_ + bucketId * BucketSize;
						auto pair = ptr[threadRank].load(cuda::memory_order_relaxed);
						auto laneId = __ffs(tile.ballot(keyEqualOperator(pair.key_, curKey)));
						curResult = laneId ? tile.shfl(pair.value_, laneId - 1)
										   : internal::kGpuHashtableSentinelPair<Pair_>.value_;
						if (valueEqualOperator(curResult, internal::kGpuHashtableSentinelPair<Pair_>.value_) &&
							__popc(tile.ballot(!keyEqualOperator(
								pair.key_, internal::kGpuHashtableSentinelPair<Pair_>.key_))) != BucketSize)
							break;
						if (!valueEqualOperator(curResult, internal::kGpuHashtableSentinelPair<Pair_>.value_))
							break;
					}

				if (threadRank == curLane)
					{
						toFind = false;
						result = curResult;
					}
			}

		return result;
	}

#ifndef NDEBUG

	auto PrintNonEmptyMemory() -> void
	{
		Pair_ *bufferCpu = nullptr, *bufferGpu = nullptr;
		Key_* keyBuffer = nullptr;

		keyBuffer = static_cast<Key_*>(malloc(sizeof(Key_) * capacity_));
		cudaMalloc(reinterpret_cast<void**>(&bufferGpu), sizeof(Pair_) * capacity_);
		bufferCpu = static_cast<Pair_*>(malloc(sizeof(Pair_) * capacity_));

		internal::LoadTableToBuffer<<<4096, 256>>>(capacity_, table_, bufferGpu);
		cudaMemcpy(bufferCpu, bufferGpu, sizeof(Pair_) * capacity_, cudaMemcpyDeviceToHost);
		cudaMemcpy(keyBuffer, activeKeys_, sizeof(Key_) * capacity_, cudaMemcpyDeviceToHost);
		cudaCheckError(cudaDeviceSynchronize());

		/* for(int i = 0; i < capacity_; ++i) */
		/* { */
		/*     if(bufferCpu[i] != internal::kGpuHashtableSentinelPair<Pair_>) */
		/*     { */
		/*         auto &p = bufferCpu[i]; */
		/*         printf("%d %d %d %d\n", p.key_.x_, p.key_.y_, p.key_.z_, p.value_); */
		/*     } */
		/* } */

		/* for(int i = 0; i < capacity_; ++i) */
		/* { */
		/*     auto& k = keyBuffer[i]; */
		/*     printf("%d %d %d %d\n", i, k.x_, k.y_, k.z_); */
		/* } */

		free(bufferCpu);
		free(keyBuffer);
		cudaFree(bufferGpu);
	}

#endif
};
}  // namespace hashtable

}  // namespace mpm
