#pragma once
#include "base.h"
#include "cuda_util.cuh"
#include "data_type.cuh"
#include "gpu_hash_table.cuh"
#include "mpm_meta.h"

#include <assert.h>
#include <cstdint>
#include <limits>

namespace mpm
{
template <typename... GridAttribute>
using MPMGridAttribute = meta::AttributeWrapper<GridAttribute...>;

template <typename GridAttribute, uint32_t kGridCount, uint32_t kBlockXSize, uint32_t kBlockYSize, uint32_t kBlockZSize>
struct MPMGridBlock
{
	typedef GridAttribute GridAttribute_;
	constexpr static uint32_t kGridCount_ = kGridCount;
	constexpr static uint32_t kBlockXSize_ = kBlockXSize;
	constexpr static uint32_t kBlockYSize_ = kBlockYSize;
	constexpr static uint32_t kBlockZSize_ = kBlockZSize;
	constexpr static uint32_t kBlockSize_ = kGridCount_ * kBlockYSize_ * kBlockYSize_ * kBlockZSize_;

	template <size_t AttributeIndex>
	MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetValue(uint32_t index) const ->
		typename GridAttribute::template AttributeWrapperAccessor<AttributeIndex>::Type
	{
		typedef typename GridAttribute::template AttributeWrapperAccessor<AttributeIndex>::Type TargetAttribute;
		assert(kBlockSize_ * GridAttribute::template AttributeWrapperAccessor<AttributeIndex>::kOffset_ +
				   index * sizeof(TargetAttribute) <
			   GridAttribute::kSize_ * kBlockSize_);
		return *reinterpret_cast<const TargetAttribute*>(
			block_ + kBlockSize_ * GridAttribute::template AttributeWrapperAccessor<AttributeIndex>::kOffset_ +
			index * sizeof(TargetAttribute));
	}

	template <size_t AttributeIndex>
	MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetValue(uint32_t index) ->
		typename GridAttribute::template AttributeWrapperAccessor<AttributeIndex>::Type&
	{
		typedef typename GridAttribute::template AttributeWrapperAccessor<AttributeIndex>::Type TargetAttribute;
		assert(kBlockSize_ * GridAttribute::template AttributeWrapperAccessor<AttributeIndex>::kOffset_ +
				   index * sizeof(TargetAttribute) <
			   GridAttribute::kSize_ * kBlockSize_);
		return *reinterpret_cast<TargetAttribute*>(
			block_ + kBlockSize_ * GridAttribute::template AttributeWrapperAccessor<AttributeIndex>::kOffset_ +
			index * sizeof(TargetAttribute));
	}

	template <size_t AttributeIndex>
	MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto SetValue(
		uint32_t index, typename GridAttribute::template AttributeWrapperAccessor<AttributeIndex>::Type value) -> void
	{
		typedef typename GridAttribute::template AttributeWrapperAccessor<AttributeIndex>::Type TargetAttribute;
		assert(kBlockSize_ * GridAttribute::template AttributeWrapperAccessor<AttributeIndex>::kOffset_ +
				   index * sizeof(TargetAttribute) <
			   GridAttribute::kSize_ * kBlockSize_);
		*reinterpret_cast<TargetAttribute*>(
			block_ + kBlockSize_ * GridAttribute::template AttributeWrapperAccessor<AttributeIndex>::kOffset_ +
			index * sizeof(TargetAttribute)) = value;
	}

	uint8_t block_[GridAttribute::kSize_ * kBlockSize_];
};

struct MPMGridBlockCoordinate : public Vector<int, 3>
{
	constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC explicit MPMGridBlockCoordinate(int blockX = 0, int blockY = 0,
																				 int blockZ = 0)
		: Vector<int, 3>(blockX, blockY, blockZ)
	{
	}

	constexpr MPM_FORCE_INLINE MPMGridBlockCoordinate(const MPMGridBlockCoordinate&) = default;

	MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto operator==(const MPMGridBlockCoordinate& rhs) const -> bool
	{
		return (*this)[0] == rhs[0] && (*this)[1] == rhs[1] && (*this)[2] == rhs[2];
	}

	MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto operator<(const MPMGridBlockCoordinate& rhs) const -> bool
	{
		return ((*this)[0] < rhs[0]) || ((*this)[0] == rhs[0] && (*this)[1] < rhs[1]) ||
			   ((*this)[0] == rhs[0] && (*this)[1] == rhs[1] && (*this)[2] < rhs[2]);
	}
};

struct MPMGridBlockCoordinateSentinel : hashtable::internal::GpuHashtableKeySentinelTag
{
	constexpr static auto kKeySentinelValue_ = MPMGridBlockCoordinate{
		std::numeric_limits<int>::max(), std::numeric_limits<int>::max(), std::numeric_limits<int>::max()};
};

template <typename Config, int kGridIndex, typename Grid>
__global__ auto ClearGrid(Config config, uint32_t blockCount, Grid grid) -> void
{
	typedef typename Grid::GridBlock_ GridBlock;
	typedef typename GridBlock::GridAttribute_ GridAttribute;
	constexpr int kAttributeCount = GridAttribute::kAttributeCount_;
	uint32_t blockIndex = blockIdx.x;

	if (blockIndex >= blockCount)
		return;
	auto block = grid.GetBlock(blockIndex);

	for (uint32_t cellIndex = threadIdx.x; cellIndex < 2 * config.GetBlockVolume(); cellIndex += blockDim.x)
		{
			if constexpr (kGridIndex == -1)
				{
					if (cellIndex >= config.GetBlockVolume())
						break;
				}
			if constexpr (kGridIndex == 1)
				{
					if (cellIndex < config.GetBlockVolume())
						continue;
				}

			meta::ConstexprLoop<0, kAttributeCount>(
				[&](auto indexWrapper) -> void
				{
					constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
					block->template SetValue<index>(cellIndex, 0);
				});
		}
}

template <typename Config, int kGridIndex, typename Grid>
__global__ auto CopyGrid(Config config, uint32_t blockCount, Grid destinationGrid, const Grid sourceGrid) -> void
{
	typedef typename Grid::GridBlock_ GridBlock;
	typedef typename GridBlock::GridAttribute_ GridAttribute;
	constexpr int kAttributeCount = GridAttribute::kAttributeCount_;
	uint32_t blockIndex = blockIdx.x;

	if (blockIndex >= blockCount)
		return;
	auto dstBlock = destinationGrid.GetBlock(blockIndex);
	const auto srcBlock = sourceGrid.GetBlock(blockIndex);

	for (uint32_t cellIndex = threadIdx.x; cellIndex < 2 * config.GetBlockVolume(); cellIndex += blockDim.x)
		{
			if constexpr (kGridIndex == -1)
				{
					if (cellIndex >= config.GetBlockVolume())
						break;
				}
			if constexpr (kGridIndex == 1)
				{
					if (cellIndex < config.GetBlockVolume())
						continue;
				}

			meta::ConstexprLoop<0, kAttributeCount>(
				[&](auto indexWrapper) -> void
				{
					constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
					dstBlock->template SetValue<index>(cellIndex, srcBlock->template GetValue<index>(cellIndex));
				});
		}
}

template <typename GridBlock>
class MPMGrid
{
   public:
	typedef GridBlock GridBlock_;

	MPM_FORCE_INLINE MPMGrid() = default;
	MPM_FORCE_INLINE MPM_HOST_DEV_FUNC MPMGrid(const MPMGrid& rhs)
		: grid_(rhs.grid_), maxGridBlockCount_(rhs.maxGridBlockCount_)
	{
	}
	MPM_FORCE_INLINE ~MPMGrid() = default;

	template <typename Allocator>
	inline auto Allocate(Allocator allocator, size_t maxGridBlockCount) -> void
	{
		assert(maxGridBlockCount_ == 0);
		maxGridBlockCount_ = maxGridBlockCount;
		allocator.Allocate(grid_, maxGridBlockCount_ * sizeof(GridBlock));
	}

	template <typename Allocator>
	inline auto Deallocate(Allocator allocator) -> void
	{
		allocator.Deallocate(grid_, maxGridBlockCount_ * sizeof(GridBlock));
		maxGridBlockCount_ = 0;
	}

	template <typename Allocator>
	inline auto Reallocate(Allocator allocator, size_t maxGridBlockCount) -> void
	{
		DeallocateGrid(allocator);
		AllocateGrid(allocator, maxGridBlockCount);
	}

	template <typename Config, int kGridIndex = 0>
	inline auto Reset(const Config& config, uint32_t blockCount) -> void
	{
		auto cuContext = CudaUtil::GetCudaContext();
		cuContext.LaunchCompute({blockCount, 2 * config.GetBlockVolume()}, ClearGrid<Config, kGridIndex, MPMGrid>,
								config, blockCount, *this);
	}

	template <typename Config, int kGridIndex = 0>
	inline auto Copy(const Config& config, uint32_t blockCount, const MPMGrid& rhs) -> void
	{
		auto cuContext = CudaUtil::GetCudaContext();
		cuContext.LaunchCompute({blockCount, 2 * config.GetBlockVolume()}, CopyGrid<Config, kGridIndex, MPMGrid>,
								config, blockCount, *this, rhs);
	}

	MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetBlock(uint32_t blockIndex) -> GridBlock* { return (grid_ + blockIndex); }

	MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto GetBlock(uint32_t blockIndex) const -> const GridBlock*
	{
		return (grid_ + blockIndex);
	}

   private:
	size_t maxGridBlockCount_ = 0;
	GridBlock* grid_ = nullptr;
};

}  // namespace mpm
