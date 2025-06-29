#pragma once
#include "base.h"

#include <array>
#include <iostream>
#include <vector>

constexpr static bool kEnableGpuDebugMessage = false;

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
		{
			fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
			if (abort)
				exit(code);
		}
}
#define cudaCheckError(ans)                   \
	{                                         \
		gpuAssert((ans), __FILE__, __LINE__); \
	}

template <typename... Argument>
MPM_FORCE_INLINE MPM_DEV_FUNC auto GpuDebugPrint(const char* message, Argument... argument) -> void
{
	if (kEnableGpuDebugMessage)
		{
			printf(message, argument...);
		}
}

struct CudaKernelLaunchParameter
{
	dim3 gridParameter_{};
	dim3 blockParameter_{};
	size_t sharedMemorySize_{0};
	cudaStream_t stream_{cudaStreamDefault};

	template <typename GridDim, typename BlockDim>
	CudaKernelLaunchParameter(GridDim g0, BlockDim b0, size_t sharedMemorySize = 0,
							  cudaStream_t stream = cudaStreamDefault)
		: gridParameter_{static_cast<uint32_t>(g0)},
		  blockParameter_{static_cast<uint32_t>(b0)},
		  sharedMemorySize_(sharedMemorySize),
		  stream_(stream)
	{
	}

	inline auto IsValid() const -> bool
	{
		return gridParameter_.x && gridParameter_.y && gridParameter_.z && blockParameter_.x && blockParameter_.y &&
			   blockParameter_.z;
	}
};

class CudaUtil
{
   private:
	CudaUtil();

   public:
	enum class StreamIndex
	{
		kCompute = 0,
		kSpare,
		kTotal = 32
	};

	enum class EventIndex
	{
		kCompute = 0,
		kSpare,
		kTotal = 32
	};
	struct CudaContext
	{
	   public:
		explicit CudaContext(int deviceIndex) : deviceIndex_(deviceIndex)
		{
			if (deviceIndex_ != -1)
				{
					printf("[Cuda] Initialize device index to %d\n", deviceIndex);
				}
		}

		inline auto SetContext() -> void { cudaSetDevice(deviceIndex_); }

		template <StreamIndex kIndex>
		inline auto SyncStream() const -> void
		{
			cudaCheckError(cudaStreamSynchronize(CudaUtil::GetInstance()->stream_[static_cast<uint32_t>(kIndex)]));
		}

		inline auto GetComputeStream() const -> cudaStream_t
		{
			return CudaUtil::GetInstance()->stream_[static_cast<uint32_t>(StreamIndex::kCompute)];
		}

		template <typename Func, typename... Argument>
		auto LaunchCompute(CudaKernelLaunchParameter&& launchParameter, Func&& f, Argument... args) -> void
		{
			static_assert(!std::disjunction_v<std::is_reference<Argument>...>, "Cannot pass refernce to cuda kernels.");
			if (launchParameter.IsValid())
				{
					std::forward<Func>(f)<<<launchParameter.gridParameter_, launchParameter.blockParameter_,
											launchParameter.sharedMemorySize_, launchParameter.stream_>>>(args...);
					cudaCheckError(cudaGetLastError());
				}
		}

		template <typename... Argument>
		auto LaunchCompute(CudaKernelLaunchParameter&& launchParameter, void (*f)(Argument...),
						   Argument... args) -> void
		{
			static_assert(!std::disjunction_v<std::is_reference<Argument>...>, "Cannot pass refernce to cuda kernels.");
			if (launchParameter.IsValid())
				{
					f<<<launchParameter.gridParameter_, launchParameter.blockParameter_,
						launchParameter.sharedMemorySize_, launchParameter.stream_>>>(args...);
					cudaCheckError(cudaGetLastError());
				}
		}

	   private:
		int deviceIndex_ = -1;
	};

	~CudaUtil() = default;
	CudaUtil(const CudaUtil&) = delete;
	CudaUtil& operator=(const CudaUtil&) = delete;
	CudaUtil(CudaUtil&&) = delete;

	inline static auto GetInstance() -> CudaUtil*
	{
		static CudaUtil instance_{};
		return &instance_;
	}

	static auto GetCudaContext() -> CudaContext& { return GetInstance()->context_; }

   private:
	CudaContext context_;
	cudaDeviceProp prop_;

	std::array<cudaStream_t, static_cast<uint32_t>(StreamIndex::kTotal)> stream_;
};
