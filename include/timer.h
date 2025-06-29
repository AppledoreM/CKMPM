#pragma once

#include <cuda.h>
#include <chrono>

namespace mico
{
enum class TimerPlatform
{
	kCpu = 0,
	kCuda = 1
};

template <TimerPlatform Platform>
class Timer;

typedef std::chrono::time_point<std::chrono::steady_clock> CpuTimestamp;

template <>
class Timer<TimerPlatform::kCpu>
{
   public:
	Timer() = default;
	Timer(const Timer&) = default;
	Timer& operator=(const Timer&) = default;
	~Timer() = default;

	inline auto RecordTimestamp() -> void
	{
		timestamp_[rollIndex_] = std::chrono::steady_clock::now();
		rollIndex_ = (rollIndex_ + 1) & 1;
	}

	template <typename ElapsedTime = float, typename OutputUnit = std::chrono::milliseconds>
	inline auto GetElapsedTime() -> ElapsedTime
	{
		return std::chrono::duration_cast<OutputUnit>(timestamp_[(rollIndex_ + 1) & 1] -
																   timestamp_[rollIndex_]).count();
	}

   private:
	CpuTimestamp clock_;
	std::chrono::time_point<std::chrono::steady_clock> timestamp_[2];
	uint32_t rollIndex_ = 0;
};

template <>
class Timer<TimerPlatform::kCuda>
{
   public:
	Timer(cudaStream_t stream) : stream_(stream)
	{
		cudaEventCreate(&timestamp_[0]);
		cudaEventCreate(&timestamp_[1]);
	}
	~Timer()
	{
		cudaEventDestroy(timestamp_[0]);
		cudaEventDestroy(timestamp_[1]);
	}

	Timer(const Timer&) = default;
	Timer& operator=(const Timer&) = default;

	auto RecordTimestamp() -> void
	{
		cudaEventRecord(timestamp_[rollIndex_], stream_);
		rollIndex_ = (rollIndex_ + 1) & 1;
	}

	auto GetElapsedTime() -> float
	{
		float elapsedTime = 0.0;
		cudaEventSynchronize(timestamp_[(rollIndex_ + 1) & 1]);
		cudaEventElapsedTime(&elapsedTime, timestamp_[rollIndex_], timestamp_[(rollIndex_ + 1) & 1]);
		return elapsedTime;
	}

   private:
	cudaStream_t stream_;
	cudaEvent_t timestamp_[2];
	uint32_t rollIndex_ = 0;
};

}  // namespace mico
