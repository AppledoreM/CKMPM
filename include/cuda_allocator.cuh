#pragma once

#include "base.h"

namespace mpm
{

template <typename DerivedAllocator>
class CudaAllocatorBase
{
   public:
	template <typename T>
	inline auto Allocate(T*& ptr, size_t size) -> void
	{
		(static_cast<DerivedAllocator*>(this))->AllocateImpl(ptr, size);
	}

	template <typename T>
	inline auto Deallocate(T*& ptr, size_t size) -> void
	{
		(static_cast<DerivedAllocator*>(this))->DeallocateImpl(ptr, size);
	}
};

class CudaDefaultAllocator : public CudaAllocatorBase<CudaDefaultAllocator>
{
   public:
	friend class CudaAllocatorBase<CudaDefaultAllocator>;

   protected:
	template <typename T>
	inline auto AllocateImpl(T*& ptr, size_t size) -> void
	{
		cudaMalloc(reinterpret_cast<void**>(&ptr), size);
	}

	template <typename T>
	inline auto DeallocateImpl(T*& ptr, size_t size) -> void
	{
		cudaFree(ptr);
	}
};

}  // namespace mpm
