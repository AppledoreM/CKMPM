#pragma once

#include <cuda.h>

#ifndef MPM_HOST_DEV_FUNC
#define MPM_HOST_DEV_FUNC __device__ __host__
#endif

#ifndef MPM_DEV_FUNC
#define MPM_DEV_FUNC __device__
#endif

#ifndef MPM_FORCE_INLINE
#define MPM_FORCE_INLINE __forceinline__
#endif
