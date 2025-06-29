#pragma once
#include <cuda_profiler_api.h>
#include <chrono>
#include <fstream>
#include <span>
#include <string>
#include <typeinfo>
#include <variant>
#include <vector>

#include "cuda_allocator.cuh"
#include "mpm_algorithm.cuh"
#include "mpm_benchmark.h"
#include "mpm_config.h"
#include "mpm_grid.cuh"
#include "mpm_io.h"
#include "mpm_material.cuh"
#include "mpm_meta.h"
#include "mpm_model.h"
#include "mpm_particle.cuh"
#include "mpm_partition.cuh"
#include "spdlog/spdlog.h"
#include "timer.h"

namespace mpm
{

template <typename MPMGridConfig>
class MPMEngine
{
   private:
	MPMEngine() = default;

	struct AuxiliaryBuffer
	{
		float* maxVelocityNorm_ = nullptr;
		uint32_t* activeBlockMark_ = nullptr;
		uint32_t* activeParticleBlockMarkSource_ = nullptr;
		uint32_t* activeParticleBlockMarkDestination_ = nullptr;
		uint32_t* binCount_ = nullptr;

		template <typename Config>
		inline auto GetTotalSize(const Config& config) -> size_t
		{
			return sizeof(float) + sizeof(uint32_t) * 4 * config.GetMaxActiveBlockCount();
		}

		template <typename Allocator, typename Config>
		inline auto Allocate(Allocator allocator, const Config& config) -> void
		{
			allocator.Allocate(data_, GetTotalSize(config));

			maxVelocityNorm_ = reinterpret_cast<float*>(data_);
			activeBlockMark_ = reinterpret_cast<uint32_t*>(data_ + sizeof(float));
			activeParticleBlockMarkSource_ =
				reinterpret_cast<uint32_t*>(data_ + sizeof(float) + sizeof(uint32_t) * config.GetMaxActiveBlockCount());
			activeParticleBlockMarkDestination_ = reinterpret_cast<uint32_t*>(
				data_ + sizeof(float) + 2 * sizeof(uint32_t) * config.GetMaxActiveBlockCount());
			binCount_ = reinterpret_cast<uint32_t*>(data_ + sizeof(float) +
													3 * sizeof(uint32_t) * config.GetMaxActiveBlockCount());
		}

		template <typename Allocator>
		inline auto Deallocate(Allocator allocator) -> void
		{
			allocator.Deallocate(data_);
		}

		uint8_t* data_ = nullptr;
	};

	AuxiliaryBuffer auxiliaryBuffer_;

   public:
	constexpr static uint32_t kDefaultCudaBlockSize_ = 256;
	constexpr static uint32_t kBufferCount_ = 2;
	constexpr static size_t kGridBlockSize_ = 4;
	constexpr static uint32_t kNDim_ = 3;

	inline static auto GetInstance() -> MPMEngine*
	{
		static MPMEngine instance_{};
		return &instance_;
	}

	~MPMEngine() = default;
	MPMEngine(const MPMEngine&) = delete;
	MPMEngine(MPMEngine&&) = delete;
	MPMEngine& operator=(const MPMEngine&) = delete;

	template <MPMConfigType Config>
	auto InitializeParticle(const Config& config,
							std::vector<MPMModelVariant>& models) -> void
	{
        models_ = &models;
		auto& cuContext = CudaUtil::GetCudaContext();
		CudaDefaultAllocator allocator;
		spdlog::info("Start initializing particles.");
		for (int modelIndex = 0; modelIndex < static_cast<int>(models.size()); ++modelIndex)
			{
                std::visit([&](auto&& model) -> void
                           {
                              constexpr auto ConstitutiveModel = std::decay_t<decltype(model)>::kConstitutiveModel_;
                            const uint32_t particleCount = model.GetParticleCount();
                            spdlog::info("\tInitializing model with particle count {}", particleCount);
                            auto& hostParticlePosition = model.GetParticlePosition();
                            auto& hostParticleVelocity = model.GetParticleVelocity();

                            float maxVelocityNorm = 0.0;
                            std::for_each(std::begin(hostParticleVelocity), std::end(hostParticleVelocity),
                                          [&maxVelocityNorm](auto& velocity) -> void
                                          { maxVelocityNorm = max(maxVelocityNorm, velocity.Norm()); });
                            if (maxVelocityNorm > 0.0)
                                {
                                    dt_ = std::min(dt_, config.GetDx() * config.GetCfl() / maxVelocityNorm);
                                    spdlog::info("Set Default dt = {}", dt_);
                                }

                            particleMaterial_.emplace_back(model.GetParticleMaterial());

                            for (int i = 0; i < kBufferCount_; ++i)
                                {
                                    particleBuffer_[i].emplace_back(
                                        MPMParticleBuffer<typename MPMMaterial<ConstitutiveModel>::Particle_>{});

                                    std::visit(
                                        [&](auto&& buffer) -> void
                                        {
                                            buffer.AllocateParticleBuffer(CudaDefaultAllocator{},
                                                                          (config.GetMaxActiveBlockCount() +
                                                                           particleCount / config.GetMaxParticleCountPerBucket()) *
                                                                              config.GetMaxParticleCountPerBucket());
                                            buffer.ReserveBucket(CudaDefaultAllocator{}, config,
                                                                 static_cast<size_t>(config.GetMaxActiveBlockCount()));
                                            buffer.SetParticleMass(model.GetParticleMass());
                                            buffer.SetParticleVolume(model.GetParticleVolume());
                                        },
                                        particleBuffer_[i].back());
                                }

                            particleCount_.emplace_back(static_cast<size_t>(particleCount));
                            velocityCount_.emplace_back(hostParticleVelocity.size());

                            //TODO: Write a pre-allocated cuda memory allocator
                            Vector<float, 3>* deviceParticlePosition = nullptr;
                            Vector<float, 3>* deviceParticleVelocity = nullptr;
                            allocator.Allocate(deviceParticlePosition, sizeof(Vector<float, kNDim_>) * particleCount);
                            allocator.Allocate(deviceParticleVelocity, sizeof(Vector<float, kNDim_>) * hostParticleVelocity.size());

                            cudaMemcpyAsync(
                                static_cast<void*>(deviceParticlePosition), static_cast<const void*>(hostParticlePosition.data()),
                                sizeof(Vector<float, kNDim_>) * particleCount, cudaMemcpyDefault, cuContext.GetComputeStream());
                            cudaMemcpyAsync(static_cast<void*>(deviceParticleVelocity),
                                            static_cast<const void*>(hostParticleVelocity.data()),
                                            sizeof(Vector<float, kNDim_>) * hostParticleVelocity.size(), cudaMemcpyDefault,
                                            cuContext.GetComputeStream());

                            deviceParticlePosition_.emplace_back(deviceParticlePosition);
                            deviceParticleVelocity_.emplace_back(deviceParticleVelocity);
                                
                            

                           }, models[modelIndex]
                           );
			}
		cuContext.template SyncStream<CudaUtil::StreamIndex::kCompute>();
		spdlog::info("Finished initializating particles.");
	}

	template <MPMConfigType Config>
	inline auto Initialize(const Config& config) -> void
	{
		spdlog::set_pattern("[%^MPM Engine %l%$] [%H:%M:%S %z] [thread %t] %v");
		auto& cuContext = CudaUtil::GetCudaContext();

		cuContext.SetContext();

		dt_ = std::min(config.GetDt(), 1.0f / config.GetFps());
		spdlog::info("Set Default dt = {}", dt_);

		for (int i = 0; i < static_cast<int>(kBufferCount_); ++i)
			{
				grid_[i].Allocate(CudaDefaultAllocator{}, config.GetMaxActiveBlockCount());
				partition_[i].Allocate(CudaDefaultAllocator{}, config.GetMaxActiveBlockCount());
				partition_[i].Reset(cuContext.GetComputeStream());
			}

		// Allocate auxiliary buffer
		auxiliaryBuffer_.Allocate(CudaDefaultAllocator{}, config);
		cuContext.template SyncStream<CudaUtil::StreamIndex::kCompute>();
	}

	inline auto GetModelCount() -> int
	{
		assert(particleBuffer_[0].size() == particleBuffer_[1].size());
		return static_cast<int>(particleBuffer_[0].size());
	}

	template <uint32_t kCountPerRoll = 1, uint32_t gridIndex = 0>
	inline auto GetNextRollIndex() -> uint32_t
	{
		assert(0 <= gridIndex && gridIndex <= kCountPerRoll);
		return (kCountPerRoll * (rollIndex_ + 1) + gridIndex) % (kCountPerRoll * kBufferCount_);
	}

	template <uint32_t kCountPerRoll = 1, uint32_t gridIndex = 0>
	inline auto GetRollIndex() -> uint32_t
	{
		assert(0 <= gridIndex && gridIndex <= kCountPerRoll);
		return (kCountPerRoll * rollIndex_ + gridIndex) % (kCountPerRoll * kBufferCount_);
	}

	inline auto IterateRollIndex() -> void { rollIndex_ = (rollIndex_ + 1) % kBufferCount_; }

	template <MPMConfigType Config>
	auto InitialSetup(const Config& config) -> void
	{
		auto& cuContext = CudaUtil::GetCudaContext();
		typedef Vector<float, 3>* ParticleArray;

		for (int i = 0; i < GetModelCount(); ++i)
			{
				cuContext.LaunchCompute({4096, kDefaultCudaBlockSize_},
										ActivateBlocksWithParticles<Config, Partition_, ParticleArray>, config,
										particleCount_[i], deviceParticlePosition_[i], partition_[GetNextRollIndex()]);
			}

		cudaCheckError(cudaMemcpyAsync(&activePartitionBlockCount_, partition_[GetNextRollIndex()].count_,
									   sizeof(uint32_t), cudaMemcpyDeviceToHost, cuContext.GetComputeStream()));

		cuContext.SyncStream<CudaUtil::StreamIndex::kCompute>();
		spdlog::info("Active block count: {}.", activePartitionBlockCount_);

		for (int i = 0; i < GetModelCount(); ++i)
			{
				std::visit(
					[&](auto&& buffer) -> void
					{
						typedef typename std::decay_t<decltype(buffer)>::Particle_ Particle;
						cuContext.LaunchCompute({4096, kDefaultCudaBlockSize_},
												BuildParticleCellBucket<Config, ParticleArray, Particle, Partition_>,
												config, particleCount_[i], deviceParticlePosition_[i], buffer,
												partition_[GetNextRollIndex()]);
					},
					particleBuffer_[GetRollIndex()][i]);
			}

		cuContext.SyncStream<CudaUtil::StreamIndex::kCompute>();

		if (activePartitionBlockCount_ > config.GetMaxActiveBlockCount())
			{
				spdlog::error("Too many active blocks. Active block count {}.", activePartitionBlockCount_);
				std::abort();
			}

		for (int i = 0; i < GetModelCount(); ++i)
			{
				std::visit(
					[&](auto&& buffer) -> void
					{
						typedef typename std::decay_t<decltype(buffer)>::Particle_ Particle;
						cudaCheckError(cudaMemsetAsync(buffer.particleBucketSize_, 0,
													   sizeof(uint32_t) * (activePartitionBlockCount_ + 1),
													   cuContext.GetComputeStream()));
						cuContext.LaunchCompute({activePartitionBlockCount_, config.GetBlockVolume()},
												ParticleCellBucketToBlock<Config, Particle>, config, buffer);
					},
					particleBuffer_[GetRollIndex()][i]);
			}

		auto* binCount = auxiliaryBuffer_.binCount_;

		for (int i = 0; i < GetModelCount(); ++i)
			{
				std::visit(
					[&](auto&& buffer, auto&& material) -> void
					{
						typedef typename std::decay_t<decltype(buffer)>::Particle_ Particle;
						cuContext.LaunchCompute({activePartitionBlockCount_ / 128 + 1, 128}, ComputeBinCount<Config>,
												config, activePartitionBlockCount_, buffer.particleBucketSize_,
												binCount);
						ExclusiveScan(activePartitionBlockCount_ + 1, binCount, buffer.particleBinOffset_,
									  cuContext.GetComputeStream());

						cuContext.SyncStream<CudaUtil::StreamIndex::kCompute>();
                        if constexpr(std::is_same_v<typename std::decay_t<decltype(material)>::Particle_, Particle>)
                        {
                            cuContext.LaunchCompute(
                                {activePartitionBlockCount_, 128},
                                CopyParticleArrayToParticleBuffer<
                                    Config, std::decay_t<decltype(material)>::kConstitutiveModel_, ParticleArray, Particle>,
                                config, particleCount_[i], material, deviceParticlePosition_[i], buffer);
                        }
					},
					particleBuffer_[GetRollIndex()][i], particleMaterial_[i]);
			}

		cuContext.LaunchCompute({NextNearestMultipleOf<128>(activePartitionBlockCount_) / 128, 128},
								ActivateBlocks<MPMActivateBlockPolicy::Neighbor, Config, Partition_>, config,
								activePartitionBlockCount_, partition_[GetNextRollIndex()]);

		cudaCheckError(cudaMemcpyAsync(&activePartitionNeighborBlockCount_, partition_[GetNextRollIndex()].count_,
									   sizeof(uint32_t), cudaMemcpyDeviceToHost, cuContext.GetComputeStream()));

		cuContext.SyncStream<CudaUtil::StreamIndex::kCompute>();

		if (activePartitionNeighborBlockCount_ > config.GetMaxActiveBlockCount())
			{
				spdlog::error("Too many active blocks. Active neighbor block count {}.",
							  activePartitionNeighborBlockCount_);
				std::abort();
			}
		spdlog::info("Activate neighbor block count: {}.", activePartitionNeighborBlockCount_);

		cuContext.LaunchCompute({NextNearestMultipleOf<128>(activePartitionBlockCount_) / 128, 128},
								ActivateBlocks<MPMActivateBlockPolicy::Exterior, Config, Partition_>, config,
								activePartitionBlockCount_, partition_[GetNextRollIndex()]);

		cudaCheckError(cudaMemcpyAsync(&activePartitionExteriorBlockCount_, partition_[GetNextRollIndex()].count_,
									   sizeof(uint32_t), cudaMemcpyDeviceToHost, cuContext.GetComputeStream()));

		cuContext.SyncStream<CudaUtil::StreamIndex::kCompute>();

		if (activePartitionExteriorBlockCount_ > config.GetMaxActiveBlockCount())
			{
				spdlog::error("Too many active blocks. Active neighbor block count {}.",
							  activePartitionExteriorBlockCount_);
				std::abort();
			}
		spdlog::info("Activate exterior block count: {}.", activePartitionExteriorBlockCount_);

		partition_[GetRollIndex()].Copy(partition_[GetNextRollIndex()], cuContext.GetComputeStream());

		for (int i = 0; i < GetModelCount(); ++i)
			{
				std::visit(
					[&](auto&& buffer) -> void
					{
						std::get<std::decay_t<decltype(buffer)>>(particleBuffer_[GetNextRollIndex()][i])
							.Copy(config, buffer, activePartitionNeighborBlockCount_, cuContext.GetComputeStream());
					},
					particleBuffer_[GetRollIndex()][i]);
			}

		grid_[GetRollIndex()].Reset(config, activePartitionNeighborBlockCount_);
		cuContext.SyncStream<CudaUtil::StreamIndex::kCompute>();

		auto allocator = mpm::CudaDefaultAllocator{};
		for (int i = 0; i < GetModelCount(); ++i)
			{
				std::visit(
					[&](auto&& buffer) -> void
					{
						spdlog::info("Rasterizing with mass {}.", buffer.GetParticleMass());
						cuContext.LaunchCompute({4096, kDefaultCudaBlockSize_},
												Rasterize<Config, GridBlock_, Partition_, ParticleArray>, config,
												particleCount_[i], deviceParticlePosition_[i], grid_[GetRollIndex()],
												partition_[GetRollIndex()], 0.0, buffer.GetParticleMass(),
												velocityCount_[i], deviceParticleVelocity_[i]);
					},
					particleBuffer_[GetRollIndex()][i]);

				std::visit(
					[&](auto&& buffer) -> void
					{
						cuContext.LaunchCompute({activePartitionBlockCount_, 128}, InitializeAdvectionBucket<Config>,
												config, buffer.particleBucketSize_, buffer.blockBucket_);
					},
					particleBuffer_[GetNextRollIndex()][i]);
				allocator.Deallocate(deviceParticleVelocity_[i], particleCount_[i] * sizeof(float) * 3);
			}

		grid_[GetNextRollIndex()].Copy(config, activePartitionNeighborBlockCount_, grid_[GetRollIndex()]);
		cuContext.SyncStream<CudaUtil::StreamIndex::kCompute>();
		spdlog::info(
			"Finished initial setup.\n"
			"\tActive partition count: {}.\n"
			"\tActive neighbor partition count: {}.\n"
			"\tActive exterior partition count: {}.\n",
			activePartitionBlockCount_, activePartitionNeighborBlockCount_, activePartitionExteriorBlockCount_);


	}

	template <typename Config>
	inline auto ExportParticleToModel(const Config& config, const std::string& exportRootPath, int microsecond, bool exportParticlePositionAsBinary=false) -> void
	{
		auto cuContext = CudaUtil::GetCudaContext();

		mico::Timer<mico::TimerPlatform::kCuda> deviceTimer(cuContext.GetComputeStream());

		auto allocator = mpm::CudaDefaultAllocator{};

		deviceTimer.RecordTimestamp();

		uint32_t* deviceParticleCount = nullptr;
		allocator.Allocate(deviceParticleCount, sizeof(uint32_t));

		for (int i = 0; i < GetModelCount(); ++i)
			{
				std::vector<std::array<float, 3>> position = {};
				uint32_t hostParticleCount = 0;
				cudaCheckError(cudaMemsetAsync(deviceParticleCount, 0, sizeof(uint32_t), cuContext.GetComputeStream()));

				std::visit(
					[&](auto&& buffer) -> void
					{
						cuContext.LaunchCompute(
							{activePartitionBlockCount_, 128},
							RetrieveParticleBuffer<Config, Partition_, std::decay_t<decltype(buffer)>,
												   Vector<float, kNDim_>*>,
							config, partition_[GetRollIndex()], partition_[GetNextRollIndex()], deviceParticleCount,
							buffer, std::get<std::decay_t<decltype(buffer)>>(particleBuffer_[GetNextRollIndex()][i]),
							deviceParticlePosition_[i]);
					},
					particleBuffer_[GetRollIndex<1>()][i]);

				cudaCheckError(cudaMemcpyAsync(&hostParticleCount, deviceParticleCount, sizeof(uint32_t),
											   cudaMemcpyDeviceToHost, cuContext.GetComputeStream()));
				cuContext.SyncStream<CudaUtil::StreamIndex::kCompute>();

				spdlog::info("Frame {}: Exporting model {} with {} particles.", currentFrameIndex_, i,
							 hostParticleCount);

				assert(particleCount_[i] >= hostParticleCount && hostParticleCount > 0);

				position.resize(hostParticleCount);

				cudaCheckError(cudaMemcpyAsync(position.data(), deviceParticlePosition_[i],
											   sizeof(float) * 3 * hostParticleCount, cudaMemcpyDeviceToHost));
				cuContext.SyncStream<CudaUtil::StreamIndex::kCompute>();

                if(exportParticlePositionAsBinary)
                {
                    std::ofstream binaryFile(exportRootPath + "/model_" + std::to_string(i) + "_particle_frame_" + std::to_string(currentFrameIndex_) + ".txt", std::ios::binary | std::ios::out);
                    if(!binaryFile.is_open())
                    {
                        std::cerr << "Failed to find binary file location.\n";
                        exit(1);
                    }

                    binaryFile.write(reinterpret_cast<char*>(position.data()), sizeof(float) * 3 * hostParticleCount);
                    binaryFile.close();
                }

				MPMParticleIoAttribute<float, 3> attribute = {};
				attribute.attributeName_ = "position";
				attribute.attribute_ = std::move(position);
				MPMParticleIoData<decltype(attribute)> ioData(std::move(attribute));

				exporter_.ExportDataAsync(
					exportRootPath + "/" + "model_" + std::to_string(i) + "_particle_frame_" + std::to_string(currentFrameIndex_) + ".bgeo", ioData);
			}
	}

	template <typename Config>
	auto Simulate(Config& config, const std::string& exportRootPath = "", bool collectConservationMetric = false, bool exportParticlePositionAsBinary = false, bool profileMode = false) -> void
	{
		spdlog::set_level(spdlog::level::debug);
		auto cuContext = CudaUtil::GetCudaContext();
		mico::Timer<mico::TimerPlatform::kCpu> hostTimer;
		mico::Timer<mico::TimerPlatform::kCuda> deviceTimer(cuContext.GetComputeStream());


		const double secondsPerFrame = 1.0 / config.GetFps();

        if(!profileMode)
        {
            exporter_.Run();
        }
        else
        {
            spdlog::set_level(spdlog::level::err);
        }

		spdlog::info("Initial Dt={}. Starting Simulation.", dt_);

		currentTimestamp_ = std::chrono::duration<double>(0.0);
		double currentDt = 0.f;
		double nextDt = dt_;


		const auto simulationStartTimestamp_ = currentTimestamp_;

        auto benchmark = MPMBenchmark::GetInstance();
        {
            benchmark->AddBenchmark("G2P2G (Cuda)");
            benchmark->AddBenchmark("Total");
            benchmark->AddBenchmark("Activate Neighbor");
            benchmark->AddBenchmark("Activate Exterior");
            benchmark->AddBenchmark("Copy Grid Data");
        }

        Vector<float, 3> *conservationMetric = nullptr;
        Vector<float, 3> conservationMetricHost[4];
        std::ofstream conservationMetricOutputFile[GetModelCount()];
        if(collectConservationMetric)
        {
            for(int i = 0; i < GetModelCount(); ++i)
            {
                conservationMetricOutputFile[i].open(exportRootPath + "/conservationMetric_model_" + std::to_string(i) + ".bin", std::ios::out | std::ios::binary);
                
                if(!conservationMetricOutputFile[i].is_open())
                {
                    spdlog::error("Failed to open conservation metric output file.");
                    exit(0);
                }
            }
            cudaMalloc(reinterpret_cast<void**>(&conservationMetric), sizeof(float) * 3 * 4);
            cudaDeviceSynchronize();
        }

		hostTimer.RecordTimestamp();
		for (currentFrameIndex_ = 0; currentFrameIndex_ < config.GetTotalSimulatedFrameCount(); ++currentFrameIndex_)
			{
                std::cout << "Frame " << currentFrameIndex_ << "\n";
				const auto nextFrameTimestamp =
					currentTimestamp_ + std::chrono::duration<double>{secondsPerFrame};
				for (; currentTimestamp_ < nextFrameTimestamp; ++step_)
					{
						float maxVelocityNorm = 0.0;

						// Update Dt
						{

							deviceTimer.RecordTimestamp();
							cudaCheckError(cudaMemsetAsync(auxiliaryBuffer_.maxVelocityNorm_, 0, sizeof(float),
														   cuContext.GetComputeStream()));

							spdlog::info("Updating Dt");
                            cuContext.LaunchCompute(
                                {NextNearestMultipleOf<16>(activePartitionNeighborBlockCount_) / 16, 512},
                                UpdateGridVelocityAndQueryMax<true, Config, GridBlock_, Partition_>, config,
                                activePartitionNeighborBlockCount_, grid_[0], partition_[GetRollIndex()],
                                currentDt, currentFrameIndex_, auxiliaryBuffer_.maxVelocityNorm_);

							cudaCheckError(cudaMemcpyAsync(&maxVelocityNorm, auxiliaryBuffer_.maxVelocityNorm_,
														   sizeof(float), cudaMemcpyDeviceToHost,
														   cuContext.GetComputeStream()));
							maxVelocityNorm = std::sqrt(maxVelocityNorm);
							deviceTimer.RecordTimestamp();
							cuContext.SyncStream<CudaUtil::StreamIndex::kCompute>();
							auto elapsedTime = deviceTimer.GetElapsedTime();

                            if (std::isinf(maxVelocityNorm))
                                {
                                    spdlog::error("Velocity reached infinity. Exiting simulation.");
                                    goto simulate_end;
                                }

                             
                            nextDt = min(dt_, static_cast<double>(std::chrono::duration<double>(nextFrameTimestamp - currentTimestamp_).count()));

                            if (maxVelocityNorm > 0.0)
                                {
                                    nextDt = min(nextDt, config.GetDx() * config.GetCfl() / maxVelocityNorm);
                                }

							spdlog::info(
								"Frame: {}/{}, Maximum velocity is {}. Current Dt: {}. Elapsed "
								"time: {}ms.",
								currentFrameIndex_, config.GetTotalSimulatedFrameCount(), maxVelocityNorm, currentDt,
								elapsedTime);
						}

                        grid_[1].Reset(config, activePartitionNeighborBlockCount_);

						deviceTimer.RecordTimestamp();

                        if(collectConservationMetric)
                        {
                            for (int i = 0; i < GetModelCount(); ++i)
                                {
                                    cudaMemsetAsync(conservationMetric, 0, sizeof(float) * 12, cuContext.GetComputeStream());
                                    std::visit(
                                        [&](auto&& buffer) -> void
                                        {
                                        cuContext.LaunchCompute(
                                            {activePartitionBlockCount_, 128},
                                            CollectConservationMetric<Config,
                                                  std::decay_t<decltype(buffer)>, Partition_, GridBlock_>,
											buffer, std::get<std::decay_t<decltype(buffer)>>(
                                                particleBuffer_[GetNextRollIndex()][i]),
                                            partition_[GetNextRollIndex()], partition_[GetRollIndex()],
                                            grid_[0], conservationMetric);
                                        },
                                        particleBuffer_[GetRollIndex()][i]);

                                cudaCheckError(cudaMemcpyAsync(&conservationMetricHost[0], conservationMetric, sizeof(float) * 3 * 4, cudaMemcpyDeviceToHost, cuContext.GetComputeStream()));
                                cuContext.SyncStream<CudaUtil::StreamIndex::kCompute>();

                                spdlog::default_logger()->log(spdlog::level::off, "Frame: {}, Conservation Metric Info:\n"
                                             "\tLagrangian Linear Momentum: ({}, {}, {}), Norm: {}\n"
                                             "\tLagrangian Angular Momentum: ({}, {}, {}), Norm: {}\n", currentFrameIndex_,
                                             conservationMetricHost[2][0], conservationMetricHost[2][1], conservationMetricHost[2][2], conservationMetricHost[2].Norm(),
                                             conservationMetricHost[3][0], conservationMetricHost[3][1], conservationMetricHost[3][2], conservationMetricHost[3].Norm());

                                float outputBuffer[1 + 4 * 3];
                                outputBuffer[0] = static_cast<float>(std::chrono::duration<double>(currentTimestamp_).count());
                                memcpy(&outputBuffer[1], &conservationMetricHost[0], sizeof(float) * 12);

                                conservationMetricOutputFile[i].write(reinterpret_cast<char*>(outputBuffer), 13 * sizeof(float));
                                }
                        }


						// G2P2G
                        deviceTimer.RecordTimestamp();
						for (int i = 0; i < GetModelCount(); ++i)
							{
								std::visit(
									[&](auto&& buffer) -> void
									{
										cudaCheckError(cudaMemsetAsync(
											buffer.cellParticleCount_, 0,
											sizeof(int) * activePartitionExteriorBlockCount_ * config.GetBlockVolume(),
											cuContext.GetComputeStream()));
									},
									particleBuffer_[GetNextRollIndex()][i]);


                                bool freezed = true;
                                double elapsedTime = static_cast<double>(std::chrono::duration<double>(currentTimestamp_ - simulationStartTimestamp_).count());
                                // std::cout << "Elapsed time: " << elapsedTime << "\n";
                                std::visit([&](auto&& model) -> void
                                           {
                                            freezed = model.IsFreezed(elapsedTime);
                                           }, (*models_)[i]);
                                // std::cout << "Model " << i << " freeze status: " << freezed << std::endl;


								std::visit(
									[&](auto&& buffer, auto&& material) -> void
									{
                                        cuContext.LaunchCompute(
                                            {activePartitionBlockCount_, 128},
                                            G2P2G<Config,
                                                  std::decay_t<decltype(material)>::kConstitutiveModel_,
                                                  std::decay_t<decltype(buffer)>, Partition_, GridBlock_>,
											currentFrameIndex_,
                                            currentDt, nextDt, freezed,  material, buffer,
                                            std::get<std::decay_t<decltype(buffer)>>(
                                                particleBuffer_[GetNextRollIndex()][i]),
                                            partition_[GetNextRollIndex()], partition_[GetRollIndex()],
                                            grid_[0], grid_[1]);
									},
									particleBuffer_[GetRollIndex()][i], particleMaterial_[i]);
							}
						deviceTimer.RecordTimestamp();
						cuContext.SyncStream<CudaUtil::StreamIndex::kCompute>();

                        benchmark->AccumulateBenchmarkEpoch("G2P2G (Cuda)", std::chrono::duration<double>(deviceTimer.GetElapsedTime() / 1000.0));

						spdlog::info("G2P2G Finished. Elapsed time: {}ms.", deviceTimer.GetElapsedTime());
						//TODO:  Check for partition resize

						//-----------------------------Preparing Partition & Particle Buffer for Next Step----------------------------

                        {

                            deviceTimer.RecordTimestamp();
                            for (int i = 0; i < GetModelCount(); ++i)
                                {
                                    std::visit(
                                        [&](auto&& buffer) -> void
                                        {
                                            cudaCheckError(cudaMemsetAsync(
                                                buffer.particleBucketSize_, 0,
                                                sizeof(uint32_t) * (activePartitionExteriorBlockCount_ + 1),
                                                cuContext.GetComputeStream()));
                                            cuContext.LaunchCompute(
                                                {activePartitionExteriorBlockCount_, config.GetBlockVolume()},
                                                ParticleCellBucketToBlock<
                                                    Config, typename std::decay_t<decltype(buffer)>::Particle_>,
                                                config, buffer);
                                        },
                                        particleBuffer_[GetNextRollIndex()][i]);
                                }

                            auto activeBlockMark = auxiliaryBuffer_.activeBlockMark_;
                            auto activeParticleBlockMarkSource = auxiliaryBuffer_.activeParticleBlockMarkSource_;
                            auto activeParticleBlockMarkDestination =
                                auxiliaryBuffer_.activeParticleBlockMarkDestination_;

                            cudaCheckError(cudaMemsetAsync(activeBlockMark, 0,
                                                           sizeof(uint32_t) * activePartitionNeighborBlockCount_,
                                                           cuContext.GetComputeStream()));

                            cuContext.LaunchCompute({NextNearestMultipleOf<128>(activePartitionNeighborBlockCount_ *
                                                                                config.GetBlockVolume()) /
                                                         128,
                                                     128},
                                                    MarkActiveGridBlocks<Config, GridBlock_>, config,
                                                    activePartitionNeighborBlockCount_, grid_[1], activeBlockMark);

                            cudaCheckError(
                                cudaMemsetAsync(activeParticleBlockMarkSource, 0,
                                                sizeof(uint32_t) * (activePartitionExteriorBlockCount_ + 1),
                                                cuContext.GetComputeStream()));

                            for (int i = 0; i < GetModelCount(); ++i)
                                {
                                    std::visit(
                                        [&](auto&& buffer) -> void
                                        {
                                            cuContext.LaunchCompute(
                                                {activePartitionExteriorBlockCount_ / 128 + 1, 128},
                                                MarkActiveParticleBlocks<Config>, config,
                                                activePartitionExteriorBlockCount_ + 1, buffer.particleBucketSize_,
                                                activeParticleBlockMarkSource);
                                        },
                                        particleBuffer_[GetNextRollIndex<1>()][i]);
                                }

                            ExclusiveScan(activePartitionExteriorBlockCount_ + 1, activeParticleBlockMarkSource,
                                          activeParticleBlockMarkDestination, cuContext.GetComputeStream());

                            cudaCheckError(
                                cudaMemcpyAsync(reinterpret_cast<void*>(partition_[GetNextRollIndex()].count_),
                                                reinterpret_cast<void*>(activeParticleBlockMarkDestination +
                                                                        activePartitionExteriorBlockCount_),
                                                sizeof(uint32_t), cudaMemcpyDefault, cuContext.GetComputeStream()));

                            cudaCheckError(
                                cudaMemcpyAsync(reinterpret_cast<void*>(&activePartitionBlockCount_),
                                                reinterpret_cast<void*>(activeParticleBlockMarkDestination +
                                                                        activePartitionExteriorBlockCount_),
                                                sizeof(uint32_t), cudaMemcpyDefault, cuContext.GetComputeStream()));

                            cuContext.LaunchCompute(
                                {NextNearestMultipleOf<256>(activePartitionExteriorBlockCount_) / 256, 256},
                                InverseExclusiveScan<uint32_t>, activePartitionExteriorBlockCount_,
                                activeParticleBlockMarkDestination, activeParticleBlockMarkSource);

                            partition_[GetNextRollIndex()].ResetPartitionTable(cuContext.GetComputeStream());

                            cuContext.SyncStream<CudaUtil::StreamIndex::kCompute>();

                            if (activePartitionBlockCount_ >= config.GetMaxActiveBlockCount())
                                {
                                    spdlog::error("Too much active blocks: {}", activePartitionBlockCount_);
                                    spdlog::dump_backtrace();
                                    std::abort();
                                }

                            cuContext.LaunchCompute(
                                {NextNearestMultipleOf<128>(activePartitionBlockCount_) / 128, 128},
                                UpdatePartition<Partition_>, activePartitionBlockCount_,
                                activeParticleBlockMarkSource, partition_[GetRollIndex()],
                                partition_[GetNextRollIndex()]);

                            for (int i = 0; i < GetModelCount(); ++i)
                                {
                                    std::visit(
                                        [&](auto&& buffer) -> void
                                        {
                                            auto& nextBuffer = std::get<std::decay_t<decltype(buffer)>>(
                                                particleBuffer_[GetRollIndex()][i]);
                                            cuContext.LaunchCompute(
                                                {activePartitionBlockCount_, 128},
                                                UpdateBucket<Config, std::decay_t<decltype(buffer)>>, config,
                                                activePartitionBlockCount_, activeParticleBlockMarkSource, buffer,
                                                nextBuffer);
                                        },
                                        particleBuffer_[GetNextRollIndex()][i]);
                                }

                            auto* binCount = auxiliaryBuffer_.binCount_;

                            for (int i = 0; i < GetModelCount(); ++i)
                                {
                                    std::visit(
                                        [&](auto&& buffer) -> void
                                        {
                                            cuContext.LaunchCompute({activePartitionBlockCount_ / 128 + 1, 128},
                                                                    ComputeBinCount<Config>, config,
                                                                    activePartitionBlockCount_,
                                                                    buffer.particleBucketSize_, binCount);

                                            ExclusiveScan(activePartitionBlockCount_ + 1, binCount,
                                                          buffer.particleBinOffset_, cuContext.GetComputeStream());

                                            cuContext.SyncStream<CudaUtil::StreamIndex::kCompute>();
                                        },
                                        particleBuffer_[GetRollIndex()][i]);
                                }

                            deviceTimer.RecordTimestamp();

                            spdlog::info("Update partition & buffer for next step finished. Elapsed time: {}ms.",
                                         deviceTimer.GetElapsedTime());

                            deviceTimer.RecordTimestamp();

                            const uint32_t previousActivePartitionNeighborBlockCount =
                                activePartitionNeighborBlockCount_;

                            cuContext.LaunchCompute(
                                {NextNearestMultipleOf<128>(activePartitionBlockCount_) / 128, 128},
                                ActivateBlocks<MPMActivateBlockPolicy::Neighbor, Config, Partition_>, config,
                                activePartitionBlockCount_, partition_[GetNextRollIndex()]);

                            cudaCheckError(cudaMemcpyAsync(&activePartitionNeighborBlockCount_,
                                                           partition_[GetNextRollIndex()].count_, sizeof(uint32_t),
                                                           cudaMemcpyDeviceToHost, cuContext.GetComputeStream()));

                            cuContext.SyncStream<CudaUtil::StreamIndex::kCompute>();

                            if (activePartitionNeighborBlockCount_ >= config.GetMaxActiveBlockCount())
                                {
                                    spdlog::error(
                                        "Too much active neighbor blocks. Neighbor block count: {}. Exiting.",
                                        activePartitionNeighborBlockCount_);
                                    spdlog::dump_backtrace();
                                    std::abort();
                                }

                            //TODO: resize grid if necessary

                            deviceTimer.RecordTimestamp();
                            benchmark->AccumulateBenchmarkEpoch("Activate Neighbor", std::chrono::duration<double>(deviceTimer.GetElapsedTime() / 1000.0));

                            grid_[0].Reset(config, activePartitionExteriorBlockCount_);

                            cuContext.LaunchCompute(
                                {previousActivePartitionNeighborBlockCount, 2 * config.GetBlockVolume()},
                                CopySelectedGridBlocks<MPMGridBlockCoordinate, Partition_, Grid_>,
                                partition_[GetRollIndex()].activeKeys_, partition_[GetNextRollIndex()],
                                activeBlockMark, grid_[1], grid_[0]);

                            cuContext.SyncStream<CudaUtil::StreamIndex::kCompute>();

                            deviceTimer.RecordTimestamp();
                            spdlog::info("Copy grid data to next step finished. Elapsed time: {}ms.",
                                         deviceTimer.GetElapsedTime());
                            benchmark->AccumulateBenchmarkEpoch("Copy Grid Data", std::chrono::duration<double>(deviceTimer.GetElapsedTime() / 1000.0));

                            //TODO: Resize grid

                            deviceTimer.RecordTimestamp();
                            cuContext.LaunchCompute(
                                {NextNearestMultipleOf<128>(activePartitionBlockCount_) / 128, 128},
                                ActivateBlocks<MPMActivateBlockPolicy::Exterior, Config, Partition_>, config,
                                activePartitionBlockCount_, partition_[GetNextRollIndex()]);

                            cudaCheckError(cudaMemcpyAsync(&activePartitionExteriorBlockCount_,
                                                           partition_[GetNextRollIndex()].count_, sizeof(uint32_t),
                                                           cudaMemcpyDefault, cuContext.GetComputeStream()));

                            cuContext.SyncStream<CudaUtil::StreamIndex::kCompute>();

                            deviceTimer.RecordTimestamp();

                            benchmark->AccumulateBenchmarkEpoch("Activate Exterior", std::chrono::duration<double>(deviceTimer.GetElapsedTime() / 1000.0));
                            spdlog::info("Activate exterior blocks for next step finished. Elapsed time: {}ms.",
                                         deviceTimer.GetElapsedTime());

                            if (activePartitionExteriorBlockCount_ >= config.GetMaxActiveBlockCount())
                                {
                                    spdlog::error(
                                        "Too much active exterior blocks. Exterior block count: {}. Exiting.",
                                        activePartitionExteriorBlockCount_);
                                    spdlog::dump_backtrace();
                                    std::abort();
                                }
                            IterateRollIndex();
                        }

                        config.UpdateConfig(static_cast<float>(currentDt), static_cast<int>(currentFrameIndex_));
						currentDt = nextDt;
						currentTimestamp_ += std::chrono::duration<double>(currentDt);

						spdlog::info("Active block count: {}. Neighbor block count: {}. Exterior block count:{}",
									 activePartitionBlockCount_, activePartitionNeighborBlockCount_,
									 activePartitionExteriorBlockCount_);
					}
				spdlog::info("Step : {}", step_);
				// std::cout << "Finished frame: " << currentFrameIndex_ << std::endl;

                if(!profileMode)
                {
                    ExportParticleToModel(config, exportRootPath, static_cast<double>(std::chrono::duration<double>(currentTimestamp_ - simulationStartTimestamp_).count()), exportParticlePositionAsBinary);
                }
			}
	simulate_end:
		(void)nullptr;

		hostTimer.RecordTimestamp();
        spdlog::set_level(spdlog::level::info);

        benchmark->AccumulateBenchmarkEpoch("Total", std::chrono::duration<double>(hostTimer.GetElapsedTime() / 1000.0));

        benchmark->PrintStatistic();

        if(!profileMode)
        {
            exporter_.Terminate();
        }

        if(collectConservationMetric)
        {
            cudaFree(conservationMetric);
            for(int i = 0; i < GetModelCount(); ++i)
            {
                conservationMetricOutputFile[i].close();
            }
        }
        exit(0);
	}

   private:
	typedef MPMGridAttribute<float, float, float, float> GridAttribute_;
	typedef MPMGridBlock<GridAttribute_, 2, kGridBlockSize_, kGridBlockSize_, kGridBlockSize_> GridBlock_;
	typedef MPMGrid<GridBlock_> Grid_;

	typedef MPMParticleBuffer<typename MPMMaterial<MPMConstitutiveModel::kLinear>::Particle_> LinearParticleBuffer_;
	typedef MPMParticleBuffer<typename MPMMaterial<MPMConstitutiveModel::kFixedCorotated>::Particle_>
		FixedCorotatedParticleBuffer_;
	typedef MPMParticleBuffer<typename MPMMaterial<MPMConstitutiveModel::kDruckerPragerStvkhencky>::Particle_>
		DruckerPragerStvkhenckyParticleBuffer_;
	typedef MPMParticleBuffer<typename MPMMaterial<MPMConstitutiveModel::kFluid>::Particle_>
		FluidParticleBuffer_;
	typedef MPMParticleBuffer<typename MPMMaterial<MPMConstitutiveModel::kNonAssociatedCamClay>::Particle_>
		NonAssociatedCamClayParticleBuffer_;
	typedef MPMParticleBuffer<typename MPMMaterial<MPMConstitutiveModel::kVonMises>::Particle_>
		VonMisesParticleBuffer_;


	typedef std::variant<typename MPMMaterial<MPMConstitutiveModel::kLinear>::Particle_,
						 typename MPMMaterial<MPMConstitutiveModel::kFixedCorotated>::Particle_,
                         typename MPMMaterial<MPMConstitutiveModel::kDruckerPragerStvkhencky>::Particle_,
                         typename MPMMaterial<MPMConstitutiveModel::kFluid>::Particle_,
						 typename MPMMaterial<MPMConstitutiveModel::kVonMises>::Particle_,
						 typename MPMMaterial<MPMConstitutiveModel::kNonAssociatedCamClay>::Particle_
                         >
		Particle_;

    typedef ParticleMaterial ParticleMaterial_;

	typedef std::variant<LinearParticleBuffer_, FixedCorotatedParticleBuffer_, DruckerPragerStvkhenckyParticleBuffer_, FluidParticleBuffer_, VonMisesParticleBuffer_, NonAssociatedCamClayParticleBuffer_> ParticleBuffer_;

	typedef MPMGridConfig MPMGridConfig_;
	typedef typename MPMGridConfig_::Domain_ Domain_;
	typedef typename Domain_::DomainRange_ DomainRange_;
	typedef typename Domain_::DomainOffset_ DomainOffset_;
	typedef MPMPartition<MPMGridConfig_> Partition_;

	// All necesssary data structures
	Grid_ grid_[2];

	std::array<std::vector<ParticleBuffer_>, kBufferCount_> particleBuffer_;
	std::vector<ParticleMaterial_> particleMaterial_;

	Partition_ partition_[2];

	std::vector<size_t> particleCount_;
	std::vector<size_t> velocityCount_;

	std::vector<Vector<float, kNDim_>*> deviceParticlePosition_;
	std::vector<Vector<float, kNDim_>*> deviceParticleVelocity_;
    std::vector<MPMModelVariant>* models_ = nullptr;
	uint32_t rollIndex_ = 0;
	uint32_t activePartitionBlockCount_ = {};
	uint32_t activePartitionNeighborBlockCount_ = {};
	uint32_t activePartitionExteriorBlockCount_ = {};
	float dt_ = 0.0;
	uint32_t simulateFrameCount_ = 0;
	uint32_t currentFrameIndex_ = 0;
	uint32_t step_ = 0;

    std::chrono::duration<double> currentTimestamp_;

	mpm::MPMParticleExporter exporter_;
};

}  // namespace mpm
