#pragma once
#include <any>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>

#include <nlohmann/json.hpp>

#include "Partio.h"
#include "PartioAttribute.h"
#include "mpm_config.h"
#include "mpm_meta.h"
#include "mpm_particle.cuh"
#include "mpm_thread_pool.h"

namespace mpm
{

class MPMImportParser
{
   public:
	virtual ~MPMImportParser() = default;
};

class MPMExportFormatter
{
   public:
	virtual ~MPMExportFormatter() = default;
	[[nodiscard]] virtual auto FormatData() const -> std::vector<char> = 0;
};

template <typename T, size_t kNDim>
struct MPMParticleIoAttribute
{
	typedef T Data_;
	constexpr static size_t kNDim_ = kNDim;
	std::string attributeName_;
	std::vector<std::array<T, kNDim>> attribute_;

	MPMParticleIoAttribute() = default;
	MPMParticleIoAttribute(const MPMParticleIoAttribute& rhs)
		: attributeName_(rhs.attributeName_), attribute_(rhs.attribute_)
	{
	}

	MPMParticleIoAttribute(MPMParticleIoAttribute&& rhs)
		: attributeName_(rhs.attributeName_), attribute_(rhs.attribute_)
	{
	}
};

template <typename... ParticleIoAttribute>
struct MPMParticleIoData
{

	MPMParticleIoData(ParticleIoAttribute&&... attribute)
		: particleData_(std::make_tuple<ParticleIoAttribute...>(std::move(attribute)...))
	{
	}

	inline auto AddToParticleIO(Partio::ParticlesDataMutable* particleIO) -> void
	{
		meta::ConstexprLoop<0, sizeof...(ParticleIoAttribute)>(
			[&](auto attributeIndexWrapper) -> void
			{
				constexpr int attributeIndex = meta::ConstexprLoopIndex(attributeIndexWrapper);
				auto& exportAttribute = std::get<attributeIndex>(particleData_);
				Partio::ParticleAttribute attribute =
					particleIO->addAttribute(exportAttribute.attributeName_.c_str(), Partio::VECTOR,
											 std::decay_t<decltype(exportAttribute)>::kNDim_);
				particleIO->addParticles(exportAttribute.attribute_.size());
				for (int i = 0; i < static_cast<int>(exportAttribute.attribute_.size()); ++i)
					{
						float* value = particleIO->dataWrite<typename std::decay_t<decltype(exportAttribute)>::Data_>(
							attribute, i);
						for (int j = 0; j < std::decay_t<decltype(exportAttribute)>::kNDim_; ++j)
							{
								value[j] = exportAttribute.attribute_[i][j];
							}
					}
			});
	}

	std::tuple<ParticleIoAttribute...> particleData_;
};

class MPMParticleExporter
{
   public:
	MPMParticleExporter() {}

	template <typename ParticleIoData>
	inline auto ExportDataAsync(const std::string& name, ParticleIoData&& data) -> void
	{
		if (!isActive_)
			return;
		std::unique_lock dataLock(dataMutex_);
		if (dataTable_[name])
			throw std::runtime_error("File name " + name + " already existed.!");
		dataTable_[name] = Partio::create();
		data.AddToParticleIO(dataTable_[name]);
		if (dataTable_.size() >= 1)
			exportCv_.notify_one();
	}

	inline auto Run() -> void
	{
		isActive_ = true;
		exportThread_ = std::thread(&MPMParticleExporter::RunImpl, this);
	}

	inline auto Terminate() -> void
	{
		isActive_ = false;
		exportCv_.notify_one();
		if (exportThread_.joinable())
			exportThread_.join();
	}

	~MPMParticleExporter() { Terminate(); }

   private:
	inline auto RunImpl() -> void
	{
		auto exportFunc = [&](auto&& dataTable) -> void
		{
			for (auto&& [filename, data] : dataTable)
				{
					Partio::write(filename.c_str(), *data);
					data->release();
				}
		};
		while (isActive_)
			{
				std::unique_lock exportLock(dataMutex_);

				exportCv_.wait(exportLock, [&]() -> bool { return !dataTable_.empty(); });

				std::map<std::string, Partio::ParticlesDataMutable*> tmpData;
				std::swap(tmpData, dataTable_);
				exportLock.unlock();

				exportFunc(tmpData);
			}

		std::unique_lock exportLock(dataMutex_);
		if (!dataTable_.empty())
			{
				exportFunc(dataTable_);
			}
	}

   private:
	std::fstream exportFile_;
	std::map<std::string, Partio::ParticlesDataMutable*> dataTable_;
	std::mutex dataMutex_;
	std::thread exportThread_;
	std::condition_variable exportCv_;
	bool isActive_ = false;
};

// template <MPMConfigType Config>
// class MPMConfigImporter;

// template <MPMDx kDx>
// class MPMConfigImporter<MPMDefaultStaticConfig<kDx>>
// {
//    public:
// 	typedef MPMDefaultStaticConfig<kDx> Config_;

// 	MPMConfigImporter() = default;
// 	~MPMConfigImporter() = default;

// 	auto ImportConfig(const std::string& filename) -> Config_
// 	{
// 		using json = nlohmann::json;

// 		std::ifstream configFile(filename);

// 		if (!configFile.is_open())
// 			{
// 				throw std::runtime_error("Failed to find configuration file " + filename + "\n");
// 			}

// 		json configData = json::parse(configFile);
// 		const auto& experimentCondition = configData["experiment_condition"];

// 		return Config_{experimentCondition["dt"].template get<float>(),
// 					   experimentCondition["fps"].template get<uint32_t>(),
// 					   experimentCondition["cfl"].template get<float>(),
// 					   experimentCondition["total_simulated_time"].template get<float>()};
// 	}
// };

}  // namespace mpm
