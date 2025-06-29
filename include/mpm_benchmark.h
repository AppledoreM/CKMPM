#include <chrono>
#include <map>
#include <string>
#include <vector>

#include "spdlog/spdlog.h"



namespace mpm
{
    class MPMBenchmark
    {
    public:

        struct MPMBenchmarkStatistic
        {
            std::string name_;
            std::chrono::duration<double> duration_;
            uint32_t count_ = 0;

            MPMBenchmarkStatistic(const std::string& name = "", std::chrono::duration<double> duration = std::chrono::duration<double>(0), uint32_t count = 0) : name_(name), duration_(duration), count_(count)
            {


            }

            ~MPMBenchmarkStatistic() = default;

            auto AccumulateEpoch(std::chrono::duration<double> duration) -> void
            {
                duration_ += duration;
                ++count_;
            }

            auto PrintStatistic() const -> void
            {
                spdlog::info("MPMBenchmark task {} | Total duration: {}s | Epoch count: {} | Average duration: {}s.", name_, duration_.count(), count_, duration_.count() / count_);
            }
        };

        static auto GetInstance() -> MPMBenchmark*
        {
            static MPMBenchmark instance;
            return &instance;
        }


        ~MPMBenchmark() = default;
        MPMBenchmark(const MPMBenchmark&) = delete;
        MPMBenchmark& operator=(const MPMBenchmark) = delete;


        auto AddBenchmark(const std::string& name) -> void
        {
            benchmark_[name] = MPMBenchmarkStatistic(name);
        }

        auto AccumulateBenchmarkEpoch(const std::string& name, std::chrono::duration<double> duration) -> void
        {
            benchmark_[name].AccumulateEpoch(duration);
        }

        auto PrintStatistic() const ->  void
        {
            for(auto&& [_, benchmarkStatistic] : benchmark_)
            {
                benchmarkStatistic.PrintStatistic();
            }
        }

        
    private:
        MPMBenchmark() = default;

        std::map<std::string, MPMBenchmarkStatistic> benchmark_;

    };

}


