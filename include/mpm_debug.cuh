#pragma once

namespace mpm
{

template <typename Derived>
class IMPMDebugBase
{
   public:
	template <typename Config>
	auto PrintDebugInformation(const Config& config) const -> void
	{
		return (static_cast<const Derived*>(this))->PrintDebugInformationImpl(config);
	}
};

}  // namespace mpm
