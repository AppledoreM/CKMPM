#pragma once
#include "base.h"
#include "cuda_util.cuh"
#include "mpm_meta.h"
#include "mpm_particle.cuh"

namespace mpm
{

enum class MPMConstitutiveModel
{
	kFixedCorotated = 0,
	kLinear = 1,
    kDruckerPragerStvkhencky = 2,
    kFluid = 3,
    kVonMises = 4,
    kNonAssociatedCamClay = 5
};

template <MPMConstitutiveModel Model>
class MPMMaterial;

template <>
class MPMMaterial<MPMConstitutiveModel::kLinear>
{
   public:
	constexpr static MPMConstitutiveModel kConstitutiveModel_ = MPMConstitutiveModel::kLinear;
	typedef MPMParticle<0, float, float, float, 
            float, float, float, 
            float, float, float,
            float, float, float,
            float, float, float, 
            float, float, float,
            float, float, float>
		Particle_;
	struct LinearMaterialParameter
	{
		float lambda_;
		float mu_;

		constexpr LinearMaterialParameter() = default;

		constexpr MPM_HOST_DEV_FUNC LinearMaterialParameter(float lambda, float mu) : lambda_(lambda), mu_(mu)
		{
		}

		constexpr MPM_HOST_DEV_FUNC LinearMaterialParameter(const LinearMaterialParameter& rhs)
			: lambda_(rhs.lambda_), mu_(rhs.mu_)
		{
		}
	};

    typedef LinearMaterialParameter MaterialParameter_;

	constexpr MPMMaterial() = default;
	constexpr MPM_HOST_DEV_FUNC MPMMaterial(MaterialParameter_ parameter) : parameter_(parameter) {}
	constexpr MPMMaterial(const MPMMaterial&) = default;

	MaterialParameter_ parameter_ = {};
};

template <>
class MPMMaterial<MPMConstitutiveModel::kFixedCorotated>
{
   public:
	struct FixedCorotatedMaterialParameter
	{
		float lambda_;
		float mu_;

		constexpr FixedCorotatedMaterialParameter() = default;

		constexpr MPM_HOST_DEV_FUNC FixedCorotatedMaterialParameter(float lambda, float mu) : lambda_(lambda), mu_(mu)
		{
		}

		constexpr MPM_HOST_DEV_FUNC FixedCorotatedMaterialParameter(const FixedCorotatedMaterialParameter& rhs)
			: lambda_(rhs.lambda_), mu_(rhs.mu_)
		{
		}
	};

	constexpr static MPMConstitutiveModel kConstitutiveModel_ = MPMConstitutiveModel::kFixedCorotated;
	typedef MPMParticle<1, float, float, float, 
                        float, float, float, 
                        float, float, float, 
                        float, float, float>
		Particle_;
	typedef FixedCorotatedMaterialParameter MaterialParameter_;

	constexpr MPMMaterial() = default;
	constexpr MPM_HOST_DEV_FUNC MPMMaterial(const MaterialParameter_& parameter) : parameter_(parameter) {}
	constexpr MPM_HOST_DEV_FUNC MPMMaterial(const MPMMaterial& rhs) : parameter_(rhs.parameter_) {}

	MaterialParameter_ parameter_ = {};
};


template<>
class MPMMaterial<MPMConstitutiveModel::kDruckerPragerStvkhencky>
{
public:
    constexpr static MPMConstitutiveModel kConstitutiveModel_ = MPMConstitutiveModel::kDruckerPragerStvkhencky;
	typedef MPMParticle<2, float, float, float, 
                        float, float, float,
                        float, float, float,
                        float, float, float, float>
		Particle_;

	struct DruckerPragerStvkhenckyMaterialParameter
	{
        static constexpr float kLogJp0 = 0.f;
		float lambda_;
		float mu_;
        

		constexpr DruckerPragerStvkhenckyMaterialParameter() = default;

		constexpr MPM_HOST_DEV_FUNC DruckerPragerStvkhenckyMaterialParameter(float lambda, float mu) : lambda_(lambda), mu_(mu)
		{
		}

		constexpr MPM_HOST_DEV_FUNC DruckerPragerStvkhenckyMaterialParameter(const DruckerPragerStvkhenckyMaterialParameter& rhs)
			: lambda_(rhs.lambda_), mu_(rhs.mu_)
		{
		}
	};

    typedef DruckerPragerStvkhenckyMaterialParameter MaterialParameter_;

	constexpr MPMMaterial() = default;
	constexpr MPM_HOST_DEV_FUNC MPMMaterial(const MaterialParameter_& parameter) : parameter_(parameter) {}
	constexpr MPM_HOST_DEV_FUNC MPMMaterial(const MPMMaterial& rhs) : parameter_(rhs.parameter_) {}

	MaterialParameter_ parameter_ = {};
};

template<>
class MPMMaterial<MPMConstitutiveModel::kFluid>
{
public:
    constexpr static MPMConstitutiveModel kConstitutiveModel_ = MPMConstitutiveModel::kFluid;
	typedef MPMParticle<3, float, float, float, float>
		Particle_;

	struct FluidMaterialParameter
	{
		float bulk_ = 0.f;
        float gamma_ = 0.f;
		float viscosity_ = 0.f;
        float J0_ = 0.f;

		constexpr FluidMaterialParameter() = default;

		constexpr MPM_HOST_DEV_FUNC FluidMaterialParameter(float bulk, float gamma, float viscosity, float J0 = 1.0f) : bulk_(bulk), gamma_(gamma), viscosity_(viscosity), J0_(J0)
		{

		}

		constexpr MPM_HOST_DEV_FUNC FluidMaterialParameter(const FluidMaterialParameter& rhs) : bulk_(rhs.bulk_), gamma_(rhs.gamma_), viscosity_(rhs.viscosity_), J0_(rhs.J0_)	
        {

		}
	};

    typedef FluidMaterialParameter MaterialParameter_;

	constexpr MPMMaterial() = default;
	constexpr MPM_HOST_DEV_FUNC MPMMaterial(const MaterialParameter_& parameter) : parameter_(parameter) {}
	constexpr MPM_HOST_DEV_FUNC MPMMaterial(const MPMMaterial& rhs) : parameter_(rhs.parameter_) {}

    constexpr MPM_HOST_DEV_FUNC auto GetJ0() const -> float { return parameter_.J0_; }

	MaterialParameter_ parameter_ = {};
};



template<>
class MPMMaterial<MPMConstitutiveModel::kNonAssociatedCamClay>
{
public:
    constexpr static MPMConstitutiveModel kConstitutiveModel_ = MPMConstitutiveModel::kNonAssociatedCamClay;
	typedef MPMParticle<4, float, float, float, float,
                        float, float, float, float, 
                        float, float, float, float, 
                        float> Particle_;

	struct NonAssociatedCamClayMaterialParameter
	{
        float mu_ = 0;
        float lambda_ = 0;
        float alpha0_ = 0.f;
        float beta_ = 0.f;
        float hardeningFactor_ = 0.f;
        float m_ = 0.f;
        bool hardeningOn_ = true;


		constexpr NonAssociatedCamClayMaterialParameter() = default;

		constexpr MPM_HOST_DEV_FUNC NonAssociatedCamClayMaterialParameter(float mu, float lambda, float alpha0, 
                                                                         float beta, float hardeningFactor, float m, 
                                                                         bool hardeningOn) : mu_(mu), lambda_(lambda), alpha0_(alpha0), beta_(beta), hardeningFactor_(hardeningFactor), m_(m), hardeningOn_(hardeningOn)
		{

		}

		constexpr MPM_HOST_DEV_FUNC NonAssociatedCamClayMaterialParameter(const NonAssociatedCamClayMaterialParameter& rhs) : 
            mu_(rhs.mu_), lambda_(rhs.lambda_), alpha0_(rhs.alpha0_), beta_(rhs.beta_), hardeningFactor_(rhs.hardeningFactor_), m_(rhs.m_), hardeningOn_(rhs.hardeningOn_)
        {

		}
	};

    typedef NonAssociatedCamClayMaterialParameter MaterialParameter_;

	constexpr MPMMaterial() = default;
	constexpr MPM_HOST_DEV_FUNC MPMMaterial(const MaterialParameter_& parameter) : parameter_(parameter) {}
	constexpr MPM_HOST_DEV_FUNC MPMMaterial(const MPMMaterial& rhs) : parameter_(rhs.parameter_) {}

	MaterialParameter_ parameter_ = {};
};


template<>
class MPMMaterial<MPMConstitutiveModel::kVonMises>
{
public:
    constexpr static MPMConstitutiveModel kConstitutiveModel_ = MPMConstitutiveModel::kVonMises;
	typedef MPMParticle<5, float, float, float, float,
                        float, float, float, float, 
                        float, float, float, float, 
                        float> Particle_;

	struct VonMisesMaterialParameter
	{
		float mu_ = 0.f;
        float lambda_ = 0.f;
		float yieldStress_ = 0.f;

		constexpr VonMisesMaterialParameter() = default;

		constexpr MPM_HOST_DEV_FUNC VonMisesMaterialParameter(float mu, float lambda, float yieldStress) : mu_(mu), lambda_(lambda), yieldStress_(yieldStress)
		{

		}

		constexpr MPM_HOST_DEV_FUNC VonMisesMaterialParameter(const VonMisesMaterialParameter& rhs) : mu_(rhs.mu_), lambda_(rhs.lambda_), yieldStress_(rhs.yieldStress_)	
        {

		}
	};

    typedef VonMisesMaterialParameter MaterialParameter_;

	constexpr MPMMaterial() = default;
	constexpr MPM_HOST_DEV_FUNC MPMMaterial(const MaterialParameter_& parameter) : parameter_(parameter) {}
	constexpr MPM_HOST_DEV_FUNC MPMMaterial(const MPMMaterial& rhs) : parameter_(rhs.parameter_) {}

	MaterialParameter_ parameter_ = {};
};


//TODO: Add support for other materials
template <typename Scalar>
struct MPMStressComputationIntermediateBuffer
{
	Scalar volume_;
	Scalar mu_;
	Scalar lambda_;
	Matrix<Scalar, 3, 3>* PF = nullptr;
	Matrix<Scalar, 3, 3>* F = nullptr;
};

template <typename Scalar>
constexpr MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto ComputeLameParameters(const Scalar E,
																		const Scalar nu) -> Vector<Scalar, 2>
{
	auto lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
	auto mu = E / (2 * (1 + nu));
	return Vector<Scalar, 2>{lambda, mu};
}

template <MPMConstitutiveModel ConstitutiveModel, typename Scalar>
MPM_FORCE_INLINE MPM_HOST_DEV_FUNC auto ComputeStress(
	const Scalar volume, const Scalar mu, const Scalar lambda, Vector<Scalar, 9>& F,
	MPMStressComputationIntermediateBuffer<Scalar>& stressIntermediateBuffer) -> void;

template <MPMConstitutiveModel ConstitutiveModel, typename Scalar>
requires(MPMConstitutiveModel::kLinear == ConstitutiveModel) MPM_FORCE_INLINE MPM_HOST_DEV_FUNC
	auto ComputeStress(const Scalar volume, const Scalar mu, const Scalar lambda, const Matrix<float, 3, 3>& F,
					   Matrix<float, 3, 3>& PF) -> void
{
    auto cauchyStrainTensor = 0.5f * (F + F.Transpose());
    PF = F.Determinant() * (lambda * cauchyStrainTensor.Trace() * Matrix<float, 3, 3>::Identity() + 2 * mu * cauchyStrainTensor);
}

template <MPMConstitutiveModel ConstitutiveModel, typename Scalar>
requires(MPMConstitutiveModel::kFixedCorotated == ConstitutiveModel) MPM_FORCE_INLINE MPM_DEV_FUNC
	auto ComputeEnergy(MPMStressComputationIntermediateBuffer<Scalar>& stressIntermediateBuffer) -> float
{
	const auto& F = *stressIntermediateBuffer.F;
	Matrix<float, 3, 3> U, V;
	Vector<float, 3> S;
	svd(F[0], F[1], F[2], F[3], F[4], F[5], F[6], F[7], F[8], U[0], U[1], U[2], U[3], U[4], U[5], U[6], U[7], U[8],
		S[0], S[1], S[2], V[0], V[1], V[2], V[3], V[4], V[5], V[6], V[7], V[8]);

	float J = S[0] * S[1] * S[2];
	float energy = 0.0;
	for (int i = 0; i < 3; ++i)
		energy += (S[i] - 1.f) * (S[i] - 1.f);
	energy = stressIntermediateBuffer.mu_ * energy + 0.5f * stressIntermediateBuffer.lambda_ * (J - 1) * (J - 1);

	return energy;
}

template <MPMConstitutiveModel ConstitutiveModel, typename Scalar>
requires(MPMConstitutiveModel::kFixedCorotated == ConstitutiveModel) MPM_FORCE_INLINE MPM_DEV_FUNC
	auto ComputeStress(const Scalar volume, const Scalar mu, const Scalar lambda, const Matrix<float, 3, 3>& F,
					   Matrix<float, 3, 3>& PF) -> void
{

	Matrix<float, 3, 3> U, V;
	Vector<float, 3> S;



	svd(F[0], F[1], F[2], F[3], F[4], F[5], F[6], F[7], F[8], U[0], U[1], U[2], U[3], U[4], U[5], U[6], U[7], U[8],
		S[0], S[1], S[2], V[0], V[1], V[2], V[3], V[4], V[5], V[6], V[7], V[8]);
	V = U;
	{
		const float J = S[0] * S[1] * S[2];
		const float scaled_mu = 2.0 * mu;
		const float scaled_lambda = lambda * J * (J - 1.f);
		const auto scale = scaled_mu * S * (S - 1.f) + scaled_lambda;

		meta::ConstexprLoop<0, 3>(
			[&](auto indexWrapper) -> void
			{
				constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
				V[index * 3 + 0] *= scale[0];
				V[index * 3 + 1] *= scale[1];
				V[index * 3 + 2] *= scale[2];
			});
	}

	PF[0] = V[0] * U[0] + V[1] * U[1] + V[2] * U[2];
	PF[1] = V[0] * U[3] + V[1] * U[4] + V[2] * U[5];
	PF[2] = V[0] * U[6] + V[1] * U[7] + V[2] * U[8];
	PF[3] = V[3] * U[0] + V[4] * U[1] + V[5] * U[2];
	PF[4] = V[3] * U[3] + V[4] * U[4] + V[5] * U[5];
	PF[5] = V[3] * U[6] + V[4] * U[7] + V[5] * U[8];
	PF[6] = V[6] * U[0] + V[7] * U[1] + V[8] * U[2];
	PF[7] = V[6] * U[3] + V[7] * U[4] + V[8] * U[5];
	PF[8] = V[6] * U[6] + V[7] * U[7] + V[8] * U[8];
}


template <MPMConstitutiveModel ConstitutiveModel, typename Scalar>
requires(MPMConstitutiveModel::kDruckerPragerStvkhencky == ConstitutiveModel) MPM_FORCE_INLINE MPM_DEV_FUNC
	auto ComputeStress(const Scalar volume, const Scalar mu, const Scalar lambda, Matrix<float, 3, 3>& F, float& logJp, 
					   Matrix<float, 3, 3>& PF) -> void
{

    constexpr float sandCohesion = 0;
    constexpr float sandBeta = 1.f;
    constexpr bool sandVolumeCorrection = true;
    constexpr float sandYs = 0.816496580927726f * 2.f * 0.5f / (3.f - 0.5f);

    float scaledMu = 2.0f * mu;

	Matrix<float, 3, 3> U, V;
    Vector<float, 3> epsilon;
    {
        Vector<float, 3> S;
        svd(F[0], F[1], F[2], F[3], F[4], F[5], F[6], F[7], F[8], U[0], U[1], U[2], U[3], U[4], U[5], U[6], U[7], U[8],
            S[0], S[1], S[2], V[0], V[1], V[2], V[3], V[4], V[5], V[6], V[7], V[8]);

        V = V.Transpose();
        meta::ConstexprLoop<0, 3>(
                                  [&](auto indexWrapper) -> void
                                  {
                                    constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
                                    epsilon[index] = logf(max(abs(S[index]), 1e-4)) - sandCohesion;
                                  }
                                 );
    }



    float epsilonSum = epsilon[0] + epsilon[1] + epsilon[2];
    float epsilonTrace = epsilonSum + logJp;

    Vector<float, 3> epsilonHat = epsilon - epsilonTrace / 3.0f;
    float epsilonHatNorm = epsilonHat.Norm();

    Vector<float, 3> newS;
    Matrix<float, 3, 3> newF;
    if(epsilonTrace >= 0.f)
    {
        newS = expf(sandCohesion);

        newF = newS[0] * U.MatrixMultiplication(V); 
        F = newF;
        
        if(sandVolumeCorrection)
        {
            logJp = sandBeta * epsilonSum + logJp;
        }
    }
    else if(mu != 0.f)
    {
        logJp = 0.f;
        float deltaGamma = epsilonHatNorm + (3.0f * lambda + scaledMu) / scaledMu * epsilonTrace * sandYs;
        Vector<float, 3> H;
        if(deltaGamma <= 0.f)
        {
            H = epsilon + sandCohesion;
        }
        else
        {
            H = epsilon - (deltaGamma / epsilonHatNorm) * epsilonHat + sandCohesion;
        }

        meta::ConstexprLoop<0, 3>(
                                  [&](auto indexWrapper) -> void
                                  {
                                  constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
                                  newS[index] = expf(H[index]);
                                  }
                                 );

        meta::ConstexprLoop<0, 3>(
                                  [&](auto indexWrapper) -> void
                                  {
                                  constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
                                  newF[3 * index] = newS[index] * V[3 * index];
                                  newF[3 * index + 1] = newS[index] * V[3 * index + 1];
                                  newF[3 * index + 2] = newS[index] * V[3 * index + 2];
                                  }
                                 );
        newF = U.MatrixMultiplication(newF);
        F = newF;
    }

    Vector<float, 3> pHat;
    {
        Vector<float, 3> newLogS = Vector<float, 3>{logf(newS[0]), logf(newS[1]), logf(newS[2])};
        Vector<float, 3> inverseS = 1.0f / newS;
        float newLogSTrace = newLogS[0] + newLogS[1] + newLogS[2];
        pHat = (scaledMu * newLogS + lambda * newLogSTrace) * inverseS;
    }
    meta::ConstexprLoop<0, 3>(
                              [&](auto indexWrapper) -> void
                              {
                              constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
                              PF[3 * index] = pHat[index] * V[3 * index];
                              PF[3 * index + 1] = pHat[index] * V[3 * index + 1];
                              PF[3 * index + 2] = pHat[index] * V[3 * index + 2];
                              }
                             );
    PF = U.MatrixMultiplication(PF).MatrixMultiplication(F.Transpose());
}

template <MPMConstitutiveModel ConstitutiveModel, typename Scalar>
requires(MPMConstitutiveModel::kFluid == ConstitutiveModel) MPM_FORCE_INLINE MPM_DEV_FUNC
	auto ComputeStress(const Scalar volume, const Scalar pressure, const Scalar viscosity, const Matrix<float, 3, 3>& covariantVelocity, Scalar& J, 
					   Matrix<float, 3, 3>& PF) -> void
{
    PF = (covariantVelocity + covariantVelocity.Transpose()) * viscosity;
    PF[0] -= pressure;
    PF[4] -= pressure;
    PF[8] -= pressure;

    PF *= J;
}

template <MPMConstitutiveModel ConstitutiveModel, typename Scalar>
requires(MPMConstitutiveModel::kVonMises == ConstitutiveModel) MPM_FORCE_INLINE MPM_DEV_FUNC
	auto ComputeStress(const Scalar volume, const Scalar mu, const Scalar lambda, const Scalar yieldStress, Matrix<float, 3, 3>& elasticDeformationGradient,
                       Matrix<float, 3, 3>& deformationGradient,
					   Matrix<float, 3, 3>& PF) -> void
{
	Matrix<float, 3, 3> U, V;
	Vector<float, 3> S;

	svd(elasticDeformationGradient[0], elasticDeformationGradient[1], elasticDeformationGradient[2],
		elasticDeformationGradient[3], elasticDeformationGradient[4], elasticDeformationGradient[5],
		elasticDeformationGradient[6], elasticDeformationGradient[7], elasticDeformationGradient[8], U[0], U[1], U[2],
		U[3], U[4], U[5], U[6], U[7], U[8], S[0], S[1], S[2], V[0], V[1], V[2], V[3], V[4], V[5], V[6], V[7], V[8]);

	Vector<float, 3> logS, invS;
	float logSSum = 0.f;
	meta::ConstexprLoop<0, 3>(
		[&](auto indexWrapper) -> void
		{
			constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
			logS[index] = logf(S[index]);
			invS[index] = 1.f / S[index];
			logSSum += logS[index];
		});

	Vector<float, 3> newS = 2.f * mu * invS * logS + lambda * logSSum * invS;

	meta::ConstexprLoop<0, 3>(
		[&](auto indexWrapper) -> void
		{
			constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
			PF[3 * index] = newS[index] * V[index];
			PF[3 * index + 1] = newS[index] * V[3 + index];
			PF[3 * index + 2] = newS[index] * V[6 + index];
		});

	//PF = U.MatrixMultiplication(PF).MatrixMultiplication(elasticDeformationGradient.Transpose());
	PF = U.MatrixMultiplication(PF).MatrixMultiplication(deformationGradient.Transpose());

}

template <MPMConstitutiveModel ConstitutiveModel, typename Scalar>
requires(MPMConstitutiveModel::kNonAssociatedCamClay == ConstitutiveModel) MPM_FORCE_INLINE MPM_DEV_FUNC
	auto ComputeStress(const typename MPMMaterial<ConstitutiveModel>::MaterialParameter_& material,
                       Matrix<float, 3, 3>& deformationGradient,
                       Scalar& logJp,
					   Matrix<float, 3, 3>& PF) -> void
{
	Matrix<float, 3, 3> U, V;
	Vector<float, 3> S;

    const float mu = material.mu_;
    const float lambda = material.lambda_;
    const float bulkModulus = 2.f * mu / 3.f + lambda;
    const float hardeningFactor = material.hardeningFactor_;
    const float beta = material.beta_;
    const float mSquare = material.m_ * material.m_;
    const bool hardeningOn = material.hardeningOn_;

	svd(deformationGradient[0], deformationGradient[1], deformationGradient[2],
		deformationGradient[3], deformationGradient[4], deformationGradient[5],
		deformationGradient[6], deformationGradient[7], deformationGradient[8], U[0], U[1], U[2],
		U[3], U[4], U[5], U[6], U[7], U[8], S[0], S[1], S[2], V[0], V[1], V[2], V[3], V[4], V[5], V[6], V[7], V[8]);


    float p0 = bulkModulus * (float(1e-5) + sinh(hardeningFactor * (-logJp > 0 ? -logJp : 0)));

    float pMin = -beta * p0;

    float Je_trial = S[0] * S[1] * S[2];
    Vector<float, 3> bHatTrial = Vector<float, 3>{S[0] * S[0], S[1] * S[1], S[2] * S[2]};
    float bHatTrialTraceDivDim = (bHatTrial[0] + bHatTrial[1] + bHatTrial[2]) / 3.f;
    float deviatoricFactor = mu * powf(Je_trial, -2.f / 3.f);
    Vector<float, 3> sHatTrial = deviatoricFactor * (bHatTrial - bHatTrialTraceDivDim);

    float hydrostaticFactor = bulkModulus * 0.5f * (Je_trial - 1.f / Je_trial);
    float pTrial = -hydrostaticFactor * Je_trial;
    
    float ysHalfCoefficient = 3.f / 2.f * (1 + 2.f * beta);
    float ypHalf = mSquare * (pTrial - pMin) * (pTrial - p0);
    float sHatTrialNormSquare = sHatTrial[0] * sHatTrial[0] + sHatTrial[1] * sHatTrial[1] + sHatTrial[2] * sHatTrial[2];
    float y = ysHalfCoefficient  * sHatTrialNormSquare + ypHalf;


    if(pTrial > p0 || pTrial < pMin)
    {

        float p = pTrial > p0 ? p0 : pMin;
        float Je_new = sqrtf(-2.f * p / bulkModulus + 1.f);

        S = powf(Je_new, 1.f / 3.f);
        Matrix<float, 3, 3> newF;

        meta::ConstexprLoop<0, 3>(
                                  [&](auto indexWrapper) -> void
                                  {
                                  constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
                                  newF[3 * index] = S[index] * V[3 * index];
                                  newF[3 * index + 1] = S[index] * V[3 * index + 1];
                                  newF[3 * index + 2] = S[index] * V[3 * index + 2];
                                  }
                                 );
        deformationGradient = U.MatrixMultiplication(newF);

        if(hardeningOn) logJp += logf(Je_trial / Je_new);
    }
    else
    {
        if(y >= 1e-4)
        {
            float bSCoefficient = powf(Je_trial, 2.f / 3.f) / mu * sqrtf(-ypHalf / ysHalfCoefficient) / sqrtf(sHatTrialNormSquare);
            meta::ConstexprLoop<0, 3>(
                                      [&](auto indexWrapper) -> void
                                      {
                                      constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
                                      S[index] = sqrtf(sHatTrial[index] * bSCoefficient + bHatTrialTraceDivDim);
                                      }
                                     );

            {
                Matrix<float, 3, 3> newF;
                meta::ConstexprLoop<0, 3>(
                                          [&](auto indexWrapper) -> void
                                          {
                                              constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
                                              newF[3 * index] = S[index] * V[3 * index];
                                              newF[3 * index + 1] = S[index] * V[3 * index + 1];
                                              newF[3 * index + 2] = S[index] * V[3 * index + 2];
                                          }
                                         );
                deformationGradient = U.MatrixMultiplication(newF);
            }

            meta::ConstexprLoop<0, 9>(
                                      [&](auto indexWrapper) -> void
                                      {
                                          constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
                                          if(hardeningOn && p0 > 1e-4 && pTrial < p0 - 1e-4 && pTrial > 1e-4 + pMin)
                                          {
                                            float pCenter = (1.f - beta) * p0 / 2;
                                            float qTrial = sqrt(3.f / 2.f * sHatTrialNormSquare);

                                            Vector<float, 2> direction = Vector<float, 2>{pCenter - pTrial, -qTrial};

                                            float directionNorm = direction.Norm();
                                            direction /= directionNorm;

                                            float A = mSquare * direction[0] * direction[0] + (1 + 2 * beta) * direction[1] * direction[1];
                                            float B = mSquare * direction[0] * (2 * pCenter - p0 - pMin);
                                            float C = mSquare * (pCenter - pMin) * (pCenter - p0);

                                            float discriminant = B * B - 4 * A * C;
                                            float l1 = (-B + sqrtf(discriminant)) / (2.f * A);
                                            float l2 = (-B - sqrtf(discriminant)) / (2.f * A);

                                            float p1 = pCenter + l1 * direction[0];
                                            float p2 = pCenter + l2 * direction[0];

                                            float pFake = (pTrial - pCenter) * (p1 - pCenter) > 0 ? p1 : p2;
                                            float Je_square = -2 * pFake / bulkModulus + 1.f;
                                            float Je_new = sqrtf(Je_square > 0.f ? Je_square : -Je_square);

                                            if(Je_new > 1e-4) logJp += logf(Je_trial / Je_new);
                                          }
                                      }
                                     );
            }


        }
        float J = S[0] * S[1] * S[2];
        Matrix<float, 3, 3> bDev, b;

        meta::ConstexprLoop<0, 9>(
                                  [&](auto indexWrapper) -> void
                                  {
                                      constexpr int index = meta::ConstexprLoopIndex(indexWrapper);
                                      constexpr int row = index / 3;
                                      constexpr int column = index % 3;
                                      b[index] = deformationGradient[row * 3] * deformationGradient[column * 3] + deformationGradient[row * 3 + 1] * deformationGradient[column * 3 + 1] + deformationGradient[row * 3 + 2] * deformationGradient[column * 3 + 2];
                                  }
                                 );
        bDev = b - (b.Trace() / 3.f) * Matrix<float, 3, 3>::Identity();

        float deviatoricCoefficient = mu * powf(J, -2.f / 3.f);
        float hydrostaticCoefficient = bulkModulus * 0.5f * (J * J - 1.f);

        PF = deviatoricFactor * bDev;
        PF[0] += hydrostaticCoefficient;
        PF[4] += hydrostaticCoefficient;
        PF[8] += hydrostaticCoefficient;

    }

	typedef std::variant<MPMMaterial<MPMConstitutiveModel::kLinear>, MPMMaterial<MPMConstitutiveModel::kFixedCorotated>,
                        MPMMaterial<MPMConstitutiveModel::kDruckerPragerStvkhencky>, MPMMaterial<MPMConstitutiveModel::kFluid>,
                        MPMMaterial<MPMConstitutiveModel::kVonMises>,
                        MPMMaterial<MPMConstitutiveModel::kNonAssociatedCamClay>>
		ParticleMaterial;
    


    

}  // namespace mpm
