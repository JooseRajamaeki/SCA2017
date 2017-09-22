


#ifndef SCA_CONTROL_H
#define SCA_CONTROL_H
#include <Eigen/Eigen> 
#include <vector>
#include "DiagonalGMM.h"
#include "TrajectoryOptimization.h"
#include "GenericDensityForest.hpp"
#include "ANN.h"
//#include "Debug.h"
#include "EigenMathUtils.h"
#include "MiscUtils.hpp"
#include "ProbUtils.hpp"


#include <future>
#include <iostream>
#include <fstream>
#include <ctime>
#include <deque>
#include <random>
#include <map>
#include <exception>

#define ENABLE_DEBUG_OUTPUT
#include <iostream> 
#include <time.h>

#ifdef SWIG
#define __stdcall  //SWIG doesn't understand __stdcall, but fortunately c# assumes it for virtual calls
#endif


#define MAX_NEAREST_NEIGHBORS 1000

namespace AaltoGames
{

	#define physicsBrokenCost 10e+20f



	class SCAControl //: public ITrajectoryOptimization
	{

	public:

#ifndef SWIG
		class TeachingSample{

		public:

			typedef float Scalar;

			Eigen::Matrix<Scalar,Eigen::Dynamic,1> state_;
			Eigen::Matrix<Scalar,Eigen::Dynamic,1> future_state_;
			Eigen::Matrix<Scalar,Eigen::Dynamic,1> control_;


			Scalar cost_to_go_;
			Scalar instantaneous_cost_;

			Eigen::Matrix<Scalar,Eigen::Dynamic,1> key_vector_;
			Eigen::VectorXf state_control_;

			Eigen::VectorXf input_for_learner_;
			Eigen::VectorXf output_for_learner_;

			bool operator==(const TeachingSample& other){

				float diff = 0.0f;
				for (int i = 0; i < state_.size(); i++){
					diff = state_[i] - other.state_[i];
					if (std::abs(diff) > 0.0f){
						return false;
					}
				}

				for (int i = 0; i < future_state_.size(); i++){
					diff = future_state_[i] - other.future_state_[i];
					if (std::abs(diff) > 0.0f){
						return false;
					}
				}

				for (int i = 0; i < control_.size(); i++){
					diff = control_[i] - other.control_[i];
					if (std::abs(diff) > 0.0f){
						return false;
					}
				}

	

				return true;

			}

			bool operator!=(TeachingSample& other){

				return !(*this == other);

			}


			TeachingSample()
			{
				state_ = Eigen::VectorXf::Zero(0);
				future_state_ = state_;
				control_ = state_;
		
				key_vector_ = state_;

				cost_to_go_ = std::numeric_limits<float>::infinity();
				instantaneous_cost_ = std::numeric_limits<float>::infinity();
	
			}

			TeachingSample(const MarginalSample& marginal_sample);

			~TeachingSample()
			{
				state_ = Eigen::VectorXf::Zero(0);
				future_state_ = state_;
				control_ = state_;
	
				key_vector_ = state_;

				cost_to_go_ = std::numeric_limits<float>::infinity();
				instantaneous_cost_ = std::numeric_limits<float>::infinity();

			}
		};
#endif
		int state_dimension_;
		int control_dimension_;

		//Random forest related parameters
		int nn_trajectories_;
		int number_of_nearest_neighbor_trees_;
		int number_of_data_in_leaf_;
		int number_of_hyperplane_tries_;
		int amount_data_in_tree_;
		int amount_recent_;
		bool use_forests_;

		//Neural network related parameters
		bool learning_;
		unsigned learning_budget_;
		bool use_machine_learning_;
		float regularization_noise_;
		int machine_learning_samples_;
		int noisy_machine_learning_samples_;

		float no_prior_trajectory_portion_;
		float stored_sample_percentage_;

		double best_cost_;

		int steps_;
		int max_steps_;
		int amount_samples_;
		int max_samples_;
		int iteration_idx_;
		bool resample_;
		int current_step_;
		int next_step_;
		int best_full_sample_idx_;
		bool time_advanced_;
		double resample_threshold_;
		bool use_sampling_;
		float validation_fraction_;


#ifndef SWIG

		std::mutex copying_transition_data_;
		std::mutex actor_mutex_;


		std::future<void> training_neural_net_;
		std::future<void> building_forest_;


		typedef TeachingSample::Scalar (*teaching_sample_distance)(TeachingSample&,TeachingSample&);
		

		GenericDensityForest<TeachingSample> ann_forest_;
		GenericDensityTree<TeachingSample> tree_under_construction_;
		std::deque<TeachingSample> adding_buffer_;


		std::vector<std::vector<MarginalSample> > experience_;
		std::vector<std::vector<MarginalSample> > previous_experience_;
		

		std::unique_ptr<MultiLayerPerceptron > actor_;
		std::unique_ptr<MultiLayerPerceptron > actor_copy_;
		std::unique_ptr<MultiLayerPerceptron > actor_in_training_;


		std::deque<std::unique_ptr<TeachingSample> > recent_samples_;
		std::deque<std::shared_ptr<TeachingSample> > validation_data_;
		std::deque<std::shared_ptr<TeachingSample> > transitions_;
		std::deque<std::shared_ptr<TeachingSample> > transitions_buffer_;


		typedef std::pair<Eigen::VectorXf, Eigen::VectorXf> DiagonalGaussian;
		std::vector<DiagonalGaussian> sampling_distributions_;
		std::vector<std::vector<DiagonalGaussian> > gaussian_distributions_;

		
		Eigen::VectorXf control_min_;
		Eigen::VectorXf control_max_;
		Eigen::VectorXf control_mean_;
		Eigen::VectorXf control_prior_std_;
		Eigen::VectorXf control_diff_prior_std_;
		Eigen::VectorXf control_mutation_std_;

		void init_neural_net(int input_dim, int output_dim, MultiLayerPerceptron& net);
		std::deque<std::string> get_settings();

#endif
		SCAControl();
		~SCAControl();


		void init(int nSampledTrajectories, int nSteps, int nStateDimensions, int nControlDimensions, const float *controlMinValues, const float *controlMaxValues, const float *controlMean, const float *controlPriorStd, const float *controlDiffPriorStd, const  float controlMutationStdScale, bool useMirroring);
		virtual double __stdcall getBestTrajectoryCost();
		virtual void __stdcall getBestControl(int timeStep, float *out_control);
		virtual const float* __stdcall getRecentControl(float* state, float *out_control, int thread);
		virtual void __stdcall getMachineLearningControl(float *state, float* out_control);
		virtual void __stdcall getBestControlState(int timeStep, float *out_state);
		virtual double __stdcall getBestTrajectoryOriginalStateCost(int timeStep);
		virtual void __stdcall setSamplingParams(const float *controlPriorStd, const float *controlDiffPriorStd, float controlMutationStdScale);

		
		virtual void __stdcall startIteration(bool advanceTime, const float *initialState);
		virtual void __stdcall startPlanningStep(int stepIdx);
		virtual int __stdcall getPreviousSampleIdx(int sampleIdx, int timeStep = -1);
		virtual void __stdcall getControl(int sampleIdx, float *out_control, const float *priorMean=0, const float *priorStd=0);
		virtual void __stdcall getAssumedStartingState(int sampleIdx, float *out_state);
		virtual void __stdcall updateResults(int sampleIdx, const float *finalControl, const float *newState, double cost, const float *priorMean=0, const float *priorStd=0);
		virtual void __stdcall endPlanningStep(int stepIdx);
		virtual void __stdcall endIteration();
		virtual void __stdcall setParams(float resampleThreshold, bool learning, int nTrajectories);
		virtual int __stdcall getBestSampleLastIdx();

		int getNumTrajectories();
		int getNumSteps();
		int getCurrentStep();
		void restart();
		void long_term_learning();


	private:

		std::vector<MarginalSample> old_best_;

		DiagonalGMM static_prior_;
		std::map<int,std::unique_ptr<DiagonalGMM> > proposals_;
		std::map<int,std::unique_ptr<DiagonalGMM> > priors_;

		std::vector<TeachingSample> keys_;

		SCAControl operator=(const SCAControl& other);
		SCAControl(const SCAControl& other);
		void resize_marginal(void);

	};

} //namespace AaltoGames


#endif


