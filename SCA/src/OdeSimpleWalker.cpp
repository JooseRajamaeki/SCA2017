
//In the paper the target_rotation was computed before the forward simulation. Thus the simulated trajectories are in the wrong coordinate frame (that of time instant zero). This is corrected here. Thus if you run the code, you will get slightly different results than in the paper.


#include <stdlib.h>
#include <cmath>
#include <vector>
#include <queue>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <Eigen/Geometry>

#include "RenderClient.h"
#include "RenderCommands.h"
#include "UnityOde.h"
#include "UnityOde_internal.h"

#include "mathutils.h"
#include "SCAController.h"
#include "FileUtils.h"



using namespace std::chrono;
using namespace AaltoGames;
using namespace Eigen;
#include "Biped.h"
#include "BipedLegsOnly.h"
#include "Quadruped.h"
#include "Monoped.h"

////Monoped-specific parameters
//typedef Monoped TestRig;
//static const float controlAccSd=deg2rad*30.0f;
//static const float angleSd=deg2rad*10.0f;  //cost function
//static const int nTrajectories=256; 
//static const int nRealtimeTrajectories=64;
//int non_groung_contact_bones[] = { 1, 0 };

////Legs only biped specific parameters
//typedef BipedLegsOnly TestRig;
//static const float controlAccSd=deg2rad*30.0f;
//static const float angleSd=deg2rad*10.0f;  //cost function
//static const int nTrajectories=64; 
//static const int nRealtimeTrajectories=16;
//int non_groung_contact_bones[] = {(int)TestRig::Bones::bLeftThigh,(int)TestRig::Bones::bRightThigh,(int)TestRig::Bones::bLeftShin,(int)TestRig::Bones::bRightShin };


//Full humanoid biped specific parameters
//Note: poseInterpolationTime=0.2, controlAccSd=deg2rad*40.0f also seems to work
typedef Biped TestRig;
static const float controlAccSd = deg2rad*10.0f;
//static const float angleSd = deg2rad*30.0f;  //cost function //run
static const float angleSd = deg2rad*10.0f;  //cost function  //walk
static const int nTrajectories = 64;
static const int nRealtimeTrajectories = 8;
int non_groung_contact_bones[] = { (int)TestRig::Bones::bPelvis,(int)TestRig::Bones::bLeftThigh,(int)TestRig::Bones::bLeftShin,(int)TestRig::Bones::bRightThigh,(int)TestRig::Bones::bRightShin,(int)TestRig::Bones::bSpine,(int)TestRig::Bones::bHead,(int)TestRig::Bones::bLeftUpperArm,(int)TestRig::Bones::bLeftForeArm,(int)TestRig::Bones::bRightUpperArm,(int)TestRig::Bones::bRightForeArm };

////Quadruped-specific parameters
//typedef Quadruped TestRig;
//static const float controlAccSd=deg2rad*10.0f;
//static const float angleSd=deg2rad*10.0f;  //cost function
//static const int nTrajectories=64; 
//static const int nRealtimeTrajectories=16;
//int non_groung_contact_bones[] = { (int)TestRig::Bones::bPelvis,(int)TestRig::Bones::bLeftThigh,(int)TestRig::Bones::bRightThigh,(int)TestRig::Bones::bLeftUpperArm,(int)TestRig::Bones::bRightUpperArm };


//Common parameters
static int num_motor_angles = 0;
static const float maxDistanceFromOrigin = 5.0f;
static const float rad2deg = 1.0f / deg2rad;
static const float torsoMinFMax = 5.0f;
static const float minFMax = 5.0f;
static const bool rigTestMode = false;
static const bool poseParameterization = true;
static const float fmaxDiffSd = 10.0f;
static const float poseSpringConstant = 10.0f;
static TestRig character;
static VectorXf controlMin, controlMax, controlRange, controlMean, controlSd, controlDiffSd, controlDiffDiffSd;
static VectorXf scaledControlSd; //control sd may be scaled, e.g., to allow wider range of movement when recovering balance after a disturbance
static SCAControl *flc;

static const float planningHorizonSeconds = 1.2f;
static int nTimeSteps = (int)(planningHorizonSeconds / timeStep);
static int nPhysicsPerStep = 1;
static const int fps = (int)(0.5f + 1.0f / timeStep);
static const bool useThreads = true;
static int resetSaveSlot = nTrajectories + 1;
static int masterContext;
static const float kmh2ms = 1.0f / 3.6f;
static float targetSpeed = 1.0f;
//static Vector3f targetVel(-6.0f*kmh2ms, 0, 0);  //run
//static Vector3f targetVel(0,0,0);
static const float angleSamplingSd = deg2rad*25.0f;  //sampling
static const float ikSd = 0.05f;
static const bool scoreAngles = true;
static const float velSd = 0.05f;
static const bool useContactVelCost = false;
static const float contactVelSd = 0.2f;
static const float comDiffSd = 0.025f;
static const float controlSdRelToRange = 8.0f;
static const float controlDiffRelSd = poseParameterization ? 100.0f : 0.25f;

static const bool useFastStep = false;
static const float resampleThreshold = 2.0f;
static const float mutationSd = useTorque ? 0.05f : 0.1f;
static const float poseTorqueRelSd = 0.5f;
static const float poseTorqueK = maxControlTorque / (1.0f*PI); //we exert maximum torque towards default pose if the angle displacement is 90 degrees 
static const float friction = 0.5f; //higher friction needed compared to the biped to prevent "cheat gaits"
static int frameIdx = 0;
static int randomSeed = 2;
static const bool onlyAdvanceIfNotFalling = false;
static const bool multiTask = false;
static bool enableRealTimeMode = true;
static bool realtimeMode = false;
static int stateDim; //character state dim + 1 for target speed
static const bool useErrorLearning = false;
static const bool test_real_time_mode = false;

//video capturing
static const bool captureVideo = true;
static const int startRealtimeModeAt = INT_MAX;// 5 * 60 * fps;
static int autoExitAt = 5000;
static const int nInitialCapturedFrames = autoExitAt;
static const int nFramesToCaptureEveryMinute = 15 * fps;

//launched spheres
static const bool useSpheres = false;
static const int sphereInterval = 5000;
static int lastSphereLaunchTime = 0; //to delay ball throwing so that gait has time to emerge with video capturing

//acceleration
static const bool useAcceleration = false;
static const float acceleration = 1.0f / 10.0f; //1 m/s in 10 seconds
static const float acceleratedMaxSpeed = 100.0f; //practically inf (meters per second)

static bool use_external_prior = true;
//random impulses 
static bool useRandomImpulses = false; //if true, learning is made more robust through application of random impulses
static const float randomImpulseMagnitude = 100.0f;
static const int randomImpulseInterval = 200;// 4 * fps;

//targeted walking
static bool useWalkTargets = false; //if true, targetVel computed every frame so that the character walks towards the walkTargets, targets switched after reached
static const int nWalkTargets = 4;
static int currentWalkTarget = 0;
static Vector3f walkTargets[nWalkTargets] = { Vector3f(-5,0,0),Vector3f(5.6f,0,0),Vector3f(0,-4.3f,0),Vector3f(0,6.2f,0) }; //walk back and forth along the x axis
static Vector3f walkTarget = Vector3f(-1000.0f, 0, 0);

static int walk_time = 200;

int get_walk_target(int number_of_walk_targets) {
	static int factor = 675;
	if (number_of_walk_targets <= 1) {
		return 0;
	}
	else {
		factor = (8121 * factor + 28411) % 134456;
		return factor % 4;
	}
}

//recovery mode !
static const bool enableRecoveryMode = true;
static const bool includeRecoveryModeInState = true;// true;  //has to be included as long as the mode directly affects fmax and spring kp
static float recoveryModePoseSdMult = 5.0f;
static float recoveryModeAccSdMult = 1.0f;
static float recoveryModeFmax = defaultFmax*2.0f;
static bool inRecoveryMode = false;
static const float recoveryModeAngleLimit = 30.0f;
static const float recoveryModeSpringKp = springKp*5.0f;
static const float recoveryModeTimeUntilReset = 2.0f; //wait this many seconds before resetting

//misc
static int groundPlane = -1;

std::deque<Eigen::VectorXf> control_sequence;
std::deque<Eigen::VectorXf> machine_learning_control_sequence;


static bool run_on_neural_network = false;

static std::string cost_file_name = "costs.csv";
static std::string started_at_time_string = "";
static std::vector<std::vector<float>> costs;
static std::vector<std::string> comments;
static bool no_settings_exit = false;

static Eigen::VectorXf minControls;
static Eigen::VectorXf maxControls;

static void write_vector_to_file(const std::string& filename, const std::vector<std::vector<float> >& data, const std::vector<std::string> comments = std::vector<std::string>()) {

	std::ofstream myfile;
	myfile.open(filename);
	myfile.clear();

	for (const std::string& comment_line : comments) {
		myfile << "//" << comment_line << std::endl;
	}

	for (const std::vector<float>& measurement : data) {
		for (unsigned i = 0; i < measurement.size(); i++) {
			myfile << measurement[i];
			if (i < (int)measurement.size() - 1) {
				myfile << ",";
			}
		}
		myfile << std::endl;
	}

	myfile.close();

}


//timing
high_resolution_clock::time_point t1;
void startPerfCount()
{
	t1 = high_resolution_clock::now();
}
int getDurationMs()
{
	high_resolution_clock::time_point t2 = high_resolution_clock::now();

	std::chrono::duration<double> time_span = duration_cast<std::chrono::duration<double>>(t2 - t1);
	return (int)(time_span.count()*1000.0);
}

class SphereData
{
public:
	int body, geom, spawnFrame;
};
static std::vector<SphereData> spheres;


class SimulationContext
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;  //needed as we have fixed size eigen matrices as members
	VectorXf stateFeatures;
	VectorXf control;
	VectorXf priorMean, priorSd;
	VectorXf angleRates;
	Vector3f initialPosition, resultPosition;
	float stateCost;
	float controlCost;
	int trajectoryIdx;
};

static SimulationContext contexts[nTrajectories + 2];

Vector3f get_target_dir(Vector3f& com) {

	Vector3f dir = walkTarget - com;
	dir.normalize();
	return dir;
}

//state vector seen by the controller, including both character's physical state and task information
int computeStateVector(float *out_state)
{

	Vector3f com;
	character.computeCOM(com);

	Vector3f target_dir = get_target_dir(com);
	const Vector3f initialDir(-1, 0, 0);
	Quaternionf targetRotation = Quaternionf::FromTwoVectors(initialDir, target_dir);
	Quaternionf stateRotation = targetRotation.inverse(); //The codes that were used in the paper had a bug here. The stateRotation was computed before running the MCTS instead of being computed here.

	const bool use_motor_angles = false;

	int nState = 0;
	if (!use_motor_angles) {
		nState = character.computeStateVector(out_state,stateRotation);

		//task variables scaled larger so that they dominate the state distances
		if (useWalkTargets)
			character.pushStateVector3f(nState, out_state, target_dir*10.0f);
		else
			out_state[nState++] = target_dir.x()*10.0f;
	}

	if (use_motor_angles) {

		//since ground contacts are highly significant, add foot bone tip y pos and vel with greater weight
		for (int i = 0; i <= 1; i++)
		{
			Vector3f footPos = character.getFootPos((OdeRig::BodySides)i);
			//Vector3f footVel(odeBodyGetLinearVel(character.getFootBone((OdeRig::BodySides)i)->body));
			out_state[nState] = footPos.z();
			nState++;
			//out_state[nState] = footVel.z();
			//nState++;
		}

		Vector3f tmp = com;
		out_state[nState] = tmp.z();
		nState++;

		character.computeMeanVel(tmp);
		out_state[nState] = tmp.z();
		nState++;



		Quaternionf q(odeBodyGetQuaternion(character.bones[0]->body));
		Quaternionf root_rotation = stateRotation*q;

		out_state[nState] = root_rotation.x();
		nState++;
		out_state[nState] = root_rotation.y();
		nState++;
		out_state[nState] = root_rotation.z();
		nState++;
		out_state[nState] = root_rotation.w();
		nState++;

		if (num_motor_angles == 0) {
			for (auto joint_ptr : character.joints) {
				num_motor_angles += joint_ptr->nMotorDof;
			}
		}

		float motor_angles[100];
		character.getCurrentMotorAngles(motor_angles);

		for (int motor_angle = 0; motor_angle < num_motor_angles; motor_angle++) {
			out_state[nState] = motor_angles[motor_angle];
			nState++;
		}

		character.getCurrentAngleRates(motor_angles);

		for (int motor_angle = 0; motor_angle < num_motor_angles; motor_angle++) {
			out_state[nState] = motor_angles[motor_angle];
			nState++;
		}

		//task variables scaled larger so that they dominate the state distances
		out_state[nState++] = target_dir.norm()*10.0f;

	}

	if (enableRecoveryMode && includeRecoveryModeInState)
	{
		out_state[nState++] = 10.0f*(inRecoveryMode ? 1.0f : 0);  //needed as state cost computed differently, and recovery mode affects the state,action -> next state mapping (through spring constants and fmax)
	}
	if (useSpheres)
	{


		bool pushZeros = true;
		if (spheres.size() > 0)
		{
			//add the relative position of the last launched sphere
			SphereData &sd = spheres.back();
			Vector3f spherePos(odeBodyGetPosition(sd.body));
			Vector3f sphereVel(odeBodyGetLinearVel(sd.body));
			Vector3f pos(odeBodyGetPosition(character.bones[0]->body));
			Vector3f relPos = stateRotation*(spherePos - pos);
			Vector3f relVel = stateRotation*sphereVel;
			//if sphere flying towards us, add it to the state
			if (relPos.norm() < 5.0f && relVel.dot(relPos.normalized()) < -1.0f)
			{
				character.pushStateVector3f(nState, out_state, relPos);
				character.pushStateVector3f(nState, out_state, relVel);
				pushZeros = false;
			}
		}
		if (pushZeros)
		{
			character.pushStateVector3f(nState, out_state, Vector3f::Zero());
			character.pushStateVector3f(nState, out_state, Vector3f::Zero());
		}
	}
	return nState;
}

std::chrono::time_point<std::chrono::system_clock> start, end;

void EXPORT_API rcInit()
{

	start = std::chrono::system_clock::now();

	if (test_real_time_mode) {
		enableRealTimeMode = true;
	}

	started_at_time_string = get_time_string();
	unsigned seed = time(nullptr);
	srand(seed);



	costs.clear();
	comments.clear();
	comments.push_back("This is a sample comment.");

	initOde(nTrajectories + 2);
	setCurrentOdeContext(ALLTHREADS);
	allocateODEDataForThread();
	odeRandSetSeed(randomSeed);
	// create world
	odeWorldSetGravity(0, 0, -9.81f);
	odeWorldSetCFM(1e-5);
	odeSetFrictionCoefficient(friction);
	//odeWorldSetLinearDamping(0.001);
	//dWorldSetAngularDamping(world, 0.001);
	//dWorldSetMaxAngularSpeed(world, 200);

	odeWorldSetContactMaxCorrectingVel(5);
	odeWorldSetContactSurfaceLayer(0.01f);
	groundPlane = odeCreatePlane(0, 0, 0, 1, 0);
	odeGeomSetCategoryBits(groundPlane, 0xf0000000);
	odeGeomSetCollideBits(groundPlane, 0xffffffff);
	character.init(false, rigTestMode);
	int nLegs = character.numberOfLegs();

	//controller init (motor target velocities are the controlled variables, one per joint)
	float tmp[1024];
	stateDim = computeStateVector(tmp);
	controlMin = Eigen::VectorXf::Zero(character.controlDim);
	controlMax = Eigen::VectorXf::Zero(character.controlDim);
	controlMean = Eigen::VectorXf::Zero(character.controlDim);
	controlSd = Eigen::VectorXf::Zero(character.controlDim);
	controlDiffSd = Eigen::VectorXf::Zero(character.controlDim);
	controlDiffDiffSd = controlDiffSd;

	int controlVarIdx = 0;
	for (size_t i = 0; i < character.joints.size(); i++)
	{
		OdeRig::Joint *j = character.joints[i];
		for (int dofIdx = 0; dofIdx < j->nMotorDof; dofIdx++)
		{
			if (useTorque)
			{
				controlMin[controlVarIdx] = -maxControlTorque;
				controlMax[controlVarIdx] = maxControlTorque;
				controlSd[controlVarIdx] = controlMax[controlVarIdx];
			}
			else if (!poseParameterization)
			{
				controlMin[controlVarIdx] = -maxControlSpeed;
				controlMax[controlVarIdx] = maxControlSpeed;
				float range = j->angleMax[dofIdx] - j->angleMin[dofIdx];
				controlSd[controlVarIdx] = controlSdRelToRange*range;
			}
			else
			{
				controlMin[controlVarIdx] = j->angleMin[dofIdx];
				controlMax[controlVarIdx] = j->angleMax[dofIdx];
				float range = controlMax[controlVarIdx] - controlMin[controlVarIdx];
				controlSd[controlVarIdx] = angleSamplingSd;
			}
			controlMean[controlVarIdx] = 0;
			controlDiffSd[controlVarIdx] = controlSd[controlVarIdx] * controlDiffRelSd;
			controlDiffDiffSd[controlVarIdx] = controlSd[controlVarIdx] * controlDiffRelSd * 4.0f;
			controlVarIdx++;
		}
	}
	if (!useTorque && (optimizeFmaxInVelocityMode))
	{
		for (int i = 0; i < character.nMotors; i++)
		{
			controlMin[controlVarIdx] = minFMax;
			if (nLegs == 2 && (i == Biped::bPelvis || i == Biped::bSpine || i == Biped::bHead))
				controlMin[controlVarIdx] = torsoMinFMax;
			controlMax[controlVarIdx] = 2.0f*defaultFmax / controlFmaxScale;
			controlMean[controlVarIdx] = 0;
			controlSd[controlVarIdx] = defaultFmax / controlFmaxScale;
			controlDiffSd[controlVarIdx] = fmaxDiffSd/controlFmaxScale;
			controlDiffDiffSd[controlVarIdx] = fmaxDiffSd / controlFmaxScale;
			controlVarIdx++;
		}
	}

	controlRange = controlMax - controlMin;
	flc = new SCAControl();
	flc->amount_data_in_tree_ = 2 * 60 * 30; //2 minutes


	flc->init(nTrajectories, nTimeSteps / nPhysicsPerStep, stateDim, character.controlDim, controlMin.data(), controlMax.data(), controlMean.data(), controlSd.data(), controlDiffSd.data(), mutationSd, false);
	flc->no_prior_trajectory_portion_ = 0.25f;
	flc->learning_budget_ = 2000;

	minControls = flc->control_min_;
	maxControls = flc->control_max_;

	for (int i = 0; i < nTrajectories + 1; i++)	//+1 because of master context
	{
		contexts[i].stateFeatures.resize(stateDim);
		contexts[i].control.resize(character.controlDim);
		contexts[i].trajectoryIdx = i;
		contexts[i].priorMean.resize(character.controlDim);
		contexts[i].priorSd.resize(character.controlDim);
		contexts[i].angleRates.resize(character.controlDim);
	}

	masterContext = nTrajectories;
	setCurrentOdeContext(masterContext);
	stepOde(timeStep, true); //allow time for joint limits to set
	stepOde(timeStep, true); //allow time for joint limits to set
	stepOde(timeStep, true); //allow time for joint limits to set
	character.saveInitialPose();
	saveOdeState(resetSaveSlot, masterContext);
	saveOdeState(masterContext, masterContext);
}

void EXPORT_API rcUninit()
{
	delete flc;
	uninitOde();
}

void EXPORT_API rcGetClientData(RenderClientData &data)
{
	data.physicsTimeStep = timeStep*(float)nPhysicsPerStep;
	data.maxAllowedTimeStep = data.physicsTimeStep; //render every step
	data.defaultMouseControlsEnabled = true;
}

static int dofIdx = 0;
static float rigTestControl[32] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
static bool paused = false;
static bool applyImpulse = false;

void throwSphere()
{
	//launch a ball
	setCurrentOdeContext(ALLTHREADS);
	float r = 0.33f;
	SphereData sd;
	sd.geom = odeCreateSphere(r);
	sd.body = odeBodyCreate();
	sd.spawnFrame = frameIdx;
	odeGeomSetBody(sd.geom, sd.body);
	dMass mass;
	odeMassSetSphereTotal(sd.body, 50.0f, r);
	setCurrentOdeContext(masterContext);
	restoreOdeState(masterContext);
	Vector3f characterPos(odeBodyGetPosition(character.bones[0]->body));
	Vector3f throwDir(randomf() - 0.5f, randomf() - 0.5f, 0);
	throwDir.normalize();
	Vector3f vel;
	character.computeMeanVel(vel);
	Vector3f spawnPos = characterPos + vel - 5.0f*throwDir;  //Add vel, as the ball will hit in 1 second
	spawnPos.z() = 2.0f;
	odeBodySetPosition(sd.body, spawnPos.x(), spawnPos.y(), spawnPos.z());
	Vector3f spawnVel = throwDir*5.0f;
	spawnVel.z() = 3.5f;
	odeBodySetLinearVel(sd.body, spawnVel.x(), spawnVel.y(), spawnVel.z());
	saveOdeState(masterContext, masterContext);
	spheres.push_back(sd);
	lastSphereLaunchTime = frameIdx;
}

static void write_vectors_to_file(const std::string& filename, const std::deque<Eigen::VectorXf>& data, const std::vector<std::string> comments = std::vector<std::string>()) {

	std::ofstream myfile;
	myfile.open(filename);

	for (const std::string& comment_line : comments) {
		myfile << "//" << comment_line << std::endl;
	}

	for (const Eigen::VectorXf& datum : data) {
		int size = datum.size();
		for (int i = 0; i < size; i++) {
			myfile << datum[i];
			if (i < size - 1) {
				myfile << ",";
			}
		}
		myfile << std::endl;
	}

	myfile.close();

}

static bool switchTargets = false; //if true, toggles target switching at next rcUpdate()  (todo: Unity-style input handling...)
void EXPORT_API rcOnKeyDown(int key)
{
	if (key == 'p')
		paused = !paused;
	if (key == '2')
	{
		dofIdx = (dofIdx + 1) % character.nTotalDofs;
		printf("Adjusting dof %d\n", dofIdx);
	}
	if (key == '1')
	{
		dofIdx = (dofIdx - 1 + character.nTotalDofs) % character.nTotalDofs;
		printf("Adjusting dof %d\n", dofIdx);
	}
	if (key == '3')
	{
		rigTestControl[dofIdx] = useTorque ? -maxControlTorque : -maxControlSpeed;
	}
	if (key == '4')
	{
		rigTestControl[dofIdx] = 0;
	}
	if (key == '5')
	{
		rigTestControl[dofIdx] = useTorque ? maxControlTorque : maxControlSpeed;
	}
	if (key == 'r')
	{
		saveOdeState(masterContext, resetSaveSlot);
		if (useAcceleration)
			targetSpeed *= 0;
	}
	if (key == 't')
	{
		realtimeMode = !realtimeMode;
	}
	if (key == ' ')
	{
		throwSphere();
	}
	if (key == 'i')
	{
		applyImpulse = true;
	}
	if (key == 'w')
	{
		switchTargets = true;
	}

	if (key == 'n') {
		run_on_neural_network = true;
	}
	if (key == 'm') {
		run_on_neural_network = false;
	}

	if (key == 'o') {
		write_vectors_to_file("controls.txt", control_sequence);
		write_vectors_to_file("ml_controls.txt", machine_learning_control_sequence);
		std::system("python density_plotter.py");
	}
}

void EXPORT_API rcOnKeyUp(int key)
{
}

float computeStateCost(const TestRig &character, bool debugOutput = false)
{
	float result = 0;
	Vector3f com;
	character.computeCOM(com);

	Vector3f target_dir = get_target_dir(com);
	const Vector3f initialDir(-1, 0, 0);
	Quaternionf targetRotation = Quaternionf::FromTwoVectors(initialDir, target_dir);

	int nLegs = character.numberOfLegs();



	//bone angle diff from initial
	if (scoreAngles)
	{
		for (size_t i = 0; i < character.bones.size(); i++)
		{
			//score bones other than arms of a humanoid biped (arms should be able to swing freely)
			if (!(nLegs == 2 && (i == Biped::bLeftFoot || i == Biped::bRightFoot || i == Biped::bLeftForeArm || i == Biped::bLeftUpperArm || i == Biped::bRightForeArm || i == Biped::bRightUpperArm)))
			{
				OdeRig::Bone *b = character.bones[i];
				Quaternionf q = ode2eigenq(odeBodyGetQuaternion(b->body));
				float weight = 1.0f;
				float sdMult = 1.0f;
				result += weight*squared(q.angularDistance(targetRotation*b->initialRotation) / (angleSd));
			}
		}
	}
	else
	{
		for (size_t i = 0; i < character.bones.size(); i++)
		{
			OdeRig::Bone *b = character.bones[i];
			Vector3f endPos = character.getBoneEndPos(b);
			result += ((endPos - com) - targetRotation*b->initialEndPos).squaredNorm() / squared(ikSd);
		}
	}

	//com over feet
	if (scoreAngles && nLegs <= 2)
	{
		Vector3f meanFeet = Vector3f::Zero();
		float footy[2];
		for (int i = 0; i < nLegs; i++)
		{
			Vector3f footPos = character.getFootPos(i);
			footy[i] = footPos.z();
			meanFeet += footPos / (float)nLegs;

		}
		meanFeet.z() = std::max(footy[0], footy[1]);
		Vector3f comDiff = com - meanFeet;
		comDiff.z() = std::min(comDiff.z(), 0.0f); //only penalize vertical difference if a foot higher than com (prevents the cheat where character lies down and holds legs up over com)
		result += comDiff.squaredNorm() / squared(comDiffSd);
	}

	//COM vel difference
	Vector3f vel;
	character.computeMeanVel(vel);
	Vector3f targetVel = targetSpeed*target_dir;
	Vector3f velDiff = vel - targetVel;
	velDiff.z() = std::min(velDiff.z(), 0.0f);  //don't penalize upwards deviation
	result += velDiff.squaredNorm() / squared(velSd);

	
	return result;
}


bool fallen()
{
	for (int bone : non_groung_contact_bones) {

		int bone_geom_id = character.bones[bone]->geoms[0];

		Vector3f pos, normal, vel;
		pos.setZero();
		normal.setZero();
		vel.setZero();
		bool contact = odeGetGeomContact(bone_geom_id, groundPlane, pos.data(), normal.data(), vel.data());

		if (contact) {
			return true;
		}

	}

	return false;
}

bool recovered()
{
	return !fallen();
}


void applyControl(const float *control)
{

	if (poseParameterization && !useTorque)
	{
		float currentAngles[256];
		float control2[256];
		memcpy(control2, control, sizeof(float)*character.controlDim);
		character.getCurrentMotorAngles(currentAngles);
		for (int i = 0; i < character.nTotalDofs; i++)
		{
			control2[i] = (control[i] - currentAngles[i]) *  poseSpringConstant;
		}
		character.applyControl(control2);


		character.setFmaxForAllMotors(defaultFmax);
		character.setMotorSpringConstants(springKp, springDamping);

		if (enableRecoveryMode && !optimizeFmaxInVelocityMode)
		{
			const bool allow_changing_spring_constants = true;
			const bool allow_changing_fmax = true;
			if (inRecoveryMode) {
				if (allow_changing_fmax) {
					character.setFmaxForAllMotors(recoveryModeFmax);
				}
				if (allow_changing_spring_constants) {
					character.setMotorSpringConstants(recoveryModeSpringKp, springDamping);
				}
			}
		}

	}
	else
		character.applyControl(control);
}

static float previousCost = 10000.0f;
static int fallCount = 0;

void EXPORT_API rcOnMouse(float,float,float,float,float,float,int,int,int) {

}

void EXPORT_API rcUpdate()
{
	//accelerated target velocity
	if (useAcceleration)
	{
		if (frameIdx == 0)
			targetSpeed *= 0;
		targetSpeed += acceleration*timeStep;
	}

	//walk targets
	if (useWalkTargets)
	{


		//pick next target location and velocity
		Vector3f com;
		character.computeCOM(com);
		com.z() = 0;
		float distToTarget = (walkTargets[currentWalkTarget] - com).norm();
		if (false && distToTarget < 0.5f || switchTargets)
		{
			currentWalkTarget = get_walk_target(nWalkTargets);
			switchTargets = false;
		}

		if (frameIdx % walk_time == 0) {
			currentWalkTarget = get_walk_target(nWalkTargets);
			switchTargets = false;
		}

		walkTarget = walkTargets[currentWalkTarget];

	}

	//visualize target as a "pole"
	dMatrix3 R;
	dRSetIdentity(R);
	rcSetColor(0.5f, 1, 0.5f);
	rcDrawCapsule(walkTarget.data(), R, 20.0f, 0.02f);
	rcSetColor(1, 1, 1);

	//multitasking with random velocity
	static float timeOnTask = 0;
	timeOnTask += timeStep;
	if (multiTask && (timeOnTask > 5.0f))
	{
		float r = (float)randInt(-1, 1); 
		targetSpeed = r;
		printf("New target vel %f\n", targetSpeed);
		timeOnTask = 0;
	}

	//setup current character state
	setCurrentOdeContext(masterContext);
	restoreOdeState(masterContext);
	VectorXf startState(stateDim);
	computeStateVector(&startState[0]);

	int currentTrajectories = nTrajectories;
	flc->setParams(previousCost*resampleThreshold, true, nTrajectories);
	flc->control_diff_prior_std_ = controlSd;


	end = std::chrono::system_clock::now();

	std::chrono::duration<double> duration = end - start;
	int seconds = (int)duration.count() % 60;
	int minutes = (int)duration.count() / 60;

	rcPrintString("Number of trajectories: %d %s", currentTrajectories, realtimeMode ? "(Realtime-mode)" : "");
	rcPrintString("Planning horizon: %.1f seconds", planningHorizonSeconds);
	rcPrintString("Time %02d:%02d:%2d (%d frames)", frameIdx / (fps * 60), (frameIdx / fps) % 60, frameIdx % fps, frameIdx);
	rcPrintString("Pruning threshold %.1f", previousCost*resampleThreshold);
	scaledControlSd = controlSd;
	if (inRecoveryMode)
	{
		scaledControlSd *= recoveryModePoseSdMult;
	}
	else {
		flc->control_min_ = minControls;
		flc->control_max_ = maxControls;
	}
	flc->setSamplingParams(scaledControlSd.data(), controlDiffSd.data(), mutationSd);

	startPerfCount();
	flc->startIteration(!onlyAdvanceIfNotFalling || (fallCount == 0), &startState[0]);
	int masterContext = nTrajectories;

	bool willFall = false;
	if (!rigTestMode)
	{

		if (!run_on_neural_network) {

			for (int step = 0; step < nTimeSteps / nPhysicsPerStep; step++)
			{
				int nUsedTrajectories = flc->getNumTrajectories();
				flc->startPlanningStep(step);
				if (step == 0)
				{
					for (int i = 0; i < nUsedTrajectories; i++)
					{
						saveOdeState(contexts[i].trajectoryIdx, masterContext);
					}
				}
				else
				{
					for (int i = 0; i < nUsedTrajectories; i++)
					{
						saveOdeState(contexts[i].trajectoryIdx, contexts[i].trajectoryIdx);
					}

				}

				std::deque<std::future<void>> workers;

				for (int t = nUsedTrajectories - 1; t >= 0; t--)
				{
					//lambda to be executed in the thread of the simulation context
					auto controlStep = [step](int data) {
						if (frameIdx == 0 && step == 0 && useThreads)
						{
							allocateODEDataForThread();
							odeRandSetSeed(randomSeed);
						}
						SimulationContext &c = contexts[data];
						setCurrentOdeContext(c.trajectoryIdx);
						restoreOdeState(flc->getPreviousSampleIdx(c.trajectoryIdx));


						if (use_external_prior) {
							//pose prior (towards zero angles)
							character.getCurrentMotorAngles(c.priorMean.data());
							if (useTorque)
							{
								c.priorMean = -poseTorqueK*c.priorMean;
								c.priorSd = controlSd*poseTorqueRelSd;
							}
							else if (!poseParameterization)
							{
								for (int i = 0; i < character.nTotalDofs; i++)
								{
									//We assume the following linear relation: pose+deltaTime*avel=resultPose
									//=> avel=(resultPose-pose)/deltaTime
									//We set a prior for the result pose, E[resultPose]=0, var[resultPose] and compute the corresponding avel prior
									//E[avel]=(E[resultPose]-pose)/deltaTime, since current pose is known and not a random variable
									//var[avel]=var[resultPose]/deltaTime^2 => sd[avel]=sd[resultPose]/deltaTime
									float posePriorSd = 15.0f; //degrees
									c.priorMean[i] = -c.priorMean[i] / timeStep;
									c.priorSd[i] = posePriorSd / timeStep;
								}
								if (optimizeFmaxInVelocityMode)
								{
									c.priorMean.tail(character.controlDim - character.nTotalDofs).setConstant(0);
									c.priorSd.tail(character.controlDim - character.nTotalDofs).setConstant(defaultFmax*100.0f); //NOP, fmax continuity not constrained
								}
							}
							else
							{
								//In pose-based control we use the extra prior to limit acceleration.
								//We first compute the predicted pose based on current pose and motor angular velocities, and then
								//set the prior there.
								c.angleRates.setZero();
								character.getCurrentAngleRates(c.angleRates.data());
								c.priorMean = c.priorMean + c.angleRates / poseSpringConstant;
								c.priorMean = c.priorMean.cwiseMax(controlMin);
								c.priorMean = c.priorMean.cwiseMin(controlMax);
								c.priorSd.setConstant(controlAccSd * (inRecoveryMode ? recoveryModeAccSdMult : 1.0f));
								if (optimizeFmaxInVelocityMode)
								{
									c.priorMean.tail(character.controlDim - character.nTotalDofs).setConstant(0);
									c.priorSd.tail(character.controlDim - character.nTotalDofs).setConstant(defaultFmax*100.0f);
								}
							}
						}

						//sample control
						if (use_external_prior) {
							flc->getControl(c.trajectoryIdx, &c.control[0], c.priorMean.data(), c.priorSd.data());
						}
						else {
							flc->getControl(c.trajectoryIdx, &c.control[0], nullptr, nullptr);
						}
						//step physics
						character.computeCOM(c.initialPosition);
						bool broken = false;
						
						float controlCost = 0;
						for (int k = 0; k < nPhysicsPerStep; k++)
						{
							applyControl(&c.control[0]);
							if (useFastStep)
								broken = !stepOdeFast(timeStep, false);
							else
								broken = !stepOde(timeStep, false);
							if (broken)
							{
								restoreOdeState(flc->getPreviousSampleIdx(c.trajectoryIdx));
								break;
							}

							if (useTorque)
								controlCost = c.control.cwiseQuotient(controlSd).squaredNorm();
							else
							{
								controlCost += character.getAppliedSqJointTorques();
								controlCost /= squared(defaultFmax);
							}
						}

						


						character.computeCOM(c.resultPosition);
						if (!broken)
						{
							float brokenDistanceThreshold = 0.25f;
							if ((c.resultPosition - c.initialPosition).norm() > brokenDistanceThreshold)
							{
								restoreOdeState(flc->getPreviousSampleIdx(c.trajectoryIdx));
								c.resultPosition = c.initialPosition;
								broken = true;
							}
						}

						//evaluate state cost
						float stateCost = computeStateCost(character);
						if (broken)
							stateCost += 1000000.0f;
						computeStateVector(&c.stateFeatures[0]);
		
						flc->updateResults(c.trajectoryIdx, c.control.data(), c.stateFeatures.data(), stateCost + controlCost);
						c.stateCost = stateCost;
						c.controlCost = controlCost;
					};
					if (!useThreads)
						controlStep(t);
					else
						workers.push_back(std::async(std::launch::async, controlStep, t));
				}
				if (useThreads)
				{
					for (std::future<void>& worker : workers) {
						worker.wait();
					}
				}

				flc->endPlanningStep(step);

				//debug visualization
				for (int t = nUsedTrajectories - 1; t >= 0; t--)
				{
					SimulationContext &c = contexts[t];
					if (flc->experience_[step + 1][t].particleRole == ParticleRole::OLD_BEST) {
						rcSetColor(0, 1, 0, 1);
					}
					else if (flc->experience_[step + 1][t].particleRole == ParticleRole::NEAREST_NEIGHBOR)
					{
						rcSetColor(0, 0, 1, 1);
					}
					else
					{
						rcSetColor(1, 1, 1, 1.0f);
					}
					rcDrawLine(c.initialPosition.x(), c.initialPosition.y(), c.initialPosition.z(), c.resultPosition.x(), c.resultPosition.y(), c.resultPosition.z());
				}
				rcSetColor(1, 1, 1, 1);
			}

			flc->endIteration();

			//print profiling info
			int controllerUpdateMs = getDurationMs();
			rcPrintString("Controller update time: %d ms", controllerUpdateMs);

			//check whether best trajectory will fall
			setCurrentOdeContext(contexts[flc->getBestSampleLastIdx()].trajectoryIdx);

			willFall = fallen();
			if (willFall)
				fallCount++;
			else
				fallCount = 0;

		}


		if (useAcceleration)
			rcPrintString("Target speed: %.2f m/s", fabs(targetSpeed));


		//step master context
		setCurrentOdeContext(masterContext);
		restoreOdeState(masterContext);

		if (run_on_neural_network) {
			bool on_the_ground = fallen();
			if (on_the_ground) {
				fallCount++;
			}
			else {
				fallCount = 0;
			}
		}

		Eigen::VectorXf machine_learning_control = contexts[masterContext].control;

		computeStateVector(contexts[masterContext].stateFeatures.data());
		flc->getMachineLearningControl(contexts[masterContext].stateFeatures.data(), machine_learning_control.data());

		if (run_on_neural_network) {
			contexts[masterContext].control = machine_learning_control;
		}
		else {
			flc->getBestControl(0, contexts[masterContext].control.data());
		}

		machine_learning_control_sequence.push_back(machine_learning_control);
		control_sequence.push_back(contexts[masterContext].control);

		float best_trajectory_cost = (float)flc->getBestTrajectoryCost();
		for (int k = 0; k < nPhysicsPerStep; k++)
		{
			//apply control (and random impulses)
			applyControl(contexts[masterContext].control.data());
			if (applyImpulse || (useRandomImpulses && !realtimeMode && ((frameIdx % randomImpulseInterval) == 0)))
			{
				applyImpulse = false;
				float impulse[3] = { randomf()*randomImpulseMagnitude,randomf()*randomImpulseMagnitude,randomf()*randomImpulseMagnitude };
				odeBodyAddForce(character.bones[0]->body, impulse);
			}
			if (useFastStep)
				stepOdeFast(timeStep, false);
			else
				stepOde(timeStep, false);
		}
		if (!onlyAdvanceIfNotFalling || (fallCount == 0)) //only progress if not fallen
			saveOdeState(masterContext, masterContext);

		float controlCost = character.getAppliedSqJointTorques();
		controlCost /= squared(defaultFmax);

		previousCost = contexts[flc->getBestSampleLastIdx()].stateCost;
		float state_cost = computeStateCost(character);
		rcPrintString("state cost %.2f, current control cost %.2f, end state cost %.2f\n", state_cost, controlCost, previousCost);

		std::vector<float> cost;
		cost.push_back(best_trajectory_cost);
		cost.push_back(state_cost + controlCost);

		costs.push_back(cost);


	} //!rigTestMode
	setCurrentOdeContext(masterContext);

	//in simple forward walking without targets, if moved too far from origin, move back
	if (!useWalkTargets)
	{
		Vector3f com;
		character.computeCOM(com);
		if (com.norm() > maxDistanceFromOrigin)
		{
			Vector3f disp = -com;
			disp.z() = 0;
			for (size_t i = 0; i < character.bones.size(); i++)
			{
				Vector3f pos(odeBodyGetPosition(character.bones[i]->body));
				pos += disp;
				odeBodySetPosition(character.bones[i]->body, pos.x(), pos.y(), pos.z());
			}
			saveOdeState(masterContext, masterContext);
		}
	}
	//check for falling
	if (fallCount > (int)((inRecoveryMode ? recoveryModeTimeUntilReset : 1.0f) / timeStep))
	{
		saveOdeState(masterContext, resetSaveSlot);

		run_on_neural_network = false;


		for (int i = 0; i <= nTrajectories; i++)
		{
			setCurrentOdeContext(contexts[i].trajectoryIdx);
			restoreOdeState(resetSaveSlot);
			saveOdeState(contexts[i].trajectoryIdx, contexts[i].trajectoryIdx);
		}
	
		fallCount = 0;
		if (useAcceleration)
			targetSpeed *= 0;
	
	}
	if (rigTestMode)
	{
		applyControl(rigTestControl);
		if (useFastStep)
			stepOdeFast(timeStep);
		else
			stepOde(timeStep, false);
		saveOdeState(masterContext, masterContext);
	}
	static bool setVP = true;
	if (setVP)
	{
		setVP = false;
		rcSetViewPoint(-(2.0f + maxDistanceFromOrigin), -3, 1.5f, -(maxDistanceFromOrigin - 1), 0, 0.9f);
		rcSetLightPosition(-0, -10, 10);
	}
	rcDrawAllObjects((dxSpace *)odeGetSpace());

	character.debugVisualize();

	//state transitions
	if (willFall)
	{
		rcPrintString("Falling predicted!");
		if (enableRecoveryMode)
			inRecoveryMode = true;
	}
	if (!willFall && recovered())
		inRecoveryMode = false;
	if (inRecoveryMode)
		rcPrintString("Recovery mode with stronger and larger movements.");

	//turn on real-time mode automatically when learned enough
	if (frameIdx > startRealtimeModeAt)
		realtimeMode = true;

	if (no_settings_exit) {
		exit(0);
	}

	//quit 
	if (frameIdx > autoExitAt)
	{
		std::deque<std::string> settings = flc->get_settings();

		for (std::string setting : settings) {
			comments.push_back(setting);
		}

		comments.push_back("Turns: " + std::to_string(useWalkTargets));

		cost_file_name = get_time_string() + "_costs.csv";
		write_vector_to_file(cost_file_name, costs, comments);
		if (fileExists("out.mp4"))
			remove("out.mp4");
		system("screencaps2mp4.bat");
		exit(0);
	}

	if (captureVideo)
	{
		//During learning, capture first 30 seconds and then 15 seconds at the start of each minutes
		if (realtimeMode || (frameIdx < nInitialCapturedFrames || (frameIdx % (60 * fps) < nFramesToCaptureEveryMinute)))
		{
			rcTakeScreenShot();
		}
	}

	//spheres
	if (useSpheres && (frameIdx > lastSphereLaunchTime + sphereInterval))
	{
		Vector3f com;
		character.computeCOM(com);
		if (useWalkTargets || (com.x() > -maxDistanceFromOrigin + 2.0f) && (com.x() < -0.5f))  //don't throw a sphere right before character is about to be teleported
			throwSphere();
	}

	for (size_t i = 0; i < spheres.size(); i++)
	{
		SphereData &sd = spheres[i];
		if (sd.spawnFrame < frameIdx - 5 * fps)
		{
			setCurrentOdeContext(ALLTHREADS);
			odeGeomDestroy(sd.geom);
			odeBodyDestroy(sd.body);
			saveOdeState(masterContext, masterContext);
			spheres[i] = spheres[spheres.size() - 1];
			spheres.resize(spheres.size() - 1);
		}
	}

	frameIdx++;

}
