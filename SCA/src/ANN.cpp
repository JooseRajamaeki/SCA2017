#include "ANN.h"

static inline float randu() {
	return (float)rand() / (float)RAND_MAX;
}

static inline void stable_exp(float& num) {
	const float cut_off_hi = 60.0f;
	const float cut_off_lo = 0.00000001f;

	float abs_tmp = std::abs(num);

	float sign = 0.0f;
	if (num >= 0.0) {
		sign = 1.0f;
	}
	else {
		sign = -1.0f;
	}

	num = std::max(cut_off_lo, std::min(cut_off_hi, abs_tmp)) * sign;
	num = std::exp(num);

}

static inline void outer_product(Eigen::MatrixXf& product, const Eigen::VectorXf& left, const Eigen::VectorXf& right) {

	int rows = left.size();
	int cols = right.size();

	product.resize(rows, cols);

	for (int row = 0; row < rows; row++) {
		for (int col = 0; col < cols; col++) {
			product(row, col) = left(row)*right(col);
		}
	}

}

Operation::~Operation()
{
}

Operation::Operation() {

}

void Operation::perform_operation()
{
	ready_ = true;
}

void Operation::get_gradients()
{
	ready_ = true;
	gradient_.resize(size_);
	gradient_.setOnes();
}

void Operation::zero_delta()
{
	deltas_.setZero();
}

void Operation::zero_activations()
{
	activations_.setZero();
}

void Operation::set_activation(const float* in)
{
	for (int i = 0; i < size_; i++) {
		activations_[i] = in[i];
	}
}

void Operation::init(unsigned dimension, int operation_index)
{
	index_ = operation_index;
	size_ = dimension;
	activations_ = Eigen::VectorXf::Zero(dimension);
	outputs_ = Eigen::VectorXf::Zero(dimension);
	deltas_ = Eigen::VectorXf::Zero(dimension);
	gradient_ = Eigen::VectorXf::Zero(dimension);

}

void Operation::prepare_forward()
{
	ready_ = false;
	activations_.resize(size_);
	activations_.setZero();
	outputs_.resize(size_);
	outputs_.setZero();
}

void Operation::prepare_backward() {
	ready_ = false;
	deltas_.resize(size_);
	deltas_.setZero();
	gradient_.resize(size_);
	gradient_.setZero();
}

void Operation::increase_dimension()
{
	size_++;

	//Resizing everything just in case.
	prepare_forward();
	prepare_backward();
}

void Operation::write_to_file(std::ofstream& file)
{
	file << std::setprecision(std::numeric_limits<double>::digits10 + 2);
	file << "//OPERATION BEGIN" << std::endl;
	file << type_ << std::endl;
	file << index_ << std::endl;
	file << ready_ << std::endl;
	file << size_ << std::endl;
}

void Operation::load_from_file(std::ifstream& filename)
{
	std::string line;
	std::getline(filename, line);

	{
		std::istringstream ss(line);
		int tmp;
		ss >> tmp;
		type_ = (OperationType)tmp;
		std::getline(filename, line);
	}

	{
		std::istringstream ss(line);
		ss >> index_;
		std::getline(filename, line);
	}

	{
		std::istringstream ss(line);
		ss >> ready_;
		std::getline(filename, line);
	}

	{
		std::istringstream ss(line);
		ss >> size_;
	}

	activations_.resize(size_);
	outputs_.resize(size_);
	gradient_.resize(size_);
	deltas_.resize(size_);

	activations_.setZero();
	outputs_.setZero();
	gradient_.setZero();
	deltas_.setZero();
}

void Connection::standard_init()
{
	adam_sample_ = 0;

	input_node_index_ = -1;
	output_node_index_ = -1;

	forward_path_ = true;
	backward_path_ = true;
	identity_connection_ = false;
	fixed_connection_ = false;

	in_operation_ = nullptr;
	out_operation_ = nullptr;

	input_dimension_ = -1;
	output_dimension_ = -1;
}

Connection::Connection()
{
	standard_init();
}

Connection::Connection(int input_oper_idx, int output_oper_idx)
{
	standard_init();
	input_node_index_ = input_oper_idx;
	output_node_index_ = output_oper_idx;
}

Connection::Connection(const Connection & other)
{
	copy_values(other);
}

Connection Connection::operator=(const Connection & other)
{
	copy_values(other);
	return *this;
}

void Connection::copy_values(const Connection & other)
{
	input_node_index_ = other.input_node_index_;
	output_node_index_ = other.output_node_index_;

	input_dimension_ = other.input_dimension_;
	output_dimension_ = other.output_dimension_;

	forward_path_ = other.forward_path_;
	backward_path_ = other.backward_path_;
	fixed_connection_ = other.fixed_connection_;
	identity_connection_ = other.identity_connection_;

	ready_ = other.ready_;

	//input_delta_ = other.input_delta_;
	input_tmp_ = other.input_tmp_;
	output_tmp_ = other.output_tmp_;

	input_map_ = other.input_map_;
	output_map_ = other.output_map_;

	weights_ = other.weights_;
	prev_weights_ = other.prev_weights_;
	prev_prev_weights_ = other.prev_prev_weights_;

	gradients_ = other.gradients_;

	gradient_samples_ = other.gradient_samples_;
	gradients_sum_ = other.gradients_sum_;

	gradients_first_moment_ = other.gradients_first_moment_;
	gradients_second_moment_ = other.gradients_second_moment_;
	adam_sample_ = other.adam_sample_;

	last_gradients_ = other.last_gradients_;
	gradient_signs_ = other.gradient_signs_;
	update_values_ = other.update_values_;


	in_operation_ = nullptr;
	out_operation_ = nullptr;
	dependencies_forward_.clear();
	dependencies_backward_.clear();
}

void Connection::set_identity()
{
	identity_connection_ = true;
	weights_.resize(0, 0);
}

void Connection::connect_to_operations(const std::vector<Operation*>& operations)
{

	out_operation_ = nullptr;
	for (const Operation* oper : operations) {
		if (oper->index_ == output_node_index_) {
			out_operation_ = (Operation*)oper;
		}
	}

	assert(out_operation_);

	in_operation_ = nullptr;
	for (const Operation* oper : operations) {
		if (oper->index_ == input_node_index_) {
			in_operation_ = (Operation*)oper;
		}
	}

	assert(in_operation_);

}





bool Connection::form_output(float drop_out_stdev)
{
	if (!forward_path_) {
		return true;
	}



	for (Connection* conn : dependencies_forward_) {
		if (!(conn->ready_)) {
			return false;
		}
	}



	if (!(in_operation_->ready_)) {
		if (drop_out_stdev > 0.0f) {
			in_operation_->outputs_ = in_operation_->activations_;

			BoxMuller(in_operation_->outputs_);
			in_operation_->outputs_ *= drop_out_stdev;
			in_operation_->outputs_.array() += 1.0f; //This is now noise from distribution N(1,drop_out_stdev^2)

			in_operation_->activations_.array() *= in_operation_->outputs_.array();
		}

		in_operation_->perform_operation();
	}



	const Eigen::VectorXf* in_vec = nullptr;
	if (in_operation_->size_ == input_dimension_) {
		in_vec = &in_operation_->outputs_;
	}
	else {
		input_tmp_.resize(input_dimension_);
		//Gather inputs
		for (int i = 0; i < input_dimension_; i++) {
			input_tmp_[i] = in_operation_->outputs_[input_map_[i]];
		}
		in_vec = &input_tmp_;
	}



	if (out_operation_->size_ == output_dimension_) {
		if (identity_connection_) {
			out_operation_->activations_ += (*in_vec);
		}
		else {
			out_operation_->activations_ += weights_ * (*in_vec);
		}
	}
	else {
		//Form outputs with inputs
		if (identity_connection_) {
			output_tmp_ = (*in_vec);
		}
		else {
			output_tmp_ = weights_ * (*in_vec);
		}
		//Distribute outputs
		for (int i = 0; i < output_dimension_; i++) {
			out_operation_->activations_[output_map_[i]] += output_tmp_[i];
		}
	}



	ready_ = true;
	return true;
}





void Connection::resize(int input_dimension, int output_dimension)
{
	if (input_dimension <= 0) {
		input_dimension_ = in_operation_->size_;
	}
	else {
		input_dimension_ = input_dimension;
	}

	if (output_dimension <= 0) {
		output_dimension_ = out_operation_->size_;
	}
	else {
		output_dimension_ = output_dimension;
	}


	if (identity_connection_) {
		fixed_connection_ = true;
		weights_ = Eigen::MatrixXf::Identity(0, 0);
	}
	else {
		weights_.conservativeResize(output_dimension_, input_dimension_);

		input_tmp_ = Eigen::VectorXf::Zero(input_dimension_);
		output_tmp_ = Eigen::VectorXf::Zero(output_dimension_);
	}


	input_map_.clear();
	output_map_.clear();

	if (in_operation_) {
		//Fully connected to input
		if (input_dimension_ == in_operation_->size_) {
			for (int i = 0; i < input_dimension_; i++) {
				input_map_[i] = i;
			}
		}
	}

	if (out_operation_) {
		//Fully connected to output
		if (output_dimension_ == out_operation_->size_) {
			for (int i = 0; i < output_dimension_; i++) {
				output_map_[i] = i;
			}
		}
	}

	gradients_sum_ = Eigen::MatrixXf::Zero(0, 0);

	gradients_first_moment_ = Eigen::MatrixXf::Zero(0, 0);
	gradients_second_moment_ = Eigen::MatrixXf::Zero(0, 0);

	last_gradients_ = Eigen::MatrixXf::Zero(0, 0);
	update_values_ = Eigen::MatrixXf::Zero(0, 0);
	prev_weights_ = Eigen::MatrixXf::Zero(0, 0);
	prev_prev_weights_ = Eigen::MatrixXf::Zero(0, 0);

	adam_sample_ = 0;

}

void Connection::duplicate_output(int dimension)
{

	int in_dim = input_dimension_;
	int out_dim = output_dimension_;

	out_dim++;

	resize(in_dim, out_dim);

	int cols = weights_.cols();

	for (int row = out_dim - 1; row > dimension; row--) {
		for (int col = 0; col < cols; col++) {
			weights_(row,col) = weights_(row - 1,col);
		}
	}

}

void Connection::duplicate_input(int dimension)
{

	int in_dim = input_dimension_;
	int out_dim = output_dimension_;

	in_dim++;

	resize(in_dim, out_dim);

	int rows = weights_.rows();

	for (int col = in_dim - 1; col > dimension; col--) {
		for (int row = 0; row < rows; row++) {
			weights_(row,col) = weights_(row,col - 1);
		}
	}

	weights_.col(dimension) *= 0.5f;
	weights_.col(dimension + 1) *= 0.5f;

}


bool Connection::compute_deltas()
{


	for (Connection* conn : dependencies_backward_) {
		if (!(conn->ready_)) {
			return false;
		}
	}

	if (!backward_path_) {
		ready_ = true;
		return true;
	}


	//If the out operation is a terminal operation and has not formed the delta, we need to form it.
	if (!out_operation_->ready_) {
		out_operation_->get_gradients();
		out_operation_->deltas_.array() *= out_operation_->gradient_.array();
		out_operation_->ready_ = true;
	}



	const Eigen::VectorXf* delta = &out_operation_->deltas_;
	if (out_operation_->deltas_.size() != output_dimension_) {
		//Using here output_vector_ as a temporary container. It's not needed for anything else right now.
		output_tmp_.resize(output_dimension_);
		for (int i = 0; i < output_dimension_; i++) {
			output_tmp_[i] = out_operation_->deltas_[output_map_[i]];
		}
		delta = &output_tmp_;
	}



	if (in_operation_->deltas_.size() == input_dimension_) {
		if (identity_connection_) {
			in_operation_->deltas_ += *delta;
		}
		else {
			//Back propagate deltas
			in_operation_->deltas_ += weights_.transpose() * (*delta);
		}
	}
	else {
		if (identity_connection_) {
			input_tmp_ = *delta;
		}
		else {
			//Back propagate deltas
			input_tmp_ = weights_.transpose() * (*delta);
		}
		for (int i = 0; i < input_dimension_; i++) {
			in_operation_->deltas_[input_map_[i]] += input_tmp_[i];
		}
	}

	ready_ = true;
	return true;

}

void Connection::compute_gradients()
{

	if (fixed_connection_) {
		return;
	}

	const Eigen::VectorXf* delta = &out_operation_->deltas_;
	const Eigen::VectorXf* input = &in_operation_->outputs_;

	if (output_dimension_ != out_operation_->deltas_.size()) {
		output_tmp_.resize(output_dimension_);
		for (int i = 0; i < output_dimension_; i++) {
			output_tmp_[i] = (*delta)[output_map_[i]];
		}
		delta = &output_tmp_;
	}

	if (input_dimension_ != in_operation_->size_) {
		input_tmp_.resize(input_dimension_);
		for (int i = 0; i < input_dimension_; i++) {
			input_tmp_[i] = (*input)[input_map_[i]];
		}
		input = &input_tmp_;
	}

	//gradients_ = delta*input.transpose();
	outer_product(gradients_, *delta, *input);

}

void Connection::zero_accumulated_gradients()
{

	gradient_samples_ = 0;
	if (gradients_sum_.rows() != weights_.rows() || gradients_sum_.cols() != weights_.cols()) {
		gradients_sum_.resize(weights_.rows(), weights_.cols());
	}
	gradients_sum_.setZero();

}

void Connection::accumulate_gradients(bool increase_counter)
{
	if (fixed_connection_) {
		return;
	}


	gradients_sum_ += gradients_;
	if (increase_counter) {
		gradient_samples_++;
	}

}

void Connection::apply_gradients(const float& learning_rate)
{
	if (fixed_connection_) {
		return;
	}

	if (gradient_samples_ > 0) {
		weights_ -= (learning_rate / (float)gradient_samples_) * gradients_sum_;
	}

}

void Connection::randomize_weights(const float min_val, const float max_val)
{

	if (identity_connection_) {
		weights_.resize(0, 0);
		return;
	}

	unsigned rows = weights_.rows();
	unsigned cols = weights_.cols();

	for (unsigned row = 0; row < rows; row++) {
		for (unsigned col = 0; col < cols; col++) {
			float& num = weights_(row, col);

			num = (float)rand() / (float)RAND_MAX;

			num = min_val + num*(max_val - min_val);

		}
	}

}

void Connection::randomize_weights_gaussian(const float stdev)
{

	if (identity_connection_) {
		weights_.resize(0, 0);
		return;
	}

	unsigned rows = weights_.rows();
	unsigned cols = weights_.cols();

	for (unsigned row = 0; row < rows; row++) {
		for (unsigned col = 0; col < cols; col++) {
			float& num = weights_(row, col);

			BoxMuller<float>(&num,1);
			num *= stdev;
		}
	}

}



void Connection::apply_adam(const float learning_rate, const float first_moment_smoothing, const float second_moment_smoothing, const float epsilon)
{
	if (fixed_connection_) {
		return;
	}

	float local_learning_rate = learning_rate;
	local_learning_rate *= std::sqrt(1.0f - std::pow(second_moment_smoothing, (float)adam_sample_));
	local_learning_rate /= 1.0f - std::pow(first_moment_smoothing, (float)adam_sample_);

	gradients_sum_ *= 1.0f / (float)gradient_samples_;

	if (gradients_first_moment_.rows() == 0) {
		adam_sample_ = 0;
		gradients_first_moment_ = gradients_sum_*0.0f;
		gradients_second_moment_ = gradients_sum_*0.0f;
	}

	adam_sample_++;

	int rows = weights_.rows();
	int cols = weights_.cols();

	for (int row = 0; row < rows; row++) {
		for (int col = 0; col < cols; col++) {


			float& gradient = gradients_sum_(row, col);

			gradients_first_moment_(row, col) *= first_moment_smoothing;
			gradients_first_moment_(row, col) += (1.0f - first_moment_smoothing)*gradient;

			gradients_second_moment_(row, col) *= second_moment_smoothing;
			gradients_second_moment_(row, col) += (1.0f - second_moment_smoothing)*gradient*gradient;




			////////////////////////////////////

			float& weight = weights_(row, col);

			//Denominator first
			float update = std::sqrt(gradients_second_moment_(row, col));
			update = std::max(update, std::numeric_limits<float>::min());


			update = local_learning_rate * gradients_first_moment_(row, col) / update;

			if (!(update > std::numeric_limits<float>::lowest() && update < std::numeric_limits<float>::max())) {
				update = 0;
			}

			weight -= update;
		}
	}


	assert(has_valid_nums(weights_));

}

void Connection::apply_adamax(const float learning_rate, const float first_moment_smoothing, const float second_moment_smoothing, const float epsilon)
{
	if (fixed_connection_) {
		return;
	}

	int rows = weights_.rows();
	int cols = weights_.cols();

	//Accumulate
	gradients_sum_ *= 1.0f / (float)gradient_samples_;

	if (gradients_first_moment_.rows() == 0) {
		adam_sample_ = 0;
		gradients_first_moment_ = gradients_sum_*0.0f;
		gradients_second_moment_ = gradients_sum_*0.0f;
	}

	adam_sample_++;

	float unbias = 1.0f - std::pow(first_moment_smoothing, (float)adam_sample_);
	unbias = 1.0f / unbias;

	const float local_learning_rate = unbias*learning_rate;

	for (int row = 0; row < rows; row++) {
		for (int col = 0; col < cols; col++) {

			const float& gradient = gradients_sum_(row, col);

			gradients_first_moment_(row, col) *= first_moment_smoothing;
			gradients_first_moment_(row, col) += (1.0f - first_moment_smoothing)*gradient;

			float& second_moment = gradients_second_moment_(row, col);
			second_moment = std::max(second_moment*second_moment_smoothing, std::abs(gradient));

			if (!(second_moment >= 0.0f && second_moment <= std::numeric_limits<float>::max())) {
				second_moment = std::numeric_limits<float>::max();
			}


			////////////////////////////////////

			//Perform updates

			float before_update = weights_(row, col);
			float& num = weights_(row, col);

			num -= local_learning_rate*gradients_first_moment_(row, col) / gradients_second_moment_(row, col);

			if (!(num > std::numeric_limits<float>::lowest() && num < std::numeric_limits<float>::max())) {
				num = before_update;
			}
		}
	}

	assert(has_valid_nums(weights_));
}

//The RPROP+ algorithm
void Connection::apply_rprop(float min_val, float max_val)
{

	if (fixed_connection_) {
		return;
	}

	prev_prev_weights_ = prev_weights_;
	prev_weights_ = weights_;

	const float increase_value = 1.2f;
	const float decrease_value = 0.5f;



	if (last_gradients_.rows() != 0) {

		gradient_signs_ = last_gradients_.array() * gradients_sum_.array();

		unsigned rows = weights_.rows();
		unsigned cols = weights_.cols();

		for (unsigned row = 0; row < rows; row++) {
			for (unsigned col = 0; col < cols; col++) {

				float& grad = gradients_sum_(row, col);
				const float& sign = gradient_signs_(row, col);
				float& update_value = update_values_(row, col);

				if (sign > 0.0f) {
					update_value *= increase_value;
					update_value = std::min(max_val, update_value);

					if (grad >= 0) {
						weights_(row, col) -= update_value;
					}

					if (grad < 0) {
						weights_(row, col) += update_value;
					}
				}

				if (sign < 0.0f) {
					update_value *= decrease_value;
					update_value = std::max(min_val, update_value);

					if (prev_prev_weights_.rows() == gradients_sum_.rows() && prev_prev_weights_.cols() == gradients_sum_.cols()) {
						weights_(row, col) = prev_prev_weights_(row, col);
					}

					grad = 0;
				}

				if (sign == 0.0f) {
					if (grad >= 0) {
						weights_(row, col) -= update_value;
					}

					if (grad < 0) {
						weights_(row, col) += update_value;
					}
				}

			}
		}

	}
	else {
		update_values_ = Eigen::MatrixXf::Random(gradients_sum_.rows(), gradients_sum_.cols()).cwiseAbs()*0.001f;
	}

	last_gradients_ = gradients_sum_;

}

void Connection::apply_rmsprop(float learning_rate)
{

	if (fixed_connection_) {
		return;
	}

	const float smoothing = 0.9f;

	gradients_sum_ /= (float)gradient_samples_;

	if (gradients_second_moment_.rows() == 0) {
		gradients_second_moment_ = gradients_sum_.cwiseAbs2();
	}
	else {
		gradients_second_moment_ *= smoothing;
		gradients_second_moment_ += (1.0f - smoothing)*gradients_sum_.cwiseAbs2();
	}

	unsigned rows = gradients_second_moment_.rows();
	unsigned cols = gradients_second_moment_.cols();

	for (unsigned row = 0; row < rows; row++) {
		for (unsigned col = 0; col < cols; col++) {
			float& grad = gradients_sum_(row, col);

			grad /= std::sqrt(gradients_second_moment_(row, col));

			if (grad - grad != grad - grad) {
				grad = 0;
			}

		}
	}

	weights_ -= learning_rate*gradients_sum_;

}


void Connection::clamp_weights(float min_weight, float max_weight)
{

	if (fixed_connection_) {
		return;
	}

	unsigned rows = weights_.rows();
	unsigned cols = weights_.cols();

	for (unsigned row = 0; row < rows; row++) {
		for (unsigned col = 0; col < cols; col++) {
			float& weight = weights_(row, col);

			weight = std::min(weight, max_weight);
			weight = std::max(weight, min_weight);
		}
	}

}

void Connection::write_matrix_to_file(std::ofstream& filename, Eigen::MatrixXf& matrix_to_write)
{
	filename << std::setprecision(std::numeric_limits<double>::digits10 + 2);
	filename << matrix_to_write.rows() << " " << matrix_to_write.cols() << std::endl;
	for (int row = 0; row < matrix_to_write.rows(); row++) {
		for (int col = 0; col < matrix_to_write.cols(); col++) {
			filename << matrix_to_write(row, col) << " ";
		}
	}
	filename << std::endl;
}

Eigen::MatrixXf Connection::load_matrix_from_file(std::ifstream& filename)
{

	unsigned rows = 0;
	unsigned cols = 0;

	std::string line;
	std::getline(filename, line);

	std::stringstream ss(line);
	ss >> rows;
	ss >> cols;

	Eigen::MatrixXf read_matrix;
	read_matrix.resize(rows, cols);

	std::getline(filename, line);
	ss = std::stringstream(line);

	for (unsigned row = 0; row < rows; row++) {
		for (unsigned col = 0; col < cols; col++) {
			ss >> read_matrix(row, col);
		}
	}

	return read_matrix;

}

static void write_vector_to_file(std::ofstream & filename, Eigen::VectorXf & matrix_to_write)
{
	filename << std::setprecision(std::numeric_limits<double>::digits10 + 2);
	filename << matrix_to_write.size() << std::endl;
	for (int row = 0; row < matrix_to_write.rows(); row++) {
		filename << matrix_to_write(row) << " ";
	}
	filename << std::endl;
}

static Eigen::VectorXf load_vector_from_file(std::ifstream & filename)
{
	unsigned rows = 0;

	std::string line;
	std::getline(filename, line);

	std::stringstream ss(line);
	ss >> rows;



	Eigen::VectorXf read_matrix;
	read_matrix.resize(rows);

	std::getline(filename, line);
	ss = std::stringstream(line);

	for (unsigned row = 0; row < rows; row++) {
		ss >> read_matrix(row);
	}


	return read_matrix;
}

void Connection::write_map_to_file(std::ofstream & filename, std::map<int, int>& map_to_write)
{
	filename << std::setprecision(std::numeric_limits<double>::digits10 + 2);
	filename << map_to_write.size() << std::endl;
	for (std::map<int, int>::iterator iter = map_to_write.begin(); iter != map_to_write.end(); iter++) {
		int key = iter->first;
		int value = iter->second;
		filename << std::to_string(key) << " " << std::to_string(value) << " ";
	}
	filename << std::endl;
}

std::map<int, int> Connection::load_map_from_file(std::ifstream & filename)
{
	std::map<int, int> read_map;

	std::string line;
	std::getline(filename, line);

	std::stringstream ss(line);
	unsigned entries = 0;
	ss >> entries;

	std::getline(filename, line);
	ss = std::stringstream(line);

	for (unsigned row = 0; row < entries; row++) {
		int key = 0;
		int value = -1;
		ss >> key;
		ss >> value;
		read_map[key] = value;
	}

	return read_map;
}

void Connection::write_to_file(std::ofstream& filename)
{
	filename << "//CONNECTION BEGIN" << std::endl;
	filename << input_node_index_ << std::endl;
	filename << output_node_index_ << std::endl;
	filename << forward_path_ << std::endl;
	filename << backward_path_ << std::endl;
	filename << fixed_connection_ << std::endl;
	filename << identity_connection_ << std::endl;
	filename << ready_ << std::endl;
	filename << input_dimension_ << std::endl;
	filename << output_dimension_ << std::endl;

	if (gradient_samples_ < 0) {
		gradient_samples_ = 0;
	}

	filename << gradient_samples_ << std::endl;
	filename << adam_sample_ << std::endl;

	write_matrix_to_file(filename, weights_);
	write_matrix_to_file(filename, prev_weights_);
	write_matrix_to_file(filename, prev_prev_weights_);
	write_matrix_to_file(filename, gradients_);
	write_matrix_to_file(filename, gradients_sum_);
	write_matrix_to_file(filename, gradients_first_moment_);
	write_matrix_to_file(filename, gradients_second_moment_);
	write_matrix_to_file(filename, last_gradients_);
	write_matrix_to_file(filename, gradient_signs_);
	write_matrix_to_file(filename, update_values_);
	write_map_to_file(filename, input_map_);
	write_map_to_file(filename, output_map_);
}

void Connection::load_from_file(std::ifstream& filename)
{
	dependencies_forward_.clear();
	dependencies_backward_.clear();
	in_operation_ = nullptr;
	out_operation_ = nullptr;

	std::string line;
	std::getline(filename, line);
	input_node_index_ = std::stoi(line);

	std::getline(filename, line);
	output_node_index_ = std::stoi(line);

	std::getline(filename, line);
	forward_path_ = (bool)std::stoi(line);

	std::getline(filename, line);
	backward_path_ = (bool)std::stoi(line);

	std::getline(filename, line);
	fixed_connection_ = (bool)std::stoi(line);

	std::getline(filename, line);
	identity_connection_ = (bool)std::stoi(line);

	std::getline(filename, line);
	ready_ = (bool)std::stoi(line);

	std::getline(filename, line);
	input_dimension_ = std::stoi(line);

	std::getline(filename, line);
	output_dimension_ = std::stoi(line);

	std::getline(filename, line);
	gradient_samples_ = std::stoi(line);

	std::getline(filename, line);
	adam_sample_ = std::stoi(line);

	weights_ = load_matrix_from_file(filename);
	prev_weights_ = load_matrix_from_file(filename);
	prev_prev_weights_ = load_matrix_from_file(filename);
	gradients_ = load_matrix_from_file(filename);
	gradients_sum_ = load_matrix_from_file(filename);
	gradients_first_moment_ = load_matrix_from_file(filename);
	gradients_second_moment_ = load_matrix_from_file(filename);
	last_gradients_ = load_matrix_from_file(filename);
	gradient_signs_ = load_matrix_from_file(filename);
	update_values_ = load_matrix_from_file(filename);
	input_map_ = load_map_from_file(filename);
	output_map_ = load_map_from_file(filename);
}

MultiLayerPerceptron::MultiLayerPerceptron()
{
	training_ = false;
	drop_out_stdev_ = 0.0f;

	learning_rate_ = 0.001f;
	connections_ = std::vector<Connection>();
	operations_ = std::vector<std::unique_ptr<Operation> >();
	input_operation_ = nullptr;
	output_operation_ = nullptr;

	min_weight_ = std::numeric_limits<float>::lowest();
	max_weight_ = std::numeric_limits<float>::max();

	adam_first_moment_smoothing_ = 0.8f;
	adam_second_moment_smoothing_ = 0.9f;

	max_gradient_norm_ = std::numeric_limits<float>::max();

	mse_ = std::numeric_limits<float>::infinity();
	prev_mse_ = std::numeric_limits<float>::infinity();

	use_input_scaling_ = false;

}

void MultiLayerPerceptron::copy(const MultiLayerPerceptron & other)
{
	training_ = false;
	drop_out_stdev_ = other.drop_out_stdev_;

	operations_.clear();
	connections_.clear();

	input_mean_ = other.input_mean_;
	input_stdev_ = other.input_stdev_;

	input_operation_ = nullptr;
	output_operation_ = nullptr;

	learning_rate_ = other.learning_rate_;

	min_weight_ = other.min_weight_;
	max_weight_ = other.max_weight_;

	adam_first_moment_smoothing_ = other.adam_first_moment_smoothing_;
	adam_second_moment_smoothing_ = other.adam_second_moment_smoothing_;

	mse_ = other.mse_;
	prev_mse_ = other.prev_mse_;

	max_gradient_norm_ = other.max_gradient_norm_;

	copy_operations(other.operations_);

	for (const Connection& other_conn : other.connections_) {
		connections_.push_back(Connection(other_conn));
	}

	//Connecting the operations to connections.
	std::vector<Operation*> oper_ptr;
	for (std::unique_ptr<Operation>& oper : operations_) {
		oper_ptr.push_back(oper.get());
	}
	for (Connection& conn : connections_) {
		conn.connect_to_operations(oper_ptr);
	}

	build_dependencies();

	for (std::unique_ptr<Operation>& oper : operations_) {
		if (oper->index_ == other.input_operation_->index_) {
			input_operation_ = oper.get();
		}

		if (oper->index_ == other.output_operation_->index_) {
			output_operation_ = oper.get();
		}
	}

	use_input_scaling_ = other.use_input_scaling_;

}

MultiLayerPerceptron::MultiLayerPerceptron(const MultiLayerPerceptron & other)
{
	copy(other);
}

MultiLayerPerceptron MultiLayerPerceptron::operator=(const MultiLayerPerceptron & other)
{
	copy(other);
	return *this;
}

void MultiLayerPerceptron::copy_operations(const std::vector<std::unique_ptr<Operation>>& operations)
{
	operations_.clear();
	for (const std::unique_ptr<Operation>& operation : operations) {
		switch (operation->type_) {

		case Operation::OperationType::MIRROR:
		{
			std::unique_ptr<MirrorOperation> tmp = std::unique_ptr<MirrorOperation>(new MirrorOperation());
			void* oper = operation.get();
			*tmp = *((MirrorOperation*)oper);
			operations_.push_back(std::move(tmp));
		}
		break;

		case Operation::OperationType::ELU:
		{
			std::unique_ptr<ELUOperation> tmp = std::unique_ptr<ELUOperation>(new ELUOperation());
			void* oper = operation.get();
			*tmp = *((ELUOperation*)oper);
			operations_.push_back(std::move(tmp));
		}
		break;


		case Operation::OperationType::SELU:
		{
			std::unique_ptr<SELUOperation> tmp = std::unique_ptr<SELUOperation>(new SELUOperation());
			void* oper = operation.get();
			*tmp = *((SELUOperation*)oper);
			operations_.push_back(std::move(tmp));
		}
		break;

		case Operation::OperationType::BSELU:
		{
			std::unique_ptr<BSELUOperation> tmp = std::unique_ptr<BSELUOperation>(new BSELUOperation());
			void* oper = operation.get();
			*tmp = *((BSELUOperation*)oper);
			operations_.push_back(std::move(tmp));
		}
		break;

		case Operation::OperationType::EVOLVING_UNITY:
		{
			std::unique_ptr<EvolvingUnityOperation> tmp = std::unique_ptr<EvolvingUnityOperation>(new EvolvingUnityOperation());
			void* oper = operation.get();
			*tmp = *((EvolvingUnityOperation*)oper);
			operations_.push_back(std::move(tmp));
		}
		break;

		case Operation::OperationType::IDENTITY:
		{
			std::unique_ptr<IdentityOperation> tmp = std::unique_ptr<IdentityOperation>(new IdentityOperation());
			void* oper = operation.get();
			*tmp = *((IdentityOperation*)oper);
			operations_.push_back(std::move(tmp));
		}
		break;

		case Operation::OperationType::LINEAR_CLIPPED:
		{
			std::unique_ptr<LinearClippedOperation> tmp = std::unique_ptr<LinearClippedOperation>(new LinearClippedOperation());
			void* oper = operation.get();
			*tmp = *((LinearClippedOperation*)oper);
			operations_.push_back(std::move(tmp));
		}
		break;

		case Operation::OperationType::LOG:
		{
			std::unique_ptr<LogOperation> tmp = std::unique_ptr<LogOperation>(new LogOperation());
			void* oper = operation.get();
			*tmp = *((LogOperation*)oper);
			operations_.push_back(std::move(tmp));
		}
		break;

		case Operation::OperationType::NEGATE:
		{
			std::unique_ptr<NegateOperation> tmp = std::unique_ptr<NegateOperation>(new NegateOperation());
			void* oper = operation.get();
			*tmp = *((NegateOperation*)oper);
			operations_.push_back(std::move(tmp));
		}
		break;


		case Operation::OperationType::SIGMOID:
		{
			std::unique_ptr<SigmoidOperation> tmp = std::unique_ptr<SigmoidOperation>(new SigmoidOperation());
			void* oper = operation.get();
			*tmp = *((SigmoidOperation*)oper);
			operations_.push_back(std::move(tmp));
		}
		break;

		case Operation::OperationType::SOFTPLUS:
		{
			std::unique_ptr<SoftPlusOperation> tmp = std::unique_ptr<SoftPlusOperation>(new SoftPlusOperation());
			void* oper = operation.get();
			*tmp = *((SoftPlusOperation*)oper);
			operations_.push_back(std::move(tmp));
		}
		break;


		case Operation::OperationType::SQUARE:
		{
			std::unique_ptr<SquareOperation> tmp = std::unique_ptr<SquareOperation>(new SquareOperation());
			void* oper = operation.get();
			*tmp = *((SquareOperation*)oper);
			operations_.push_back(std::move(tmp));
		}
		break;


		case Operation::OperationType::TANH:
		{
			std::unique_ptr<TanhOperation> tmp = std::unique_ptr<TanhOperation>(new TanhOperation());
			void* oper = operation.get();
			*tmp = *((TanhOperation*)oper);
			operations_.push_back(std::move(tmp));
		}
		break;


		case Operation::OperationType::UNITY:
		{
			std::unique_ptr<UnityOperation> tmp = std::unique_ptr<UnityOperation>(new UnityOperation());
			void* oper = operation.get();
			*tmp = *((UnityOperation*)oper);
			operations_.push_back(std::move(tmp));
		}
		break;

		}
	}
}

void MultiLayerPerceptron::build_dependencies()
{

	for (Connection& conn : connections_) {
		conn.dependencies_forward_.clear();
		conn.dependencies_backward_.clear();
	}

	for (unsigned i = 0; i < connections_.size(); i++) {

		Connection& conn1 = connections_[i];

		for (unsigned j = 0; j < connections_.size(); j++) {

			if (i == j) {
				continue;
			}

			Connection& conn2 = connections_[j];

			if (conn1.output_node_index_ == conn2.input_node_index_) {
				conn2.dependencies_forward_.push_back(&conn1);
				conn1.dependencies_backward_.push_back(&conn2);
			}

		}

	}

}

void MultiLayerPerceptron::reset_training()
{
	for (Connection& conn : connections_) {
		conn.prev_prev_weights_.resize(0, 0);
		conn.prev_weights_.resize(0, 0);
		conn.adam_sample_ = 0;
		conn.update_values_.resize(0, 0);
		conn.last_gradients_.resize(0, 0);
	}
}

void MultiLayerPerceptron::build_network(const std::vector<unsigned>& layer_widths)
{

	connections_.clear();
	operations_.clear();


	input_mean_.resize(layer_widths[0]);
	input_mean_.setZero();
	input_stdev_ = input_mean_;
	input_stdev_.setOnes();


	int operation_idx = 0;
	int previous_layer_idx = -1;
	for (unsigned i = 0; i < layer_widths.size(); i++) {

		if (i == 0) {//Input layer

			operations_.push_back(std::unique_ptr<Operation>(new IdentityOperation()));
			operations_.back()->init(layer_widths[i], operation_idx);

			input_operation_ = operations_.back().get();
			previous_layer_idx = operation_idx;
			operation_idx++;
		}
		else if (i == layer_widths.size() - 1) {//Output layer


			//Bias
			int bias_idx = operation_idx;
			operations_.push_back(std::unique_ptr<Operation>(new EvolvingUnityOperation()));
			operations_.back()->init(1, operation_idx);
			operation_idx++;

			//Output
			operations_.push_back(std::unique_ptr<Operation>(new IdentityOperation()));
			operations_.back()->init(layer_widths[i], operation_idx);

			output_operation_ = operations_.back().get();

			connections_.push_back(Connection(bias_idx, operation_idx));
			connections_.push_back(Connection(previous_layer_idx, operation_idx));
			operation_idx++;
		}
		else {//Hidden layers


			//Bias
			int bias_idx = operation_idx;
			operations_.push_back(std::unique_ptr<Operation>(new EvolvingUnityOperation()));
			operations_.back()->init(1, operation_idx);
			operation_idx++;

			//Hidden layer
			operations_.push_back(std::unique_ptr<Operation>(new ELUOperation()));
			operations_.back()->init(layer_widths[i], operation_idx);

			connections_.push_back(Connection(bias_idx, operation_idx));
			connections_.push_back(Connection(previous_layer_idx, operation_idx));

			previous_layer_idx = operation_idx;
			operation_idx++;

		}
	}


	std::vector<Operation*> oper_ptr;
	for (std::unique_ptr<Operation>& oper : operations_) {
		oper_ptr.push_back(oper.get());
	}
	for (Connection& conn : connections_) {
		conn.connect_to_operations(oper_ptr);
		conn.resize();
	}
	build_dependencies();

	float standard_min_weight = -0.1f;
	float standard_max_weight = 0.1f;
	randomize_weights(standard_min_weight, standard_max_weight);

}

void MultiLayerPerceptron::build_input_connected_network(const std::vector<unsigned>& layer_widths)
{
	connections_.clear();
	operations_.clear();

	int operation_idx = 0;
	int previous_layer_idx = -1;
	int input_layer_idx = -1;
	for (unsigned i = 0; i < layer_widths.size(); i++) {

		if (i == 0) {//Input layer

			operations_.push_back(std::unique_ptr<Operation>(new IdentityOperation()));
			operations_.back()->init(layer_widths[i], operation_idx);

			input_operation_ = operations_.back().get();
			previous_layer_idx = operation_idx;
			input_layer_idx = operation_idx;
			operation_idx++;
		}
		else if (i == layer_widths.size() - 1) {//Output layer


												//Bias
			int bias_idx = operation_idx;
			operations_.push_back(std::unique_ptr<Operation>(new EvolvingUnityOperation()));
			operations_.back()->init(1, operation_idx);
			operation_idx++;

			//Output
			operations_.push_back(std::unique_ptr<Operation>(new IdentityOperation()));
			operations_.back()->init(layer_widths[i], operation_idx);

			output_operation_ = operations_.back().get();

			connections_.push_back(Connection(bias_idx, operation_idx));
			connections_.push_back(Connection(previous_layer_idx, operation_idx));
			operation_idx++;
		}
		else {//Hidden layers


			  //Bias
			int bias_idx = operation_idx;
			operations_.push_back(std::unique_ptr<Operation>(new EvolvingUnityOperation()));
			operations_.back()->init(1, operation_idx);
			operation_idx++;

			//Hidden layer
			operations_.push_back(std::unique_ptr<Operation>(new ELUOperation()));
			operations_.back()->init(layer_widths[i], operation_idx);

			connections_.push_back(Connection(bias_idx, operation_idx));
			connections_.push_back(Connection(previous_layer_idx, operation_idx));
			connections_.push_back(Connection(input_layer_idx, operation_idx));

			previous_layer_idx = operation_idx;
			operation_idx++;

		}
	}


	std::vector<Operation*> oper_ptr;
	for (std::unique_ptr<Operation>& oper : operations_) {
		oper_ptr.push_back(oper.get());
	}
	for (Connection& conn : connections_) {
		conn.connect_to_operations(oper_ptr);
		conn.resize();
	}
	build_dependencies();

	float standard_min_weight = -0.1f;
	float standard_max_weight = 0.1f;
	randomize_weights(standard_min_weight, standard_max_weight);
}

void MultiLayerPerceptron::build_structured_network(const unsigned input_dim, const unsigned output_dim, const unsigned layer_width, std::vector<std::vector<unsigned>> input_groupings, std::vector<std::vector<unsigned>> output_groupings)
{
	connections_.clear();
	operations_.clear();

	int operation_idx = 0;

	if (output_groupings.size() == 0) {
		output_groupings.push_back(std::vector<unsigned>());
		for (unsigned dim = 0; dim < output_dim; dim++) {
			output_groupings.back().push_back(dim);
		}
	}

	if (input_groupings.size() == 0) {
		input_groupings.push_back(std::vector<unsigned>());
		for (unsigned dim = 0; dim < input_dim; dim++) {
			input_groupings.back().push_back(dim);
		}
	}

	//Input layer
	const int input_layer_idx = operation_idx;
	operations_.push_back(std::unique_ptr<Operation>(new IdentityOperation()));
	operations_.back()->init(input_dim, operation_idx);
	input_operation_ = operations_.back().get();
	operation_idx++;

	//Output layer bias
	const int output_bias_idx = operation_idx;
	operations_.push_back(std::unique_ptr<Operation>(new EvolvingUnityOperation()));
	operations_.back()->init(1, operation_idx);
	operation_idx++;

	//Output layer
	const int output_layer_idx = operation_idx;
	operations_.push_back(std::unique_ptr<Operation>(new IdentityOperation()));
	operations_.back()->init(output_dim, operation_idx);
	output_operation_ = operations_.back().get();

	//Output bias connection
	connections_.push_back(Connection(output_bias_idx, operation_idx));
	operation_idx++;

	std::deque<int> output_idxs;
	std::deque<int> input_idxs;

	std::vector<std::unique_ptr<Operation>> previous_layer;
	std::vector<std::unique_ptr<Operation>> current_layer;

	for (std::vector<unsigned> grouping : input_groupings) {
		//Bias
		int bias_idx = operation_idx;
		operations_.push_back(std::unique_ptr<Operation>(new EvolvingUnityOperation()));
		operations_.back()->init(1, operation_idx);
		operation_idx++;

		//Hidden layer
		previous_layer.push_back(std::unique_ptr<Operation>(new ELUOperation()));
		previous_layer.back()->init(layer_width, operation_idx);

		connections_.push_back(Connection(bias_idx, operation_idx));
		connections_.push_back(Connection(input_layer_idx, operation_idx));
		input_idxs.push_back(operation_idx);

		Connection& input_connection = connections_.back();
		input_connection.resize(grouping.size(), layer_width);
		input_connection.input_map_.clear();
		for (unsigned i = 0; i < grouping.size(); i++) {
			input_connection.input_map_[i] = grouping[i];
		}

		operation_idx++;
	}

	while (previous_layer.size() > 1) {
		current_layer.clear();

		for (int pair_idx = 0; pair_idx < (int)previous_layer.size() - 1; pair_idx++) {

			Operation* previous1 = previous_layer[pair_idx].get();
			Operation* previous2 = previous_layer[pair_idx + 1].get();

			//Bias
			int bias_idx = operation_idx;
			operations_.push_back(std::unique_ptr<Operation>(new EvolvingUnityOperation()));
			operations_.back()->init(1, operation_idx);
			operation_idx++;

			//Hidden layer
			current_layer.push_back(std::unique_ptr<Operation>(new ELUOperation()));
			current_layer.back()->init(layer_width, operation_idx);
			input_idxs.push_back(operation_idx);

			connections_.push_back(Connection(bias_idx, operation_idx));
			connections_.push_back(Connection(previous1->index_, operation_idx));
			connections_.push_back(Connection(previous2->index_, operation_idx));
			operation_idx++;
		}

		for (std::unique_ptr<Operation>& oper : previous_layer) {
			operations_.push_back(std::move(oper));
		}
		previous_layer.clear();

		for (std::unique_ptr<Operation>& oper : current_layer) {
			previous_layer.push_back(std::move(oper));
		}
	}

	std::unique_ptr<Operation> input_joint = std::move(previous_layer.front());
	previous_layer.clear();




	for (std::vector<unsigned> grouping : output_groupings) {
		//Bias
		int bias_idx = operation_idx;
		operations_.push_back(std::unique_ptr<Operation>(new EvolvingUnityOperation()));
		operations_.back()->init(1, operation_idx);
		operation_idx++;

		//Hidden layer
		previous_layer.push_back(std::unique_ptr<Operation>(new ELUOperation()));
		previous_layer.back()->init(layer_width, operation_idx);
		output_idxs.push_back(operation_idx);

		connections_.push_back(Connection(bias_idx, operation_idx));
		connections_.push_back(Connection(operation_idx, output_layer_idx));

		Connection& output_connection = connections_.back();
		output_connection.resize(layer_width, grouping.size());
		output_connection.output_map_.clear();
		for (unsigned i = 0; i < grouping.size(); i++) {
			output_connection.output_map_[i] = grouping[i];
		}

		operation_idx++;
	}



	while (previous_layer.size() > 1) {
		current_layer.clear();

		for (int pair_idx = 0; pair_idx < (int)previous_layer.size() - 1; pair_idx++) {

			Operation* previous1 = previous_layer[pair_idx].get();
			Operation* previous2 = previous_layer[pair_idx + 1].get();

			//Bias
			int bias_idx = operation_idx;
			operations_.push_back(std::unique_ptr<Operation>(new EvolvingUnityOperation()));
			operations_.back()->init(1, operation_idx);
			operation_idx++;

			//Hidden layer
			current_layer.push_back(std::unique_ptr<Operation>(new ELUOperation()));
			current_layer.back()->init(layer_width, operation_idx);
			output_idxs.push_back(operation_idx);

			connections_.push_back(Connection(bias_idx, operation_idx));
			connections_.push_back(Connection(operation_idx, previous1->index_));
			connections_.push_back(Connection(operation_idx, previous2->index_));
			operation_idx++;
		}

		for (std::unique_ptr<Operation>& oper : previous_layer) {
			operations_.push_back(std::move(oper));
		}
		previous_layer.clear();

		for (std::unique_ptr<Operation>& oper : current_layer) {
			previous_layer.push_back(std::move(oper));
		}
	}

	////Connecting input and output side.
	//connections_.push_back(Connection(input_joint->index_, previous_layer.front()->index_));
	operations_.push_back(std::move(input_joint));
	operations_.push_back(std::move(previous_layer.front()));
	previous_layer.clear();

	for (int in_idx : input_idxs) {
		for (int out_idx : output_idxs) {
			connections_.push_back(Connection(in_idx, out_idx));
		}
	}


	std::vector<Operation*> oper_ptr;
	for (std::unique_ptr<Operation>& oper : operations_) {
		oper_ptr.push_back(oper.get());
	}
	for (Connection& conn : connections_) {
		conn.connect_to_operations(oper_ptr);
		if (conn.in_operation_ != input_operation_ && conn.out_operation_ != output_operation_) {
			conn.resize();
		}
		if (conn.in_operation_->type_ == Operation::OperationType::EVOLVING_UNITY && conn.in_operation_->size_ == 1) {
			conn.resize();
		}
	}
	build_dependencies();

	float standard_min_weight = -0.1f;
	float standard_max_weight = 0.1f;
	randomize_weights(standard_min_weight, standard_max_weight);

}

void MultiLayerPerceptron::build_sigmoid_network(const std::vector<unsigned>& layer_widths)
{
	connections_.clear();
	operations_.clear();

	int operation_idx = 0;
	int previous_layer_idx = -1;
	for (unsigned i = 0; i < layer_widths.size(); i++) {

		if (i == 0) {//Input layer

			operations_.push_back(std::unique_ptr<Operation>(new IdentityOperation()));
			operations_.back()->init(layer_widths[i], operation_idx);

			input_operation_ = operations_.back().get();
			previous_layer_idx = operation_idx;
			operation_idx++;
		}
		else if (i == layer_widths.size() - 1) {//Output layer


			//Bias
			int bias_idx = operation_idx;
			operations_.push_back(std::unique_ptr<Operation>(new EvolvingUnityOperation()));
			operations_.back()->init(1, operation_idx);
			operation_idx++;

			//Output
			operations_.push_back(std::unique_ptr<Operation>(new SigmoidOperation()));
			operations_.back()->init(layer_widths[i], operation_idx);

			output_operation_ = operations_.back().get();

			connections_.push_back(Connection(bias_idx, operation_idx));
			connections_.push_back(Connection(previous_layer_idx, operation_idx));
			operation_idx++;
		}
		else {//Hidden layers


			  //Bias
			int bias_idx = operation_idx;
			operations_.push_back(std::unique_ptr<Operation>(new EvolvingUnityOperation()));
			operations_.back()->init(1, operation_idx);
			operation_idx++;

			//Hidden layer
			operations_.push_back(std::unique_ptr<Operation>(new ELUOperation()));
			operations_.back()->init(layer_widths[i], operation_idx);

			connections_.push_back(Connection(bias_idx, operation_idx));
			connections_.push_back(Connection(previous_layer_idx, operation_idx));

			previous_layer_idx = operation_idx;
			operation_idx++;

		}
	}


	std::vector<Operation*> oper_ptr;
	for (std::unique_ptr<Operation>& oper : operations_) {
		oper_ptr.push_back(oper.get());
	}
	for (Connection& conn : connections_) {
		conn.connect_to_operations(oper_ptr);
		conn.resize();
	}
	build_dependencies();

	float standard_min_weight = -0.1f;
	float standard_max_weight = 0.1f;
	randomize_weights(standard_min_weight, standard_max_weight);
}

void MultiLayerPerceptron::build_residual_network(const std::vector<unsigned>& layer_widths)
{
	connections_.clear();
	operations_.clear();

	int operation_idx = 0;
	int previous_layer_idx = -1;
	for (unsigned i = 0; i < layer_widths.size(); i++) {

		for (Connection& conn : connections_) {
			if (conn.output_node_index_ < 0) {
				conn.output_node_index_ = operation_idx;
			}
		}

		if (i == 0) {
			previous_layer_idx = operation_idx;
			operations_.push_back(std::unique_ptr<Operation>(new IdentityOperation()));
			operations_.back()->init(layer_widths[i], operation_idx);
			operation_idx++;

			input_operation_ = operations_.back().get();
		}
		else if (i == layer_widths.size() - 1) {

			connections_.push_back(Connection());
			Connection* current_connection = &connections_.back();
			current_connection->input_node_index_ = previous_layer_idx;
			current_connection->output_node_index_ = operation_idx;

			operations_.push_back(std::unique_ptr<Operation>(new IdentityOperation()));
			operations_.back()->init(layer_widths[i], operation_idx);
			operation_idx++;
			output_operation_ = operations_.back().get();
		}
		else {

			int previous_operation_idx = previous_layer_idx;
			std::vector<int> current_layer;

			////Non-linear part
			{
				connections_.push_back(Connection());
				Connection* current_connection = &connections_.back();
				current_connection->input_node_index_ = previous_operation_idx;
				current_connection->output_node_index_ = operation_idx;

				current_layer.push_back(operation_idx);
				operations_.push_back(std::unique_ptr<Operation>(new ELUOperation()));
				operations_.back()->init(layer_widths[i], operation_idx++);



				//connections_.push_back(Connection());
				//current_connection = &connections_.back();
				//current_connection->input_node_index_ = previous_operation_idx;
				//current_connection->output_node_index_ = operation_idx;

				//current_layer.push_back(operation_idx);
				//operations_.push_back(std::unique_ptr<Operation>(new MirrorOperation()));
				//operations_.back()->init(layer_widths[i], operation_idx++);
			}

			////Bias
			Connection* bias_connection = nullptr;
			{
				connections_.push_back(Connection());
				bias_connection = &connections_.back();
				bias_connection->input_node_index_ = operation_idx;

				operations_.push_back(std::unique_ptr<Operation>(new EvolvingUnityOperation()));
				operations_.back()->init(1, operation_idx++);
			}


			//Bypass connection
			Connection* bypass_connection = nullptr;
			if (layer_widths[i - 1] == layer_widths[i]) {
				connections_.push_back(Connection());
				bypass_connection = &connections_.back();
				bypass_connection->input_node_index_ = previous_operation_idx;
				bypass_connection->identity_connection_ = true;
			}

			//Gathering operation
			previous_layer_idx = operation_idx;
			operations_.push_back(std::unique_ptr<Operation>(new IdentityOperation()));
			operations_.back()->init(layer_widths[i], operation_idx++);

			if (bypass_connection) {
				bypass_connection->output_node_index_ = previous_layer_idx;
			}
			bias_connection->output_node_index_ = previous_layer_idx;


			for (int current_operation_idx : current_layer) {
				connections_.push_back(Connection());
				Connection* current_connection = &connections_.back();
				current_connection->input_node_index_ = current_operation_idx;
				current_connection->output_node_index_ = previous_layer_idx;//The previous layer idx is quite misleading here since it refers to the identity operation being the previous layer for the next layer
				current_connection->identity_connection_ = true;
			}


		}


	}


	std::vector<Operation*> oper_ptr;
	for (std::unique_ptr<Operation>& oper : operations_) {
		oper_ptr.push_back(oper.get());
	}
	for (Connection& conn : connections_) {
		conn.connect_to_operations(oper_ptr);
		conn.resize();
	}
	build_dependencies();
}

void MultiLayerPerceptron::run(const float * in, float * out)
{

	for (std::unique_ptr<Operation>& operation : operations_) {
		operation->prepare_forward();
	}

	std::deque<Connection*> flow_queue;

	for (Connection& conn : connections_) {
		conn.ready_ = false;
		flow_queue.push_back(&conn);
	}

	unsigned input_dim = input_operation_->size_;
	if (!use_input_scaling_) {
		for (unsigned i = 0; i < input_dim; i++) {
			input_operation_->activations_[i] = in[i];
		}
	}
	else {
		for (unsigned i = 0; i < input_dim; i++) {
			float normalized = (in[i] - input_mean_[i]) / input_stdev_[i];
			input_operation_->activations_[i] = normalized;
		}
	}


	perform_flow(flow_queue);


	output_operation_->perform_operation();

	if (out) {
		unsigned output_dim = output_operation_->size_;
		for (unsigned i = 0; i < output_dim; i++) {
			out[i] = output_operation_->outputs_[i];
		}
	}

}

float MultiLayerPerceptron::mse(const float ** in, const float ** out, unsigned data_points)
{
	Eigen::VectorXf prediction = output_operation_->outputs_;
	unsigned out_dim = prediction.size();
	prediction.setZero();

	float mean_square_error = 0.0f;

	for (unsigned i = 0; i < data_points; i++) {

		float error = 0.0f;

		run(in[i], prediction.data());

		for (unsigned dim = 0; dim < out_dim; dim++) {
			float diff = prediction[dim] - out[i][dim];
			error += diff*diff;
		}

		mean_square_error += error;

	}

	if (data_points > 0) {
		mean_square_error /= (float)data_points;
	}

	return mean_square_error;
}

void MultiLayerPerceptron::backpropagate_deltas(const float * in, const float * out, bool is_gradient, const float* output_scales)
{

	for (std::unique_ptr<Operation>& oper : operations_) {
		oper->prepare_backward();;
	}

	unsigned out_dim = output_operation_->size_;
	Eigen::Map<const Eigen::VectorXf> output_target = Eigen::Map<const Eigen::VectorXf>(out, out_dim);

	if (is_gradient) {
		output_operation_->deltas_ = output_target;
	}
	else {
		output_operation_->deltas_ = -(output_target - output_operation_->outputs_);
	}

	if (output_scales) {
		for (unsigned i = 0; i < out_dim; i++) {
			float stdev = std::max(std::numeric_limits<float>::epsilon(), output_scales[i]);
			output_operation_->deltas_[i] /= stdev;
		}
	}

	for (unsigned i = 0; i < out_dim; i++) {
		float& delta = output_operation_->deltas_[i];
		if (delta - delta != delta - delta) {
			return;
		}
	}


	if (max_gradient_norm_ < std::numeric_limits<float>::max()) {

		const bool clamping = true;

		if (clamping) {
			for (unsigned i = 0; i < (unsigned)output_operation_->deltas_.size(); i++) {
				float& num = output_operation_->deltas_[i];

				if (num < -max_gradient_norm_) {
					num = -max_gradient_norm_;
				}

				if (num > max_gradient_norm_) {
					num = max_gradient_norm_;
				}

			}
		}
		else {
			float norm = output_operation_->deltas_.norm();

			output_operation_->deltas_.normalize();

			norm = std::min(norm, max_gradient_norm_);

			output_operation_->deltas_ *= norm;
		}

	}

	//Forming the flow queue
	std::deque<Connection*> flow_queue;
	for (int i = (int)connections_.size() - 1; i >= 0; i--) {
		Connection& conn = connections_[i];
		conn.ready_ = false;
		flow_queue.push_back(&conn);
	}

	//////DEBUG
	//for (int i = 0; i < flow_queue.size(); i++) {
	//	int rand_idx = rand() % flow_queue.size();
	//	Connection* tmp = flow_queue[rand_idx];
	//	flow_queue[rand_idx] = flow_queue[i];
	//	flow_queue[i] = tmp;
	//}
	//Back propagating the deltas
	while (flow_queue.size() > 0) {
		Connection* connection = flow_queue.front();
		flow_queue.pop_front();

		bool delta_not_computed = !(connection->compute_deltas());
		if (delta_not_computed) {
			flow_queue.push_back(connection);
		}

	}

	input_operation_->get_gradients();
	input_operation_->deltas_.array() = input_operation_->deltas_.array() * input_operation_->gradient_.array();

}

void MultiLayerPerceptron::compute_gradients(const float * in, const float * out, bool is_gradient, const float* output_scales)
{

	run(in);
	backpropagate_deltas(in, out, is_gradient, output_scales);

	for (Connection& conn : connections_) {
		conn.compute_gradients();
	}

}

void MultiLayerPerceptron::compute_gradients_log_bernoulli(const float * in, const float * out)
{

	run(in);

	float grad = output_operation_->outputs_[0];

	if (*out > 0.5f) {
		grad = -grad;
		sigmoid(grad);
		grad = -grad;
	}
	else {
		sigmoid(grad);
	}


	const bool is_gradient = true;
	backpropagate_deltas(in, &grad, is_gradient);

	for (Connection& conn : connections_) {
		conn.compute_gradients();
	}

}

void MultiLayerPerceptron::train_back_prop(const float ** in, const float ** out, unsigned data_points, unsigned minibatch_size, bool is_gradient, const float* output_scales)
{

	if (use_input_scaling_) {
		train_input_normalization(in, data_points);
	}

	training_ = true;

	std::vector<unsigned> random_index(data_points, 0);
	for (unsigned i = 0; i < data_points; i++) {
		random_index[i] = i;
	}
	shuffle(random_index);
	shuffle(random_index);


	if (minibatch_size <= 0) {
		training_ = false;
		return;
	}

	int batch_idx = 0;
	while (random_index.size() > 0) {

		if (batch_idx == 0) {
			for (Connection& conn : connections_) {
				conn.zero_accumulated_gradients();
			}
		}

		unsigned rand_idx = random_index.back();
		random_index.pop_back();

		compute_gradients(in[rand_idx], out[rand_idx], is_gradient, output_scales);


		for (Connection& conn : connections_) {
			conn.accumulate_gradients();
		}

		batch_idx++;

		if (batch_idx >= minibatch_size) {
			for (Connection& conn : connections_) {
				conn.apply_gradients(learning_rate_);
				conn.clamp_weights(min_weight_, max_weight_);
			}
			batch_idx = 0;
		}

	}

	training_ = false;
}

void MultiLayerPerceptron::train_adam(const float ** in, const float ** out, unsigned data_points, unsigned minibatch_size, bool is_gradient, const float* output_scales)
{

	if (use_input_scaling_) {
		train_input_normalization(in, data_points);
	}

	training_ = true;

	std::vector<unsigned> random_index(data_points, 0);
	for (unsigned i = 0; i < data_points; i++) {
		random_index[i] = i;
	}
	shuffle(random_index);
	shuffle(random_index);


	if (minibatch_size <= 0) {
		training_ = false;
		return;
	}

	int batch_idx = 0;
	while (random_index.size() > 0) {

		if (batch_idx == 0) {
			for (Connection& conn : connections_) {
				conn.zero_accumulated_gradients();
			}
		}

		unsigned rand_idx = random_index.back();
		random_index.pop_back();

		compute_gradients(in[rand_idx], out[rand_idx], is_gradient, output_scales);


		for (Connection& conn : connections_) {
			conn.accumulate_gradients();
		}

		batch_idx++;

		if (batch_idx >= minibatch_size) {
			for (Connection& conn : connections_) {
				conn.apply_adam(learning_rate_, adam_first_moment_smoothing_, adam_second_moment_smoothing_);
				conn.clamp_weights(min_weight_, max_weight_);
			}
			batch_idx = 0;
		}

	}

	training_ = false;
}

void MultiLayerPerceptron::train_adamax(const float ** in, const float ** out, unsigned data_points, unsigned minibatch_size, bool is_gradient, const float* output_scales)
{

	if (use_input_scaling_) {
		train_input_normalization(in, data_points);
	}

	training_ = true;

	std::vector<unsigned> random_index(data_points, 0);
	for (unsigned i = 0; i < data_points; i++) {
		random_index[i] = i;
	}
	shuffle(random_index);
	shuffle(random_index);


	if (minibatch_size <= 0) {
		training_ = false;
		return;
	}

	int batch_idx = 0;
	while (random_index.size() > 0) {

		if (batch_idx == 0) {
			for (Connection& conn : connections_) {
				conn.zero_accumulated_gradients();
			}
		}

		unsigned rand_idx = random_index.back();
		random_index.pop_back();

		compute_gradients(in[rand_idx], out[rand_idx], is_gradient, output_scales);


		for (Connection& conn : connections_) {
			conn.accumulate_gradients();
		}

		batch_idx++;

		if (batch_idx >= minibatch_size) {
			for (Connection& conn : connections_) {
				conn.apply_adamax(learning_rate_, adam_first_moment_smoothing_, adam_second_moment_smoothing_);
				conn.clamp_weights(min_weight_, max_weight_);
			}
			batch_idx = 0;
		}

	}

	training_ = false;
}

void MultiLayerPerceptron::train_rprop(const float** in, const float** out, unsigned data_points, bool is_gradient, const float* output_scales)
{

	if (use_input_scaling_) {
		train_input_normalization(in, data_points);
	}

	for (Connection& conn : connections_) {
		conn.zero_accumulated_gradients();
	}

	for (unsigned index = 0; index < data_points; index++) {


		compute_gradients(in[index], out[index], is_gradient, output_scales);


		for (Connection& conn : connections_) {
			conn.accumulate_gradients();
		}

	}

	for (Connection& conn : connections_) {
		conn.apply_rprop();
		conn.clamp_weights(min_weight_, max_weight_);
	}

}

void MultiLayerPerceptron::train_rmsprop(const float ** in, const float ** out, unsigned data_points, unsigned minibatch_size, bool is_gradient, const float* output_scales)
{

	if (use_input_scaling_) {
		train_input_normalization(in, data_points);
	}

	training_ = true;

	std::vector<unsigned> random_index(data_points, 0);
	for (unsigned i = 0; i < data_points; i++) {
		random_index[i] = i;
	}
	shuffle(random_index);
	shuffle(random_index);


	if (minibatch_size <= 0) {
		training_ = false;
		return;
	}

	int batch_idx = 0;
	while (random_index.size() > 0) {

		if (batch_idx == 0) {
			for (Connection& conn : connections_) {
				conn.zero_accumulated_gradients();
			}
		}

		unsigned rand_idx = random_index.back();
		random_index.pop_back();

		compute_gradients(in[rand_idx], out[rand_idx], is_gradient, output_scales);


		for (Connection& conn : connections_) {
			conn.accumulate_gradients();
		}

		batch_idx++;

		if (batch_idx >= minibatch_size) {
			for (Connection& conn : connections_) {
				conn.apply_rmsprop(learning_rate_);
				conn.clamp_weights(min_weight_, max_weight_);
			}
			batch_idx = 0;
		}

	}

	training_ = false;


}

void MultiLayerPerceptron::train_particle_swarm_optimization(const float ** in, const float ** out, unsigned data_points, int population)
{

	if (population < 2) {
		return;
	}

	while (pso_population_.size() < population) {
		store_parameters_to_pso_population();
		float min_weight = -0.2f;
		float max_weight = 0.2f;
		randomize_weights_and_biases(min_weight, max_weight);
	}

	for (std::unique_ptr<CandidateSolution>& sample : pso_population_) {

		sample->fitness_ = 0.0f;
		set_parameters(sample->parameters_);

		for (int i = 0; i < data_points; i++) {
			Eigen::Map<const Eigen::VectorXf> true_out(out[i], output_operation_->size_);

			run(in[i]);

			float loss = (true_out - output_operation_->outputs_).squaredNorm();
			sample->fitness_ += loss;
		}

	}



	CandidateSolution* best = nullptr;
	for (std::unique_ptr<CandidateSolution>& sample : pso_population_) {
		if (!best) {
			best = sample.get();
		}
		else {
			if (best->fitness_ > sample->fitness_) {
				best = sample.get();
			}
		}
	}


	for (std::unique_ptr<CandidateSolution>& sample : pso_population_) {

		for (int i = 0; i < sample->parameters_.size(); i++) {

			if (sample.get() != best) {

				float force = sampleUniform<float>();

				sample->velocity_[i] *= sample->momentum_;
				sample->velocity_[i] += force*(best->parameters_[i] - sample->parameters_[i]);

				sample->parameters_[i] += sample->velocity_[i];
			}



		}

	}

	set_parameters(best->parameters_);

}

void MultiLayerPerceptron::randomize_weights(float min_val, float max_val, float bias_val)
{
	for (Connection& conn : connections_) {


		if (conn.in_operation_->type_ == Operation::EVOLVING_UNITY || conn.in_operation_->type_ == Operation::UNITY) {
			conn.weights_.setConstant(bias_val);
		}
		else {
			conn.randomize_weights(min_val, max_val);
		}
	}
}

void MultiLayerPerceptron::randomize_weights()
{
	float stdev = (float)get_amount_parameters();
	stdev = 1.0f / stdev;

	for (Connection& conn : connections_) {
		conn.randomize_weights_gaussian(stdev);
	}

}

void MultiLayerPerceptron::randomize_weights_and_biases(float min_val, float max_val)
{
	for (Connection& conn : connections_) {
		conn.randomize_weights(min_val, max_val);
	}
}

void MultiLayerPerceptron::train_input_normalization(const float ** in, unsigned data_points)
{

	float normalization_learning_rate = learning_rate_;

	for (int i = 0; i < data_points; i++) {
		Eigen::Map<const Eigen::VectorXf> input(in[i], input_operation_->size_);

		Eigen::VectorXf diff = input - input_mean_;
		input_mean_ += normalization_learning_rate*diff;

		diff = input - input_mean_;
		input_stdev_ = input_stdev_.cwiseAbs2();
		input_stdev_ += (normalization_learning_rate*(diff.cwiseAbs2() - input_stdev_)).eval();
		input_stdev_ = input_stdev_.cwiseSqrt();
	}

	for (int dim = 0; dim < input_stdev_.size(); dim++) {
		if (input_stdev_[dim] < std::numeric_limits<float>::epsilon()) {
			input_stdev_[dim] = std::numeric_limits<float>::epsilon();
		}
	}


}

void MultiLayerPerceptron::train_bernoulli_classifier_adam(const float ** in, const float ** out, unsigned data_points, unsigned minibatch_size)
{
	if (use_input_scaling_) {
		train_input_normalization(in, data_points);
	}

	training_ = true;

	std::vector<unsigned> random_index(data_points, 0);
	for (unsigned i = 0; i < data_points; i++) {
		random_index[i] = i;
	}
	shuffle(random_index);
	shuffle(random_index);


	if (minibatch_size <= 0) {
		training_ = false;
		return;
	}

	int batch_idx = 0;
	while (random_index.size() > 0) {

		if (batch_idx == 0) {
			for (Connection& conn : connections_) {
				conn.zero_accumulated_gradients();
			}
		}

		unsigned rand_idx = random_index.back();
		random_index.pop_back();

		compute_gradients_log_bernoulli(in[rand_idx], out[rand_idx]);

		for (Connection& conn : connections_) {
			conn.accumulate_gradients();
		}

		batch_idx++;

		if (batch_idx >= minibatch_size) {
			for (Connection& conn : connections_) {
				conn.apply_adam(learning_rate_, adam_first_moment_smoothing_, adam_second_moment_smoothing_);
				conn.clamp_weights(min_weight_, max_weight_);
			}
			batch_idx = 0;
		}

	}

	training_ = false;
}



void MultiLayerPerceptron::write_to_file(std::string filename)
{
	std::ofstream net_file(filename);
	net_file.clear();
	net_file << learning_rate_ << std::endl;
	net_file << min_weight_ << std::endl;
	net_file << max_weight_ << std::endl;
	net_file << adam_first_moment_smoothing_ << std::endl;
	net_file << adam_second_moment_smoothing_ << std::endl;

	mse_ = std::min(mse_, std::numeric_limits<float>::max());
	prev_mse_ = std::min(prev_mse_, std::numeric_limits<float>::max());

	net_file << mse_ << std::endl;
	net_file << prev_mse_ << std::endl;

	net_file << max_gradient_norm_ << std::endl;

	net_file << input_operation_->index_ << std::endl;
	net_file << output_operation_->index_ << std::endl;

	net_file << drop_out_stdev_ << std::endl;

	net_file << use_input_scaling_ << std::endl;

	write_vector_to_file(net_file, input_mean_);
	write_vector_to_file(net_file, input_stdev_);

	for (std::unique_ptr<Operation>& oper : operations_) {
		oper->write_to_file(net_file);
	}

	for (Connection& conn : connections_) {
		conn.write_to_file(net_file);
	}

	net_file.close();
}

void MultiLayerPerceptron::load_from_file(std::string filename)
{
	std::ifstream net_file(filename);

	std::string line;

	input_operation_ = nullptr;
	output_operation_ = nullptr;

	net_file >> learning_rate_;
	net_file >> min_weight_;
	net_file >> max_weight_;
	net_file >> adam_first_moment_smoothing_;
	net_file >> adam_second_moment_smoothing_;
	net_file >> mse_;
	net_file >> prev_mse_;


	net_file >> max_gradient_norm_;

	int input_operation_index;
	net_file >> input_operation_index;

	int output_operation_index;
	net_file >> output_operation_index;

	net_file >> drop_out_stdev_;

	int tmp;
	net_file >> tmp;
	use_input_scaling_ = (bool)tmp;

	std::getline(net_file, line);

	input_mean_ = load_vector_from_file(net_file);
	input_stdev_ = load_vector_from_file(net_file);


	line = read_operations(net_file);
	line = read_connections(net_file, line);

	net_file.close();

	//Connecting the operations to connections.
	std::vector<Operation*> oper_ptr;
	for (std::unique_ptr<Operation>& oper : operations_) {
		oper_ptr.push_back(oper.get());
	}
	for (Connection& conn : connections_) {
		conn.connect_to_operations(oper_ptr);
	}

	build_dependencies();

	for (std::unique_ptr<Operation>& oper : operations_) {
		if (oper->index_ == input_operation_index) {
			input_operation_ = oper.get();
		}

		if (oper->index_ == output_operation_index) {
			output_operation_ = oper.get();
		}
	}
}

std::string MultiLayerPerceptron::read_operations(std::ifstream & input_file)
{
	std::string read_line;
	std::getline(input_file, read_line);
	std::vector<std::unique_ptr<Operation>> operations;

	while (read_line.compare("//OPERATION BEGIN") == 0) {
		operations.push_back(std::unique_ptr<Operation>(new Operation()));
		Operation& new_oper = *operations.back();
		new_oper.load_from_file(input_file);
		std::getline(input_file, read_line);
	}

	copy_operations(operations);

	return read_line;

}

std::string MultiLayerPerceptron::read_connections(std::ifstream & input_file, std::string line)
{


	connections_.clear();

	while (line.compare("//CONNECTION BEGIN") == 0) {
		connections_.push_back(Connection());
		connections_.back().load_from_file(input_file);
		std::getline(input_file, line);
	}

	return line;
}

IdentityOperation::IdentityOperation()
{
	type_ = OperationType::IDENTITY;
}

IdentityOperation::~IdentityOperation()
{
}

void IdentityOperation::perform_operation()
{
	ready_ = true;
	outputs_ = activations_;

}

void IdentityOperation::get_gradients()
{
	ready_ = true;
	gradient_.resize(size_);
	gradient_.setOnes();
}

TanhOperation::TanhOperation()
{
	type_ = OperationType::TANH;
}

TanhOperation::~TanhOperation()
{
}

void TanhOperation::perform_operation()
{
	ready_ = true;
	outputs_ = activations_;
	for (int i = 0; i < size_; i++) {
		float& num = outputs_[i];
		num = std::tanh(num);
	}
}

void TanhOperation::get_gradients()
{

	ready_ = true;
	gradient_ = activations_;
	unsigned size = gradient_.size();

	for (unsigned i = 0; i < size; i++) {
		float& num = gradient_[i];
		num = std::tanh(num);
		num *= -num;
		num += 1.0f;
	}

}

UnityOperation::UnityOperation()
{
	type_ = OperationType::UNITY;
}

UnityOperation::~UnityOperation()
{
}

void UnityOperation::perform_operation()
{
	ready_ = true;
	activations_.setOnes();
	outputs_ = activations_;
}

void UnityOperation::get_gradients()
{
	ready_ = true;
	gradient_.resize(size_);
	gradient_.setZero();
}

SquareOperation::SquareOperation()
{
	type_ = OperationType::SQUARE;
}

SquareOperation::~SquareOperation()
{
}

void SquareOperation::perform_operation()
{
	ready_ = true;
	outputs_ = (desired_ - activations_).cwiseAbs2();
}

void SquareOperation::get_gradients()
{
	ready_ = true;
	gradient_ = -(desired_ - activations_);
}

void SquareOperation::init(unsigned dimension, int operation_index)
{
	Operation::init(dimension, operation_index);
	desired_.resize(dimension);
	desired_.setZero();
}

LogOperation::LogOperation()
{
	type_ = OperationType::LOG;
}

LogOperation::~LogOperation()
{
}

void LogOperation::perform_operation()
{
	ready_ = true;
	outputs_ = activations_;
	for (int i = 0; i < size_; i++) {
		outputs_[i] = std::max(std::numeric_limits<float>::epsilon(), outputs_[i]);
		outputs_[i] = std::log(outputs_[i]);
	}

}

void LogOperation::get_gradients()
{

	ready_ = true;
	gradient_ = activations_;

	unsigned size = gradient_.size();
	for (unsigned i = 0; i < size; i++) {
		gradient_[i] = std::max(std::numeric_limits<float>::epsilon(), gradient_[i]);
		gradient_[i] = 1.0f / gradient_[i];
	}

}

SigmoidOperation::SigmoidOperation() {
	type_ = OperationType::SIGMOID;
}

SigmoidOperation::~SigmoidOperation()
{
}

void SigmoidOperation::perform_operation()
{
	ready_ = true;
	outputs_ = activations_;

	for (int i = 0; i < size_; i++) {
		float& tmp = outputs_[i];

		tmp = -tmp;
		stable_exp(tmp);
		tmp = 1.0f / (1.0f + tmp);

	}
}

void SigmoidOperation::get_gradients()
{

	ready_ = true;
	gradient_ = activations_;

	unsigned size = gradient_.size();
	for (unsigned i = 0; i < size; i++) {
		float& tmp = gradient_[i];

		tmp = -tmp;
		stable_exp(tmp);
		tmp = 1.0f / (1.0f + tmp);

		tmp = tmp*(1.0f - tmp);

	}

}

NegateOperation::NegateOperation()
{
	type_ = OperationType::NEGATE;
}

NegateOperation::~NegateOperation()
{

}

void NegateOperation::perform_operation()
{
	ready_ = true;
	outputs_ = -activations_;
}

void NegateOperation::get_gradients()
{
	ready_ = true;
	gradient_.resize(size_);
	for (int i = 0; i < size_; i++) {
		gradient_[i] = -1.0f;
	}
}


void MultiLayerPerceptron::perform_flow(std::deque<Connection*>& flow_queue)
{
	//////DEBUG
	//for (int i = 0; i < flow_queue.size(); i++) {
	//	int rand_idx = rand() % flow_queue.size();
	//	Connection* tmp = flow_queue[rand_idx];
	//	flow_queue[rand_idx] = flow_queue[i];
	//	flow_queue[i] = tmp;
	//}
	while (flow_queue.size() > 0) {
		Connection* connection = flow_queue.front();
		flow_queue.pop_front();

		float drop_out_stdev = 0.0f;
		if (training_) {
			if (connection->in_operation_ != input_operation_) {
				drop_out_stdev = drop_out_stdev_;
			}
		}

		bool output_not_formed = !(connection->form_output(drop_out_stdev));
		if (output_not_formed) {
			flow_queue.push_back(connection);
		}

	}
}

int MultiLayerPerceptron::get_amount_parameters()
{
	int amount_parameters = 0;

	for (const Connection& conn : connections_) {

		amount_parameters += conn.weights_.rows() * conn.weights_.cols();

	}

	return amount_parameters;
}

std::vector<float> MultiLayerPerceptron::get_parameters_as_one_vector()
{
	std::vector<float> parameter_vector;
	parameter_vector.reserve(500000);

	for (int i = 0; i < input_mean_.size(); i++) {
		parameter_vector.push_back(input_mean_[i]);
	}

	for (int i = 0; i < input_stdev_.size(); i++) {
		parameter_vector.push_back(input_stdev_[i]);
	}

	std::vector<Connection*> connec = get_connections_in_order();

	for (const Connection* conn : connec) {

		const Eigen::MatrixXf& weight_matrix = conn->weights_;

		int rows = weight_matrix.rows();
		int cols = weight_matrix.cols();

		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				parameter_vector.push_back(weight_matrix(row, col));
			}
		}

	}

	return parameter_vector;
}

void MultiLayerPerceptron::set_parameters(const std::vector<float>& parameters)
{
	std::vector<Connection*> connec = get_connections_in_order();

	int idx = 0;

	for (int i = 0; i < input_mean_.size(); i++) {
		input_mean_[i] = parameters[idx];
		idx++;
	}

	for (int i = 0; i < input_stdev_.size(); i++) {
		input_stdev_[i] = parameters[idx];
		idx++;
	}


	for (Connection* conn : connec) {

		Eigen::MatrixXf& weight_matrix = conn->weights_;

		int rows = weight_matrix.rows();
		int cols = weight_matrix.cols();

		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				weight_matrix(row, col) = parameters[idx];
				idx++;
			}
		}

	}


	assert(idx == parameters.size());


}

std::vector<Connection*> MultiLayerPerceptron::get_connections_in_order()
{
	std::vector<Connection*> connec;

	for (Connection& conn : connections_) {
		connec.push_back(&conn);
	}

	auto order_lambda = [](Connection* first, Connection* second) {
		if (first->input_node_index_ < second->input_node_index_) {
			return true;
		}

		if (first->input_node_index_ > second->input_node_index_) {
			return false;
		}

		if (first->output_node_index_ <= second->output_node_index_) {
			return true;
		}

		return false;

	};

	std::sort(connec.begin(), connec.end(), order_lambda);

	return connec;
}

void MultiLayerPerceptron::increase_capacity(int operation_idx, int dimension)
{

	Operation* operation = nullptr;

	for (std::unique_ptr<Operation>& oper : operations_) {

		if (oper->index_ == operation_idx) {

			operation = oper.get();

			//Can resize only hidden layers or biases
			if (input_operation_ == oper.get()) {
				operation = nullptr;
			}

			if (output_operation_ == oper.get()) {
				operation = nullptr;
			}

		}

		if (operation) {
			break;
		}

	}

	if (operation) {

		if (dimension < operation->size_) {

			operation->increase_dimension();

			for (Connection& conn : connections_) {

				if (conn.output_node_index_ == operation_idx) {
					conn.duplicate_output(dimension);
				}

				if (conn.input_node_index_ == operation_idx) {
					conn.duplicate_input(dimension);
				}

			}

		}

	}

}

void MultiLayerPerceptron::store_parameters_to_pso_population()
{
	std::unique_ptr<CandidateSolution> individual(new CandidateSolution());

	individual->fitness_ = 0.0f;
	individual->parameters_ = get_parameters_as_one_vector();
	individual->momentum_ = 0.8f;
	individual->velocity_ = std::vector<float>(individual->parameters_.size(), 0.0f);
	pso_population_.push_back(std::move(individual));
}

std::pair<int, int> MultiLayerPerceptron::find_hidden_neuron_with_biggest_variation(const float ** in, unsigned data_points)
{
	std::pair<int, int> operation_and_dimension;

	//Storing the means to operations' gradients and stdevs to operations' deltas

	for (std::unique_ptr<Operation>& oper : operations_) {
		oper->prepare_backward();
	}



	for (unsigned i = 0; i < data_points; i++) {

		run(in[i]);

		for (std::unique_ptr<Operation>& oper : operations_) {
			oper->gradient_ += oper->activations_;
		}

	}
	for (std::unique_ptr<Operation>& oper : operations_) {
		oper->gradient_ *= 1.0 / (float)data_points;
	}



	for (unsigned i = 0; i < data_points; i++) {

		run(in[i]);

		for (std::unique_ptr<Operation>& oper : operations_) {
			oper->deltas_ += (oper->activations_ - oper->gradient_).cwiseAbs2();
		}

	}
	for (std::unique_ptr<Operation>& oper : operations_) {
		oper->deltas_ *= 1.0 / (float)data_points;
		oper->deltas_ = oper->deltas_.cwiseSqrt();
	}




	float max_stdev = -1.0f;

	for (std::unique_ptr<Operation>& oper : operations_) {
		if (oper.get() != input_operation_ && oper.get() != output_operation_) {

			for (unsigned i = 0; i < oper->deltas_.size(); i++) {

				const float& stdev = oper->deltas_[i];

				if (stdev > max_stdev) {
					max_stdev = stdev;
					operation_and_dimension = std::make_pair(oper->index_, i);
				}

			}

		}
	}

	return operation_and_dimension;
}

std::vector<std::pair<int, int>> MultiLayerPerceptron::find_hidden_neurons_with_biggest_variation(const float ** in, unsigned data_points)
{
	std::pair<int, int> operation_and_dimension;
	std::vector<std::pair<int, int>> return_value;

	//Storing the means to operations' gradients and stdevs to operations' deltas

	for (std::unique_ptr<Operation>& oper : operations_) {
		oper->prepare_backward();
	}



	for (unsigned i = 0; i < data_points; i++) {

		run(in[i]);

		for (std::unique_ptr<Operation>& oper : operations_) {
			oper->gradient_ += oper->activations_;
		}

	}
	for (std::unique_ptr<Operation>& oper : operations_) {
		oper->gradient_ *= 1.0 / (float)data_points;
	}



	for (unsigned i = 0; i < data_points; i++) {

		run(in[i]);

		for (std::unique_ptr<Operation>& oper : operations_) {
			oper->deltas_ += (oper->activations_ - oper->gradient_).cwiseAbs2();
		}

	}
	for (std::unique_ptr<Operation>& oper : operations_) {
		oper->deltas_ *= 1.0 / (float)data_points;
		oper->deltas_ = oper->deltas_.cwiseSqrt();
	}




	for (std::unique_ptr<Operation>& oper : operations_) {
		if (oper.get() != input_operation_ && oper.get() != output_operation_) {

			if (oper->type_ == Operation::EVOLVING_UNITY || oper->type_ == Operation::UNITY) {
				continue;
			}

			float max_stdev = -1.0f;
			operation_and_dimension.first = oper->index_;

			for (unsigned i = 0; i < oper->deltas_.size(); i++) {

				const float& stdev = oper->deltas_[i];

				if (stdev > max_stdev) {
					max_stdev = stdev;
					operation_and_dimension.second = i;
				}

			}

			return_value.push_back(operation_and_dimension);

		}
	}

	return return_value;
}

std::pair<int, int> MultiLayerPerceptron::find_hidden_neuron_with_greatest_blame(const float** in, const float** out, unsigned data_points)
{
	std::pair<int, int> operation_and_dimension;

	//Storing the means to operations' gradients and stdevs to operations' deltas

	std::map<int, Eigen::VectorXf> errors;
	for (std::unique_ptr<Operation>& oper : operations_) {
		oper->prepare_forward();
		errors[oper->index_] = oper->activations_*0.0f;
	}


	for (unsigned i = 0; i < data_points; i++) {

		run(in[i]);
		backpropagate_deltas(in[i], out[i]);

		for (std::unique_ptr<Operation>& oper : operations_) {
			errors[oper->index_] += oper->deltas_;
		}

	}



	float max_blame = -1.0f;

	for (std::unique_ptr<Operation>& oper : operations_) {
		if (oper.get() != input_operation_ && oper.get() != output_operation_) {

			if (oper->type_ == Operation::EVOLVING_UNITY || oper->type_ == Operation::UNITY) {
				continue;
			}

			Eigen::VectorXf& error = errors[oper->index_];

			for (unsigned i = 0; i < error.size(); i++) {

				const float& stdev = error[i];

				if (std::abs(stdev) > max_blame) {
					max_blame = std::abs(stdev);
					operation_and_dimension = std::make_pair(oper->index_, i);
				}

			}

		}
	}

	return operation_and_dimension;
}

std::vector<std::pair<int, int>> MultiLayerPerceptron::find_hidden_neurons_with_greatest_blame(const float ** in, const float ** out, unsigned data_points)
{
	std::pair<int, int> operation_and_dimension;
	std::vector<std::pair<int, int>> return_value;

	//Storing the means to operations' gradients and stdevs to operations' deltas

	std::map<int, Eigen::VectorXf> errors;
	for (std::unique_ptr<Operation>& oper : operations_) {
		oper->prepare_forward();
		errors[oper->index_] = oper->activations_*0.0f;
	}


	for (unsigned i = 0; i < data_points; i++) {

		run(in[i]);
		backpropagate_deltas(in[i], out[i]);

		for (std::unique_ptr<Operation>& oper : operations_) {
			errors[oper->index_] += oper->deltas_;
		}

	}


	for (std::unique_ptr<Operation>& oper : operations_) {
		if (oper.get() != input_operation_ && oper.get() != output_operation_) {

			if (oper->type_ == Operation::EVOLVING_UNITY || oper->type_ == Operation::UNITY) {
				continue;
			}

			Eigen::VectorXf& error = errors[oper->index_];
			float max_blame = -1.0f;
			operation_and_dimension.first = oper->index_;

			for (unsigned i = 0; i < error.size(); i++) {

				const float& stdev = error[i];

				if (std::abs(stdev) > max_blame) {
					max_blame = std::abs(stdev);
					operation_and_dimension.second = i;
				}

			}

			return_value.push_back(operation_and_dimension);

		}
	}

	return return_value;
}

EvolvingUnityOperation::EvolvingUnityOperation()
{
	type_ = OperationType::EVOLVING_UNITY;
}

EvolvingUnityOperation::~EvolvingUnityOperation()
{
}

void EvolvingUnityOperation::perform_operation()
{
	ready_ = true;
	activations_.setOnes();
	outputs_ = activations_;
}

void EvolvingUnityOperation::get_gradients()
{
	ready_ = true;
	gradient_.resize(size_);
	gradient_.setOnes();
}

LinearClippedOperation::LinearClippedOperation()
{
	type_ = OperationType::LINEAR_CLIPPED;
}

LinearClippedOperation::~LinearClippedOperation()
{
}

void LinearClippedOperation::perform_operation()
{
	ready_ = true;
	outputs_ = activations_;

	for (int i = 0; i < size_; i++) {
		float& num = outputs_[i];

		num = std::max(0.0f, std::min(1.0f, num));

	}
}

void LinearClippedOperation::get_gradients()
{
	ready_ = true;
	gradient_.resize(size_);
	gradient_.setOnes();
}

SoftPlusOperation::SoftPlusOperation()
{
	type_ = OperationType::SOFTPLUS;
}

SoftPlusOperation::~SoftPlusOperation()
{
}

void SoftPlusOperation::perform_operation()
{
	ready_ = true;
	outputs_ = activations_;

	for (int i = 0; i < size_; i++) {

		float& num = outputs_[i];

		num = std::exp(num);
		num += 1.0f;
		num = std::log(num);

	}

}

void SoftPlusOperation::get_gradients()
{
	ready_ = true;
	gradient_ = activations_;

	unsigned size = gradient_.size();

	for (unsigned i = 0; i < size; i++) {
		float& num = gradient_[i];

		num = std::exp(-num);
		num = 1.0f / (1.0f + num);

	}


}

ELUOperation::ELUOperation()
{
	type_ = OperationType::ELU;
}

ELUOperation::~ELUOperation()
{
}

void ELUOperation::perform_operation()
{
	ready_ = true;
	outputs_ = activations_;

	for (int i = 0; i < size_; i++) {

		float& num = outputs_[i];

		if (num < 0.0f) {

			stable_exp(num);
			num -= 1.0f;
		}

	}
}

void ELUOperation::get_gradients()
{
	ready_ = true;
	gradient_ = activations_;

	unsigned size = gradient_.size();

	for (unsigned i = 0; i < size; i++) {
		float& num = gradient_[i];

		if (num >= 0.0f) {
			num = 1.0f;
		}
		else {
			stable_exp(num);
		}

	}
}

MirrorOperation::MirrorOperation()
{
	type_ = OperationType::MIRROR;
}

MirrorOperation::~MirrorOperation()
{
}

void MirrorOperation::perform_operation()
{
	ready_ = true;
	outputs_ = activations_;

	for (int i = 0; i < size_; i++) {

		float& num = outputs_[i];

		if (num < 0.0f) {
			num = -num;
		}

	}
}

void MirrorOperation::get_gradients()
{
	ready_ = true;
	gradient_ = activations_;

	unsigned size = gradient_.size();

	for (unsigned i = 0; i < size; i++) {
		float& num = gradient_[i];

		if (num > 0.0f) {
			num = 1.0f;
		}
		else if (num < 0.0f) {
			num = -1.0f;
		}
		else {
			//Subgradient in the interval [-1,1]
			num = (float)rand() / (float)RAND_MAX;
			num = 2.0f * num - 1.0f;
		}


	}
}

SELUOperation::SELUOperation() : lambda_(1.5f)
{

	type_ = OperationType::SELU;
}

SELUOperation SELUOperation::operator=(const SELUOperation& other)
{
	type_ = other.type_;
	index_ = other.index_;
	ready_ = other.ready_;
	size_ = other.size_;

	init(size_, index_);
	return *this;
}

SELUOperation::~SELUOperation()
{
}

void SELUOperation::perform_operation()
{

	ready_ = true;
	outputs_ = activations_;

	for (int i = 0; i < size_; i++) {

		float& num = outputs_[i];

		if (num < 0.0f) {

			stable_exp(num);
			num -= 1.0f;
		}

		num *= lambda_;

	}

}

void SELUOperation::get_gradients()
{

	ready_ = true;
	gradient_ = activations_;

	unsigned size = gradient_.size();

	for (unsigned i = 0; i < size; i++) {
		float& num = gradient_[i];

		if (num >= 0.0f) {
			num = 1.0f;
		}
		else {
			stable_exp(num);
		}

		num *= lambda_;

	}
}

BSELUOperation::BSELUOperation() : lambda_(1.5f)
{
	type_ = OperationType::BSELU;
}

BSELUOperation BSELUOperation::operator=(const BSELUOperation& other)
{
	type_ = other.type_;
	index_ = other.index_;
	ready_ = other.ready_;
	size_ = other.size_;

	init(size_, index_);
	return *this;
}

BSELUOperation::~BSELUOperation()
{
}

void BSELUOperation::perform_operation()
{

	ready_ = true;
	outputs_ = activations_;

	for (int i = 0; i < size_; i++) {

		float& num = outputs_[i];

		if (i % 2 == 0) {
			num *= -1.0f;
		}

		if (num < 0.0f) {

			stable_exp(num);
			num -= 1.0f;
		}


		if (i % 2 == 0) {
			num *= -lambda_;
		}
		else {
			num *= lambda_;
		}

	}

}

void BSELUOperation::get_gradients()
{

	ready_ = true;
	gradient_ = activations_;

	unsigned size = gradient_.size();

	for (unsigned i = 0; i < size; i++) {
		float& num = gradient_[i];

		if (i % 2 == 0) {
			num *= -1.0f;
		}

		if (num >= 0.0f) {
			num = 1.0f;
		}
		else {
			stable_exp(num);
		}

		num *= lambda_;

	}

}
