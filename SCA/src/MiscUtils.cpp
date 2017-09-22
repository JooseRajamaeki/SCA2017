#include "MiscUtils.hpp"

std::string get_time_string() {
	time_t now = time(0);
	tm localtm;
	localtime_s(&localtm, &now);
	std::string result(1000,' ');
	
	asctime_s((char*)result.data(), 1000, &localtm);

	
	do {

		if (result.back() == '\n') {
			result.pop_back();
			break;
		}

		result.pop_back();

		if (result.size() == 0) {
			break;
		}

	} while (true);

	result = result.substr(4, result.size());

	std::string year_string = result.substr(result.size() - 4, result.size());
	result = result.substr(0, result.size() - 5);

	result = year_string + "_" + result;

	for (char& character : result) {
		if (character == ':') {
			character = '.';
		}
		if (character == ' ') {
			character = '_';
		}
	}

	return result;
}

void repulse(float** sequence1, int dim, int data_points, float shift_amount)
{

	std::vector<std::vector<float>> data;
	data.reserve(data_points*dim);

	for (int i = 0; i < data_points; i++) {

		std::vector<float> tmp(dim,0.0f);

		Eigen::Map<Eigen::VectorXf> datum(tmp.data(), dim);

		Eigen::Map<Eigen::VectorXf> current(sequence1[i], dim);

		Eigen::VectorXf diff;

		for (int j = 0; j < data_points; j++) {

			if (i == j) {
				continue;
			}

			Eigen::Map<Eigen::VectorXf> other(sequence1[j], dim);

			diff = current - other;
			float norm = diff.norm();
			if (norm > std::numeric_limits<float>::min()) {
				diff /= norm;
			}

			datum += diff;

		}

		float norm = datum.norm();
		if (norm > std::numeric_limits<float>::min()) {
			datum /= norm;
		}
		datum *= shift_amount;

		datum += current;

		data.push_back(tmp);

	}


	for (int i = 0; i < data_points; i++) {
		for (int j = 0; j < dim; j++) {
			sequence1[i][j] = data[i][j];
		}
	}
}

void project_to_unit_sphere(float** sequence, int dim, int data_points)
{

	for (int i = 0; i < data_points; i++) {
		Eigen::Map<Eigen::VectorXf> current(sequence[i], dim);
		current.normalize();
	}


}
