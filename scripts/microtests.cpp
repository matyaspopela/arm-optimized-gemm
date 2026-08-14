#include <iostream>
#include <ostream>
#include <string>


constexpr const char* name = "naive";
constexpr const char* folder = "../testfolder/";


int main() {
	std::string result = std::string(folder) + name;
	result += ".csv";

	std::cout << result << std::endl;

	std::ifstream file(result);

	return 0;
}