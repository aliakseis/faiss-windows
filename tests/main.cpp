#define BOOST_TEST_NO_MAIN
#include <boost/test/unit_test.hpp>

class __declspec(dllexport) TestRunner {
public:
	void run(int, char**);
	static bool init();
};

void TestRunner::run(int n, char** argc) {
	boost::unit_test::unit_test_main(&TestRunner::init, n, argc);
}
bool TestRunner::init() {
	return true;
}
int main(int argv, char** argc) {
	TestRunner runner;
	runner.run(argv, argc);
}