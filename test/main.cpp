#include <gtest/gtest.h>

class ESMTrackingEnvironment : public ::testing::Environment{
    public:
        virtual ~ESMTrackingEnvironment() {
        }

        virtual void SetUp() {
        }

        virtual void TearDown() {
        }
};

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new ESMTrackingEnvironment);
    bool rv = RUN_ALL_TESTS();
    return rv;
}

