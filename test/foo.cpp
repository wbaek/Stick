#include <gtest/gtest.h>

#include <foo.hpp>

TEST(Foo, Foo_can_creatable) {
    Foo f;
}

TEST(Foo, bar) {
    Foo f;
    EXPECT_EQ(0, f.bar());
}
