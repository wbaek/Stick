#include <gtest/gtest.h>

#include <model/homography.hpp>

TEST(Homography, create) {
    Stick::Homography model;
}

TEST(Homography, set_get_initialize) {
    Stick::Homography model;

    cv::Mat pose = model.get();
    EXPECT_EQ(1.0, pose.at<double>(0));
    EXPECT_EQ(0.0, pose.at<double>(1));
    EXPECT_EQ(0.0, pose.at<double>(2));

    EXPECT_EQ(0.0, pose.at<double>(3));
    EXPECT_EQ(1.0, pose.at<double>(4));
    EXPECT_EQ(0.0, pose.at<double>(5));

    EXPECT_EQ(0.0, pose.at<double>(6));
    EXPECT_EQ(0.0, pose.at<double>(7));
    EXPECT_EQ(1.0, pose.at<double>(8));

    EXPECT_EQ(1.0, pose.at<double>(cv::Point(0, 0)));
    EXPECT_EQ(0.0, pose.at<double>(cv::Point(1, 0)));
    EXPECT_EQ(0.0, pose.at<double>(cv::Point(2, 0)));

    EXPECT_EQ(0.0, pose.at<double>(cv::Point(0, 1)));
    EXPECT_EQ(1.0, pose.at<double>(cv::Point(1, 1)));
    EXPECT_EQ(0.0, pose.at<double>(cv::Point(2, 1)));

    EXPECT_EQ(0.0, pose.at<double>(cv::Point(0, 2)));
    EXPECT_EQ(0.0, pose.at<double>(cv::Point(1, 2)));
    EXPECT_EQ(1.0, pose.at<double>(cv::Point(2, 2)));

    pose.at<double>(cv::Point(2, 0)) = 2.0;
    pose.at<double>(cv::Point(2, 1)) = 3.0;
    pose = model.get();
 
    EXPECT_EQ(1.0, pose.at<double>(0));
    EXPECT_EQ(0.0, pose.at<double>(1));
    EXPECT_EQ(0.0, pose.at<double>(2));

    EXPECT_EQ(0.0, pose.at<double>(3));
    EXPECT_EQ(1.0, pose.at<double>(4));
    EXPECT_EQ(0.0, pose.at<double>(5));

    EXPECT_EQ(0.0, pose.at<double>(6));
    EXPECT_EQ(0.0, pose.at<double>(7));
    EXPECT_EQ(1.0, pose.at<double>(8));

    EXPECT_EQ(1.0, pose.at<double>(cv::Point(0, 0)));
    EXPECT_EQ(0.0, pose.at<double>(cv::Point(1, 0)));
    EXPECT_EQ(0.0, pose.at<double>(cv::Point(2, 0)));

    EXPECT_EQ(0.0, pose.at<double>(cv::Point(0, 1)));
    EXPECT_EQ(1.0, pose.at<double>(cv::Point(1, 1)));
    EXPECT_EQ(0.0, pose.at<double>(cv::Point(2, 1)));

    EXPECT_EQ(0.0, pose.at<double>(cv::Point(0, 2)));
    EXPECT_EQ(0.0, pose.at<double>(cv::Point(1, 2)));
    EXPECT_EQ(1.0, pose.at<double>(cv::Point(2, 2)));

   
    pose.at<double>(cv::Point(2, 0)) = 2.0;
    pose.at<double>(cv::Point(2, 1)) = 3.0;
    model.set(pose);

    pose = model.get();
    EXPECT_EQ(1.0, pose.at<double>(0));
    EXPECT_EQ(0.0, pose.at<double>(1));
    EXPECT_EQ(2.0, pose.at<double>(2));

    EXPECT_EQ(0.0, pose.at<double>(3));
    EXPECT_EQ(1.0, pose.at<double>(4));
    EXPECT_EQ(3.0, pose.at<double>(5));

    EXPECT_EQ(0.0, pose.at<double>(6));
    EXPECT_EQ(0.0, pose.at<double>(7));
    EXPECT_EQ(1.0, pose.at<double>(8));
 
    EXPECT_EQ(1.0, pose.at<double>(cv::Point(0, 0)));
    EXPECT_EQ(0.0, pose.at<double>(cv::Point(1, 0)));
    EXPECT_EQ(2.0, pose.at<double>(cv::Point(2, 0)));

    EXPECT_EQ(0.0, pose.at<double>(cv::Point(0, 1)));
    EXPECT_EQ(1.0, pose.at<double>(cv::Point(1, 1)));
    EXPECT_EQ(3.0, pose.at<double>(cv::Point(2, 1)));

    EXPECT_EQ(0.0, pose.at<double>(cv::Point(0, 2)));
    EXPECT_EQ(0.0, pose.at<double>(cv::Point(1, 2)));
    EXPECT_EQ(1.0, pose.at<double>(cv::Point(2, 2)));

    model.initialize();
    pose = model.get();
    EXPECT_EQ(1.0, pose.at<double>(0));
    EXPECT_EQ(0.0, pose.at<double>(1));
    EXPECT_EQ(0.0, pose.at<double>(2));

    EXPECT_EQ(0.0, pose.at<double>(3));
    EXPECT_EQ(1.0, pose.at<double>(4));
    EXPECT_EQ(0.0, pose.at<double>(5));

    EXPECT_EQ(0.0, pose.at<double>(6));
    EXPECT_EQ(0.0, pose.at<double>(7));
    EXPECT_EQ(1.0, pose.at<double>(8));

    EXPECT_EQ(1.0, pose.at<double>(cv::Point(0, 0)));
    EXPECT_EQ(0.0, pose.at<double>(cv::Point(1, 0)));
    EXPECT_EQ(0.0, pose.at<double>(cv::Point(2, 0)));

    EXPECT_EQ(0.0, pose.at<double>(cv::Point(0, 1)));
    EXPECT_EQ(1.0, pose.at<double>(cv::Point(1, 1)));
    EXPECT_EQ(0.0, pose.at<double>(cv::Point(2, 1)));

    EXPECT_EQ(0.0, pose.at<double>(cv::Point(0, 2)));
    EXPECT_EQ(0.0, pose.at<double>(cv::Point(1, 2)));
    EXPECT_EQ(1.0, pose.at<double>(cv::Point(2, 2)));
}


TEST(Homography, transform) {
    Stick::Homography model;

    cv::Point pt = model.transform(cv::Point(2, 3));
    EXPECT_EQ(2, pt.x);
    EXPECT_EQ(3, pt.y);

    cv::Mat in = cv::Mat::zeros(cv::Size(1, 3), cv::DataType<double>::type);
    in.at<double>(0) = 2;
    in.at<double>(1) = 3;

    cv::Mat out = model.transform(in);
    EXPECT_EQ(2, out.at<double>(0));
    EXPECT_EQ(3, out.at<double>(1));

    cv::Mat pose = model.get();
    pose.at<double>(0, 2) = 2.0;
    pose.at<double>(1, 2) = 3.0;
    model.set(pose);

    pt = model.transform(cv::Point(2, 3));
    EXPECT_EQ(4, pt.x);
    EXPECT_EQ(6, pt.y);

    out = model.transform(in);
    EXPECT_EQ(4, out.at<double>(0));
    EXPECT_EQ(6, out.at<double>(1));
}

TEST(Homography, compose) {
    Stick::Homography model;

    cv::Mat pose = model.get();
    cv::Point pt;

    model.initialize();
    pose.at<double>(0, 0) = 1;
    pose.at<double>(0, 1) = 2;
    pose.at<double>(0, 2) = 3;
    pose.at<double>(1, 0) = 4;
    pose.at<double>(1, 1) = 5;
    pose.at<double>(1, 2) = 6;

    pt = model.transform(cv::Point(3, 4));
    EXPECT_EQ(3, pt.x);
    EXPECT_EQ(4, pt.y);

    model.compose( pose );
    pt = model.transform(cv::Point(3, 4));
    EXPECT_EQ(14, pt.x);
    EXPECT_EQ(38, pt.y);

    model.compose( pose );
    pt = model.transform(cv::Point(3, 4));
    EXPECT_EQ(1*14+2*38+3, pt.x);
    EXPECT_EQ(4*14+5*38+6, pt.y);
}

TEST(Homography, inverse) {
    Stick::Homography model;

    cv::Mat pose = model.get();
    cv::Point pt;

    model.initialize();
    pose.at<double>(0, 0) = 1;
    pose.at<double>(0, 1) = 2;
    pose.at<double>(0, 2) = 3;
    pose.at<double>(1, 0) = 4;
    pose.at<double>(1, 1) = 5;
    pose.at<double>(1, 2) = 6;

    model.compose( pose );
    pt = model.transform(cv::Point(3, 4));
    EXPECT_EQ(14, pt.x);
    EXPECT_EQ(38, pt.y);

    model.set( model.inverse() );
    pt = model.transform(cv::Point(14, 38));
    EXPECT_EQ(3, pt.x);
    EXPECT_EQ(4, pt.y);
}

TEST(Homography, jacobian) {
    Stick::Homography model;

    cv::Mat jacobian = model.jacobian(cv::Point(3, 4));
    EXPECT_EQ(3, jacobian.at<double>(0, 0));
    EXPECT_EQ(4, jacobian.at<double>(0, 1));
    EXPECT_EQ(1, jacobian.at<double>(0, 2));
    EXPECT_EQ(0, jacobian.at<double>(0, 3));
    EXPECT_EQ(0, jacobian.at<double>(0, 4));
    EXPECT_EQ(0, jacobian.at<double>(0, 5));
    EXPECT_EQ(-9, jacobian.at<double>(0, 6));
    EXPECT_EQ(-12, jacobian.at<double>(0, 7));
    EXPECT_EQ(0, jacobian.at<double>(0, 8));

    EXPECT_EQ(0, jacobian.at<double>(1, 0));
    EXPECT_EQ(0, jacobian.at<double>(1, 1));
    EXPECT_EQ(0, jacobian.at<double>(1, 2));
    EXPECT_EQ(3, jacobian.at<double>(1, 3));
    EXPECT_EQ(4, jacobian.at<double>(1, 4));
    EXPECT_EQ(1, jacobian.at<double>(1, 5));
    EXPECT_EQ(-12, jacobian.at<double>(1, 6));
    EXPECT_EQ(-16, jacobian.at<double>(1, 7));
    EXPECT_EQ(0, jacobian.at<double>(1, 8));
}

