#include <ros/ros.h>
#include <stdio.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/PointCloud.h>

int main(int argc, char **argv)
{
    // setup ros for this node and get handle to ros system
    ros::init(argc, argv, "fake_pointcloud");
    ros::start();

    // get node handle
    ros::NodeHandle n;
    ros::Rate loopRate(1.0);
    std::string topicName = "cloud_topic";

    ros::Publisher demoPublisher = n.advertise<pcl::PointCloud<pcl::PointXYZ> >(topicName.c_str(),10);
    // ros::Publisher demoPublisher = n.advertise<sensor_msgs::PointCloud>(topicName.c_str(),10);

    ROS_INFO("Publishing point cloud on topic \"%s\" once every second.", topicName.c_str());

    while (ros::ok())
    {
        // create point cloud object
        pcl::PointCloud<pcl::PointXYZ> myCloud;
        // sensor_msgs::PointCloud myCloud;
        myCloud.header.frame_id = "camera_rgb_optical_frame";

        // fill cloud with random points
        for (int v=0; v<1000; ++v)
        {
            pcl::PointXYZ newPoint;
            newPoint.x = (rand() * 100.0) / RAND_MAX;
            newPoint.y = (rand() * 100.0) / RAND_MAX;
            newPoint.z = (rand() * 100.0) / RAND_MAX;
            myCloud.points.push_back(newPoint);
        }
        // for (int v=0; v < 1000; ++v)
        // {
        //     geometry_msgs::Point32 newPoint;
        //     newPoint.x = (rand() * 100.0) / RAND_MAX;
        //     newPoint.y = (rand() * 100.0) / RAND_MAX;
        //     newPoint.z = (rand() * 100.0) / RAND_MAX;
        //     myCloud.points.push_back(newPoint);

        //     sensor_msgs::ChannelFloat32 newChannel;
        //     newChannel.name = "rgb";
        //     newChannel.values = std::vector<float>{0, 255, 255};
        //     myCloud.channels.push_back(newChannel);
        // }

        // publish point cloud
        demoPublisher.publish(myCloud);

        // pause for loop delay
        loopRate.sleep();
    }

    return 1;
}