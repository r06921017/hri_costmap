#include <ros/ros.h>
#include <ros/console.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>

#include <iostream>
#include <cmath>

using namespace std;

inline double getTFDist(tf::StampedTransform input_tf)
{
    return sqrt(pow(input_tf.getOrigin().x(), 2) + pow(input_tf.getOrigin().y(), 2) + pow(input_tf.getOrigin().z(), 2));
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "costmap_marker");
    ros::NodeHandle nh;
    tf::TransformListener tf_listener;
    tf::TransformBroadcaster br;

    int human_id = 0;
    int human_num = 2;
    int joint_num = 25;
    int needed_tf = 4;
    vector<tf::StampedTransform> body_tfs(human_num * joint_num);
    vector<geometry_msgs::PoseStamped> cur_states(human_num * joint_num);
    string camera_frame_id = "camera_rgb_optical_frame";

    //Using Kinect ordering for joints
    enum {SPINEBASE, SPINEMID, NECK, HEAD, 
        SHOULDERLEFT, ELBOWLEFT, WRISTLEFT, HANDLEFT, SHOULDERRIGHT, ELBOWRIGHT, WRISTRIGHT, HANDRIGHT, 
        HIPLEFT, KNEELEFT, ANKLELEFT, FOOTLEFT, HIPRIGHT, KNEERIGHT, ANKLERIGHT, FOOTRIGHT, 
        SPINESHOULDER, HANDTIPLEFT , THUMBLEFT, HANDTIPRIGHT, THUMBRIGHT};

    vector<string> joint_names({"SPINEBASE", "SPINEMID", "NECK", "HEAD", 
        "SHOULDERLEFT", "ELBOWLEFT", "WRISTLEFT", "HANDLEFT", "SHOULDERRIGHT", "ELBOWRIGHT", "WRISTRIGHT", "HANDRIGHT", 
        "HIPLEFT", "KNEELEFT", "ANKLELEFT", "FOOTLEFT", "HIPRIGHT", "KNEERIGHT", "ANKLERIGHT", "FOOTRIGHT", 
        "SPINESHOULDER", "HANDTIPLEFT", "THUMBLEFT", "HANDTIPRIGHT", "THUMBRIGHT"});

    // Create cylinder marker
    ros::Publisher state_marker_array_pub = nh.advertise<visualization_msgs::MarkerArray>("/cur_states", 10);
    visualization_msgs::MarkerArray marker_array;
    vector<int> needed_joint({SPINEBASE, SPINEMID, HEAD, HANDLEFT, HANDRIGHT});

    ros::Rate rate(10.0);
    bool exception_flag = true;
    while(ros::ok())
    {
        for (int idx = 0; idx < (int) needed_joint.size(); idx ++)
        {
            int joint_id = joint_num * human_id + needed_joint[idx];
            string src_tf = "/hri_costmap_"+to_string(human_id)+"_"+joint_names[needed_joint[idx]];
            string tar_tf = camera_frame_id;
            
            try
            {
                ros::Time now = ros::Time::now();
                tf_listener.waitForTransform(tar_tf, src_tf, now, ros::Duration(1.0));
                tf_listener.lookupTransform(tar_tf, src_tf, now, body_tfs[joint_id]);

                ROS_INFO("%s, x: %f", joint_names[needed_joint[idx]].c_str(), body_tfs[joint_id].getOrigin().x());
                ROS_INFO("%s, y: %f", joint_names[needed_joint[idx]].c_str(), body_tfs[joint_id].getOrigin().y());
                ROS_INFO("%s, z: %f", joint_names[needed_joint[idx]].c_str(), body_tfs[joint_id].getOrigin().z());

                geometry_msgs::PoseStamped tmp_joint;
                tmp_joint.header.stamp = now;
                tmp_joint.header.frame_id = tar_tf;
                tmp_joint.pose.position.x = floor((body_tfs[joint_id].getOrigin().x() * 10) + .5) / 10;
                tmp_joint.pose.position.y = floor((body_tfs[joint_id].getOrigin().y() * 10) + .5) / 10;
                tmp_joint.pose.position.z = floor((body_tfs[joint_id].getOrigin().z() * 10) + .5) / 10;
                tmp_joint.pose.orientation.x = 0.0;
                tmp_joint.pose.orientation.y = 0.0;
                tmp_joint.pose.orientation.z = 0.0;
                tmp_joint.pose.orientation.w = 1.0;

                ROS_INFO("%s, x: %f  %f", joint_names[needed_joint[idx]].c_str(), body_tfs[joint_id].getOrigin().x(), tmp_joint.pose.position.x);
                ROS_INFO("%s, y: %f  %f", joint_names[needed_joint[idx]].c_str(), body_tfs[joint_id].getOrigin().y(), tmp_joint.pose.position.y);
                ROS_INFO("%s, z: %f  %f", joint_names[needed_joint[idx]].c_str(), body_tfs[joint_id].getOrigin().z(), tmp_joint.pose.position.z);
                
                cur_states[joint_id] = tmp_joint;

                // Visualization
                visualization_msgs::Marker marker;
                marker.ns = "hri_costmap";
                marker.id = joint_id;
                marker.lifetime = ros::Duration(0.5);
                marker.header.stamp = ros::Time();
                marker.header.frame_id = tar_tf;
                marker.frame_locked = false;
                marker.type = visualization_msgs::Marker::CUBE;
                marker.action = visualization_msgs::Marker::ADD;

                marker.pose.position.x = tmp_joint.pose.position.x;
                marker.pose.position.y = tmp_joint.pose.position.y;
                marker.pose.position.z = tmp_joint.pose.position.z;
                marker.pose.orientation.x = tmp_joint.pose.orientation.x;
                marker.pose.orientation.y = tmp_joint.pose.orientation.y;
                marker.pose.orientation.z = tmp_joint.pose.orientation.z;
                marker.pose.orientation.w = tmp_joint.pose.orientation.w;
                
                marker.color.a = 0.5;
                marker.color.r = (human_id)?1:0;
                marker.color.g = 1;
                marker.color.b = (!human_id)?1:0;
                marker.scale.x = 0.1;
                marker.scale.y = 0.1;
                marker.scale.z = 0.1;

                marker_array.markers.push_back(marker);
            }
            catch(tf::TransformException ex)
            {
                ROS_ERROR("%s", ex.what());
                ros::Duration(1.0).sleep();
                continue;
            }
        }
        state_marker_array_pub.publish(marker_array);       
        rate.sleep();
    }

    return 0;
}
