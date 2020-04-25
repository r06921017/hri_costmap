#include <ros/ros.h>
#include <ros/console.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/TransformStamped.h>

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
    vector<tf::StampedTransform> body_tfs(human_num * needed_tf);

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
	ros::Publisher costmap_marker_array_pub = nh.advertise<visualization_msgs::MarkerArray>("/hri_costmap_marker", 1);
    visualization_msgs::MarkerArray marker_array;

    ros::Rate rate(10.0);
    bool exception_flag = true;
    while(ros::ok())
    {
        for (int idx = 0; idx < needed_tf; idx ++)
        {
            int joint_id = needed_tf * human_id + idx;
            string tar_tf;
            string src_tf;
            
            switch (idx)
            {
            case 0:
                tar_tf = "/hri_costmap_"+to_string(human_id)+"_"+joint_names[SPINEMID];
                src_tf = "/temp_link";
                break;

            case 1:
                tar_tf = "/hri_costmap_"+to_string(human_id)+"_"+joint_names[HEAD];
                src_tf = "/hri_costmap_"+to_string(human_id)+"_"+joint_names[SPINEBASE];
                break;

            case 2:
                tar_tf = "/hri_costmap_"+to_string(human_id)+"_"+joint_names[SPINEMID];
                src_tf = "/hri_costmap_"+to_string(human_id)+"_"+joint_names[HANDLEFT];
                break;

            case 3:
                tar_tf = "/hri_costmap_"+to_string(human_id)+"_"+joint_names[SPINEMID];
                src_tf = "/hri_costmap_"+to_string(human_id)+"_"+joint_names[HANDRIGHT];
                break;

            default:
                break;
            }
            
            try
            {
                ros::Time now = ros::Time::now();
                tf_listener.waitForTransform(tar_tf, src_tf, now, ros::Duration(1.0));
                tf_listener.lookupTransform(tar_tf, src_tf, now, body_tfs[joint_id]);
            }
            catch(tf::TransformException ex)
            {
                ROS_ERROR("%s", ex.what());
                ros::Duration(1.0).sleep();
                continue;
            }
        }       

        ROS_INFO("%f,  %f, %f", body_tfs[needed_tf*human_id+3].getOrigin().x(), body_tfs[needed_tf*human_id+3].getOrigin().y(), body_tfs[needed_tf*human_id+3].getOrigin().z());
        double cyl_h = getTFDist(body_tfs[needed_tf*human_id+1]);
        double cyl_w = sqrt(pow(body_tfs[needed_tf*human_id+3].getOrigin().x(), 2)+pow(body_tfs[needed_tf*human_id+3].getOrigin().z(), 2));

        tf::Transform cyl_tf;
        cyl_tf.setOrigin(tf::Vector3(0.0, 0.0, 0.0));
        cyl_tf.setRotation(tf::createQuaternionFromRPY(-M_PI/2, 0, 0));
        br.sendTransform(
            tf::StampedTransform(   cyl_tf, 
                                    ros::Time::now(), 
                                    body_tfs[needed_tf * human_id].frame_id_, 
                                    "/costmap_link"));


        visualization_msgs::Marker marker;
        marker.ns = "hri_costmap";
        marker.id = human_id;
        marker.lifetime = ros::Duration(0.5);
        marker.header.stamp = ros::Time::now();
        marker.header.frame_id = "/costmap_link";
        marker.frame_locked = false;
        marker.type = visualization_msgs::Marker::CYLINDER;
        marker.action = visualization_msgs::Marker::ADD;

        marker.color.a = 0.25;
        marker.color.r = (human_id)?1:0;
        marker.color.g = 0;
        marker.color.b = (!human_id)?1:0;

        marker.scale.x = cyl_w;
        marker.scale.y = cyl_w;
        marker.scale.z = cyl_h;

        ROS_INFO("width: %f, height: %f", cyl_w, cyl_h);

        marker_array.markers.push_back(marker);
        costmap_marker_array_pub.publish(marker_array);

        rate.sleep();
    }

    return 0;
}
