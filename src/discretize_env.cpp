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
#include <fstream>

using namespace std;

inline double getTFDist(tf::StampedTransform input_tf)
{
    return sqrt(pow(input_tf.getOrigin().x(), 2) + pow(input_tf.getOrigin().y(), 2) + pow(input_tf.getOrigin().z(), 2));
}

void saveResults(const string& file_name, string joint_name, int frame, double x1, double y1, double z1, double x2, double y2, double z2)
{
	std::ifstream infile(file_name);
	bool exist = infile.good();
	infile.close();
	if (!exist)
	{
		ofstream addHeads(file_name);
		addHeads << "joint,frame,x1,y1,z1,x2,y2,z2" << endl;
		addHeads.close();
	}

	ofstream stats(file_name, std::ios::app);
	stats << joint_name << "," << frame << "," << x1 << "," << y1 << "," << z1 << "," << x2 << "," << y2 << "," << z2 << endl;
	stats.close();
    return;
}

void split(const std::string& s, std::vector<std::string>& sv, const char* delim = " ") 
{
    sv.clear();
    char* buffer = new char[s.size() + 1];
    buffer[s.size()] = '\0';

    std::copy(s.begin(), s.end(), buffer);
    char* p = std::strtok(buffer, delim);

    do {
        sv.emplace_back(p);
    } while ((p = std::strtok(NULL, delim)));

    delete[] buffer;

    return;
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


    // Save trajectory files
    int frame_count;
    nh.getParam("/hri_costmap_node/frame_count", frame_count);
    vector<int> needed_joint({HANDRIGHT});
    vector<bool> get_frame(frame_count, false);
    vector<vector<geometry_msgs::PoseStamped>> output_states(frame_count);

    ros::Rate rate(5.0);
    bool exception_flag = true;
    int cur_frame;

    while(ros::ok())
    {
        nh.getParam("/hri_costmap_node/cur_frame", cur_frame);                
        cur_states =vector<geometry_msgs::PoseStamped>(human_num * joint_num);
        ROS_INFO("frame: %d", cur_frame);
        for (int human_id = 0; human_id < human_num; human_id++)
        {
            ROS_INFO("human: %d", human_id);
            for (int idx = 0; idx < (int) needed_joint.size(); idx ++)
            {
                int joint_id = joint_num * human_id + needed_joint[idx];
                string src_tf = "/hri_costmap_"+to_string(human_id)+"_"+joint_names[needed_joint[idx]];
                string tar_tf = camera_frame_id;
                
                try
                {
                    ros::Time now = ros::Time::now();
                    tf_listener.waitForTransform(tar_tf, src_tf, ros::Time(), ros::Duration(1.0));
                    tf_listener.lookupTransform(tar_tf, src_tf, ros::Time(), body_tfs[joint_id]);

                    geometry_msgs::PoseStamped tmp_joint;
                    tmp_joint.header.stamp = now;
                    tmp_joint.header.frame_id = tar_tf;
                    // tmp_joint.pose.position.x = floor((body_tfs[joint_id].getOrigin().x() * 10) + .5) / 10;
                    // tmp_joint.pose.position.y = floor((body_tfs[joint_id].getOrigin().y() * 10) + .5) / 10;
                    // tmp_joint.pose.position.z = floor((body_tfs[joint_id].getOrigin().z() * 10) + .5) / 10;

                    tmp_joint.pose.position.x = body_tfs[joint_id].getOrigin().x();
                    tmp_joint.pose.position.y = body_tfs[joint_id].getOrigin().y();
                    tmp_joint.pose.position.z = body_tfs[joint_id].getOrigin().z();
                    
                    tmp_joint.pose.orientation.x = 0.0;
                    tmp_joint.pose.orientation.y = 0.0;
                    tmp_joint.pose.orientation.z = 0.0;
                    tmp_joint.pose.orientation.w = 1.0;

                    cur_states[joint_id] = tmp_joint;
                    ROS_INFO("h %d: %f, %f, %f", human_id, tmp_joint.pose.position.x, tmp_joint.pose.position.y, tmp_joint.pose.position.z);

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
        }

        output_states[cur_frame] = cur_states;
        get_frame[cur_frame] = true;
        state_marker_array_pub.publish(marker_array);
        nh.setParam("/hri_costmap_node/received", true);

        if (std::all_of(get_frame.begin(), get_frame.end(), [](bool v) { return v; }))
        {
            break;
        }

        rate.sleep();
    }

    ROS_INFO("Saving file....");
    string file_name;
    vector<string> file_name_vec;
    nh.getParam("/hri_costmap_node/filename", file_name);
    split(file_name, file_name_vec, "/");
    file_name = file_name_vec.back();
    split(file_name, file_name_vec, ".");
    file_name = "/home/rdaneel/hri_costmap_traj/" + file_name_vec[0] + ".csv";
    ROS_INFO("%s", file_name.c_str());

    int save_frame = 0;
    for (const auto & it : output_states)
    {
        for (int idx = 0; idx < (int) needed_joint.size(); idx ++)
        {
            int joint_id = 0 + needed_joint[idx];
            double x1 = it[joint_id].pose.position.x;
            double y1 = it[joint_id].pose.position.y;
            double z1 = it[joint_id].pose.position.z;

            joint_id = joint_num + needed_joint[idx];
            double x2 = it[joint_id].pose.position.x;
            double y2 = it[joint_id].pose.position.y;
            double z2 = it[joint_id].pose.position.z;

            saveResults(file_name, joint_names[needed_joint[idx]].c_str(), save_frame, x1, y1, z1, x2, y2, z2);
        }
        save_frame ++;
    }

    ROS_INFO("Done!");
    return 0;
}
