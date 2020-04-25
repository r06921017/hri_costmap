/* Visualize the sphere for each joint*/

#include <ros/ros.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <shape_msgs/SolidPrimitive.h>
#include <moveit_msgs/CollisionObject.h>
#include <moveit_msgs/DisplayTrajectory.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/move_group_interface/move_group_interface.h>

#include <geometric_shapes/shape_operations.h>
#include <geometric_shapes/shapes.h>
#include <geometric_shapes/mesh_operations.h>

#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf/transform_listener.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <string>
#include <utility>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

//Using Kinect ordering for joints
enum {SPINEBASE, SPINEMID, NECK, HEAD, SHOULDERLEFT, ELBOWLEFT, WRISTLEFT, HANDLEFT, SHOULDERRIGHT, ELBOWRIGHT, WRISTRIGHT, HANDRIGHT, HIPLEFT, KNEELEFT, ANKLELEFT, FOOTLEFT, HIPRIGHT, KNEERIGHT, ANKLERIGHT, FOOTRIGHT, SPINESHOULDER, HANDTIPLEFT , THUMBLEFT, HANDTIPRIGHT, THUMBRIGHT};
vector<string> joint_names({"SPINEBASE", "SPINEMID", "NECK", "HEAD", "SHOULDERLEFT", "ELBOWLEFT", "WRISTLEFT", "HANDLEFT", "SHOULDERRIGHT", "ELBOWRIGHT", "WRISTRIGHT", "HANDRIGHT", "HIPLEFT", "KNEELEFT", "ANKLELEFT", "FOOTLEFT", "HIPRIGHT", "KNEERIGHT", "ANKLERIGHT", "FOOTRIGHT", "SPINESHOULDER", "HANDTIPLEFT", "THUMBLEFT", "HANDTIPRIGHT", "THUMBRIGHT"});
int num_of_joint = 25;
int num_of_people = 2;

int main(int argc, char** argv)
{
	ros::init(argc, argv, "hri_costmap_viz");
	ros::NodeHandle nh;
	ros::NodeHandle p_nh("~");

	moveit::planning_interface::MoveGroupInterface control_group("right_arm");
	moveit::planning_interface::PlanningSceneInterface current_scene;
	ros::Publisher display_publisher = nh.advertise<moveit_msgs::DisplayTrajectory>("/move_group/display_planned_path", 1, true);
	moveit_msgs::DisplayTrajectory display_trajectory;
	sleep(5.0);

	string skeleton_filename;
	
	if(!p_nh.getParam("skeleton_filename", skeleton_filename))
	{
		ROS_INFO("Please include the filename in the node's parameters in the launch file with the key \"skeleton_filename\"");
		ros::shutdown();
		return -1;
	}

	string camera_frame_id = p_nh.param(string("camera_frame_id"), string("camera"));
	
	static tf2_ros::TransformBroadcaster br;
	ros::Publisher skeleton_marker_array_pub = nh.advertise<visualization_msgs::MarkerArray>("/hri_costmap_viz", 10);
	
	std::vector<moveit_msgs::CollisionObject> collision_objects;

	/* Define a box to add to the world. */
	shape_msgs::SolidPrimitive primitive;
	primitive.type = primitive.CYLINDER;
	primitive.dimensions.resize(2);
	primitive.dimensions[primitive.CYLINDER_HEIGHT] = 1;
	primitive.dimensions[primitive.CYLINDER_RADIUS] = 0.5;

	moveit_msgs::CollisionObject collision_object;
	collision_object.header.frame_id = control_group.getPlanningFrame();
	collision_object.id = "obs_1"; 

	/* A pose for the box (specified relative to frame_id) */
	geometry_msgs::Pose box_pose;
	box_pose.orientation.w = 1.0;
	box_pose.position.x =  0.6;
	box_pose.position.y = -0.4;
	box_pose.position.z =  1.2;

	collision_object.primitives.push_back(primitive);
	collision_object.primitive_poses.push_back(box_pose);
	collision_object.operation = collision_object.ADD;
	collision_objects.push_back(collision_object);

	// add the collision object into the world
	ROS_INFO("Add an object into the world");
	current_scene.addCollisionObjects(collision_objects);

	// Read and visualize body joints
	vector<geometry_msgs::TransformStamped> body_transforms(num_of_joint * num_of_people);  //25 per person and maximum 2 people in a frame
	vector<double> radius(num_of_joint*num_of_people);
	visualization_msgs::MarkerArray marker_array;
	
	for(int i = 0; i < 2; i ++)
	{
		for(int j = 0; j < 25; j ++)
		{
			int index = i*25 + j;
			body_transforms[index].header.seq = 0;
			body_transforms[index].header.frame_id = camera_frame_id;
			body_transforms[index].child_frame_id = "hri_costmap_"+to_string(i)+"_"+joint_names[j];

			visualization_msgs::Marker marker;
			marker.ns = "hri_costmap";
			marker.id = index;
			marker.lifetime = ros::Duration(0.5);
			marker.header.stamp = ros::Time();
			marker.header.frame_id = body_transforms[index].child_frame_id;
			marker.frame_locked = false;
			marker.type = visualization_msgs::Marker::CUBE;
			marker.action = visualization_msgs::Marker::ADD;

			marker.color.a = 1;
			marker.color.r = (i)?1:0;
			marker.color.g = 0;
			marker.color.b = (!i)?1:0;

			marker.scale.x = 0.1;
			marker.scale.y = 0.1;
			marker.scale.z = 0.1;

			marker_array.markers.push_back(marker);
		}
	}

	ROS_INFO("Opening file %s", skeleton_filename.c_str());
		
	ifstream skeleton_file(skeleton_filename);
	if(!skeleton_file.is_open())
	{
		ROS_INFO("Unable to open file %s", skeleton_filename.c_str());
		ros::shutdown();
		return -1;
	}

	ros::Rate r(10);
	while(ros::ok())
	{
		skeleton_file.seekg(0);
		int frame_count;
		skeleton_file >> frame_count;
		if(frame_count==0)
		{
			ROS_INFO("No frames present");
			ros::shutdown();
			return -1;
		}

		for(int frame = 0; frame < frame_count; frame ++)
		{
			int body_count;
			skeleton_file >> body_count; // no. of observed skeletons in current frame
			
			ros::Time t = ros::Time::now();

			for(int body_num = 0; body_num < body_count; body_num ++)
			{
				long int bodyID;
				skeleton_file >> bodyID;
				
				int clipedEdges, handLeftConfidence, handLeftState, handRightConfidence, handRightState, isResticted;
				skeleton_file >> clipedEdges >> handLeftConfidence >> handLeftState >> handRightConfidence >> handRightState >> isResticted;

				float leanX, leanY;
				skeleton_file >> leanX >> leanY;

				int bodyTrackingState;
				skeleton_file >> bodyTrackingState;

				int joint_count;
				skeleton_file >> joint_count; 

				for (int joint_num = 0; joint_num < joint_count; joint_num ++)
				{
					float x, y, z, depthX, depthY, colorX, colorY, orientationW, orientationX, orientationY, orientationZ;
					skeleton_file >> x >> y >> z >> depthX >> depthY >> colorX >> colorY >> orientationW >> orientationX >> orientationY >> orientationZ;
					
					int jointTrackingState;
					skeleton_file >> jointTrackingState;					

					if(bodyTrackingState and jointTrackingState)
					{
						body_transforms[body_num*25 + joint_num].transform.translation.x = x;
						body_transforms[body_num*25 + joint_num].transform.translation.y = y;
						body_transforms[body_num*25 + joint_num].transform.translation.z = z;

						if(orientationX==orientationY && orientationY==orientationZ and orientationZ==orientationW and orientationW==0.)
							orientationW = 1.;

						body_transforms[body_num*25 + joint_num].transform.rotation.x = orientationX;
						body_transforms[body_num*25 + joint_num].transform.rotation.y = orientationY;
						body_transforms[body_num*25 + joint_num].transform.rotation.z = orientationZ;
						body_transforms[body_num*25 + joint_num].transform.rotation.w = orientationW;

						body_transforms[body_num*25 + joint_num].header.stamp = t;
						
						ROS_INFO("TRACKING JOINT %d of SKELETON %d in FRAME %d as %.3f %.3f %.3f %.3f %.3f %.3f %.3f",joint_num, body_num, frame, x, y, z, orientationX, orientationY, orientationZ, orientationW);
					}

					// if (joint_num == SHOULDERLEFT and body_num == 1)
					// {

					// }
				}
			}

			br.sendTransform(body_transforms);
			skeleton_marker_array_pub.publish(marker_array);
			r.sleep();
		}
	}
	skeleton_file.close();

	return 0;
}
