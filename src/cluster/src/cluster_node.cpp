#include <ros/ros.h>
#include <pcl/io/pcd_io.h>
#include <pcl/segmentation/extract_clusters.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/filter.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/angles.h>
#include <pcl/common/common.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <opencv2/opencv.hpp>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/plane_clipper3D.h>

struct ClusterInfo
{
    pcl::PointCloud<pcl::PointXYZI> cloud;
    Eigen::Vector3f center;
    Eigen::Vector3f size;
    Eigen::Matrix3f pose;
};

double lidarHeight, clusterTolerance, planeTolerance;
int minClusterSize;

pcl::visualization::PCLVisualizer::Ptr initViewer(int &v0)
{
    pcl::visualization::PCLVisualizer::Ptr pViewer(
        new pcl::visualization::PCLVisualizer("cluster"));

    pViewer->createViewPort(0, 0, 1.0, 1.0, v0);
    pViewer->setBackgroundColor(0, 0, 0, v0);
    pViewer->addCoordinateSystem(1.0, "cs0", v0);
    pViewer->initCameraParameters();

    return pViewer;
}


Eigen::Vector3f groundRemoval(pcl::PointCloud<pcl::PointXYZI> laser,
                              pcl::PointCloud<pcl::PointXYZI> &ground,
                              pcl::PointCloud<pcl::PointXYZI> &left)
{
    Eigen::Vector4f groundDirection = Eigen::Vector4f(0.0, 0.0, 1.0, lidarHeight - planeTolerance * 2.0);
    pcl::PlaneClipper3D<pcl::PointXYZI> clipperFilter(groundDirection);

    pcl::PointIndices::Ptr pInliers(new pcl::PointIndices);
    clipperFilter.clipPointCloud3D(laser, pInliers->indices);

    pcl::PointCloud<pcl::PointXYZI> candidate;
    pcl::ExtractIndices<pcl::PointXYZI> ex;
    ex.setInputCloud(laser.makeShared());
    ex.setIndices(pInliers);
    ex.setNegative(true);
    ex.filter(candidate);
    ex.setNegative(false);
    ex.filter(left);

    pcl::SACSegmentation<pcl::PointXYZI> sac;
    sac.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
    sac.setMethodType(pcl::SAC_RANSAC);
    sac.setDistanceThreshold(planeTolerance);
    sac.setMaxIterations(100);
    sac.setProbability(0.8);
    sac.setAxis(Eigen::Vector3f::UnitZ());
    sac.setEpsAngle(pcl::deg2rad(30.0));
    sac.setInputCloud(candidate.makeShared());

    pInliers->indices.clear();
    pcl::ModelCoefficients::Ptr pCoeffs(new pcl::ModelCoefficients);
    sac.segment(*pInliers, *pCoeffs);

    ex.setInputCloud(candidate.makeShared());
    ex.setIndices(pInliers);
    ex.filter(ground);

    ex.setNegative(true);
    pcl::PointCloud<pcl::PointXYZI> tmp;
    ex.filter(tmp);

    left += tmp;

    return Eigen::Vector3f(pCoeffs->values[0], pCoeffs->values[1], pCoeffs->values[2]);
}


void callback(const sensor_msgs::PointCloud2::ConstPtr &msg, pcl::visualization::PCLVisualizer::Ptr pViewer)
{
    pcl::PointCloud<pcl::PointXYZI> laser;
    pcl::fromROSMsg(*msg, laser);

    // remove NAN
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(laser, laser, indices);

    pcl::CropBox<pcl::PointXYZI> cropFilter;
    cropFilter.setInputCloud(laser.makeShared());
    cropFilter.setMin(Eigen::Vector4f(-1.0, -1.0, -100.0, 1.0));
    cropFilter.setMax(Eigen::Vector4f(1.0, 1.0, 100.0, 1.0));
    cropFilter.setNegative(true);
    cropFilter.filter(laser);

    pcl::PointCloud<pcl::PointXYZI> ground, left, leftViewer;
    Eigen::Vector3f groundDirection = groundRemoval(laser, ground, left);
	pcl::copyPointCloud(left, leftViewer);

    float angle = acos(groundDirection.dot(Eigen::Vector3f::UnitZ()));
    Eigen::Vector3f direction = groundDirection.cross(Eigen::Vector3f::UnitZ());
    direction.normalize();
    Eigen::AngleAxisf q(angle, direction);
    Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
    trans.block<3,3>(0,0) = q.matrix();
    pcl::transformPointCloud(left, left, trans);

    pcl::PointCloud<pcl::PointXYZI> laser1;
	pcl::PointCloud<pcl::PointXYZI> laser2;
    for(pcl::PointXYZI point : left)
    {
        if (point.z > 0.0)
            continue;
        float dis = sqrt(point.x * point.x + point.y * point.y);
        if (dis < 50.0)
            laser1.push_back(point);
		else
			laser2.push_back(point);
    }

	/*
    pcl::RadiusOutlierRemoval<pcl::PointXYZI> radiusFilter;
    radiusFilter.setInputCloud(laser1.makeShared());
    radiusFilter.setRadiusSearch(0.1);
    radiusFilter.setMinNeighborsInRadius(4);
    radiusFilter.filter(laser1);
	*/

    pcl::search::KdTree<pcl::PointXYZI>::Ptr pTree(new pcl::search::KdTree<pcl::PointXYZI>);
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> clusterExtraction;
    clusterExtraction.setSearchMethod(pTree);
    clusterExtraction.setMinClusterSize(minClusterSize);
    clusterExtraction.setMaxClusterSize(INT_MAX);

    std::vector<pcl::PointIndices> clusters1;
    clusterExtraction.setClusterTolerance(clusterTolerance);
    clusterExtraction.setInputCloud(laser1.makeShared());
    clusterExtraction.extract(clusters1);

	std::vector<pcl::PointIndices> clusters2;
	clusterExtraction.setClusterTolerance(clusterTolerance * 2.0);
	clusterExtraction.setInputCloud(laser2.makeShared());
	clusterExtraction.extract(clusters2);

    pViewer->removeAllPointClouds();
    pViewer->removeAllShapes();

    pcl::PointCloud<pcl::PointXYZI>::Ptr pLaser(new pcl::PointCloud<pcl::PointXYZI>(leftViewer));
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI>
        colorGreen(pLaser, 0.0, 255.0, 0.0);
    pViewer->addPointCloud(pLaser, colorGreen, "cloud");

    pcl::PointCloud<pcl::PointXYZI>::Ptr pGround(new pcl::PointCloud<pcl::PointXYZI>(ground));
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI>
        colorRed(pGround, 255.0, 0.0, 0.0);
    pViewer->addPointCloud(pGround, colorRed, "ground");

    for(int i = 0; i < int(clusters1.size()); i ++)
    {
        pcl::PointCloud<pcl::PointXYZI> clusterCloud;
        pcl::copyPointCloud(laser1, clusters1[i], clusterCloud);

        pcl::PointXYZI minPt, maxPt;
        pcl::getMinMax3D(clusterCloud, minPt, maxPt);
        if (maxPt.z - minPt.z < 0.1)
            continue;

        std::vector<cv::Point2f> cvPoints;
        for(pcl::PointXYZI point : clusterCloud)
            cvPoints.push_back(cv::Point2f(point.x, point.y));
    
        cv::RotatedRect rect = cv::minAreaRect(cvPoints);

        ClusterInfo info;
        info.center = Eigen::Vector3f(rect.center.x, rect.center.y, (maxPt.z + minPt.z) / 2.0);
        info.size = Eigen::Vector3f(rect.size.width, rect.size.height, maxPt.z - minPt.z);
        Eigen::AngleAxisf q2(pcl::deg2rad(rect.angle), Eigen::Vector3f::UnitZ());
        info.pose.col(0) = q2.matrix() * Eigen::Vector3f::UnitX();
        info.pose.col(2) = Eigen::Vector3f::UnitZ();
        info.pose.col(1) = -(info.pose.col(0).cross(Eigen::Vector3f::UnitZ()));
        info.pose.col(1).normalize();
        info.cloud += clusterCloud;

        info.center = q.matrix().inverse() * info.center;
        info.pose = q.matrix().inverse() * info.pose;
        Eigen::Matrix4f transInv = Eigen::Matrix4f::Identity();
        transInv.block<3,3>(0,0) = q.matrix().inverse();
        pcl::transformPointCloud(info.cloud, info.cloud, transInv);

        Eigen::Quaternionf q_guess(info.pose);
        pViewer->addCube(info.center, q_guess, info.size(0), info.size(1), info.size(2), "cube" + std::to_string(i));
        pViewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
            pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "cube" + std::to_string(i));
    }

	for(int i = 0; i < int(clusters2.size()); i ++)
	{
		pcl::PointCloud<pcl::PointXYZI> clusterCloud;
		pcl::copyPointCloud(laser2, clusters2[i], clusterCloud);

		pcl::PointXYZI minPt, maxPt;
		if (maxPt.z - minPt.z < 0.1)
			continue;

		std::vector<cv::Point2f> cvPoints;
		for(pcl::PointXYZI point: clusterCloud)
			cvPoints.push_back(cv::Point2f(point.x, point.y));

		cv::RotatedRect rect = cv::minAreaRect(cvPoints);
	
        ClusterInfo info;
        info.center = Eigen::Vector3f(rect.center.x, rect.center.y, (maxPt.z + minPt.z) / 2.0);
        info.size = Eigen::Vector3f(rect.size.width, rect.size.height, maxPt.z - minPt.z);
        Eigen::AngleAxisf q2(pcl::deg2rad(rect.angle), Eigen::Vector3f::UnitZ());
        info.pose.col(0) = q2.matrix() * Eigen::Vector3f::UnitX();
        info.pose.col(2) = Eigen::Vector3f::UnitZ();
        info.pose.col(1) = -(info.pose.col(0).cross(Eigen::Vector3f::UnitZ()));
        info.pose.col(1).normalize();
        info.cloud += clusterCloud;

        info.center = q.matrix().inverse() * info.center;
        info.pose = q.matrix().inverse() * info.pose;
        Eigen::Matrix4f transInv = Eigen::Matrix4f::Identity();
        transInv.block<3,3>(0,0) = q.matrix().inverse();
        pcl::transformPointCloud(info.cloud, info.cloud, transInv);

        Eigen::Quaternionf q_guess(info.pose);
        pViewer->addCube(info.center, q_guess, info.size(0), info.size(1), info.size(2), "cube" + std::to_string(i));
        pViewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
            pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "cube" + std::to_string(i));
	}

    pViewer->spinOnce(10);

}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "cluster");
    ros::NodeHandle nh("~");

    nh.param("lidar_height", lidarHeight, 1.0);
    nh.param("cluster_tolerance", clusterTolerance, 0.2);
    nh.param("min_cluster_size", minClusterSize, 20);
    nh.param("plane_tolerance", planeTolerance, 0.1);

    std::cout << "lidar_height: " << lidarHeight << std::endl;
    std::cout << "cluster_tolerance: " << clusterTolerance << std::endl;
    std::cout << "min_cluster_size: " << minClusterSize << std::endl;
    std::cout << "plane_tolerance: " << planeTolerance << std::endl;

    int v0 = 0;
    pcl::visualization::PCLVisualizer::Ptr pViewer = initViewer(v0);

    ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/cti/sensor/rslidar/PointCloud2", 100, boost::bind(&callback, _1, pViewer));

    ros::spin();

    return 0;
}
