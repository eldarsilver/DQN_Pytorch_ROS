#! /usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan
import math
import numpy



def discretize_observation(data, new_ranges):
    """
    Discards all the laser readings that are not multiple in index of new_ranges
    value.
    """
    

    discretized_ranges = []
    filtered_range = []
    #mod = len(data.ranges)/new_ranges
    mod = new_ranges

    max_laser_value = data.range_max
    min_laser_value = data.range_min

    rospy.logdebug("data=" + str(data))
    rospy.logwarn("data.range_max= %s" % data.range_max)
    rospy.logwarn("data.range_min= %s" % data.range_min)
    rospy.logwarn("len(data.ranges)= %s" % len(data.ranges))
    rospy.logwarn("data.angle_min= %s" % data.angle_min)
    rospy.logwarn("data.angle_max= %s" % data.angle_max)
    rospy.logwarn("data.angle_increment= %s" % data.angle_increment)
    rospy.logwarn("mod=" + str(mod))
    min_range = 0.5

    for i, item in enumerate(data.ranges):
        if (i % mod == 0):
            if item == float('Inf') or numpy.isinf(item):
                # discretized_ranges.append(self.max_laser_value)
                discretized_ranges.append(
                    round(max_laser_value, 1))
            elif numpy.isnan(item):
                # discretized_ranges.append(self.min_laser_value)
                discretized_ranges.append(
                    round(min_laser_value, 1))
            else:
                # discretized_ranges.append(int(item))
                discretized_ranges.append(round(item, 1))

            if (min_range > item > 0):
                rospy.logerr("done Validation >>> item=" +
                             str(item)+"< "+str(min_range))
        
            else:
                rospy.logwarn("NOT done Validation >>> item=" +
                                  str(item)+"< "+str(min_range))
            # We add last value appended
            filtered_range.append(discretized_ranges[-1])
        else:
            # We add value zero
            filtered_range.append(0.1)

    rospy.logdebug(
        "Size of observations, discretized_ranges==>"+str(len(discretized_ranges)))

    return discretized_ranges


 
def callback(msg):
    distance=0.8
    print(len(msg.ranges)) # 360
    rospy.loginfo('right msg.ranges[90] %s' % msg.ranges[90])
    rospy.loginfo('left msg.ranges[269] %s ' % msg.ranges[269]) 
    rospy.loginfo('back msg.ranges[359] %s' % msg.ranges[359])
    rospy.loginfo('back msg.ranges[0] %s' % msg.ranges[0])
    rospy.loginfo('front msg.ranges[179] %s' % msg.ranges[179])
    closest = min(msg.ranges)         
    rospy.loginfo('Laser distancia  %s' % closest)
    new_ranges = int(math.ceil(float(len(msg.ranges)) / float(6)))

    discretized_observations = discretize_observation(msg, new_ranges)
    rospy.loginfo('back right 60 discretized_observations[0] %s' % discretized_observations[0])  
    rospy.loginfo('right 120 discretized_observations[1] %s' % discretized_observations[1]) 
    rospy.loginfo('left 240 discretized_observations[3] %s ' % discretized_observations[3]) 
    rospy.loginfo('back 360 discretized_observations[5] %s' % discretized_observations[5])
    rospy.loginfo('back left 300 discretized_observations[4] %s' % discretized_observations[4])
    rospy.loginfo('front 180 discretized_observations[2] %s' % discretized_observations[2])       
    if msg.ranges[179] < distance:             
        rospy.loginfo('Parar robot')             


 
rospy.init_node('scan_values')
sub = rospy.Subscriber('/scan', LaserScan, callback)
rospy.spin()
