import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import time
import random
import cv2
import copy
import numpy as np
from queue import Queue
from queue import Empty
from scipy import ndimage as nd

IM_WIDTH = 640
IM_HEIGHT = 480

def sensor_callback(sensor_data, sensor_queue, sensor_name):
    # Do stuff with the sensor_data data like save it to disk
    # Then you just need to add to the queue
    sensor_queue.put((sensor_data.frame, sensor_name, sensor_data))


def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def get_image_point(loc, K, w2c):
    # Calculate 2D projection of 3D coordinate

    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc.x, loc.y, loc.z, 1])
    # transform to camera coordinates
    point_camera = np.dot(w2c, point)

    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth component also
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    # now project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera)
    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2]


def parse_rgb_img_bounding_box(s_data, vehicle, camera, world, K):
    img = np.reshape(np.copy(s_data.raw_data), (s_data.height, s_data.width, 4))
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

    for npc in world.get_actors().filter('*vehicle*'):

        # Filter out the ego vehicle
        if npc.id != vehicle.id:

            bb = npc.bounding_box
            dist = npc.get_transform().location.distance(vehicle.get_transform().location)

            # Filter for the vehicles within 50m
            if dist < 50:

                # Calculate the dot product between the forward vector
                # of the vehicle and the vector between the vehicle
                # and the other vehicle. We threshold this dot product
                # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
                forward_vec = vehicle.get_transform().get_forward_vector()
                ray = npc.get_transform().location - vehicle.get_transform().location

                if forward_vec.dot(ray) > 1:
                    p1 = get_image_point(bb.location, K, world_2_camera)
                    verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                    x_max = -10000
                    x_min = 10000
                    y_max = -10000
                    y_min = 10000

                    for vert in verts:
                        p = get_image_point(vert, K, world_2_camera)
                        # Find the rightmost vertex
                        if p[0] > x_max:
                            x_max = p[0]
                        # Find the leftmost vertex
                        if p[0] < x_min:
                            x_min = p[0]
                        # Find the highest vertex
                        if p[1] > y_max:
                            y_max = p[1]
                        # Find the lowest  vertex
                        if p[1] < y_min:
                            y_min = p[1]

                    cv2.line(img, (int(x_min), int(y_min)), (int(x_max), int(y_min)),
                             (0, 0, 255, 255), 1)
                    cv2.line(img, (int(x_min), int(y_max)), (int(x_max), int(y_max)),
                             (0, 0, 255, 255), 1)
                    cv2.line(img, (int(x_min), int(y_min)), (int(x_min), int(y_max)),
                             (0, 0, 255, 255), 1)
                    cv2.line(img, (int(x_max), int(y_min)), (int(x_max), int(y_max)),
                             (0, 0, 255, 255), 1)

    return img


def parse_lidar(lidar_data):
    points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
    points = copy.deepcopy(points)
    points = np.reshape(points, (int(points.shape[0] / 4), 4))
    return points


def get_ground_truth(lidar, camera_int_mat, lidar_transform, camera_transform):
    lidar_out = np.array([lidar[:, 1], -lidar[:, 2], lidar[:, 0]]).reshape(-1, 3)
    idx = np.where(lidar_out[:,0]>2)[0]
    points = lidar_out[idx, :]
    points = points[:,0:3]

    Tlidar = np.array(lidar_transform.get_inverse_matrix())
    Tcamera = np.array(camera_transform.get_matrix())
    Tlidar_camera = np.dot(Tcamera, Tlidar)

    P_lidar_to_img = camera_int_mat.dot(Tlidar_camera[:3, :])
    img = project(points, P_lidar_to_img)
    img = np.around(img)

    # points_depth = points[:, 0]
    #
    # dim_y, dim_x = IM_HEIGHT, IM_WIDTH
    # depth_map = np.full((dim_y, dim_x), np.nan)
    #
    # lidar_y_min = np.inf
    # for iter_, (xx, yy) in enumerate(img):
    #     if xx >= 0  and xx <= dim_x-1 and yy >= 0 and yy <= dim_y-1 and points_depth[iter_] >= 0:
    #         depth_map[int(yy), int(xx)] = points_depth[iter_]
    #         if lidar_y_min > yy:
    #             lidar_y_min = yy
    #
    # depth_map[:int(lidar_y_min), :] = -1
    # return fill(depth_map)
    return dense_map(img, n, m, grid)


def fill(data, invalid=None):
    if invalid is None: invalid = np.isnan(data)
    ind = nd.distance_transform_edt(invalid,
                                    return_distances=False,
                                    return_indices=True)
    return data[tuple(ind)]


def project(point_clouds, proj_lidar_to_img):
        # dimension of data and projection matrix
        dim_norm = proj_lidar_to_img.shape[0]
        dim_proj = proj_lidar_to_img.shape[1]

        p2_in = point_clouds
        if point_clouds.shape[1] < dim_proj:
            ones_vec = np.ones((point_clouds.shape[0],1), dtype=np.float32)
            p2_in = np.append(point_clouds, ones_vec, axis=1)

        p2_out = np.transpose(proj_lidar_to_img.dot(np.transpose(p2_in)))
        # normalize homogeneous coordinates:
        p_out = p2_out[:,0:dim_norm-1] / np.outer(p2_out[:,dim_norm-1, None], np.ones(dim_norm-1))
        return p_out


def dense_map(Pts, n, m, grid):
    ng = 2 * grid + 1

    mX = np.zeros((m, n)) + np.float("inf")
    mY = np.zeros((m, n)) + np.float("inf")
    mD = np.zeros((m, n))
    mX[np.int32(Pts[1]), np.int32(Pts[0])] = Pts[0] - np.round(Pts[0])
    mY[np.int32(Pts[1]), np.int32(Pts[0])] = Pts[1] - np.round(Pts[1])
    mD[np.int32(Pts[1]), np.int32(Pts[0])] = Pts[2]

    KmX = np.zeros((ng, ng, m - ng, n - ng))
    KmY = np.zeros((ng, ng, m - ng, n - ng))
    KmD = np.zeros((ng, ng, m - ng, n - ng))

    for i in range(ng):
        for j in range(ng):
            KmX[i, j] = mX[i: (m - ng + i), j: (n - ng + j)] - grid - 1 + i
            KmY[i, j] = mY[i: (m - ng + i), j: (n - ng + j)] - grid - 1 + i
            KmD[i, j] = mD[i: (m - ng + i), j: (n - ng + j)]
    S = np.zeros_like(KmD[0, 0])
    Y = np.zeros_like(KmD[0, 0])

    for i in range(ng):
        for j in range(ng):
            s = 1 / np.sqrt(KmX[i, j] * KmX[i, j] + KmY[i, j] * KmY[i, j])
            Y = Y + s * KmD[i, j]
            S = S + s

    S[S == 0] = 1
    out = np.zeros((m, n))
    out[grid + 1: -grid, grid + 1: -grid] = Y / S
    return out

def main():
    actor_list = []
    sensor_list = []
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    # world = client.load_world('Town01')
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    weather = carla.WeatherParameters(
        cloudiness=80.0,
        precipitation=30.0,
        sun_altitude_angle=70.0)

    # world.set_weather(weather)

    world.set_weather(carla.WeatherParameters.MidRainyNoon)  #change weather here

    try:

        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        world.apply_settings(settings)
        # set the spectator to follow the ego vehicle
        spectator = world.get_spectator()
        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.global_percentage_speed_difference(65.0)

        # create the ego vehicle
        ego_vehicle_bp = random.choice(blueprint_library.filter('vehicle.*.*'))
        # black color
        ego_vehicle_bp.set_attribute('color', '0, 0, 0')
        # get a random valid occupation in the world
        spawn_points = world.get_map().get_spawn_points()
        transform = random.choice(spawn_points)
        # spawn the vehicle
        ego_vehicle = world.spawn_actor(ego_vehicle_bp, transform)
        # set the vehicle autopilot mode
        ego_vehicle.set_autopilot(True)
        # collect all actors to destroy when we quit the script
        actor_list.append(ego_vehicle)

        for i in range(100):
            vehicle_bp = random.choice(blueprint_library.filter('vehicle'))
            npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
            if npc:
                npc.set_autopilot(True)

        # # Set up the set of bounding boxes from the level
        # # We filter for traffic lights and traffic signs
        # bounding_box_set = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
        # bounding_box_set.extend(world.get_level_bbs(carla.CityObjectLabel.TrafficSigns))
        #
        # # Remember the edge pairs
        # edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5], [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]

        # create sensor queue
        sensor_queue = Queue()

        # add a camera
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', "{}".format(IM_WIDTH))
        camera_bp.set_attribute('image_size_y', "{}".format(IM_HEIGHT))
        camera_bp.set_attribute('fov', '120')
        camera_bp.set_attribute('sensor_tick', '3')
        camera_transform = carla.Transform(carla.Location(x=1.5, y=-0.06, z=2.4))
        camera_left = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)

        # Get the attributes from the camera
        image_w = camera_bp.get_attribute("image_size_x").as_int()
        image_h = camera_bp.get_attribute("image_size_y").as_int()
        fov = camera_bp.get_attribute("fov").as_float()
        # Calculate the camera projection matrix to project from 3D -> 2D
        K = build_projection_matrix(image_w, image_h, fov)
        # set the callback function
        camera_left.listen(lambda image: sensor_callback(image, sensor_queue, "rgb_left"))
        sensor_list.append(camera_left)

        # add another camera
        camera_transform = carla.Transform(carla.Location(x=1.5, y=0.06, z=2.4))
        camera_right = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
        # set the callback function
        camera_right.listen(lambda image: sensor_callback(image, sensor_queue, "rgb_right"))
        sensor_list.append(camera_right)


        # add a depth camera
        camera_dep = blueprint_library.find('sensor.camera.depth')
        camera_dep.set_attribute('image_size_x', "{}".format(IM_WIDTH))
        camera_dep.set_attribute('image_size_y', "{}".format(IM_HEIGHT))
        camera_dep.set_attribute('fov', '120')
        camera_dep.set_attribute('sensor_tick', '3')

        camera_transform = carla.Transform(carla.Location(x=1.5, y=0, z=2.4))
        camera_d = world.spawn_actor(camera_dep, camera_transform, attach_to=ego_vehicle)
        # set the callback function
        camera_d.listen(lambda image: sensor_callback(image, sensor_queue, "depth"))
        sensor_list.append(camera_d)

        #we also add a lidar on it
        # lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        # lidar_bp.set_attribute('channels', str(32))
        # lidar_bp.set_attribute('points_per_second', str(90000))
        # lidar_bp.set_attribute('rotation_frequency', str(40))
        # lidar_bp.set_attribute('range', str(20))
        #
        # # set the relative location
        # lidar_location = carla.Location(0, 0, 2)
        # lidar_rotation = carla.Rotation(0, 0, 0)
        # lidar_transform = carla.Transform(lidar_location, lidar_rotation)
        # # spawn the lidar
        # lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=ego_vehicle)
        # lidar.listen(
        #     lambda point_cloud: sensor_callback(point_cloud, sensor_queue, "lidar"))
        # sensor_list.append(lidar)
        count = 0
        break_num = 200

        while True:
            world.tick()

            transform = ego_vehicle.get_transform()
            spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50),
                                                    carla.Rotation(pitch=-90)))

            w_frame = world.get_snapshot().frame
            print("\nWorld's frame: %d" % w_frame)


            try:
                rgbs = []
                for i in range(0, len(sensor_list)):
                    s_frame, s_name, s_data = sensor_queue.get(True, 1.0)
                    print("    Frame: %d   Sensor: %s" % (s_frame, s_name))
                    sensor_type = s_name.split('_')[0]
                    if sensor_type == 'rgb':
                        if s_name.split('_')[1] == 'left':
                            # img = parse_rgb_img_bounding_box(s_data, ego_vehicle, camera_left, world, K)
                            s_data.save_to_disk(
                                os.path.join('./outputs/DefaultMap/MidRainyNoon', 'DefaultMap_MidRainyNoon_%06d_left.png' % s_data.frame))
                            count += 1
                            print(count)
                            # Now draw the image into the OpenCV display window
                            # cv2.imshow('ImageWindowName', img)
                            # cv2.waitKey(1)
                            # rgbs.append(img)
                        if s_name.split('_')[1] == 'right':
                            # img = parse_rgb_img_bounding_box(s_data, ego_vehicle, camera_right, world, K)
                            s_data.save_to_disk(
                                os.path.join('./outputs/DefaultMap/MidRainyNoon', 'DefaultMap_MidRainyNoon_%06d_right.png' % s_data.frame))
                            # Now draw the image into the OpenCV display window
                            # cv2.imshow('ImageWindowName', img)
                            # cv2.waitKey(1)
                            # rgbs.append(img)

                    elif sensor_type == 'depth':
                        s_data.save_to_disk(
                            os.path.join('./outputs/DefaultMap/MidRainyNoon','DefaultMap_MidRainyNoon_%06d_dep.jpg' % s_data.frame), carla.ColorConverter.LogarithmicDepth)

                    elif sensor_type == 'lidar':
                        lidar_out = parse_lidar(s_data)
                        canvas = get_ground_truth(lidar_out, K, lidar_transform, camera_transform)
                        cv2.imshow('ImageWindowName', canvas)
                        cv2.waitKey(1)
                #     elif sensor_type == 'imu':
                #         imu_yaw = s_data.compass
                #     elif sensor_type == 'gnss':
                #         gnss = s_data
                #
                # rgb = np.concatenate(rgbs, axis=1)[..., :3]
                # cv2.imshow('vizs', visualize_data(rgb, lidar, imu_yaw, gnss))
                # cv2.waitKey(100)
                # if rgb is None or args.save_path is not None:
                #     mkdir_folder(args.save_path)
                #
                #     filename = args.save_path + 'rgb/' + str(w_frame) + '.png'
                #     cv2.imwrite(filename, np.array(rgb[..., ::-1]))
                #     filename = args.save_path + 'lidar/' + str(w_frame) + '.npy'
                #     np.save(filename, lidar)

            except Empty:
                print("    Some of the sensor information is missed")

            if count == break_num:
                break

    finally:
        world.apply_settings(original_settings)
        print('destroying actors')
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        for sensor in sensor_list:
            sensor.destroy()
        print('done.')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')