import numpy as np
import open3d as o3d
import torch
import json

# Transforms
class PointcloudNoise(object):
    ''' Point cloud noise transformation class.

    It adds noise to point cloud data.

    Args:
        stddev (int): standard deviation
    '''

    def __init__(self, stddev):
        self.stddev = stddev

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        data_out = data.copy()
        points = data[None]
        noise = self.stddev * np.random.randn(*points.shape)
        noise = noise.astype(np.float32)
        data_out[None] = points + noise
        return data_out


class SubsamplePointcloud(object):
    ''' Point cloud subsampling transformation class.

    It subsamples the point cloud data.

    Args:
        N (int): number of points to be subsampled
    '''

    def __init__(self, N):
        if isinstance(N, int):
            self.N = N
        else:
            with open(N) as json_file:
                grasps_dict = json.load(json_file)
            self.N = grasps_dict

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dict): data dictionary
        '''
        data_out = data.copy()
        points = data[None]
        normals = data['normals']

        if isinstance(self.N, int):
            indices = np.random.randint(points.shape[0], size=self.N)

        else:
            obj_name = data['obj_name']

            obj_grasps_dict = self.N[obj_name]

            total_grasps = len(obj_grasps_dict)
            n_grasps = torch.randint(5, total_grasps, (1,)).item()
            grasps = torch.ones(total_grasps).multinomial(n_grasps, replacement=False)

            indices = []
            for grasp in grasps:
                start, end = obj_grasps_dict[str(grasp.item())]
                indices += list(range(start, end))

            indices = np.random.randint(len(indices), size=3_000)  # To make all points the same size

        del data_out['obj_name']
        data_out[None] = points[indices, :]
        data_out['normals'] = normals[indices, :]

        return data_out


class SubsamplePoints(object):
    ''' Points subsampling transformation class.

    It subsamples the points data.

    Args:
        N (int): number of points to be subsampled
    '''
    def __init__(self, N):
        self.N = N

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        points = data[None]
        occ = data['occ']

        data_out = data.copy()
        if isinstance(self.N, int):
            idx = np.random.randint(points.shape[0], size=self.N)
            data_out.update({
                None: points[idx, :],
                'occ':  occ[idx],
            })
        else:
            # We will have 70% points or all of them if there are less than Nt_in
            Nt_out, Nt_in = self.N
            Nt = Nt_out + Nt_in # Output size

            occ_binary = (occ >= 0.5)
            points0 = points[~occ_binary]
            points1 = points[occ_binary]

            # If there are not enough ones, we have to fill with zeros
            Nt_in = min(Nt_in, points1.shape[0])
            Nt_out = Nt - Nt_in

            idx0 = torch.randint(points0.shape[0], (Nt_out,))
            unif = torch.ones(points1.shape[0])
            idx1 = unif.multinomial(Nt_in, replacement=False)

            points0 = points0[idx0, :]
            points1 = points1[idx1, :]
            points = np.concatenate([points0, points1], axis=0)

            occ0 = np.zeros(Nt_out, dtype=np.float32)
            occ1 = np.ones(Nt_in, dtype=np.float32)
            occ = np.concatenate([occ0, occ1], axis=0)

            volume = occ_binary.sum() / len(occ_binary)
            volume = volume.astype(np.float32)

            data_out.update({
                None: points,
                'occ': occ,
                'volume': volume,
            })
        return data_out


class ScaleAndRotatePoints(object):
    ''' Point scale and rotate transformation class

    It scales and rotates the points data.

    Args:
        scale (float): scale factor to be applied.
        rotatation [float, float, float]: angle to rotate the data (in degrees).
    '''

    def __init__(self, threshold=0.02):
        self.threshold = threshold
        self.rotation = (0, 0, 0)
        self.scale = 1.

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        r1 = 0.8
        r2 = 1.
        self.scale = (r1 - r2) * torch.rand(1) + r2
        self.rotation = torch.rand(3) * 2 * np.pi

        points = data[None]
        occ = data['occ']
        data_out = data.copy()

        # Pointcloud of random points
        pcd_rnd = o3d.geometry.PointCloud()
        pcd_rnd.points = o3d.utility.Vector3dVector(points)

        # Get object points from occupancy vector
        ind = np.nonzero(occ)[0]
        obj_pts = points[ind]

        # Create pointcloud of object
        pcd_obj = o3d.geometry.PointCloud()
        pcd_obj.points = o3d.utility.Vector3dVector(obj_pts)

        # Rotate object
        R = pcd_obj.get_rotation_matrix_from_xyz(self.rotation)
        pcd_obj.rotate(R, center=(0, 0, 0))
        # Scale object
        pcd_obj.scale(self.scale, center=(0, 0, 0))

        # Compute new occupancy vector with rotated and scaled object
        occ_out = np.where(np.asarray(pcd_rnd.compute_point_cloud_distance(pcd_obj)) < self.threshold, 1., 0.).astype(
            dtype='float32')

        data_out.update({
            # None: points,
            'occ': occ_out,
            'scale': self.scale.item(),
            'rotation': self.rotation.tolist()
        })

        return data_out


class ScaleAndRotatePointcloud(object):
    ''' Point scale and rotate transformation class

    It scales and rotates the points data.

    Args:
        scale (float): scale factor to be applied.
        rotatation [float, float, float]: angle to rotate the data (in degrees).
    '''

    def __init__(self, threshold=0.02):
        self.threshold = threshold

    def __call__(self, data, scale, rotation):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        points = data[None]
        data_out = data.copy()

        # Create pointcloud of input object
        pcd_obj = o3d.geometry.PointCloud()
        pcd_obj.points = o3d.utility.Vector3dVector(points)

        # Rotate object
        R = pcd_obj.get_rotation_matrix_from_xyz(rotation)
        pcd_obj.rotate(R, center=(0, 0, 0))
        # Scale object
        pcd_obj.scale(scale, center=(0, 0, 0))

        data_out.update({
            None: np.asarray(pcd_obj.points, dtype='float32'),
        })

        return data_out
