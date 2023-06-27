import torch
import open3d as o3d
import pyvista
import numpy as np
import os

import py3d.smooth as smt
import py3d.arap as arap


def pc_normalize(pc):
    B, N, _ = pc.shape
    centroid = torch.mean(pc, dim=1).view(B, 1, 3)
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc**2, dim=-1)), dim=-1).values.view(B, 1, 1)
    pc = pc / m
    return pc, centroid, m

def to_numpy(a):
    if isinstance(a, torch.Tensor):
        return a.cpu().detach().numpy()
    return a

def regulate_face(face):
    """
        face: (N, 3)
    """
    for i, f in enumerate(face):
        if f[0]==-1:
            face = face[:i]
            break
    return face

def read_mesh(path):
    _, ext = os.path.splitext(path)
    if ext == ".off":
        mesh = o3d.io.read_triangle_mesh(path)
        xyz = torch.Tensor(np.asarray(mesh.vertices)).cuda().float().unsqueeze(0).contiguous()
        face = torch.Tensor(np.asarray(mesh.triangles)).cuda().int().unsqueeze(0)
    else:
        mesh = pyvista.PolyData(path)
        xyz = torch.Tensor(np.asarray(mesh.points)).cuda().float().unsqueeze(0).contiguous()
        face = torch.Tensor(np.asarray(mesh.faces)).cuda().int()
        face = face.reshape(-1, 4)[:, 1:].unsqueeze(0).contiguous()
    return xyz, face

def to_vista_face(face):
    M, D = face.shape
    if D==4:
        return face
    else:
        return np.concatenate([np.full((M, 1), 3), face], axis=-1)
    
def write_mesh(path, pc, face):
    pc, face = to_numpy(pc), to_numpy(face)
    face = regulate_face(face)
    if path[-3:] == "off":
        pass
    else:
        face = to_vista_face(face)
        mesh = pyvista.PolyData(pc, face)
        mesh.save(path)

def visual_mesh(pc, face, idx=None):
    '''
        pc: (N, 3) Tensor or ndarray
        faces: (N, 3)
        idx: (N, nsample)
    '''
    pc, face = to_numpy(pc), to_numpy(face)
    face = regulate_face(face)
    face = to_vista_face(face)
    mesh = pyvista.PolyData(pc, face)
    ploter_args = dict(window_size=[1024, 768])
    p = pyvista.Plotter(**ploter_args)
    p.add_mesh(mesh, show_edges=False)
    p.show(screenshot=os.path.join("vis.png"))

def visual_path(path):
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])



def test(path):
    # path = "/data_HDD/zhuxingyu/vscode/py3d/data/buddha.off"
    mesh = o3d.io.read_triangle_mesh(path)
    xyz = torch.Tensor(np.asarray(mesh.vertices)).float().unsqueeze(0).contiguous()
    face = torch.Tensor(np.asarray(mesh.triangles)).int().unsqueeze(0)
    xyz, _, _ = pc_normalize(xyz)

    print(xyz.shape, face.shape)
    # new_xyz = smt.implicit_laplacian_smoothing(xyz, face, 10)
    handles = torch.randint(0, xyz.shape[1], (1, 1)).int()
    moves = torch.randn((1, 3))
    moves = (moves / moves.norm(p=2, dim=-1))
    print(moves)
    # new_xyz = arap.arap_deform(xyz, face, handles, moves)

    mesh = o3d.geometry.TriangleMesh(
        vertices = o3d.utility.Vector3dVector(np.asarray(xyz[0])),
        triangles = o3d.utility.Vector3iVector(np.asarray(face[0])),
    )
    handle_pos = []
    for handle in handle_pos:
        handle_pos.append(xyz[0][handle])
    mesh.deform_as_rigid_as_possible(
        o3d.utility.IntVector(handles[0].numpy()),
        o3d.utility.Vector3dVector(handle_pos),
        max_iter=20
    )
    print(mesh)
    # print(new_xyz)
    # write_mesh("/data_HDD/zhuxingyu/vscode/py3d/data/out.off", new_xyz[0], face[0])
    # visual_mesh(new_xyz[0], face[0])


if __name__ == "__main__":
    # test_implicit_laplacian_smoothing()
    path = "/data_HDD/zhuxingyu/vscode/py3d/dat" \
           "a/out.off"
    # path = "/data_HDD/zhuxingyu/vscode/py3d/data/decimated_knight.off"
    # path = "/data_HDD/zhuxingyu/.dataset/modelnet40_processed2500/train/airplane/airplane_0001.off"
    # path = "/data_HDD/zhuxingyu/.dataset/modelnet40_processed500/train/airplane/airplane_0005.obj"
    # path = "/data_HDD/zhuxingyu/.dataset/ModelNet40/door/test/door_0110.off"
    # test(path)
    visual_path(path)

    # path1 = "/data_HDD/zhuxingyu/vscode/py3d/data/decimated_knight.off"
    # xyz, _ = read_mesh(path)
    # xyz, _, _ = pc_normalize(xyz)
    # xyz1, _ = read_mesh(path1)
    # xyz1, _, _ = pc_normalize(xyz1)
    #
    # print((xyz - xyz1).sum())