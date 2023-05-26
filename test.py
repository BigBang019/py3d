import torch
import open3d as o3d
import pyvista
import numpy as np
import os

import py3d.smooth as smt

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

def test_implicit_laplacian_smoothing():
    path = "/data_HDD/zhuxingyu/vscode/pymesh/data/buddha.off"
    mesh = o3d.io.read_triangle_mesh(path)
    xyz = torch.Tensor(np.asarray(mesh.vertices)).float().unsqueeze(0).contiguous()
    face = torch.Tensor(np.asarray(mesh.triangles)).int().unsqueeze(0)
    print(xyz.shape, face.shape)
    new_xyz = smt.implicit_laplacian_smoothing(xyz, face, 10)
    print(new_xyz)
    write_mesh("/data_HDD/zhuxingyu/vscode/pymesh/data/out1.ply", new_xyz[0], face[0])
    visual_mesh(new_xyz[0], face[0])


if __name__ == "__main__":
    test_implicit_laplacian_smoothing()