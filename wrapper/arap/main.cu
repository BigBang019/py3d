//
// Created by xingyu on 5/24/23.
//
#include "core/cuda_utils.h"
#include "core/utils.h"
#include "arap.h"


#include <iostream>

//typedef Eigen::MatrixXd<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::Dynamic> Matrix3f;

void save_mesh(at::Tensor xyz, at::Tensor faces, std::string filename) {
    std::ofstream file(filename);
    defer {file.close(); };
    file << "OFF\n";
    const int N = xyz.size(0);
    const int M = faces.size(0);

    file << N << " " << M << " " << 0 << "\n";
    for (int i = 0; i < N; ++i) {
        file << xyz[i][0].item<float>() << " " << xyz[i][1].item<float>() << " " << xyz[i][2].item<float>() << "\n";
    }
    for (int i = 0; i < M; ++i) {
        file << "3 " << faces[i][0].item<int>() << " " << faces[i][1].item<int>() << " " << faces[i][2].item<int>()<< "\n";
    }
}

int main() {
    std::string filename = "/data_HDD/zhuxingyu/vscode/py3d/data/decimated_knight.off";
//    std::string filename = "/data_HDD/zhuxingyu/.dataset/ModelNet40/door/test/door_0110.off";
    std::ifstream mesh_file(filename);
    defer {mesh_file.close(); };

//    mesh_file.ignore(1);
    std::string head;
    mesh_file >> head;
    int V, F, VN;
    mesh_file >> V >> F >> VN;
    std::cout << V << " " << F << " " << VN << std::endl;
    std::vector<float> vs(V * 3);
    std::vector<int> fs(F * 3);
    for (int i = 0; i < V; ++i) {
        float a, b, c;
        mesh_file >> a >> b >> c;
        vs[3*i] = a;
        vs[3*i+1] = b;
        vs[3*i+2] = c;
    }
    for (int i = 0; i < F; ++i) {
        int n, a, b, c;

        mesh_file >> n >> a >> b >> c;
        fs[3*i] = a;
        fs[3*i+1] = b;
        fs[3*i+2] = c;
    }
    std::cout << vs.size() << " " << fs.size() << std::endl;

    at::Tensor xyz = torch::from_blob(vs.data(), {1, V, 3}, at::device(at::kCPU).dtype(at::ScalarType::Float)).contiguous();
    at::Tensor faces = torch::from_blob(fs.data(), {1, F, 3}, at::device(at::kCPU).dtype(at::ScalarType::Int));
    at::Tensor move = torch::randn({1, 3}, at::device(at::kCPU).dtype(at::ScalarType::Float));
    at::Tensor handle = torch::randint(V, {1, 5}, at::device(at::kCPU).dtype(at::ScalarType::Int));

    auto max_values = std::get<0>(xyz.max(1, true));
    auto min_values = std::get<0>(xyz.min(1, true));
    auto ranges = max_values - min_values;
    xyz = (xyz - min_values) / ranges;

    move = move / move.norm(2, -1, true);
    DEBUG(move);

    at::Tensor new_xyz = arap_deform(xyz, faces, handle, move);
    DEBUG((new_xyz - xyz).sum());
    new_xyz = (new_xyz + min_values) * ranges;
    save_mesh(new_xyz[0], faces[0], "/data_HDD/zhuxingyu/vscode/py3d/data/out.off");
    DEBUG("heyehye");
    return 0;
}