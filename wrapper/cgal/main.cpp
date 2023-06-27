//
// Created by xingyu on 5/29/23.
//

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>

#include "utils.h"
#include "intersection.h"

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
    std::string filename = "/data_HDD/zhuxingyu/.dataset/ModelNet40/bathtub/train/bathtub_0001.off";
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
    at::Tensor xyz = torch::from_blob(vs.data(), {V, 3}, at::device(at::kCPU).dtype(at::ScalarType::Float)).contiguous();
    at::Tensor faces = torch::from_blob(fs.data(), {F, 3}, at::device(at::kCPU).dtype(at::ScalarType::Int));

    at::Tensor face_pairs = self_intersections(xyz, faces);
    DEBUG(face_pairs[0]);
    std::vector<int> facelist;
    for (int i = 100; i < face_pairs.size(0); ++i) {
        auto f1 = face_pairs[i][0].item<int>();
        auto f2 = face_pairs[i][1].item<int>();
        if (f1 == f2) continue;
        DEBUG(f1, f2);
        facelist.push_back(faces[f1][0].item<int>());
        facelist.push_back(faces[f1][1].item<int>());
        facelist.push_back(faces[f1][2].item<int>());
        facelist.push_back(faces[f2][0].item<int>());
        facelist.push_back(faces[f2][1].item<int>());
        facelist.push_back(faces[f2][2].item<int>());
        break;
    }
    at::Tensor new_faces = torch::from_blob(facelist.data(), {2, 3}, at::device(at::kCPU).dtype(at::ScalarType::Int));
    save_mesh(xyz, new_faces, "/data_HDD/zhuxingyu/vscode/py3d/data/out.off");
    return 0;
}