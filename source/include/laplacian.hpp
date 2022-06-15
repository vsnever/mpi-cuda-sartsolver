
#pragma once

#include <vector>
#include <string>


class LaplacianMatrix {
    // sparse flattened Laplacian matrix
    size_t nvox;
    std::vector<size_t> index1d;
    std::vector<float> val;

public:

    inline size_t nvoxel() const {return nvox;}
    inline size_t size() const {return val.size();}

    const std::vector<float>& value() const;
    inline float value(size_t i) const {return val[i];}

    const std::vector<size_t>& index() const;
    inline size_t index(size_t i) const {return index1d[i];}

    float matrix(size_t i, size_t j) const;

    int read_hdf5(const std::string& filename);

    LaplacianMatrix(size_t nvoxel=0);

};
