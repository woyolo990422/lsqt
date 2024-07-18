
#pragma once
#include "gpu_vector.cuh"
#include "error.cuh"
#include "common.cuh"

class LSQT
{
public:

    void preprocess(
        double start_energy,
        double end_energy,
        double maximum_energy,
        double time_step,
        int number_of_moments,
        int number_of_orbitals,
        int number_of_energy_points
    );

    void process(
        int step,
        int* NN,
        int* NL,
        int number_of_energy_points,
        int number_of_orbitals,
        int number_of_moments,
        int direction,
        double maximum_energy,
        double* Hr_onsite_val,
        double* Hr_hopping_val,
        double* Hi_hopping_val,
        int number_of_atoms,
        double volume,
        double* xx,
        int* type_length,
        int* dev_orbit_offset,
        int* dev_orbit_offset2,
        int* dev_orbit_atom_id,
        int* dev_type_content
    );

    void find_dos_and_velocity(
        int* NN,
        int* NL,
        int number_of_energy_points,
        int number_of_orbitals,
        int number_of_moments,
        double maximum_energy,
        double* Hr_onsite_val,
        double* Hr_hopping_val,
        double* Hi_hopping_val,
        double* xx,
        int number_of_atoms,
        int* type_length,
        int* dev_orbit_offset,
        int* dev_orbit_offset2,
        int* dev_orbit_atom_id,
        int* dev_type_content
    );

    void find_sigma(
        int step,
        int number_of_moments,
        int number_of_energy_points,
        int number_of_orbitals,
        int number_of_atoms,
        int direction,
        double maximum_energy,
        int* NN,
        int* NL,
        double* Hr_onsite_val,
        double* Hr_hopping_val,
        double* Hi_hopping_val,
        double* xx,
        double volume,
        int* type_length,
        int* dev_orbit_offset,
        int* dev_orbit_offset2,
        int* dev_orbit_atom_id,
        int* dev_type_content);

    int transport_direction;

private:

    GPU_Vector<double> xx;
    GPU_Vector<double> slr;
    GPU_Vector<double> sli;
    GPU_Vector<double> srr;
    GPU_Vector<double> sri;
    GPU_Vector<double> scr;
    GPU_Vector<double> sci;

    std::vector<int> orbit_atom_id;
    std::vector<double> E;
    std::vector<double> sigma;
    double time_step;

};