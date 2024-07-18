#pragma once
#include <vector>
#include <map>
#include "lsqt.cuh"
#include "pytb.cuh"

class Run
{
public:
	Run();
	Pytb pytb;
    LSQT lsqt;
private:

	void perform_a_run_lsqt(
        int* NN_orbit,
        int* NL_orbit,
        int* type_number,
        int* atoms_type,
        double* Hr_onsite_val,
        double* Hr_hopping_val,
        double* Hi_hopping_val,
        double* xx,
        int* orbit_offset,
        int* orbit_offset2,
        int* orbit_atom_id,
        int num_of_atoms,
        int num_of_orbits,
        double max_energy,
        double start_energy,
        double end_energy,
        double volume,
        double time_step,
        int num_moments,
        int num_energies,
        int transport_direction,
        int num_of_steps
    );

	double cutoff;   // run time of entire simulation (fs)
    int num_of_steps;
};

