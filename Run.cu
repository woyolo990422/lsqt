#include "error.cuh"
#include "Run.cuh"
#include "gpu_vector.cuh"
Run::Run()
{
    print_line_1();
    printf("Started initializing real space hamiltonian matrix.\n");
    fflush(stdout);
    print_line_2();

    pytb.pytb_init();

    print_line_1();
    printf("Finished initializing real space hamiltonian matrix.\n");
    fflush(stdout);
    print_line_2();



    //pytb.dev_NN.data(),
//pytb.dev_NL.data(),
    perform_a_run_lsqt(
        pytb.dev_NN_orbit.data(),
        pytb.dev_NL_orbit.data(),
        pytb.dev_orbit_size.data(),
        pytb.dev_atoms_type.data(),
        pytb.dev_Hr_onsite_val.data(),
        pytb.dev_Hr_hopping_val.data(),
        pytb.dev_Hi_hopping_val.data(),
        pytb.dev_xx.data(),
        pytb.dev_orbit_offset.data(),
        pytb.dev_orbit_offset2.data(),
        pytb.dev_orbit_atom_id.data(),
        pytb.num_of_atoms,
        pytb.num_of_orbits,
        pytb.max_energy,
        pytb.start_energy,
        pytb.end_energy,
        pytb.volume,
        pytb.time_step,
        pytb.num_moments,
        pytb.num_energies,
        pytb.transport_direction,
        pytb.num_of_steps
    );




}


void Run::perform_a_run_lsqt(
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
)
{



    printf("LSQT Run %d steps.\n", num_of_steps);
    


    lsqt.preprocess(
        start_energy,
        end_energy,
        max_energy,
        time_step,
        num_moments,
        num_of_orbits,
        num_energies        
    );


    clock_t time_begin = clock();
    
    for (int step = 0; step < num_of_steps; ++step) {
        
        lsqt.process(
            step,
            NN_orbit,
            NL_orbit,
            num_energies,
            num_of_orbits,
            num_moments,
            transport_direction,
            max_energy,
            Hr_onsite_val,
            Hr_hopping_val,
            Hi_hopping_val,
            num_of_atoms,
            volume,
            xx,
            type_number,
            orbit_offset,
            orbit_offset2,
            orbit_atom_id,
            atoms_type);
            

        int base = (10 <= num_of_steps) ? (num_of_steps / 10) : 1;
        if (0 == (step + 1) % base) {
            printf("    %d steps completed.\n", step + 1);
            fflush(stdout);
        }
    }
    
    print_line_1();
    clock_t time_finish = clock();
    double time_used = (time_finish - time_begin) / (double)CLOCKS_PER_SEC;

    printf("Time used for this run = %g second.\n", time_used);
    double run_speed = num_of_atoms * (num_of_steps / time_used);
    printf("Speed of this run = %g atom*step/second.\n", run_speed);
    print_line_2();

}

