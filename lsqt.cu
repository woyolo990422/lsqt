
#include "lsqt.cuh"
#include <vector>
#include <algorithm>
#include <cmath>


namespace
{
#define BLOCK_SIZE_EC 512 // do not change this

    // copy state: so = si
    __global__ void gpu_copy_state(int N, double* sir, double* sii, double* sor, double* soi)
    {
        int n = blockIdx.x * blockDim.x + threadIdx.x;
        if (n < N) {
            sor[n] = sir[n];
            soi[n] = sii[n];
        }
    }

    // will be used for U(t)
    __global__ void gpu_chebyshev_01(
        int N,
        double* s0r,
        double* s0i,
        double* s1r,
        double* s1i,
        double* sr,
        double* si,
        double b0,
        double b1,
        int direction)
    {
        int n = blockIdx.x * blockDim.x + threadIdx.x;
        if (n < N) {
            double bessel_0 = b0;
            double bessel_1 = b1 * direction;
            sr[n] = bessel_0 * s0r[n] + bessel_1 * s1i[n];
            si[n] = bessel_0 * s0i[n] - bessel_1 * s1r[n];
        }
    }

    // will be used for U(t)
    __global__ void gpu_chebyshev_2(
        int N,
        double Em_inv,
        int* NN,
        int* NL,
        double* Hr_onsite_val,
        double* Hr_hopping_val,
        double* Hi_hopping_val,
        double* s0r,
        double* s0i,
        double* s1r,
        double* s1i,
        double* s2r,
        double* s2i,
        double* sr,
        double* si,
        double bessel_m,
        int label,
        int* type_length,
        int number_of_atoms,
        int* orbitals_offset,
        int* orbitals_offset2,
        int* orbitals_atom_id,
        int* type_content
    )
    {
        int n = blockIdx.x * blockDim.x + threadIdx.x;
        if (n < N) {
            double temp_real = 0;
            double temp_imag = 0;
            // add for the deform matrix because the physical mean of strain

            int atom_id = orbitals_atom_id[n];

            int orbit_id = n - orbitals_offset[atom_id];

            int type_n = type_content[atom_id];


            //int onsite_index = orbit_id * (type_length[type_n] + 1) + orbitals_offset2[atom_id];
            //temp_real = Hr_onsite_val[onsite_index] * s1r[n]; // on-site
            //temp_imag = Hr_onsite_val[onsite_index] * s1i[n]; // on-site

            //ergodic itself orbits
            for (int io = 0; io < type_length[type_n]; ++io) {
                int onsite_index = orbit_id + io * type_length[type_n] + orbitals_offset2[atom_id];
                temp_real += Hr_onsite_val[onsite_index] * s1r[n]; // on-site
                temp_imag += Hr_onsite_val[onsite_index] * s1i[n]; // on-site
            }


            int neighbor_number = NN[n];
            for (int m = 0; m < neighbor_number; ++m) {
                int index_1 = m * N + n;
                int index_2 = NL[index_1];
                double a = Hr_hopping_val[index_1];
                double b = Hi_hopping_val[index_1];
                double c = s1r[index_2];
                double d = s1i[index_2];
                temp_real += a * c - b * d; // hopping
                temp_imag += a * d + b * c; // hopping
            }
            temp_real *= Em_inv; // scale
            temp_imag *= Em_inv; // scale

            temp_real = 2.0 * temp_real - s0r[n];
            temp_imag = 2.0 * temp_imag - s0i[n];
            switch (label) {
            case 1: {
                sr[n] += bessel_m * temp_real;
                si[n] += bessel_m * temp_imag;
                break;
            }
            case 2: {
                sr[n] -= bessel_m * temp_real;
                si[n] -= bessel_m * temp_imag;
                break;
            }
            case 3: {
                sr[n] += bessel_m * temp_imag;
                si[n] -= bessel_m * temp_real;
                break;
            }
            case 4: {
                sr[n] -= bessel_m * temp_imag;
                si[n] += bessel_m * temp_real;
                break;
            }
            }
            s2r[n] = temp_real;
            s2i[n] = temp_imag;
        }
    }

    // for KPM
    __global__ void gpu_kernel_polynomial(
        int N,
        double Em_inv,
        int* NN,
        int* NL,
        double* Hr_onsite_val,
        double* Hr_hopping_val,
        double* Hi_hopping_val,
        double* s0r,
        double* s0i,
        double* s1r,
        double* s1i,
        double* s2r,
        double* s2i,
        int* type_length,
        int number_of_atoms,
        int* orbitals_offset,
        int* orbitals_offset2,
        int* dev_orbit_atom_id,
        int* type_content
    )
    {
        int n = blockIdx.x * blockDim.x + threadIdx.x;

        if (n < N) {

            double temp_real = 0;
            double temp_imag = 0;
            // add for the deform matrix because the physical mean of strain

            int atom_id = dev_orbit_atom_id[n];

            int orbit_id = n - orbitals_offset[atom_id];

            int type_n = type_content[atom_id];


            //int onsite_index = orbit_id * (type_length[type_n]+1) + orbitals_offset2[atom_id];
            //temp_real = Hr_onsite_val[onsite_index] * s1r[n]; // on-site
            //temp_imag = Hr_onsite_val[onsite_index] * s1i[n]; // on-site

            

            //ergodic itself orbits
            for (int io = 0; io < type_length[type_n]; ++io) {
                int onsite_index = orbit_id +io* type_length[type_n]+orbitals_offset2[atom_id];
                temp_real += Hr_onsite_val[onsite_index]  * s1r[n]; // on-site
                temp_imag += Hr_onsite_val[onsite_index]  * s1i[n]; // on-site
            }

            int neighbor_number = NN[n];

            for (int m = 0; m < neighbor_number; ++m) {

                int index_1 = m * N + n;
                int index_2 = NL[index_1];
                double a = Hr_hopping_val[index_1];
                double b = Hi_hopping_val[index_1];
                double c = s1r[index_2];
                double d = s1i[index_2];
                temp_real += a * c - b * d; // hopping
                temp_imag += a * d + b * c; // hopping
                //printf("%d,%d,%d,%d\n", n,m,index_1, index_2);
            }

            temp_real *= Em_inv; // scale
            temp_imag *= Em_inv; // scale

            temp_real = 2.0 * temp_real - s0r[n];
            temp_imag = 2.0 * temp_imag - s0i[n];
 
            s2r[n] = temp_real;
            s2i[n] = temp_imag;

        }
    }

    // apply the Hamiltonian: H * si = so
    __global__ void gpu_apply_hamiltonian(
        int N,
        double Em_inv,
        int* NN,
        int* NL,
        double* Hr_onsite_val,
        double* Hr_hopping_val,
        double* Hi_hopping_val,
        double* sir,
        double* sii,
        double* sor,
        double* soi,
        int* type_length,
        int number_of_atoms,
        int* dev_orbitals_offset,
        int* dev_orbitals_offset2,
        int* dev_orbit_atom_id,
        int* dev_type_content)
    {
        int n = blockIdx.x * blockDim.x + threadIdx.x;

        if (n < N) {

            double temp_real = 0;
            double temp_imag = 0;

            // add for the deform matrix because the physical mean of strain

            int atom_id = dev_orbit_atom_id[n];

            int orbit_id = n - dev_orbitals_offset[atom_id];

            int type_n = dev_type_content[atom_id];

            //int onsite_index = orbit_id * (type_length[type_n] + 1) + dev_orbitals_offset2[atom_id];
            //temp_real = Hr_onsite_val[onsite_index] * sir[n]; // on-site
            //temp_imag = Hr_onsite_val[onsite_index] * sii[n]; // on-site

            //ergodic itself orbits
            for (int io = 0; io < type_length[type_n]; ++io) {
                int onsite_index = orbit_id +io* type_length[type_n]+dev_orbitals_offset2[atom_id];
                temp_real += Hr_onsite_val[onsite_index]  * sir[n]; // on-site
                temp_imag += Hr_onsite_val[onsite_index]  * sii[n]; // on-site
            }

            int neighbor_number = NN[n];

            for (int m = 0; m < neighbor_number; ++m) {
                int index_1 = m * N + n;
                int index_2 = NL[index_1];
                double a = Hr_hopping_val[index_1];
                double b = Hi_hopping_val[index_1];

                double c = sir[index_2];
                double d = sii[index_2];
                temp_real += a * c - b * d; // hopping
                temp_imag += a * d + b * c; // hopping
                //printf("see:%d,%d,%d,%d,%f\n", n, m, index_1, index_2,a);
            }

            temp_real *= Em_inv; // scale
            temp_imag *= Em_inv; // scale
         
            sor[n] = temp_real;
            soi[n] = temp_imag;

        }
    }

    // so = V * si (no scaling; no on-site)
    __global__ void gpu_apply_current(
        int N,
        int* NN,
        int* NL,
        double* Hr_hopping_val,
        double* Hi_hopping_val,
        double* g_xx,
        double* sir,
        double* sii,
        double* sor,
        double* soi)
    {
        int n = blockIdx.x * blockDim.x + threadIdx.x;

        if (n < N) {
            
            double temp_real = 0.0;
            double temp_imag = 0.0;
            int neighbor_number = NN[n];

            for (int m = 0; m < neighbor_number; ++m) {
                int index_1 = m * N + n;
                int index_2 = NL[index_1];
                double a = Hr_hopping_val[index_1];
                double b = Hi_hopping_val[index_1];
                double c = sir[index_2];
                double d = sii[index_2];
                double xx = g_xx[index_1];
                temp_real += (a * c - b * d) * xx;
                temp_imag += (a * d + b * c) * xx;

            }
            sor[n] = +temp_imag;
            soi[n] = -temp_real;
        }
    }
    // 1st step of <sl|sr>
    static __global__ void gpu_find_inner_product_1(
        int N, double* srr, double* sri, double* slr, double* sli, double* moments, int offset)
    {
        int tid = threadIdx.x;
        int n = blockIdx.x * blockDim.x + tid;
        __shared__ double s_data[BLOCK_SIZE_EC];
        s_data[tid] = 0.0;
        if (n < N) {
            s_data[tid] = (srr[n] * slr[n] + sri[n] * sli[n]);
            
        }
        __syncthreads();
        for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
            if (tid < offset) {
                s_data[tid] += s_data[tid + offset];
                
            }
            __syncthreads();
        }
        if (tid == 0) {
            moments[blockIdx.x + offset] = s_data[0];
        }
    }

    // 2nd step of <sl|sr>
    __global__ void gpu_find_inner_product_2(
        int number_of_blocks, int number_of_patches, double* moments_tmp, double* moments)
    {
        int tid = threadIdx.x;
        __shared__ double s_data[BLOCK_SIZE_EC];
        s_data[tid] = 0.0;
        for (int patch = 0; patch < number_of_patches; ++patch) {
            int n = tid + patch * BLOCK_SIZE_EC;
            if (n < number_of_blocks) {
                s_data[tid] += moments_tmp[blockIdx.x * number_of_blocks + n];
            }
        }
        __syncthreads();
        for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
            if (tid < offset) {
                s_data[tid] += s_data[tid + offset];
            }
            __syncthreads();
        }
        if (tid == 0)
            moments[blockIdx.x] = s_data[0];
    }


    // get the Chebyshev moments: <sl|T_m(H)|sr>
    void find_moments_chebyshev(
        int N,
        int Nm,
        double Em,
        int* NN,
        int* NL,
        double* Hr_onsite_val,
        double* Hr_hopping_val,
        double* Hi_hopping_val,
        double* slr,
        double* sli,
        double* srr,
        double* sri,
        double* moments,
        int* type_length,
        int number_of_atoms,
        int* dev_orbit_offset,
        int* dev_orbit_offset2,
        int* dev_orbit_atom_id,
        int* dev_type_content
    )
    {

        int grid_size = (N - 1) / BLOCK_SIZE_EC + 1;
        int number_of_blocks = grid_size;
        int number_of_patches = (number_of_blocks - 1) / BLOCK_SIZE_EC + 1;

        int memory_moments = sizeof(double) * Nm;
        int memory_moments_tmp = memory_moments * grid_size;
        double Em_inv = 1.0 / Em;

        double* s0r, * s1r, * s2r, * s0i, * s1i, * s2i, * moments_tmp;
        cudaMalloc((void**)&s0r, sizeof(double) * N);
        cudaMalloc((void**)&s1r, sizeof(double) * N);
        cudaMalloc((void**)&s2r, sizeof(double) * N);
        cudaMalloc((void**)&s0i, sizeof(double) * N);
        cudaMalloc((void**)&s1i, sizeof(double) * N);
        cudaMalloc((void**)&s2i, sizeof(double) * N);
        cudaMalloc((void**)&moments_tmp, memory_moments_tmp);

        gpu_copy_state << <grid_size, BLOCK_SIZE_EC >> > (N, srr, sri, s0r, s0i);
        CUDA_CHECK_KERNEL

        gpu_find_inner_product_1 << <grid_size, BLOCK_SIZE_EC >> > (
                N, s0r, s0i, slr, sli, moments_tmp, 0 * grid_size);
        CUDA_CHECK_KERNEL

            // T_1(H)
        gpu_apply_hamiltonian << <grid_size, BLOCK_SIZE_EC >> > (
            N, Em_inv, NN, NL,
            Hr_onsite_val,
            Hr_hopping_val,
            Hi_hopping_val,
            s0r, s0i, s1r, s1i,
            type_length,
            number_of_atoms,
            dev_orbit_offset,
            dev_orbit_offset2,
            dev_orbit_atom_id,
            dev_type_content);
        CUDA_CHECK_KERNEL

            gpu_find_inner_product_1 << <grid_size, BLOCK_SIZE_EC >> > (
                N, s1r, s1i, slr, sli, moments_tmp, 1 * grid_size);
        CUDA_CHECK_KERNEL
          
            // T_m(H) (m >= 2)
            for (int m = 2; m < Nm; ++m) {

                gpu_kernel_polynomial << <grid_size, BLOCK_SIZE_EC >> > (
                    N, Em_inv, NN, NL,
                    Hr_onsite_val,
                    Hr_hopping_val,
                    Hi_hopping_val,
                    s0r, s0i, s1r, s1i, s2r, s2i,
                    type_length,
                    number_of_atoms,
                    dev_orbit_offset,
                    dev_orbit_offset2,
                    dev_orbit_atom_id,
                    dev_type_content);

                CUDA_CHECK_KERNEL

                    gpu_find_inner_product_1 << <grid_size, BLOCK_SIZE_EC >> > (
                        N, s2r, s2i, slr, sli, moments_tmp, m * grid_size);
                CUDA_CHECK_KERNEL


                    // permute the pointers; do not need to copy the data
                double* temp_real;
                double* temp_imag;
                temp_real = s0r;
                temp_imag = s0i;
                s0r = s1r;
                s0i = s1i;
                s1r = s2r;
                s1i = s2i;
                s2r = temp_real;
                s2i = temp_imag;
            }


        gpu_find_inner_product_2 << <Nm, BLOCK_SIZE_EC >> > (
            number_of_blocks, number_of_patches, moments_tmp, moments);
        CUDA_CHECK_KERNEL

        cudaFree(s0r);
        cudaFree(s0i);
        cudaFree(s1r);
        cudaFree(s1i);
        cudaFree(s2r);
        cudaFree(s2i);
        cudaFree(moments_tmp);
    }

    // Jackson damping
    void apply_damping(int Nm, double* moments)
    {
        for (int k = 0; k < Nm; ++k) {
            double a = 1.0 / (Nm + 1.0);
            double damping = (1.0 - k * a) * cos(k * PI * a) + sin(k * PI * a) * a / tan(PI * a);
            moments[k] *= damping;
        }
    }

    // kernel polynomial summation
    void perform_chebyshev_summation(
        int Nm, int Ne, double Em, double* E, double* moments, double* correlation)
    {
        for (int step1 = 0; step1 < Ne; ++step1) {
            double energy_scaled = E[step1] / Em;
            double chebyshev_0 = 1.0;
            double chebyshev_1 = energy_scaled;
            double chebyshev_2;
            double temp = moments[1] * chebyshev_1;
            for (int step2 = 2; step2 < Nm; ++step2) {
                chebyshev_2 = 2.0 * energy_scaled * chebyshev_1 - chebyshev_0;
                chebyshev_0 = chebyshev_1;
                chebyshev_1 = chebyshev_2;
                temp += moments[step2] * chebyshev_2;
            }
            temp *= 2.0;
            temp += moments[0];
            temp *= 2.0 / (PI * sqrt(1.0 - energy_scaled * energy_scaled));
            correlation[step1] = temp / Em;
        }
    }

    // direction = +1: U(+t) |state>
    // direction = -1: U(-t) |state>
    void evolve(
        int N,
        double Em,
        int direction,
        double time_step_scaled,
        int* NN,
        int* NL,
        double* Hr_onsite_val,
        double* Hr_hopping_val,
        double* Hi_hopping_val,
        double* sr,
        double* si,
        int* type_length,
        int number_of_atoms,
        int* dev_orbit_offset,
        int* dev_orbit_offset2,
        int* dev_orbit_atom_id,
        int* dev_type_content
    )
    {
        int grid_size = (N - 1) / BLOCK_SIZE_EC + 1;
        double Em_inv = 1.0 / Em;
        double* s0r;
        double* s1r;
        double* s2r;
        double* s0i;
        double* s1i;
        double* s2i;
        cudaMalloc((void**)&s0r, sizeof(double) * N);
        cudaMalloc((void**)&s0i, sizeof(double) * N);
        cudaMalloc((void**)&s1r, sizeof(double) * N);
        cudaMalloc((void**)&s1i, sizeof(double) * N);
        cudaMalloc((void**)&s2r, sizeof(double) * N);
        cudaMalloc((void**)&s2i, sizeof(double) * N);


        // T_0(H) |psi> = |psi>
        gpu_copy_state << <grid_size, BLOCK_SIZE_EC >> > (N, sr, si, s0r, s0i);
        CUDA_CHECK_KERNEL


            // T_1(H) |psi> = H |psi>
            gpu_apply_hamiltonian << <grid_size, BLOCK_SIZE_EC >> > (
                N, Em_inv, NN, NL,
                Hr_onsite_val,
                Hr_hopping_val,
                Hi_hopping_val,
                sr, si, s1r, s1i,
                type_length,
                number_of_atoms,
                dev_orbit_offset,
                dev_orbit_offset2,
                dev_orbit_atom_id,
                dev_type_content);
        CUDA_CHECK_KERNEL

            // |final_state> = c_0 * T_0(H) |psi> + c_1 * T_1(H) |psi>
            double bessel_0 = j0(time_step_scaled);
        double bessel_1 = 2.0 * j1(time_step_scaled);
        gpu_chebyshev_01 << <grid_size, BLOCK_SIZE_EC >> > (
            N, s0r, s0i, s1r, s1i, sr, si, bessel_0, bessel_1, direction);
        CUDA_CHECK_KERNEL

            for (int m = 2; m < 1000000; ++m) {
                double bessel_m = jn(m, time_step_scaled);
                if (bessel_m < 1.0e-15 && bessel_m > -1.0e-15) {
                    break;
                }
                bessel_m *= 2.0;
                int label;
                int m_mod_4 = m % 4;
                if (m_mod_4 == 0) {
                    label = 1;
                }
                else if (m_mod_4 == 2) {
                    label = 2;
                }
                else if ((m_mod_4 == 1 && direction == 1) || (m_mod_4 == 3 && direction == -1)) {
                    label = 3;
                }
                else {
                    label = 4;
                }
                gpu_chebyshev_2 << <grid_size, BLOCK_SIZE_EC >> > (
                    N, Em_inv, NN, NL,
                    Hr_onsite_val,
                    Hr_hopping_val,
                    Hi_hopping_val,
                    s0r, s0i, s1r, s1i, s2r, s2i, sr, si, bessel_m, label,
                    type_length,
                    number_of_atoms,
                    dev_orbit_offset,
                    dev_orbit_offset2,
                    dev_orbit_atom_id,
                    dev_type_content
                    );
                CUDA_CHECK_KERNEL

                    // permute the pointers; do not need to copy the data
                    double* temp_real, * temp_imag;
                temp_real = s0r;
                temp_imag = s0i;
                s0r = s1r;
                s0i = s1i;
                s1r = s2r;
                s1i = s2i;
                s2r = temp_real;
                s2i = temp_imag;
            }
        cudaFree(s0r);
        cudaFree(s0i);
        cudaFree(s1r);
        cudaFree(s1i);
        cudaFree(s2r);
        cudaFree(s2i);
    }



    void find_dos_or_others(
        int N,
        int Nm,
        int Ne,
        double Em,
        double* E,
        int* NN,
        int* NL,
        double* Hr_onsite_val,
        double* Hr_hopping_val,
        double* Hi_hopping_val,
        int number_of_atoms,
        double* slr,
        double* sli,
        double* srr,
        double* sri,
        double* dos_or_others,
        int* type_length,
        int* dev_orbit_offset,
        int* dev_orbit_offset2,
        int* dev_orbit_atom_id,
        int* dev_type_content)
    {
        std::vector<double> moments_cpu(Nm);
        GPU_Vector<double> moments_gpu(Nm);

        find_moments_chebyshev(
            N, Nm, Em, NN, NL, 
            Hr_onsite_val, Hr_hopping_val, Hi_hopping_val,
            slr, sli, srr, sri, 
            moments_gpu.data(),
            type_length,
            number_of_atoms, 
            dev_orbit_offset, 
            dev_orbit_offset2,
            dev_orbit_atom_id, 
            dev_type_content);

        moments_gpu.copy_to_host(moments_cpu.data());
        apply_damping(Nm, moments_cpu.data());
        perform_chebyshev_summation(Nm, Ne, Em, E, moments_cpu.data(), dos_or_others);
    }

    void initialize_state(int N, GPU_Vector<double>& sr, GPU_Vector<double>& si)
    {
        std::vector<double> sr_cpu(N);
        std::vector<double> si_cpu(N);
        for (int n = 0; n < N; ++n) {
            double random_phase = rand() / double(RAND_MAX) * 2.0 * PI;
            sr_cpu[n] = cos(random_phase);
            si_cpu[n] = sin(random_phase);
        }
        sr.copy_from_host(sr_cpu.data());
        si.copy_from_host(si_cpu.data());
    }
}// namespace

void LSQT::preprocess(
    double start_energy,
    double end_energy,
    double maximum_energy,
    double time_step,
    int number_of_moments,
    int number_of_orbitals,
    int number_of_energy_points
)
{

    printf("Compute LSQT.\n");
    printf("    number of moments is %d.\n", number_of_moments);
    printf("    number of energy points is %d.\n", number_of_energy_points);
    printf("    starting energy is %g eV.\n", start_energy);
    printf("    ending energy is %g eV.\n", end_energy);
    printf("    maximum energy is %g eV.\n", maximum_energy);

    E.resize(number_of_energy_points);

    double delta_energy = (end_energy - start_energy) / (number_of_energy_points - 1);
    for (int n = 0; n < number_of_energy_points; ++n) {
        E[n] = start_energy + n * delta_energy;
    }

    time_step /= TIME_UNIT_CONVERSION;

    this->time_step = time_step * 15.46692; // from GPUMD unit to hbar/eV
    

    sigma.resize(number_of_energy_points);

    slr.resize(number_of_orbitals);
    sli.resize(number_of_orbitals);
    srr.resize(number_of_orbitals);
    sri.resize(number_of_orbitals);
    scr.resize(number_of_orbitals);
    sci.resize(number_of_orbitals);




}

void LSQT::process(
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
)
{

    find_dos_and_velocity(
        NN,
        NL,
        number_of_energy_points,
        number_of_orbitals,
        number_of_moments,
        maximum_energy,
        Hr_onsite_val,
        Hr_hopping_val,
        Hi_hopping_val,
        xx,
        number_of_atoms,
        type_length,
        dev_orbit_offset,
        dev_orbit_offset2,
        dev_orbit_atom_id,
        dev_type_content
    );


    find_sigma(
        step,
        number_of_moments,
        number_of_energy_points,
        number_of_orbitals,
        number_of_atoms,
        direction,
        maximum_energy,
        NN,
        NL,
        Hr_onsite_val,
        Hr_hopping_val,
        Hi_hopping_val,
        xx,
        volume,
        type_length,
        dev_orbit_offset,
        dev_orbit_offset2,
        dev_orbit_atom_id,
        dev_type_content);
}

void LSQT::find_dos_and_velocity(
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
)
{
    std::vector<double> dos(number_of_energy_points);
    std::vector<double> velocity(number_of_energy_points);

    GPU_Vector<double> sr(number_of_orbitals);
    GPU_Vector<double> si(number_of_orbitals);
    GPU_Vector<double> sxr(number_of_orbitals);
    GPU_Vector<double> sxi(number_of_orbitals);

    initialize_state(number_of_orbitals, sr, si);

    // dos

    find_dos_or_others(
        number_of_orbitals,
        number_of_moments,
        number_of_energy_points,
        maximum_energy,
        E.data(),
        NN,
        NL,
        Hr_onsite_val,
        Hr_hopping_val,
        Hi_hopping_val,
        number_of_atoms,
        sr.data(),
        si.data(),
        sr.data(),
        si.data(),
        dos.data(),
        type_length,
        dev_orbit_offset, dev_orbit_offset2, dev_orbit_atom_id, dev_type_content);


    FILE* os_dos = my_fopen("lsqt_dos.out", "a");

    //std::reverse(dos.begin(), dos.end());

    for (int n = 0; n < number_of_energy_points; ++n)
        fprintf(os_dos, "%25.15e", dos[n] / number_of_atoms); // state/eV/atom
    fprintf(os_dos, "\n");
    fclose(os_dos);

    int grid_size = (number_of_orbitals - 1) / BLOCK_SIZE_EC + 1;

    // velocity
    gpu_apply_current << <grid_size, BLOCK_SIZE_EC >> > (
        number_of_orbitals,
        NN,
        NL,
        Hr_hopping_val,
        Hi_hopping_val,
        xx,
        sr.data(),
        si.data(),
        sxr.data(),
        sxi.data());

    find_dos_or_others(
        number_of_orbitals,
        number_of_moments,
        number_of_energy_points,
        maximum_energy,
        E.data(),
        NN,
        NL,
        Hr_onsite_val,
        Hr_hopping_val,
        Hi_hopping_val,
        number_of_atoms,
        sxr.data(),
        sxi.data(),
        sxr.data(),
        sxi.data(),
        velocity.data(),
        type_length,
        dev_orbit_offset, dev_orbit_offset2, dev_orbit_atom_id, dev_type_content);

    FILE* os_vel = my_fopen("lsqt_velocity.out", "a");
    const double m_per_s_conversion = 1.60217663e5 / 1.054571817;

    std::reverse(velocity.begin(), velocity.end());

    for (int n = 0; n < number_of_energy_points; ++n)
        fprintf(os_vel, "%25.15e", sqrt(velocity[n] / dos[n]) * m_per_s_conversion);
    fprintf(os_vel, "\n");
    fclose(os_vel);
}

void LSQT::find_sigma(
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
    int* dev_type_content)
{
    double time_step_scaled = time_step * maximum_energy;
    double V = volume;
    int grid_size = (number_of_orbitals - 1) / BLOCK_SIZE_EC + 1;
    if (step == 0) {
        initialize_state(number_of_orbitals, slr, sli);

        
        // velocity
        gpu_apply_current << <grid_size, BLOCK_SIZE_EC >> > (
            number_of_orbitals,
            NN,
            NL,
            Hr_hopping_val,
            Hi_hopping_val,
            xx,
            slr.data(),
            sli.data(),
            srr.data(),
            sri.data());
        CUDA_CHECK_KERNEL
    }
    else {
        evolve(
            number_of_orbitals,
            maximum_energy,
            direction,
            time_step_scaled,
            NN,
            NL,
            Hr_onsite_val,
            Hr_hopping_val,
            Hi_hopping_val,
            slr.data(),
            sli.data(),
            type_length,
            number_of_atoms,
            dev_orbit_offset,
            dev_orbit_offset2,
            dev_orbit_atom_id,
            dev_type_content);

        evolve(
            number_of_orbitals,
            maximum_energy,
            direction,
            time_step_scaled,
            NN,
            NL,
            Hr_onsite_val,
            Hr_hopping_val,
            Hi_hopping_val,
            srr.data(),
            sri.data(),
            type_length,
            number_of_atoms,
            dev_orbit_offset,
            dev_orbit_offset2,
            dev_orbit_atom_id,
            dev_type_content);
    }

    // velocity
    gpu_apply_current << <grid_size, BLOCK_SIZE_EC >> > (
        number_of_orbitals,
        NN,
        NL,
        Hr_hopping_val,
        Hi_hopping_val,
        xx,
        slr.data(),
        sli.data(),
        scr.data(),
        sci.data());
    CUDA_CHECK_KERNEL

        std::vector<double> vac(number_of_energy_points);


    find_dos_or_others(
        number_of_orbitals,
        number_of_moments,
        number_of_energy_points,
        maximum_energy,
        E.data(),
        NN,
        NL,
        Hr_onsite_val,
        Hr_hopping_val,
        Hi_hopping_val,
        number_of_atoms,
        scr.data(),
        sci.data(),
        srr.data(),
        sri.data(),
        vac.data(),
        type_length,
        dev_orbit_offset, dev_orbit_offset2, dev_orbit_atom_id, dev_type_content);

    FILE* os_sigma = my_fopen("lsqt_sigma.out", "a");
    const double S_per_m_conversion = 7.748091729e5 * PI;

    std::reverse(vac.begin(), vac.end());

    for (int n = 0; n < number_of_energy_points; ++n) {
        sigma[n] += vac[n] * time_step / V;
        fprintf(os_sigma, "%25.15e", sigma[n] * S_per_m_conversion);
    }
    fprintf(os_sigma, "\n");
    fclose(os_sigma);
}
