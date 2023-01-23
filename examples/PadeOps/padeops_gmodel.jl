module padeops_gmodel

using DocStringExtensions

using Random
using Distributions
using LinearAlgebra
using FFTW
using Statistics
using Printf

export run_G_ensemble
export padeops_run
export padeops_wait
export padeops_postprocess


"""
$(DocStringExtensions.TYPEDEF)

Structure to hold all information to run the forward model *G*.

# Fields
$(DocStringExtensions.TYPEDFIELDS)
"""

# Settings for PadeOps forward model using an input file
mutable struct PSettings
    # Nondimensional end time
    tstop::Float64
    # RunID
    runid::Int32
    # starting time index for post-processing
    tidx_start::Int32
    # interval between time indexes
    delta_tidx::Int32
    # ending time index for post-processing
    tidx_end::Int32
    # time indexes between starting and ending time index
    tidx_steps::Int32
    # phi
    phi::Int32
    # angular velocity of Earth
    omega::Float64
    # Characteristic length
    L::Int32
    # Characteristic velocity
    G::Int32
    # Rossby number
    Ro::Float64
    # Nondimensional length x
    Lx
    # Nondimensional length y
    Ly
    # Nondimensional length z
    Lz
    # x grid points
    nx
    # y grid points
    ny
    # z grid points
    nz 
    # G model statistics type
    stats_type::Int32
    # input file
    inputfile::String  
end

# These are my parameters of interest
struct PParams
    # Roughness length (nondimensional)
    z0::Float64
end


"""
$(DocStringExtensions.TYPEDSIGNATURES)

Run the forward model *G* for an array of parameters by iteratively
calling `run_G` for each of the *N\\_ensemble* parameter values.

- `params` - array of size (*N\\_ensemble* × *N\\_parameters*) containing the parameters for 
  which *G* will be run.
- `settings` - a [GSetttings](@ref) struct.

Returns `g_ens`, an array of size (*N\\_ensemble* × *N\\_data*), where g_ens[j,:] = G(params[j,:]).
"""
function run_G_ensemble(params::Array{FT, 2}, settings::PSettings, rng_seed = 42) where {FT <: AbstractFloat}

    # Initialize ensemble
    N_ens = size(params, 2) # params is N_params x N_ens
    nd = 2   # number of stats? 2: mean & variance
    runid_ens = Array{Int64,1}(undef,N_ens);  # empty array for an array of runid
    job_ens = Array{Int64,1}(undef,N_ens);  # empty array for an array of job numbers
    g_ens = zeros(nd, N_ens);  # empty array for an ensemble of post-processed statistics   (e.g., 2x10)

    # Parameters to estimate
    z0 = params[1, :]       # roughness length
    
    # Phase 1: Run the model for an ensemble and output job numbers in an array
    Random.seed!(rng_seed)
    for i in 1:N_ens
        settings.runid = settings.runid + i - 1;
        runid_ens[i] = settings.runid;
        padeops_params = padeops_gmodel.PParams(z0[i])        # run the model with the current parameters, i.e., map θ to G(θ)
        job_ens[i] = padeops_run(settings, padeops_params);   # output the job numbers
    end
    
    # Phase 2: Wait for all ensemble members to finish running
    padeops_wait(job_ens);
    
    # Phase 3: Post-process the ensemble of data and output 
    for i in 1:N_ens
        g_ens[:, i] = padeops_postprocess(settings, runid_ens[i]);
    end
    
    return g_ens
end



# Takes in the settings and parameters for the model and then runs it, then returns post-processed statistics 
# Input: settings from .dat input file
# Output: Post-processed statistics from simulation
function padeops_run(settings::PSettings, params::PParams)
    
    inputfile = settings.inputfile;
    # Implement changes to input file
    tstop_string = string("tstop                         = ", settings.tstop, "D0    ! Physical time to stop the simulation");
    run(`sed -i -E "s+tstop.*+$tstop_string+" $inputfile`);
    runid_string = string("RunID                         = ", @sprintf("%2.2i",settings.runid), "          ! Run Label (All output files will be tagged with this number)")
    run(`sed -i -E "s+RunID.*+$runid_string+" $inputfile`);
    ksrunid_string = string("KSRunID                       = 99         ! RunID tag for KS files")
    run(`sed -i -E "s+KSRunID.*+$ksrunid_string+" $inputfile`);
    tidx_start_string = string("t_dataDump                    = ", settings.tidx_start, "         ! Data dumping frequency (# of timesteps)");
    run(`sed -i -E "s+t_dataDump.*+$tidx_start_string+" $inputfile`);
    
    
    # Run simulation on cluster using sbatch command
    bashfile_path = "/work2/09033/youngin/stampede2/bash_scripts/"
    bashfile_name = "submit_opti_neutral_skxnorm.sh"
    bashfile = joinpath(bashfile_path, bashfile_name)
    # Add input file into the bash file
    inputfile_string = string("export inputFile=\"Run", @sprintf("%2.2i",settings.runid), ".dat\"");
    run(`sed -i -E "s+export inputFile=\".*+$inputfile_string+" $bashfile`);
    run(pipeline(`sbatch $bashfile`; stdout="log.txt"));
    println("Script submitted");
    
    # Find job number 
    f = "log.txt";
    job_number = 0;
    for ln in eachline(open(f))
        if  (contains(ln,"Submitted batch job "))
            job_number_str = replace(ln, "Submitted batch job "=>"");
            job_number = parse(Int64, job_number_str);  
        end
    end
    println(job_number);
    
    return job_number;    
end




 





# Takes in an array of job numbers as input
# Finishes when all jobs have finished running
# No output 
function padeops_wait(job_ens)
    
    # Check to see if all ensemble members have started running
    all_running = false;
    counter = 0;
    ens_state = zeros(1,length(job_ens))
    while (!all_running)
        counter = counter + 1;
        println("Waiting 1 minute to see if all ensemble members have begun running...");
        sleep(60)  # pause for 1 minutes
        # terminal point : 100 cycles approx 100 minutes
        if counter == 100;   
            break;
        end
        
        # check to see if job has started running
        run(pipeline(`squeue -u youngin`, stdout="checkjob.txt"));    
        for i in 1:length(job_ens)
            contain = false;
            for ln in eachline(open("checkjob.txt"))
                if (contains(ln, string(job_ens[i])))
                    println(ln);
                    if (contains(ln, "R"))
                        println("Job ", job_ens[i], " running.");
                        ens_state[i] = 1;
                        break;
                    elseif (contains(ln, "PD"))
                        println("Job ", job_ens[i], " pending.");
                    end
                    contain = true;
                end
            end
            if contain == false;
                println("Job ", job_ens[i], " not found.");
            end
        end
        
        # check if all ensemble members are running
        if mean(ens_state) == 1.0
            println("All running.");
            all_running = true;
        end
    end
    
    # Check to see if all ensemble members have finished running
    all_finished = false;
    counter = 0;
    while (!all_finished)
        counter = counter + 1;
        println("Waiting 1 minute to see if all ensemble members have finished running...");
        sleep(60)  # pause for 1 minutes
        # terminal point : 100 cycles approx 100 minutes
        if counter == 100;   
            break;
        end
        # check to see if job has finished
        run(pipeline(`squeue -u youngin`, stdout="checkjob.txt"));    
        for i in 1:length(job_ens)
            contain = false;
            for ln in eachline(open("checkjob.txt"))
                if (contains(ln, string(job_ens[i])))
                    println("Job ", job_ens[i], " still running.");
                    contain = true;
                    break;
                end
            end
            if contain == false
                println("Job ", job_ens[i], " has finished.");
                ens_state[i] = 2;
            end
        end
        
        println("ens_state: ", ens_state);
        # check if all ensemble members have finished running
        if mean(ens_state) == 2.0
            all_finished = true;
        end
    end   
end





### Functions for post-processing
###
# function for reading in fortran files
function read_fortran_file(fname,nx,ny,nz)
    f = open(fname,"r")
    dat = reinterpret(Float64, read(f))
    dat = reshape(dat,(nx,ny,nz))
    return dat
end



function padeops_postprocess(settings::PSettings, run_id) 
    
    println("Starting post-processing...")

    # directory for all simulation data
    basedir = "/scratch/09033/youngin/ces_padeops/neutral_pbl_test";
    
    # variables from settings needed for post-processing
    runid = run_id;
    phi = settings.phi;
    omega = settings.omega;
    L = settings.L;
    G = settings.G;
    Ro = settings.Ro;
    # Domain
    Lx = settings.Lx; nx = settings.nx;
    Ly = settings.Ly; ny = settings.ny;
    Lz = settings.Lz; nz = settings.nz;
    # Grid
    dz = Lz/nz; dx = Lx/nx; dy = Ly/ny;
    zEdge = LinRange(0,Lz,nz+1); zCell = 0.5*(zEdge[2:end]+zEdge[1:end-1])';
    x = 0:dx:Lx-dx; y = 0:dy:Ly-dy;
    
    # time settings
    tidx_start = settings.tidx_start;
    delta_tidx = settings.delta_tidx;
    tidx_end = settings.tidx_end;
    tidx_steps = settings.tidx_steps;
    
    u = zeros(nx,ny,nz,tidx_steps);
    v = zeros(nx,ny,nz,tidx_steps);
    w = zeros(nx,ny,nz,tidx_steps);
    potT = zeros(nx,ny,nz,tidx_steps);
    nSGS = zeros(nx,ny,nz,tidx_steps);
    # dimensional domain
    x = L*x; y = L*y; z = L*zCell';
    
    global m = 1;  # counter
    # over a time interval
    for tidx in tidx_start:delta_tidx:tidx_end
        label = "uVel";
        fname = string(basedir,"/Run",@sprintf("%2.2i",runid),"_",label,"_t",@sprintf("%6.6i",tidx),".out")
        u[:,:,:,m] = read_fortran_file(fname,nx,ny,nz);
        global m = m + 1;
    end
    
    # Mean over x,y, and x,y,t, and x,y,z,t
    u_mean_xy = dropdims(mean(u,dims=(1,2)),dims=(1,2));
    u_mean_xyt = dropdims(mean(u,dims=(1,2,4)),dims=(1,2));
    u_mean_xyzt = dropdims(mean(u,dims=(1,2,3,4)),dims=(1,2,3));
    # Variance
    um_LES = mean(u,dims=(1,2,4));
    uf = u .- um_LES;
    ufuf = dropdims(mean(uf.*uf,dims=(1,2,3,4)),dims=(1,2,3));
    # Combine statistics (currently vertical columns of mean and variance)
    gt = vcat(u_mean_xyzt, ufuf)

    println("gt: ",gt);
    return gt
    
    println("Post-processing finished.")
end


end # end of module



