using Distributions
using JLD
using NCDatasets
using LinearAlgebra
using Glob

function write_ekp()
    # Get versions
    version_files = glob("versions_*.txt")
    n_params = 2
    ens_all = Array{Float64, 2}[]
    for file in version_files
        versions = readlines(file)
        u = zeros(length(versions), n_params)
        for (ens_index, version_) in enumerate(versions)
            open("$(version_).output/$(version_)", "r") do io
                u[ens_index, :] = [parse(Float64, line) for (index, line) in enumerate(eachline(io)) if index%3 == 0]
            end
        end
        push!(ens_all, u)
    end
    save( string("ekp_clima.jld"), "eki_u", ens_all)
    return
end

write_ekp()
