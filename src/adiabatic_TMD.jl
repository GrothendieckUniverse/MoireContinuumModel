module AdiabaticTMD

using LinearAlgebra, MLStyle

using CMP_Utils

"Chong Wang's parameters from large-scale DFT fitting, see `PhysRevLett.132.036501`"
const params_ChongWang = Dict(
    "a" => 3.52, # Angstrom
    "m" => 0.6, # Electron mass
    "V" => 20.75, # meV ~ 20.8 meV
    "ψ" => -107.7, # deg
    "w" => -23.82, # meV ~ 23.8 meV
    "θ" => 3.89, # deg
    "d" => 300, # gating distance
    "interlayer_distance" => 7.4, # in unit of Å, see Fig.(1d) in `PhysRevLett.132.036501`
    "eE" => 0.0, # meV/Å, so the layer potential difference is `eE * interlayer_distance`
)


function initialize_adiabatic_TMD_Hamiltonian(; params::Dict{String,<:Number}=params_ChongWang, nG_cutoff::Int64=1, show_params::Bool=false, rotational_symmetric_cutoff::Bool=true)
    params = merge(params_ChongWang, params)
    if show_params
        @info let io = IOBuffer()
            write(io, "Initializing R-Stack Moire Continuum Model with parameters:\n")
            write(io, "\t$params") # show(io, "text/plain", params)
            String(take!(io))
        end
    end

    aM = params["a"] / (2 * sin(deg2rad(params["θ"] / 2))) # in unit of Å
    params["aM"] = aM
    brav_vec_list = aM * [[sqrt(3) / 2, -1 / 2], [0, 1]] # this choice of the bravias vector is important: it ensures `reciprocal_vec_list≡[𝐆1,𝐆2]` to be stored in the order of `𝐆1=[|𝐆|,0.0]` and `𝐆2≡e^{i2π/6}𝐆1`
    reciprocal_vec_list = CMP_Utils.dual_basis_vec_list(brav_vec_list)
    @assert reciprocal_vec_list[1][2] ≈ 0.0 # check convention for the moire reciprocal vector: `𝐆1=[|𝐆|,0.0]`
    @assert angle(reciprocal_vec_list[2][1] + im * reciprocal_vec_list[2][2]) ≈ π / 3 # check convention for the moire reciprocal vector: `𝐆2≡e^{i2π/6}𝐆1`


    G_int_list = if rotational_symmetric_cutoff
        G_int_list_enlarged = [[iG1, iG2] for iG2 in -4*nG_cutoff:4*nG_cutoff for iG1 in -4*nG_cutoff:4*nG_cutoff]

        # filter out those `G_int` such that the `G_crys_list` is rotation symmetric
        G_crys_list_enlarged = [sum(reciprocal_vec_list .* G_int) for G_int in G_int_list_enlarged]
        G_crys_threshold = nG_cutoff * norm(reciprocal_vec_list[1]) * (1.0 + 1.0E-10) # the threshold is set to be slightly larger than `nG_cutoff * |𝐆1|`
        G_crys_list = filter(G_crys -> norm(G_crys) < G_crys_threshold, G_crys_list_enlarged)
        G_ind_list = [findfirst(x -> x == G_crys, G_crys_list_enlarged) for G_crys in G_crys_list]
        G_int_list = G_int_list_enlarged[G_ind_list]
    else
        [[iG1, iG2] for iG2 in -nG_cutoff:nG_cutoff for iG1 in -nG_cutoff:nG_cutoff]
    end
    G_int_to_iG_dict = Dict(G_int => iG for (iG, G_int) in enumerate(G_int_list))
    nG = length(G_int_list)


    G_int_list
end



end # module AdiabaticTMD