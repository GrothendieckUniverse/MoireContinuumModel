module RStackMoireContinuumModel


export R_Stack_Moire_Model, params_FengchengWu, params_ChongWang
export initialize_r_stack_moire_continuum_model, plot_moire_bands

using PhysicalConstants.CODATA2018: ħ, m_e
using Unitful

using LinearAlgebra, SparseArrays, KrylovKit
using Plots, MLStyle


"""
Struct `R_Stack_Moire_Model` for R-Stacked (Parallel-Stacked) Twisted Bilayer TMD Model
---
- Fields:
    - `brav_vec_list::Vector{Vector{Float64}}`: Bravais vectors of the moire superlattice
    - `reciprocal_vec_list::Vector{Vector{Float64}}`: Reciprocal lattice vectors of the moire superlattice
    - `params::Dict{String,<:Number}`: Parameters of the model
    - `nG_cutoff::Int64`: The cutoff of the reciprocal lattice vectors
    - `G_int_list::Vector{Vector{Int64}}`: List of reciprocal lattice vectors
    - `G_int_to_iG_dict::Dict{Vector{Int64},Int64}`: Mapping from reciprocal lattice vectors to indices
    - `nG::Int64`: Length of `G_int_list`
    - `potential_dict::Dict{Vector{Int64},AbstractMatrix}`: Dictionary of potential blocks
    - `nlayer::Int64`: Number of layers
    - `hk_matrix_for_valley::F where {F<:Function}`: Function `hk_matrix_for_valley(k_crys::Vector{Float64}, valley::Int)` to get the Hamiltonian matrix for a given moire crystal momentum and valley index
"""
mutable struct R_Stack_Moire_Model
    brav_vec_list::Vector{Vector{Float64}}
    reciprocal_vec_list::Vector{Vector{Float64}}
    params::Dict{String,<:Number}
    nG_cutoff::Int64
    G_int_list::Vector{Vector{Int64}}
    G_int_to_iG_dict::Dict{Vector{Int64},Int64}
    nG::Int64

    potential_dict::Dict{Vector{Int64},AbstractMatrix} # includes both moire potential and interlayer tunneling blocks
    nlayer::Int64

    hk_matrix_for_valley::F where {F<:Function} # `(k_crys, valley_index::Int) -> moire Hamiltonian matrix`
end

"Fengcheng Wu's parameters from local twisting fitting"
const params_FengchengWu = Dict(
    "a" => 3.472, # Angstrom
    "m" => 0.62, # Electron mass
    "V" => 8.0, # meV
    "ψ" => -89.6, # deg
    "w" => -8.5, # meV
    "θ" => 1.2, # deg
    "d" => 300, # gating distance
    "interlayer_distance" => 7.8, # in unit of Å, see suplemental material of `PhysRevLett.122.086402`
    "eE" => 0.0, # meV/Å so the layer potential difference is `eE * d`
)
"Chong Wang's parameters from large-scale DFT fitting"
const params_ChongWang = Dict(
    "a" => 3.52, # Angstrom
    "m" => 0.6, # Electron mass
    "V" => 20.75, # meV ~ 20.8 meV
    "ψ" => 107.7, # deg
    "w" => -23.82, # meV ~ 23.8 meV
    "θ" => 3.89, # deg
    "d" => 300, # gating distance
    "interlayer_distance" => 7.4, # in unit of Å, see Fig.(1d) in `PhysRevLett.132.036501`
    "eE" => 0.0, # meV/Å, so the layer potential difference is `eE * d`
)

"""
Valley Hamiltonian Matrix from `H_{v;l,g,l',g'}(k) = ⟨v,l,g,k|H|v,l′,g′,k⟩`
---
for a given valley index and moire crystal momentum. The order of slots here is important!
"""
function _get_Hamiltonian_for_valley(k_crys, valley_index::Int64; G_int_list::Vector{<:Vector{Int}}, potential_dict::Dict{Vector{Int64},<:Matrix}, mass::Float64, reciprocal_vec_list::Vector{<:Vector{Float64}})
    nG = length(G_int_list)
    @assert valley_index in 1:2 "valley_index must be 1 or 2, representing valley K or K′"
    nlayer = size(first(values(potential_dict)), 1) # layer degrees of freedom

    hk_array_for_valley = zeros(ComplexF64, nlayer, nG, nlayer, nG) # `H_{v;l,g,l',g'}(k) = ⟨v,l,g,k|H|v,l′,g′,k⟩`. Note: it is OK to change the order for slots for `hk_array_for_valley` here. But because we want to splat it into the Hamiltonian matrix using `reshape` method, with the result that the column/row are of the form `g + κ_l`, the order of slots are almost fixed here. Indeed, you can only swith the order of the first two and last two slots, but never mix them.


    for (iG, G) in enumerate(G_int_list)
        hk_array_for_valley[:, iG, :, iG] += @match valley_index begin
            1 => _kinetic_block(k_crys + G; mass=mass, reciprocal_vec_list=reciprocal_vec_list) # valley `K`
            2 => conj(_kinetic_block(-k_crys - G; mass=mass, reciprocal_vec_list=reciprocal_vec_list)) # valley `K′` as the time-reversal of valley `K`
        end
        for (iG′, G′) in enumerate(G_int_list)
            # input moire potential
            ΔG = G - G′
            if ΔG in keys(potential_dict)
                hk_array_for_valley[:, iG, :, iG′] += @match valley_index begin
                    1 => potential_dict[ΔG]
                    2 => conj(potential_dict[-ΔG])
                end
            end
        end
    end

    return Hermitian(reshape(hk_array_for_valley, nlayer * nG, nlayer * nG))


    # hk_array_for_valley = zeros(ComplexF64, nlayer, 2 * nG_cutoff + 1, 2 * nG_cutoff + 1, nlayer, 2 * nG_cutoff + 1, 2 * nG_cutoff + 1) # layer, G1, G2, layer′, G1′, G2′
    # for iG1 in axes(hk_array_for_valley, 2), iG2 in axes(hk_array_for_valley, 3)
    #     G = [iG1 - nG_cutoff - 1, iG2 - nG_cutoff - 1]
    #     hk_array_for_valley[:, iG1, iG2, :, iG1, iG2] += @match valley_index begin
    #         1 => _kinetic_block(k_crys + G, m) # valley `K`
    #         2 => conj(_kinetic_block(-k_crys - G, m)) # valley `K′` as the time-reversal of valley `K`
    #     end
    #     for iG1′ in axes(hk_array_for_valley, 5), iG2′ in axes(hk_array_for_valley, 6)
    #         G′ = [iG1′ - nG_cutoff - 1, iG2′ - nG_cutoff - 1]
    #         ΔG = G - G′
    #         if ΔG in keys(potential_dict)
    #             hk_array_for_valley[:, iG1, iG2, :, iG1′, iG2′] += @match valley_index begin
    #                 1 => potential_dict[ΔG]
    #                 2 => conj(potential_dict[-ΔG])
    #             end
    #         end
    #     end
    # end

    # return Hermitian(reshape(hk_array_for_valley, nlayer * nG, nlayer * nG))
end



function _kinetic_block(k_crys; mass::Float64, reciprocal_vec_list::Vector{<:Vector{Float64}})
    κ_plus = [2 / 3, -1 / 3] # moire `κ+`
    κ_minus = [1 / 3, 1 / 3] # moire `κ-`

    #(1.0546 x 10^-34)^2*10^20/(9.109*10^(-31))/(1.6022*10^(-19))*1000=7621
    unit_const = ustrip(u"meV * Å^2", ħ^2 / m_e) # see `PhysicalConstants.CODATA2018`

    Ek_t = -1 / (2 * mass) * norm(sum(reciprocal_vec_list .* (k_crys - κ_plus)))^2
    Ek_b = -1 / (2 * mass) * norm(sum(reciprocal_vec_list .* (k_crys - κ_minus)))^2

    diagm(unit_const * [Ek_t, Ek_b])
end




function initialize_r_stack_moire_continuum_model(; params::Dict{String,<:Number}, nG_cutoff::Int64=5)
    @info let io = IOBuffer()
        write(io, "Initializing R-Stack Moire Continuum Model\n")
        write(io, "\twith parameters: $params") # # show(io, "text/plain", params)
        String(take!(io))
    end

    aM = params["a"] / (2 * sin(deg2rad(params["θ"] / 2))) # Angstrom
    params["aM"] = aM # add moire lattice constant to the parameter dict

    brav_vec_list = aM * [[sqrt(3) / 2, -1 / 2], [0, 1]] # Angstrom
    (aM1, aM2) = Tuple(brav_vec_list)
    # brav_vec_mat = hcat(brav_vec_list...)

    # G_list = [4 * pi / (sqrt(3) * aM) * [cos(pi * (j - 1) / 3), sin(pi * (j - 1) / 3)] for j in 1:6]
    reciprocal_vec_list = begin
        aM1_3D = push!(deepcopy(aM1), 1)
        aM2_3D = push!(deepcopy(aM2), 1)
        aM3_3D = [0, 0, 1]
        cell_volume = abs(aM1[1] * aM2[2] - aM1[2] * aM2[1]) # `cell_volume = |aM1∧aM2|`

        b1 = 2 * pi * cross(aM2_3D, aM3_3D)[1:2] / cell_volume
        b2 = 2 * pi * cross(aM3_3D, aM1_3D)[1:2] / cell_volume
        # b3 = cross(a1, a2)

        [b1, b2]
    end
    # reciprocal_vec_mat = 2 * pi * inv(brav_vec_mat)'


    nG_one_direction = 2 * nG_cutoff + 1
    G_int_list = [[iG_1 - nG_cutoff - 1, iG_2 - nG_cutoff - 1] for iG_2 in 1:nG_one_direction for iG_1 in 1:nG_one_direction] # the order of the loop for G1 and G2 is important here (it must be consistent with the loop when inputing the Hamiltonian)

    G_int_to_iG_dict = Dict{Vector{Int64},Int64}(G_int => iG for (iG, G_int) in enumerate(G_int_list))
    nG = length(G_int_list) # also equal to `(2 * nG_cutoff + 1)^2`


    potential_dict = Dict{Vector{Int64},Matrix{ComplexF64}}()
    moire_potential_dict = Dict{Vector{Int64},Matrix{ComplexF64}}()
    interlayer_tunneling_dict = Dict{Vector{Int64},Matrix{ComplexF64}}()
    let
        w = params["w"]
        V = params["V"]
        ψ = deg2rad(params["ψ"])
        eE = params["eE"]
        d = params["d"]
        interlayer_distance = params["interlayer_distance"]
        begin
            # keep the first g-shell only, i.e., input in the order of `g1, g2, g3, g4, g5, g6`. Here `g1` is along x-direction, and `gi=R((i-1)*π/3)g1`
            # here first g-shell means a honeycomb grids spanned with a single unit of moire reciprocal vector `b1` for either layers (so we have two overlapping honeycombs differing by a shift of `(κ+)-(κ-)`). 
            # Note: truncation of the first g-shell means that every hopping matrix elements must be included in the honeycomb of the source sites.
            moire_potential_dict[[1, 0]] = diagm([V * exp(im * ψ), V * exp(-im * ψ)])
            moire_potential_dict[[-1, 1]] = diagm([V * exp(im * ψ), V * exp(-im * ψ)])
            moire_potential_dict[[0, -1]] = diagm([V * exp(im * ψ), V * exp(-im * ψ)])
            moire_potential_dict[[0, 1]] = diagm([V * exp(-im * ψ), V * exp(im * ψ)])
            moire_potential_dict[[-1, 0]] = diagm([V * exp(-im * ψ), V * exp(im * ψ)])
            moire_potential_dict[[1, -1]] = diagm([V * exp(-im * ψ), V * exp(im * ψ)])


            # displacment field induced sublattice potential differences (which effectively contributes to `g=0` moire potential). Note: this term can also be directly added to kinetic block
            moire_potential_dict[[0, 0]] = diagm([-eE * interlayer_distance / 2, eE * interlayer_distance / 2])


            interlayer_tunneling_dict[[0, 0]] = [0 w; conj(w) 0] # for the zeroth shell: both directions of hoppings are within their own honeycombs
            interlayer_tunneling_dict[[1, 0]] = zeros(ComplexF64, 2, 2) # for g1: both directions of hoppings get outside of their own honeycombs
            interlayer_tunneling_dict[[0, 1]] = [0 0; conj(w) 0] # for g2: only hopping from top honeycomb to bottom honeycomb is within the top honeycomb. The reverse get outside of the bottom honeycomb
            interlayer_tunneling_dict[[-1, 1]] = [0 0; conj(w) 0] # for g3: only hopping from top honeycomb to bottom honeycomb is within the top honeycomb. The reverse get outside of the bottom honeycomb
            interlayer_tunneling_dict[[-1, 0]] = zeros(ComplexF64, 2, 2) # for g4: both directions of hoppings get outside of their own honeycombs
            interlayer_tunneling_dict[[0, -1]] = [0 w; 0 0] # for g4: only hopping from bottom honeycomb to top honeycomb is within the bottom honeycomb. The reverse get outside of the top honeycomb
            interlayer_tunneling_dict[[1, -1]] = [0 w; 0 0] # for g5: only hopping from bottom honeycomb to top honeycomb is within the bottom honeycomb. The reverse get outside of the top honeycomb
        end
    end
    @assert size(first(values(moire_potential_dict))) == size(first(values(interlayer_tunneling_dict))) "Illegal input: the moire potential block must be of the same shape as the interlayer tunneling block!"

    nlayer = size(first(values(moire_potential_dict)), 1) # here is layer degrees of freedom

    g_shell_range = (-1):1
    for ig1 in g_shell_range, ig2 in g_shell_range
        potential_dict[[ig1, ig2]] = get(moire_potential_dict, [ig1, ig2], zeros(ComplexF64, 2, 2)) + get(interlayer_tunneling_dict, [ig1, ig2], zeros(ComplexF64, 2, 2))
    end

    nlayer = size(first(values(potential_dict)), 1) # here is layer degrees of freedom


    function hk_matrix_for_valley(k_crys, valley_index::Int64)::Matrix{ComplexF64}
        _get_Hamiltonian_for_valley(k_crys, valley_index;
            :G_int_list => G_int_list,
            :potential_dict => potential_dict,
            :mass => params["m"],
            :reciprocal_vec_list => reciprocal_vec_list
        )
    end

    return R_Stack_Moire_Model(
        brav_vec_list,
        reciprocal_vec_list,
        params,
        nG_cutoff,
        G_int_list,
        G_int_to_iG_dict,
        nG,
        potential_dict,
        nlayer,
        hk_matrix_for_valley
    )
end


"""
Plot Moire Bands Along a Given `k_crys_path`
---
- Args:
    - `moire_model::R_Stack_Moire_Model`: The moire model
- Named Args:
    - `valley_index::Int64=1`: Valley index
    - `k_crys_path::Vector{Vector{Float64}}=[[0.0, 0.0], [0.5, 0.0], [1 / 3, 1 / 3], [0.0, 0.0]]`: Path in the crystal momentum space
    - `nk_per_segment=25`: Number of k-points per segment
    - `nband=4`: Number of bands to plot
    - `plot_range=(-50, 0)`: Energy range to plot
    - `align_spec_to_zero::Bool=true`: Align the spectrum to zero
    - `show_spectrum::Bool=false`: Show the spectrum
"""
function plot_moire_bands(moire_model::R_Stack_Moire_Model; valley_index::Int64=1, k_crys_path::Vector{Vector{Float64}}=[[0.0, 0.0], [0.5, 0.0], [1 / 3, 1 / 3], [0.0, 0.0]], nk_per_segment=25, nband=4, plot_range=(-50, 0), align_spec_to_zero::Bool=true, show_spectrum::Bool=false)

    G1 = [1, 0]
    G2 = [cos(π / 3), sin(π / 3)]
    G_mat = hcat(G1, G2)
    G_mat_inv = inv(G_mat)

    k_cart_path = [sum([G1, G2] .* k_crys) for k_crys in k_crys_path]
    n_segment = length(k_crys_path) - 1

    turning_point_nk_list = Vector{Int}()
    k_crys_list = Vector{Vector{Float64}}()

    for i in 1:n_segment
        for j in 0:nk_per_segment
            k_cart = k_cart_path[i] + (k_cart_path[i+1] - k_cart_path[i]) / nk_per_segment * j # with head included and tail excluded
            k_crys = G_mat_inv * k_cart
            push!(k_crys_list, k_crys)
        end
    end
    push!(k_crys_list, k_crys_path[end]) # add the last turning point


    current_turning_point = 1
    for _ in 1:n_segment
        push!(turning_point_nk_list, current_turning_point)
        current_turning_point += nk_per_segment + 1
    end
    push!(turning_point_nk_list, current_turning_point) # add the last turning point
    # @show turning_point_nk_list


    spec_list = Vector{Vector{Float64}}()
    for k_crys in k_crys_list
        eigvals, eigvecs, info = KrylovKit.eigsolve(moire_model.hk_matrix_for_valley(k_crys, valley_index), nband, :LR)
        push!(spec_list, eigvals[1:nband])
    end
    spec_mat = hcat(spec_list...)
    (spec_min, spec_max) = minimum(spec_mat), maximum(spec_mat)
    spec_range = spec_max - spec_min
    if align_spec_to_zero
        spec_mat = spec_mat .- maximum(spec_mat)
    end
    if show_spectrum
        println()
        show(stdout, "text/plain", spec_mat)
        println()
    end

    x_scaling_factor = 1 / 2 * length(k_crys_list) / spec_range # `2/5` is chosen to compare with the plot in `PhysRevLett.132.036501`
    x_data = 1:length(k_crys_list)
    x_data_scaled = x_scaling_factor * x_data

    p = Plots.plot(x_data_scaled, transpose(spec_mat),
        xlims=(minimum(x_data_scaled) - 1, maximum(x_data_scaled) + 1), ylims=plot_range,
        frame=:box,
        lw=2.5,
        aspect_ratio=:equal,
        legend=:false, ylabel="Energy (meV)", xticks=:false,)
    vline!(p, x_scaling_factor * turning_point_nk_list, lw=1, lc=:black, alpha=0.5)

    return p
end



end # module