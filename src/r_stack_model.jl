module RStackMoireContinuumModel


export R_Stack_Moire_Model, params_FengchengWu, params_ChongWang
export initialize_r_stack_moire_continuum_model, plot_moire_bands

using PhysicalConstants.CODATA2018: ħ, m_e
using Unitful

using LinearAlgebra, SparseArrays, KrylovKit
using Plots, MLStyle
using HilbertSpace, CMP_Utils

"""
Struct `R_Stack_Moire_Model` for R-Stacked (Parallel-Stacked) Twisted Bilayer TMD Model
---
- Fields:
    - `brav_vec_list::Vector{Vector{Float64}}`: bravais vectors of the moire superlattice
    - `reciprocal_vec_list::Vector{Vector{Float64}}`: reciprocal lattice vectors of the moire superlattice
    - `params::Dict{String,<:Number}`: parameters of the model
    - `nG_cutoff::Int64`: cutoff of `G_int` in either directions
    - `G_int_list::Vector{Vector{Int64}}`: list of `G_int` that are used to construct the moire Hamiltonian
    - `G_int_to_iG_dict::Dict{Vector{Int64},Int64}`: dict `G_int -> iG`
    - `nG::Int64`: length of `G_int_list`
    - `nl::Int64`: number of layers
    - `moire_Hamitonian_basis::HilbertSpace.Finite_Dimensional_Single_Particle_Hilbert_Space`: single-particle basis that spans the moire Hamiltonian matrix
    - `hk_matrix_for_valley::F where {F<:Function}`: function `(k_crys, valley_index::Int) -> moire Hamiltonian matrix for that valley`
"""
mutable struct R_Stack_Moire_Model
    brav_vec_list::Vector{Vector{Float64}}
    reciprocal_vec_list::Vector{Vector{Float64}}
    params::Dict{String,<:Number}
    nG_cutoff::Int64
    G_int_list::Vector{Vector{Int64}}
    G_int_to_iG_dict::Dict{Vector{Int64},Int64}
    nG::Int64
    nl::Int64

    moire_Hamitonian_basis::HilbertSpace.Finite_Dimensional_Single_Particle_Hilbert_Space
    hk_matrix_for_valley::F where {F<:Function} # function `(k_crys, valley_index::Int) -> moire Hamiltonian matrix for that valley`
end

"Fengcheng Wu's parameters from local stacking fitting, see `PhysRevLett.122.086402`"
const params_FengchengWu = Dict(
    "a" => 3.472, # Angstrom
    "m" => 0.62, # Electron mass
    "V" => 8.0, # meV
    "ψ" => -89.6, # deg
    "w" => -8.5, # meV
    "θ" => 1.2, # deg
    "d" => 300, # gating distance
    "interlayer_distance" => 7.8, # in unit of Å, see suplemental material of `PhysRevLett.122.086402`
    "eE" => 0.0, # meV/Å so the layer potential difference is `eE * interlayer_distance`
)

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



"""
Kinetic Part of the Moire Hamiltonian within the Layer Block
---
including the displacment-field-induced potential differences.
- Args:
    - `k_crys::Vector{Float64}`: moire crystal momentum
- Named Args:
    - `reciprocal_vec_list::Vector{<:Vector{Float64}}`: reciprocal lattice vectors of the moire superlattice
    - `params::Dict{String,<:Number}`: parameters of the model
"""
function _kinetic_block(k_crys; reciprocal_vec_list::Vector{<:Vector{Float64}}, params::Dict{String,<:Number})
    κ_plus = [2 / 3, -1 / 3] # moire `κ+ = (G1+G6)/3 = (2/3,-1/3)`
    κ_minus = [1 / 3, 1 / 3] # moire `κ- = (G1+G2)/3 = (1/3,1/3)`

    #(1.0546 x 10^-34)^2*10^20/(9.109*10^(-31))/(1.6022*10^(-19))*1000=7621
    unit = ustrip(u"meV * Å^2", ħ^2 / m_e) # see `PhysicalConstants.CODATA2018`
    m = params["m"]

    eE = params["eE"]
    interlayer_distance = params["interlayer_distance"]

    E_t = norm(sum(reciprocal_vec_list .* (k_crys - κ_plus)))^2
    E_b = norm(sum(reciprocal_vec_list .* (k_crys - κ_minus)))^2

    E_k = -unit * 1 / (2 * m) * diagm([E_t, E_b])
    E_eE = diagm([-eE * interlayer_distance / 2, eE * interlayer_distance / 2]) # displacment field induced sublattice potential differences (which effectively contributes to the `g=0` moire potential)

    return E_k + E_eE
end

"""
Generate the Dictionary `ΔG_int_to_layer_block_dict` within the First g-shell
---
We keep the difference of the moire-reciprocal vectors within the first g-shell only, i.e., `Δg≡g-g′` is constrained within the honeycomb spanned by `g1, g2, g3, g4, g5, g6` with `|gi|=|b1|=|b2|` *of their own layers* (so for top/bottom layers we have two overlapping honeycombs shifted by `(κ+)-(κ-)`). In our convention, `g1` is along x-direction, and `gi=R((i-1)*π/3)g1`.

Note: truncation within the first g-shell means that only those hopping matrix elements connecting sites within the first-g shell of the corresponding layers are included.
___
- Named Args:
    - `params::Dict{String,<:Number}`: parameters of the model
"""
function _get_ΔG_int_to_layer_block_dict_within_first_g_shell(; params::Dict{String,<:Number})::Dict{Vector{Int},Matrix{ComplexF64}}
    w = params["w"]
    V = params["V"]
    ψ = deg2rad(params["ψ"]) # the original input is in degree

    intralayer_moire_potential_dict = Dict{Vector{Int},Matrix{ComplexF64}}()
    interlayer_tunneling_dict = Dict{Vector{Int},Matrix{ComplexF64}}()
    ΔG_int_to_layer_block_dict = Dict{Vector{Int},Matrix{ComplexF64}}()

    # for intralayer moire potential, every hopping is within the first-g-shell honeycomb of the corresponding layer
    intralayer_moire_potential_dict[[1, 0]] = diagm([V * exp(im * ψ), V * exp(-im * ψ)])
    intralayer_moire_potential_dict[[-1, 1]] = diagm([V * exp(im * ψ), V * exp(-im * ψ)])
    intralayer_moire_potential_dict[[0, -1]] = diagm([V * exp(im * ψ), V * exp(-im * ψ)])
    intralayer_moire_potential_dict[[0, 1]] = diagm([V * exp(-im * ψ), V * exp(im * ψ)])
    intralayer_moire_potential_dict[[-1, 0]] = diagm([V * exp(-im * ψ), V * exp(im * ψ)])
    intralayer_moire_potential_dict[[1, -1]] = diagm([V * exp(-im * ψ), V * exp(im * ψ)])


    interlayer_tunneling_dict[[0, 0]] = [0 w; conj(w) 0] # for the zeroth shell: both directions of hoppings are within their own honeycombs
    # interlayer_tunneling_dict[[1, 0]] = zeros(ComplexF64, 2, 2) # for g1: both directions of hoppings get outside of their own honeycombs, no need to store the zero matrix
    interlayer_tunneling_dict[[0, 1]] = [0 0; conj(w) 0] # for g2: only hopping from top honeycomb to bottom honeycomb is within the top honeycomb. The reverse get outside of the bottom honeycomb
    interlayer_tunneling_dict[[-1, 1]] = [0 0; conj(w) 0] # for g3: only hopping from top honeycomb to bottom honeycomb is within the top honeycomb. The reverse get outside of the bottom honeycomb
    # interlayer_tunneling_dict[[-1, 0]] = zeros(ComplexF64, 2, 2) # for g4: both directions of hoppings get outside of their own honeycombs, no need to store the zero matrix
    interlayer_tunneling_dict[[0, -1]] = [0 w; 0 0] # for g5: only hopping from bottom honeycomb to top honeycomb is within the bottom honeycomb. The reverse get outside of the top honeycomb
    interlayer_tunneling_dict[[1, -1]] = [0 w; 0 0] # for g6: only hopping from bottom honeycomb to top honeycomb is within the bottom honeycomb. The reverse get outside of the top honeycomb


    for (k, v) in intralayer_moire_potential_dict
        ΔG_int_to_layer_block_dict[k] = get!(ΔG_int_to_layer_block_dict, k, zeros(ComplexF64, 2, 2)) + v
    end
    for (k, v) in interlayer_tunneling_dict
        ΔG_int_to_layer_block_dict[k] = get!(ΔG_int_to_layer_block_dict, k, zeros(ComplexF64, 2, 2)) + v
    end

    return ΔG_int_to_layer_block_dict
end

"""
Construct the Function of Moire Hamiltonian Matrix for a Given Valley
---
for a given moire crystal momentum `k_crys` and a valley index `iv∈(1,2)` representing valley-K or K′. The full moire Hamiltonian matrix elements for valley-v are expanded with the moire plane wave `⟨k,g,l,v|H|k,g′,l′,v⟩` for moire reciprocal vectors `g,g′`, and layer indices `l,l′`. Here we want to output the moire Hamiltonian for each valley as a function `(k_crys, iv) -> hk_matrix_for_valley`, *without* explicit construction of the full list of the moire Hamiltonian for each `k_crys` and each `iv`, thus here both crystal-momentum `k_crys` and valley index `iv` should be **excluded** from the single-particle basis that spans the moire Hamiltonian matrix.
___ 
Practically, we first construct the single-particle basis of type `HilbertSpace.Finite_Dimensional_Single_Particle_Hilbert_Space` spanned by the left degrees of freedom: the moire reciprocal vectors `iG`, and the layer index `il`. And then input the Hamiltonian matrix elements of the `hk_matrix_for_valley` in terms of this single-particle basis.
___
- Args:
    - `k_crys::Vector{<:Number}`: moire crystal momentum
    - `iv::Int`: valley index
- Named Args:
    - `reciprocal_vec_list::Vector{<:Vector{Float64}}`: reciprocal lattice vectors of the moire superlattice
    - `G_int_list::Vector{Vector{Int}}`: the truncated list of moire reciprocal vectors that are used to construct the moire Hamiltonian
    - `ΔG_int_to_layer_block_dict::Dict{Vector{Int},<:Matrix}`: dictionary from `ΔG_int` to the Hamiltonian matrix within the layer block
    - `params::Dict{String,<:Number}`: parameters of the model
"""
function _get_hk_matrix_for_valley(k_crys::Vector{<:Number}, iv::Int;
    reciprocal_vec_list::Vector{<:Vector{<:Float64}},
    G_int_list::Vector{Vector{Int}},
    ΔG_int_to_layer_block_dict::Dict{Vector{Int},Matrix{ComplexF64}},
    moire_Hamitonian_basis::HilbertSpace.Finite_Dimensional_Single_Particle_Hilbert_Space,
    params::Dict{String,<:Number}
)::Matrix{ComplexF64}
    @assert iv ∈ (1, 2) "Check input: the valley index `iv` must be 1 or 2, representing valley-K or K′"

    hk_matrix_for_valley = zeros(ComplexF64, moire_Hamitonian_basis.nstate, moire_Hamitonian_basis.nstate)

    for (iψ, ψ) in enumerate(moire_Hamitonian_basis.state_list)
        (iG, il) = ψ.dof_indices
        G_int = G_int_list[iG]

        layer_block = @match iv begin
            1 => _kinetic_block(k_crys + G_int; reciprocal_vec_list=reciprocal_vec_list, params=params)
            2 => conj(_kinetic_block(-(k_crys + G_int); reciprocal_vec_list=reciprocal_vec_list, params=params))
        end

        if layer_block[il, il] != 0.0
            hk_matrix_for_valley[iψ, iψ] += layer_block[il, il]
        end

        for (iψ′, ψ′) in enumerate(moire_Hamitonian_basis.state_list)
            (iG′, il′) = ψ′.dof_indices
            G′_int = G_int_list[iG′]

            ΔG = G_int - G′_int
            if ΔG in keys(ΔG_int_to_layer_block_dict)
                layer_block = @match iv begin
                    1 => ΔG_int_to_layer_block_dict[ΔG]
                    2 => conj(ΔG_int_to_layer_block_dict[-ΔG])
                end
                if layer_block[il, il′] != 0.0
                    hk_matrix_for_valley[iψ, iψ′] += layer_block[il, il′]
                end
            end
        end
    end

    return hk_matrix_for_valley
end


"""
Constructor of `R_Stack_Moire_Model`
---
- Named Args:
    - `params::Dict{String,<:Number}`: parameters of the model (default to be `params_ChongWang`)
    - `nG_cutoff::Int64=5`: cutoff of `G_int` in either directions
"""
function initialize_r_stack_moire_continuum_model(; params::Dict{String,<:Number}=params_ChongWang, nG_cutoff::Int64=5, show_params::Bool=false, rotational_symmetric_cutoff::Bool=false)::R_Stack_Moire_Model
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



    nl = 2 # number of layers, must be consistent with the size of values of `ΔG_int_to_layer_block_dict`

    moire_Hamitonian_basis = HilbertSpace.Finite_Dimensional_Single_Particle_Hilbert_Space(;
        dof_range_map=Dict(
            "G" => 1:nG,
            "l" => 1:nl
        )
    )

    ΔG_int_to_layer_block_dict = _get_ΔG_int_to_layer_block_dict_within_first_g_shell(; params=params)

    hk_matrix_for_valley = (k_crys::Vector{<:Number}, iv::Int) -> _get_hk_matrix_for_valley(k_crys, iv; reciprocal_vec_list=reciprocal_vec_list, G_int_list=G_int_list, ΔG_int_to_layer_block_dict=ΔG_int_to_layer_block_dict, moire_Hamitonian_basis=moire_Hamitonian_basis, params=params)


    return R_Stack_Moire_Model(
        brav_vec_list,
        reciprocal_vec_list,
        params,
        nG_cutoff,
        G_int_list,
        G_int_to_iG_dict,
        nG,
        nl,
        moire_Hamitonian_basis,
        hk_matrix_for_valley
    )
end



function rotation_eigval(n::Int; moire_model::R_Stack_Moire_Model, valley_index::Int=1)

    # for C3-system, we need to investigate three high-symmetry point `γ, k, k′`
    # [TODO!] first, one needs to derive the rotation transformation for the layer-block moire Hamiltonian to get the representation of the rotation
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
function plot_moire_bands(moire_model::R_Stack_Moire_Model; valley_index::Int64=1, k_crys_path::Vector{Vector{Float64}}=[[0.0, 0.0], [0.5, 0.0], [1 / 3, 1 / 3], [0.0, 0.0]], nk_per_segment=25, nband=5, plot_range=(-50, 0), align_spec_to_zero::Bool=true, show_spectrum::Bool=false)

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
        eigvals, eigvecs, info = KrylovKit.eigsolve(moire_model.hk_matrix_for_valley(k_crys, valley_index), nband, :LR, ishermitian=true)
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