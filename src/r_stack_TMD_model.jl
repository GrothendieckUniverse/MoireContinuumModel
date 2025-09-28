module RStackMoireContinuumModel


export R_Stack_Moire_Model, params_FengchengWu, params_ChongWang
export initialize_r_stack_moire_continuum_model, plot_moire_bands, plot_Ωxy_and_trace_G, plot_Ωxy_and_trace_G_with_eE

using PhysicalConstants.CODATA2018: ħ, m_e
using Unitful

using LinearAlgebra, SparseArrays, Arpack
using Plots, CairoMakie, MLStyle
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
    - `moire_Hamiltonian_basis::HilbertSpace.Single_Particle_Hilbert_Space`: single-particle basis that spans the moire Hamiltonian matrix
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

    moire_Hamiltonian_basis::HilbertSpace.Single_Particle_Hilbert_Space
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
Practically, we first construct the single-particle basis of type `HilbertSpace.Single_Particle_Hilbert_Space` spanned by the left degrees of freedom: the moire reciprocal vectors `iG`, and the layer index `il`. And then input the Hamiltonian matrix elements of the `hk_matrix_for_valley` in terms of this single-particle basis.
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
    moire_Hamiltonian_basis::HilbertSpace.Single_Particle_Hilbert_Space,
    params::Dict{String,<:Number}
)::Matrix{ComplexF64}
    @assert iv ∈ (1, 2) "Check input: the valley index `iv` must be 1 or 2, representing valley-K or K′"

    hk_matrix_for_valley = zeros(ComplexF64, moire_Hamiltonian_basis.nstate, moire_Hamiltonian_basis.nstate)

    for (iψ, ψ) in enumerate(moire_Hamiltonian_basis.state_list)
        (iG, il) = ψ.dof_indices
        G_int = G_int_list[iG]

        layer_block = @match iv begin
            1 => _kinetic_block(k_crys + G_int; reciprocal_vec_list=reciprocal_vec_list, params=params)
            2 => conj(_kinetic_block(-(k_crys + G_int); reciprocal_vec_list=reciprocal_vec_list, params=params))
        end

        if layer_block[il, il] != 0.0
            hk_matrix_for_valley[iψ, iψ] += layer_block[il, il]
        end

        for (iψ′, ψ′) in enumerate(moire_Hamiltonian_basis.state_list)
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

    moire_Hamiltonian_basis = HilbertSpace.Single_Particle_Hilbert_Space(; dof_name_to_range_pair_list=["G" => collect(1:nG), "l" => collect(1:nl)])

    ΔG_int_to_layer_block_dict = _get_ΔG_int_to_layer_block_dict_within_first_g_shell(; params=params)

    hk_matrix_for_valley = (k_crys::Vector{<:Number}, iv::Int) -> _get_hk_matrix_for_valley(k_crys, iv; reciprocal_vec_list=reciprocal_vec_list, G_int_list=G_int_list, ΔG_int_to_layer_block_dict=ΔG_int_to_layer_block_dict, moire_Hamiltonian_basis=moire_Hamiltonian_basis, params=params)


    return R_Stack_Moire_Model(
        brav_vec_list,
        reciprocal_vec_list,
        params,
        nG_cutoff,
        G_int_list,
        G_int_to_iG_dict,
        nG,
        nl,
        moire_Hamiltonian_basis,
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
        eigvals, eigvecs = Arpack.eigs(moire_model.hk_matrix_for_valley(k_crys, valley_index); nev=nband, which=:LR)
        eigvals = real.(eigvals) # ensure the eigenvalues are real
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



function plot_Ωxy_and_trace_G(; params::Dict{String,<:Number}=params_ChongWang, valley_index::Int=1, band_index::Int=1, which::Symbol=:LR, N::Int=10, BZ_shape::Symbol=:hexagon)
    @assert BZ_shape in [:parallelogram, :hexagon]

    params = merge(params_ChongWang, params)
    moire_model = initialize_r_stack_moire_continuum_model(; params=params, nG_cutoff=4)
    @show moire_model.reciprocal_vec_list

    hk_cart::Function = k_cart::Vector{Float64} -> begin
        # convert `k_cart` to `k_crys`
        G_mat = reduce(hcat, moire_model.reciprocal_vec_list)
        k_crys = inv(G_mat) * k_cart

        moire_model.hk_matrix_for_valley(k_crys, valley_index)
    end

    # compute `Ω_xy` and `trace(G)` within appropriate `k_cart` range
    @show G_norm = norm(moire_model.reciprocal_vec_list[1]) # norm of the moire reciprocal vector `G1`
    k_cart_list = if BZ_shape == :parallelogram
        k_crys_list = [[i / N, j / N] for i in 0:N for j in 0:N]
        k_cart_list = [reduce(+, k_crys .* moire_model.reciprocal_vec_list) for k_crys in k_crys_list] # convert `k_crys` to `k_cart`
    elseif BZ_shape == :hexagon
        k_crys_list = [[i / N, j / N] for i in -2N:2N for j in -2N:2N]
        k_cart_list = [reduce(+, k_crys .* moire_model.reciprocal_vec_list) for k_crys in k_crys_list] # convert `k_crys` to `k_cart`
        filter(k_cart -> abs(k_cart[1]) < G_norm && abs(k_cart[2]) < G_norm, k_cart_list) # filter out those `k_cart` that are outside the moire Brillouin zone
    end
    @show length(k_cart_list) # number of k-points in the moire Brillouin zone
    x_list = [k_cart[1] for k_cart in k_cart_list]
    y_list = [k_cart[2] for k_cart in k_cart_list]

    moire_cell_volume = let
        @show (G1, G2) = Tuple(moire_model.reciprocal_vec_list)
        G1_complex = G1[1] + im * G1[2]
        G2_complex = G2[1] + im * G2[2]
        begin
            norm(G1_complex) * norm(G2_complex) * sin(angle(G1_complex + im * G2_complex)) |> real # area of the moire unit cell
        end
    end
    @show moire_cell_volume


    Ω_list = [CMP_Utils.berry_curvature_Ωxy(hk_cart, k_cart, band_index; which=which) for k_cart in k_cart_list] * 1 / (2π) * moire_cell_volume # make it dimensionless

    trace_G_list = [CMP_Utils.trace_of_quantum_metric_tensor_G(hk_cart, k_cart, band_index; which=which) for k_cart in k_cart_list] * 1 / (2π) * moire_cell_volume # make it dimensionless

    # colorrange = (0.5 * min(minimum(abs.(Ω_list)), minimum(abs.(trace_G_list))), max(maximum(abs.(Ω_list)), maximum(abs.(trace_G_list))) * 1.5) # manually set the colorange for colorscaling
    # colorrange = (0.0, max(maximum(abs.(Ω_list)), maximum(abs.(trace_G_list))) * 1.5) # manually set the colorange for colorscaling
    colorrange = (0.0, 3)
    @show colorrange

    if BZ_shape == :parallelogram
        fig = Figure(size=(600, 300))
        ax1 = Axis(fig[1, 1]; aspect=sqrt(3), title=L"\Omega",)

        p1 = CairoMakie.scatter!(ax1, x_list, y_list;
            marker=:hexagon, markersize=24, # tune makersize to compactly tile the moire Brillouin zone
            color=Ω_list,
            # colorrange=extrema(Ω_list), # manually set the colorange for colorscaling
        )
        hidedecorations!(ax1)
        hidespines!(ax1)
        CairoMakie.Colorbar(fig[1, 2], p1; minorticksvisible=true)
        CairoMakie.colsize!(fig.layout, 1, Aspect(1, sqrt(3)))


        ax2 = Axis(fig[2, 1]; aspect=sqrt(3), title=L"\mathrm{tr}\mathcal G",)
        p2 = CairoMakie.scatter!(ax2, x_list, y_list;
            marker=:hexagon, markersize=24, # tune makersize to compactly tile the moire Brillouin zone
            color=trace_G_list,

            # colorrange=extrema(trace_G_list), # manually set the colorange for colorscalingng
        )
        hidedecorations!(ax2)
        hidespines!(ax2)
        CairoMakie.Colorbar(fig[2, 2], p2; minorticksvisible=true)
        CairoMakie.colsize!(fig.layout, 1, Aspect(1, sqrt(3)))

        @show sum(Ω_list) / length(Ω_list)
        @show sum(trace_G_list) / length(trace_G_list)

        display(fig)
        CairoMakie.save("/home/hxd/mBZ_berry_curvature_and_trace_G.pdf", fig)

    elseif BZ_shape == :hexagon
        fig = Figure(size=(300, 400))
        ax1 = Axis(fig[1, 1];
            aspect=1, title="Berry Curvature",
            xgridvisible=false, ygridvisible=false,
            xticks=[-0.1, 0.0, 0.1],
            yticks=[-0.1, 0.0, 0.1],
        )


        p1 = CairoMakie.scatter!(ax1, x_list, y_list;
            marker=:hexagon, markersize=18, # tune makersize to compactly tile the mBZ
            color=-Ω_list, # revert the sign of Berry curvature for comparison with `tr(G)` below
            colorrange=colorrange, # manually set the colorange for colorscaling
        )
        CairoMakie.xlims!(ax1, -0.8 * G_norm, 0.8 * G_norm)
        CairoMakie.ylims!(ax1, -0.8 * G_norm, 0.8 * G_norm)
        # hidedecorations!(ax1)
        # hidespines!(ax1)
        CairoMakie.Colorbar(fig[1, 2], p1)

        ax2 = Axis(fig[2, 1];
            aspect=1, title="Trace Quantum Metric",
            xgridvisible=false, ygridvisible=false,
            xticks=[-0.1, 0.0, 0.1],
            yticks=[-0.1, 0.0, 0.1],
        )

        p2 = CairoMakie.scatter!(ax2, x_list, y_list;
            marker=:hexagon, markersize=18, # tune makersize to compactly tile the mBZ
            color=trace_G_list,
            colorrange=colorrange, # manually set the colorange for colorscaling
        )
        CairoMakie.xlims!(ax2, -0.8 * G_norm, 0.8 * G_norm)
        CairoMakie.ylims!(ax2, -0.8 * G_norm, 0.8 * G_norm)
        # hidedecorations!(ax2)
        # hidespines!(ax2)
        CairoMakie.Colorbar(fig[2, 2], p2)


        # add BZ boundary
        # first, find six rotate-related BZ edges
        BZ_edge_init = 1 / 3 * reduce(+, moire_model.reciprocal_vec_list) # the initial BZ edge is `1/3 * (𝐆1+𝐆2)`
        BZ_edge_init_complex = BZ_edge_init[1] + im * BZ_edge_init[2]
        BZ_edge_complex_list = [BZ_edge_init_complex * exp(im * π / 3 * i) for i in 0:6] # the six rotate-related BZ edges 
        BZ_edge_list = [[real(BZ_edge_complex), imag(BZ_edge_complex)] for BZ_edge_complex in BZ_edge_complex_list] # convert to Cartesian coordinates
        xs = [edge[1] for edge in BZ_edge_list]
        ys = [edge[2] for edge in BZ_edge_list]
        Makie.lines!(ax1, xs, ys; color=:tomato, linewidth=2)
        Makie.lines!(ax2, xs, ys; color=:tomato, linewidth=2)

        # scale colorbars
        CairoMakie.colsize!(fig.layout, 1, Aspect(1, 1))

        # display(fig)
        CairoMakie.save("/home/hxd/mBZ_berry_curvature_and_trace_G.pdf", fig)
        CairoMakie.save("/home/hxd/mBZ_berry_curvature_and_trace_G.svg", fig)
    end
    return Ω_list, trace_G_list
end



function plot_Ωxy_and_trace_G_with_eE(; params::Dict{String,<:Number}=params_ChongWang, eE_pool::AbstractRange=0.0:0.5:2.6, N::Int=10, valley_index::Int=1, band_index::Int=1, which::Symbol=:LR, new_run::Bool=false)
    Ω_avg_list = Vector{Float64}()
    ΔΩ_list = Vector{Float64}()
    trace_G_avg_list = Vector{Float64}()
    Δtrace_G_list = Vector{Float64}()

    band_width_list = Vector{Float64}()

    if new_run
        # write data to file 
        data_dir = @__DIR__
        data_path = "$data_dir/geometry_data.txt"
        if isfile(data_path)
            @info "Overwriting existing data file $data_path"
        else
            @info "Creating new data file $data_path"
        end
        @info "Writing data to $data_path"

        for eE in eE_pool
            @info "Current eE = $(round(eE, digits=2)) [meV/Å]"
            params["eE"] = eE # set the electric field strength
            params = merge(params_ChongWang, params)
            moire_model = initialize_r_stack_moire_continuum_model(; params=params, nG_cutoff=3)

            hk_cart::Function = k_cart::Vector{Float64} -> begin
                # convert `k_cart` to `k_crys`
                G_mat = reduce(hcat, moire_model.reciprocal_vec_list)
                k_crys = inv(G_mat) * k_cart

                moire_model.hk_matrix_for_valley(k_crys, valley_index)
            end

            # compute `Ω_xy` and `trace(G)` within the range from `k_crys ∈ [0, 2π] × [0, 2π]` (no need bother to pull back to the hexagonal moire BZ)
            k_crys_list = [[i / N, j / N] for i in 0:N for j in 0:N]
            k_cart_list = [reduce(+, k_crys .* moire_model.reciprocal_vec_list) for k_crys in k_crys_list] # convert `k_crys` to `k_cart`
            @show length(k_cart_list) # number of k-points in the moire Brillouin zone


            moire_cell_volume = let
                (G1, G2) = Tuple(moire_model.reciprocal_vec_list)
                G1_complex = G1[1] + im * G1[2]
                G2_complex = G2[1] + im * G2[2]
                begin
                    norm(G1_complex) * norm(G2_complex) * sin(angle(G1_complex + im * G2_complex)) |> real # area of the moire unit cell
                end
            end


            Ω_list = [CMP_Utils.berry_curvature_Ωxy(hk_cart, k_cart, band_index; which=which) for k_cart in k_cart_list] * 1 / (2π) * moire_cell_volume # make it dimensionless

            trace_G_list = [CMP_Utils.trace_of_quantum_metric_tensor_G(hk_cart, k_cart, band_index; which=which) for k_cart in k_cart_list] * 1 / (2π) * moire_cell_volume # make it dimensionless


            eigval_list = [Arpack.eigs(moire_model.hk_matrix_for_valley(k_crys, valley_index); nev=band_index, which=which)[1] |> first |> real for k_crys in k_crys_list]
            @show band_width = maximum(eigval_list) - minimum(eigval_list)
            push!(band_width_list, band_width)

            push!(Ω_avg_list, sum(Ω_list) / length(Ω_list))
            push!(ΔΩ_list, maximum(Ω_list) - minimum(Ω_list))
            push!(trace_G_avg_list, sum(trace_G_list) / length(trace_G_list))
            push!(Δtrace_G_list, maximum(trace_G_list) - minimum(trace_G_list))
        end

        open(data_path, "w") do io
            write(io, "eE\tΔtrace_G\tΔΩ\ttrace_G_avg\tΩ_avg\tband width\n")
            for (eE, Δtrace_G, ΔΩ, trace_G_avg, Ω_avg, band_width) in zip(eE_pool, Δtrace_G_list, ΔΩ_list, trace_G_avg_list, Ω_avg_list, band_width_list)
                write(io, "$eE\t$Δtrace_G\t$ΔΩ\t$trace_G_avg\t$Ω_avg\t$band_width\n")
            end
        end
    else
        # read data from file
        data_dir = @__DIR__
        data_path = "$data_dir/geometry_data.txt"
        if !isfile(data_path)
            error("Data file $data_path does not exist. Please run the function with `new_run=true` to generate the data.")
            return nothing
        end
        @info "Reading data from $data_path"
        open(data_path, "r") do io
            # skip the header line
            readline(io)
            while !eof(io)
                line = readline(io)
                eE, Δtrace_G, ΔΩ, trace_G_avg, Ω_avg, band_width = split(line, '\t')
                eE = parse(Float64, eE)
                Δtrace_G = parse(Float64, Δtrace_G)
                ΔΩ = parse(Float64, ΔΩ)
                trace_G_avg = parse(Float64, trace_G_avg)
                Ω_avg = parse(Float64, Ω_avg)
                band_width = parse(Float64, band_width) # band width is not used in the plot, but we still read it for completeness
                @show eE, Δtrace_G, ΔΩ, trace_G_avg, Ω_avg, band_width

                push!(Δtrace_G_list, Δtrace_G)
                push!(ΔΩ_list, ΔΩ)
                push!(trace_G_avg_list, trace_G_avg)
                push!(Ω_avg_list, Ω_avg)
                push!(band_width_list, band_width)
            end
        end
    end


    fig = CairoMakie.Figure(size=(600, 400))
    # create twin axes 
    ax = CairoMakie.Axis(fig[1, 1];
        backgroundcolor=:transparent,
        xlabel="eE [meV/Å]",
        yminorticksvisible=true,
        # limits=(nothing, (-0.2, 10.2)),
        title="Trace Condition and Fluctuations",
        xticks=0.0:0.4:2.0,
    )
    CairoMakie.scatter!(ax, eE_pool, Δtrace_G_list; marker=:rect, markersize=16, label=L"$|\Delta\mathrm{tr}\mathcal{G}|$")
    CairoMakie.lines!(ax, eE_pool, Δtrace_G_list; linestyle=:dash, linewidth=2)
    CairoMakie.scatter!(ax, eE_pool, ΔΩ_list; marker=:rect, markersize=16, label=L"$|\Delta\Omega|$")
    CairoMakie.lines!(ax, eE_pool, ΔΩ_list; linestyle=:dash, linewidth=2)
    CairoMakie.scatter!(ax, eE_pool, (trace_G_avg_list .+ Ω_avg_list); marker=:diamond, color=:tomato, markersize=16, label=L"($\mathrm{tr}\mathcal{G} + \Omega)$")
    CairoMakie.lines!(ax, eE_pool, (trace_G_avg_list .+ Ω_avg_list); linestyle=:dash, linewidth=2, color=:tomato)

    CairoMakie.axislegend(ax, position=:lt)
    colsize!(fig.layout, 1, Aspect(1, 0.8)) # check `# Box(f[1, 1], color=(:red, 0.2), strokewidth=0)`


    # # CairoMakie.xlims!(ax, (-0.1, maximum(eE_pool) * 1.4))
    # ax2 = CairoMakie.Axis(fig[1, 1];
    #     yticklabelcolor=:tomato, yaxisposition=:right,
    #     yticks=range(0.0, stop=maximum(band_width_list) * 1.4, length=5),
    #     yminorticksvisible=true,)
    # CairoMakie.hidespines!(ax2)
    # CairoMakie.hidexdecorations!(ax2)


    # CairoMakie.scatter!(ax2, eE_pool, band_width_list; marker=:hexagon, markersize=16, color=:tomato, label="Band Width [meV]")
    # CairoMakie.lines!(ax2, eE_pool, band_width_list; linestyle=:dash, linewidth=2, color=:tomato)

    # CairoMakie.axislegend(ax2, position=:lt)

    display(fig)


    CairoMakie.save("/home/hxd/quantum_geometry_with_eE.pdf", fig)
    CairoMakie.save("/home/hxd/quantum_geometry_with_eE.svg", fig)

    return nothing
end



end # module