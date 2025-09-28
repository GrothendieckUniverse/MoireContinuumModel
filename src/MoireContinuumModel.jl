module MoireContinuumModel

include("r_stack_TMD_model.jl")
using .RStackMoireContinuumModel
export R_Stack_Twisted_Moire_Model, params_FengchengWu, params_ChongWang
export initialize_r_stack_moire_continuum_model, plot_moire_bands, plot_Ωxy_and_trace_G

end # module MoireContinuumModel
