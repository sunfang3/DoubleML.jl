# Propensity score post-processing (Python doubleml.utils.PSProcessor)

"""
    PSProcessor

Clip / bound propensity scores after cross-fitting (Python `PSProcessor` /
`PSProcessorConfig` replacement for bare `trimming_threshold`).

# Fields
- `clipping_threshold`: main clip into `[ε, 1−ε]` (default `0.01`)
- `extreme_threshold`: hard floor before clipping (default `1e-12`)
- `calibration_method`: reserved (`nothing` = none; `"platt"` not yet implemented)
"""
struct PSProcessor
    clipping_threshold::Float64
    extreme_threshold::Float64
    calibration_method::Union{Nothing,String}
end

function PSProcessor(; clipping_threshold::Real=0.01,
                     extreme_threshold::Real=1e-12,
                     calibration_method::Union{Nothing,AbstractString}=nothing)
    (0 < clipping_threshold < 0.5) ||
        throw(ArgumentError("clipping_threshold must be in (0, 0.5)"))
    (0 < extreme_threshold <= clipping_threshold) ||
        throw(ArgumentError("extreme_threshold must be in (0, clipping_threshold]"))
    cal = if calibration_method === nothing
        nothing
    else
        s = String(calibration_method)
        s in ("platt",) || throw(ArgumentError("calibration_method must be nothing or \"platt\""))
        s
    end
    return PSProcessor(Float64(clipping_threshold), Float64(extreme_threshold), cal)
end

"""Alias matching Python `PSProcessorConfig` name."""
const PSProcessorConfig = PSProcessor

"""
    process_propensity(p, proc::PSProcessor) -> Vector{Float64}

Apply extreme then clipping bounds. Calibration hooks reserved for later.
"""
function process_propensity(p::AbstractVector{<:Real}, proc::PSProcessor)
    out = Float64.(p)
    ε0 = proc.extreme_threshold
    ε = proc.clipping_threshold
    @inbounds for i in eachindex(out)
        x = out[i]
        x = clamp(x, ε0, 1 - ε0)
        out[i] = clamp(x, ε, 1 - ε)
    end
    return out
end

"""
    process_propensity(p, trimming_threshold::Real)

Backward-compatible clip using a bare threshold (legacy `trimming_threshold`).
"""
process_propensity(p::AbstractVector{<:Real}, trimming_threshold::Real) =
    _clip_ps(p, Float64(trimming_threshold))

"""
Resolve a propensity processor from either a `PSProcessor` or a bare threshold.
"""
function resolve_ps_processor(ps_processor::Union{Nothing,PSProcessor},
                              trimming_threshold::Real)
    if ps_processor !== nothing
        return ps_processor
    end
    thr = Float64(trimming_threshold)
    # map bare threshold: extreme slightly smaller when thr is large
    ext = min(1e-12, thr)
    return PSProcessor(clipping_threshold=max(thr, 1e-12),
                       extreme_threshold=max(ext, 1e-15))
end

function Base.show(io::IO, p::PSProcessor)
    print(io, "PSProcessor(clip=$(p.clipping_threshold), extreme=$(p.extreme_threshold)",
          p.calibration_method === nothing ? ")" : ", cal=$(p.calibration_method))")
end
